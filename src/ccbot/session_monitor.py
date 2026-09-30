"""Session monitoring service — watches JSONL files for new messages.

Runs an async polling loop; each cycle (poll_once, serialised by a lock) does:
  1. Loads the current session_map to know which sessions to watch.
  2. Detects session_map changes (new/changed/deleted windows) and cleans up.
  3. Locates the transcript of exactly those sessions (hook-reported path,
     in-memory cache, one glob per session id with a negative-cache retry).
  4. Reads new JSONL lines from each session file using byte-offset tracking.
  5. Parses entries via TranscriptParser and emits NewMessage objects to a callback.

poll_now() lets other code force a cycle on demand (e.g. right before reading
fresh output). Optimizations: no directory scan (cost tracks active sessions,
not history); mtime cache skips unchanged files; byte offset avoids re-reading.

Key classes: SessionMonitor, NewMessage, SessionInfo.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiofiles

from .config import config
from .monitor_state import MonitorState, TrackedSession
from .transcript_parser import ToolCall, TranscriptParser
from .utils import read_json_cached

logger = logging.getLogger(__name__)

# Minimum gap between glob lookups for a session whose transcript wasn't found
TRANSCRIPT_RETRY_SECONDS = 5.0
# Longest a poll_now() caller (the status poller) waits for its cycle
POLL_NOW_TIMEOUT = 5.0


@dataclass
class SessionInfo:
    """Information about a Claude Code session."""

    session_id: str
    file_path: Path


@dataclass
class NewMessage:
    """A new message detected by the monitor."""

    session_id: str
    text: str
    is_complete: bool  # True when stop_reason is set (final message)
    content_type: str = "text"  # "text" or "thinking"
    tool_use_id: str | None = None
    role: str = "assistant"  # "user" or "assistant"
    tool_name: str | None = None  # For tool_use messages, the tool name
    image_data: list[tuple[str, bytes]] | None = None  # From tool_result images
    timestamp: str | None = None  # ISO timestamp of the JSONL entry
    stop_reason: str | None = None  # "end_turn" on the last message of a turn
    api_message_id: str | None = None
    raw: str | None = None  # unformatted text / thinking body
    tool: ToolCall | None = None  # structured tool_use / tool_result data


class SessionMonitor:
    """Monitors Claude Code sessions for new assistant messages.

    Uses simple async polling with aiofiles for non-blocking I/O.
    Emits both intermediate and complete assistant messages.
    """

    def __init__(
        self,
        projects_path: Path | None = None,
        poll_interval: float | None = None,
        state_file: Path | None = None,
    ):
        self.projects_path = (
            projects_path if projects_path is not None else config.claude_projects_path
        )
        self.poll_interval = (
            poll_interval if poll_interval is not None else config.monitor_poll_interval
        )

        self.state = MonitorState(state_file=state_file or config.monitor_state_file)
        self.state.load()

        self._running = False
        self._task: asyncio.Task | None = None
        self._message_callback: Callable[[NewMessage], Awaitable[None]] | None = None
        # Per-session pending tool_use state carried across poll cycles
        self._pending_tools: dict[str, dict[str, Any]] = {}  # session_id -> pending
        # Track last known session_map for detecting changes
        # Keys may be window_id (@12) or window_name (old format) during transition
        self._last_session_map: dict[str, str] = {}  # window_key -> session_id
        # In-memory mtime cache for quick file change detection (not persisted)
        self._file_mtimes: dict[str, float] = {}  # session_id -> last_seen_mtime
        # Sessions that entered session_map while running, before their JSONL
        # existed: every byte is new, so they are tracked from offset 0.
        self._fresh_sessions: set[str] = set()
        # session_id -> resolved transcript path (avoids re-globbing)
        self._transcript_cache: dict[str, Path] = {}
        # session_id -> monotonic time before which a failed lookup isn't retried
        self._transcript_retry_at: dict[str, float] = {}
        # Serialises poll cycles: offsets/pending tools must not race
        self._poll_lock = asyncio.Lock()
        # poll_now() requests that haven't started their cycle yet share it
        self._requested: asyncio.Future[None] | None = None
        self._requested_tasks: set[asyncio.Task[None]] = set()

    def set_message_callback(
        self, callback: Callable[[NewMessage], Awaitable[None]]
    ) -> None:
        self._message_callback = callback

    @staticmethod
    def _hook_transcript_paths() -> dict[str, str]:
        """session_id -> transcript_path as reported by the SessionStart hook."""
        # Deferred import to avoid circular dependency
        from .session import session_manager

        return {
            st.session_id: st.transcript_path
            for st in session_manager.window_states.values()
            if st.session_id and st.transcript_path
        }

    def _find_transcript(
        self, session_id: str, hook_paths: dict[str, str]
    ) -> Path | None:
        """Locate a session's transcript without scanning the projects tree.

        Order: transcript_path reported by the SessionStart hook (authoritative,
        survives `cd` and renamed project dirs), the in-memory cache, then one
        glob for `*/<session_id>.jsonl`. Misses (a fresh session whose file
        doesn't exist yet) are negative-cached and the glob retried at most
        every TRANSCRIPT_RETRY_SECONDS.
        """
        hook_path = hook_paths.get(session_id)
        if hook_path:
            path = Path(hook_path)
            if path.exists():
                self._transcript_cache[session_id] = path
                self._transcript_retry_at.pop(session_id, None)
                return path

        cached = self._transcript_cache.get(session_id)
        if cached is not None:
            if cached.exists():
                return cached
            del self._transcript_cache[session_id]  # moved/deleted: look again

        now = time.monotonic()
        if now < self._transcript_retry_at.get(session_id, 0.0):
            return None
        try:
            found = next(iter(self.projects_path.glob(f"*/{session_id}.jsonl")), None)
        except OSError as e:
            logger.debug("Error looking up transcript for %s: %s", session_id, e)
            found = None
        if found is None:
            self._transcript_retry_at[session_id] = now + TRANSCRIPT_RETRY_SECONDS
            return None
        self._transcript_cache[session_id] = found
        self._transcript_retry_at.pop(session_id, None)
        return found

    def _find_session_files(self, active_session_ids: set[str]) -> list[SessionInfo]:
        """Resolve transcript files for exactly the sessions in session_map."""
        hook_paths = self._hook_transcript_paths()
        # Drop lookup state for sessions that are no longer active
        for cache in (self._transcript_cache, self._transcript_retry_at):
            for sid in [k for k in cache if k not in active_session_ids]:
                del cache[sid]

        sessions = []
        for sid in active_session_ids:
            path = self._find_transcript(sid, hook_paths)
            if path is not None:
                sessions.append(SessionInfo(session_id=sid, file_path=path))
        return sessions

    async def _read_new_lines(
        self, session: TrackedSession, file_path: Path
    ) -> list[dict]:
        """Read new lines from a session file using byte offset for efficiency.

        Detects file truncation (e.g. after /clear) and resets offset.
        Recovers from corrupted offsets (mid-line) by scanning to next line.
        """
        new_entries = []
        try:
            async with aiofiles.open(
                file_path, "r", encoding="utf-8", errors="replace"
            ) as f:
                # Get file size to detect truncation
                await f.seek(0, 2)  # Seek to end
                file_size = await f.tell()

                # Detect file truncation: if offset is beyond file size, reset
                if session.last_byte_offset > file_size:
                    logger.info(
                        "File truncated for session %s "
                        "(offset %d > size %d). Resetting.",
                        session.session_id,
                        session.last_byte_offset,
                        file_size,
                    )
                    session.last_byte_offset = 0

                # Seek to last read position for incremental reading
                await f.seek(session.last_byte_offset)

                # Detect corrupted offset: if we're mid-line (not at '{'),
                # scan forward to the next line start. This can happen if
                # the state file was manually edited or corrupted.
                if session.last_byte_offset > 0:
                    first_char = await f.read(1)
                    if first_char and first_char != "{":
                        logger.warning(
                            "Corrupted offset %d in session %s (mid-line), "
                            "scanning to next line",
                            session.last_byte_offset,
                            session.session_id,
                        )
                        await f.readline()  # Skip rest of partial line
                        session.last_byte_offset = await f.tell()
                        return []
                    await f.seek(session.last_byte_offset)  # Reset for normal read

                # Read only new lines from the offset.
                # Track safe_offset: only advance past lines that parsed
                # successfully. A non-empty line without a trailing newline is
                # a partial write at EOF — retry next cycle. A COMPLETE line
                # (newline-terminated) that fails to parse is permanently
                # corrupt; skip it, or it would stall this session forever.
                safe_offset = session.last_byte_offset
                async for line in f:
                    data = TranscriptParser.parse_line(line)
                    if data:
                        new_entries.append(data)
                        safe_offset = await f.tell()
                    elif line.strip():
                        if line.endswith("\n"):
                            logger.warning(
                                "Skipping unparseable JSONL line in session %s "
                                "(%d bytes)",
                                session.session_id,
                                len(line),
                            )
                            safe_offset = await f.tell()
                        else:
                            # Partial write at EOF — don't advance offset
                            logger.debug(
                                "Partial JSONL line in session %s, "
                                "will retry next cycle",
                                session.session_id,
                            )
                            break
                    else:
                        # Empty line — safe to skip
                        safe_offset = await f.tell()

                session.last_byte_offset = safe_offset

        except OSError as e:
            logger.error("Error reading session file %s: %s", file_path, e)
        return new_entries

    async def check_for_updates(self, active_session_ids: set[str]) -> list[NewMessage]:
        """Check all sessions for new assistant messages.

        Reads from last byte offset. Emits both intermediate
        (stop_reason=null) and complete messages.

        Args:
            active_session_ids: Set of session IDs currently in session_map
        """
        new_messages = []

        # Look up the transcript of each session in session_map (no dir scan)
        sessions = self._find_session_files(active_session_ids)

        for session_info in sessions:
            try:
                tracked = self.state.get_session(session_info.session_id)

                if tracked is None:
                    # For new sessions, initialize offset to end of file
                    # to avoid re-processing old messages — unless the session
                    # was created while we were watching, when all of it is new.
                    fresh = session_info.session_id in self._fresh_sessions
                    self._fresh_sessions.discard(session_info.session_id)
                    try:
                        file_size = session_info.file_path.stat().st_size
                        current_mtime = session_info.file_path.stat().st_mtime
                    except OSError:
                        file_size = 0
                        current_mtime = 0.0
                    tracked = TrackedSession(
                        session_id=session_info.session_id,
                        file_path=str(session_info.file_path),
                        last_byte_offset=0 if fresh else file_size,
                    )
                    self.state.update_session(tracked)
                    self._file_mtimes[session_info.session_id] = current_mtime
                    logger.info(f"Started tracking session: {session_info.session_id}")
                    continue

                # Check mtime + file size to see if file has changed
                try:
                    st = session_info.file_path.stat()
                    current_mtime = st.st_mtime
                    current_size = st.st_size
                except OSError:
                    continue

                last_mtime = self._file_mtimes.get(session_info.session_id, 0.0)
                if (
                    current_mtime <= last_mtime
                    and current_size <= tracked.last_byte_offset
                ):
                    # File hasn't changed, skip reading
                    continue

                # File changed, read new content from last offset
                new_entries = await self._read_new_lines(
                    tracked, session_info.file_path
                )
                self._file_mtimes[session_info.session_id] = current_mtime

                if new_entries:
                    logger.debug(
                        f"Read {len(new_entries)} new entries for "
                        f"session {session_info.session_id}"
                    )

                # Parse new entries using the shared logic, carrying over pending tools
                carry = self._pending_tools.get(session_info.session_id, {})
                parsed_entries, remaining = TranscriptParser.parse_entries(
                    new_entries,
                    pending_tools=carry,
                )
                if remaining:
                    self._pending_tools[session_info.session_id] = remaining
                else:
                    self._pending_tools.pop(session_info.session_id, None)

                for entry in parsed_entries:
                    if not entry.text and not entry.image_data:
                        continue
                    # User messages are always emitted (they mark turn
                    # starts); the bot decides whether to display them.
                    new_messages.append(
                        NewMessage(
                            session_id=session_info.session_id,
                            text=entry.text,
                            is_complete=True,
                            content_type=entry.content_type,
                            tool_use_id=entry.tool_use_id,
                            role=entry.role,
                            tool_name=entry.tool_name,
                            image_data=entry.image_data,
                            timestamp=entry.timestamp,
                            stop_reason=entry.stop_reason,
                            api_message_id=entry.api_message_id,
                            raw=entry.raw,
                            tool=entry.tool,
                        )
                    )

                self.state.update_session(tracked)

            except OSError as e:
                logger.debug(f"Error processing session {session_info.session_id}: {e}")

        self.state.save_if_dirty()
        return new_messages

    async def _load_current_session_map(self) -> dict[str, str]:
        """Load current session_map and return window_key -> session_id mapping.

        Keys in session_map are formatted as "tmux_session:window_id"
        (e.g. "ccbot:@12"). Old-format keys ("ccbot:window_name") are also
        accepted so that sessions running before a code upgrade continue
        to be monitored until the hook re-fires with new format.
        Only entries matching our tmux_session_name are processed.
        """
        window_to_session: dict[str, str] = {}
        if config.session_map_file.exists():
            try:
                session_map = await read_json_cached(config.session_map_file)
                prefix = f"{config.tmux_session_name}:"
                for key, info in session_map.items():
                    # Only process entries for our tmux session
                    if not key.startswith(prefix):
                        continue
                    window_key = key[len(prefix) :]
                    session_id = info.get("session_id", "")
                    if session_id:
                        window_to_session[window_key] = session_id
            except (json.JSONDecodeError, OSError):
                pass
        return window_to_session

    async def _cleanup_all_stale_sessions(self) -> None:
        """Clean up all tracked sessions not in current session_map (used on startup)."""
        current_map = await self._load_current_session_map()
        active_session_ids = set(current_map.values())

        stale_sessions = []
        for session_id in self.state.tracked_sessions.keys():
            if session_id not in active_session_ids:
                stale_sessions.append(session_id)

        if stale_sessions:
            logger.info(
                f"[Startup cleanup] Removing {len(stale_sessions)} stale sessions"
            )
            for session_id in stale_sessions:
                self.state.remove_session(session_id)
                self._file_mtimes.pop(session_id, None)
            self.state.save_if_dirty()

    async def _detect_and_cleanup_changes(self) -> dict[str, str]:
        """Detect session_map changes and cleanup replaced/removed sessions.

        Returns current session_map for further processing.
        """
        current_map = await self._load_current_session_map()

        sessions_to_remove: set[str] = set()

        # Check for window session changes (window exists in both, but session_id changed)
        for window_id, old_session_id in self._last_session_map.items():
            new_session_id = current_map.get(window_id)
            if new_session_id and new_session_id != old_session_id:
                logger.info(
                    "Window '%s' session changed: %s -> %s",
                    window_id,
                    old_session_id,
                    new_session_id,
                )
                sessions_to_remove.add(old_session_id)

        # Check for deleted windows (window in old map but not in current)
        old_windows = set(self._last_session_map.keys())
        current_windows = set(current_map.keys())
        deleted_windows = old_windows - current_windows

        for window_id in deleted_windows:
            old_session_id = self._last_session_map[window_id]
            logger.info(
                "Window '%s' deleted, removing session %s",
                window_id,
                old_session_id,
            )
            sessions_to_remove.add(old_session_id)

        # A session that appears while we're running and has no transcript yet
        # is brand new: nothing it writes has been seen, so track it from byte 0
        # rather than from wherever EOF happens to be when a poll first finds
        # the file. Sessions whose file already exists (e.g. --resume) still
        # start at EOF so their history isn't replayed.
        known_session_ids = set(self._last_session_map.values())
        new_session_ids = set(current_map.values()) - known_session_ids
        hook_paths = self._hook_transcript_paths() if new_session_ids else {}
        for session_id in new_session_ids:
            if (
                self.state.get_session(session_id) is None
                and self._find_transcript(session_id, hook_paths) is None
            ):
                self._fresh_sessions.add(session_id)

        # Perform cleanup
        if sessions_to_remove:
            for session_id in sessions_to_remove:
                self.state.remove_session(session_id)
                self._file_mtimes.pop(session_id, None)
                self._fresh_sessions.discard(session_id)
            self.state.save_if_dirty()

        # Update last known map
        self._last_session_map = current_map

        return current_map

    async def poll_once(self) -> None:
        """Run one full monitor cycle and deliver its messages, in order.

        Serialised by a lock so cycles never overlap (offsets would race).
        Errors are logged, never raised.
        """
        async with self._poll_lock:
            await self._cycle()

    async def _cycle(self) -> None:
        """Body of one poll cycle; the caller holds ``_poll_lock``."""
        # Deferred import to avoid circular dependency
        from .session import session_manager

        try:
            # Load hook-based session map updates
            await session_manager.load_session_map()

            # Detect session_map changes and cleanup replaced/removed sessions
            current_map = await self._detect_and_cleanup_changes()
            active_session_ids = set(current_map.values())

            # Check for new messages (all I/O is async)
            new_messages = await self.check_for_updates(active_session_ids)

            for msg in new_messages:
                status = "complete" if msg.is_complete else "streaming"
                preview = msg.text[:80] + ("..." if len(msg.text) > 80 else "")
                logger.info("[%s] session=%s: %s", status, msg.session_id, preview)
                if self._message_callback:
                    try:
                        await self._message_callback(msg)
                    except Exception as e:
                        logger.error(f"Message callback error: {e}")

        except Exception as e:
            logger.error(f"Monitor loop error: {e}")

    async def poll_now(self, timeout: float = POLL_NOW_TIMEOUT) -> None:
        """Run a monitor cycle on demand and wait (at most ``timeout``) for it.

        The cycle starts after this call, so it sees anything written before
        the call — a cycle already in progress doesn't count. Callers that
        arrive while a requested cycle is still waiting for the lock share
        it instead of queueing one cycle each.
        """
        fut = self._requested
        if fut is None:
            fut = asyncio.get_running_loop().create_future()
            self._requested = fut
            task = asyncio.create_task(self._run_requested(fut))
            self._requested_tasks.add(task)
            task.add_done_callback(self._requested_tasks.discard)
        try:
            await asyncio.wait_for(asyncio.shield(fut), timeout)
        except TimeoutError:
            logger.warning("On-demand transcript poll still running after %ss", timeout)

    async def _run_requested(self, fut: asyncio.Future[None]) -> None:
        try:
            async with self._poll_lock:
                if self._requested is fut:
                    self._requested = None  # later callers need a newer cycle
                await self._cycle()
        finally:
            if self._requested is fut:
                self._requested = None
            if not fut.done():
                fut.set_result(None)

    async def _monitor_loop(self) -> None:
        """Background loop: poll_once() every poll_interval seconds."""
        logger.info("Session monitor started, polling every %ss", self.poll_interval)

        async with self._poll_lock:
            # Clean up all stale sessions on startup
            await self._cleanup_all_stale_sessions()
            # Initialize last known session_map
            self._last_session_map = await self._load_current_session_map()

        while self._running:
            await self.poll_once()
            await asyncio.sleep(self.poll_interval)

        logger.info("Session monitor stopped")

    def start(self) -> None:
        if self._running:
            logger.warning("Monitor already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        self.state.save()
        logger.info("Session monitor stopped and state saved")
