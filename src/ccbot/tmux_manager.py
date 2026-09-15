"""Tmux session/window management via libtmux.

Wraps libtmux to provide async-friendly operations on a single tmux session:
  - list_windows / find_window_by_name: discover Claude Code windows.
  - capture_pane: read terminal content (plain or with ANSI colors).
  - send_keys: forward user input or control keys to a window.
  - create_window / kill_window: lifecycle management.

All blocking libtmux calls are wrapped in asyncio.to_thread().

Key class: TmuxManager (singleton instantiated as `tmux_manager`).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import libtmux

from .config import SENSITIVE_ENV_VARS, config
from .terminal_parser import (
    AUTO_ANSWER_DIALOGS,
    extract_interactive_content,
    find_menu_option,
    is_blocking_dialog,
    is_prompt_ready,
)

# Launch modes → extra `claude` CLI flags. Keys double as the canonical
# permission-mode names (as reported by hooks / `--permission-mode`).
LAUNCH_MODES: dict[str, str] = {
    "default": "",
    "acceptEdits": "--permission-mode acceptEdits",
    "plan": "--permission-mode plan",
    "bypassPermissions": "--dangerously-skip-permissions",
}
LAUNCH_MODE_LABELS: dict[str, str] = {
    "default": "🟢 Normal",
    "acceptEdits": "✏️ Accept edits",
    "plan": "📝 Plan",
    "bypassPermissions": "⚡ Skip permissions",
}
# Accepted aliases for /restart <mode> and /mode <mode>
LAUNCH_MODE_ALIASES: dict[str, str] = {
    "default": "default",
    "normal": "default",
    "manual": "default",
    "acceptedits": "acceptEdits",
    "accept": "acceptEdits",
    "edit": "acceptEdits",
    "plan": "plan",
    "bypasspermissions": "bypassPermissions",
    "bypass": "bypassPermissions",
    "skip": "bypassPermissions",
    "dangerous": "bypassPermissions",
    "yolo": "bypassPermissions",
}


def normalize_launch_mode(value: str | None) -> str | None:
    """Map a user-typed mode name to a LAUNCH_MODES key (None if unknown)."""
    if not value:
        return None
    return LAUNCH_MODE_ALIASES.get(value.strip().lower())


def build_claude_command(
    mode: str = "default", resume_session_id: str | None = None
) -> str:
    """Build the shell command that starts Claude Code in a window.

    Non-bypass modes get --allow-dangerously-skip-permissions so that
    Shift+Tab (and /mode) can still reach bypass later — except as root,
    where Claude Code refuses the flag.
    """
    parts = [config.claude_command]
    flag = LAUNCH_MODES.get(mode, "")
    if flag:
        parts.append(flag)
    if mode != "bypassPermissions" and os.geteuid() != 0:
        parts.append("--allow-dangerously-skip-permissions")
    if resume_session_id:
        parts.append(f"--resume {resume_session_id}")
    return " ".join(parts)


# pane_current_command values meaning "Claude Code is not running here"
SHELL_COMMANDS = frozenset(
    {"zsh", "bash", "fish", "sh", "dash", "tcsh", "ksh", "nu", "login"}
)

logger = logging.getLogger(__name__)

# Claude session IDs are UUIDs (JSONL filename stems)
_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


@dataclass
class TmuxWindow:
    """Information about a tmux window."""

    window_id: str
    window_name: str
    cwd: str  # Current working directory
    pane_current_command: str = ""  # Process running in active pane


class TmuxManager:
    """Manages tmux windows for Claude Code sessions."""

    def __init__(self, session_name: str | None = None):
        """Initialize tmux manager.

        Args:
            session_name: Name of the tmux session to use (default from config)
        """
        self.session_name = session_name or config.tmux_session_name
        self._server: libtmux.Server | None = None
        self._list_lock = asyncio.Lock()
        self._windows_cache: list[TmuxWindow] | None = None
        self._windows_cache_at = 0.0

    @property
    def server(self) -> libtmux.Server:
        """Get or create tmux server connection."""
        if self._server is None:
            self._server = libtmux.Server()
        return self._server

    def get_session(self) -> libtmux.Session | None:
        """Get the tmux session if it exists."""
        try:
            return self.server.sessions.get(session_name=self.session_name)
        except Exception:
            return None

    def get_or_create_session(self) -> libtmux.Session:
        """Get existing session or create a new one."""
        session = self.get_session()
        if session:
            self._scrub_session_env(session)
            return session

        # Create new session with main window named specifically
        session = self.server.new_session(
            session_name=self.session_name,
            start_directory=str(Path.home()),
        )
        # Rename the default window to the main window name
        if session.windows:
            session.windows[0].rename_window(config.tmux_main_window_name)
        self._scrub_session_env(session)
        return session

    @staticmethod
    def _scrub_session_env(session: libtmux.Session) -> None:
        """Remove sensitive env vars from the tmux session environment.

        Prevents new windows (and their child processes like Claude Code)
        from inheriting secrets such as TELEGRAM_BOT_TOKEN.
        """
        for var in SENSITIVE_ENV_VARS:
            try:
                session.unset_environment(var)
            except Exception:
                pass  # var not set in session env — nothing to remove

    # list_windows() is called by the 1s status poll for every binding plus
    # every send/capture, so it is cached briefly and done with a single
    # tmux fork instead of libtmux's per-window list-panes calls.
    _LIST_WINDOWS_TTL = 0.5
    # tmux ≤3.5 escapes control characters in -F output ("\037"), so the
    # separator must be printable; ␞ (U+241E) is what libtmux uses too.
    _LIST_SEP = "\u241e"
    _LIST_FORMAT = _LIST_SEP.join(
        [
            "#{window_id}",
            "#{window_name}",
            "#{pane_current_path}",
            "#{pane_current_command}",
            "#{pane_active}",
        ]
    )

    async def list_windows(self) -> list[TmuxWindow]:
        """List all windows in the session with their working directories.

        Results are cached for _LIST_WINDOWS_TTL seconds. An empty result
        is double-checked against the session's existence so a transient
        tmux hiccup never looks like "all windows are gone" to callers
        that unbind topics based on it.

        Returns:
            List of TmuxWindow with window info and cwd
        """
        async with self._list_lock:
            now = time.monotonic()
            if self._windows_cache is not None and (
                now - self._windows_cache_at < self._LIST_WINDOWS_TTL
            ):
                return list(self._windows_cache)

            windows = await asyncio.to_thread(self._sync_list_windows)
            if not windows and not await asyncio.to_thread(self.get_session):
                # Session unreachable — surface an empty list but do NOT
                # cache it so the next call re-probes immediately.
                logger.warning("tmux session '%s' unreachable", self.session_name)
                return []

            self._windows_cache = windows
            self._windows_cache_at = now
            return list(windows)

    def invalidate_windows_cache(self) -> None:
        """Drop the list_windows cache (after create/kill/rename)."""
        self._windows_cache = None

    def _sync_list_windows(self) -> list[TmuxWindow]:
        """Single `tmux list-panes -s` call, parsed with a \x1f separator.

        Avoids libtmux's per-window queries and its zip(strict=True) crash
        when any format field contains a newline.
        """
        try:
            result = subprocess.run(
                [
                    "tmux",
                    "list-panes",
                    "-s",
                    "-t",
                    self.session_name,
                    "-F",
                    self._LIST_FORMAT,
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as e:
            logger.debug("tmux list-panes failed: %s", e)
            return []
        if result.returncode != 0:
            logger.debug("tmux list-panes failed: %s", result.stderr.strip())
            return []

        windows: list[TmuxWindow] = []
        seen: set[str] = set()
        for line in result.stdout.splitlines():
            parts = line.split(self._LIST_SEP)
            if len(parts) != 5:
                continue
            window_id, name, cwd, pane_cmd, active = parts
            # One entry per window: prefer the active pane
            if active != "1" or window_id in seen:
                continue
            if name == config.tmux_main_window_name:
                continue
            seen.add(window_id)
            windows.append(
                TmuxWindow(
                    window_id=window_id,
                    window_name=name,
                    cwd=cwd,
                    pane_current_command=pane_cmd,
                )
            )
        return windows

    async def find_window_by_name(self, window_name: str) -> TmuxWindow | None:
        """Find a window by its name.

        Args:
            window_name: The window name to match

        Returns:
            TmuxWindow if found, None otherwise
        """
        windows = await self.list_windows()
        for window in windows:
            if window.window_name == window_name:
                return window
        logger.debug("Window not found by name: %s", window_name)
        return None

    async def find_window_by_id(self, window_id: str) -> TmuxWindow | None:
        """Find a window by its tmux window ID (e.g. '@0', '@12').

        Args:
            window_id: The tmux window ID to match

        Returns:
            TmuxWindow if found, None otherwise
        """
        windows = await self.list_windows()
        for window in windows:
            if window.window_id == window_id:
                return window
        logger.debug("Window not found by id: %s", window_id)
        return None

    async def capture_pane(self, window_id: str, with_ansi: bool = False) -> str | None:
        """Capture the visible text content of a window's active pane.

        Args:
            window_id: The window ID to capture
            with_ansi: If True, capture with ANSI color codes

        Returns:
            The captured text, or None on failure.
        """
        if with_ansi:
            # Use async subprocess to call tmux capture-pane -e for ANSI colors
            try:
                proc = await asyncio.create_subprocess_exec(
                    "tmux",
                    "capture-pane",
                    "-e",
                    "-p",
                    "-t",
                    window_id,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode == 0:
                    return stdout.decode("utf-8")
                logger.error(
                    f"Failed to capture pane {window_id}: {stderr.decode('utf-8')}"
                )
                return None
            except Exception as e:
                logger.error(f"Unexpected error capturing pane {window_id}: {e}")
                return None

        # Original implementation for plain text - wrap in thread
        def _sync_capture() -> str | None:
            session = self.get_session()
            if not session:
                return None
            try:
                window = session.windows.get(window_id=window_id)
                if not window:
                    return None
                pane = window.active_pane
                if not pane:
                    return None
                lines = pane.capture_pane()
                return "\n".join(lines) if isinstance(lines, list) else str(lines)
            except Exception as e:
                logger.error(f"Failed to capture pane {window_id}: {e}")
                return None

        return await asyncio.to_thread(_sync_capture)

    async def send_key(self, window_id: str, key: str) -> bool:
        """Send a single tmux key name (``Down``, ``Enter``, ``Escape``, ``BTab``)."""
        return await self.send_keys(window_id, key, enter=False, literal=False)

    async def auto_answer_dialog(self, window_id: str, pane_text: str) -> str | None:
        """Answer a startup dialog the bot handles itself (trust folder, …).

        Moves the ``❯`` cursor onto the wanted option with Up/Down, verifies
        it landed there, then presses Enter. Never blind-types a digit —
        option order has changed between Claude Code versions.

        Returns the dialog name when one was answered, else None.
        """
        ui = extract_interactive_content(pane_text)
        if ui is None or ui.name not in AUTO_ANSWER_DIALOGS:
            return None
        needle = AUTO_ANSWER_DIALOGS[ui.name]
        if needle is None:
            await self.send_key(window_id, "Escape")
            logger.info("Dismissed %s in %s", ui.name, window_id)
            return ui.name

        offsets = find_menu_option(pane_text, needle)
        if offsets is None:
            logger.warning("%s in %s: option %r not found", ui.name, window_id, needle)
            return None
        cursor, target = offsets
        steps = target - cursor
        key = "Down" if steps > 0 else "Up"
        for _ in range(abs(steps)):
            await self.send_key(window_id, key)
            await asyncio.sleep(0.15)
        if steps:
            await asyncio.sleep(0.3)
            pane = await self.capture_pane(window_id)
            check = find_menu_option(pane or "", needle)
            if check is None or check[0] != check[1]:
                logger.warning(
                    "%s in %s: cursor did not land on %r, not confirming",
                    ui.name,
                    window_id,
                    needle,
                )
                return None
        await self.send_key(window_id, "Enter")
        logger.info("Answered %s in %s with %r", ui.name, window_id, needle)
        return ui.name

    async def wait_for_claude_ready(
        self, window_id: str, timeout: float = 30.0
    ) -> tuple[bool, str]:
        """Wait until Claude Code shows its input prompt, answering startup dialogs.

        Returns ``(ready, note)`` where note lists dialogs answered on the way
        or the reason for giving up.
        """
        deadline = time.monotonic() + timeout
        answered: list[str] = []
        shell_seen = 0
        while time.monotonic() < deadline:
            pane = await self.capture_pane(window_id)
            if pane:
                if is_prompt_ready(pane):
                    note = ", ".join(answered) if answered else "ready"
                    return True, note
                handled = await self.auto_answer_dialog(window_id, pane)
                if handled:
                    answered.append(handled)
                    await asyncio.sleep(1.0)
                    continue
            window = await self.find_window_by_id(window_id)
            if window is None:
                return False, "window closed"
            if window.pane_current_command in SHELL_COMMANDS:
                # Claude may not have started yet; give it a few polls
                shell_seen += 1
                if shell_seen >= 8:
                    return False, "Claude Code is not running (shell prompt)"
            else:
                shell_seen = 0
            await asyncio.sleep(0.5)
        return False, f"timed out after {timeout:.0f}s"

    async def clear_blocking_dialog(self, window_id: str) -> tuple[bool, str]:
        """Make the pane safe to type into.

        Answers known startup dialogs, escapes unknown modals (up to 3×).
        Returns ``(ok, reason)``; on failure the reason is user-facing.
        """
        pane = await self.capture_pane(window_id)
        if not pane:
            return True, ""
        handled = await self.auto_answer_dialog(window_id, pane)
        if handled:
            await asyncio.sleep(1.0)
            pane = await self.capture_pane(window_id) or ""
        ui = extract_interactive_content(pane)
        if ui is None or ui.name != "Modal":
            return True, ""
        for _ in range(3):
            await self.send_key(window_id, "Escape")
            await asyncio.sleep(0.3)
            pane = await self.capture_pane(window_id) or ""
            if not is_blocking_dialog(pane):
                return True, ""
        return (
            False,
            "Claude Code is showing a dialog the bot can't handle — "
            "check /screenshot and answer it there.",
        )

    async def stop_claude(self, window_id: str, timeout: float = 6.0) -> bool:
        """Stop the Claude Code process in a window, leaving the shell.

        Escape + two Ctrl-C is Claude Code's exit sequence; if the pane is
        still not at a shell after ``timeout`` the pane is respawned with
        ``tmux respawn-pane -k`` (keeps the window id).
        """
        window = await self.find_window_by_id(window_id)
        if window is None:
            return False
        if window.pane_current_command not in SHELL_COMMANDS:
            await self.send_key(window_id, "Escape")
            await asyncio.sleep(0.3)
            await self.send_key(window_id, "C-c")
            await asyncio.sleep(0.4)
            await self.send_key(window_id, "C-c")
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                await asyncio.sleep(0.5)
                self.invalidate_windows_cache()
                w = await self.find_window_by_id(window_id)
                if w is None:
                    return False
                if w.pane_current_command in SHELL_COMMANDS:
                    return True
            logger.warning("Claude in %s did not exit; respawning pane", window_id)
            proc = await asyncio.create_subprocess_exec(
                "tmux",
                "respawn-pane",
                "-k",
                "-t",
                window_id,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.error("respawn-pane failed: %s", stderr.decode().strip())
                return False
            await asyncio.sleep(0.5)
            self.invalidate_windows_cache()
        return True

    async def start_claude(self, window_id: str, command: str) -> bool:
        """Type ``command`` into the window's shell and press Enter.

        A shell that is still redrawing its prompt (right after Claude
        exited) can swallow the first keystrokes, so the typed line is
        verified against the pane before Enter and retyped once if needed.
        """
        await asyncio.sleep(1.0)  # let the prompt settle
        for attempt in range(2):
            if attempt:
                await self.send_key(window_id, "C-u")
                await asyncio.sleep(0.3)
            if not await self.send_keys(window_id, command, enter=False, literal=True):
                return False
            await asyncio.sleep(0.6)
            pane = await self.capture_pane(window_id) or ""
            tail = "".join(pane.rstrip().split("\n")[-3:])
            # Prompt themes may wrap or decorate the line; compare without spaces
            if command.replace(" ", "") in tail.replace(" ", ""):
                return await self.send_key(window_id, "Enter")
            logger.warning(
                "Typed command not visible in %s (attempt %d)", window_id, attempt + 1
            )
        return False

    async def send_keys(
        self, window_id: str, text: str, enter: bool = True, literal: bool = True
    ) -> bool:
        """Send keys to a specific window.

        Args:
            window_id: The window ID to send to
            text: Text to send
            enter: Whether to press enter after the text
            literal: If True, send text literally. If False, interpret special keys
                     like "Up", "Down", "Left", "Right", "Escape", "Enter".

        Returns:
            True if successful, False otherwise
        """
        if literal and enter:
            # Split into text + delay + Enter via libtmux.
            # Claude Code's TUI sometimes interprets a rapid-fire Enter
            # (arriving in the same input batch as the text) as a newline
            # rather than submit.  A 500ms gap lets the TUI process the
            # text before receiving Enter.
            def _send_literal(chars: str) -> bool:
                session = self.get_session()
                if not session:
                    logger.error("No tmux session found")
                    return False
                try:
                    window = session.windows.get(window_id=window_id)
                    if not window:
                        logger.error(f"Window {window_id} not found")
                        return False
                    pane = window.active_pane
                    if not pane:
                        logger.error(f"No active pane in window {window_id}")
                        return False
                    pane.send_keys(chars, enter=False, literal=True)
                    return True
                except Exception as e:
                    logger.error(f"Failed to send keys to window {window_id}: {e}")
                    return False

            def _send_enter() -> bool:
                session = self.get_session()
                if not session:
                    return False
                try:
                    window = session.windows.get(window_id=window_id)
                    if not window:
                        return False
                    pane = window.active_pane
                    if not pane:
                        return False
                    pane.send_keys("", enter=True, literal=False)
                    return True
                except Exception as e:
                    logger.error(f"Failed to send Enter to window {window_id}: {e}")
                    return False

            # Claude Code's ! command mode: send "!" first so the TUI
            # switches to bash mode, wait 1s, then send the rest.
            if text.startswith("!"):
                if not await asyncio.to_thread(_send_literal, "!"):
                    return False
                rest = text[1:]
                if rest:
                    await asyncio.sleep(1.0)
                    if not await asyncio.to_thread(_send_literal, rest):
                        return False
            else:
                if not await asyncio.to_thread(_send_literal, text):
                    return False
            await asyncio.sleep(0.5)
            return await asyncio.to_thread(_send_enter)

        # Other cases: special keys (literal=False) or no-enter
        def _sync_send_keys() -> bool:
            session = self.get_session()
            if not session:
                logger.error("No tmux session found")
                return False

            try:
                window = session.windows.get(window_id=window_id)
                if not window:
                    logger.error(f"Window {window_id} not found")
                    return False

                pane = window.active_pane
                if not pane:
                    logger.error(f"No active pane in window {window_id}")
                    return False

                pane.send_keys(text, enter=enter, literal=literal)
                return True

            except Exception as e:
                logger.error(f"Failed to send keys to window {window_id}: {e}")
                return False

        return await asyncio.to_thread(_sync_send_keys)

    async def rename_window(self, window_id: str, new_name: str) -> bool:
        """Rename a tmux window by its ID."""

        def _sync_rename() -> bool:
            session = self.get_session()
            if not session:
                return False
            try:
                window = session.windows.get(window_id=window_id)
                if not window:
                    return False
                window.rename_window(new_name)
                self.invalidate_windows_cache()
                logger.info("Renamed window %s to '%s'", window_id, new_name)
                return True
            except Exception as e:
                logger.error(f"Failed to rename window {window_id}: {e}")
                return False

        return await asyncio.to_thread(_sync_rename)

    async def kill_window(self, window_id: str) -> bool:
        """Kill a tmux window by its ID."""

        def _sync_kill() -> bool:
            session = self.get_session()
            if not session:
                return False
            try:
                window = session.windows.get(window_id=window_id)
                if not window:
                    return False
                window.kill()
                self.invalidate_windows_cache()
                logger.info("Killed window %s", window_id)
                return True
            except Exception as e:
                logger.error(f"Failed to kill window {window_id}: {e}")
                return False

        return await asyncio.to_thread(_sync_kill)

    async def create_window(
        self,
        work_dir: str,
        window_name: str | None = None,
        start_claude: bool = True,
        resume_session_id: str | None = None,
        mode: str = "default",
    ) -> tuple[bool, str, str, str]:
        """Create a new tmux window and optionally start Claude Code.

        Args:
            work_dir: Working directory for the new window
            window_name: Optional window name (defaults to directory name)
            start_claude: Whether to start claude command
            resume_session_id: If set, append --resume <id> to claude command
            mode: Launch mode key from LAUNCH_MODES

        Returns:
            Tuple of (success, message, window_name, window_id)
        """
        # Validate directory first
        path = Path(work_dir).expanduser().resolve()
        if not path.exists():
            return False, f"Directory does not exist: {work_dir}", "", ""
        if not path.is_dir():
            return False, f"Not a directory: {work_dir}", "", ""

        # resume_session_id is interpolated into a shell command line below;
        # it comes from JSONL filenames on disk, but validate defensively —
        # Claude session IDs are always UUIDs.
        if resume_session_id and not _UUID_RE.fullmatch(resume_session_id):
            logger.error("Rejecting non-UUID resume_session_id: %r", resume_session_id)
            return False, "Invalid session ID for resume", "", ""

        # Create window name, adding suffix if name already exists
        final_window_name = window_name if window_name else path.name

        # Check for existing window name
        base_name = final_window_name
        counter = 2
        while await self.find_window_by_name(final_window_name):
            final_window_name = f"{base_name}-{counter}"
            counter += 1

        # Create window in thread
        def _create_and_start() -> tuple[bool, str, str, str]:
            session = self.get_or_create_session()
            try:
                # Create new window
                window = session.new_window(
                    window_name=final_window_name,
                    start_directory=str(path),
                )

                wid = window.window_id or ""

                # Prevent Claude Code from overriding window name
                window.set_window_option("allow-rename", "off")

                # Start Claude Code if requested
                if start_claude:
                    pane = window.active_pane
                    if pane:
                        cmd = build_claude_command(mode, resume_session_id)
                        pane.send_keys(cmd, enter=True)

                self.invalidate_windows_cache()
                logger.info(
                    "Created window '%s' (id=%s) at %s",
                    final_window_name,
                    wid,
                    path,
                )
                return (
                    True,
                    f"Created window '{final_window_name}' at {path}",
                    final_window_name,
                    wid,
                )

            except Exception as e:
                logger.error(f"Failed to create window: {e}")
                return False, f"Failed to create window: {e}", "", ""

        return await asyncio.to_thread(_create_and_start)


# Global instance with default session name
tmux_manager = TmuxManager()
