"""Unit tests for SessionMonitor: JSONL reading, transcript lookup, poll cycles."""

import asyncio
import json

import pytest

from ccbot.monitor_state import TrackedSession
from ccbot.session_monitor import SessionMonitor


class TestReadNewLinesOffsetRecovery:
    """Tests for _read_new_lines offset corruption recovery."""

    @pytest.fixture
    def monitor(self, tmp_path):
        """Create a SessionMonitor with temp state file."""
        return SessionMonitor(
            projects_path=tmp_path / "projects",
            state_file=tmp_path / "monitor_state.json",
        )

    @pytest.mark.asyncio
    async def test_mid_line_offset_recovery(self, monitor, tmp_path, make_jsonl_entry):
        """Recover from corrupted offset pointing mid-line."""
        # Create JSONL file with two valid lines
        jsonl_file = tmp_path / "session.jsonl"
        entry1 = make_jsonl_entry(msg_type="assistant", content="first message")
        entry2 = make_jsonl_entry(msg_type="assistant", content="second message")
        jsonl_file.write_text(
            json.dumps(entry1) + "\n" + json.dumps(entry2) + "\n",
            encoding="utf-8",
        )

        # Calculate offset pointing into the middle of line 1
        line1_bytes = len(json.dumps(entry1).encode("utf-8")) // 2
        session = TrackedSession(
            session_id="test-session",
            file_path=str(jsonl_file),
            last_byte_offset=line1_bytes,  # Mid-line (corrupted)
        )

        # Read should recover and return empty (offset moved to next line)
        result = await monitor._read_new_lines(session, jsonl_file)

        # Should return empty list (recovery skips to next line, no new content yet)
        assert result == []

        # Offset should now point to start of line 2
        line1_full = len(json.dumps(entry1).encode("utf-8")) + 1  # +1 for newline
        assert session.last_byte_offset == line1_full

    @pytest.mark.asyncio
    async def test_valid_offset_reads_normally(
        self, monitor, tmp_path, make_jsonl_entry
    ):
        """Normal reading when offset points to line start."""
        jsonl_file = tmp_path / "session.jsonl"
        entry1 = make_jsonl_entry(msg_type="assistant", content="first")
        entry2 = make_jsonl_entry(msg_type="assistant", content="second")
        jsonl_file.write_text(
            json.dumps(entry1) + "\n" + json.dumps(entry2) + "\n",
            encoding="utf-8",
        )

        # Offset at 0 should read both lines
        session = TrackedSession(
            session_id="test-session",
            file_path=str(jsonl_file),
            last_byte_offset=0,
        )

        result = await monitor._read_new_lines(session, jsonl_file)

        assert len(result) == 2
        assert session.last_byte_offset == jsonl_file.stat().st_size

    @pytest.mark.asyncio
    async def test_poison_line_skipped(self, monitor, tmp_path, make_jsonl_entry):
        """A complete (newline-terminated) but corrupt line must be skipped.

        Regression: previously any unparseable line was treated as a partial
        write and retried forever, permanently stalling the session.
        """
        jsonl_file = tmp_path / "session.jsonl"
        good = make_jsonl_entry(msg_type="assistant", content="ok")
        jsonl_file.write_text(
            json.dumps(good) + "\n" + "{corrupt json!!!\n" + json.dumps(good) + "\n",
            encoding="utf-8",
        )
        session = TrackedSession(
            session_id="test-session",
            file_path=str(jsonl_file),
            last_byte_offset=0,
        )

        result = await monitor._read_new_lines(session, jsonl_file)

        # Both good lines returned; offset advanced past the poison line
        assert len(result) == 2
        assert session.last_byte_offset == jsonl_file.stat().st_size

    @pytest.mark.asyncio
    async def test_partial_line_at_eof_retried(
        self, monitor, tmp_path, make_jsonl_entry
    ):
        """An unterminated line at EOF is a partial write — retry next cycle."""
        jsonl_file = tmp_path / "session.jsonl"
        good = make_jsonl_entry(msg_type="assistant", content="ok")
        good_line = json.dumps(good) + "\n"
        jsonl_file.write_text(good_line + '{"type": "assis', encoding="utf-8")
        session = TrackedSession(
            session_id="test-session",
            file_path=str(jsonl_file),
            last_byte_offset=0,
        )

        result = await monitor._read_new_lines(session, jsonl_file)

        assert len(result) == 1
        # Offset stops at the partial line, not EOF
        assert session.last_byte_offset == len(good_line.encode("utf-8"))

        # Writer completes the line — next cycle picks it up
        with open(jsonl_file, "a", encoding="utf-8") as f:
            f.write('tant", "message": {"content": "done"}}\n')
        result2 = await monitor._read_new_lines(session, jsonl_file)
        assert len(result2) == 1
        assert session.last_byte_offset == jsonl_file.stat().st_size

    @pytest.mark.asyncio
    async def test_truncation_detection(self, monitor, tmp_path, make_jsonl_entry):
        """Detect file truncation and reset offset."""
        jsonl_file = tmp_path / "session.jsonl"
        entry = make_jsonl_entry(msg_type="assistant", content="content")
        jsonl_file.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        # Set offset beyond file size (simulates truncation)
        session = TrackedSession(
            session_id="test-session",
            file_path=str(jsonl_file),
            last_byte_offset=9999,  # Beyond file size
        )

        result = await monitor._read_new_lines(session, jsonl_file)

        # Should reset offset to 0 and read the line
        assert session.last_byte_offset == jsonl_file.stat().st_size
        assert len(result) == 1


class TestFreshSessionTracking:
    """Sessions created while the monitor runs are tracked from byte 0."""

    @pytest.fixture
    def monitor(self, tmp_path, monkeypatch):
        from ccbot import session_monitor as sm
        from ccbot.session import session_manager

        monkeypatch.setattr(session_manager, "window_states", {})
        projects = tmp_path / "projects"
        projects.mkdir()
        monkeypatch.setattr(
            sm.config, "session_map_file", tmp_path / "session_map.json"
        )
        monkeypatch.setattr(sm.config, "tmux_session_name", "ccbot")
        return sm.SessionMonitor(
            projects_path=projects,
            state_file=tmp_path / "monitor_state.json",
        )

    @staticmethod
    def _write_map(path, entries: dict[str, str]) -> None:
        path.write_text(
            json.dumps(
                {
                    f"ccbot:{wid}": {"session_id": sid, "cwd": "/proj"}
                    for wid, sid in entries.items()
                }
            )
        )

    @pytest.mark.asyncio
    async def test_new_session_without_file_starts_at_zero(
        self, monitor, tmp_path, make_jsonl_entry
    ):
        map_file = tmp_path / "session_map.json"
        self._write_map(map_file, {})
        await monitor._detect_and_cleanup_changes()

        # Hook fires before Claude writes the transcript
        self._write_map(map_file, {"@1": "fresh-sid"})
        await monitor._detect_and_cleanup_changes()
        assert "fresh-sid" in monitor._fresh_sessions

        # Transcript appears with content between polls
        jsonl = monitor.projects_path / "-proj" / "fresh-sid.jsonl"
        jsonl.parent.mkdir()
        jsonl.write_text(
            json.dumps(make_jsonl_entry(msg_type="assistant", content="hi")) + "\n"
        )

        monitor._transcript_retry_at.clear()  # skip the 5 s negative-cache wait
        await monitor.check_for_updates({"fresh-sid"})

        tracked = monitor.state.get_session("fresh-sid")
        assert tracked is not None
        assert tracked.last_byte_offset == 0
        assert "fresh-sid" not in monitor._fresh_sessions

    @pytest.mark.asyncio
    async def test_existing_file_starts_at_eof(
        self, monitor, tmp_path, make_jsonl_entry
    ):
        map_file = tmp_path / "session_map.json"
        self._write_map(map_file, {})
        await monitor._detect_and_cleanup_changes()

        # Resumed session: transcript already exists when the hook fires
        jsonl = monitor.projects_path / "-proj" / "old-sid.jsonl"
        jsonl.parent.mkdir()
        jsonl.write_text(
            json.dumps(make_jsonl_entry(msg_type="assistant", content="old")) + "\n"
        )
        self._write_map(map_file, {"@1": "old-sid"})
        await monitor._detect_and_cleanup_changes()
        assert "old-sid" not in monitor._fresh_sessions

        await monitor.check_for_updates({"old-sid"})

        tracked = monitor.state.get_session("old-sid")
        assert tracked is not None
        assert tracked.last_byte_offset == jsonl.stat().st_size


@pytest.fixture
def lookup_monitor(tmp_path, monkeypatch):
    """Monitor with isolated session_map / window_states for lookup + poll tests."""
    from ccbot import session_monitor as sm
    from ccbot.session import session_manager

    monkeypatch.setattr(session_manager, "window_states", {})
    monkeypatch.setattr(sm.config, "session_map_file", tmp_path / "session_map.json")
    monkeypatch.setattr(sm.config, "tmux_session_name", "ccbot")
    projects = tmp_path / "projects"
    projects.mkdir()
    return sm.SessionMonitor(
        projects_path=projects, state_file=tmp_path / "monitor_state.json"
    )


class TestTranscriptLookup:
    """Transcripts are found for active session ids only, without a dir scan."""

    def test_glob_hit_is_cached(self, lookup_monitor):
        m = lookup_monitor
        f = m.projects_path / "-proj" / "sid-1.jsonl"
        f.parent.mkdir()
        f.write_text("{}\n")

        assert m._find_transcript("sid-1", {}) == f
        # Second lookup must not glob again
        m.projects_path = m.projects_path / "gone"
        assert m._find_transcript("sid-1", {}) == f

    def test_cached_path_deleted_is_dropped(self, lookup_monitor):
        m = lookup_monitor
        f = m.projects_path / "-proj" / "sid-1.jsonl"
        f.parent.mkdir()
        f.write_text("{}\n")
        assert m._find_transcript("sid-1", {}) == f
        f.unlink()
        assert m._find_transcript("sid-1", {}) is None
        assert "sid-1" not in m._transcript_cache

    def test_hook_path_wins_over_glob(self, lookup_monitor, tmp_path):
        m = lookup_monitor
        elsewhere = tmp_path / "renamed-dir" / "sid-1.jsonl"
        elsewhere.parent.mkdir()
        elsewhere.write_text("{}\n")
        globbed = m.projects_path / "-proj" / "sid-1.jsonl"
        globbed.parent.mkdir()
        globbed.write_text("{}\n")

        assert m._find_transcript("sid-1", {"sid-1": str(elsewhere)}) == elsewhere

    def test_negative_cache_limits_glob_retries(self, lookup_monitor, monkeypatch):
        from pathlib import Path

        from ccbot import session_monitor as sm

        m = lookup_monitor
        now = [1000.0]
        monkeypatch.setattr(sm.time, "monotonic", lambda: now[0])
        globs: list[str] = []
        real_glob = Path.glob

        def counting_glob(self, pattern, *args, **kwargs):
            globs.append(pattern)
            return real_glob(self, pattern, *args, **kwargs)

        monkeypatch.setattr(Path, "glob", counting_glob)

        assert m._find_transcript("sid-1", {}) is None
        assert len(globs) == 1
        # File appears, but we're inside the retry window: no glob, no hit
        f = m.projects_path / "-proj" / "sid-1.jsonl"
        f.parent.mkdir()
        f.write_text("{}\n")
        now[0] += sm.TRANSCRIPT_RETRY_SECONDS - 0.1
        assert m._find_transcript("sid-1", {}) is None
        assert len(globs) == 1
        # Window elapsed: retried and found
        now[0] += 0.2
        assert m._find_transcript("sid-1", {}) == f
        assert len(globs) == 2
        assert "sid-1" not in m._transcript_retry_at

    def test_only_active_sessions_are_resolved(self, lookup_monitor):
        m = lookup_monitor
        for sid in ("active", "other"):
            f = m.projects_path / "-proj" / f"{sid}.jsonl"
            f.parent.mkdir(exist_ok=True)
            f.write_text("{}\n")

        found = m._find_session_files({"active", "missing"})

        assert [s.session_id for s in found] == ["active"]
        assert "other" not in m._transcript_cache

    def test_inactive_session_state_pruned(self, lookup_monitor):
        m = lookup_monitor
        m._transcript_cache["old"] = m.projects_path / "old.jsonl"
        m._transcript_retry_at["gone"] = 1.0
        m._find_session_files({"new"})
        assert "old" not in m._transcript_cache
        assert "gone" not in m._transcript_retry_at

    @pytest.mark.asyncio
    async def test_check_for_updates_uses_hook_path(
        self, lookup_monitor, tmp_path, make_jsonl_entry
    ):
        from ccbot.session import WindowState, session_manager

        m = lookup_monitor
        f = tmp_path / "somewhere" / "sid-h.jsonl"
        f.parent.mkdir()
        f.write_text("")
        session_manager.window_states["@1"] = WindowState(
            session_id="sid-h", transcript_path=str(f)
        )
        await m.check_for_updates({"sid-h"})  # starts tracking at EOF
        f.write_text(
            json.dumps(make_jsonl_entry(msg_type="assistant", content="hi")) + "\n"
        )

        msgs = await m.check_for_updates({"sid-h"})

        assert [x.text for x in msgs] == ["hi"]


class TestPollOnce:
    """poll_once runs a full cycle under a lock; poll_now waits for it."""

    @pytest.mark.asyncio
    async def test_poll_once_delivers_messages_in_order(
        self, lookup_monitor, tmp_path, monkeypatch, make_jsonl_entry
    ):
        from ccbot.session import session_manager

        m = lookup_monitor
        f = m.projects_path / "-proj" / "sid-1.jsonl"
        f.parent.mkdir()
        f.write_text("")
        (tmp_path / "session_map.json").write_text(
            json.dumps({"ccbot:@1": {"session_id": "sid-1", "cwd": "/proj"}})
        )

        async def no_load():
            return None

        monkeypatch.setattr(session_manager, "load_session_map", no_load)
        got: list[str] = []

        async def cb(msg):
            got.append(msg.text)

        m.set_message_callback(cb)
        await m.poll_once()  # begins tracking at EOF
        f.write_text(
            "".join(
                json.dumps(make_jsonl_entry(msg_type="assistant", content=t)) + "\n"
                for t in ("one", "two", "three")
            )
        )
        await m.poll_once()

        assert got == ["one", "two", "three"]

    @pytest.mark.asyncio
    async def test_cycles_never_overlap_and_poll_now_runs_fresh_cycle(
        self, lookup_monitor, monkeypatch
    ):
        from ccbot.session import session_manager

        m = lookup_monitor
        events: list[str] = []
        release = asyncio.Event()
        running = 0
        max_running = 0

        async def slow_load():
            nonlocal running, max_running
            running += 1
            max_running = max(max_running, running)
            events.append("start")
            if len(events) == 1:
                await release.wait()  # first cycle blocks until released
            events.append("end")
            running -= 1

        monkeypatch.setattr(session_manager, "load_session_map", slow_load)

        first = asyncio.create_task(m.poll_once())
        await asyncio.sleep(0.01)
        second = asyncio.create_task(m.poll_now())
        await asyncio.sleep(0.01)
        # Second cycle is waiting on the lock, not running concurrently
        assert events == ["start"]
        release.set()
        await asyncio.gather(first, second)

        assert events == ["start", "end", "start", "end"]
        assert max_running == 1

    @pytest.mark.asyncio
    async def test_poll_once_swallows_errors(self, lookup_monitor, monkeypatch):
        from ccbot.session import session_manager

        async def boom():
            raise RuntimeError("nope")

        monkeypatch.setattr(session_manager, "load_session_map", boom)
        await lookup_monitor.poll_once()  # must not raise
        # lock released: a later cycle can still run
        await lookup_monitor.poll_now()
