"""Unit tests for SessionMonitor JSONL reading and offset handling."""

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
        from ccbot import session_monitor as sm

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

        async def fake_scan():
            return [sm.SessionInfo(session_id="fresh-sid", file_path=jsonl)]

        monitor.scan_projects = fake_scan  # type: ignore[method-assign]
        await monitor.check_for_updates({"fresh-sid"})

        tracked = monitor.state.get_session("fresh-sid")
        assert tracked is not None
        assert tracked.last_byte_offset == 0
        assert "fresh-sid" not in monitor._fresh_sessions

    @pytest.mark.asyncio
    async def test_existing_file_starts_at_eof(
        self, monitor, tmp_path, make_jsonl_entry
    ):
        from ccbot import session_monitor as sm

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

        async def fake_scan():
            return [sm.SessionInfo(session_id="old-sid", file_path=jsonl)]

        monitor.scan_projects = fake_scan  # type: ignore[method-assign]
        await monitor.check_for_updates({"old-sid"})

        tracked = monitor.state.get_session("old-sid")
        assert tracked is not None
        assert tracked.last_byte_offset == jsonl.stat().st_size
