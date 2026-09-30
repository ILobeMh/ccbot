"""Tests for TmuxManager.list_windows parsing and caching (tmux is mocked)."""

import asyncio
import subprocess
from unittest.mock import MagicMock

import pytest

import ccbot.tmux_manager as tm


def _completed(stdout: str, rc: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr="")


@pytest.fixture
def mgr(monkeypatch) -> tm.TmuxManager:
    monkeypatch.setattr(tm.config, "tmux_main_window_name", "__main__")
    m = tm.TmuxManager(session_name="ccbot")
    # Session always "exists" unless a test says otherwise
    monkeypatch.setattr(m, "get_session", lambda: MagicMock())
    return m


class TestListWindows:
    @pytest.mark.asyncio
    async def test_parses_active_panes_and_skips_main(self, mgr, monkeypatch):
        out = "\n".join(
            [
                "@0\u241e__main__\u241e/home/mh\u241ezsh\u241e1",
                "@1\u241eproj\u241e/home/mh/proj\u241eclaude\u241e1",
                "@1\u241eproj\u241e/home/mh/proj\u241ezsh\u241e0",  # inactive split
                "@2\u241ename with\u241esp aces\u241e/tmp\u241ezsh\u241e1",  # 6 fields: ignored
            ]
        )
        monkeypatch.setattr(tm.subprocess, "run", lambda *a, **k: _completed(out))
        windows = await mgr.list_windows()
        assert [w.window_id for w in windows] == ["@1"]
        assert windows[0].cwd == "/home/mh/proj"
        assert windows[0].pane_current_command == "claude"

    @pytest.mark.asyncio
    async def test_cached_within_ttl(self, mgr, monkeypatch):
        calls = 0

        def fake_run(*a, **k):
            nonlocal calls
            calls += 1
            return _completed("@1\u241eproj\u241e/p\u241eclaude\u241e1")

        monkeypatch.setattr(tm.subprocess, "run", fake_run)
        await mgr.list_windows()
        await mgr.list_windows()
        assert calls == 1
        mgr.invalidate_windows_cache()
        await mgr.list_windows()
        assert calls == 2

    @pytest.mark.asyncio
    async def test_empty_result_with_dead_session_not_cached(self, mgr, monkeypatch):
        monkeypatch.setattr(mgr, "get_session", lambda: None)
        calls = 0

        def fake_run(*a, **k):
            nonlocal calls
            calls += 1
            return _completed("", rc=1)

        monkeypatch.setattr(tm.subprocess, "run", fake_run)
        assert await mgr.list_windows() == []
        assert await mgr.list_windows() == []
        assert calls == 2  # re-probed, not served from cache

    @pytest.mark.asyncio
    async def test_empty_session_is_cached(self, mgr, monkeypatch):
        calls = 0

        def fake_run(*a, **k):
            nonlocal calls
            calls += 1
            return _completed("@0\u241e__main__\u241e/h\u241ezsh\u241e1")

        monkeypatch.setattr(tm.subprocess, "run", fake_run)
        assert await mgr.list_windows() == []
        assert await mgr.list_windows() == []
        assert calls == 1


class TestBuildClaudeCommand:
    @pytest.fixture(autouse=True)
    def _cfg(self, monkeypatch):
        monkeypatch.setattr(tm.config, "claude_command", "claude")
        monkeypatch.setattr(tm.os, "geteuid", lambda: 1000)

    def test_default_allows_bypass_cycle(self):
        assert tm.build_claude_command("default") == (
            "claude --allow-dangerously-skip-permissions"
        )

    def test_bypass(self):
        assert tm.build_claude_command("bypassPermissions") == (
            "claude --dangerously-skip-permissions"
        )

    def test_plan_with_resume(self):
        sid = "550e8400-e29b-41d4-a716-446655440000"
        assert tm.build_claude_command("plan", sid) == (
            f"claude --permission-mode plan --allow-dangerously-skip-permissions "
            f"--resume {sid}"
        )

    def test_root_never_gets_allow_flag(self, monkeypatch):
        monkeypatch.setattr(tm.os, "geteuid", lambda: 0)
        assert tm.build_claude_command("acceptEdits") == (
            "claude --permission-mode acceptEdits"
        )

    def test_custom_claude_command(self, monkeypatch):
        monkeypatch.setattr(tm.config, "claude_command", "IS_SANDBOX=1 claude")
        assert tm.build_claude_command("bypassPermissions").startswith(
            "IS_SANDBOX=1 claude --dangerously"
        )


class TestNormalizeLaunchMode:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("normal", "default"),
            ("Default", "default"),
            ("accept", "acceptEdits"),
            ("acceptEdits", "acceptEdits"),
            ("plan", "plan"),
            ("bypass", "bypassPermissions"),
            ("yolo", "bypassPermissions"),
            ("nonsense", None),
            ("", None),
            (None, None),
        ],
    )
    def test_aliases(self, raw, expected):
        assert tm.normalize_launch_mode(raw) == expected


class _FakeProc:
    def __init__(self, stdout=b"", stderr=b"", rc=0, hang=False):
        self._out, self._err, self.returncode = stdout, stderr, rc
        self._hang = hang
        self.killed = False

    async def communicate(self):
        if self._hang:
            await asyncio.sleep(3600)
        return self._out, self._err

    def kill(self):
        self.killed = True

    async def wait(self):
        return self.returncode


class TestCapturePane:
    @pytest.fixture
    def spawn(self, monkeypatch):
        calls: list[tuple] = []
        result: dict = {"proc": _FakeProc()}

        async def fake_exec(*args, **kwargs):
            calls.append(args)
            return result["proc"]

        monkeypatch.setattr(tm.asyncio, "create_subprocess_exec", fake_exec)
        return calls, result

    @pytest.mark.asyncio
    async def test_plain_single_fork_trims_trailing_empty_lines(self, mgr, spawn):
        calls, result = spawn
        result["proc"] = _FakeProc(stdout=b"hello  \n\n  indented\nlast\n\n\n")

        out = await mgr.capture_pane("@3")

        assert out == "hello  \n\n  indented\nlast"
        assert calls == [("tmux", "capture-pane", "-p", "-t", "@3")]

    @pytest.mark.asyncio
    async def test_ansi_passes_dash_e_and_keeps_output(self, mgr, spawn):
        calls, result = spawn
        result["proc"] = _FakeProc(stdout=b"\x1b[31mred\x1b[0m\n")

        out = await mgr.capture_pane("@3", with_ansi=True)

        assert out == "\x1b[31mred\x1b[0m\n"
        assert calls == [("tmux", "capture-pane", "-e", "-p", "-t", "@3")]

    @pytest.mark.asyncio
    async def test_nonzero_exit_returns_none(self, mgr, spawn):
        _, result = spawn
        result["proc"] = _FakeProc(stderr=b"can't find window: @9", rc=1)
        assert await mgr.capture_pane("@9") is None

    @pytest.mark.asyncio
    async def test_timeout_kills_process(self, mgr, spawn, monkeypatch):
        _, result = spawn
        proc = _FakeProc(hang=True)
        result["proc"] = proc
        monkeypatch.setattr(tm, "CAPTURE_TIMEOUT_SECONDS", 0.01)

        assert await mgr.capture_pane("@3") is None
        assert proc.killed

    @pytest.mark.asyncio
    async def test_spawn_error_returns_none(self, mgr, monkeypatch):
        async def boom(*a, **k):
            raise FileNotFoundError("tmux")

        monkeypatch.setattr(tm.asyncio, "create_subprocess_exec", boom)
        assert await mgr.capture_pane("@3") is None
