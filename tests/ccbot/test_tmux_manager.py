"""Tests for TmuxManager.list_windows parsing and caching (tmux is mocked)."""

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
                "@0\x1f__main__\x1f/home/mh\x1fzsh\x1f1",
                "@1\x1fproj\x1f/home/mh/proj\x1fclaude\x1f1",
                "@1\x1fproj\x1f/home/mh/proj\x1fzsh\x1f0",  # inactive split
                "@2\x1fname with\x1fsp aces\x1f/tmp\x1fzsh\x1f1",  # 6 fields: ignored
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
            return _completed("@1\x1fproj\x1f/p\x1fclaude\x1f1")

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
            return _completed("@0\x1f__main__\x1f/h\x1fzsh\x1f1")

        monkeypatch.setattr(tm.subprocess, "run", fake_run)
        assert await mgr.list_windows() == []
        assert await mgr.list_windows() == []
        assert calls == 1
