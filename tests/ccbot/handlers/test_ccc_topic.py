"""Tests for the ccc special topic: parsing, dashboard, quota watcher, client."""

import json
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import ccbot.handlers.ccc_topic as cc
from ccbot.markdown_v2 import convert_markdown

FIXTURE = Path(__file__).parent.parent / "fixtures" / "ccc" / "list.json"


@pytest.fixture
def accounts() -> list[cc.Account]:
    return cc.parse_accounts(json.loads(FIXTURE.read_text()))


class TestParse:
    def test_shape(self, accounts):
        claude = [a for a in accounts if a.provider == "claude"]
        assert len(claude) == 2
        cur = next(a for a in claude if a.current)
        assert cur.plan == "pro"
        assert {w.name for w in cur.windows} >= {"five_hour", "seven_day"}
        assert [w.label for w in cur.main_windows] == ["5h", "7d"]
        assert cur.main_windows[0].resets_at is not None
        assert cur.main_windows[0].resets_at.tzinfo is not None

    def test_free_codex_uses_30d_window(self, accounts):
        free = [a for a in accounts if a.plan == "free"]
        assert free and all(a.main_windows[0].label == "30d" for a in free)
        assert not any(a.usable for a in free)

    def test_reset_credits(self, accounts):
        assert any(a.reset_credits > 0 for a in accounts if a.provider == "codex")


class TestRender:
    def test_dashboard_renders_and_converts(self, accounts):
        text, kb = cc.render_dashboard(accounts)
        assert "Claude" in text and "Codex" in text
        assert "free accounts" in text
        convert_markdown(text)  # MarkdownV2 must not raise
        buttons = [b for row in kb.inline_keyboard for b in row]
        data = [b.callback_data for b in buttons]
        # No "use" button for the accounts already in use
        current_ids = {a.id for a in accounts if a.current}
        assert not any(d == f"{cc.CB_CCC_USE}{i}" for i in current_ids for d in data)
        assert f"{cc.CB_CCC_NEXT}claude" in data
        assert cc.CB_CCC_RESTART in data
        assert all(len(d) <= 64 for d in data)

    def test_exhausted_flag_and_due(self, accounts):
        a = next(x for x in accounts if x.provider == "claude" and x.current)
        a.windows[0].remaining = 0
        a.windows[0].resets_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        text, _ = cc.render_dashboard(accounts)
        assert "⛔" in text
        assert "↻ due" in text


class TestWatcher:
    def test_events_only_after_priming(self, accounts):
        w = cc.QuotaWatcher()
        assert w.observe(accounts) == []
        cur = next(a for a in accounts if a.provider == "claude" and a.current)
        cur.windows[0].remaining = 0
        ev = w.observe(accounts)
        assert [(e.kind, e.account.id) for e in ev] == [("exhausted", cur.id)]
        assert w.observe(accounts) == []  # no repeat
        cur.windows[0].remaining = 100
        ev = w.observe(accounts)
        assert [(e.kind, e.account.id) for e in ev] == [("available", cur.id)]

    def test_inactive_account_only_reports_available(self, accounts):
        w = cc.QuotaWatcher()
        w.observe(accounts)
        other = next(a for a in accounts if a.provider == "claude" and not a.current)
        other.windows[0].remaining = 0
        assert w.observe(accounts) == []  # not in use: no "exhausted" alarm
        other.windows[0].remaining = 50
        assert [e.kind for e in w.observe(accounts)] == ["available"]

    def test_next_wakeup_targets_earliest_reset(self, accounts):
        cur = next(a for a in accounts if a.provider == "claude" and a.current)
        cur.windows[0].remaining = 0
        cur.windows[0].resets_at = datetime.now(timezone.utc) + timedelta(minutes=10)
        secs = cc.QuotaWatcher.next_wakeup(accounts, 3600)
        assert 600 < secs < 700
        assert cc.QuotaWatcher.next_wakeup([], 300) == 300


class TestClient:
    @pytest.fixture
    def fake_ccc(self, tmp_path, monkeypatch):
        script = tmp_path / "ccc"
        script.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "list" ]; then echo "warning: x" >&2; cat "$CCC_FIXTURE"; exit 0; fi\n'
            'if [ "$1" = "use" ]; then printf \'{"current":{"name":"%s","provider":"claude"}}\' "$2"; exit 0; fi\n'
            'echo "boom: $1" >&2; exit 2\n'
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("CCC_FIXTURE", str(FIXTURE))
        monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
        return cc.CccClient(str(script))

    @pytest.mark.asyncio
    async def test_list_ignores_stderr_warnings(self, fake_ccc):
        accs = await fake_ccc.list(cached=True)
        assert len(accs) == 12

    @pytest.mark.asyncio
    async def test_use_returns_current(self, fake_ccc):
        r = await fake_ccc.use("abc")
        assert r["current"]["name"] == "abc"

    @pytest.mark.asyncio
    async def test_error_raises(self, fake_ccc):
        with pytest.raises(cc.CccError, match="boom: refresh"):
            await fake_ccc.refresh()

    def test_available(self, fake_ccc):
        assert fake_ccc.available()
        assert not cc.CccClient("/definitely/missing/ccc").available()
