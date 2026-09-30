"""Tests for the ccc special topic: parsing, dashboard, quota watcher, client."""

import json
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import ccbot.handlers.ccc_topic as cc
from ccbot.handlers import callback_data
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


def _buttons(kb):
    return [b for row in kb.inline_keyboard for b in row]


def _data(kb):
    return [b.callback_data for b in _buttons(kb)]


def _line(text: str, marker: str) -> str:
    return next(ln for ln in text.splitlines() if marker in ln)


class TestRender:
    def test_home_lists_all_providers_with_in_use_account(self, accounts):
        text, kb = cc.render_dashboard(accounts)
        claude = _line(text, "**Claude**")
        codex = _line(text, "**Codex**")
        grok = _line(text, "**Grok**")
        assert "✳" in claude and "`i***@g…` (pro)" in claude and "5h" in claude
        assert "❉" in codex and "(plus)" in codex and "7 free" in codex
        assert "✦" in grok and "`g***@x…` (SuperGrok)" in grok and "7d 100%" in grok
        assert "_updated " in text and text.endswith("UTC_")
        convert_markdown(text)  # MarkdownV2 must not raise
        # one button per provider, then refresh / restart
        assert [[b.text for b in row] for row in kb.inline_keyboard] == [
            ["✳ Claude ›", "❉ Codex ›", "✦ Grok ›"],
            ["🔄 Refresh", "🔁 Restart Claude sessions"],
        ]
        assert _data(kb) == [
            f"{cc.CB_CCC_PROV}claude",
            f"{cc.CB_CCC_PROV}codex",
            f"{cc.CB_CCC_PROV}grok",
            f"{cc.CB_CCC_REFRESH}net",
            cc.CB_CCC_RESTART,
        ]

    def test_home_shows_reset_only_below_full(self, accounts):
        grok = next(a for a in accounts if a.provider == "grok" and a.current)
        grok.windows[0].resets_at = datetime.now(timezone.utc) + timedelta(hours=3)
        grok.windows[0].remaining = 100
        text, _ = cc.render_dashboard(accounts)
        assert "↻" not in _line(text, "**Grok**")
        grok.windows[0].remaining = 40
        text, _ = cc.render_dashboard(accounts)
        line = _line(text, "**Grok**")
        assert "7d ▕" in line and "40%" in line and "↻ 2h5" in line

    def test_home_flags_exhausted_in_use_account(self, accounts):
        grok = next(a for a in accounts if a.provider == "grok" and a.current)
        grok.windows[0].remaining = 0
        text, _ = cc.render_dashboard(accounts)
        assert "⛔" in _line(text, "**Grok**")
        convert_markdown(text)

    def test_home_without_accounts(self):
        text, kb = cc.render_dashboard([])
        assert "No accounts" in text
        assert _data(kb) == [f"{cc.CB_CCC_REFRESH}net", cc.CB_CCC_RESTART]
        convert_markdown(text)

    def test_provider_page_lists_accounts_and_buttons(self, accounts):
        text, kb = cc.render_provider(accounts, "grok")
        assert text.startswith("✦ **Grok** — 7 accounts")
        for name in ("g***@x***.com", "k***@y***.io", "q***@x***.org"):
            assert name in text
        assert "```" in text and "reauth_required" in text and "stale" in text
        assert all(len(ln) <= 34 for ln in text.splitlines() if not ln.startswith("_"))
        convert_markdown(text)
        # use buttons for every non-current account, none for the one in use
        data = _data(kb)
        for a in (a for a in accounts if a.provider == "grok"):
            assert (f"{cc.CB_CCC_USE}{a.id}" in data) == (not a.current)
        assert data[-2:] == [f"{cc.CB_CCC_NEXT}grok", cc.CB_CCC_HOME]
        texts = [b.text for b in _buttons(kb)]
        assert "▶ k***@y***.io ⛔" in texts  # exhausted
        assert texts[-2:] == ["⏭ Next Grok", "« Back"]
        assert all(len(row) <= 2 for row in kb.inline_keyboard)

    def test_provider_page_free_summary(self, accounts):
        text, kb = cc.render_provider(accounts, "codex")
        assert "○ 7 free accounts (6 with quota, 1 need login)" in text
        # free accounts get no button of their own
        free_ids = {a.id for a in accounts if a.free}
        assert not any(d == f"{cc.CB_CCC_USE}{i}" for i in free_ids for d in _data(kb))
        assert "🎟2" in text
        convert_markdown(text)

    def test_current_free_account_is_listed(self, accounts):
        free = next(a for a in accounts if a.free and a.status == "ready")
        free.current = True
        text, kb = cc.render_provider(accounts, "codex")
        assert f"● {free.name}" in text
        assert "○ 6 free accounts" in text
        assert f"{cc.CB_CCC_USE}{free.id}" not in _data(kb)

    def test_unknown_provider_gets_fallback_section(self, accounts):
        odd = cc.parse_accounts(
            {
                "accounts": [
                    {
                        "account": {
                            "id": "x1",
                            "provider": "mistral",
                            "name": "a***@b.com",
                            "status": "ready",
                        },
                        "current": True,
                        "quota": {
                            "plan": "pro",
                            "windows": [
                                {"name": "some_odd_window", "remainingPercent": 55}
                            ],
                        },
                    }
                ]
            }
        )
        text, kb = cc.render_dashboard(accounts + odd)
        line = _line(text, "Mistral")
        assert line.startswith("• **Mistral**") and "55%" in line
        assert f"{cc.CB_CCC_PROV}mistral" in _data(kb)
        ptext, pkb = cc.render_provider(odd, "mistral")
        assert "• **Mistral** — 1 account" in ptext and "some odd window" in ptext
        assert f"{cc.CB_CCC_NEXT}mistral" in _data(pkb)
        convert_markdown(text)
        convert_markdown(ptext)

    def test_unknown_provider_page_is_just_back(self, accounts):
        text, kb = cc.render_provider(accounts, "nope")
        assert "no accounts" in text
        assert _data(kb) == [cc.CB_CCC_HOME]

    def test_float_remaining_is_shown_as_int(self, accounts):
        floaty = next(a for a in accounts if a.name == "n***@z***.net")
        assert floaty.windows[0].remaining == 99
        assert isinstance(floaty.windows[0].remaining, int)
        text, _ = cc.render_provider(accounts, "grok")
        assert "98.5" not in text and "99%" in text

    def test_pct_parsing(self):
        assert cc._pct(98.581362) == 99
        assert cc._pct("42") == 42
        assert cc._pct(None) is None and cc._pct(True) is None
        assert cc._pct("n/a") is None and cc._pct([1]) is None
        assert cc._pct(-3) == 0 and cc._pct(140) == 100

    def test_main_windows_fallbacks(self, accounts):
        claude = next(a for a in accounts if a.provider == "claude" and not a.current)
        assert {w.name for w in claude.windows} >= {"iguana_necktie", "five_hour"}
        assert [w.name for w in claude.main_windows] == ["five_hour", "seven_day"]
        only_odd = cc.Account(
            "i",
            "p",
            "n",
            False,
            "ready",
            "pro",
            False,
            windows=[
                cc.Window("weird_one", 30, None),
                cc.Window("weird_two", 20, None),
                cc.Window("weird_three", 10, None),
            ],
        )
        assert [w.name for w in only_odd.main_windows] == ["weird_one", "weird_two"]
        thirty = cc.Account(
            "i",
            "p",
            "n",
            False,
            "ready",
            "free",
            False,
            windows=[cc.Window("weird", 30, None), cc.Window("43200_minute", 30, None)],
        )
        assert [w.label for w in thirty.main_windows] == ["30d"]

    def test_window_labels(self):
        assert cc.Window("300_minute", 1, None).label == "5h"
        assert cc.Window("10080_minute", 1, None).label == "7d"
        assert cc.Window("90_minute", 1, None).label == "90m"
        assert cc.Window("nimbus_quill", 1, None).label == "nimbus quill"

    def test_all_callback_data_within_limit(self, accounts):
        for text, kb in (
            cc.render_dashboard(accounts),
            *(cc.render_provider(accounts, p) for p in ("claude", "codex", "grok")),
        ):
            assert all(len(d.encode()) <= 64 for d in _data(kb))
            convert_markdown(text)

    def test_callback_prefixes_registered_and_disjoint(self):
        prefixes = cc.CccTopic.callback_prefixes
        assert cc.CB_CCC_HOME in prefixes and cc.CB_CCC_PROV in prefixes
        others = [
            v
            for k, v in vars(callback_data).items()
            if k.startswith("CB_") and isinstance(v, str) and v not in prefixes
        ]
        for p in prefixes:
            assert not any(o.startswith(p) or p.startswith(o) for o in others)

    def test_exhausted_flag_and_due(self, accounts):
        a = next(x for x in accounts if x.provider == "claude" and x.current)
        a.windows[0].remaining = 0
        a.windows[0].resets_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        text, _ = cc.render_provider(accounts, "claude")
        assert "⛔" in text
        assert "↻ due" in text
        home, _ = cc.render_dashboard(accounts)
        assert "⛔" in home


class TestCallbacks:
    @pytest.fixture
    def topic(self, accounts, monkeypatch):
        client = MagicMock()
        client.list = AsyncMock(return_value=accounts)
        client.use = AsyncMock(
            return_value={"current": {"name": "h***@x***.com", "provider": "grok"}}
        )
        client.use_next = AsyncMock(return_value={"current": {"name": "d***@g***.com"}})
        edit = AsyncMock()
        monkeypatch.setattr(cc, "safe_edit", edit)
        t = cc.CccTopic(client)
        t._last = accounts
        return t, edit

    @staticmethod
    async def _press(topic, edit, data):
        q = MagicMock()
        q.answer = AsyncMock()
        await topic.handle_callback(q, MagicMock(), {}, data)
        return q, edit.await_args

    @pytest.mark.asyncio
    async def test_provider_page_navigation(self, topic):
        t, edit = topic
        q, call = await self._press(t, edit, f"{cc.CB_CCC_PROV}grok")
        q.answer.assert_awaited_once()
        text = call.args[1]
        assert text.startswith("✦ **Grok**") and "g***@x***.com" in text
        assert cc.CB_CCC_HOME in _data(call.kwargs["reply_markup"])
        t.client.list.assert_awaited_with(cached=True)

    @pytest.mark.asyncio
    async def test_home_navigation(self, topic):
        t, edit = topic
        _, call = await self._press(t, edit, cc.CB_CCC_HOME)
        text = call.args[1]
        assert "**Claude**" in text and "**Grok**" in text
        assert f"{cc.CB_CCC_PROV}grok" in _data(call.kwargs["reply_markup"])

    @pytest.mark.asyncio
    async def test_use_rerenders_provider_page_with_note(self, topic, accounts):
        t, edit = topic
        target = next(a for a in accounts if a.name == "h***@x***.com")
        _, call = await self._press(t, edit, f"{cc.CB_CCC_USE}{target.id}")
        t.client.use.assert_awaited_once_with(target.id)
        text = call.args[1]
        assert text.startswith("✅ Now using `h***@x***.com` (grok).")
        assert "✦ **Grok** — 7 accounts" in text
        assert cc.CB_CCC_HOME in _data(call.kwargs["reply_markup"])
        convert_markdown(text)

    @pytest.mark.asyncio
    async def test_next_rerenders_provider_page(self, topic):
        t, edit = topic
        _, call = await self._press(t, edit, f"{cc.CB_CCC_NEXT}codex")
        t.client.use_next.assert_awaited_once_with("codex")
        text = call.args[1]
        assert "✅ Switched codex to `d***@g***.com`." in text
        assert "❉ **Codex**" in text

    @pytest.mark.asyncio
    async def test_refresh_rerenders_home_uncached(self, topic):
        t, edit = topic
        _, call = await self._press(t, edit, f"{cc.CB_CCC_REFRESH}net")
        t.client.list.assert_awaited_with(cached=False)
        text = call.args[1]
        assert "**Grok**" in text and "_updated " in text

    @pytest.mark.asyncio
    async def test_restart_without_handler(self, topic):
        t, edit = topic
        q, call = await self._press(t, edit, cc.CB_CCC_RESTART)
        assert call is None
        assert q.answer.await_args.kwargs.get("show_alert") is True

    @pytest.mark.asyncio
    async def test_ccc_error_alerts(self, topic):
        t, edit = topic
        t.client.list.side_effect = cc.CccError("boom")
        q, call = await self._press(t, edit, cc.CB_CCC_HOME)
        assert call is None
        assert "boom" in q.answer.await_args.args[0]


class TestAnnounce:
    @pytest.mark.asyncio
    async def test_grok_exhausted_alert(self, accounts, monkeypatch):
        sent = AsyncMock()
        monkeypatch.setattr(cc, "safe_send", sent)
        monkeypatch.setattr(cc.config, "ccc_alerts", True, raising=False)
        monkeypatch.setattr(cc, "in_quiet_hours", lambda: False)
        monkeypatch.setattr("ccbot.handlers.notifications_topic.notify", AsyncMock())
        t = cc.CccTopic(MagicMock())
        t._bot, t._chat_id, t._thread_id = MagicMock(), 1, 2
        grok = next(a for a in accounts if a.provider == "grok" and a.current)
        grok.windows[0].remaining = 0
        await t._announce(cc.QuotaEvent("exhausted", grok), accounts)
        text = sent.await_args.args[2]
        assert "✦ **Grok**" in text and "is exhausted" in text
        assert f"{cc.CB_CCC_NEXT}grok" in _data(sent.await_args.kwargs["reply_markup"])
        convert_markdown(text)


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
        assert len(accs) == 19

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
