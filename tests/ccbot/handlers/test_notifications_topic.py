"""Tests for the notifications topic: gating, dedupe, turn-done and UI events."""

from unittest.mock import AsyncMock

import pytest

import ccbot.handlers.notifications_topic as nt
from ccbot.config import config
from ccbot.session import SessionManager


@pytest.fixture
def ready(monkeypatch):
    monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
    monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
    mgr = SessionManager()
    mgr.bind_thread(1, 55, "@3", window_name="proj")
    monkeypatch.setattr(nt, "session_manager", mgr)
    send = AsyncMock()
    monkeypatch.setattr(nt, "safe_send", send)
    monkeypatch.setattr(nt, "_s", nt._State())
    nt._s.bot, nt._s.chat_id, nt._s.thread_id = object(), -1001234567890, 99
    for attr in (
        "notify_needs_input",
        "notify_turn_done",
        "notify_lifecycle",
        "notify_errors",
        "notify_ccc",
    ):
        monkeypatch.setattr(config, attr, True)
    monkeypatch.setattr(config, "notify_turn_min", 0.0)
    monkeypatch.setattr(config, "quiet_hours", "")
    monkeypatch.setattr(nt, "TURN_DONE_DELAY", 0)
    return send


class TestNotify:
    @pytest.mark.asyncio
    async def test_sends_with_topic_link(self, ready):
        assert await nt.notify("done", "finished", "@3")
        kwargs = ready.await_args.kwargs
        assert kwargs["message_thread_id"] == 99
        text = ready.await_args.args[2]
        assert "proj" in text and "finished" in text
        url = kwargs["reply_markup"].inline_keyboard[0][0].url
        assert url == "https://t.me/c/1234567890/55"

    @pytest.mark.asyncio
    async def test_message_link(self, ready):
        assert await nt.notify("input", "q", "@3", message_id=777)
        url = ready.await_args.kwargs["reply_markup"].inline_keyboard[0][0].url
        assert url == "https://t.me/c/1234567890/55/777"

    @pytest.mark.asyncio
    async def test_turn_done_links_last_reply(self, ready):
        await nt.record_sent("@3", "text", 4242, "All done.")
        await nt.mark_working("@3", True, None)
        await nt.mark_working("@3", False, None)
        url = ready.await_args.kwargs["reply_markup"].inline_keyboard[0][0].url
        assert url.endswith("/55/4242")

    @pytest.mark.asyncio
    async def test_merged_thinking_message_counts_as_reply(self, ready):
        await nt.record_sent("@3", "thinking", 4300, "…")
        await nt.mark_working("@3", True, None)
        await nt.mark_working("@3", False, None)
        url = ready.await_args.kwargs["reply_markup"].inline_keyboard[0][0].url
        assert url.endswith("/55/4300")

    @pytest.mark.asyncio
    async def test_error_notified_from_record_sent(self, ready):
        await nt.record_sent("@3", "error", 9, "🚨 You've hit your session limit")
        assert ready.await_count == 1
        assert "session limit" in ready.await_args.args[2]
        assert (
            ready.await_args.kwargs["reply_markup"]
            .inline_keyboard[0][0]
            .url.endswith("/9")
        )

    @pytest.mark.asyncio
    async def test_disabled_kind(self, ready, monkeypatch):
        monkeypatch.setattr(config, "notify_errors", False)
        assert not await nt.notify("error", "boom", "@3")
        ready.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_quiet_hours(self, ready, monkeypatch):
        monkeypatch.setattr(nt, "in_quiet_hours", lambda: True)
        assert not await nt.notify("input", "q", "@3")

    @pytest.mark.asyncio
    async def test_dedupe(self, ready):
        assert await nt.notify("error", "same", "@3", signature="x")
        assert not await nt.notify("error", "same again", "@3", signature="x")
        assert ready.await_count == 1

    @pytest.mark.asyncio
    async def test_not_ready(self, ready):
        nt._s.thread_id = None
        assert not await nt.notify("done", "x", "@3")


class TestTurnDone:
    @pytest.mark.asyncio
    async def test_busy_then_idle_fires_once(self, ready):
        nt.record_assistant_text("@3", "All done.\nmore")
        await nt.mark_working("@3", True, None)
        await nt.mark_working("@3", True, None)
        await nt.mark_working("@3", False, "Worked for 3s")
        await nt.mark_working("@3", False, "Worked for 3s")
        assert ready.await_count == 1
        text = ready.await_args.args[2]
        assert "finished" in text and "Worked for 3s" in text and "All done." in text

    @pytest.mark.asyncio
    async def test_idle_without_prior_busy_is_silent(self, ready):
        await nt.mark_working("@3", False, "Worked for 3s")
        ready.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_min_duration(self, ready, monkeypatch):
        monkeypatch.setattr(config, "notify_turn_min", 3600.0)
        await nt.mark_working("@3", True, None)
        await nt.mark_working("@3", False, None)
        ready.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dialog_pauses_instead_of_finishing(self, ready):
        await nt.mark_working("@3", True, None)
        await nt.mark_working("@3", False, None, paused=True)
        ready.assert_not_awaited()
        await nt.mark_working("@3", False, "done")
        assert ready.await_count == 1


class TestUi:
    @pytest.mark.asyncio
    async def test_announced_once_until_cleared(self, ready):
        await nt.mark_ui("@3", "AskUserQuestion", "Which db?\n☐ a\n☐ b")
        await nt.mark_ui("@3", "AskUserQuestion", "Which db?\n☐ a\n☐ b")
        assert ready.await_count == 1
        assert "Which db?" in ready.await_args.args[2]
        await nt.mark_ui("@3", None)
        await nt.mark_ui("@3", "AskUserQuestion", "Which db?\n☐ a\n☐ b")
        assert ready.await_count == 2

    @pytest.mark.asyncio
    async def test_permission_label(self, ready):
        await nt.mark_ui("@3", "PermissionPrompt", "Do you want to proceed?")
        assert "permission needed" in ready.await_args.args[2]
