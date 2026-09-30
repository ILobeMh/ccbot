"""Tests for message_queue ordering of content vs. interactive UIs.

The question / permission UI must never overtake the content that led to
it: it is a queue task processed in order, and content that still arrives
while a UI is open re-posts the UI below it (or deletes it once answered).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from ccbot.handlers import interactive_ui, message_queue

UI_PANE = "  Do you want to proceed?\n  ❯ 1. Yes\n    2. No\n\n  Esc to cancel\n"


@pytest.fixture
def events():
    return []


@pytest.fixture
def env(events, monkeypatch):
    """Patch the queue's Telegram / tmux edges and record what happens."""
    interactive_ui._interactive_mode.clear()
    interactive_ui._interactive_msgs.clear()
    message_queue._pending_interactive.clear()
    message_queue._ui_send_failed_at.clear()

    next_id = iter(range(1000, 2000))

    async def fake_send(bot, chat_id, text, **kwargs):
        events.append(("send", text))
        msg = MagicMock()
        msg.message_id = next(next_id)
        return msg

    ui_result = {"value": True}

    async def fake_ui(bot, user_id, wid, thread_id=None, *, force_new=False):
        events.append(("ui", force_new))
        if ui_result["value"]:
            interactive_ui._interactive_msgs[(user_id, thread_id or 0)] = 55
            interactive_ui._interactive_mode[(user_id, thread_id or 0)] = wid
        return ui_result["value"]

    async def fake_clear(user_id, bot=None, thread_id=None):
        events.append(("clear_ui",))
        interactive_ui._interactive_msgs.pop((user_id, thread_id or 0), None)
        interactive_ui._interactive_mode.pop((user_id, thread_id or 0), None)

    tmux = MagicMock()
    tmux.find_window_by_id = AsyncMock(return_value=None)
    tmux.capture_pane = AsyncMock(return_value=UI_PANE)
    sm = MagicMock()
    sm.resolve_chat_id.return_value = -100

    monkeypatch.setattr(message_queue, "send_with_fallback", fake_send)
    monkeypatch.setattr(message_queue, "handle_interactive_ui", fake_ui)
    monkeypatch.setattr(message_queue, "clear_interactive_msg", fake_clear)
    monkeypatch.setattr(message_queue, "tmux_manager", tmux)
    monkeypatch.setattr(message_queue, "session_manager", sm)
    monkeypatch.setattr(message_queue, "record_sent", AsyncMock())
    monkeypatch.setattr(message_queue, "INTERACTIVE_RENDER_WAIT", 0.0)
    yield {"tmux": tmux, "ui_result": ui_result}
    interactive_ui._interactive_mode.clear()
    interactive_ui._interactive_msgs.clear()
    message_queue._pending_interactive.clear()
    message_queue._ui_send_failed_at.clear()


async def _drain(user_id: int = 1) -> None:
    queue = message_queue.get_message_queue(user_id)
    assert queue is not None
    try:
        await queue.join()
    finally:
        await message_queue.shutdown_workers()


@pytest.mark.asyncio
async def test_ui_waits_for_content_queued_before_it(env, events):
    bot = AsyncMock()
    await message_queue.enqueue_content_message(
        bot,
        1,
        "@5",
        ["thinking before the question"],
        content_type="thinking",
        thread_id=42,
    )
    await message_queue.enqueue_interactive(bot, 1, "@5", 42)
    assert message_queue.has_pending_interactive(1, 42)
    await _drain()
    assert events == [("send", "thinking before the question"), ("ui", False)]
    assert not message_queue.has_pending_interactive(1, 42)


@pytest.mark.asyncio
async def test_late_content_reposts_open_ui_below_it(env, events):
    bot = AsyncMock()
    interactive_ui._interactive_msgs[(1, 42)] = 55
    interactive_ui._interactive_mode[(1, 42)] = "@5"
    await message_queue.enqueue_content_message(
        bot, 1, "@5", ["late thinking"], content_type="thinking", thread_id=42
    )
    await _drain()
    assert events == [("send", "late thinking"), ("ui", True)]


@pytest.mark.asyncio
async def test_answered_ui_is_deleted_before_new_content(env, events):
    bot = AsyncMock()
    env["tmux"].capture_pane.return_value = "idle output\n"
    interactive_ui._interactive_msgs[(1, 42)] = 55
    interactive_ui._interactive_mode[(1, 42)] = "@5"
    await message_queue.enqueue_content_message(
        bot, 1, "@5", ["answer received"], content_type="text", thread_id=42
    )
    await _drain()
    assert events == [("clear_ui",), ("send", "answer received")]


@pytest.mark.asyncio
async def test_ui_that_never_renders_falls_back_to_tool_use_text(env, events):
    bot = AsyncMock()
    env["ui_result"]["value"] = False
    env["tmux"].capture_pane.return_value = "no dialog on screen\n"
    await message_queue.enqueue_interactive(
        bot, 1, "@5", 42, fallback_parts=["**AskUserQuestion**(Pick one)"]
    )
    await _drain()
    assert events == [("ui", False), ("send", "**AskUserQuestion**(Pick one)")]
    assert interactive_ui.get_interactive_window(1, 42) is None
    assert not message_queue.has_pending_interactive(1, 42)


@pytest.mark.asyncio
async def test_error_is_never_merged_into_other_content(env, events):
    bot = AsyncMock()
    await message_queue.enqueue_content_message(
        bot,
        1,
        "@5",
        ["⚠️ API error: retrying"],
        content_type="warning",
        text="⚠️ API error: retrying",
        thread_id=42,
    )
    await message_queue.enqueue_content_message(
        bot,
        1,
        "@5",
        ["🚨 failed"],
        content_type="error",
        text="🚨 failed",
        thread_id=42,
    )
    # queued while the worker is idle → both are in the queue before it runs
    await _drain()
    assert [e for e in events if e[0] == "send"] == [
        ("send", "⚠️ API error: retrying"),
        ("send", "🚨 failed"),
    ]
    kinds = [c.args[1] for c in message_queue.record_sent.await_args_list]  # type: ignore[attr-defined]
    assert kinds == ["warning", "error"]


@pytest.mark.asyncio
async def test_merge_keeps_text_of_all_parts():
    first = message_queue.MessageTask(
        task_type="content", text="a", window_id="@5", parts=["a"]
    )
    queue: asyncio.Queue[message_queue.MessageTask] = asyncio.Queue()
    queue.put_nowait(
        message_queue.MessageTask(
            task_type="content", text="b", window_id="@5", parts=["b"]
        )
    )
    merged, count = await message_queue._merge_content_tasks(
        queue, first, asyncio.Lock()
    )
    assert count == 1
    assert merged.parts == ["a", "b"]
    assert merged.text == "a\n\nb"


@pytest.mark.asyncio
async def test_drawn_ui_telegram_refuses_does_not_hold_queue(env, events):
    """UI on screen but the send fails: give up at once, back off the poller."""
    bot = AsyncMock()
    env["ui_result"]["value"] = False  # handle_interactive_ui fails to send
    await message_queue.enqueue_interactive(
        bot, 1, "@5", 42, fallback_parts=["**AskUserQuestion**(Pick one)"]
    )
    await message_queue.enqueue_content_message(
        bot, 1, "@5", ["next"], content_type="text", thread_id=42
    )
    await _drain()
    assert events == [("ui", False), ("send", "next")]
    assert interactive_ui.get_interactive_window(1, 42) is None
    assert message_queue.interactive_backoff_active(1, 42)
    message_queue.clear_pending_interactive(1, 42)
    assert not message_queue.interactive_backoff_active(1, 42)


@pytest.mark.asyncio
async def test_flood_control_waits_for_interactive_instead_of_dropping(
    env, events, monkeypatch
):
    """A flood ban drops status updates only; a queued UI still shows."""
    import time

    bot = AsyncMock()
    monkeypatch.setattr(message_queue.asyncio, "sleep", AsyncMock())
    message_queue._flood_until[1] = time.monotonic() + 30
    try:
        await message_queue.enqueue_interactive(bot, 1, "@5", 42)
        await _drain()
    finally:
        message_queue._flood_until.pop(1, None)
    assert events == [("ui", False)]
    assert not message_queue.has_pending_interactive(1, 42)


@pytest.mark.asyncio
async def test_retry_after_reposts_ui_once_after_content(env, events, monkeypatch):
    """A 429 on the content send: retry sends content, then re-posts UI once."""
    from aiogram.exceptions import TelegramRetryAfter
    from aiogram.methods import SendMessage

    bot = AsyncMock()
    interactive_ui._interactive_msgs[(1, 42)] = 55
    interactive_ui._interactive_mode[(1, 42)] = "@5"
    real_send = message_queue.send_with_fallback
    calls = {"n": 0}

    async def flaky_send(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TelegramRetryAfter(
                method=SendMessage(chat_id=1, text="x"), message="flood", retry_after=1
            )
        return await real_send(*args, **kwargs)

    monkeypatch.setattr(message_queue, "send_with_fallback", flaky_send)
    monkeypatch.setattr(message_queue.asyncio, "sleep", AsyncMock())
    await message_queue.enqueue_content_message(
        bot, 1, "@5", ["late thinking"], content_type="thinking", thread_id=42
    )
    await _drain()
    assert events == [("send", "late thinking"), ("ui", True)]
