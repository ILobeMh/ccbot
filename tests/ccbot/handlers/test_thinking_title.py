"""Tests for the rich queue's thinking titles and task-notification replies.

A short reply line right after a thinking block becomes the block's summary
(merged in one message, or edited into the thinking message already sent);
a task notification replies to the tool call that started the task.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from ccbot.handlers import interactive_ui, message_queue
from ccbot.handlers.response_builder import thinking_body, titled_thinking
from ccbot.rich_render import RenderOptions, render_thinking, text_caption

THINKING = (
    "Since this touches the shared contracts, a full gate is usual.\nBut not here."
)
REPLY = "Now updating the lessons entry, then running the full gate:"


@pytest_asyncio.fixture
async def calls(monkeypatch):
    log: list[tuple] = []
    ids = iter(range(500, 600))

    def _msg():
        m = MagicMock()
        m.message_id = next(ids)
        return m

    async def send_rich(bot, chat_id, text, **kw):
        log.append(("send", text, kw.get("reply_parameters")))
        return _msg()

    async def edit_rich(bot, chat_id, message_id, text, **kw):
        log.append(("edit", message_id, text))
        return True

    tmux = MagicMock()
    tmux.find_window_by_id = AsyncMock(return_value=None)
    tmux.capture_pane = AsyncMock(return_value="")
    sm = MagicMock()
    sm.resolve_chat_id.return_value = -100
    monkeypatch.setattr(message_queue, "send_rich", send_rich)
    monkeypatch.setattr(message_queue, "edit_rich", edit_rich)
    monkeypatch.setattr(message_queue, "tmux_manager", tmux)
    monkeypatch.setattr(message_queue, "session_manager", sm)
    monkeypatch.setattr(message_queue, "record_sent", AsyncMock())
    interactive_ui._interactive_msgs.clear()
    for d in (
        message_queue._tool_msg_ids,
        message_queue._status_msg_info,
        message_queue._open_thinking,
        message_queue._recent_tool_msgs,
    ):
        d.clear()
    yield log
    await message_queue.shutdown_workers()
    message_queue._open_thinking.clear()
    message_queue._recent_tool_msgs.clear()


def _thinking_part() -> str:
    (part,) = render_thinking(THINKING, RenderOptions())
    return part


async def _thinking(bot, window: str = "@5") -> None:
    await message_queue.enqueue_content_message(
        bot,
        1,
        window,
        [_thinking_part()],
        content_type="thinking",
        thread_id=42,
        rich=True,
        thinking=THINKING,
    )


async def _reply(bot, text: str = REPLY) -> None:
    await message_queue.enqueue_content_message(
        bot,
        1,
        "@5",
        [text],
        content_type="text",
        thread_id=42,
        rich=True,
        caption=text_caption(text),
    )


async def _settle() -> None:
    queue = message_queue.get_message_queue(1)
    assert queue is not None
    await queue.join()


def test_caption_only_for_short_single_lines():
    assert text_caption(REPLY) == REPLY
    assert text_caption("Run `a && b` now") == "Run `a && b` now"
    assert text_caption("Costs $5 & more") == "Costs \\$5 &amp; more"
    assert text_caption("two\nlines") is None
    assert text_caption("- a list item") is None
    assert text_caption("| a | b |") is None
    assert text_caption("x" * 300) is None


def test_titled_thinking_replaces_the_preview():
    titled = titled_thinking(THINKING, REPLY)
    assert titled is not None
    assert titled.startswith(f"<details><summary>💭 {REPLY}</summary>")
    assert "Since this touches" in titled  # the thinking stays inside
    assert thinking_body("x", None) == "x"


@pytest.mark.asyncio
async def test_reply_in_the_same_batch_titles_the_thinking(calls):
    bot = AsyncMock()
    await _thinking(bot)
    await _reply(bot)
    await _settle()
    assert len(calls) == 1
    kind, text, _ = calls[0]
    assert kind == "send"
    assert text == titled_thinking(THINKING, REPLY)
    assert text.count(REPLY) == 1  # not repeated below the block


@pytest.mark.asyncio
async def test_late_reply_is_edited_into_the_sent_thinking(calls):
    bot = AsyncMock()
    await _thinking(bot)
    await _settle()
    await _reply(bot)
    await _settle()
    assert calls[0] == ("send", _thinking_part(), None)
    assert calls[1] == ("edit", 500, titled_thinking(THINKING, REPLY))
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_long_or_final_replies_stay_separate(calls):
    bot = AsyncMock()
    await _thinking(bot)
    await _settle()
    long_reply = "## Summary\n\nAll done."
    await _reply(bot, long_reply)
    await _settle()
    assert [c[0] for c in calls] == ["send", "send"]
    assert calls[1][1] == long_reply


@pytest.mark.asyncio
async def test_something_sent_in_between_blocks_the_edit(calls):
    bot = AsyncMock()
    await _thinking(bot)
    await _settle()
    await message_queue.enqueue_content_message(
        bot,
        1,
        "@5",
        ["⚙️ **run**"],
        tool_use_id="t1",
        content_type="tool_use",
        thread_id=42,
        rich=True,
    )
    await _settle()
    await _reply(bot)
    await _settle()
    assert [c[0] for c in calls] == ["send", "send", "send"]
    assert calls[2][1] == REPLY


@pytest.mark.asyncio
async def test_other_window_or_stale_thinking_is_not_edited(calls, monkeypatch):
    bot = AsyncMock()
    await _thinking(bot, window="@9")
    await _settle()
    await _reply(bot)
    await _settle()
    assert [c[0] for c in calls] == ["send", "send"]

    calls.clear()
    await _thinking(bot)
    await _settle()
    monkeypatch.setattr(message_queue, "TITLE_EDIT_WINDOW", 0.0)
    await _reply(bot)
    await _settle()
    assert [c[0] for c in calls] == ["send", "send"]


@pytest.mark.asyncio
async def test_task_notification_replies_to_its_tool_call(calls):
    bot = AsyncMock()
    await message_queue.enqueue_content_message(
        bot,
        1,
        "@5",
        ["⚙️ **Run the gate** · background"],
        tool_use_id="toolu_1",
        content_type="tool_use",
        thread_id=42,
        rich=True,
    )
    await message_queue.enqueue_content_message(
        bot,
        1,
        "@5",
        ["⚙️ **Run the gate** · background\n⎿ ✅ running"],
        tool_use_id="toolu_1",
        content_type="tool_result",
        thread_id=42,
        rich=True,
    )
    await _settle()
    await message_queue.enqueue_content_message(
        bot,
        1,
        "@5",
        ["✅ **Run the gate** · finished · exit 0"],
        tool_use_id="toolu_1",
        content_type="task_notification",
        thread_id=42,
        rich=True,
    )
    await message_queue.enqueue_content_message(
        bot, 1, "@5", ["All green."], content_type="text", thread_id=42, rich=True
    )
    await _settle()
    kind, text, reply = calls[-2]
    assert (kind, text) == ("send", "✅ **Run the gate** · finished · exit 0")
    assert reply is not None and reply.message_id == 500
    # the notification is a message of its own (never merged with the reply)
    assert calls[-1][1] == "All green." and calls[-1][2] is None


@pytest.mark.asyncio
async def test_interactive_result_without_its_message_is_dropped(calls):
    """The question UI message already records the answer: its tool_result is
    only an edit of the fallback message, never a new message."""
    bot = AsyncMock()
    await message_queue.enqueue_content_message(
        bot,
        1,
        "@5",
        ["⎿ ✅ Answered\n\n- **A** → b"],
        tool_use_id="toolu_q",
        content_type="tool_result",
        thread_id=42,
        rich=True,
        edit_only=True,
    )
    await _settle()
    assert calls == []
