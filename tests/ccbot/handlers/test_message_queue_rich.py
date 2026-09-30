"""Tests for the rich-message path of the queue worker and response builder."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ccbot.config import config
from ccbot.handlers import interactive_ui, message_queue
from ccbot.handlers.response_builder import build_rich_parts
from ccbot.transcript_parser import ToolCall


@pytest.fixture
def calls(monkeypatch):
    """Record rich / classic sends and edits made by the worker."""
    log: list[tuple] = []
    ids = iter(range(500, 600))

    def _msg():
        m = MagicMock()
        m.message_id = next(ids)
        return m

    async def send_rich(bot, chat_id, text, **kw):
        log.append(("send_rich", text, kw.get("message_thread_id")))
        return _msg()

    async def send_classic(bot, chat_id, text, **kw):
        log.append(("send_classic", text))
        return _msg()

    async def edit_rich(bot, chat_id, message_id, text, **kw):
        log.append(("edit_rich", message_id, text))
        return True

    tmux = MagicMock()
    tmux.find_window_by_id = AsyncMock(return_value=None)
    tmux.capture_pane = AsyncMock(return_value="")
    sm = MagicMock()
    sm.resolve_chat_id.return_value = -100
    monkeypatch.setattr(message_queue, "send_rich", send_rich)
    monkeypatch.setattr(message_queue, "send_with_fallback", send_classic)
    monkeypatch.setattr(message_queue, "edit_rich", edit_rich)
    monkeypatch.setattr(message_queue, "tmux_manager", tmux)
    monkeypatch.setattr(message_queue, "session_manager", sm)
    monkeypatch.setattr(message_queue, "record_sent", AsyncMock())
    interactive_ui._interactive_msgs.clear()
    interactive_ui._interactive_mode.clear()
    message_queue._tool_msg_ids.clear()
    message_queue._status_msg_info.clear()
    yield log
    message_queue._tool_msg_ids.clear()
    message_queue._status_msg_info.clear()


async def _drain(user_id: int = 1) -> None:
    queue = message_queue.get_message_queue(user_id)
    assert queue is not None
    try:
        await queue.join()
    finally:
        await message_queue.shutdown_workers()


@pytest.mark.asyncio
async def test_consecutive_rich_parts_are_packed_into_one_message(calls):
    bot = AsyncMock()
    await message_queue.enqueue_content_message(
        bot,
        1,
        "@5",
        ["<details><summary>💭 hm</summary>\n\nx\n\n</details>"],
        content_type="thinking",
        thread_id=42,
        rich=True,
    )
    await message_queue.enqueue_content_message(
        bot, 1, "@5", ["the answer"], content_type="text", thread_id=42, rich=True
    )
    await _drain()
    assert [c[0] for c in calls] == ["send_rich"]
    assert calls[0][1].endswith("</details>\n\nthe answer")
    assert calls[0][2] == 42


@pytest.mark.asyncio
async def test_rich_and_classic_tasks_do_not_merge(calls):
    bot = AsyncMock()
    await message_queue.enqueue_content_message(
        bot, 1, "@5", ["rich"], thread_id=42, rich=True
    )
    await message_queue.enqueue_content_message(bot, 1, "@5", ["classic"], thread_id=42)
    await _drain()
    assert [c[0] for c in calls] == ["send_rich", "send_classic"]


@pytest.mark.asyncio
async def test_tool_result_edits_first_part_and_continues(calls):
    bot = AsyncMock()
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
    await message_queue.enqueue_content_message(
        bot,
        1,
        "@5",
        ["⚙️ **run**\n⎿ ✅ 3 lines", "_continued (2/3)_ b", "_continued (3/3)_ c"],
        tool_use_id="t1",
        content_type="tool_result",
        thread_id=42,
        rich=True,
    )
    await _drain()
    assert calls[0] == ("send_rich", "⚙️ **run**", 42)
    tool_msg = 500
    assert calls[1] == ("edit_rich", tool_msg, "⚙️ **run**\n⎿ ✅ 3 lines")
    # continuation parts are packed into one new message
    assert calls[2][0] == "send_rich"
    assert calls[2][1] == "_continued (2/3)_ b\n\n_continued (3/3)_ c"


@pytest.mark.asyncio
async def test_status_message_is_edited_into_rich_content(calls):
    bot = AsyncMock()
    message_queue._status_msg_info[(1, 42)] = (77, "@5", "Working…")
    await message_queue.enqueue_content_message(
        bot, 1, "@5", ["hello"], thread_id=42, rich=True
    )
    await _drain()
    assert calls == [("edit_rich", 77, "hello")]


def test_pack_rich_respects_budget():
    big = "x" * (message_queue.RICH_CHAR_BUDGET - 2)  # + "\n\n" + "y" > budget
    assert message_queue._pack_rich([big, "y"]) == [big, "y"]
    assert message_queue._pack_rich(["a", "b", "c"]) == ["a\n\nb\n\nc"]


class TestBuildRichParts:
    def test_todo_result_adds_nothing(self):
        call = ToolCall(
            name="TodoWrite", input={"todos": []}, result_text="Todos modified"
        )
        assert build_rich_parts("x", "tool_result", "assistant", tool=call) is None

    def test_bash_use_renders_description(self):
        call = ToolCall(name="Bash", input={"command": "ls", "description": "List"})
        (part,) = build_rich_parts("**Bash**(ls)", "tool_use", "assistant", tool=call)
        assert part.startswith("⚙️ **List**")

    def test_thinking_uses_raw_body_and_settings(self, monkeypatch):
        monkeypatch.setattr(config, "thinking_max_chars", 10)
        (part,) = build_rich_parts(
            "\x02EXPQUOTE_START\x02long thinking text\x02EXPQUOTE_END\x02",
            "thinking",
            "assistant",
            raw="long thinking text here",
        )
        assert "EXPQUOTE" not in part and "(truncated)" in part

    def test_tool_result_of_unknown_tool_falls_back_to_text(self):
        call = ToolCall(name="", result_text="orphan")
        (part,) = build_rich_parts(
            "\x02EXPQUOTE_START\x02orphan\x02EXPQUOTE_END\x02",
            "tool_result",
            "assistant",
            tool=call,
        )
        assert part == "orphan"

    def test_user_and_notices(self):
        assert build_rich_parts("hi $X", "text", "user") == ["👤 hi \\$X"]
        assert build_rich_parts("⚠️ API error: x", "warning", "assistant") == [
            "⚠️ API error: x"
        ]


def test_turn_footer_threshold():
    assert build_rich_parts(
        "✻ Worked for 27s", "turn_duration", "assistant", raw="27.4"
    ) == ["_✻ Worked for 27s_"]
    assert (
        build_rich_parts("✻ Worked for 3s", "turn_duration", "assistant", raw="3")
        is None
    )


@pytest.mark.asyncio
async def test_turn_footer_is_packed_onto_the_reply(calls):
    bot = AsyncMock()
    await message_queue.enqueue_content_message(
        bot, 1, "@5", ["Done: all tests pass."], thread_id=42, rich=True, ends_turn=True
    )
    await message_queue.enqueue_content_message(
        bot,
        1,
        "@5",
        ["_✻ Worked for 27s_"],
        content_type="turn_duration",
        thread_id=42,
        rich=True,
    )
    await _drain()
    assert calls == [("send_rich", "Done: all tests pass.\n\n_✻ Worked for 27s_", 42)]
