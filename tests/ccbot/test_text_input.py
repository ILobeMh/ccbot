"""Tests for text merging (split pastes) and the confirm-before-send stage."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccbot import bot as bot_mod
from ccbot.config import config


def _text_update(text: str, user_id: int = 1) -> MagicMock:
    update = MagicMock()
    update.effective_user.id = user_id
    update.message.text = text
    return update


@pytest.fixture(autouse=True)
def _clean():
    bot_mod._pending_merges.clear()
    bot_mod._pending_texts.clear()
    yield
    bot_mod._pending_merges.clear()
    bot_mod._pending_texts.clear()


class TestTextMerge:
    @pytest.mark.asyncio
    async def test_parts_within_window_are_joined(self):
        handled: list[str] = []

        async def _record(_u, _c, text):
            handled.append(text)

        with (
            patch("ccbot.bot.is_user_allowed", return_value=True),
            patch("ccbot.bot._get_thread_id", return_value=42),
            patch("ccbot.bot._handle_text", side_effect=_record),
            patch.object(config, "text_merge_window", 0.05),
        ):
            for part in ("part one", "part two", "part three"):
                await bot_mod.text_handler(_text_update(part), MagicMock())
            assert handled == []
            await asyncio.sleep(0.2)
        assert handled == ["part one\npart two\npart three"]

    @pytest.mark.asyncio
    async def test_window_zero_processes_immediately(self):
        handled: list[str] = []

        async def _record(_u, _c, text):
            handled.append(text)

        with (
            patch("ccbot.bot.is_user_allowed", return_value=True),
            patch("ccbot.bot._handle_text", side_effect=_record),
            patch.object(config, "text_merge_window", 0.0),
        ):
            await bot_mod.text_handler(_text_update("a"), MagicMock())
            await bot_mod.text_handler(_text_update("b"), MagicMock())
        assert handled == ["a", "b"]


class TestConfirmText:
    @pytest.mark.asyncio
    async def test_stage_and_append(self):
        msg = MagicMock()
        msg.reply_text = AsyncMock(
            return_value=MagicMock(edit_reply_markup=AsyncMock())
        )
        await bot_mod._stage_text(msg, 1, 42, "@1", "hello")
        await bot_mod._stage_text(msg, 1, 42, "@1", "more")
        pending = bot_mod._pending_texts[(1, 42)]
        assert pending["text"] == "hello\n\nmore"
        assert msg.reply_text.await_count == 2
        first_prompt = msg.reply_text.return_value
        first_prompt.edit_reply_markup.assert_awaited()

    def test_prompt_preview_is_truncated(self):
        text = "x" * 1000
        prompt = bot_mod._text_confirm_prompt(text)
        assert "(1000 chars)" in prompt
        assert prompt.endswith(" …")
        assert len(prompt) < 500
