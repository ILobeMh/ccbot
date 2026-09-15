"""Tests for the MarkdownV2 → plain-text fallback policy in message_sender."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut

from ccbot.handlers.message_sender import (
    edit_with_fallback,
    safe_reply,
    safe_send,
    send_with_fallback,
)


def _bot(send_side_effects: list) -> MagicMock:
    bot = MagicMock()
    bot.send_message = AsyncMock(side_effect=send_side_effects)
    bot.edit_message_text = AsyncMock(side_effect=send_side_effects)
    return bot


class TestSendWithFallback:
    @pytest.mark.asyncio
    async def test_bad_request_falls_back_to_plain(self):
        bot = _bot([BadRequest("Can't parse entities"), "sent-plain"])
        result = await send_with_fallback(bot, 1, "**bold**")
        assert result == "sent-plain"
        assert bot.send_message.await_count == 2
        # Second call has no parse_mode
        assert "parse_mode" not in bot.send_message.await_args_list[1].kwargs

    @pytest.mark.asyncio
    async def test_timeout_is_not_retried(self):
        bot = _bot([TimedOut()])
        result = await send_with_fallback(bot, 1, "text")
        assert result is None
        assert bot.send_message.await_count == 1

    @pytest.mark.asyncio
    async def test_network_error_is_not_retried(self):
        bot = _bot([NetworkError("boom")])
        assert await send_with_fallback(bot, 1, "text") is None
        assert bot.send_message.await_count == 1

    @pytest.mark.asyncio
    async def test_retry_after_propagates(self):
        bot = _bot([RetryAfter(3)])
        with pytest.raises(RetryAfter):
            await send_with_fallback(bot, 1, "text")

    @pytest.mark.asyncio
    async def test_success_first_try(self):
        bot = _bot(["ok"])
        assert await send_with_fallback(bot, 1, "text") == "ok"
        assert bot.send_message.await_count == 1


class TestSafeSendAndReply:
    @pytest.mark.asyncio
    async def test_safe_send_bad_request_then_plain(self):
        bot = _bot([BadRequest("bad"), None])
        await safe_send(bot, 1, "text", message_thread_id=7)
        assert bot.send_message.await_count == 2
        assert bot.send_message.await_args_list[1].kwargs["message_thread_id"] == 7

    @pytest.mark.asyncio
    async def test_safe_reply_timeout_raises(self):
        message = MagicMock()
        message.reply_text = AsyncMock(side_effect=[TimedOut()])
        with pytest.raises(TimedOut):
            await safe_reply(message, "text")
        assert message.reply_text.await_count == 1

    @pytest.mark.asyncio
    async def test_safe_reply_bad_request_falls_back(self):
        message = MagicMock()
        message.reply_text = AsyncMock(side_effect=[BadRequest("bad"), "plain"])
        assert await safe_reply(message, "text") == "plain"


class TestEditWithFallback:
    @pytest.mark.asyncio
    async def test_success(self):
        bot = _bot([None])
        assert await edit_with_fallback(bot, 1, 10, "text") is True

    @pytest.mark.asyncio
    async def test_not_modified_is_success(self):
        bot = _bot([BadRequest("Message is not modified: nothing changed")])
        assert await edit_with_fallback(bot, 1, 10, "text") is True
        assert bot.edit_message_text.await_count == 1

    @pytest.mark.asyncio
    async def test_parse_error_then_plain_success(self):
        bot = _bot([BadRequest("Can't parse entities"), None])
        assert await edit_with_fallback(bot, 1, 10, "text") is True
        assert bot.edit_message_text.await_count == 2

    @pytest.mark.asyncio
    async def test_message_gone_returns_false(self):
        bot = _bot(
            [
                BadRequest("Message to edit not found"),
                BadRequest("Message to edit not found"),
            ]
        )
        assert await edit_with_fallback(bot, 1, 10, "text") is False

    @pytest.mark.asyncio
    async def test_timeout_assumed_delivered(self):
        bot = _bot([TimedOut()])
        assert await edit_with_fallback(bot, 1, 10, "text") is True
        assert bot.edit_message_text.await_count == 1

    @pytest.mark.asyncio
    async def test_retry_after_propagates(self):
        bot = _bot([RetryAfter(1)])
        with pytest.raises(RetryAfter):
            await edit_with_fallback(bot, 1, 10, "text")
