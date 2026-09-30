"""Tests for the MarkdownV2 → plain-text fallback policy in message_sender."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramNetworkError,
    TelegramRetryAfter,
)
from aiogram.methods import SendMessage
from aiogram.types import CallbackQuery, Message

from ccbot.handlers.message_sender import (
    edit_with_fallback,
    safe_edit,
    safe_reply,
    safe_send,
    send_with_fallback,
)

_M = SendMessage(chat_id=1, text="x")


def BadRequest(msg: str) -> TelegramBadRequest:  # noqa: N802 — reads like the old API
    return TelegramBadRequest(method=_M, message=msg)


def TimedOut() -> TelegramNetworkError:  # noqa: N802
    return TelegramNetworkError(method=_M, message="Request timeout error")


def NetworkError(msg: str) -> TelegramNetworkError:  # noqa: N802
    return TelegramNetworkError(method=_M, message=msg)


def RetryAfter(secs: int) -> TelegramRetryAfter:  # noqa: N802
    return TelegramRetryAfter(method=_M, message="flood", retry_after=secs)


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
        # Second call is plain text (explicit None beats any bot default)
        assert bot.send_message.await_args_list[1].kwargs["parse_mode"] is None

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
        with pytest.raises(TelegramRetryAfter):
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
        message.reply = AsyncMock(side_effect=[TimedOut()])
        with pytest.raises(TelegramNetworkError):
            await safe_reply(message, "text")
        assert message.reply.await_count == 1

    @pytest.mark.asyncio
    async def test_safe_reply_bad_request_falls_back(self):
        message = MagicMock()
        message.reply = AsyncMock(side_effect=[BadRequest("bad"), "plain"])
        assert await safe_reply(message, "text") == "plain"
        # the plain retry explicitly disables parse mode
        assert message.reply.await_args_list[1].kwargs["parse_mode"] is None


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
        with pytest.raises(TelegramRetryAfter):
            await edit_with_fallback(bot, 1, 10, "text")


class TestSafeEditTargets:
    @pytest.mark.asyncio
    async def test_message_target_uses_edit_text(self):
        message = MagicMock(spec=Message)
        message.edit_text = AsyncMock()
        await safe_edit(message, "hi")
        message.edit_text.assert_awaited_once()
        assert message.edit_text.await_args.kwargs["parse_mode"] == "MarkdownV2"

    @pytest.mark.asyncio
    async def test_query_target_edits_its_message(self):
        query = MagicMock(spec=CallbackQuery)
        query.message = MagicMock(spec=Message)
        query.message.edit_text = AsyncMock()
        await safe_edit(query, "hi")
        query.message.edit_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_inaccessible_message_is_skipped(self):
        query = MagicMock(spec=CallbackQuery)
        query.message = MagicMock()  # InaccessibleMessage: not a Message
        await safe_edit(query, "hi")  # must not raise
