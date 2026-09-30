"""Safe message sending helpers with MarkdownV2 fallback.

Provides utility functions for sending Telegram messages with automatic
format conversion and fallback to plain text on failure.

Uses telegramify-markdown for MarkdownV2 formatting.

Functions:
  - send_with_fallback: Send with formatting → plain text fallback
  - send_photo: Photo sending (single or media group)
  - safe_reply: Reply with formatting, fallback to plain text
  - safe_edit: Edit message with formatting, fallback to plain text
  - safe_send: Send message with formatting, fallback to plain text
  - edit_with_fallback: Edit by (chat_id, message_id), returns success bool
  - send_rich / edit_rich: Rich messages (Bot API 10.1, rich_render markdown);
    a rejected rich text is resent / re-edited as plain text
  - run_with_fallback: The shared policy behind all of the above

Fallback policy (see run_with_fallback): only a TelegramBadRequest (i.e.
the MarkdownV2 parser rejected the text) triggers the plain-text retry.
TelegramNetworkError (timeouts included) usually means the request *was*
delivered and the response got lost, so retrying would duplicate the
message — those are logged and treated as failure without a resend.

Rate limiting is handled by telegram_client.TelegramRateLimiter on the bot
session. TelegramRetryAfter that survives its retries is re-raised so
callers (queue worker) can handle it.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from aiogram import Bot
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramNetworkError,
    TelegramRetryAfter,
    TelegramServerError,
)
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InputMediaPhoto,
    InputRichMessage,
    LinkPreviewOptions,
    Message,
)

from ..markdown_v2 import convert_markdown
from ..rich_render import is_rtl
from ..telegram_sender import TELEGRAM_MAX_MESSAGE_LENGTH, split_message, utf16_len
from ..transcript_parser import TranscriptParser

logger = logging.getLogger(__name__)

T = TypeVar("T")


def strip_sentinels(text: str) -> str:
    """Strip expandable quote sentinel markers for plain text fallback."""
    for s in (
        TranscriptParser.EXPANDABLE_QUOTE_START,
        TranscriptParser.EXPANDABLE_QUOTE_END,
    ):
        text = text.replace(s, "")
    return text


def _ensure_formatted(text: str) -> str:
    """Convert markdown to MarkdownV2."""
    return convert_markdown(text)


PARSE_MODE = "MarkdownV2"


# Disable link previews in all messages to reduce visual noise
NO_LINK_PREVIEW = LinkPreviewOptions(is_disabled=True)

_NOT_MODIFIED = "message is not modified"

# The request may well have reached Telegram: never resend / report failure.
# 5xx gateway errors (TelegramServerError) count too — PTB mapped them to
# NetworkError, and treating them as failure made the queue post duplicates.
_TRANSPORT_ERRORS = (TelegramNetworkError, TelegramServerError)


def _is_not_modified(e: TelegramBadRequest) -> bool:
    return _NOT_MODIFIED in str(e).lower()


async def run_with_fallback(
    primary: Callable[[], Awaitable[T]],
    fallback: Callable[[], Awaitable[T]],
    what: str,
    *,
    raise_on_failure: bool = False,
) -> T | None:
    """Run ``primary``; on a TelegramBadRequest run ``fallback`` (plain text).

    - TelegramBadRequest → the formatted text was rejected → try ``fallback``.
    - "Message is not modified" (edits) → treated as success, returns None.
    - TelegramNetworkError (incl. timeouts) → logged, returns None. Not
      retried: the request has usually reached Telegram already and a resend
      duplicates.
    - TelegramRetryAfter → re-raised for the queue worker.
    - Anything else → logged (re-raised when ``raise_on_failure``).
    """
    try:
        return await primary()
    except TelegramRetryAfter:
        raise
    except TelegramBadRequest as e:
        if _is_not_modified(e):
            return None
        logger.warning("%s: MarkdownV2 rejected, falling back to plain: %s", what, e)
    except _TRANSPORT_ERRORS as e:
        logger.warning("%s: transport error, not retrying: %s", what, e)
        if raise_on_failure:
            raise
        return None
    except Exception as e:
        logger.error("%s failed: %s", what, e)
        if raise_on_failure:
            raise
        return None

    try:
        return await fallback()
    except TelegramRetryAfter:
        raise
    except TelegramBadRequest as e:
        if _is_not_modified(e):
            return None
        logger.error("%s: plain-text fallback rejected too: %s", what, e)
        if raise_on_failure:
            raise
        return None
    except Exception as e:
        logger.error("%s: plain-text fallback failed: %s", what, e)
        if raise_on_failure:
            raise
        return None


async def send_with_fallback(
    bot: Bot,
    chat_id: int,
    text: str,
    **kwargs: Any,
) -> Message | None:
    """Send message with MarkdownV2, falling back to plain text on failure.

    Returns the sent Message on success, None on failure.
    TelegramRetryAfter is re-raised for caller handling.
    """
    kwargs.setdefault("link_preview_options", NO_LINK_PREVIEW)
    return await run_with_fallback(
        lambda: bot.send_message(
            chat_id=chat_id,
            text=_ensure_formatted(text),
            parse_mode=PARSE_MODE,
            **kwargs,
        ),
        lambda: bot.send_message(
            chat_id=chat_id, text=strip_sentinels(text), parse_mode=None, **kwargs
        ),
        f"send_message({chat_id})",
    )


async def send_photo(
    bot: Bot,
    chat_id: int,
    image_data: list[tuple[str, bytes]],
    **kwargs: Any,
) -> None:
    """Send photo(s) to chat. Sends as media group if multiple images.

    Rate limiting is handled by the bot session's TelegramRateLimiter.

    Args:
        bot: Telegram Bot instance
        chat_id: Target chat ID
        image_data: List of (media_type, raw_bytes) tuples
        **kwargs: Extra kwargs passed to send_photo/send_media_group
    """
    if not image_data:
        return
    files = [
        BufferedInputFile(raw_bytes, filename=f"image{i}.{_extension(media_type)}")
        for i, (media_type, raw_bytes) in enumerate(image_data, 1)
    ]
    try:
        if len(files) == 1:
            await bot.send_photo(chat_id=chat_id, photo=files[0], **kwargs)
        else:
            media: list[Any] = [InputMediaPhoto(media=f) for f in files]
            await bot.send_media_group(chat_id=chat_id, media=media, **kwargs)
    except TelegramRetryAfter:
        raise
    except Exception as e:
        logger.error("Failed to send photo to %d: %s", chat_id, e)


def _extension(media_type: str) -> str:
    """ "image/png" → "png" (Telegram sniffs the content; this is cosmetic)."""
    sub = media_type.partition("/")[2].split(";")[0].strip()
    return {"jpeg": "jpg", "svg+xml": "svg"}.get(sub, sub or "png")


async def safe_reply(message: Message, text: str, **kwargs: Any) -> Message:
    """Reply (quoting ``message``, in its topic) with formatting → plain fallback."""
    kwargs.setdefault("link_preview_options", NO_LINK_PREVIEW)
    sent = await run_with_fallback(
        lambda: message.reply(
            _ensure_formatted(text),
            parse_mode=PARSE_MODE,
            **kwargs,
        ),
        lambda: message.reply(strip_sentinels(text), parse_mode=None, **kwargs),
        "reply_text",
        raise_on_failure=True,
    )
    if sent is None:  # only reachable via "not modified", impossible for replies
        raise RuntimeError("reply_text returned no message")
    return sent


async def safe_edit(target: CallbackQuery | Message, text: str, **kwargs: Any) -> None:
    """Edit message with formatting, falling back to plain text on failure.

    ``target`` is a CallbackQuery (its message is edited) or a Message —
    e.g. a progress message returned by safe_reply.
    """
    kwargs.setdefault("link_preview_options", NO_LINK_PREVIEW)
    message = target.message if isinstance(target, CallbackQuery) else target
    if not isinstance(message, Message):
        logger.debug("edit_message_text: message is inaccessible (too old)")
        return
    await run_with_fallback(
        lambda: message.edit_text(
            _ensure_formatted(text), parse_mode=PARSE_MODE, **kwargs
        ),
        lambda: message.edit_text(strip_sentinels(text), parse_mode=None, **kwargs),
        "edit_message_text",
    )


async def edit_with_fallback(
    bot: Bot,
    chat_id: int,
    message_id: int,
    text: str,
    **kwargs: Any,
) -> bool:
    """Edit a message by id with formatting → plain-text fallback.

    Returns True when the edit succeeded, the text was already identical,
    or the outcome is unknown (transport error — assume delivered so the
    caller doesn't send a duplicate). Returns False when Telegram rejected
    the edit (message deleted / too old / both renderings invalid) so the
    caller can send a fresh message instead.
    """
    kwargs.setdefault("link_preview_options", NO_LINK_PREVIEW)
    what = f"edit_message({message_id})"
    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=_ensure_formatted(text),
            parse_mode=PARSE_MODE,
            **kwargs,
        )
        return True
    except TelegramRetryAfter:
        raise
    except TelegramBadRequest as e:
        if _is_not_modified(e):
            return True
        logger.warning("%s: MarkdownV2 rejected, falling back to plain: %s", what, e)
    except _TRANSPORT_ERRORS as e:
        logger.warning("%s: transport error, not retrying: %s", what, e)
        return True
    except Exception as e:
        logger.error("%s failed: %s", what, e)
        return False

    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=strip_sentinels(text),
            parse_mode=None,
            **kwargs,
        )
        return True
    except TelegramRetryAfter:
        raise
    except TelegramBadRequest as e:
        if _is_not_modified(e):
            return True
        logger.debug("%s: plain-text fallback rejected: %s", what, e)
        return False
    except _TRANSPORT_ERRORS as e:
        logger.warning("%s: transport error on fallback, not retrying: %s", what, e)
        return True
    except Exception as e:
        logger.error("%s: plain-text fallback failed: %s", what, e)
        return False


async def safe_send(
    bot: Bot,
    chat_id: int,
    text: str,
    message_thread_id: int | None = None,
    **kwargs: Any,
) -> None:
    """Send message with formatting, falling back to plain text on failure."""
    kwargs.setdefault("link_preview_options", NO_LINK_PREVIEW)
    if message_thread_id is not None:
        kwargs.setdefault("message_thread_id", message_thread_id)
    await run_with_fallback(
        lambda: bot.send_message(
            chat_id=chat_id,
            text=_ensure_formatted(text),
            parse_mode=PARSE_MODE,
            **kwargs,
        ),
        lambda: bot.send_message(
            chat_id=chat_id, text=strip_sentinels(text), parse_mode=None, **kwargs
        ),
        f"send_message({chat_id})",
    )


def _rich(markdown: str) -> InputRichMessage:
    return InputRichMessage(
        markdown=markdown, is_rtl=True if is_rtl(markdown) else None
    )


_EDIT_GONE = ("message to edit not found", "message can't be edited")


async def send_rich(
    bot: Bot,
    chat_id: int,
    markdown: str,
    **kwargs: Any,
) -> Message | None:
    """Send a rich message; if Telegram rejects the markdown, send it as plain text.

    Returns the sent Message, or None on failure. Transport errors are not
    retried (the message may have been delivered). TelegramRetryAfter is
    re-raised for the queue worker.
    """
    try:
        return await bot.send_rich_message(
            chat_id=chat_id, rich_message=_rich(markdown), **kwargs
        )
    except TelegramRetryAfter:
        raise
    except _TRANSPORT_ERRORS as e:
        logger.warning(
            "send_rich_message(%s): transport error, not retrying: %s", chat_id, e
        )
        return None
    except Exception as e:
        # BadRequest (markup rejected) or anything else the API refuses:
        # the content must not be lost — send it as plain text instead
        logger.warning("send_rich_message(%s) failed, sending plain: %s", chat_id, e)
    kwargs.pop("link_preview_options", None)
    markup = kwargs.pop("reply_markup", None)
    chunks = split_message(markdown) or [markdown]
    sent: Message | None = None
    for i, chunk in enumerate(chunks):
        last = i == len(chunks) - 1
        try:
            sent = await bot.send_message(
                chat_id=chat_id,
                text=chunk,
                parse_mode=None,
                link_preview_options=NO_LINK_PREVIEW,
                reply_markup=markup if last else None,
                **kwargs,
            )
        except TelegramRetryAfter:
            raise
        except Exception as e:
            logger.error("send_message(%s) plain fallback failed: %s", chat_id, e)
            return sent
    return sent


async def edit_rich(
    bot: Bot,
    chat_id: int,
    message_id: int,
    markdown: str,
    **kwargs: Any,
) -> bool:
    """Edit a message (rich or plain) into a rich one.

    Same contract as edit_with_fallback: True when edited, unchanged, or the
    outcome is unknown (transport error); False when the message can't be
    edited any more (caller sends a new one). Markdown Telegram rejects is
    re-edited as plain text.
    """
    what = f"edit_rich({message_id})"
    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            rich_message=_rich(markdown),
            **kwargs,
        )
        return True
    except TelegramRetryAfter:
        raise
    except TelegramBadRequest as e:
        if _is_not_modified(e):
            return True
        if any(g in str(e).lower() for g in _EDIT_GONE):
            logger.debug("%s: %s", what, e)
            return False
        logger.warning("%s: rich text rejected, editing plain: %s", what, e)
    except _TRANSPORT_ERRORS as e:
        logger.warning("%s: transport error, not retrying: %s", what, e)
        return True
    except Exception as e:
        logger.error("%s failed: %s", what, e)
        return False
    if utf16_len(markdown) > TELEGRAM_MAX_MESSAGE_LENGTH:
        return False  # too long for plain text: caller sends it anew (split)
    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=markdown,
            parse_mode=None,
            link_preview_options=NO_LINK_PREVIEW,
            **kwargs,
        )
        return True
    except TelegramRetryAfter:
        raise
    except TelegramBadRequest as e:
        return _is_not_modified(e)
    except _TRANSPORT_ERRORS:
        return True
    except Exception as e:
        logger.error("%s: plain fallback failed: %s", what, e)
        return False
