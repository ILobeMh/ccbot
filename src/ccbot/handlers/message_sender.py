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
  - run_with_fallback: The shared policy behind all of the above

Fallback policy (see run_with_fallback): only a BadRequest (i.e. the
MarkdownV2 parser rejected the text) triggers the plain-text retry.
TimedOut / NetworkError usually mean the request *was* delivered and the
response got lost, so retrying would duplicate the message — those are
logged and treated as failure without a resend.

Rate limiting is handled globally by AIORateLimiter on the Application.
RetryAfter exceptions are re-raised so callers (queue worker) can handle them.
"""

import io
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from telegram import Bot, InputMediaPhoto, LinkPreviewOptions, Message
from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut

from ..markdown_v2 import convert_markdown
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


def _is_not_modified(e: BadRequest) -> bool:
    return _NOT_MODIFIED in str(e).lower()


async def run_with_fallback(
    primary: Callable[[], Awaitable[T]],
    fallback: Callable[[], Awaitable[T]],
    what: str,
    *,
    raise_on_failure: bool = False,
) -> T | None:
    """Run ``primary``; on a BadRequest run ``fallback`` (plain text).

    - BadRequest → the formatted text was rejected → try ``fallback``.
    - "Message is not modified" (edits) → treated as success, returns None.
    - TimedOut / NetworkError → logged, returns None. Not retried: the
      request has usually reached Telegram already and a resend duplicates.
    - RetryAfter → re-raised for the queue worker.
    - Anything else → logged (re-raised when ``raise_on_failure``).
    """
    try:
        return await primary()
    except RetryAfter:
        raise
    except BadRequest as e:
        if _is_not_modified(e):
            return None
        logger.warning("%s: MarkdownV2 rejected, falling back to plain: %s", what, e)
    except (TimedOut, NetworkError) as e:
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
    except RetryAfter:
        raise
    except BadRequest as e:
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
    RetryAfter is re-raised for caller handling.
    """
    kwargs.setdefault("link_preview_options", NO_LINK_PREVIEW)
    return await run_with_fallback(
        lambda: bot.send_message(
            chat_id=chat_id,
            text=_ensure_formatted(text),
            parse_mode=PARSE_MODE,
            **kwargs,
        ),
        lambda: bot.send_message(chat_id=chat_id, text=strip_sentinels(text), **kwargs),
        f"send_message({chat_id})",
    )


async def send_photo(
    bot: Bot,
    chat_id: int,
    image_data: list[tuple[str, bytes]],
    **kwargs: Any,
) -> None:
    """Send photo(s) to chat. Sends as media group if multiple images.

    Rate limiting is handled globally by AIORateLimiter on the Application.

    Args:
        bot: Telegram Bot instance
        chat_id: Target chat ID
        image_data: List of (media_type, raw_bytes) tuples
        **kwargs: Extra kwargs passed to send_photo/send_media_group
    """
    if not image_data:
        return
    try:
        if len(image_data) == 1:
            _media_type, raw_bytes = image_data[0]
            await bot.send_photo(
                chat_id=chat_id,
                photo=io.BytesIO(raw_bytes),
                **kwargs,
            )
        else:
            media = [
                InputMediaPhoto(media=io.BytesIO(raw_bytes))
                for _media_type, raw_bytes in image_data
            ]
            await bot.send_media_group(
                chat_id=chat_id,
                media=media,
                **kwargs,
            )
    except RetryAfter:
        raise
    except Exception as e:
        logger.error("Failed to send photo to %d: %s", chat_id, e)


async def safe_reply(message: Message, text: str, **kwargs: Any) -> Message:
    """Reply with formatting, falling back to plain text on failure."""
    kwargs.setdefault("link_preview_options", NO_LINK_PREVIEW)
    sent = await run_with_fallback(
        lambda: message.reply_text(
            _ensure_formatted(text),
            parse_mode=PARSE_MODE,
            **kwargs,
        ),
        lambda: message.reply_text(strip_sentinels(text), **kwargs),
        "reply_text",
        raise_on_failure=True,
    )
    if sent is None:  # only reachable via "not modified", impossible for replies
        raise RuntimeError("reply_text returned no message")
    return sent


async def safe_edit(target: Any, text: str, **kwargs: Any) -> None:
    """Edit message with formatting, falling back to plain text on failure."""
    kwargs.setdefault("link_preview_options", NO_LINK_PREVIEW)
    await run_with_fallback(
        lambda: target.edit_message_text(
            _ensure_formatted(text),
            parse_mode=PARSE_MODE,
            **kwargs,
        ),
        lambda: target.edit_message_text(strip_sentinels(text), **kwargs),
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
    except RetryAfter:
        raise
    except BadRequest as e:
        if _is_not_modified(e):
            return True
        logger.warning("%s: MarkdownV2 rejected, falling back to plain: %s", what, e)
    except (TimedOut, NetworkError) as e:
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
            **kwargs,
        )
        return True
    except RetryAfter:
        raise
    except BadRequest as e:
        if _is_not_modified(e):
            return True
        logger.debug("%s: plain-text fallback rejected: %s", what, e)
        return False
    except (TimedOut, NetworkError) as e:
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
        lambda: bot.send_message(chat_id=chat_id, text=strip_sentinels(text), **kwargs),
        f"send_message({chat_id})",
    )
