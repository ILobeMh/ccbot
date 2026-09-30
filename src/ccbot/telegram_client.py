"""aiogram client setup: Bot factory, outgoing rate limiting, per-user data.

The bot talks to Telegram through one aiogram ``Bot`` whose session carries
a request middleware (``TelegramRateLimiter``) that keeps outgoing calls
inside Telegram's limits and retries flood-control errors, and the
dispatcher gets ``UserDataMiddleware`` so handlers receive a per-user
``user_data`` dict (the equivalent of python-telegram-bot's
``context.user_data``: in memory, lost on restart).

Rate limits (https://core.telegram.org/bots/faq#my-bot-is-hitting-limits):
  - ~30 requests/s overall → every call with a ``chat_id``.
  - 20 messages/min per group → only calls that create a message in a group.
    Edits, deletes and chat actions don't count against it: charging them
    (as PTB's AIORateLimiter did) let 1 s status-line edits and "typing…"
    actions starve real output, since all topics share one group.
  - A 429 (``TelegramRetryAfter``) pauses every request, then the call is
    retried (up to ``max_retries``), like AIORateLimiter.

Key components: build_bot(), TelegramRateLimiter, UserDataMiddleware.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from aiogram import BaseMiddleware, Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.session.middlewares.base import (
    BaseRequestMiddleware,
    NextRequestMiddlewareType,
)
from aiogram.exceptions import TelegramRetryAfter
from aiogram.types import LinkPreviewOptions, TelegramObject, User
from aiolimiter import AsyncLimiter

if TYPE_CHECKING:
    from aiogram.methods import Response, TelegramMethod
    from aiogram.methods.base import TelegramType

logger = logging.getLogger(__name__)

# API methods that create a new message in the chat (count against the
# per-group budget). Everything else with a chat_id only uses the global one.
MESSAGE_CREATING_METHODS = frozenset(
    {
        "sendMessage",
        "sendRichMessage",
        "sendPhoto",
        "sendDocument",
        "sendMediaGroup",
        "sendVoice",
        "sendAudio",
        "sendVideo",
        "sendAnimation",
        "sendSticker",
        "sendPoll",
        "sendLocation",
        "sendContact",
        "sendDice",
        "copyMessage",
        "forwardMessage",
    }
)


def _is_group(chat_id: object) -> bool:
    """Groups/channels have negative ids (or are addressed by @username)."""
    if isinstance(chat_id, str):
        try:
            return int(chat_id) < 0
        except ValueError:
            return True  # "@username"
    return isinstance(chat_id, int) and chat_id < 0


class TelegramRateLimiter(BaseRequestMiddleware):
    """Client-side rate limiting + flood-control retry for outgoing calls."""

    def __init__(
        self,
        overall_per_second: float = 30,
        group_per_minute: float = 20,
        max_retries: int = 5,
    ) -> None:
        self._overall = AsyncLimiter(overall_per_second, 1)
        # Telegram's server-side counter survives our restarts: start with
        # the bucket full so capacity drains in over ~1 s instead of bursting
        self._overall._level = self._overall.max_rate
        self._group_rate = group_per_minute
        self._groups: dict[int | str, AsyncLimiter] = {}
        self._max_retries = max_retries
        self._not_paused = asyncio.Event()
        self._not_paused.set()

    def _group_limiter(self, chat_id: int | str) -> AsyncLimiter:
        limiter = self._groups.get(chat_id)
        if limiter is None:
            limiter = self._groups[chat_id] = AsyncLimiter(self._group_rate, 60)
        return limiter

    async def __call__(
        self,
        make_request: NextRequestMiddlewareType[TelegramType],
        bot: Bot,
        method: TelegramMethod[TelegramType],
    ) -> Response[TelegramType]:
        chat_id = getattr(method, "chat_id", None)
        if chat_id is None:  # getUpdates, getMe, getFile, …: not limited
            return await make_request(bot, method)
        group = (
            self._group_limiter(chat_id)
            if _is_group(chat_id) and method.__api_method__ in MESSAGE_CREATING_METHODS
            else None
        )
        for attempt in range(self._max_retries + 1):
            await self._not_paused.wait()
            try:
                if group is not None:
                    async with group, self._overall:
                        return await make_request(bot, method)
                async with self._overall:
                    return await make_request(bot, method)
            except TelegramRetryAfter as e:
                if attempt >= self._max_retries:
                    raise
                logger.info(
                    "Rate limit hit (%s). Retrying after %ss",
                    method.__api_method__,
                    e.retry_after,
                )
                self._not_paused.clear()
                try:
                    await asyncio.sleep(e.retry_after + 0.1)
                finally:
                    self._not_paused.set()
        raise AssertionError("unreachable")  # pragma: no cover


class UserDataMiddleware(BaseMiddleware):
    """Inject ``user_data``: a per-user dict kept in memory for the process."""

    def __init__(self) -> None:
        self._store: dict[int, dict[str, Any]] = {}

    def for_user(self, user_id: int) -> dict[str, Any]:
        return self._store.setdefault(user_id, {})

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user: User | None = data.get("event_from_user")
        data["user_data"] = self.for_user(user.id) if user else {}
        return await handler(event, data)


def build_bot(token: str, *, request_timeout: float = 30.0) -> Bot:
    """The bot client: aiohttp session with rate limiting, previews off."""
    session = AiohttpSession(timeout=request_timeout)
    session.middleware(TelegramRateLimiter())
    return Bot(
        token,
        session=session,
        default=DefaultBotProperties(link_preview=LinkPreviewOptions(is_disabled=True)),
    )
