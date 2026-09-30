"""Bot-owned "special" forum topics that are not bound to a Claude Code window.

A special topic is created by the bot itself (e.g. ``shell``, ``ccc``) and
every message in it is handled by a registered SpecialTopic implementation
instead of the normal topic→window→session routing.

Key components:
  - SpecialTopic: interface (name, handle_text, handle_command, callbacks)
  - register(): add an implementation to the registry (done at import time
    by the feature modules; see bot.build_dispatcher)
  - ensure_special_topics(bot): create missing topics in the forum group and
    persist their thread ids in state.json (special_topics)
  - special_message_router / special_callback_router: handlers on the
    router that bot.build_dispatcher checks before the main one; they
    consume updates for special topics (so the regular handlers never see
    them) and raise SkipHandler for everything else
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from aiogram import Bot
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.enums import ChatType
from aiogram.exceptions import (
    TelegramAPIError,
    TelegramBadRequest,
    TelegramNetworkError,
)
from aiogram.types import CallbackQuery, Message

from ..config import config
from ..session import session_manager
from .message_sender import safe_reply

logger = logging.getLogger(__name__)


class SpecialTopic(Protocol):
    """A bot-owned topic. ``name`` is also the forum topic title."""

    name: str
    callback_prefixes: tuple[str, ...]

    async def handle_text(
        self, message: Message, bot: Bot, user_data: dict[str, Any], text: str
    ) -> None: ...

    async def handle_callback(
        self, query: CallbackQuery, bot: Bot, user_data: dict[str, Any], data: str
    ) -> None: ...

    async def on_ready(self, bot: Bot, chat_id: int, thread_id: int) -> None:
        """Called once per bot start after the topic is known to exist."""
        ...


_REGISTRY: dict[str, SpecialTopic] = {}


def register(topic: SpecialTopic) -> None:
    if topic.name in config.special_topics:
        _REGISTRY[topic.name] = topic
    else:
        logger.info("Special topic %r disabled by CCBOT_SPECIAL_TOPICS", topic.name)


def registered() -> dict[str, SpecialTopic]:
    return dict(_REGISTRY)


def topic_for_thread(thread_id: int | None) -> SpecialTopic | None:
    """The special topic bound to ``thread_id``, if any."""
    if thread_id is None:
        return None
    for name, tid in session_manager.special_topics.items():
        if tid == thread_id and name in _REGISTRY:
            return _REGISTRY[name]
    return None


def is_special_thread(thread_id: int | None) -> bool:
    return topic_for_thread(thread_id) is not None


def forum_chat_id() -> int | None:
    """The supergroup the bot lives in (CCBOT_FORUM_CHAT_ID or learned)."""
    if config.forum_chat_id:
        return config.forum_chat_id
    for chat_id in session_manager.group_chat_ids.values():
        if chat_id < 0:
            return chat_id
    return None


async def _topic_exists(bot: Bot, chat_id: int, thread_id: int) -> bool:
    try:
        # Silent no-op when nothing is pinned; fails with Topic_id_invalid
        # when the topic was deleted.
        await bot.unpin_all_forum_topic_messages(
            chat_id=chat_id, message_thread_id=thread_id
        )
        return True
    except TelegramBadRequest as e:
        # aiogram keeps Telegram's raw text ("Bad Request: TOPIC_ID_INVALID")
        text = str(e).lower()
        if "topic_id_invalid" in text or "thread not found" in text:
            return False
        logger.debug("Topic probe for %s: %s", thread_id, e)
        return True
    except (TelegramAPIError, TelegramNetworkError) as e:
        logger.debug("Topic probe for %s: %s", thread_id, e)
        return True


async def ensure_special_topics(bot: Bot) -> None:
    """Create any missing special topics and notify their implementations."""
    if not _REGISTRY:
        return
    chat_id = forum_chat_id()
    if chat_id is None:
        logger.info(
            "Special topics: forum chat id unknown yet — will create on first "
            "group message (or set CCBOT_FORUM_CHAT_ID)"
        )
        return
    for name, topic in _REGISTRY.items():
        thread_id = session_manager.special_topics.get(name)
        if thread_id is not None and not await _topic_exists(bot, chat_id, thread_id):
            logger.info(
                "Special topic %r (thread %s) is gone; recreating", name, thread_id
            )
            thread_id = None
        if thread_id is None:
            try:
                created = await bot.create_forum_topic(chat_id=chat_id, name=name)
            except (TelegramAPIError, TelegramNetworkError) as e:
                logger.error(
                    "Cannot create special topic %r (needs 'Manage Topics' admin "
                    "right in the group): %s",
                    name,
                    e,
                )
                continue
            thread_id = created.message_thread_id
            session_manager.set_special_topic(name, thread_id)
            logger.info("Created special topic %r as thread %s", name, thread_id)
        try:
            await topic.on_ready(bot, chat_id, thread_id)
        except Exception as e:
            logger.error("Special topic %r on_ready failed: %s", name, e)


async def special_message_router(
    message: Message, bot: Bot, user_data: dict[str, Any]
) -> None:
    """First-checked handler: dispatch messages in special topics and stop.

    Anything outside a special topic raises SkipHandler so the main router
    still gets it.
    """
    user = message.from_user
    if not user or not message.text:
        raise SkipHandler()
    chat = message.chat
    thread_id = message.message_thread_id
    if thread_id == 1:
        thread_id = None
    if chat.type in (ChatType.GROUP, ChatType.SUPERGROUP):
        session_manager.set_group_chat_id(user.id, thread_id, chat.id)
        # Topics could not be created at startup without a known chat id
        if _REGISTRY and not all(
            n in session_manager.special_topics for n in _REGISTRY
        ):
            await ensure_special_topics(bot)

    topic = topic_for_thread(thread_id)
    if topic is None:
        raise SkipHandler()
    if not config.is_user_allowed(user.id):
        await safe_reply(message, "You are not authorized to use this bot.")
        return
    try:
        await topic.handle_text(message, bot, user_data, message.text)
    except Exception as e:
        logger.exception("Special topic %r failed", topic.name)
        await safe_reply(message, f"❌ {topic.name}: {e}")


async def special_callback_router(
    query: CallbackQuery, bot: Bot, user_data: dict[str, Any]
) -> None:
    """First-checked handler: dispatch callback queries owned by special topics."""
    if not query.data:
        raise SkipHandler()
    for topic in _REGISTRY.values():
        if topic.callback_prefixes and query.data.startswith(topic.callback_prefixes):
            user = query.from_user
            if not config.is_user_allowed(user.id):
                await query.answer("Not authorized")
                return
            try:
                await topic.handle_callback(query, bot, user_data, query.data)
            except Exception as e:
                logger.exception("Special topic %r callback failed", topic.name)
                await query.answer(f"Error: {e}"[:200], show_alert=True)
            return
    raise SkipHandler()
