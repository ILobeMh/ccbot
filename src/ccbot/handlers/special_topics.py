"""Bot-owned "special" forum topics that are not bound to a Claude Code window.

A special topic is created by the bot itself (e.g. ``shell``, ``ccc``) and
every message in it is handled by a registered SpecialTopic implementation
instead of the normal topic→window→session routing.

Key components:
  - SpecialTopic: interface (name, handle_text, handle_command, callbacks)
  - register(): add an implementation to the registry (done at import time
    by the feature modules; see bot.create_bot)
  - ensure_special_topics(bot): create missing topics in the forum group and
    persist their thread ids in state.json (special_topics)
  - special_message_router / special_callback_router: PTB handlers placed in
    a negative group; they raise ApplicationHandlerStop for special topics so
    the regular handlers never see those updates
"""

from __future__ import annotations

import logging
from typing import Protocol

from telegram import Bot, Update
from telegram.error import BadRequest, TelegramError
from telegram.ext import ApplicationHandlerStop, ContextTypes

from ..config import config
from ..session import session_manager
from .message_sender import safe_reply

logger = logging.getLogger(__name__)


class SpecialTopic(Protocol):
    """A bot-owned topic. ``name`` is also the forum topic title."""

    name: str
    callback_prefixes: tuple[str, ...]

    async def handle_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str
    ) -> None: ...

    async def handle_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, data: str
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
    except BadRequest as e:
        if "Topic_id_invalid" in str(e) or "thread not found" in str(e).lower():
            return False
        logger.debug("Topic probe for %s: %s", thread_id, e)
        return True
    except TelegramError as e:
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
            except TelegramError as e:
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
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Group -1 handler: dispatch messages in special topics and stop."""
    user = update.effective_user
    msg = update.message
    if not user or not msg or not msg.text:
        return
    chat = update.effective_chat
    thread_id = getattr(msg, "message_thread_id", None)
    if thread_id == 1:
        thread_id = None
    if chat and chat.type in ("group", "supergroup"):
        session_manager.set_group_chat_id(user.id, thread_id, chat.id)
        # Topics could not be created at startup without a known chat id
        if _REGISTRY and not all(
            n in session_manager.special_topics for n in _REGISTRY
        ):
            await ensure_special_topics(context.bot)

    topic = topic_for_thread(thread_id)
    if topic is None:
        return
    if not config.is_user_allowed(user.id):
        await safe_reply(msg, "You are not authorized to use this bot.")
        raise ApplicationHandlerStop
    try:
        await topic.handle_text(update, context, msg.text)
    except Exception as e:
        logger.exception("Special topic %r failed", topic.name)
        await safe_reply(msg, f"❌ {topic.name}: {e}")
    raise ApplicationHandlerStop


async def special_callback_router(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Group -1 handler: dispatch callback queries owned by special topics."""
    query = update.callback_query
    if not query or not query.data:
        return
    for topic in _REGISTRY.values():
        if topic.callback_prefixes and query.data.startswith(topic.callback_prefixes):
            user = update.effective_user
            if not user or not config.is_user_allowed(user.id):
                await query.answer("Not authorized")
                raise ApplicationHandlerStop
            try:
                await topic.handle_callback(update, context, query.data)
            except Exception as e:
                logger.exception("Special topic %r callback failed", topic.name)
                await query.answer(f"Error: {e}"[:200], show_alert=True)
            raise ApplicationHandlerStop
