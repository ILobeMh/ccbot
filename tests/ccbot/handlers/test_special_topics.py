"""Tests for the special (bot-owned) topic registry and routers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import BadRequest
from telegram.ext import ApplicationHandlerStop

import ccbot.handlers.special_topics as st
from ccbot.session import SessionManager


class FakeTopic:
    name = "shell"
    callback_prefixes = ("sh:",)

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.callbacks: list[str] = []
        self.ready: list[tuple[int, int]] = []

    async def handle_text(self, update, context, text):
        self.texts.append(text)

    async def handle_callback(self, update, context, data):
        self.callbacks.append(data)

    async def on_ready(self, bot, chat_id, thread_id):
        self.ready.append((chat_id, thread_id))


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
    monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
    mgr = SessionManager()
    monkeypatch.setattr(st, "session_manager", mgr)
    monkeypatch.setattr(st.config, "special_topics", {"shell"})
    monkeypatch.setattr(st.config, "forum_chat_id", None)
    monkeypatch.setattr(st.config, "allowed_users", {1})
    monkeypatch.setattr(st, "_REGISTRY", {})
    topic = FakeTopic()
    st.register(topic)
    return mgr, topic


def _update(text: str, thread_id: int | None, user_id: int = 1, chat_type="supergroup"):
    msg = MagicMock()
    msg.text = text
    msg.message_thread_id = thread_id
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        message=msg,
        effective_chat=SimpleNamespace(id=-100, type=chat_type),
        callback_query=None,
    )


class TestRegistry:
    def test_register_respects_config(self, env, monkeypatch):
        monkeypatch.setattr(st.config, "special_topics", set())
        monkeypatch.setattr(st, "_REGISTRY", {})
        st.register(FakeTopic())
        assert st.registered() == {}

    def test_topic_for_thread(self, env):
        mgr, topic = env
        mgr.special_topics["shell"] = 42
        assert st.topic_for_thread(42) is topic
        assert st.topic_for_thread(43) is None
        assert st.is_special_thread(42)

    def test_forum_chat_id_learned(self, env, monkeypatch):
        mgr, _ = env
        assert st.forum_chat_id() is None
        mgr.group_chat_ids["1:5"] = -100123
        assert st.forum_chat_id() == -100123
        monkeypatch.setattr(st.config, "forum_chat_id", -100999)
        assert st.forum_chat_id() == -100999


class TestEnsure:
    @pytest.mark.asyncio
    async def test_creates_missing_topic(self, env):
        mgr, topic = env
        mgr.group_chat_ids["1:5"] = -100123
        bot = MagicMock()
        bot.create_forum_topic = AsyncMock(
            return_value=SimpleNamespace(message_thread_id=77)
        )
        bot.unpin_all_forum_topic_messages = AsyncMock()
        await st.ensure_special_topics(bot)
        bot.create_forum_topic.assert_awaited_once_with(chat_id=-100123, name="shell")
        assert mgr.special_topics == {"shell": 77}
        assert topic.ready == [(-100123, 77)]

    @pytest.mark.asyncio
    async def test_recreates_deleted_topic(self, env):
        mgr, topic = env
        mgr.group_chat_ids["1:5"] = -100123
        mgr.special_topics["shell"] = 5
        bot = MagicMock()
        bot.unpin_all_forum_topic_messages = AsyncMock(
            side_effect=BadRequest("Topic_id_invalid")
        )
        bot.create_forum_topic = AsyncMock(
            return_value=SimpleNamespace(message_thread_id=78)
        )
        await st.ensure_special_topics(bot)
        assert mgr.special_topics == {"shell": 78}

    @pytest.mark.asyncio
    async def test_keeps_existing_topic(self, env):
        mgr, topic = env
        mgr.group_chat_ids["1:5"] = -100123
        mgr.special_topics["shell"] = 5
        bot = MagicMock()
        bot.unpin_all_forum_topic_messages = AsyncMock()
        bot.create_forum_topic = AsyncMock()
        await st.ensure_special_topics(bot)
        bot.create_forum_topic.assert_not_awaited()
        assert topic.ready == [(-100123, 5)]

    @pytest.mark.asyncio
    async def test_no_chat_id_yet(self, env):
        _, topic = env
        bot = MagicMock()
        bot.create_forum_topic = AsyncMock()
        await st.ensure_special_topics(bot)
        bot.create_forum_topic.assert_not_awaited()
        assert topic.ready == []


class TestRouters:
    @pytest.mark.asyncio
    async def test_special_thread_is_handled_and_stops(self, env):
        mgr, topic = env
        mgr.special_topics["shell"] = 42
        ctx = SimpleNamespace(bot=MagicMock())
        with pytest.raises(ApplicationHandlerStop):
            await st.special_message_router(_update("ls -la", 42), ctx)
        assert topic.texts == ["ls -la"]

    @pytest.mark.asyncio
    async def test_other_thread_passes_through(self, env):
        mgr, topic = env
        mgr.special_topics["shell"] = 42
        ctx = SimpleNamespace(bot=MagicMock())
        await st.special_message_router(_update("hello", 7), ctx)  # no raise
        assert topic.texts == []

    @pytest.mark.asyncio
    async def test_unauthorized_user_blocked(self, env, monkeypatch):
        mgr, topic = env
        mgr.special_topics["shell"] = 42
        monkeypatch.setattr(st, "safe_reply", AsyncMock())
        ctx = SimpleNamespace(bot=MagicMock())
        with pytest.raises(ApplicationHandlerStop):
            await st.special_message_router(_update("rm -rf /", 42, user_id=2), ctx)
        assert topic.texts == []

    @pytest.mark.asyncio
    async def test_callback_router(self, env):
        _, topic = env
        query = MagicMock()
        query.data = "sh:kill:1"
        query.answer = AsyncMock()
        update = SimpleNamespace(
            callback_query=query, effective_user=SimpleNamespace(id=1)
        )
        with pytest.raises(ApplicationHandlerStop):
            await st.special_callback_router(update, SimpleNamespace())
        assert topic.callbacks == ["sh:kill:1"]

    @pytest.mark.asyncio
    async def test_callback_other_prefix_passes(self, env):
        _, topic = env
        query = MagicMock()
        query.data = "db:sel:1"
        update = SimpleNamespace(
            callback_query=query, effective_user=SimpleNamespace(id=1)
        )
        await st.special_callback_router(update, SimpleNamespace())
        assert topic.callbacks == []
