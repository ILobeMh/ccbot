"""Tests for the special (bot-owned) topic registry and routers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import UnpinAllForumTopicMessages

import ccbot.handlers.special_topics as st
from ccbot.session import SessionManager


class FakeTopic:
    name = "shell"
    callback_prefixes = ("sh:",)

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.callbacks: list[str] = []
        self.ready: list[tuple[int, int]] = []

    async def handle_text(self, message, bot, user_data, text):
        self.texts.append(text)

    async def handle_callback(self, query, bot, user_data, data):
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


def _message(
    text: str, thread_id: int | None, user_id: int = 1, chat_type="supergroup"
):
    msg = MagicMock()
    msg.text = text
    msg.message_thread_id = thread_id
    msg.from_user = SimpleNamespace(id=user_id)
    msg.chat = SimpleNamespace(id=-100, type=chat_type)
    return msg


def _query(data: str, user_id: int = 1):
    query = MagicMock()
    query.data = data
    query.from_user = SimpleNamespace(id=user_id)
    query.answer = AsyncMock()
    return query


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
        # aiogram keeps Telegram's raw description (PTB capitalized it)
        bot.unpin_all_forum_topic_messages = AsyncMock(
            side_effect=TelegramBadRequest(
                method=UnpinAllForumTopicMessages(chat_id=-100123, message_thread_id=5),
                message="Bad Request: TOPIC_ID_INVALID",
            )
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
        # Returning (no SkipHandler) consumes the update
        await st.special_message_router(_message("ls -la", 42), MagicMock(), {})
        assert topic.texts == ["ls -la"]

    @pytest.mark.asyncio
    async def test_other_thread_passes_through(self, env):
        mgr, topic = env
        mgr.special_topics["shell"] = 42
        with pytest.raises(SkipHandler):
            await st.special_message_router(_message("hello", 7), MagicMock(), {})
        assert topic.texts == []

    @pytest.mark.asyncio
    async def test_group_chat_id_captured_on_pass_through(self, env):
        mgr, _ = env
        mgr.special_topics["shell"] = 42
        with pytest.raises(SkipHandler):
            await st.special_message_router(_message("hello", 7), MagicMock(), {})
        assert -100 in mgr.group_chat_ids.values()

    @pytest.mark.asyncio
    async def test_unauthorized_user_blocked(self, env, monkeypatch):
        mgr, topic = env
        mgr.special_topics["shell"] = 42
        reply = AsyncMock()
        monkeypatch.setattr(st, "safe_reply", reply)
        await st.special_message_router(
            _message("rm -rf /", 42, user_id=2), MagicMock(), {}
        )
        assert topic.texts == []
        reply.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_callback_router(self, env):
        _, topic = env
        await st.special_callback_router(_query("sh:kill:1"), MagicMock(), {})
        assert topic.callbacks == ["sh:kill:1"]

    @pytest.mark.asyncio
    async def test_callback_unauthorized_answered(self, env):
        _, topic = env
        query = _query("sh:kill:1", user_id=2)
        await st.special_callback_router(query, MagicMock(), {})
        assert topic.callbacks == []
        query.answer.assert_awaited_once_with("Not authorized")

    @pytest.mark.asyncio
    async def test_callback_other_prefix_passes(self, env):
        _, topic = env
        with pytest.raises(SkipHandler):
            await st.special_callback_router(_query("db:sel:1"), MagicMock(), {})
        assert topic.callbacks == []
