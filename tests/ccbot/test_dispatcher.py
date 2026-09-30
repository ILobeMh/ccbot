"""Tests for bot.build_dispatcher — router order, filters and middlewares.

The structural tests pin the handler order that python-telegram-bot's
groups/registration order used to give; the feed_update tests push fake
updates through the real dispatcher (no network: the bot session fails
loudly if anything tries to reach Telegram).
"""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.types import (
    CallbackQuery,
    Chat,
    Document,
    ForumTopicCreated,
    Message,
    MessageEntity,
    PhotoSize,
    Sticker,
    Update,
    User,
)

from ccbot import bot as bot_mod
from ccbot.handlers import special_topics as st
from ccbot.session import SessionManager
from ccbot.telegram_client import UserDataMiddleware

USER_ID = 1
CHAT_ID = -100123


def _main_router(dp: Dispatcher):
    return next(r for r in dp.sub_routers if r.name == "main")


def _special_router(dp: Dispatcher):
    return next(r for r in dp.sub_routers if r.name == "special_topics")


class TestStructure:
    def test_special_router_checked_before_main(self):
        dp = bot_mod.build_dispatcher()
        assert [r.name for r in dp.sub_routers] == ["special_topics", "main"]
        special = _special_router(dp)
        assert [h.callback for h in special.message.handlers] == [
            st.special_message_router
        ]
        assert [h.callback for h in special.callback_query.handlers] == [
            st.special_callback_router
        ]

    def test_main_message_handler_order(self):
        dp = bot_mod.build_dispatcher()
        main = _main_router(dp)
        assert [h.callback for h in main.message.handlers] == [
            bot_mod.start_command,
            bot_mod.history_command,
            bot_mod.screenshot_command,
            bot_mod.esc_command,
            bot_mod.restart_command,
            bot_mod.mode_command,
            bot_mod.info_command,
            bot_mod.settings_command,
            bot_mod.resume_command,
            bot_mod.sessions_command,
            bot_mod.kill_command,
            bot_mod.unbind_command,
            bot_mod.usage_command,
            bot_mod.topic_closed_handler,
            bot_mod.topic_edited_handler,
            bot_mod.forward_command_handler,
            bot_mod.text_handler,
            bot_mod.photo_handler,
            bot_mod.voice_handler,
            bot_mod.unsupported_content_handler,
        ]
        assert [h.callback for h in main.callback_query.handlers] == [
            bot_mod.callback_handler
        ]

    def test_user_data_middleware_is_shared_outer(self):
        dp = bot_mod.build_dispatcher()
        on_message = [
            m for m in dp.message.outer_middleware if isinstance(m, UserDataMiddleware)
        ]
        on_callback = [
            m
            for m in dp.callback_query.outer_middleware
            if isinstance(m, UserDataMiddleware)
        ]
        assert len(on_message) == 1 and len(on_callback) == 1
        assert on_message[0] is on_callback[0]


# ── feed_update: real routing through the dispatcher ─────────────────────


class FakeTopic:
    name = "shell"
    callback_prefixes = ("sh:",)

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def handle_text(self, message, bot, user_data, text):
        self.texts.append(text)

    async def handle_callback(self, query, bot, user_data, data):
        pass

    async def on_ready(self, bot, chat_id, thread_id):
        pass


@pytest.fixture
def env(monkeypatch):
    """Isolated session state with a "shell" special topic on thread 99."""
    monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
    monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
    mgr = SessionManager()
    mgr.special_topics["shell"] = 99
    monkeypatch.setattr(st, "session_manager", mgr)
    monkeypatch.setattr(st.config, "special_topics", {"shell"})
    monkeypatch.setattr(st.config, "allowed_users", {USER_ID})
    monkeypatch.setattr(st, "_REGISTRY", {})
    topic = FakeTopic()
    st.register(topic)
    return topic


@pytest.fixture
def bot(monkeypatch):
    b = Bot("123456:TEST")
    # getMe answer cached up front: Command() checks "/cmd@mention" against it
    b._me = User(id=123456, is_bot=True, first_name="bot", username="ccbot_test")
    monkeypatch.setattr(
        b.session,
        "make_request",
        AsyncMock(side_effect=AssertionError("unexpected Telegram API call")),
    )
    return b


def _record(calls: list[tuple[str, Any]], name: str):
    async def handler(message: Message, user_data: dict[str, Any]) -> None:
        calls.append((name, message))

    return handler


@pytest.fixture
def recorded(monkeypatch):
    """Replace the main message handlers with recorders (before building)."""
    calls: list[tuple[str, Any]] = []
    for name in (
        "start_command",
        "forward_command_handler",
        "text_handler",
        "photo_handler",
        "voice_handler",
        "topic_closed_handler",
        "unsupported_content_handler",
    ):
        monkeypatch.setattr(bot_mod, name, _record(calls, name))
    monkeypatch.setattr(
        bot_mod,
        "_COMMAND_HANDLERS",
        (("start", bot_mod.start_command),) + bot_mod._COMMAND_HANDLERS[1:],
    )
    return calls


def _message(thread_id: int | None = 42, **fields: Any) -> Message:
    return Message(
        message_id=1,
        date=datetime.now(timezone.utc),
        chat=Chat(id=CHAT_ID, type="supergroup", is_forum=True),
        from_user=User(id=USER_ID, is_bot=False, first_name="u"),
        message_thread_id=thread_id,
        is_topic_message=thread_id is not None,
        **fields,
    )


def _command(text: str) -> dict[str, Any]:
    """Text with the bot_command entity Telegram adds to a leading /command."""
    length = len(text.split()[0])
    return {
        "text": text,
        "entities": [MessageEntity(type="bot_command", offset=0, length=length)],
    }


async def _feed(dp: Dispatcher, bot: Bot, message: Message) -> None:
    await dp.feed_update(bot, Update(update_id=1, message=message))


class TestFeedUpdate:
    @pytest.mark.asyncio
    async def test_text_in_unbound_topic_reaches_text_handler(
        self, env, bot, monkeypatch
    ):
        seen: list[tuple[Message, dict[str, Any], str]] = []

        async def fake_handle_text(message, _bot, user_data, text):
            seen.append((message, user_data, text))

        monkeypatch.setattr(bot_mod, "_handle_text", fake_handle_text)
        monkeypatch.setattr(bot_mod.config, "text_merge_window", 0.0)
        dp = bot_mod.build_dispatcher()
        await _feed(dp, bot, _message(text="hello"))
        assert len(seen) == 1
        message, user_data, text = seen[0]
        assert text == "hello" and message.message_thread_id == 42
        assert isinstance(user_data, dict)
        assert env.texts == []

    @pytest.mark.asyncio
    async def test_text_in_special_topic_does_not_reach_main_router(
        self, env, bot, recorded
    ):
        dp = bot_mod.build_dispatcher()
        await _feed(dp, bot, _message(thread_id=99, text="ls -la"))
        assert env.texts == ["ls -la"]
        assert recorded == []

    @pytest.mark.asyncio
    async def test_user_data_persists_per_user(self, env, bot, monkeypatch):
        stores: list[dict[str, Any]] = []

        async def fake_handle_text(message, _bot, user_data, text):
            user_data.setdefault("n", 0)
            user_data["n"] += 1
            stores.append(user_data)

        monkeypatch.setattr(bot_mod, "_handle_text", fake_handle_text)
        monkeypatch.setattr(bot_mod.config, "text_merge_window", 0.0)
        dp = bot_mod.build_dispatcher()
        await _feed(dp, bot, _message(text="a"))
        await _feed(dp, bot, _message(text="b"))
        assert stores[0] is stores[1] and stores[1]["n"] == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("fields", "expected"),
        [
            (_command("/start"), "start_command"),
            (_command("/Start"), "start_command"),
            (_command("/start@ccbot_test now"), "start_command"),
            (_command("/model"), "forward_command_handler"),
            (_command("/start@other_bot"), "forward_command_handler"),
            # No bot_command entity: a typed path is text, not a command
            ({"text": "/home/me/project"}, "text_handler"),
            ({"text": "hello"}, "text_handler"),
            (
                {
                    "photo": [
                        PhotoSize(file_id="f", file_unique_id="u", width=1, height=1)
                    ]
                },
                "photo_handler",
            ),
            (
                {
                    "document": Document(
                        file_id="f", file_unique_id="u", mime_type="image/png"
                    )
                },
                "photo_handler",
            ),
            (
                {
                    "document": Document(
                        file_id="f", file_unique_id="u", mime_type="application/pdf"
                    )
                },
                "unsupported_content_handler",
            ),
            (
                {
                    "sticker": Sticker(
                        file_id="f",
                        file_unique_id="u",
                        type="regular",
                        width=1,
                        height=1,
                        is_animated=False,
                        is_video=False,
                    )
                },
                "unsupported_content_handler",
            ),
            ({"forum_topic_closed": {}}, "topic_closed_handler"),
        ],
    )
    async def test_message_routing(self, env, bot, recorded, fields, expected):
        dp = bot_mod.build_dispatcher()
        await _feed(dp, bot, _message(**fields))
        assert [name for name, _ in recorded] == [expected]

    @pytest.mark.asyncio
    async def test_service_message_is_not_unsupported(self, env, bot, recorded):
        dp = bot_mod.build_dispatcher()
        created = ForumTopicCreated(name="new", icon_color=0)
        await _feed(dp, bot, _message(forum_topic_created=created))
        assert recorded == []

    @pytest.mark.asyncio
    async def test_callback_outside_special_topics_reaches_callback_handler(
        self, env, bot, monkeypatch
    ):
        seen: list[tuple[str, dict[str, Any]]] = []

        async def fake_callback(query: CallbackQuery, user_data: dict[str, Any]):
            seen.append((query.data or "", user_data))

        monkeypatch.setattr(bot_mod, "callback_handler", fake_callback)
        dp = bot_mod.build_dispatcher()
        query = CallbackQuery(
            id="q",
            from_user=User(id=USER_ID, is_bot=False, first_name="u"),
            chat_instance="c",
            message=_message(text="menu"),
            data="noop",
        )
        await dp.feed_update(bot, Update(update_id=2, callback_query=query))
        assert [d for d, _ in seen] == ["noop"]
        assert isinstance(seen[0][1], dict)
