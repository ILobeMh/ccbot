"""Tests for telegram_client: rate limiter request middleware, user_data."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram.exceptions import TelegramRetryAfter
from aiogram.methods import DeleteMessage, EditMessageText, GetMe, SendMessage

from ccbot import telegram_client as tc


def _limiter(**kw) -> tc.TelegramRateLimiter:
    kw.setdefault("prefill", False)  # tests shouldn't wait for the pre-fill
    return tc.TelegramRateLimiter(**kw)


@pytest.mark.asyncio
async def test_retry_after_is_retried_and_pauses(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    monkeypatch.setattr(tc.asyncio, "sleep", fake_sleep)
    lim = _limiter(max_retries=2)
    method = SendMessage(chat_id=-100, text="hi")
    calls = {"n": 0}

    async def make_request(bot, m):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TelegramRetryAfter(method=m, message="flood", retry_after=3)
        return "ok"

    assert await lim(make_request, MagicMock(), method) == "ok"
    assert calls["n"] == 2
    assert sleeps == [3.1]
    assert lim._not_paused.is_set()


@pytest.mark.asyncio
async def test_retry_after_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(tc.asyncio, "sleep", AsyncMock())
    lim = _limiter(max_retries=1)
    method = SendMessage(chat_id=-100, text="hi")

    async def make_request(bot, m):
        raise TelegramRetryAfter(method=m, message="flood", retry_after=1)

    with pytest.raises(TelegramRetryAfter):
        await lim(make_request, MagicMock(), method)
    assert lim._not_paused.is_set()


@pytest.mark.asyncio
async def test_group_budget_only_for_message_creating_calls():
    lim = _limiter()
    ok = AsyncMock(return_value="ok")
    await lim(ok, MagicMock(), EditMessageText(chat_id=-100, message_id=1, text="x"))
    await lim(ok, MagicMock(), DeleteMessage(chat_id=-100, message_id=1))
    assert lim._groups == {}
    await lim(ok, MagicMock(), SendMessage(chat_id=-100, text="x"))
    assert set(lim._groups) == {-100}
    await lim(ok, MagicMock(), SendMessage(chat_id=42, text="private"))
    assert set(lim._groups) == {-100}


@pytest.mark.asyncio
async def test_calls_without_chat_id_bypass_limits():
    lim = tc.TelegramRateLimiter(overall_per_second=1)  # pre-filled: 1 s wait
    ok = AsyncMock(return_value="me")
    assert await asyncio.wait_for(lim(ok, MagicMock(), GetMe()), 0.5) == "me"


@pytest.mark.asyncio
async def test_prefill_really_throttles_the_first_burst():
    """After a restart the bucket starts full: a burst drains in at the rate."""
    lim = tc.TelegramRateLimiter(overall_per_second=100)
    ok = AsyncMock(return_value="ok")
    loop = asyncio.get_running_loop()
    start = loop.time()
    for _ in range(20):
        await lim(ok, MagicMock(), SendMessage(chat_id=5, text="x"))
    assert loop.time() - start >= 0.15  # ~20 / 100 s, not instant

    cold = tc.TelegramRateLimiter(overall_per_second=100, prefill=False)
    start = loop.time()
    for _ in range(20):
        await cold(ok, MagicMock(), SendMessage(chat_id=5, text="x"))
    assert loop.time() - start < 0.1


@pytest.mark.parametrize(
    ("chat_id", "expected"),
    [(-1001, True), (5, False), ("@chan", True), ("-100", True), ("7", False)],
)
def test_is_group(chat_id, expected):
    assert tc._is_group(chat_id) is expected


@pytest.mark.asyncio
async def test_user_data_is_per_user_and_persistent():
    mw = tc.UserDataMiddleware()
    seen: list[dict] = []

    async def handler(event, data):
        seen.append(data["user_data"])
        data["user_data"]["n"] = data["user_data"].get("n", 0) + 1

    u1, u2 = MagicMock(id=1), MagicMock(id=2)
    await mw(handler, MagicMock(), {"event_from_user": u1})
    await mw(handler, MagicMock(), {"event_from_user": u1})
    await mw(handler, MagicMock(), {"event_from_user": u2})
    assert seen[0] is seen[1] and seen[0]["n"] == 2
    assert seen[2] == {"n": 1}


def test_build_bot_installs_rate_limiter():
    bot = tc.build_bot("123456:ABCDEF")
    assert any(
        isinstance(m, tc.TelegramRateLimiter)
        for m in bot.session.middleware._middlewares
    )
