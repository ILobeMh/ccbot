"""Tests for photo_handler — single photos, image documents and albums."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccbot import bot as bot_mod
from ccbot.bot import build_image_prompt, photo_handler


def _make_photo_update(
    *,
    unique_id: str,
    caption: str | None = None,
    media_group_id: str | None = None,
    user_id: int = 1,
    thread_id: int = 42,
) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    msg = MagicMock()
    update.message = msg
    msg.message_thread_id = thread_id
    msg.is_topic_message = True
    msg.caption = caption
    msg.media_group_id = media_group_id
    msg.document = None
    msg.chat = MagicMock()
    msg.chat.type = "supergroup"
    msg.chat.id = 100
    msg.chat.send_action = AsyncMock()
    photo = MagicMock()
    photo.file_unique_id = unique_id
    tg_file = MagicMock()
    tg_file.download_to_drive = AsyncMock()
    photo.get_file = AsyncMock(return_value=tg_file)
    msg.photo = [photo]
    return update


def _patches(sent: list[str]):
    async def _send(_wid: str, text: str) -> tuple[bool, str]:
        sent.append(text)
        return True, "ok"

    return (
        patch("ccbot.bot.is_user_allowed", return_value=True),
        patch("ccbot.bot._get_thread_id", return_value=42),
        patch.object(
            bot_mod.session_manager, "get_window_for_thread", return_value="@1"
        ),
        patch.object(bot_mod.session_manager, "set_group_chat_id"),
        patch.object(bot_mod.session_manager, "send_to_window", side_effect=_send),
        patch.object(
            bot_mod.tmux_manager,
            "find_window_by_id",
            AsyncMock(return_value=MagicMock()),
        ),
        patch("ccbot.bot.safe_reply", AsyncMock()),
        patch("ccbot.bot.clear_status_msg_info"),
        patch("ccbot.bot._ALBUM_FLUSH_DELAY", 0.05),
    )


class TestBuildImagePrompt:
    def test_single_with_caption(self):
        text = build_image_prompt("what is this?", [Path("/i/a.jpg")])
        assert text == "what is this?\n\n(image attached: /i/a.jpg)"

    def test_single_without_caption(self):
        text = build_image_prompt("", [Path("/i/a.jpg")])
        assert text.startswith("Please look at the attached image.")
        assert "(image attached: /i/a.jpg)" in text

    def test_multiple_with_caption(self):
        text = build_image_prompt("compare", [Path("/i/a.jpg"), Path("/i/b.png")])
        assert text.startswith("compare\n\n(2 images attached:\n")
        assert "- /i/a.jpg\n- /i/b.png)" in text


class TestPhotoHandler:
    def setup_method(self):
        bot_mod._pending_images.clear()
        bot_mod._pending_albums.clear()

    @pytest.mark.asyncio
    async def test_single_photo_is_staged_not_sent(self):
        sent: list[str] = []
        update = _make_photo_update(unique_id="u1", caption="hi")
        ps = _patches(sent)
        for p in ps:
            p.start()
        try:
            await photo_handler(update, MagicMock())
        finally:
            for p in ps:
                p.stop()
        assert sent == []
        pending = bot_mod._pending_images[(1, 42)]
        assert pending["caption"] == "hi"
        assert len(pending["paths"]) == 1

    @pytest.mark.asyncio
    async def test_text_after_staging_sends_images_with_text(self):
        sent: list[str] = []
        photo = _make_photo_update(unique_id="u1", caption="cap")
        ps = _patches(sent)
        for p in ps:
            p.start()
        try:
            await photo_handler(photo, MagicMock())
            reply = MagicMock()
            reply.chat.send_action = AsyncMock()
            consumed = await bot_mod._deliver_pending_images(reply, 1, 42, "long text")
        finally:
            for p in ps:
                p.stop()
        assert consumed
        assert len(sent) == 1
        assert sent[0].startswith("cap\n\nlong text\n\n(image attached: ")
        assert sent[0].endswith("_u1.jpg)")
        assert (1, 42) not in bot_mod._pending_images

    @pytest.mark.asyncio
    async def test_skip_sends_images_without_text(self):
        sent: list[str] = []
        photo = _make_photo_update(unique_id="u1")
        ps = _patches(sent)
        for p in ps:
            p.start()
        try:
            await photo_handler(photo, MagicMock())
            reply = MagicMock()
            reply.chat.send_action = AsyncMock()
            await bot_mod._deliver_pending_images(reply, 1, 42, "")
        finally:
            for p in ps:
                p.stop()
        assert len(sent) == 1
        assert sent[0].startswith("Please look at the attached image.")

    @pytest.mark.asyncio
    async def test_cancel_discards_images(self, tmp_path):
        f = tmp_path / "x.jpg"
        f.write_bytes(b"x")
        bot_mod._pending_images[(1, 42)] = {
            "paths": [f],
            "caption": "",
            "wid": "@1",
            "prompt_msg": None,
        }
        assert bot_mod._discard_pending_images(1, 42) == 1
        assert not f.exists()
        assert (1, 42) not in bot_mod._pending_images

    @pytest.mark.asyncio
    async def test_album_is_staged_as_one_group(self):
        sent: list[str] = []
        u1 = _make_photo_update(unique_id="a1", caption="look", media_group_id="g")
        u2 = _make_photo_update(unique_id="a2", media_group_id="g")
        u3 = _make_photo_update(unique_id="a3", media_group_id="g")
        ps = _patches(sent)
        for p in ps:
            p.start()
        try:
            for u in (u1, u2, u3):
                await photo_handler(u, MagicMock())
            assert (1, 42) not in bot_mod._pending_images  # album not settled
            await asyncio.sleep(0.2)
            pending = bot_mod._pending_images[(1, 42)]
            assert len(pending["paths"]) == 3
            assert pending["caption"] == "look"
            reply = MagicMock()
            reply.chat.send_action = AsyncMock()
            await bot_mod._deliver_pending_images(reply, 1, 42, "compare them")
        finally:
            for p in ps:
                p.stop()
        assert len(sent) == 1
        text = sent[0]
        assert text.startswith("look\n\ncompare them\n\n(3 images attached:\n")
        for uid in ("a1", "a2", "a3"):
            assert f"_{uid}.jpg" in text
        assert not bot_mod._pending_albums
