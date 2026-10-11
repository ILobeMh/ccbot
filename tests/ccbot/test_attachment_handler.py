"""Tests for attachment_handler — photos, files of any kind, and albums."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram.types import Animation, Document, Sticker, Video, VideoNote

from ccbot import bot as bot_mod
from ccbot.bot import attachment_handler, build_attachment_prompt


def _make_photo_message(
    *,
    unique_id: str,
    caption: str | None = None,
    media_group_id: str | None = None,
    user_id: int = 1,
    thread_id: int = 42,
) -> MagicMock:
    msg = MagicMock()
    msg.from_user = MagicMock()
    msg.from_user.id = user_id
    msg.message_thread_id = thread_id
    msg.is_topic_message = True
    msg.caption = caption
    msg.media_group_id = media_group_id
    msg.document = None
    msg.chat = MagicMock()
    msg.chat.type = "supergroup"
    msg.chat.id = 100
    photo = MagicMock()
    photo.file_id = f"file-{unique_id}"
    photo.file_unique_id = unique_id
    msg.photo = [photo]
    return msg


def _make_bot() -> MagicMock:
    bot = MagicMock()
    bot.download = AsyncMock()
    bot.send_chat_action = AsyncMock()
    return bot


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


class TestBuildAttachmentPrompt:
    def test_single_with_caption_mention_first(self):
        text = build_attachment_prompt("what is this?", [Path("/i/a.jpg")])
        assert text == "@/i/a.jpg\nwhat is this?"

    def test_single_without_caption(self):
        text = build_attachment_prompt("", [Path("/i/a.pdf")])
        assert text == "@/i/a.pdf\nPlease look at the attached file."

    def test_whitespace_caption_counts_as_no_caption(self):
        text = build_attachment_prompt("  \n ", [Path("/i/a.pdf")])
        assert text == "@/i/a.pdf\nPlease look at the attached file."

    def test_multiple_with_caption(self):
        text = build_attachment_prompt("compare", [Path("/i/a.jpg"), Path("/i/b.png")])
        assert text == "@/i/a.jpg\n@/i/b.png\ncompare"

    def test_caption_is_kept_verbatim(self):
        text = build_attachment_prompt(" lead\n\ttab ", [Path("/i/a.zip")])
        assert text == "@/i/a.zip\n lead\n\ttab "


class TestPhotoHandler:
    def setup_method(self):
        bot_mod._pending_attachments.clear()
        bot_mod._pending_albums.clear()

    @pytest.mark.asyncio
    async def test_single_photo_is_staged_not_sent(self):
        sent: list[str] = []
        message = _make_photo_message(unique_id="u1", caption="hi")
        bot = _make_bot()
        ps = _patches(sent)
        for p in ps:
            p.start()
        try:
            await attachment_handler(message, bot)
        finally:
            for p in ps:
                p.stop()
        assert sent == []
        bot.download.assert_awaited_once()
        assert bot.download.await_args.args[0] == "file-u1"
        dest = bot.download.await_args.kwargs["destination"]
        assert str(dest).endswith("_u1.jpg")
        pending = bot_mod._pending_attachments[(1, 42)]
        assert pending["caption"] == "hi"
        assert len(pending["paths"]) == 1

    @pytest.mark.asyncio
    async def test_text_after_staging_sends_images_with_text(self):
        sent: list[str] = []
        photo = _make_photo_message(unique_id="u1", caption="cap")
        ps = _patches(sent)
        for p in ps:
            p.start()
        try:
            bot = _make_bot()
            await attachment_handler(photo, bot)
            consumed = await bot_mod._deliver_pending_attachments(
                MagicMock(), bot, 1, 42, "long text"
            )
        finally:
            for p in ps:
                p.stop()
        assert consumed
        assert len(sent) == 1
        mention, rest = sent[0].split("\n", 1)
        assert mention.startswith("@") and mention.endswith("_u1.jpg")
        assert rest == "cap\n\nlong text"
        assert (1, 42) not in bot_mod._pending_attachments

    @pytest.mark.asyncio
    async def test_skip_sends_images_without_text(self):
        sent: list[str] = []
        photo = _make_photo_message(unique_id="u1")
        ps = _patches(sent)
        for p in ps:
            p.start()
        try:
            bot = _make_bot()
            await attachment_handler(photo, bot)
            await bot_mod._deliver_pending_attachments(MagicMock(), bot, 1, 42, "")
        finally:
            for p in ps:
                p.stop()
        assert len(sent) == 1
        assert sent[0].startswith("@")
        assert sent[0].endswith("\nPlease look at the attached file.")

    @pytest.mark.asyncio
    async def test_cancel_discards_images(self, tmp_path):
        f = tmp_path / "x.jpg"
        f.write_bytes(b"x")
        bot_mod._pending_attachments[(1, 42)] = {
            "paths": [f],
            "caption": "",
            "wid": "@1",
            "prompt_msg": None,
        }
        assert bot_mod._discard_pending_attachments(1, 42) == 1
        assert not f.exists()
        assert (1, 42) not in bot_mod._pending_attachments

    @pytest.mark.asyncio
    async def test_album_is_staged_as_one_group(self):
        sent: list[str] = []
        u1 = _make_photo_message(unique_id="a1", caption="look", media_group_id="g")
        u2 = _make_photo_message(unique_id="a2", media_group_id="g")
        u3 = _make_photo_message(unique_id="a3", media_group_id="g")
        ps = _patches(sent)
        for p in ps:
            p.start()
        try:
            bot = _make_bot()
            for u in (u1, u2, u3):
                await attachment_handler(u, bot)
            assert (1, 42) not in bot_mod._pending_attachments  # album not settled
            await asyncio.sleep(0.2)
            pending = bot_mod._pending_attachments[(1, 42)]
            assert len(pending["paths"]) == 3
            assert pending["caption"] == "look"
            await bot_mod._deliver_pending_attachments(
                MagicMock(), bot, 1, 42, "compare them"
            )
        finally:
            for p in ps:
                p.stop()
        assert len(sent) == 1
        text = sent[0]
        mentions = text.split("\n")[:3]
        for uid, line in zip(("a1", "a2", "a3"), mentions):
            assert line.startswith("@") and line.endswith(f"_{uid}.jpg")
        assert text.endswith("\nlook\n\ncompare them")
        assert not bot_mod._pending_albums


def _file_message(**media) -> MagicMock:
    """A message whose only media is the given aiogram object (other kinds unset)."""
    msg = _make_photo_message(unique_id="unused")
    msg.photo = None
    msg.document = msg.video = msg.audio = msg.animation = None
    msg.video_note = msg.sticker = None
    for field, obj in media.items():
        setattr(msg, field, obj)
    return msg


class TestFileAttachments:
    def setup_method(self):
        bot_mod._pending_attachments.clear()
        bot_mod._pending_albums.clear()

    @pytest.mark.asyncio
    async def test_document_is_downloaded_and_staged(self):
        sent: list[str] = []
        doc = Document(file_id="f", file_unique_id="u9", file_name="report final.pdf")
        message = _file_message(document=doc)
        bot = _make_bot()
        ps = _patches(sent)
        for p in ps:
            p.start()
        try:
            await attachment_handler(message, bot)
        finally:
            for p in ps:
                p.stop()
        assert sent == []
        dest = bot.download.await_args.kwargs["destination"]
        assert dest.parent == bot_mod._UPLOADS_DIR
        assert dest.name.endswith("_u9_report_final.pdf")
        assert len(bot_mod._pending_attachments[(1, 42)]["paths"]) == 1

    @pytest.mark.asyncio
    async def test_sender_name_cannot_escape_uploads_dir(self):
        doc = Document(file_id="f", file_unique_id="u1", file_name="../../etc/passwd")
        message = _file_message(document=doc)
        bot = _make_bot()
        ps = _patches([])
        for p in ps:
            p.start()
        try:
            await attachment_handler(message, bot)
        finally:
            for p in ps:
                p.stop()
        dest = bot.download.await_args.kwargs["destination"]
        assert dest.parent == bot_mod._UPLOADS_DIR
        assert dest.name.endswith("_u1_passwd")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("field", "media", "suffix"),
        [
            (
                "video",
                Video(
                    file_id="f",
                    file_unique_id="u",
                    width=1,
                    height=1,
                    duration=1,
                    mime_type="video/mp4",
                ),
                ".mp4",
            ),
            (
                "animation",
                Animation(
                    file_id="f", file_unique_id="u", width=1, height=1, duration=1
                ),
                "",
            ),
            (
                "video_note",
                VideoNote(file_id="f", file_unique_id="u", length=1, duration=1),
                ".mp4",
            ),
            (
                "sticker",
                Sticker(
                    file_id="f",
                    file_unique_id="u",
                    type="regular",
                    width=1,
                    height=1,
                    is_animated=True,
                    is_video=False,
                ),
                ".tgs",
            ),
            (
                "sticker",
                Sticker(
                    file_id="f",
                    file_unique_id="u",
                    type="regular",
                    width=1,
                    height=1,
                    is_animated=False,
                    is_video=True,
                ),
                ".webm",
            ),
            (
                "sticker",
                Sticker(
                    file_id="f",
                    file_unique_id="u",
                    type="regular",
                    width=1,
                    height=1,
                    is_animated=False,
                    is_video=False,
                ),
                ".webp",
            ),
        ],
    )
    async def test_nameless_media_gets_an_extension(self, field, media, suffix):
        message = _file_message(**{field: media})
        bot = _make_bot()
        ps = _patches([])
        for p in ps:
            p.start()
        try:
            await attachment_handler(message, bot)
        finally:
            for p in ps:
                p.stop()
        dest = bot.download.await_args.kwargs["destination"]
        # "<time>_<unique id><ext>" — no sender name, so the extension is all that follows
        assert dest.name.split("_")[1] == "u" + suffix

    @pytest.mark.asyncio
    async def test_file_over_20_mb_is_refused_without_download(self):
        doc = Document(file_id="f", file_unique_id="u", file_size=25 * 1024 * 1024)
        message = _file_message(document=doc)
        bot = _make_bot()
        reply = AsyncMock()
        ps = _patches([])
        for p in ps:
            p.start()
        try:
            with patch("ccbot.bot.safe_reply", reply):
                await attachment_handler(message, bot)
        finally:
            for p in ps:
                p.stop()
        bot.download.assert_not_awaited()
        assert "too big" in reply.await_args.args[1]
        assert (1, 42) not in bot_mod._pending_attachments
