"""The ``settings`` special topic and ``/settings``: live bot configuration.

Shows one dashboard message listing every runtime setting (see
ccbot.settings.SETTINGS) with a button per setting: booleans toggle,
choices advance to the next option. Changes apply immediately and persist
to settings.json. The dashboard in the topic is pinned; ``/settings`` sends
an unpinned copy wherever it is used.

Key components:
  - render_settings(): text + inline keyboard
  - SettingsTopic: SpecialTopic implementation registered as "settings"
  - settings_command(): the /settings command handler
"""

from __future__ import annotations

import logging

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from .. import settings
from ..config import config
from . import special_topics
from .callback_data import CB_SETTING
from .message_sender import safe_edit, safe_reply

logger = logging.getLogger(__name__)


def render_settings() -> tuple[str, InlineKeyboardMarkup]:
    lines = [
        "⚙️ **Settings** — tap to toggle / cycle. Applied immediately, saved to `settings.json`.",
        "",
    ]
    rows: list[list[InlineKeyboardButton]] = []
    group = None
    for s in settings.SETTINGS:
        if s.group != group:
            group = s.group
            lines.append(f"**{group}**")
        value = settings.display(s)
        help_text = f" — _{s.help}_" if s.help else ""
        lines.append(f"• {s.label}: **{value}**{help_text}")
        arrow = "" if s.kind == "bool" else " ▸"
        rows.append(
            [
                InlineKeyboardButton(
                    f"{s.label}: {value}{arrow}", callback_data=f"{CB_SETTING}{s.key}"
                )
            ]
        )
    lines.append("")
    lines.append(f"_file: `{settings.settings_file()}`_")
    return "\n".join(lines), InlineKeyboardMarkup(rows)


class SettingsTopic:
    name = "settings"
    callback_prefixes: tuple[str, ...] = (CB_SETTING,)

    async def on_ready(self, bot: Bot, chat_id: int, thread_id: int) -> None:
        text, kb = render_settings()
        try:
            msg = await bot.send_message(
                chat_id=chat_id,
                message_thread_id=thread_id,
                text="Loading settings…",
            )
            await safe_edit(msg, text, reply_markup=kb)
            await bot.unpin_all_forum_topic_messages(
                chat_id=chat_id, message_thread_id=thread_id
            )
            await bot.pin_chat_message(
                chat_id=chat_id, message_id=msg.message_id, disable_notification=True
            )
        except TelegramError as e:
            logger.warning("settings dashboard: %s", e)

    async def handle_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str
    ) -> None:
        if update.message is None:
            return
        body, kb = render_settings()
        await safe_reply(update.message, body, reply_markup=kb)

    async def handle_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, data: str
    ) -> None:
        query = update.callback_query
        if query is None:
            return
        key = data[len(CB_SETTING) :]
        try:
            value = settings.cycle(key)
        except (KeyError, ValueError) as e:
            await query.answer(f"Unknown setting: {e}", show_alert=True)
            return
        setting = next(s for s in settings.SETTINGS if s.key == key)
        await query.answer(f"{setting.label}: {settings.display(setting)}")
        logger.info("settings: %s -> %r", key, value)
        body, kb = render_settings()
        await safe_edit(query, body, reply_markup=kb)


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the settings dashboard in the current topic."""
    user = update.effective_user
    if not user or not config.is_user_allowed(user.id) or not update.message:
        return
    body, kb = render_settings()
    await safe_reply(update.message, body, reply_markup=kb)


settings_topic = SettingsTopic()
special_topics.register(settings_topic)
