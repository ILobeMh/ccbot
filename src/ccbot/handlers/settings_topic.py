"""The ``settings`` special topic and ``/settings``: live bot configuration.

A small menu instead of one long list: the home screen shows one button per
section (Output, Input, Sessions, …) with a one-line summary; a section page
lists its settings — on/off switches toggle in place, multiple-choice
settings open a picker with every option and the current one highlighted.
Changes apply immediately and persist to settings.json. The home screen in
the topic is pinned; ``/settings`` sends an unpinned copy wherever it is
used. Navigation edits the message in place.

Key components:
  - render_home() / render_section() / render_picker(): text + keyboard
  - SettingsTopic: SpecialTopic implementation registered as "settings"
  - settings_command(): the /settings command handler
"""

from __future__ import annotations

import logging
from typing import Any

from aiogram import Bot
from aiogram.enums import ButtonStyle
from aiogram.exceptions import TelegramAPIError, TelegramNetworkError
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from .. import settings
from ..config import config
from ..settings import Setting
from . import special_topics
from .callback_data import CB_SETTING, CB_SETTINGS_NAV
from .message_sender import safe_edit, safe_reply

logger = logging.getLogger(__name__)

# Section order, icon and one-line description (Setting.group → …)
SECTIONS: dict[str, tuple[str, str]] = {
    "Output": ("📤", "What Claude's output looks like here"),
    "Input": ("⌨️", "How your messages reach Claude Code"),
    "Sessions": ("🖥", "Launching Claude Code"),
    "Notifications": ("🔔", "What the notifications topic reports"),
    "Topics": ("🧩", "The shell and ccc topics"),
    "Misc": ("🌙", "Screenshots and quiet hours"),
}

_HOME = f"{CB_SETTINGS_NAV}home"
_SEC = f"{CB_SETTINGS_NAV}sec:"
_PICK = f"{CB_SETTINGS_NAV}pick:"
_VAL = f"{CB_SETTINGS_NAV}val:"


def _in(group: str) -> list[Setting]:
    return [s for s in settings.SETTINGS if s.group == group]


def _sections() -> list[str]:
    groups = list(dict.fromkeys(s.group for s in settings.SETTINGS))
    return [g for g in SECTIONS if g in groups] + [
        g for g in groups if g not in SECTIONS
    ]


def _short(s: Setting) -> str:
    if s.kind == "bool":
        return f"{s.label} {'✅' if settings.get(s.key) else '❌'}"
    return f"{s.label}: {s.fmt(settings.get(s.key))}"


def _back(to: str, label: str = "« Back") -> list[InlineKeyboardButton]:
    return [InlineKeyboardButton(text=label, callback_data=to)]


def render_home() -> tuple[str, InlineKeyboardMarkup]:
    lines = ["⚙️ **Settings**", "_Tap a section. Changes apply immediately._", ""]
    buttons: list[InlineKeyboardButton] = []
    for group in _sections():
        icon, _ = SECTIONS.get(group, ("•", ""))
        summary = " · ".join(_short(s) for s in _in(group)[:3])
        lines.append(f"{icon} **{group}** — {summary}")
        buttons.append(
            InlineKeyboardButton(text=f"{icon} {group}", callback_data=f"{_SEC}{group}")
        )
    rows = [buttons[i : i + 2] for i in range(0, len(buttons), 2)]
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


def render_section(group: str) -> tuple[str, InlineKeyboardMarkup]:
    icon, about = SECTIONS.get(group, ("•", ""))
    lines = [f"{icon} **{group}**"]
    if about:
        lines.append(f"_{about}_")
    lines.append("")
    rows: list[list[InlineKeyboardButton]] = []
    for s in _in(group):
        lines.append(f"**{s.label}** — {settings.display(s)}")
        if s.help:
            lines.append(f"_{s.help}_")
        if s.kind == "bool":
            on = bool(settings.get(s.key))
            rows.append(
                [
                    InlineKeyboardButton(
                        text=f"{'✅' if on else '⬜'} {s.label}",
                        callback_data=f"{CB_SETTING}{s.key}",
                        style=ButtonStyle.SUCCESS if on else None,
                    )
                ]
            )
        else:
            rows.append(
                [
                    InlineKeyboardButton(
                        text=f"{s.label}: {settings.display(s)} ›",
                        callback_data=f"{_PICK}{s.key}",
                    )
                ]
            )
    rows.append(_back(_HOME, "« All settings"))
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


def render_picker(key: str) -> tuple[str, InlineKeyboardMarkup]:
    s = settings.setting(key)
    current = settings.get(key)
    lines = [f"**{s.label}**"]
    if s.help:
        lines.append(f"_{s.help}_")
    buttons = []
    for i, choice in enumerate(s.choices):
        selected = settings.same(choice, current)
        buttons.append(
            InlineKeyboardButton(
                text=("● " if selected else "") + s.fmt(choice),
                callback_data=f"{_VAL}{key}:{i}",
                style=ButtonStyle.PRIMARY if selected else None,
            )
        )
    per_row = 2 if max((len(b.text) for b in buttons), default=0) > 14 else 3
    rows = [buttons[i : i + per_row] for i in range(0, len(buttons), per_row)]
    rows.append(_back(f"{_SEC}{s.group}"))
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


def render_settings() -> tuple[str, InlineKeyboardMarkup]:
    """The entry screen (home)."""
    return render_home()


class SettingsTopic:
    name = "settings"
    callback_prefixes: tuple[str, ...] = (CB_SETTING, CB_SETTINGS_NAV)

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
        except (TelegramAPIError, TelegramNetworkError) as e:
            logger.warning("settings dashboard: %s", e)

    async def handle_text(
        self, message: Message, bot: Bot, user_data: dict[str, Any], text: str
    ) -> None:
        body, kb = render_settings()
        await safe_reply(message, body, reply_markup=kb)

    async def handle_callback(
        self, query: CallbackQuery, bot: Bot, user_data: dict[str, Any], data: str
    ) -> None:
        try:
            body, kb, toast = self._navigate(data)
        except (KeyError, ValueError, IndexError) as e:
            await query.answer(f"Unknown setting: {e}", show_alert=True)
            return
        await query.answer(toast)
        await safe_edit(query, body, reply_markup=kb)

    @staticmethod
    def _navigate(data: str) -> tuple[str, InlineKeyboardMarkup, str | None]:
        """Apply a settings callback; return the screen to show and a toast."""
        if data == _HOME:
            return (*render_home(), None)
        if data.startswith(_SEC):
            return (*render_section(data[len(_SEC) :]), None)
        if data.startswith(_PICK):
            return (*render_picker(data[len(_PICK) :]), None)
        if data.startswith(_VAL):
            key, _, idx = data[len(_VAL) :].rpartition(":")
            s = settings.setting(key)
            settings.set_value(key, s.choices[int(idx)])
            logger.info("settings: %s -> %r", key, settings.get(key))
            return (*render_section(s.group), f"{s.label}: {settings.display(s)}")
        # CB_SETTING: toggle a switch (or cycle, from an old pinned dashboard)
        key = data[len(CB_SETTING) :]
        settings.cycle(key)
        s = settings.setting(key)
        logger.info("settings: %s -> %r", key, settings.get(key))
        return (*render_section(s.group), f"{s.label}: {settings.display(s)}")


async def settings_command(message: Message) -> None:
    """Show the settings dashboard in the current topic."""
    user = message.from_user
    if not user or not config.is_user_allowed(user.id):
        return
    body, kb = render_settings()
    await safe_reply(message, body, reply_markup=kb)


settings_topic = SettingsTopic()
special_topics.register(settings_topic)
