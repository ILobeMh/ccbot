"""Interactive UI handling for Claude Code prompts.

Handles interactive terminal UIs displayed by Claude Code:
  - AskUserQuestion: Multi-choice question prompts
  - ExitPlanMode: Plan mode exit confirmation
  - Permission Prompt: Tool permission requests
  - RestoreCheckpoint: Checkpoint restoration selection

Provides:
  - Keyboard navigation (up/down/left/right/enter/esc)
  - Terminal capture and display
  - Interactive mode tracking per user and thread

State dicts are keyed by (user_id, thread_id_or_0) for Telegram topic support.
"""

import asyncio
import logging
import re
from typing import Any

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message

from ..config import config
from ..session import session_manager
from ..terminal_parser import extract_interactive_content, is_interactive_ui
from ..tmux_manager import tmux_manager
from ..ui_choices import (
    DIGIT_UIS,
    Choice,
    ChoiceView,
    button_style,
    button_text,
    cursor_number,
    escape_literal,
    has_hidden_options,
    label_hash,
    parse_choices,
    render_view,
)
from .callback_data import (
    CB_ASK_DOWN,
    CB_ASK_ENTER,
    CB_ASK_ESC,
    CB_ASK_LEFT,
    CB_ASK_REFRESH,
    CB_ASK_RIGHT,
    CB_ASK_SPACE,
    CB_ASK_TAB,
    CB_ASK_UP,
    CB_CHOICE,
    CB_KEYPAD,
)
from .message_sender import (
    NO_LINK_PREVIEW,
    edit_rich,
    edit_with_fallback,
    send_rich,
    send_with_fallback,
)
from .notifications_topic import mark_ui

logger = logging.getLogger(__name__)

# Tool names that trigger interactive UI via JSONL (terminal capture + inline keyboard)
INTERACTIVE_TOOL_NAMES = frozenset({"AskUserQuestion", "ExitPlanMode"})

# Track interactive UI message IDs: (user_id, thread_id_or_0) -> message_id
_interactive_msgs: dict[tuple[int, int], int] = {}

# Track interactive mode: (user_id, thread_id_or_0) -> window_id
_interactive_mode: dict[tuple[int, int], str] = {}

# Topics showing the raw keypad instead of option buttons
_keypad_mode: set[tuple[int, int]] = set()
# Last parsed view per topic (for the answered-summary)
_last_views: dict[tuple[int, int], ChoiceView | None] = {}
# window_id -> AskUserQuestion "questions" input (titles for tabs)
_questions: dict[str, list[dict[str, Any]]] = {}
# Topics where a tapped option is being typed right now
_answering: set[tuple[int, int]] = set()


async def _capture_ui(window_id: str) -> str | None:
    """Pane text; for a picker that scrolls ("… +10 models") the window is
    briefly grown so more of its options are drawn."""
    pane = await tmux_manager.capture_pane(window_id)
    if pane and has_hidden_options(pane) and is_interactive_ui(pane):
        return await tmux_manager.capture_pane_expanded(window_id, before=pane) or pane
    return pane


def get_interactive_window(user_id: int, thread_id: int | None = None) -> str | None:
    """Get the window_id for user's interactive mode."""
    return _interactive_mode.get((user_id, thread_id or 0))


def set_interactive_mode(
    user_id: int,
    window_id: str,
    thread_id: int | None = None,
) -> None:
    """Set interactive mode for a user."""
    logger.debug(
        "Set interactive mode: user=%d, window_id=%s, thread=%s",
        user_id,
        window_id,
        thread_id,
    )
    _interactive_mode[(user_id, thread_id or 0)] = window_id


def clear_interactive_mode(user_id: int, thread_id: int | None = None) -> None:
    """Clear interactive mode for a user (without deleting message)."""
    logger.debug("Clear interactive mode: user=%d, thread=%s", user_id, thread_id)
    _interactive_mode.pop((user_id, thread_id or 0), None)


def get_interactive_msg_id(user_id: int, thread_id: int | None = None) -> int | None:
    """Get the interactive message ID for a user."""
    return _interactive_msgs.get((user_id, thread_id or 0))


def _build_interactive_keyboard(
    window_id: str,
    ui_name: str = "",
    back_to_choices: bool = False,
) -> InlineKeyboardMarkup:
    """Build keyboard for interactive UI navigation.

    ``ui_name`` controls the layout: ``RestoreCheckpoint`` omits ←/→ keys
    since only vertical selection is needed.
    """
    vertical_only = ui_name == "RestoreCheckpoint"

    rows: list[list[InlineKeyboardButton]] = []
    # Row 1: directional keys
    rows.append(
        [
            InlineKeyboardButton(
                text="␣ Space", callback_data=f"{CB_ASK_SPACE}{window_id}"[:64]
            ),
            InlineKeyboardButton(
                text="↑", callback_data=f"{CB_ASK_UP}{window_id}"[:64]
            ),
            InlineKeyboardButton(
                text="⇥ Tab", callback_data=f"{CB_ASK_TAB}{window_id}"[:64]
            ),
        ]
    )
    if vertical_only:
        rows.append(
            [
                InlineKeyboardButton(
                    text="↓", callback_data=f"{CB_ASK_DOWN}{window_id}"[:64]
                ),
            ]
        )
    else:
        rows.append(
            [
                InlineKeyboardButton(
                    text="←", callback_data=f"{CB_ASK_LEFT}{window_id}"[:64]
                ),
                InlineKeyboardButton(
                    text="↓", callback_data=f"{CB_ASK_DOWN}{window_id}"[:64]
                ),
                InlineKeyboardButton(
                    text="→", callback_data=f"{CB_ASK_RIGHT}{window_id}"[:64]
                ),
            ]
        )
    # Row 2: action keys
    rows.append(
        [
            InlineKeyboardButton(
                text="⎋ Esc", callback_data=f"{CB_ASK_ESC}{window_id}"[:64]
            ),
            InlineKeyboardButton(
                text="🔄", callback_data=f"{CB_ASK_REFRESH}{window_id}"[:64]
            ),
            InlineKeyboardButton(
                text="⏎ Enter", callback_data=f"{CB_ASK_ENTER}{window_id}"[:64]
            ),
        ]
    )
    if back_to_choices:
        rows.append(
            [
                InlineKeyboardButton(
                    text="« Choices", callback_data=f"{CB_KEYPAD}{window_id}"[:64]
                )
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def handle_interactive_ui(
    bot: Bot,
    user_id: int,
    window_id: str,
    thread_id: int | None = None,
    *,
    force_new: bool = False,
) -> bool:
    """Capture terminal and send interactive UI content to user.

    Handles AskUserQuestion, ExitPlanMode, Permission Prompt, and
    RestoreCheckpoint UIs. Returns True if UI was detected and sent,
    False otherwise. An existing UI message is edited in place, unless
    ``force_new`` asks to re-post it as the newest message in the topic
    (the old one is deleted once the new one is sent).
    """
    ikey = (user_id, thread_id or 0)
    chat_id = session_manager.resolve_chat_id(user_id, thread_id)
    w = await tmux_manager.find_window_by_id(window_id)
    if not w:
        return False

    # Capture plain text (no ANSI colors)
    pane_text = await _capture_ui(w.window_id)
    if not pane_text:
        logger.debug("No pane text captured for window_id %s", window_id)
        return False

    # Quick check if it looks like an interactive UI
    if not is_interactive_ui(pane_text):
        logger.debug(
            "No interactive UI detected in window_id %s (last 3 lines: %s)",
            window_id,
            pane_text.strip().split("\n")[-3:],
        )
        return False

    # Extract content between separators
    content = extract_interactive_content(pane_text)
    if not content:
        return False

    view = parse_choices(content, pane_text, _questions.get(window_id))
    use_choices = (
        view is not None and content.name in DIGIT_UIS and ikey not in _keypad_mode
    )
    if use_choices and view is not None:
        text = render_view(view)
        keyboard = _choice_keyboard(window_id, view)
    else:
        # Raw terminal text + arrow keypad (unknown UI, or the user asked
        # for the keys); plain text: terminal content is not markdown
        text = content.content
        keyboard = _build_interactive_keyboard(
            window_id, ui_name=content.name, back_to_choices=view is not None
        )

    existing_msg_id = _interactive_msgs.get(ikey)
    if existing_msg_id and not force_new:
        if await _edit(bot, chat_id, existing_msg_id, text, keyboard, use_choices):
            _interactive_mode[ikey] = window_id
            _last_views[ikey] = view
            await mark_ui(window_id, content.name, content.content, existing_msg_id)
            return True
        logger.debug("Edit failed for interactive msg %s, sending new", existing_msg_id)

    logger.info(
        "Sending interactive UI to user %d for window_id %s", user_id, window_id
    )
    sent = await _send(bot, chat_id, thread_id, text, keyboard, use_choices)
    if sent is None:
        return False
    _interactive_msgs[ikey] = sent.message_id
    _interactive_mode[ikey] = window_id
    _last_views[ikey] = view
    await mark_ui(window_id, content.name, content.content, sent.message_id)
    # New message sent successfully — now safe to delete the old one
    if existing_msg_id:
        try:
            await bot.delete_message(chat_id=chat_id, message_id=existing_msg_id)
        except Exception:
            pass  # Old message may already be gone
    return True


async def _send(
    bot: Bot,
    chat_id: int,
    thread_id: int | None,
    text: str,
    keyboard: InlineKeyboardMarkup,
    markdown: bool,
) -> Message | None:
    kwargs: dict[str, Any] = {"reply_markup": keyboard}
    if thread_id is not None:
        kwargs["message_thread_id"] = thread_id
    if not markdown:
        try:
            return await bot.send_message(
                chat_id=chat_id,
                text=text,
                link_preview_options=NO_LINK_PREVIEW,
                **kwargs,
            )
        except Exception as e:
            logger.error("Failed to send interactive UI: %s", e)
            return None
    if config.message_format == "rich":
        return await send_rich(bot, chat_id, text, **kwargs)
    return await send_with_fallback(bot, chat_id, text, **kwargs)


async def _edit(
    bot: Bot,
    chat_id: int,
    message_id: int,
    text: str,
    keyboard: InlineKeyboardMarkup | None,
    markdown: bool,
) -> bool:
    if not markdown:
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                reply_markup=keyboard,
                link_preview_options=NO_LINK_PREVIEW,
            )
            return True
        except TelegramBadRequest as e:
            return "message is not modified" in str(e).lower()
        except Exception as e:
            logger.debug("Edit failed for interactive msg %s: %s", message_id, e)
            return False
    if config.message_format == "rich":
        return await edit_rich(bot, chat_id, message_id, text, reply_markup=keyboard)
    return await edit_with_fallback(
        bot, chat_id, message_id, text, reply_markup=keyboard
    )


def _choice_keyboard(window_id: str, view: ChoiceView) -> InlineKeyboardMarkup:
    """One button per option (tap = answer), plus navigation and escape."""

    def cb(c: Choice) -> str:
        # option label hash + prompt fingerprint: a tap only acts on the very
        # prompt it was shown for (see answer_choice)
        return f"{CB_CHOICE}{window_id}:{c.number}:{choice_token(view, c)}"[:64]

    options = view.choices
    main = [
        InlineKeyboardButton(
            text=button_text(c), callback_data=cb(c), style=button_style(c)
        )
        for c in options
        if c.kind == "option"
    ]
    per_row = 1 if any(len(b.text) > 16 for b in main) else 2
    rows = [main[i : i + per_row] for i in range(0, len(main), per_row)]
    extra = [
        InlineKeyboardButton(text=button_text(c), callback_data=cb(c))
        for c in options
        if c.kind != "option"
    ]
    if extra:
        rows.append(extra)
    if view.tabs or view.multi:
        tab_nav = [
            InlineKeyboardButton(
                text="‹ Prev", callback_data=f"{CB_ASK_LEFT}{window_id}"[:64]
            )
        ]
        if not view.is_submit_tab:  # nothing after the Submit tab
            tab_nav.append(
                InlineKeyboardButton(
                    text="Next ›", callback_data=f"{CB_ASK_RIGHT}{window_id}"[:64]
                )
            )
        rows.append(tab_nav)
    rows.append(
        [
            InlineKeyboardButton(
                text="⌨️ Keys", callback_data=f"{CB_KEYPAD}{window_id}"[:64]
            ),
            InlineKeyboardButton(
                text="🔄", callback_data=f"{CB_ASK_REFRESH}{window_id}"[:64]
            ),
            InlineKeyboardButton(
                text="⎋ Esc",
                callback_data=f"{CB_ASK_ESC}{window_id}"[:64],
                style="danger",
            ),
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def choice_token(view: ChoiceView, choice: Choice) -> str:
    """8 hex chars identifying ``choice`` within ``view``'s prompt."""
    return label_hash(choice.label) + view.fingerprint


def remember_questions(window_id: str, tool_input: dict[str, Any] | None) -> None:
    """Keep the AskUserQuestion input so tabs can be titled by their header."""
    questions = (tool_input or {}).get("questions")
    if isinstance(questions, list):
        _questions[window_id] = [q for q in questions if isinstance(q, dict)]


def toggle_keypad(user_id: int, thread_id: int | None) -> bool:
    """Switch between option buttons and the raw keypad; True = keypad now."""
    ikey = (user_id, thread_id or 0)
    if ikey in _keypad_mode:
        _keypad_mode.discard(ikey)
        return False
    _keypad_mode.add(ikey)
    return True


# UIs where a digit may only move the cursor (pickers): confirm with Enter
_CONFIRM_AFTER_DIGIT = {"Settings", "Modal", "SwitchModel", "RestoreCheckpoint"}
_CURSOR_ON_RE = r"^\s*❯\s*{n}\."


async def answer_choice(
    bot: Bot,
    user_id: int,
    thread_id: int | None,
    window_id: str,
    number: int,
    token: str,
) -> str:
    """Type option ``number`` into the UI, after checking it's still the same.

    ``token`` (choice_token) must match both the option and the prompt on
    screen now; otherwise nothing is typed and the message is refreshed.
    Returns the toast to show on the tapped button.
    """
    ikey = (user_id, thread_id or 0)
    _answering.add(ikey)
    try:
        return await _answer_choice(bot, user_id, thread_id, window_id, number, token)
    finally:
        _answering.discard(ikey)


def is_answering(user_id: int, thread_id: int | None) -> bool:
    """True while a tap is being typed (the poller must not clear the UI)."""
    return (user_id, thread_id or 0) in _answering


async def _answer_choice(
    bot: Bot,
    user_id: int,
    thread_id: int | None,
    window_id: str,
    number: int,
    token: str,
) -> str:
    pane = await _capture_ui(window_id)
    ui = extract_interactive_content(pane) if pane else None
    view = parse_choices(ui, pane, _questions.get(window_id)) if ui else None
    choice = (
        next((c for c in view.choices if c.number == number), None) if view else None
    )
    if view is None or choice is None or choice_token(view, choice) != token:
        if not await handle_interactive_ui(bot, user_id, window_id, thread_id):
            await clear_interactive_msg(user_id, bot, thread_id)
        return "That question has changed — showing the current one"

    if number > 9:
        # No single key: a typed "10" would pick option 1 first. Walk the
        # cursor there instead and confirm
        if not await _walk_cursor(window_id, number, pane or ""):
            return "Couldn't reach that option — use ⌨️ Keys"
        await tmux_manager.send_keys(window_id, "Enter", enter=False, literal=False)
        await asyncio.sleep(0.4)
    else:
        await tmux_manager.send_keys(window_id, str(number), enter=False, literal=True)
        await asyncio.sleep(0.5)
    if (
        number <= 9
        and view.ui_name in _CONFIRM_AFTER_DIGIT
        and not view.multi
        and choice.kind == "option"
    ):
        pane2 = await tmux_manager.capture_pane(window_id) or ""
        ui2 = extract_interactive_content(pane2)
        view2 = parse_choices(ui2, pane2) if ui2 else None
        if (
            view2 is not None
            and view2.signature == view.signature
            and re.search(_CURSOR_ON_RE.format(n=number), pane2, re.MULTILINE)
        ):
            await tmux_manager.send_keys(window_id, "Enter", enter=False, literal=False)
            await asyncio.sleep(0.4)

    if choice.kind == "other":
        await handle_interactive_ui(bot, user_id, window_id, thread_id)
        return "✍️ Now type your answer as a message"
    if not await handle_interactive_ui(bot, user_id, window_id, thread_id):
        # A half-drawn capture can look like "no UI": look once more before
        # recording the prompt as answered
        await asyncio.sleep(0.4)
        if not await handle_interactive_ui(bot, user_id, window_id, thread_id):
            await _finalize(bot, user_id, thread_id, view, choice)
    return f"✓ {choice.label[:40]}"


async def _walk_cursor(window_id: str, number: int, pane: str) -> bool:
    """Move the picker's ``❯`` cursor onto option ``number`` with Up / Down."""
    start = cursor_number(pane)
    if start is None:
        return False
    key = "Down" if number > start else "Up"
    for _ in range(abs(number - start)):
        await tmux_manager.send_keys(window_id, key, enter=False, literal=False)
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.3)
    now = await _capture_ui(window_id)
    return now is not None and cursor_number(now) == number


async def _finalize(
    bot: Bot,
    user_id: int,
    thread_id: int | None,
    view: ChoiceView,
    choice: Choice,
) -> None:
    """The UI is gone (answered): turn its message into a one-line record."""
    ikey = (user_id, thread_id or 0)
    msg_id = _interactive_msgs.pop(ikey, None)
    _interactive_mode.pop(ikey, None)
    _last_views.pop(ikey, None)
    _keypad_mode.discard(ikey)
    if msg_id is None:
        return
    title = view.title or "Question"
    if view.review:
        answer = "; ".join(f"{q} → {a}" for q, a in view.review)
        summary = (
            f"✅ {title}: {answer}"
            if choice.label.startswith("Submit")
            else "✖ Cancelled"
        )
    else:
        icon = "❌" if button_style(choice) == "danger" else "✅"
        summary = f"{icon} {title} → {choice.label}"
    chat_id = session_manager.resolve_chat_id(user_id, thread_id)
    await _edit(bot, chat_id, msg_id, escape_literal(summary), None, True)


async def clear_interactive_msg(
    user_id: int,
    bot: Bot | None = None,
    thread_id: int | None = None,
) -> None:
    """Clear tracked interactive message, delete from chat, and exit interactive mode."""
    ikey = (user_id, thread_id or 0)
    msg_id = _interactive_msgs.pop(ikey, None)
    _interactive_mode.pop(ikey, None)
    _last_views.pop(ikey, None)
    _keypad_mode.discard(ikey)
    logger.debug(
        "Clear interactive msg: user=%d, thread=%s, msg_id=%s",
        user_id,
        thread_id,
        msg_id,
    )
    if bot and msg_id:
        chat_id = session_manager.resolve_chat_id(user_id, thread_id)
        try:
            await bot.delete_message(chat_id=chat_id, message_id=msg_id)
        except Exception:
            pass  # Message may already be deleted or too old
