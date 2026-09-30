"""▶ Resume offers for topics whose Claude window is gone.

A topic loses its tmux window when it is killed via the bot, when the
server reboots or tmux dies, or when the window is closed outside the bot.
In every case the session transcript survives, so the topic gets a message
with the session's cwd / id / launch mode and a button that relaunches it
with ``claude --resume`` (the button is handled by the CB_RESUME_SESSION
branch of bot.callback_handler, which reads ``session_manager.killed_sessions``).

Key function: offer_resume().
"""

import logging

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

from ..session import session_manager
from ..tmux_manager import LAUNCH_MODE_LABELS
from .callback_data import CB_RESUME_SESSION
from .message_sender import safe_send

logger = logging.getLogger(__name__)


def resume_keyboard(session_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "▶ Resume this session",
                    callback_data=f"{CB_RESUME_SESSION}{session_id}"[:64],
                )
            ]
        ]
    )


async def offer_resume(
    bot: Bot,
    user_id: int,
    thread_id: int,
    *,
    headline: str,
    session_id: str,
    cwd: str,
    mode: str,
    name: str,
) -> None:
    """Post ``headline`` plus session details and a ▶ Resume button in the topic.

    Also records the session in ``killed_sessions`` so the button relaunches
    it in the same directory and launch mode.
    """
    session_manager.remember_killed_session(session_id, cwd, mode, name)
    mode_label = LAUNCH_MODE_LABELS.get(mode, mode)
    try:
        await safe_send(
            bot,
            session_manager.resolve_chat_id(user_id, thread_id),
            f"{headline}\n"
            f"cwd `{cwd}`\nsession `{session_id}`\nmode {mode_label}\n\n"
            f"Tap ▶ or send `/resume {session_id}`.",
            message_thread_id=thread_id,
            reply_markup=resume_keyboard(session_id),
        )
    except Exception as e:  # never let a notice break the caller's cleanup
        logger.warning("Resume offer failed (thread %d): %s", thread_id, e)
