"""▶ Resume offers for topics whose Claude window is gone.

A topic loses its tmux window when it is killed via the bot, when the
server reboots or tmux dies, or when the window is closed outside the bot.
In every case the session transcript survives, so the topic gets a message
with the session's cwd / id / launch mode and a button that relaunches it
with ``claude --resume`` (the button is handled by the CB_RESUME_SESSION
branch of bot.callback_handler, which reads ``session_manager.killed_sessions``).

Key functions: offer_resume(), offer_lost_bindings() (startup; retried in
the background while Telegram is unreachable, e.g. right after a reboot).
"""

import asyncio
import logging

from aiogram import Bot
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from ..session import LostBinding, session_manager
from ..tmux_manager import LAUNCH_MODE_LABELS
from .callback_data import CB_RESUME_SESSION
from .message_sender import send_with_fallback

logger = logging.getLogger(__name__)

# Delays between retries of startup offers that could not be sent
RETRY_DELAYS = (15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0)
_retry_tasks: set[asyncio.Task[None]] = set()


def resume_keyboard(session_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="▶ Resume this session",
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
) -> bool:
    """Post ``headline`` plus session details and a ▶ Resume button in the topic.

    Also records the session in ``killed_sessions`` so the button relaunches
    it in the same directory and launch mode. Returns whether the message
    reached Telegram; never raises.
    """
    session_manager.remember_killed_session(session_id, cwd, mode, name)
    mode_label = LAUNCH_MODE_LABELS.get(mode, mode)
    try:
        sent = await send_with_fallback(
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
        return False
    return sent is not None


async def _offer_lost(bot: Bot, lost: LostBinding) -> bool:
    return await offer_resume(
        bot,
        lost.user_id,
        lost.thread_id,
        headline=f"⚠️ `{lost.name}` stopped while the bot was down "
        "(server reboot or tmux restart).",
        session_id=lost.session_id,
        cwd=lost.cwd,
        mode=lost.mode,
        name=lost.name,
    )


async def offer_lost_bindings(bot: Bot, lost: list[LostBinding]) -> None:
    """Offer ▶ Resume in topics that lost their window during startup.

    Offers that fail (network not up yet after a reboot) are retried in the
    background with growing delays, so the topics aren't left silently
    unbound.
    """
    failed = [b for b in lost if not await _offer_lost(bot, b)]
    if failed:
        task = asyncio.create_task(_retry_offers(bot, failed))
        _retry_tasks.add(task)
        task.add_done_callback(_retry_tasks.discard)


async def _retry_offers(bot: Bot, pending: list[LostBinding]) -> None:
    for delay in RETRY_DELAYS:
        await asyncio.sleep(delay)
        pending = [b for b in pending if not await _offer_lost(bot, b)]
        if not pending:
            return
    for b in pending:
        logger.error(
            "Could not post resume offer for session %s (thread %d); "
            "/resume %s still works",
            b.session_id,
            b.thread_id,
            b.session_id,
        )
