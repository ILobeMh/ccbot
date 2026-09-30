"""Terminal status line polling for thread-bound windows.

Provides background polling of terminal status lines for all active users:
  - Detects Claude Code status (working, waiting, etc.)
  - Detects interactive UIs (permission prompts) not triggered via JSONL
  - Answers startup dialogs (trust folder, …) on the bot's behalf
  - Notifies once when Claude Code exited (shell prompt) or an update is
    installed, with Restart / Kill buttons
  - Updates status messages in Telegram
  - Polls thread_bindings (each topic = one window); a topic whose window
    vanished is unbound and offered ▶ Resume (resume_offer.offer_resume)
  - Periodically probes topic existence via unpin_all_forum_topic_messages
    (silent no-op when no pins); cleans up deleted topics (kills tmux window
    + unbinds thread)

Key components:
  - STATUS_POLL_INTERVAL: Polling frequency (1 second)
  - TOPIC_CHECK_INTERVAL: Topic existence probe frequency (60 seconds)
  - status_poll_loop: Background polling task
  - update_status_message: Poll and enqueue status updates
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest

from ..config import config
from ..session import session_manager
from ..settings import in_quiet_hours
from ..terminal_parser import (
    AUTO_ANSWER_DIALOGS,
    extract_interactive_content,
    has_update_pending,
    is_interactive_ui,
    parse_status_line,
)
from ..tmux_manager import SHELL_COMMANDS, tmux_manager
from .callback_data import CB_KILL, CB_RESTART
from .cleanup import clear_topic_state
from .interactive_ui import (
    clear_interactive_msg,
    get_interactive_window,
)
from .message_queue import (
    enqueue_interactive,
    enqueue_status_update,
    get_message_queue,
    has_pending_interactive,
    interactive_backoff_active,
)
from .message_sender import safe_send
from .notifications_topic import mark_ui, notify
from .resume_offer import offer_resume

logger = logging.getLogger(__name__)

# Status polling interval
STATUS_POLL_INTERVAL = 1.0  # seconds - faster response (rate limiting at send layer)

# Topic existence probe interval
TOPIC_CHECK_INTERVAL = 60.0  # seconds

# Consecutive polls a bound window must be missing before the topic is
# unbound (a single failed/odd tmux listing must not cost a binding)
VANISHED_AFTER_POLLS = 3
_missing_polls: dict[str, int] = {}


# Runs one transcript-monitor cycle on demand (SessionMonitor.poll_now),
# registered by the bot at startup.
_transcript_poller: Callable[[], Awaitable[None]] | None = None


def set_transcript_poller(fn: Callable[[], Awaitable[None]] | None) -> None:
    """Register the monitor's on-demand poll (see update_status_message)."""
    global _transcript_poller
    _transcript_poller = fn


# Windows the bot is currently (re)starting Claude in — health checks are
# suppressed for them so the transient shell prompt isn't reported.
_launching: set[str] = set()
# Consecutive polls that saw a shell prompt, per window
_shell_polls: dict[str, int] = {}
# Windows already notified about "Claude exited" / "update pending"
_exit_notified: set[str] = set()
_update_notified: set[str] = set()
SHELL_POLLS_BEFORE_NOTIFY = 5


def mark_launching(window_id: str, launching: bool) -> None:
    """Suppress health notifications while the bot restarts Claude itself."""
    if launching:
        _launching.add(window_id)
    else:
        _launching.discard(window_id)
        _shell_polls.pop(window_id, None)
        _exit_notified.discard(window_id)
        _update_notified.discard(window_id)


def _health_keyboard(window_id: str, with_kill: bool) -> InlineKeyboardMarkup:
    row = [
        InlineKeyboardButton(
            "🔄 Restart", callback_data=f"{CB_RESTART}{window_id}"[:64]
        )
    ]
    if with_kill:
        row.append(
            InlineKeyboardButton("🗑 Kill", callback_data=f"{CB_KILL}{window_id}"[:64])
        )
    return InlineKeyboardMarkup([row])


async def _check_window_health(
    bot: Bot,
    user_id: int,
    window_id: str,
    thread_id: int | None,
    pane_cmd: str,
    pane_text: str,
) -> None:
    """Notify (once) when Claude exited or an update wants a restart."""
    if window_id in _launching or in_quiet_hours():
        return
    chat_id = session_manager.resolve_chat_id(user_id, thread_id)
    display = session_manager.get_display_name(window_id)

    if pane_cmd in SHELL_COMMANDS:
        n = _shell_polls.get(window_id, 0) + 1
        _shell_polls[window_id] = n
        if n >= SHELL_POLLS_BEFORE_NOTIFY and window_id not in _exit_notified:
            _exit_notified.add(window_id)
            await notify("lifecycle", "Claude Code exited (shell prompt)", window_id)
            await safe_send(
                bot,
                chat_id,
                f"⚠️ Claude Code is not running in '{display}' (shell prompt). "
                "Restart it or kill the window.",
                message_thread_id=thread_id,
                reply_markup=_health_keyboard(window_id, with_kill=True),
            )
        return

    if _shell_polls.pop(window_id, None):
        _exit_notified.discard(window_id)

    if has_update_pending(pane_text):
        if window_id not in _update_notified:
            _update_notified.add(window_id)
            await notify(
                "lifecycle",
                "Claude Code update installed — restart to apply",
                window_id,
            )
            await safe_send(
                bot,
                chat_id,
                f"✔ Claude Code update installed in '{display}'. "
                "Restart to apply it (the session is resumed).",
                message_thread_id=thread_id,
                reply_markup=_health_keyboard(window_id, with_kill=False),
            )
    else:
        _update_notified.discard(window_id)


async def update_status_message(
    bot: Bot,
    user_id: int,
    window_id: str,
    thread_id: int | None = None,
    skip_status: bool = False,
) -> None:
    """Poll terminal and check for interactive UIs and status updates.

    UI detection always happens regardless of skip_status. When skip_status=True,
    only UI detection runs (used when message queue is non-empty to avoid
    flooding the queue with status updates).

    Also detects permission prompt UIs (not triggered via JSONL) and enters
    interactive mode when found.
    """
    w = await tmux_manager.find_window_by_id(window_id)
    if not w:
        # Window gone, enqueue clear (unless skipping status)
        if not skip_status:
            await enqueue_status_update(
                bot, user_id, window_id, None, thread_id=thread_id
            )
        return

    pane_text = await tmux_manager.capture_pane(w.window_id)
    if not pane_text:
        # Transient capture failure - keep existing status message
        return

    await _check_window_health(
        bot, user_id, window_id, thread_id, w.pane_current_command, pane_text
    )
    if w.pane_current_command in SHELL_COMMANDS:
        return  # nothing to parse in a shell

    # Notifications: a dialog that disappeared may be announced again later
    ui_now = extract_interactive_content(pane_text)
    if ui_now is None:
        await mark_ui(window_id, None)  # dialog gone → next one is announced again

    interactive_window = get_interactive_window(user_id, thread_id)
    should_check_new_ui = True

    if interactive_window == window_id:
        # User is in interactive mode for THIS window
        if is_interactive_ui(pane_text):
            # Interactive UI still showing — skip status update (user is interacting)
            return
        if has_pending_interactive(user_id, thread_id):
            # Queued from the transcript; the worker waits for it to render
            return
        # Interactive UI gone — clear interactive mode, fall through to status check.
        # Don't re-check for new UI this cycle (the old one just disappeared).
        await clear_interactive_msg(user_id, bot, thread_id)
        should_check_new_ui = False
    elif interactive_window is not None:
        # User is in interactive mode for a DIFFERENT window (window switched)
        # Clear stale interactive mode
        await clear_interactive_msg(user_id, bot, thread_id)

    # Startup dialogs (trust folder, bypass warning, …) are answered by the
    # bot itself rather than shown as a keyboard.
    if should_check_new_ui:
        ui = extract_interactive_content(pane_text)
        if ui is not None and ui.name in AUTO_ANSWER_DIALOGS:
            await tmux_manager.auto_answer_dialog(window_id, pane_text)
            return

    # Check for permission prompt (interactive UI not triggered via JSONL)
    # ALWAYS check UI, regardless of skip_status
    if should_check_new_ui and is_interactive_ui(pane_text):
        logger.debug(
            "Interactive UI detected in polling (user=%d, window=%s, thread=%s)",
            user_id,
            window_id,
            thread_id,
        )
        # The terminal often shows the UI before the transcript lines that
        # led to it (thinking, text) were read: read them now so they are
        # queued first, then queue the UI behind them.
        if _transcript_poller is not None:
            await _transcript_poller()
        if get_interactive_window(
            user_id, thread_id
        ) != window_id and not interactive_backoff_active(user_id, thread_id):
            await enqueue_interactive(bot, user_id, window_id, thread_id)
        return

    # Normal status line check — skip if queue is non-empty
    if skip_status:
        return

    if not config.status_updates:
        return

    status_line = parse_status_line(pane_text)

    if status_line:
        await enqueue_status_update(
            bot,
            user_id,
            window_id,
            status_line,
            thread_id=thread_id,
        )
    # If no status line, keep existing status message (don't clear on transient state)


async def handle_vanished_window(
    bot: Bot, user_id: int, thread_id: int, window_id: str
) -> bool:
    """Unbind a topic whose window disappeared and offer ▶ Resume in it.

    Returns True when a resume offer was posted.
    """
    if session_manager.get_window_for_thread(user_id, thread_id) != window_id:
        return False  # unbound meanwhile (e.g. /kill) — whoever did it reports it
    ws = session_manager.window_states.get(window_id)
    sid, cwd = (ws.session_id, ws.cwd) if ws else ("", "")
    mode = session_manager.get_launch_info(window_id).get("mode") or "default"
    display = session_manager.get_display_name(window_id)
    session_manager.unbind_thread(user_id, thread_id)
    await session_manager.remove_session_map_entry(window_id)
    await clear_topic_state(user_id, thread_id, bot)
    logger.info(
        "Cleaned up stale binding: user=%d thread=%d window_id=%s",
        user_id,
        thread_id,
        window_id,
    )
    if not (sid and cwd):
        return False
    return await offer_resume(
        bot,
        user_id,
        thread_id,
        headline=f"⚠️ The tmux window of `{display}` is gone.",
        session_id=sid,
        cwd=cwd,
        mode=mode,
        name=display,
    )


async def _poll_binding(bot: Bot, user_id: int, thread_id: int, wid: str) -> None:
    """One status-poll tick for one topic (never raises)."""
    try:
        # Clean up stale bindings (window no longer exists) — only after it
        # was missing from several listings in a row, never on one miss
        w = await tmux_manager.find_window_by_id(wid)
        if not w:
            misses = _missing_polls.get(wid, 0) + 1
            _missing_polls[wid] = misses
            if misses >= VANISHED_AFTER_POLLS:
                _missing_polls.pop(wid, None)
                await handle_vanished_window(bot, user_id, thread_id, wid)
            return
        _missing_polls.pop(wid, None)

        # UI detection happens unconditionally in update_status_message.
        # Status enqueue is skipped inside update_status_message when
        # interactive UI is detected (returns early) or when queue is non-empty.
        queue = get_message_queue(user_id)
        skip_status = queue is not None and not queue.empty()

        await update_status_message(
            bot,
            user_id,
            wid,
            thread_id=thread_id,
            skip_status=skip_status,
        )
    except Exception as e:
        logger.debug(f"Status update error for user {user_id} thread {thread_id}: {e}")


async def status_poll_loop(bot: Bot) -> None:
    """Background task to poll terminal status for all thread-bound windows."""
    logger.info("Status polling started (interval: %ss)", STATUS_POLL_INTERVAL)
    last_topic_check = 0.0
    while True:
        try:
            # Periodic topic existence probe
            now = time.monotonic()
            if now - last_topic_check >= TOPIC_CHECK_INTERVAL:
                last_topic_check = now
                for user_id, thread_id, wid in list(
                    session_manager.iter_thread_bindings()
                ):
                    try:
                        await bot.unpin_all_forum_topic_messages(
                            chat_id=session_manager.resolve_chat_id(user_id, thread_id),
                            message_thread_id=thread_id,
                        )
                    except BadRequest as e:
                        if "Topic_id_invalid" in str(e):
                            # Topic deleted — kill window, unbind, and clean up state
                            w = await tmux_manager.find_window_by_id(wid)
                            if w:
                                await tmux_manager.kill_window(w.window_id)
                            session_manager.unbind_thread(user_id, thread_id)
                            await clear_topic_state(user_id, thread_id, bot)
                            logger.info(
                                "Topic deleted: killed window_id '%s' and "
                                "unbound thread %d for user %d",
                                wid,
                                thread_id,
                                user_id,
                            )
                        else:
                            logger.debug(
                                "Topic probe error for %s: %s",
                                wid,
                                e,
                            )
                    except Exception as e:
                        logger.debug(
                            "Topic probe error for %s: %s",
                            wid,
                            e,
                        )

            # Topics are independent (one window each): poll them together so
            # one slow capture doesn't delay every other topic.
            await asyncio.gather(
                *(
                    _poll_binding(bot, user_id, thread_id, wid)
                    for user_id, thread_id, wid in list(
                        session_manager.iter_thread_bindings()
                    )
                )
            )
        except Exception as e:
            logger.error(f"Status poll loop error: {e}")

        await asyncio.sleep(STATUS_POLL_INTERVAL)
