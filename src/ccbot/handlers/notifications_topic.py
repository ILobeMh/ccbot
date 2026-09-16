"""The ``notifications`` special topic: one feed of events from every session.

Anything that needs the user's attention or is worth knowing about lands
here, tagged with the session's topic and a button that jumps to it:
  - needs input: AskUserQuestion, permission / bash approval, plan approval,
    unknown dialogs (deduplicated while the same dialog stays on screen)
  - turn finished (after at least ``notify_turn_min`` seconds of work),
    with the last reply's first line
  - lifecycle: session created/resumed, restarted, Claude exited, update
    installed
  - API errors from the transcript (rate limits, session limits, 5xx)
  - ccc quota alerts (mirrored from the ccc topic when enabled)
Each kind has its own on/off in Settings; quiet hours apply to all.

Key components:
  - notify(kind, …): the single entry point used by the rest of the bot
  - NotificationsTopic: SpecialTopic implementation registered as "notifications"
  - mark_working()/mark_ui(): state used to detect turn-done and dedupe UIs
"""

from __future__ import annotations

import asyncio
import logging
import time

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from ..config import config
from ..session import session_manager
from ..settings import in_quiet_hours
from . import special_topics
from .message_sender import safe_reply, safe_send

logger = logging.getLogger(__name__)

# kind -> (config attribute that enables it, icon)
KINDS: dict[str, tuple[str, str]] = {
    "input": ("notify_needs_input", "❓"),
    "done": ("notify_turn_done", "✅"),
    "lifecycle": ("notify_lifecycle", "🔄"),
    "error": ("notify_errors", "🚨"),
    "ccc": ("notify_ccc", "🔀"),
}


class _State:
    bot: Bot | None = None
    chat_id: int | None = None
    thread_id: int | None = None
    # window_id -> monotonic time the current turn started (None = idle)
    working_since: dict[str, float] = {}
    # window_id -> last assistant text (first line) for "turn done"
    last_text: dict[str, str] = {}
    # window_id -> Telegram message id of the last assistant text message
    last_text_msg: dict[str, int] = {}
    # window_id -> signature of the UI already announced, and how many
    # dialogs have come and gone (so a re-shown dialog isn't deduped away)
    announced_ui: dict[str, str] = {}
    ui_epoch: dict[str, int] = {}
    # (kind, window_id, signature) -> monotonic time, for generic dedupe
    recent: dict[tuple[str, str, str], float] = {}
    # delayed announcements in flight (kept referenced until done)
    pending: set[asyncio.Task[None]] = set()


_s = _State()
DEDUPE_SECONDS = 60.0
TURN_DONE_DELAY = 4.0


def _topic_link(
    chat_id: int, thread_id: int | None, message_id: int | None = None
) -> str | None:
    """Deep link to a forum topic, or to one message inside it.

    Supergroup ids are -100<internal>; t.me/c/<internal>/<thread>/<message>
    opens the topic scrolled to that message.
    """
    raw = str(chat_id)
    if not raw.startswith("-100") or thread_id is None:
        return None
    link = f"https://t.me/c/{raw[4:]}/{thread_id}"
    return f"{link}/{message_id}" if message_id else link


def _enabled(kind: str) -> bool:
    attr, _ = KINDS[kind]
    return bool(getattr(config, attr, True))


def _thread_for_window(window_id: str) -> int | None:
    for _user_id, thread_id, wid in session_manager.iter_thread_bindings():
        if wid == window_id:
            return thread_id
    return None


async def notify(
    kind: str,
    text: str,
    window_id: str | None = None,
    *,
    signature: str | None = None,
    buttons: list[InlineKeyboardButton] | None = None,
    message_id: int | None = None,
) -> bool:
    """Post an event to the notifications topic if that kind is enabled.

    ``signature`` suppresses repeats of the same event within
    DEDUPE_SECONDS. ``message_id`` makes the jump button open the topic at
    that message instead of at the top. Returns True when a message was sent.
    """
    if kind not in KINDS or not _enabled(kind):
        return False
    if _s.bot is None or _s.chat_id is None or _s.thread_id is None:
        return False
    if in_quiet_hours():
        logger.info("notification muted by quiet hours: %s %s", kind, text[:60])
        return False
    sig_key = (kind, window_id or "", signature or text)
    now = time.monotonic()
    _s.recent = {k: t for k, t in _s.recent.items() if now - t < DEDUPE_SECONDS}
    if sig_key in _s.recent:
        return False
    _s.recent[sig_key] = now

    _attr, icon = KINDS[kind]
    logger.info(
        "notification %s window=%s msg=%s: %s", kind, window_id, message_id, text[:80]
    )
    where = ""
    rows: list[list[InlineKeyboardButton]] = []
    if window_id:
        display = session_manager.get_display_name(window_id)
        where = f"**{display}** · "
        link = _topic_link(_s.chat_id, _thread_for_window(window_id), message_id)
        if link:
            rows.append([InlineKeyboardButton(f"↗ {display}", url=link)])
    if buttons:
        rows.append(buttons)
    await safe_send(
        _s.bot,
        _s.chat_id,
        f"{icon} {where}{text}",
        message_thread_id=_s.thread_id,
        reply_markup=InlineKeyboardMarkup(rows) if rows else None,
    )
    return True


# ── State hooks called by the rest of the bot ─────────────────────────────


def record_assistant_text(window_id: str, text: str) -> None:
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if first:
        _s.last_text[window_id] = first[:160]


async def record_sent(
    window_id: str, content_type: str, message_id: int, text: str
) -> None:
    """Called by the queue after a content message reached Telegram.

    Remembers the last reply's message id (for the turn-done jump link)
    and raises the API-error notification once its message exists.
    """
    # Merged queue tasks carry the first part's content_type (often
    # "thinking"), so treat every non-tool message as "the latest reply".
    if content_type in ("text", "thinking"):
        _s.last_text_msg[window_id] = message_id
    elif content_type == "error":
        await notify(
            "error",
            text[:300],
            window_id,
            signature=text[:80],
            message_id=message_id,
        )


async def mark_working(
    window_id: str, working: bool, status: str | None, paused: bool = False
) -> None:
    """Track busy→idle transitions per window; emit 'turn done'.

    ``paused`` (a dialog is on screen) freezes the state: the turn is
    neither running nor finished, so no event fires until it clears.
    """
    if paused:
        return
    since = _s.working_since.get(window_id)
    if working:
        if since is None:
            _s.working_since[window_id] = time.monotonic()
        return
    if since is None:
        return
    _s.working_since.pop(window_id, None)
    duration = time.monotonic() - since
    if duration < config.notify_turn_min:
        return
    # The footer goes idle before the queue has delivered the last reply;
    # announce a moment later (off the polling loop) so the jump link points
    # at that message.
    coro = _announce_done(window_id, since, duration, status)
    if TURN_DONE_DELAY:
        _s.pending.add(asyncio.create_task(coro))
    else:
        await coro


async def _announce_done(
    window_id: str, since: float, duration: float, status: str | None
) -> None:
    if TURN_DONE_DELAY:
        await asyncio.sleep(TURN_DONE_DELAY)
    mins, secs = divmod(int(duration), 60)
    took = f"{mins}m{secs:02d}s" if mins else f"{secs}s"
    snippet = _s.last_text.get(window_id, "")
    tail = f"\n{snippet}" if snippet else ""
    status_note = f" ({status})" if status else ""
    await notify(
        "done",
        f"finished after {took}{status_note}{tail}",
        window_id,
        signature=f"done:{int(since)}",
        message_id=_s.last_text_msg.get(window_id),
    )
    _s.pending = {t for t in _s.pending if not t.done()}


async def mark_ui(
    window_id: str,
    ui_name: str | None,
    content: str = "",
    message_id: int | None = None,
) -> None:
    """Announce a newly shown interactive dialog once; reset when it clears.

    Called with the Telegram message id of the rendered dialog (from
    interactive_ui) so the jump button lands on it; ``None`` resets.
    """
    if ui_name is None:
        if _s.announced_ui.pop(window_id, None) is not None:
            _s.ui_epoch[window_id] = _s.ui_epoch.get(window_id, 0) + 1
        return
    first = next((ln.strip() for ln in content.splitlines() if ln.strip()), "")
    signature = f"{ui_name}:{first[:80]}"
    if _s.announced_ui.get(window_id) == signature:
        return
    _s.announced_ui[window_id] = signature
    labels = {
        "AskUserQuestion": "Claude is asking you a question",
        "ExitPlanMode": "plan ready — approve?",
        "PermissionPrompt": "permission needed",
        "BashApproval": "bash command needs approval",
        "Modal": "a dialog needs an answer",
        "Settings": "a menu is open",
    }
    what = labels.get(ui_name, f"{ui_name} needs input")
    detail = (
        f"\n{first[:160]}"
        if first and ui_name in ("AskUserQuestion", "PermissionPrompt", "BashApproval")
        else ""
    )
    epoch = _s.ui_epoch.get(window_id, 0)
    await notify(
        "input",
        f"{what}{detail}",
        window_id,
        signature=f"{epoch}:{signature}",
        message_id=message_id,
    )


# ── Topic ────────────────────────────────────────────────────────────────


class NotificationsTopic:
    name = "notifications"
    callback_prefixes: tuple[str, ...] = ()

    async def on_ready(self, bot: Bot, chat_id: int, thread_id: int) -> None:
        _s.bot, _s.chat_id, _s.thread_id = bot, chat_id, thread_id
        kinds = ", ".join(
            f"{icon} {k}"
            for k, (attr, icon) in KINDS.items()
            if getattr(config, attr, True)
        )
        await safe_send(
            bot,
            chat_id,
            "🔔 **notifications** — events from all sessions land here: questions & "
            "permission prompts, finished turns, session lifecycle, API errors, "
            f"ccc quota.\nEnabled: {kinds}. Change in the settings topic or /settings.",
            message_thread_id=thread_id,
        )

    async def handle_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str
    ) -> None:
        if update.message is None:
            return
        from .settings_topic import render_settings

        body, kb = render_settings()
        await safe_reply(update.message, body, reply_markup=kb)

    async def handle_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, data: str
    ) -> None:
        return


notifications_topic = NotificationsTopic()
special_topics.register(notifications_topic)
