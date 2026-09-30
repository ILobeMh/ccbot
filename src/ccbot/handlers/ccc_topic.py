"""The ``ccc`` special topic: AI coding tool account switching via `ccc`.

Wraps the `ccc` CLI (https://github.com/…/claude-code-codex-sw, JSON output)
so accounts (Claude, Codex, Grok, … — any provider ccc knows) and their
quota can be seen and switched from Telegram, and watches quota so the topic
gets a message when the account in use runs out and when an exhausted
account is usable again. One message, edited in place: a home view with one
line per provider, and a page per provider listing its accounts.

Key components:
  - CccClient: async wrapper over `ccc … --json` (list / status / use /
    use-next / refresh); disabled when the binary is missing
  - Account / parse_accounts(): the subset of ccc's JSON the UI needs
  - render_dashboard() / render_provider(): home / provider page, text +
    inline keyboard (provider buttons, ▶ use, ⏭ next, « back, 🔄 refresh,
    🔁 restart Claude sessions)
  - QuotaWatcher: background loop; emits "exhausted" / "available again"
    events, waking up right after the earliest known reset time
  - CccTopic: SpecialTopic implementation registered as "ccc"
"""

from __future__ import annotations

import asyncio
import builtins
import json
import logging
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aiogram import Bot
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from ..config import config
from ..settings import in_quiet_hours
from . import special_topics
from .callback_data import (
    CB_CCC_HOME,
    CB_CCC_NEXT,
    CB_CCC_PROV,
    CB_CCC_REFRESH,
    CB_CCC_RESTART,
    CB_CCC_USE,
)
from .message_sender import safe_edit, safe_reply, safe_send

logger = logging.getLogger(__name__)

PROVIDER_ICON = {"claude": "✳", "codex": "❉", "grok": "✦"}
PROVIDER_TITLE = {"claude": "Claude", "codex": "Codex", "grok": "Grok"}
# Windows shown in the main view, in display order (others are per-model extras)
MAIN_WINDOWS = ("five_hour", "seven_day")
# Target width of the monospace account blocks (phone screens)
BLOCK_WIDTH = 34
# A window at or below this many percent counts as exhausted
EXHAUSTED_PCT = 1


# ── ccc CLI wrapper ──────────────────────────────────────────────────────


class CccError(RuntimeError):
    pass


class CccClient:
    def __init__(self, command: str | None = None) -> None:
        self.command = command or config.ccc_command

    def available(self) -> bool:
        return shutil.which(self.command) is not None

    async def run(self, *args: str, timeout: float = 90.0) -> dict:
        proc = await asyncio.create_subprocess_exec(
            self.command,
            *args,
            "--json",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise CccError(f"ccc {' '.join(args)} timed out after {timeout:.0f}s")
        text = out.decode("utf-8", errors="replace").strip()
        stderr = err.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            raise CccError(stderr or text or f"ccc exited with {proc.returncode}")
        # ccc prints warnings on stderr and JSON on stdout
        try:
            return json.loads(text) if text else {}
        except json.JSONDecodeError as e:
            raise CccError(f"ccc returned invalid JSON: {e}: {text[:200]}")

    async def list(self, cached: bool = True) -> builtins.list[Account]:
        args = ["list"] + (["--cached"] if cached else [])
        return parse_accounts(await self.run(*args))

    async def use(self, selector: str) -> dict:
        return await self.run("use", selector)

    async def use_next(self, provider: str) -> dict:
        return await self.run("use-next", provider)

    async def refresh(self, selector: str | None = None) -> dict:
        return await self.run("refresh", *([selector] if selector else []))


# ── Data model ───────────────────────────────────────────────────────────


@dataclass
class Window:
    name: str
    remaining: int | None
    resets_at: datetime | None

    @property
    def exhausted(self) -> bool:
        return self.remaining is not None and self.remaining <= EXHAUSTED_PCT

    @property
    def label(self) -> str:
        known = {
            "five_hour": "5h",
            "seven_day": "7d",
            "seven_day_opus": "7d opus",
        }
        if self.name in known:
            return known[self.name]
        minutes = self.name.removesuffix("_minute")
        if minutes != self.name and minutes.isdigit():
            return _minutes_label(int(minutes))
        return self.name.replace("_", " ")


@dataclass
class Account:
    id: str
    provider: str
    name: str
    current: bool
    status: str
    plan: str
    stale: bool
    windows: list[Window] = field(default_factory=list)
    reset_credits: int = 0
    fetched_at: datetime | None = None

    @property
    def selector(self) -> str:
        return self.id

    @property
    def free(self) -> bool:
        return self.plan.lower() == "free"

    @property
    def main_windows(self) -> list[Window]:
        """5h / 7d; else the 30-day window; else whatever the provider reports."""
        main = sorted(
            (w for w in self.windows if w.name in MAIN_WINDOWS),
            key=lambda w: MAIN_WINDOWS.index(w.name),
        )
        return (
            main
            or [w for w in self.windows if w.name == "43200_minute"]
            or self.windows[:2]
        )

    @property
    def exhausted(self) -> bool:
        return any(w.exhausted for w in self.main_windows)

    @property
    def usable(self) -> bool:
        return self.status == "ready" and not self.free and not self.exhausted

    @property
    def next_reset(self) -> datetime | None:
        times = [w.resets_at for w in self.main_windows if w.exhausted and w.resets_at]
        return min(times) if times else None


def _minutes_label(minutes: int) -> str:
    """300 → ``5h``, 10080 → ``7d``, 43200 → ``30d``."""
    if minutes and minutes % 1440 == 0:
        return f"{minutes // 1440}d"
    if minutes and minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}m"


def _pct(raw: object) -> int | None:
    """remainingPercent as a whole number (ccc may send floats like 98.58)."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return None
    try:
        return max(0, min(100, round(float(raw))))
    except (TypeError, ValueError, OverflowError):
        return None


def _parse_time(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_accounts(data: dict) -> list[Account]:
    out: list[Account] = []
    for entry in data.get("accounts", []):
        acc = entry.get("account") or {}
        quota = entry.get("quota") or {}
        windows = [
            Window(
                name=str(w.get("name", "")),
                remaining=_pct(w.get("remainingPercent")),
                resets_at=_parse_time(w.get("resetsAt")),
            )
            for w in quota.get("windows", [])
            if isinstance(w, dict)
        ]
        credits = quota.get("resetCredits") or {}
        out.append(
            Account(
                id=str(acc.get("id", "")),
                provider=str(acc.get("provider", "")),
                name=str(acc.get("name") or acc.get("email") or "?"),
                current=bool(entry.get("current")),
                status=str(acc.get("status", "")),
                plan=str(quota.get("plan") or "?"),
                stale=bool(entry.get("stale")),
                windows=windows,
                reset_credits=int(credits.get("availableCount") or 0),
                fetched_at=_parse_time(quota.get("fetchedAt")),
            )
        )
    return out


# ── Rendering ────────────────────────────────────────────────────────────


def _rel(dt: datetime | None, now: datetime | None = None) -> str:
    if dt is None:
        return "?"
    now = now or datetime.now(timezone.utc)
    secs = int((dt - now).total_seconds())
    sign = "" if secs >= 0 else "-"
    secs = abs(secs)
    d, rem = divmod(secs, 86400)
    h, rem = divmod(rem, 3600)
    m = rem // 60
    if d:
        return f"{sign}{d}d{h}h"
    if h:
        return f"{sign}{h}h{m:02d}m"
    return f"{sign}{m}m"


def _gauge(pct: int | None) -> str:
    if pct is None:
        return "▕??????▏"
    filled = round(pct / 100 * 6)
    return "▕" + "█" * filled + "░" * (6 - filled) + "▏"


def _short(name: str, n: int = 18) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def _icon(provider: str) -> str:
    return PROVIDER_ICON.get(provider, "•")


def _title(provider: str) -> str:
    return PROVIDER_TITLE.get(provider, provider.title())


def _providers(accounts: list[Account]) -> list[str]:
    """Providers that have accounts: known ones first, then any others."""
    seen = list(dict.fromkeys(a.provider for a in accounts))
    known = [p for p in PROVIDER_TITLE if p in seen]
    return known + [p for p in seen if p not in PROVIDER_TITLE]


def _reset_text(w: Window, now: datetime) -> str:
    """``↻ 5d8h`` while the window is below 100%; ``↻ due`` once it has passed."""
    if w.resets_at and w.resets_at <= now:
        return "↻ due"  # cached quota older than its reset; 🔄 refreshes
    if w.resets_at and (w.remaining is None or w.remaining < 100):
        return f"↻ {_rel(w.resets_at, now)}"
    return ""


def _gauge_line(w: Window, now: datetime) -> str:
    """One aligned gauge row for a code block: ``5h ▕███░░░▏  51%  ↻ 5d8h``."""
    pct = "?" if w.remaining is None else f"{w.remaining}%"
    reset = _reset_text(w, now)
    return f"  {w.label:<3}{_gauge(w.remaining)} {pct:>4}" + (
        f"  {reset}" if reset else ""
    )


def _account_block(a: Account, now: datetime) -> list[str]:
    mark = "●" if a.current else "○"
    flags = []
    if a.exhausted:
        flags.append("⛔")
    if a.status != "ready":
        flags.append(a.status)
    if a.reset_credits:
        flags.append(f"🎟{a.reset_credits}")
    if a.stale:
        flags.append("stale")
    head = [f"{mark} {_short(a.name)}  {a.plan}"]
    if flags:
        joined = " ".join(flags)
        if len(head[0]) + 2 + len(joined) <= BLOCK_WIDTH:
            head[0] += "  " + joined
        else:  # keep phone-width: flags go on their own line
            head.append("  " + joined)
    return head + [_gauge_line(w, now) for w in a.main_windows]


def _updated(now: datetime) -> str:
    return f"_updated {now.strftime('%H:%M')} UTC_"


def _home_windows(a: Account, now: datetime) -> str:
    """``5h ▕░░░░░░▏ 4% ↻ 1h12m · 7d 51%``: gauge + reset only on the tightest window."""
    windows = a.main_windows
    if not windows:
        return "no quota data"
    below = [w for w in windows if w.remaining is not None and w.remaining < 100]
    tight = min(below, key=lambda w: w.remaining or 0, default=None)
    pieces = []
    for w in windows:
        pct = "?" if w.remaining is None else f"{w.remaining}%"
        if w.exhausted:
            pct = f"⛔ {pct}"
        if w is tight:
            reset = _reset_text(w, now)
            pieces.append(
                f"{w.label} {_gauge(w.remaining)} {pct}"
                + (f" {reset}" if reset else "")
            )
        else:
            pieces.append(f"{w.label} {pct}")
    return " · ".join(pieces)


def _home_line(provider: str, accs: list[Account], now: datetime) -> str:
    in_use = next((a for a in accs if a.current), None)
    parts = [f"{_icon(provider)} **{_title(provider)}**"]
    if in_use is None:
        parts.append("_none in use_")
    else:
        parts.append(f"`{_short(in_use.name, 7)}` ({in_use.plan})")
        parts.append(_home_windows(in_use, now))
    others = [a for a in accs if a is not in_use]
    free = sum(1 for a in others if a.free)
    if free:
        parts.append(f"{free} free")
    broken = sum(1 for a in others if a.status != "ready")
    if broken:
        parts.append(f"⚠ {broken} need login")
    return " · ".join(parts)


def render_dashboard(accounts: list[Account]) -> tuple[str, InlineKeyboardMarkup]:
    """Home view: one compact line per provider, one button per provider."""
    now = datetime.now(timezone.utc)
    lines = ["🔀 **ccc** — accounts & quota", ""]
    buttons: list[InlineKeyboardButton] = []
    providers = _providers(accounts)
    for provider in providers:
        accs = [a for a in accounts if a.provider == provider]
        lines.append(_home_line(provider, accs, now))
        buttons.append(
            InlineKeyboardButton(
                text=f"{_icon(provider)} {_title(provider)} ›",
                callback_data=f"{CB_CCC_PROV}{provider}"[:64],
            )
        )
    if not providers:
        lines.append("No accounts known to ccc yet (`ccc init` / `ccc add`).")
    lines += ["", _updated(now)]
    rows = [buttons[i : i + 3] for i in range(0, len(buttons), 3)]
    rows.append(
        [
            InlineKeyboardButton(
                text="🔄 Refresh", callback_data=f"{CB_CCC_REFRESH}net"
            ),
            InlineKeyboardButton(
                text="🔁 Restart Claude sessions", callback_data=CB_CCC_RESTART
            ),
        ]
    )
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


def render_provider(
    accounts: list[Account], provider: str
) -> tuple[str, InlineKeyboardMarkup]:
    """Provider page: a monospace block per account, use / next / back buttons.

    Paid accounts (and the one in use) get a line plus one aligned gauge line
    per quota window, so nothing wraps and the bars line up; free accounts
    are summarised in one line.
    """
    now = datetime.now(timezone.utc)
    back = [InlineKeyboardButton(text="« Back", callback_data=CB_CCC_HOME)]
    accs = [a for a in accounts if a.provider == provider]
    if not accs:
        text = f"{_icon(provider)} **{_title(provider)}** — no accounts"
        return text, InlineKeyboardMarkup(inline_keyboard=[back])
    shown = sorted(
        (a for a in accs if not a.free or a.current),
        key=lambda x: (not x.current, x.name),
    )
    free = [a for a in accs if a.free and not a.current]
    count = f"{len(accs)} account{'s' if len(accs) != 1 else ''}"
    lines = [f"{_icon(provider)} **{_title(provider)}** — {count}"]
    block: list[str] = []
    for a in shown:
        block.extend(_account_block(a, now))
    if free:
        ok = sum(1 for a in free if a.status == "ready" and not a.exhausted)
        broken = sum(1 for a in free if a.status != "ready")
        summary = f"○ {len(free)} free account{'s' if len(free) != 1 else ''} ({ok} with quota"
        if broken:
            summary += f", {broken} need login"
        block.append(summary + ")")
    lines.append("```\n" + "\n".join(block) + "\n```")
    lines.append(
        f"_● in use · ↻ resets in · 🎟 reset credits · updated "
        f"{now.strftime('%H:%M')} UTC_"
    )

    rows: list[list[InlineKeyboardButton]] = []
    use_row: list[InlineKeyboardButton] = []
    for a in shown:
        if a.current:
            continue
        label = f"▶ {_short(a.name, 14)}" + (" ⛔" if a.exhausted else "")
        use_row.append(
            InlineKeyboardButton(text=label, callback_data=f"{CB_CCC_USE}{a.id}"[:64])
        )
        if len(use_row) == 2:
            rows.append(use_row)
            use_row = []
    if use_row:
        rows.append(use_row)
    rows.append(
        [
            InlineKeyboardButton(
                text=f"⏭ Next {_title(provider)}",
                callback_data=f"{CB_CCC_NEXT}{provider}"[:64],
            )
        ]
    )
    rows.append(back)
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


# ── Quota watcher ────────────────────────────────────────────────────────


@dataclass
class QuotaEvent:
    kind: str  # "exhausted" | "available"
    account: Account


class QuotaWatcher:
    """Turn successive account snapshots into exhausted / available events."""

    def __init__(self) -> None:
        self._exhausted: dict[str, bool] = {}
        self._primed = False

    def observe(self, accounts: list[Account]) -> list[QuotaEvent]:
        events: list[QuotaEvent] = []
        for a in accounts:
            was = self._exhausted.get(a.id)
            now = a.exhausted
            self._exhausted[a.id] = now
            if not self._primed or was is None or was == now:
                continue
            if now and a.current:
                events.append(QuotaEvent("exhausted", a))
            elif was and not now and not a.free:
                events.append(QuotaEvent("available", a))
        self._primed = True
        return events

    @staticmethod
    def next_wakeup(accounts: list[Account], default: float) -> float:
        """Seconds to sleep: until just after the earliest reset, capped."""
        now = datetime.now(timezone.utc)
        soonest = default
        for a in accounts:
            reset = a.next_reset
            if reset is None:
                continue
            secs = (reset - now).total_seconds() + 45
            soonest = min(soonest, max(30.0, secs))
        return soonest


# ── Topic ────────────────────────────────────────────────────────────────

RestartAll = Callable[[], Awaitable[str]]


class CccTopic:
    name = "ccc"
    callback_prefixes: tuple[str, ...] = (
        CB_CCC_USE,
        CB_CCC_NEXT,
        CB_CCC_REFRESH,
        CB_CCC_RESTART,
        CB_CCC_HOME,
        CB_CCC_PROV,
    )

    def __init__(self, client: CccClient | None = None) -> None:
        self.client = client or CccClient()
        self.watcher = QuotaWatcher()
        self._task: asyncio.Task[None] | None = None
        self._bot: Bot | None = None
        self._chat_id: int | None = None
        self._thread_id: int | None = None
        self._restart_all: RestartAll | None = None
        self._last: list[Account] = []

    def set_restart_handler(self, fn: RestartAll) -> None:
        self._restart_all = fn

    async def stop(self) -> None:
        """Cancel the quota watcher (bot shutdown)."""
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def on_ready(self, bot: Bot, chat_id: int, thread_id: int) -> None:
        self._bot, self._chat_id, self._thread_id = bot, chat_id, thread_id
        if not self.client.available():
            await safe_send(
                bot,
                chat_id,
                f"⚠️ `{self.client.command}` is not installed on this host — account "
                "switching is unavailable. Set CCBOT_CCC_COMMAND if it lives elsewhere.",
                message_thread_id=thread_id,
            )
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._watch_loop())
        try:
            accounts = await self.client.list(cached=True)
            self.watcher.observe(accounts)  # prime without emitting events
            text, kb = render_dashboard(accounts)
            await safe_send(
                bot,
                chat_id,
                text,
                message_thread_id=thread_id,
                reply_markup=kb,
            )
        except CccError as e:
            await safe_send(bot, chat_id, f"❌ ccc: {e}", message_thread_id=thread_id)

    # -- watcher

    async def _watch_loop(self) -> None:
        logger.info("ccc quota watcher started (every %ss)", config.ccc_poll_interval)
        while True:
            delay = config.ccc_poll_interval
            try:
                # `ccc list` (no --cached) applies ccc's own refresh policy:
                # active accounts every 10 min, one due inactive per call.
                accounts = await self.client.list(cached=False)
                # Exhausted accounts whose reset time has passed: refresh
                # just those so "available again" fires promptly.
                now = datetime.now(timezone.utc)
                for a in accounts:
                    reset = a.next_reset
                    if a.exhausted and reset and reset < now and a.status == "ready":
                        try:
                            await self.client.refresh(a.selector)
                        except CccError as e:
                            logger.debug("ccc refresh %s: %s", a.name, e)
                        accounts = await self.client.list(cached=True)
                        break
                self._last = accounts
                for ev in self.watcher.observe(accounts):
                    await self._announce(ev, accounts)
                delay = QuotaWatcher.next_wakeup(accounts, config.ccc_poll_interval)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("ccc watcher: %s", e)
            await asyncio.sleep(delay)

    async def _announce(self, ev: QuotaEvent, accounts: list[Account]) -> None:
        if self._bot is None or self._chat_id is None:
            return
        if not config.ccc_alerts or in_quiet_hours():
            logger.info("ccc alert suppressed (%s %s)", ev.kind, ev.account.name)
            return
        a = ev.account
        icon = _icon(a.provider)
        if ev.kind == "exhausted":
            reset = _rel(a.next_reset)
            others = [
                x
                for x in accounts
                if x.provider == a.provider and not x.current and x.usable
            ]
            text = (
                f"⛔ {icon} **{_title(a.provider)}** account "
                f"`{a.name}` (in use) is exhausted — resets in {reset}."
            )
            rows: list[list[InlineKeyboardButton]] = []
            if others:
                best = max(
                    others,
                    key=lambda x: (
                        min(w.remaining or 0 for w in x.main_windows)
                        if x.main_windows
                        else 0
                    ),
                )
                text += f"\nBest alternative: `{best.name}`."
                rows.append(
                    [
                        InlineKeyboardButton(
                            text=f"▶ Use {_short(best.name, 20)}",
                            callback_data=f"{CB_CCC_USE}{best.id}"[:64],
                        )
                    ]
                )
            else:
                text += "\nNo other usable account right now."
            rows.append(
                [
                    InlineKeyboardButton(
                        text=f"⏭ Next {_title(a.provider)}",
                        callback_data=f"{CB_CCC_NEXT}{a.provider}",
                    )
                ]
            )
            kb = InlineKeyboardMarkup(inline_keyboard=rows)
        else:
            text = (
                f"✅ {icon} **{_title(a.provider)}** account "
                f"`{a.name}` is available again ("
                + ", ".join(f"{w.label} {w.remaining}%" for w in a.main_windows)
                + ")."
            )
            kb = (
                InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text=f"▶ Use {_short(a.name, 20)}",
                                callback_data=f"{CB_CCC_USE}{a.id}"[:64],
                            )
                        ]
                    ]
                )
                if not a.current
                else None
            )
        await safe_send(
            self._bot,
            self._chat_id,
            text,
            message_thread_id=self._thread_id,
            reply_markup=kb,
        )
        # Deferred import: notifications_topic imports nothing from here
        from .notifications_topic import notify

        await notify("ccc", text, signature=f"{ev.kind}:{a.id}")

    # -- interaction

    async def handle_text(
        self, message: Message, bot: Bot, user_data: dict[str, Any], text: str
    ) -> None:
        msg = message
        if not self.client.available():
            await safe_reply(msg, f"⚠️ `{self.client.command}` is not installed here.")
            return
        cmd = text.strip().lower().lstrip("/")
        cached = cmd not in ("refresh", "r")
        try:
            accounts = await self.client.list(cached=cached)
        except CccError as e:
            await safe_reply(msg, f"❌ ccc: {e}")
            return
        self._last = accounts
        body, kb = render_dashboard(accounts)
        await safe_reply(msg, body, reply_markup=kb)

    async def handle_callback(
        self, query: CallbackQuery, bot: Bot, user_data: dict[str, Any], data: str
    ) -> None:
        view = ""  # provider page to re-render; "" = home
        try:
            if data == CB_CCC_HOME:
                await query.answer()
                note = ""
            elif data.startswith(CB_CCC_PROV):
                await query.answer()
                note = ""
                view = data[len(CB_CCC_PROV) :]
            elif data.startswith(CB_CCC_USE):
                acc_id = data[len(CB_CCC_USE) :]
                target = next((a for a in self._last if a.id == acc_id), None)
                label = target.name if target else acc_id[:8]
                await query.answer(f"Switching to {label}…")
                result = await self.client.use(acc_id)
                cur = result.get("current") or {}
                note = f"✅ Now using `{cur.get('name', label)}` ({cur.get('provider', '')})."
                if cur.get("provider") == "claude":
                    note += " Running Claude Code sessions keep their old login until restarted."
                view = str(cur.get("provider") or (target.provider if target else ""))
            elif data.startswith(CB_CCC_NEXT):
                provider = data[len(CB_CCC_NEXT) :]
                await query.answer(f"Picking the best {provider} account…")
                result = await self.client.use_next(provider)
                cur = result.get("current") or result.get("selected") or {}
                note = f"✅ Switched {provider} to `{cur.get('name', '?')}`."
                if provider == "claude":
                    note += " Running Claude Code sessions keep their old login until restarted."
                view = provider
            elif data.startswith(CB_CCC_REFRESH):
                await query.answer("Refreshing from the providers…")
                note = ""
            elif data == CB_CCC_RESTART:
                if self._restart_all is None:
                    await query.answer("Not wired up", show_alert=True)
                    return
                await query.answer("Restarting sessions…")
                if self._bot is not None and self._chat_id is not None:
                    await safe_send(
                        self._bot,
                        self._chat_id,
                        "🔁 Restarting all Claude Code sessions…",
                        message_thread_id=self._thread_id,
                    )
                summary = await self._restart_all()
                if self._bot is not None and self._chat_id is not None:
                    await safe_send(
                        self._bot,
                        self._chat_id,
                        summary,
                        message_thread_id=self._thread_id,
                    )
                return
            else:
                await query.answer()
                return
            accounts = await self.client.list(
                cached=not data.startswith(CB_CCC_REFRESH)
            )
        except CccError as e:
            await query.answer(f"ccc: {e}"[:200], show_alert=True)
            return
        self._last = accounts
        self.watcher.observe(accounts)
        body, kb = (
            render_provider(accounts, view) if view else render_dashboard(accounts)
        )
        await safe_edit(query, (note + "\n\n" if note else "") + body, reply_markup=kb)


ccc_topic = CccTopic()
special_topics.register(ccc_topic)
