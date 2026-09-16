"""The ``ccc`` special topic: Claude Code / Codex account switching via `ccc`.

Wraps the `ccc` CLI (https://github.com/…/claude-code-codex-sw, JSON output)
so accounts and their five-hour / weekly quota can be seen and switched
from Telegram, and watches quota so the topic gets a message when the
account in use runs out and when an exhausted account is usable again.

Key components:
  - CccClient: async wrapper over `ccc … --json` (list / status / use /
    use-next / refresh); disabled when the binary is missing
  - Account / parse_accounts(): the subset of ccc's JSON the UI needs
  - render_dashboard(): text + inline keyboard (▶ use, ⏭ next, 🔄 refresh,
    🔁 restart Claude sessions)
  - QuotaWatcher: background loop; emits "exhausted" / "available again"
    events, waking up right after the earliest known reset time
  - CccTopic: SpecialTopic implementation registered as "ccc"
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from ..config import config
from . import special_topics
from .callback_data import (
    CB_CCC_NEXT,
    CB_CCC_REFRESH,
    CB_CCC_RESTART,
    CB_CCC_USE,
)
from .message_sender import safe_edit, safe_reply, safe_send

logger = logging.getLogger(__name__)

PROVIDER_ICON = {"claude": "✳", "codex": "❉"}
PROVIDER_TITLE = {"claude": "Claude", "codex": "Codex"}
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

    async def list(self, cached: bool = True) -> list[Account]:
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
        return {
            "five_hour": "5h",
            "seven_day": "7d",
            "seven_day_opus": "7d opus",
            "43200_minute": "30d",
        }.get(self.name, self.name.replace("_", " "))


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
    def main_windows(self) -> list[Window]:
        return [w for w in self.windows if w.name in ("five_hour", "seven_day")] or [
            w for w in self.windows if w.name == "43200_minute"
        ]

    @property
    def exhausted(self) -> bool:
        return any(w.exhausted for w in self.main_windows)

    @property
    def usable(self) -> bool:
        return self.status == "ready" and self.plan != "free" and not self.exhausted

    @property
    def next_reset(self) -> datetime | None:
        times = [w.resets_at for w in self.main_windows if w.exhausted and w.resets_at]
        return min(times) if times else None


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
                remaining=w.get("remainingPercent"),
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
        return "?"
    filled = round(pct / 100 * 6)
    return "▕" + "█" * filled + "░" * (6 - filled) + "▏"


def _short(name: str, n: int = 18) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def render_dashboard(accounts: list[Account]) -> tuple[str, InlineKeyboardMarkup]:
    now = datetime.now(timezone.utc)
    lines: list[str] = []
    rows: list[list[InlineKeyboardButton]] = []
    for provider in ("claude", "codex"):
        accs = [a for a in accounts if a.provider == provider]
        if not accs:
            continue
        paid = [a for a in accs if a.plan != "free"]
        free = [a for a in accs if a.plan == "free"]
        lines.append(
            f"{PROVIDER_ICON[provider]} **{PROVIDER_TITLE[provider]}** — {len(accs)} accounts"
        )
        for a in sorted(paid, key=lambda x: (not x.current, x.name)):
            mark = "●" if a.current else "○"
            parts = [f"{mark} `{_short(a.name)}` · {a.plan}"]
            for w in a.main_windows:
                reset = ""
                if w.resets_at:
                    # A reset time in the past means the cached quota is due
                    # for a refresh (the watcher does it; 🔄 forces it).
                    reset = (
                        " ↻due" if w.resets_at <= now else f" ↻{_rel(w.resets_at, now)}"
                    )
                parts.append(f"{w.label} {_gauge(w.remaining)}{w.remaining}%{reset}")
            flags = []
            if a.exhausted:
                flags.append("⛔ exhausted")
            if a.status != "ready":
                flags.append(a.status)
            if a.reset_credits:
                flags.append(f"🎟{a.reset_credits}")
            if a.stale:
                flags.append("stale")
            if flags:
                parts.append(" ".join(flags))
            lines.append("  " + " · ".join(parts))
        if free:
            ok = sum(1 for a in free if not a.exhausted)
            lines.append(f"  ○ {len(free)} free accounts ({ok} with quota)")
        lines.append("")
        use_row: list[InlineKeyboardButton] = []
        for a in sorted(paid, key=lambda x: x.name):
            if a.current:
                continue
            label = f"▶ {_short(a.name, 14)}" + (" ⛔" if a.exhausted else "")
            use_row.append(
                InlineKeyboardButton(label, callback_data=f"{CB_CCC_USE}{a.id}"[:64])
            )
            if len(use_row) == 2:
                rows.append(use_row)
                use_row = []
        if use_row:
            rows.append(use_row)
        rows.append(
            [
                InlineKeyboardButton(
                    f"⏭ Next {PROVIDER_TITLE[provider]}",
                    callback_data=f"{CB_CCC_NEXT}{provider}",
                )
            ]
        )
    if not lines:
        lines.append("No accounts known to ccc yet (`ccc init` / `ccc add`).")
    rows.append(
        [
            InlineKeyboardButton("🔄 Refresh", callback_data=f"{CB_CCC_REFRESH}net"),
            InlineKeyboardButton(
                "🔁 Restart Claude sessions", callback_data=CB_CCC_RESTART
            ),
        ]
    )
    lines.append(f"_updated {now.strftime('%H:%M:%S')} UTC_")
    return "\n".join(lines), InlineKeyboardMarkup(rows)


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
            elif was and not now and a.plan != "free":
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
                "🔀 **ccc** — accounts & quota\n" + text,
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
        a = ev.account
        icon = PROVIDER_ICON.get(a.provider, "")
        if ev.kind == "exhausted":
            reset = _rel(a.next_reset)
            others = [
                x
                for x in accounts
                if x.provider == a.provider and not x.current and x.usable
            ]
            text = (
                f"⛔ {icon} **{PROVIDER_TITLE.get(a.provider, a.provider)}** account "
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
                            f"▶ Use {_short(best.name, 20)}",
                            callback_data=f"{CB_CCC_USE}{best.id}"[:64],
                        )
                    ]
                )
            else:
                text += "\nNo other usable account right now."
            rows.append(
                [
                    InlineKeyboardButton(
                        f"⏭ Next {PROVIDER_TITLE.get(a.provider, '')}",
                        callback_data=f"{CB_CCC_NEXT}{a.provider}",
                    )
                ]
            )
            kb = InlineKeyboardMarkup(rows)
        else:
            text = (
                f"✅ {icon} **{PROVIDER_TITLE.get(a.provider, a.provider)}** account "
                f"`{a.name}` is available again ("
                + ", ".join(f"{w.label} {w.remaining}%" for w in a.main_windows)
                + ")."
            )
            kb = (
                InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                f"▶ Use {_short(a.name, 20)}",
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

    # -- interaction

    async def handle_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str
    ) -> None:
        msg = update.message
        if msg is None:
            return
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
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, data: str
    ) -> None:
        query = update.callback_query
        if query is None:
            return
        try:
            if data.startswith(CB_CCC_USE):
                acc_id = data[len(CB_CCC_USE) :]
                target = next((a for a in self._last if a.id == acc_id), None)
                label = target.name if target else acc_id[:8]
                await query.answer(f"Switching to {label}…")
                result = await self.client.use(acc_id)
                cur = result.get("current") or {}
                note = f"✅ Now using `{cur.get('name', label)}` ({cur.get('provider', '')})."
                if cur.get("provider") == "claude":
                    note += " Running Claude Code sessions keep their old login until restarted."
            elif data.startswith(CB_CCC_NEXT):
                provider = data[len(CB_CCC_NEXT) :]
                await query.answer(f"Picking the best {provider} account…")
                result = await self.client.use_next(provider)
                cur = result.get("current") or result.get("selected") or {}
                note = f"✅ Switched {provider} to `{cur.get('name', '?')}`."
                if provider == "claude":
                    note += " Running Claude Code sessions keep their old login until restarted."
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
        body, kb = render_dashboard(accounts)
        await safe_edit(query, (note + "\n\n" if note else "") + body, reply_markup=kb)


ccc_topic = CccTopic()
special_topics.register(ccc_topic)
