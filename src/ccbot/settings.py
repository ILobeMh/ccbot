"""Runtime settings: user-adjustable knobs persisted in <CCBOT_DIR>/settings.json.

Environment variables stay the defaults; anything changed from the
``settings`` topic (or ``/settings``) is stored in settings.json and applied
to the live ``config`` object immediately, so the rest of the code keeps
reading ``config.<attr>`` and needs no restart.

Key components:
  - SETTINGS: ordered registry of Setting (key = config attribute name)
  - load(): apply persisted overrides onto config at startup
  - set_value()/cycle(): change one setting, persist, run its on_change hook
  - in_quiet_hours(): shared check for alert-style notifications
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .config import config
from .utils import atomic_write_json

logger = logging.getLogger(__name__)


@dataclass
class Setting:
    key: str  # attribute on config
    label: str
    group: str
    kind: str  # "bool" | "choice"
    choices: list[Any] = field(default_factory=list)  # for "choice"
    fmt: Callable[[Any], str] = lambda v: str(v)
    help: str = ""
    on_change: Callable[[Any], None] | None = None


def _secs(v: Any) -> str:
    v = int(v)
    return f"{v // 60}m" if v >= 60 and v % 60 == 0 else f"{v}s"


def _thinking(v: Any) -> str:
    return {0: "full (split in parts)", 500: "short (500)"}.get(int(v), f"{v} chars")


def _quiet(v: Any) -> str:
    return "off" if not v else str(v).replace("-", "–")


def _mode(v: Any) -> str:
    return {
        "default": "Normal",
        "acceptEdits": "Accept edits",
        "plan": "Plan",
        "bypassPermissions": "Skip permissions",
    }.get(str(v), str(v))


def _forget_last_modes(_: Any) -> None:
    # Deferred import: session imports config, settings must not import session at top
    from .session import session_manager

    session_manager.last_launch_modes.clear()
    session_manager._save_state()


SETTINGS: list[Setting] = [
    Setting(
        "show_thinking",
        "Thinking",
        "Output",
        "bool",
        help="Show Claude's thinking as collapsed quotes",
    ),
    Setting(
        "thinking_max_chars",
        "Thinking length",
        "Output",
        "choice",
        choices=[500, 1500, 3000, 0],
        fmt=_thinking,
        help="Truncate thinking, or 'full' to send it all in [i/N] parts",
    ),
    Setting(
        "show_tool_calls",
        "Tool calls",
        "Output",
        "bool",
        help="tool_use / tool_result messages",
    ),
    Setting("show_user_messages", "Mirror my prompts (👤)", "Output", "bool"),
    Setting(
        "status_updates",
        "Status line",
        "Output",
        "bool",
        help="'Moseying… (7s)' progress edits",
    ),
    Setting(
        "claude_permission_mode",
        "Default launch mode",
        "Sessions",
        "choice",
        choices=["default", "acceptEdits", "plan", "bypassPermissions"],
        fmt=_mode,
        help="Listed first in the mode picker (also resets remembered choices)",
        on_change=_forget_last_modes,
    ),
    Setting(
        "auto_trust_dirs",
        "Auto-trust folders",
        "Sessions",
        "bool",
        help="Pre-accept the workspace trust dialog",
    ),
    Setting(
        "shell_timeout",
        "shell: timeout",
        "Topics",
        "choice",
        choices=[30.0, 60.0, 120.0, 300.0, 600.0],
        fmt=_secs,
    ),
    Setting(
        "ccc_alerts",
        "ccc: quota alerts",
        "Topics",
        "bool",
        help="Exhausted / available-again messages",
    ),
    Setting(
        "ccc_poll_interval",
        "ccc: poll every",
        "Topics",
        "choice",
        choices=[120.0, 300.0, 600.0, 1800.0],
        fmt=_secs,
    ),
    Setting(
        "screenshot_font_size",
        "Screenshot font",
        "Misc",
        "choice",
        choices=[20, 24, 28, 32, 36],
        fmt=lambda v: f"{v}px",
    ),
    Setting(
        "quiet_hours",
        "Quiet hours",
        "Misc",
        "choice",
        choices=["", "22-07", "23-08", "00-08"],
        fmt=_quiet,
        help="No ccc / health alerts in this window (server local time)",
    ),
]
_BY_KEY = {s.key: s for s in SETTINGS}


def settings_file():
    return config.config_dir / "settings.json"


def get(key: str) -> Any:
    return getattr(config, key)


def load() -> None:
    """Apply persisted overrides to config. Unknown/invalid keys are ignored."""
    path = settings_file()
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("Ignoring unreadable %s: %s", path, e)
        return
    if not isinstance(data, dict):
        return
    for key, value in data.items():
        setting = _BY_KEY.get(key)
        if setting is None or not _valid(setting, value):
            logger.warning("settings.json: ignoring %s=%r", key, value)
            continue
        setattr(config, key, _coerce(setting, value))
    logger.info("Applied %d runtime settings from %s", len(data), path)


def _valid(setting: Setting, value: Any) -> bool:
    if setting.kind == "bool":
        return isinstance(value, bool)
    return any(_same(c, value) for c in setting.choices)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return False
    return a == b


def _coerce(setting: Setting, value: Any) -> Any:
    if setting.kind == "bool":
        return bool(value)
    for c in setting.choices:
        if _same(c, value):
            return c
    return value


def _persist() -> None:
    data = {s.key: get(s.key) for s in SETTINGS}
    atomic_write_json(settings_file(), data)


def set_value(key: str, value: Any) -> Any:
    setting = _BY_KEY[key]
    if not _valid(setting, value):
        raise ValueError(f"invalid value {value!r} for {key}")
    value = _coerce(setting, value)
    setattr(config, key, value)
    _persist()
    if setting.on_change:
        setting.on_change(value)
    logger.info("Setting %s = %r", key, value)
    return value


def cycle(key: str) -> Any:
    """Toggle a bool or advance a choice to the next option (wrapping)."""
    setting = _BY_KEY[key]
    current = get(key)
    if setting.kind == "bool":
        return set_value(key, not current)
    idx = next((i for i, c in enumerate(setting.choices) if _same(c, current)), -1)
    return set_value(key, setting.choices[(idx + 1) % len(setting.choices)])


def display(setting: Setting) -> str:
    value = get(setting.key)
    if setting.kind == "bool":
        return "✅ on" if value else "❌ off"
    return setting.fmt(value)


def in_quiet_hours(now: datetime | None = None) -> bool:
    """True when alert-style notifications should be held back."""
    window = str(getattr(config, "quiet_hours", "") or "")
    if not window:
        return False
    try:
        start_s, end_s = window.split("-")
        start, end = int(start_s), int(end_s)
    except ValueError:
        return False
    hour = (now or datetime.now()).hour
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end
