"""Terminal output parser — detects Claude Code UI elements in pane text.

Parses captured tmux pane content to detect:
  - Interactive UIs (AskUserQuestion, ExitPlanMode, Permission Prompt,
    TrustDialog, BypassWarning, generic Modal, …) via regex-based UIPattern
    matching with top/bottom delimiters.
  - Startup dialogs the bot answers itself (AUTO_ANSWER_DIALOGS).
  - Status line (spinner characters + working text) by scanning from bottom
    up, skipping chrome such as "⎿ Tip:" blocks.
  - Prompt readiness, the permission mode shown in the footer, and the
    "Update installed · Restart to update" hint.

All Claude Code text patterns live here. To support a new UI type or
a changed Claude Code version, edit UI_PATTERNS / STATUS_SPINNERS.

Key functions: is_interactive_ui(), extract_interactive_content(),
parse_status_line(), is_prompt_ready(), parse_permission_mode(),
strip_pane_chrome(), extract_bash_output().
"""

import re
from dataclasses import dataclass


@dataclass
class InteractiveUIContent:
    """Content extracted from an interactive UI."""

    content: str  # The extracted display content
    name: str = ""  # Pattern name that matched (e.g. "AskUserQuestion")


@dataclass(frozen=True)
class UIPattern:
    """A text-marker pair that delimits an interactive UI region.

    Extraction scans lines top-down: the first line matching any `top` pattern
    marks the start, the first subsequent line matching any `bottom` pattern
    marks the end.  Both boundary lines are included in the extracted content.

    ``top`` and ``bottom`` are tuples of compiled regexes — any single match
    is sufficient.  This accommodates wording changes across Claude Code
    versions (e.g. a reworded confirmation prompt).
    """

    name: str  # Descriptive label (not used programmatically)
    top: tuple[re.Pattern[str], ...]
    bottom: tuple[re.Pattern[str], ...]
    min_gap: int = 2  # minimum lines between top and bottom (inclusive)


# ── UI pattern definitions (order matters — first match wins) ────────────

UI_PATTERNS: list[UIPattern] = [
    # ── Startup / one-off dialogs (also auto-answered, see AUTO_ANSWER_DIALOGS)
    UIPattern(
        name="TrustDialog",
        top=(
            # 2.1.26x wording; "No, exit" is the default option
            re.compile(r"^\s*Quick safety check"),
            # older wording
            re.compile(r"^\s*Do you trust the files in this folder"),
        ),
        bottom=(re.compile(r"Enter to confirm"),),
    ),
    UIPattern(
        name="BypassWarning",
        top=(
            re.compile(r"Bypass Permissions mode", re.IGNORECASE),
            re.compile(r"^\s*WARNING: Claude Code running in"),
        ),
        bottom=(
            re.compile(r"Enter to confirm"),
            re.compile(r"^\s*Esc to"),
        ),
    ),
    UIPattern(
        name="ResumeSession",
        top=(re.compile(r"^\s*Resume from (summary|full session)"),),
        bottom=(re.compile(r"Enter to (confirm|select)"),),
    ),
    UIPattern(
        name="ExitPlanMode",
        top=(
            re.compile(r"^\s*Would you like to proceed\?"),
            # v2.1.29+: longer prefix that may wrap across lines
            re.compile(r"^\s*Claude has written up a plan"),
        ),
        bottom=(
            re.compile(r"^\s*ctrl-g to edit in "),
            re.compile(r"^\s*Esc to (cancel|exit)"),
        ),
    ),
    UIPattern(
        name="AskUserQuestion",
        top=(re.compile(r"^\s*←\s+[☐✔☒]"),),  # Multi-tab: no bottom needed
        bottom=(),
        min_gap=1,
    ),
    UIPattern(
        name="AskUserQuestion",
        top=(re.compile(r"^\s*[☐✔☒]"),),  # Single-tab: bottom required
        bottom=(re.compile(r"^\s*Enter to select"),),
        min_gap=1,
    ),
    UIPattern(
        name="PermissionPrompt",
        top=(
            re.compile(r"^\s*Do you want to proceed\?"),
            re.compile(r"^\s*Do you want to make this edit"),
            re.compile(r"^\s*Do you want to create \S"),
            re.compile(r"^\s*Do you want to delete \S"),
        ),
        bottom=(re.compile(r"^\s*Esc to cancel"),),
    ),
    UIPattern(
        # Permission menu with numbered choices (no "Esc to cancel" line)
        name="PermissionPrompt",
        top=(re.compile(r"^\s*❯\s*1\.\s*Yes"),),
        bottom=(),
        min_gap=2,
    ),
    UIPattern(
        # Bash command approval
        name="BashApproval",
        top=(
            re.compile(r"^\s*Bash command\s*$"),
            re.compile(r"^\s*This command requires approval"),
        ),
        bottom=(re.compile(r"^\s*Esc to cancel"),),
    ),
    UIPattern(
        name="RestoreCheckpoint",
        top=(re.compile(r"^\s*Restore the code"),),
        bottom=(re.compile(r"^\s*Enter to continue"),),
    ),
    UIPattern(
        name="SwitchModel",
        top=(re.compile(r"^\s*Switch model\?"),),
        bottom=(),
    ),
    UIPattern(
        name="Settings",
        top=(
            re.compile(r"^\s*Settings:.*tab to cycle"),
            re.compile(r"^\s*Select model"),
        ),
        bottom=(
            re.compile(r"Esc to cancel"),
            re.compile(r"Esc to exit"),
            re.compile(r"Enter to confirm"),
            re.compile(r"^\s*Type to filter"),
        ),
    ),
]

# Dialogs the bot answers on its own (startup chores that would otherwise
# leave Claude Code stuck before the SessionStart hook ever fires).
# name -> substring of the option line to select before pressing Enter,
# or None to dismiss with Escape.
AUTO_ANSWER_DIALOGS: dict[str, str | None] = {
    "TrustDialog": "Yes, I trust",
    "BypassWarning": "Yes, I accept",
    "ResumeSession": "full session",
}

# Generic modal fallback: Claude Code's menus/dialogs all end with a hint
# line like "↑/↓ to navigate · Enter to confirm · Esc to cancel". If none
# of the named patterns matched, this catches new/unknown dialogs so the
# bot never types a prompt into a menu.
_RE_MODAL_HINT = re.compile(
    r"(↑/↓ to navigate|↑↓ to navigate|Enter to (confirm|select|continue|submit)"
    r"|Esc to (cancel|back|exit|dismiss|close))"
)
_RE_MENU_CURSOR = re.compile(r"^\s*❯\s*\S")


# ── Post-processing ──────────────────────────────────────────────────────

_RE_LONG_DASH = re.compile(r"^─{5,}$")


def _shorten_separators(text: str) -> str:
    """Replace lines of 5+ ─ characters with exactly ─────."""
    return "\n".join(
        "─────" if _RE_LONG_DASH.match(line) else line for line in text.split("\n")
    )


# ── Core extraction ──────────────────────────────────────────────────────


def _try_extract(lines: list[str], pattern: UIPattern) -> InteractiveUIContent | None:
    """Try to extract content matching a single UI pattern.

    When ``pattern.bottom`` is empty, the region extends from the top marker
    to the last non-empty line (used for multi-tab AskUserQuestion where the
    bottom delimiter varies by tab).
    """
    top_idx: int | None = None
    bottom_idx: int | None = None

    for i, line in enumerate(lines):
        if top_idx is None:
            if any(p.search(line) for p in pattern.top):
                top_idx = i
        elif pattern.bottom and any(p.search(line) for p in pattern.bottom):
            bottom_idx = i
            break

    if top_idx is None:
        return None

    # No bottom patterns → use last non-empty line as boundary
    if not pattern.bottom:
        for i in range(len(lines) - 1, top_idx, -1):
            if lines[i].strip():
                bottom_idx = i
                break

    if bottom_idx is None or bottom_idx - top_idx < pattern.min_gap:
        return None

    content = "\n".join(lines[top_idx : bottom_idx + 1]).rstrip()
    return InteractiveUIContent(content=_shorten_separators(content), name=pattern.name)


# ── Public API ───────────────────────────────────────────────────────────


def extract_interactive_content(pane_text: str) -> InteractiveUIContent | None:
    """Extract content from an interactive UI in terminal output.

    Tries each UI pattern in declaration order; first match wins.
    Returns None if no recognizable interactive UI is found.
    """
    if not pane_text:
        return None

    lines = pane_text.strip().split("\n")
    for pattern in UI_PATTERNS:
        result = _try_extract(lines, pattern)
        if result:
            return result
    return _try_extract_modal(lines)


def _try_extract_modal(lines: list[str]) -> InteractiveUIContent | None:
    """Fallback for unrecognised dialogs.

    Matches when the last non-blank line is a navigation hint *and* a
    ``❯`` menu cursor appears somewhere in the preceding ~15 lines, but the
    pane is not simply showing the idle input box (``❯`` right above the
    footer separator).
    """
    end = len(lines) - 1
    while end >= 0 and not lines[end].strip():
        end -= 1
    if end < 0 or not _RE_MODAL_HINT.search(lines[end]):
        return None
    if is_prompt_ready("\n".join(lines)):
        return None
    start = max(0, end - 15)
    cursor_idx = next(
        (i for i in range(end - 1, start - 1, -1) if _RE_MENU_CURSOR.match(lines[i])),
        None,
    )
    if cursor_idx is None:
        return None
    # Walk up to the dialog's top: first blank line or separator above the cursor
    top = cursor_idx
    for i in range(cursor_idx - 1, start - 1, -1):
        if not lines[i].strip() or _is_chrome_separator(lines[i]):
            break
        top = i
    content = "\n".join(lines[top : end + 1]).rstrip()
    return InteractiveUIContent(content=_shorten_separators(content), name="Modal")


def is_interactive_ui(pane_text: str) -> bool:
    """Check if terminal currently shows an interactive UI."""
    return extract_interactive_content(pane_text) is not None


def is_blocking_dialog(pane_text: str) -> bool:
    """True when a dialog/menu would swallow typed text as navigation keys."""
    ui = extract_interactive_content(pane_text)
    return ui is not None and ui.name not in ("AskUserQuestion",)


def find_menu_option(pane_text: str, needle: str) -> tuple[int, int] | None:
    """Locate a menu option in a dialog.

    Returns ``(cursor_offset, option_offset)`` — the number of lines from
    the dialog's first option to the currently highlighted (``❯``) one and
    to the option whose text contains ``needle``. Pressing Down
    ``option_offset - cursor_offset`` times moves the cursor onto it.
    Returns None when either line can't be found.
    """
    lines = pane_text.split("\n")
    # The dialog is at the bottom of the pane: take the *last* line with the
    # option text, then the nearest ``❯`` within 10 lines of it (a shell
    # prompt echo like "❯ claude" further up must not be mistaken for it).
    target_idx = next(
        (i for i in range(len(lines) - 1, -1, -1) if needle in lines[i]), None
    )
    if target_idx is None:
        return None
    lo, hi = max(0, target_idx - 10), min(len(lines), target_idx + 11)
    cursor_idx = min(
        (i for i in range(lo, hi) if _RE_MENU_CURSOR.match(lines[i])),
        key=lambda i: abs(i - target_idx),
        default=None,
    )
    if cursor_idx is None:
        return None
    return cursor_idx, target_idx


# ── Status line parsing ─────────────────────────────────────────────────

# Spinner characters Claude Code uses in its status line
STATUS_SPINNERS = frozenset(["·", "✻", "✽", "✶", "✳", "✢"])

# Lines that may sit between the status line and the footer separator:
# "⎿  Tip: …" blocks and their wrapped continuation lines (3+ spaces), todo
# HUD rows, and "⏵" hints.
_RE_STATUS_SKIPPABLE = re.compile(r"^\s*[⎿⏵]|^\s{2,}\S")
_STATUS_SCAN_LINES = 16

# Labelled separators such as "──── ultracode ────" (2.1.2xx) still count
# as chrome; require mostly ─ characters and a minimum width.
_RE_SEPARATOR_LABEL = re.compile(r"^─{8,}(\s*[\w /:+.-]{0,40}\s*─{4,})?$")

# "· done 1:47 PM", "· 1 shell still running", elapsed timers etc. change
# every second; strip them so identical statuses dedupe at the send layer.
_RE_STATUS_NOISE = re.compile(
    r"\s*·\s*(done( at)? \d{1,2}:\d{2}(:\d{2})?( ?[AP]M)?|\d+ shells? still running)",
)


def _is_chrome_separator(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 20:
        return False
    if all(c == "─" for c in stripped):
        return True
    return bool(_RE_SEPARATOR_LABEL.match(stripped))


def _find_chrome_separator(lines: list[str], window: int = 10) -> int | None:
    """Index of the topmost separator within the last ``window`` lines."""
    search_start = max(0, len(lines) - window)
    for i in range(search_start, len(lines)):
        if _is_chrome_separator(lines[i]):
            return i
    return None


def parse_status_line(pane_text: str) -> str | None:
    """Extract the Claude Code status line from terminal output.

    The status line (spinner + working text) sits above the chrome
    separator (a full line of ``─``), possibly with a "⎿ Tip:" block or a
    todo HUD in between. We locate the separator first, then scan upward
    past blank and skippable lines — this avoids false positives from
    ``·`` bullets in Claude's regular output.

    Returns the text after the spinner, or None if no status line found.
    """
    if not pane_text:
        return None

    lines = pane_text.split("\n")
    chrome_idx = _find_chrome_separator(lines)
    if chrome_idx is None:
        return None  # No chrome visible — can't determine status

    for i in range(chrome_idx - 1, max(chrome_idx - _STATUS_SCAN_LINES - 1, -1), -1):
        raw = lines[i]
        line = raw.strip()
        if not line:
            continue
        if line[0] in STATUS_SPINNERS:
            return _RE_STATUS_NOISE.sub("", line[1:]).strip()
        if _RE_STATUS_SKIPPABLE.match(raw):
            continue
        # First real content line above the separator isn't a spinner → no status
        return None
    return None


def is_prompt_ready(pane_text: str) -> bool:
    """True when Claude Code shows its idle input box.

    Layout::

        ────────────────  (separator)
        ❯ …               (input line, possibly with a ghost suggestion)
        ────────────────  (separator)
          ⏵⏵ accept edits on …   (footer)
    """
    if not pane_text:
        return False
    lines = pane_text.rstrip().split("\n")
    # Scan the last 8 lines for "separator, ❯ line(s), separator"
    tail = lines[-8:]
    for i in range(len(tail) - 2):
        if not _is_chrome_separator(tail[i]):
            continue
        if not tail[i + 1].lstrip().startswith("❯"):
            continue
        for j in range(i + 2, min(i + 5, len(tail))):
            if _is_chrome_separator(tail[j]):
                return True
    return False


def is_working(pane_text: str) -> bool:
    """True while Claude is busy (footer shows "esc to interrupt")."""
    return "esc to interrupt" in pane_text[-600:].lower()


# Footer text → permission mode as used by `claude --permission-mode`.
_PERMISSION_MODE_FOOTER: tuple[tuple[str, str], ...] = (
    ("bypass permissions on", "bypassPermissions"),
    ("accept edits on", "acceptEdits"),
    ("plan mode on", "plan"),
    ("auto mode on", "auto"),
    ("don't ask on", "dontAsk"),
    ("manual mode on", "default"),
)


def parse_permission_mode(pane_text: str) -> str | None:
    """Return the permission mode shown in the footer, or None.

    Footer examples: ``⏵⏵ bypass permissions on (shift+tab to cycle)``,
    ``⏸ plan mode on``, ``⏵⏵ accept edits on · 1 shell · esc to interrupt``.
    The footer is the last few lines, so only those are inspected.
    """
    if not pane_text:
        return None
    tail = "\n".join(pane_text.rstrip().split("\n")[-4:]).lower()
    for needle, mode in _PERMISSION_MODE_FOOTER:
        if needle in tail:
            return mode
    return None


_RE_UPDATE_PENDING = re.compile(r"Update installed\s*·\s*Restart to update")


def has_update_pending(pane_text: str) -> bool:
    """True when Claude Code shows "✔ Update installed · Restart to update"."""
    return bool(pane_text and _RE_UPDATE_PENDING.search(pane_text))


# ── Pane chrome stripping & bash output extraction ─────────────────────


def strip_pane_chrome(lines: list[str]) -> list[str]:
    """Strip Claude Code's bottom chrome (prompt area + status bar).

    The bottom of the pane looks like::

        ────────────────────────  (separator)
        ❯                        (prompt)
        ────────────────────────  (separator)
          [Opus 4.6] Context: 34%
          ⏵⏵ bypass permissions…

    This function finds the topmost ``────`` separator in the last 10 lines
    and strips everything from there down.
    """
    idx = _find_chrome_separator(lines)
    return lines[:idx] if idx is not None else lines


def extract_bash_output(pane_text: str, command: str) -> str | None:
    """Extract ``!`` command output from a captured tmux pane.

    Searches from the bottom for the ``! <command>`` echo line, then
    returns that line and everything below it (including the ``⎿`` output).
    Returns *None* if the command echo wasn't found.
    """
    lines = strip_pane_chrome(pane_text.splitlines())

    # Find the last "! <command>" echo line (search from bottom).
    # Match on the first 10 chars of the command in case the line is truncated.
    cmd_idx: int | None = None
    match_prefix = command[:10]
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped.startswith(f"! {match_prefix}") or stripped.startswith(
            f"!{match_prefix}"
        ):
            cmd_idx = i
            break

    if cmd_idx is None:
        return None

    # Include the command echo line and everything after it
    raw_output = lines[cmd_idx:]

    # Strip trailing empty lines
    while raw_output and not raw_output[-1].strip():
        raw_output.pop()

    if not raw_output:
        return None

    return "\n".join(raw_output).strip()


# ── Usage modal parsing ──────────────────────────────────────────────────────────


@dataclass
class UsageInfo:
    """Parsed output from Claude Code's /usage modal."""

    raw_text: str  # Full captured pane text
    parsed_lines: list[str]  # Cleaned content lines from the modal


def parse_usage_output(pane_text: str) -> UsageInfo | None:
    """Extract usage information from Claude Code's /usage settings tab.

    The /usage modal shows a Settings overlay with a "Usage" tab containing
    progress bars and reset times.  This parser looks for the Settings header
    line, then collects all content until "Esc to cancel".

    Returns UsageInfo with cleaned lines, or None if not detected.
    """
    if not pane_text:
        return None

    lines = pane_text.strip().split("\n")

    # Find the Settings header that indicates we're in the usage modal
    start_idx: int | None = None
    end_idx: int | None = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if start_idx is None:
            # The usage tab header line
            if "Settings:" in stripped and "Usage" in stripped:
                start_idx = i + 1  # skip the header itself
        else:
            if stripped.startswith("Esc to"):
                end_idx = i
                break

    if start_idx is None:
        return None
    if end_idx is None:
        end_idx = len(lines)

    # Collect content lines, stripping progress bar characters and whitespace
    cleaned: list[str] = []
    for line in lines[start_idx:end_idx]:
        # Strip the line but preserve meaningful content
        stripped = line.strip()
        if not stripped:
            continue
        # Remove progress bar block characters but keep the rest
        # Progress bars are like: █████▋   38% used
        # Strip leading block chars, keep the percentage
        stripped = re.sub(r"^[\u2580-\u259f\s]+", "", stripped).strip()
        if stripped:
            cleaned.append(stripped)

    if cleaned:
        return UsageInfo(raw_text=pane_text, parsed_lines=cleaned)

    return None
