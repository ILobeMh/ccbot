"""Structured view of Claude Code's interactive terminal UIs (tap-to-answer).

Parses the region extracted by terminal_parser (AskUserQuestion, permission
prompts, plan approval, pickers) into a title, question, tabs and numbered
options, and renders it as a clean message with one button per option.

Claude Code selects a numbered option when its digit is typed (verified on
2.1.285: a digit answers a single-select question, toggles a multi-select
checkbox, approves / denies a permission prompt). Each option button carries
a short hash of the option's label; before typing the digit the caller
re-parses the pane and checks the hash, so a tap on an outdated message can
never answer a different question.

Key components: Choice, ChoiceView, parse_choices(), render_view(),
button_text() / button_style(), label_hash(), escape_literal().
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .terminal_parser import InteractiveUIContent

_OPTION_RE = re.compile(r"^(\s*)(?:❯\s*)?(\d+)\.\s+(.*\S)\s*$")
_CHECK_RE = re.compile(r"^\[([ ✔✓x×])\]\s*(.*)$")
_TABS_RE = re.compile(r"^\s*←\s+(.*?)\s*→\s*$")
_TAB_ITEM_RE = re.compile(r"([☐☒✔])\s+(.+?)(?=\s{2,}[☐☒✔]|$)")
_SINGLE_HEADER_RE = re.compile(r"^\s*([☐☒✔])\s+(\S.*)$")
_SEPARATOR_RE = re.compile(r"^\s*[─╌━▔▁]{5,}\s*$")
_FOOTER_RE = re.compile(
    r"(Enter to (select|confirm|continue|submit)|to navigate|Esc to (cancel|exit)"
    r"|Tab to amend|ctrl[-+]g to edit|Type to filter)"
)
_REVIEW_RE = re.compile(r"^\s*●\s+(.*)$")
_ARROW_ANSWER_RE = re.compile(r"^\s*→\s+(.*)$")

# UIs whose option continuation lines are descriptions (AskUserQuestion) vs
# wrapped labels (menus: permission prompts, plan approval, pickers)
_DESCRIBED = {"AskUserQuestion"}
# UIs where typing a digit is known to act on the option directly
DIGIT_UIS = {
    "AskUserQuestion",
    "PermissionPrompt",
    "BashApproval",
    "ExitPlanMode",
    "Settings",
    "Modal",
    "RestoreCheckpoint",
    "SwitchModel",
}


@dataclass
class Choice:
    number: int
    label: str
    description: str = ""
    checked: bool | None = None  # multi-select state; None = single-select
    kind: str = "option"  # "option" | "other" (type your own) | "chat"


@dataclass
class ChoiceView:
    ui_name: str
    title: str = ""
    question: str = ""
    context: list[str] = field(default_factory=list)
    tabs: list[tuple[str, str]] = field(default_factory=list)  # (state, label)
    choices: list[Choice] = field(default_factory=list)
    review: list[tuple[str, str]] = field(default_factory=list)  # (q, answer)

    @property
    def multi(self) -> bool:
        return any(c.checked is not None for c in self.choices)

    @property
    def signature(self) -> str:
        """Identity of what's on screen (question + tab states), for staleness."""
        tabs = "|".join(f"{s}{t}" for s, t in self.tabs)
        return label_hash(f"{self.title}\n{self.question}\n{tabs}")


def label_hash(text: str) -> str:
    return hashlib.sha1(text.strip().encode()).hexdigest()[:4]


def _kind(label: str) -> str:
    low = label.lower().rstrip(".")
    if low == "type something":
        return "other"
    if low == "chat about this":
        return "chat"
    return "option"


_FRIENDLY_TITLE = {
    "PermissionPrompt": "Permission needed",
    "BashApproval": "Permission needed",
    "Settings": "Settings",
    "Modal": "Dialog",
    "SwitchModel": "Switch model?",
    "RestoreCheckpoint": "Restore checkpoint",
    "AskUserQuestion": "Question",
}
_COLUMNS_RE = re.compile(r"^(.*?\S)\s{2,}(\S.*)$")


def _dialog_header(pane_text: str, content: str) -> list[str]:
    """Lines of the dialog box above ``content`` (up to its top border).

    Permission prompts are matched on "Do you want to …?", but the box starts
    higher up with what is being asked for ("Bash command", the command and
    its description; "Create file", the file name and a preview).
    """
    first = next((ln for ln in content.split("\n") if ln.strip()), "")
    lines = pane_text.split("\n")
    idx = next((i for i in range(len(lines) - 1, -1, -1) if lines[i] == first), None)
    if idx is None:
        return []
    header: list[str] = []
    for ln in reversed(lines[max(0, idx - 30) : idx]):
        if re.match(r"^\s*─{5,}\s*$", ln):  # the box's top border
            break
        header.append(ln)
    return list(reversed(header))


def parse_choices(
    ui: InteractiveUIContent,
    pane_text: str | None = None,
    questions: list[dict] | None = None,
) -> ChoiceView | None:
    """Structured view of ``ui``; None when it has no numbered options.

    ``pane_text`` (the whole capture) lets permission prompts recover the
    header above their question; ``questions`` (the AskUserQuestion tool
    input) names the active question when the tab strip can't tell.
    """
    view = ChoiceView(ui_name=ui.name)
    lines = ui.content.split("\n")
    if pane_text and ui.name in ("PermissionPrompt", "BashApproval"):
        lines = _dialog_header(pane_text, ui.content) + lines
    first_opt = next((i for i, ln in enumerate(lines) if _OPTION_RE.match(ln)), None)
    if first_opt is None:
        return None

    # -- above the options: tabs / header, context paragraphs, the question
    paragraphs: list[list[str]] = [[]]
    for ln in lines[:first_opt]:
        m = _TABS_RE.match(ln)
        if m and any(ch in m.group(1) for ch in "☐☒✔"):
            view.tabs = [
                ({"☐": "pending", "☒": "answered", "✔": "submit"}[s], label.strip())
                for s, label in _TAB_ITEM_RE.findall(m.group(1))
            ]
            continue
        m = _SINGLE_HEADER_RE.match(ln)
        if m and not view.title and not view.tabs:
            view.title = m.group(2).strip()
            continue
        if _SEPARATOR_RE.match(ln) or not ln.strip():
            if paragraphs[-1]:
                paragraphs.append([])
            continue
        paragraphs[-1].append(ln.strip())
    paragraphs = [p for p in paragraphs if p]

    if ui.name == "AskUserQuestion" and view.tabs:
        # Submit tab: "Review your answers" + "● question / → answer" pairs
        flat = [ln for p in paragraphs for ln in p]
        if flat and flat[0].startswith("Review your answers"):
            view.title = "Review your answers"
            pending_q = ""
            for ln in flat[1:]:
                if m := _REVIEW_RE.match(ln):
                    pending_q = m.group(1)
                elif (m := _ARROW_ANSWER_RE.match(ln)) and pending_q:
                    view.review.append((pending_q, m.group(1)))
                    pending_q = ""
                elif not ln.startswith("You have not"):
                    view.question = ln
            paragraphs = []
        else:
            active = next((t for s, t in view.tabs if s == "pending"), "")
            view.title = active or view.title

    if paragraphs:
        view.question = " ".join(paragraphs[-1])
        context = [ln for p in paragraphs[:-1] for ln in p]
        if ui.name == "ExitPlanMode":
            view.title = "Plan ready"
            context = []  # the terminal only shows a clipped plan preview
        elif not view.title and context:
            view.title, context = context[0], context[1:]
        elif (
            not view.title and len(paragraphs[-1]) > 1 and ui.name != "AskUserQuestion"
        ):
            # "Select model" + explanation lines: first line is the title
            view.title = paragraphs[-1][0]
            view.question = " ".join(paragraphs[-1][1:])
        view.context = context

    if ui.name == "AskUserQuestion" and questions and view.question:
        wanted = " ".join(view.question.split())
        for q in questions:
            if (
                isinstance(q, dict)
                and " ".join(str(q.get("question", "")).split()) == wanted
            ):
                view.title = str(q.get("header") or view.title)
                break
    view.title = view.title or _FRIENDLY_TITLE.get(ui.name, "")

    # -- the options (continuation lines: description or wrapped label)
    described = ui.name in _DESCRIBED
    current: Choice | None = None
    for ln in lines[first_opt:]:
        if _FOOTER_RE.search(ln) and not _OPTION_RE.match(ln):
            break
        if _SEPARATOR_RE.match(ln):
            continue
        if not ln.strip():
            current = None  # a blank line ends an option's continuation
            continue
        m = _OPTION_RE.match(ln)
        if m:
            label = m.group(3)
            checked: bool | None = None
            if cm := _CHECK_RE.match(label):
                checked = cm.group(1) != " "
                label = cm.group(2)
            description = ""
            if cols := _COLUMNS_RE.match(label):  # picker: "Name     what it is"
                label, description = cols.group(1), cols.group(2)
            current = Choice(int(m.group(2)), label, description, checked=checked)
            current.kind = _kind(label)
            view.choices.append(current)
            continue
        if current is None:
            continue
        text = ln.strip()
        if text.lower() == "submit":  # multi-select's own submit row
            continue
        if "shift+tab" in text and "(" not in text:
            continue  # "shift+tab to approve with this feedback" hint
        if described or current.description:
            current.description = f"{current.description} {text}".strip()
        else:
            current.label = f"{current.label} {text}"
    return view if view.choices else None


# ── rendering ───────────────────────────────────────────────────────────

_TITLE_ICON = {
    "AskUserQuestion": "❓",
    "PermissionPrompt": "🔐",
    "BashApproval": "🔐",
    "ExitPlanMode": "📋",
    "RestoreCheckpoint": "⏪",
}


def _code(text: str, lang: str = "") -> str:
    longest = max((len(r) for r in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{text}\n{fence}"


def escape_literal(text: str) -> str:
    """Markdown-safe literal text (both classic and rich renderers)."""
    return re.sub(r"([\\`*_\[\]{}()#+\-.!|~=$<>&])", r"\\\1", text)


def render_view(view: ChoiceView) -> str:
    """Markdown for the question message (options are also buttons)."""
    icon = _TITLE_ICON.get(view.ui_name, "❔")
    out: list[str] = []
    title = view.title or _FRIENDLY_TITLE.get(view.ui_name, view.ui_name)
    head = f"{icon} **{escape_literal(title)}**"
    if view.tabs:
        strip = " · ".join(
            f"{'☒' if s == 'answered' else '☐' if s == 'pending' else '✔'} {escape_literal(t)}"
            for s, t in view.tabs
        )
        head += f"\n_{strip}_"
    out.append(head)
    if view.context:
        if view.ui_name in ("PermissionPrompt", "BashApproval"):
            out.append(_code("\n".join(view.context)))
        else:
            out.append(escape_literal("\n".join(view.context)))
    if view.review:
        out.append(
            "\n".join(
                f"• {escape_literal(q)} → **{escape_literal(a)}**"
                for q, a in view.review
            )
        )
    if view.question:
        suffix = " _(choose any, then Next ›)_" if view.multi else ""
        out.append(escape_literal(view.question) + suffix)
    lines = []
    for c in view.choices:
        if c.kind != "option":
            continue
        mark = (
            ("☑" if c.checked else "☐") + " "
            if c.checked is not None
            else f"**{c.number}.** "
        )
        desc = f" — _{escape_literal(c.description)}_" if c.description else ""
        lines.append(f"{mark}{escape_literal(c.label)}{desc}")
    if lines:
        out.append("\n".join(lines))
    return "\n\n".join(out)


def button_text(c: Choice, max_len: int = 30) -> str:
    if c.kind == "other":
        return "✍️ Other…"
    if c.kind == "chat":
        return "💬 Chat about this"
    label = c.label if len(c.label) <= max_len else c.label[: max_len - 1] + "…"
    if c.checked is not None:
        return f"{'☑' if c.checked else '☐'} {label}"
    return f"{c.number} · {label}"


def button_style(c: Choice) -> str | None:
    low = c.label.lower()
    if low.startswith("yes"):
        return "success"
    if low.startswith("no") or low == "cancel":
        return "danger"
    return None
