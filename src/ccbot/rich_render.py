"""Rich-message rendering of Claude Code output (Telegram Bot API 10.1+).

Turns transcript entries (text, thinking, tool calls, local commands,
notices) into Telegram *Rich Markdown* — GitHub-flavoured markdown sent with
``sendRichMessage`` / ``editMessageText(rich_message=…)``. A rich message
holds up to 32768 characters and renders headings, tables, fenced code with
a language and collapsible ``<details>`` blocks, so nothing is truncated
here: content over the budget is split across several messages instead.

Pure functions only (no Telegram I/O); the queue decides how to send.

Key components:
  - RenderOptions: display knobs (from settings)
  - render_text / render_thinking / render_user_text / render_notice /
    render_local_command / render_tool_use / render_tool_result:
    entry → list of message texts (first one first)
  - escape_prose(): neutralise Rich-Markdown-only syntax ($math$, ==mark==,
    ||spoiler||, raw HTML) in Claude's prose, leaving code untouched
  - split_rich(): fence-aware splitting within the size / block budgets
  - is_rtl(): base direction for Persian / Arabic / Hebrew replies
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

from .transcript_parser import ToolCall

# Telegram allows 32768 characters and 500 blocks per rich message; stay
# clear of both (escapes and our own chrome add a little on top).
RICH_CHAR_BUDGET = 30000
RICH_LINE_BUDGET = 400  # non-empty prose lines ≈ upper bound on blocks
# Long commands are folded into a <details> block past this many lines
COMMAND_INLINE_LINES = 25


@dataclass(frozen=True)
class RenderOptions:
    """How much of each kind of output to show (see settings.py)."""

    # Tool output: "full" (collapsed block with everything), "preview" (first
    # ``preview_lines`` lines inline) or "summary" (status line only)
    tool_output: str = "full"
    preview_lines: int = 15
    expand_output: bool = False  # open the output <details> by default
    thinking_max_chars: int = 0  # 0 = all of it


# ── escaping ────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_INLINE_CODE_RE = re.compile(r"(`+)(.+?)(?<!`)\1(?!`)", re.DOTALL)
_ASCII_PUNCT_RE = re.compile(r"([\\`*_\[\]{}()#+\-.!|~=$<>&])")


def escape_inline(text: str) -> str:
    """Literal text: every markdown-significant character escaped."""
    return _ASCII_PUNCT_RE.sub(r"\\\1", text)


def _escape_prose_segment(text: str) -> str:
    """Escape Rich-only syntax in a code-free stretch of GFM text."""
    text = text.replace("&", "&amp;").replace("<", "&lt;")
    text = text.replace("$", "\\$").replace("==", "\\=\\=")
    lines = []
    for line in text.split("\n"):
        if "||" in line and not line.lstrip().startswith("|"):
            line = line.replace("||", "\\|\\|")
        lines.append(line)
    return "\n".join(lines)


def _escape_outside_inline_code(line_block: str) -> str:
    out: list[str] = []
    pos = 0
    for m in _INLINE_CODE_RE.finditer(line_block):
        out.append(_escape_prose_segment(line_block[pos : m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(_escape_prose_segment(line_block[pos:]))
    return "".join(out)


def escape_prose(text: str) -> str:
    """Make Claude's markdown safe for Rich Markdown, code left untouched.

    Rich Markdown extends GFM with ``$math$``, ``==marked==``,
    ``||spoiler||`` and arbitrary HTML; Claude writes plain GFM, where a
    ``$PATH`` or ``a < b`` must stay literal.
    """
    out: list[str] = []
    prose: list[str] = []
    fence: str | None = None

    def flush_prose() -> None:
        if prose:
            out.append(_escape_outside_inline_code("\n".join(prose)))
            prose.clear()

    for line in text.split("\n"):
        m = _FENCE_RE.match(line)
        if fence is None:
            if m:
                flush_prose()
                fence = m.group(1)
                out.append(line)
            else:
                prose.append(line)
        else:
            out.append(line)
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                if not line.strip()[len(m.group(1)) :].strip():
                    fence = None
    flush_prose()
    return "\n".join(out)


def code_block(content: str, lang: str = "") -> str:
    """Fenced code block whose fence can't be closed by the content."""
    longest = max((len(r) for r in re.findall(r"`+", content)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{content.rstrip(chr(10))}\n{fence}"


def inline_code(text: str) -> str:
    longest = max((len(r) for r in re.findall(r"`+", text)), default=0)
    ticks = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{ticks}{pad}{text}{pad}{ticks}"


def details(summary: str, body: str, *, open_: bool = False) -> str:
    """Collapsible block; ``summary`` is inline markdown, ``body`` markdown."""
    attr = " open" if open_ else ""
    return f"<details{attr}><summary>{summary}</summary>\n\n{body}\n\n</details>"


# ── splitting ───────────────────────────────────────────────────────────


_DETAILS_OPEN_RE = re.compile(r"^<details( open)?><summary>.*</summary>$")
_DETAILS_CLOSE = "</details>"


def split_rich(
    text: str,
    max_chars: int = RICH_CHAR_BUDGET,
    max_lines: int = RICH_LINE_BUDGET,
) -> list[str]:
    """Split markdown on line boundaries within both budgets.

    Blocks cut in two — fenced code and our ``<details>`` blocks (opening
    line ``<details><summary>…</summary>``, closing line ``</details>``) —
    are closed at the end of one chunk and re-opened at the start of the
    next, so every chunk renders on its own.
    """
    if len(text) <= max_chars and text.count("\n") < max_lines:
        return [text]
    chunks: list[str] = []
    stack: list[str] = []  # open block lines: "<details…>" or "```lang"
    cur: list[str] = []
    cur_len = 0
    cur_lines = 0

    def closers() -> list[str]:
        out = []
        for opener in reversed(stack):
            if opener.startswith("<details"):
                out += ["", _DETAILS_CLOSE]
            else:
                out.append(_fence_marker(opener))
        return out

    def openers() -> list[str]:
        out = []
        for opener in stack:
            out += [opener, ""] if opener.startswith("<details") else [opener]
        return out

    def flush() -> None:
        nonlocal cur, cur_len, cur_lines
        body = "\n".join(cur + closers()).strip("\n")
        if body.strip():
            chunks.append(body)
        cur = openers()
        cur_len = sum(len(x) + 1 for x in cur)
        cur_lines = 0

    in_fence = lambda: bool(stack) and not stack[-1].startswith("<details")  # noqa: E731

    for line in text.split("\n"):
        pieces = [line]
        if len(line) > max_chars // 2:  # a single enormous line
            step = max_chars // 2
            pieces = [line[i : i + step] for i in range(0, len(line), step)]
        for piece in pieces:
            reserve = sum(len(x) + 1 for x in closers())
            if cur and (
                cur_len + len(piece) + 1 + reserve > max_chars
                or cur_lines + 1 > max_lines
            ):
                flush()
            cur.append(piece)
            cur_len += len(piece) + 1
            if piece.strip():
                cur_lines += 1
        # track what this line opened / closed
        m = _FENCE_RE.match(line)
        if in_fence():
            marker = _fence_marker(stack[-1])
            if m and m.group(1)[0] == marker[0] and len(m.group(1)) >= len(marker):
                if not line.strip()[len(m.group(1)) :].strip():
                    stack.pop()
        elif m:
            stack.append(line.strip())
        elif _DETAILS_OPEN_RE.match(line.strip()):
            stack.append(line.strip())
        elif line.strip() == _DETAILS_CLOSE and stack:
            stack.pop()
    body = "\n".join(cur).strip("\n")
    if body.strip() and body.strip() not in {o.strip() for o in openers()}:
        chunks.append(body)
    return chunks


def _fence_marker(fence_line: str) -> str:
    m = _FENCE_RE.match(fence_line)
    return m.group(1) if m else "```"


def _numbered(chunks: list[str], label: str) -> list[str]:
    if len(chunks) <= 1:
        return chunks
    n = len(chunks)
    return [
        chunk if i == 0 else f"_{label} ({i + 1}/{n})_\n\n{chunk}"
        for i, chunk in enumerate(chunks)
    ]


# ── direction ───────────────────────────────────────────────────────────

_RTL_RE = re.compile(r"[֐-ࣿיִ-﷿ﹰ-﻿]")
_LTR_RE = re.compile(r"[A-Za-zÀ-ɏͰ-ϿЀ-ӿ]")
_CODE_RE = re.compile(r"(`{3,}|~{3,})[\s\S]*?\1|`[^`\n]*`")


def is_rtl(markdown: str) -> bool:
    """True when the prose (code ignored) is mostly right-to-left script."""
    prose = _CODE_RE.sub(" ", markdown)
    rtl = len(_RTL_RE.findall(prose))
    return rtl > 0 and rtl >= len(_LTR_RE.findall(prose))


# ── small helpers ───────────────────────────────────────────────────────


def _lines(text: str) -> int:
    text = text.rstrip("\n")
    return text.count("\n") + 1 if text else 0


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if minutes else f"{hours}h"


def _duration(call: ToolCall) -> str | None:
    start, end = _parse_ts(call.started_at), _parse_ts(call.finished_at)
    if not start or not end:
        return None
    secs = (end - start).total_seconds()
    return format_duration(secs) if secs >= 1 else None


_LANG_BY_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".swift": "swift",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
    ".xml": "xml",
    ".dockerfile": "dockerfile",
}


def _lang_for(path: str) -> str:
    p = PurePosixPath(path)
    if p.name.lower() == "dockerfile":
        return "dockerfile"
    return _LANG_BY_SUFFIX.get(p.suffix.lower(), "")


def _output_block(
    content: str, label: str, opts: RenderOptions, lang: str = "text"
) -> str:
    """Tool output per ``opts.tool_output`` ("" when nothing is shown)."""
    content = content.rstrip("\n")
    if not content or opts.tool_output == "summary":
        return ""
    n = _lines(content)
    if opts.tool_output == "preview":
        shown = content.split("\n")[: opts.preview_lines]
        block = code_block("\n".join(shown), lang)
        more = n - len(shown)
        return block + (f"\n_… {_plural(more, 'more line')}_" if more > 0 else "")
    return details(
        f"{label} · {_plural(n, 'line')}",
        code_block(content, lang),
        open_=opts.expand_output,
    )


# ── entry renderers ─────────────────────────────────────────────────────


def render_text(text: str) -> list[str]:
    """Claude's reply text (GFM) — tables, headings, code all render natively."""
    return split_rich(escape_prose(text.strip()))


def render_thinking(text: str, opts: RenderOptions) -> list[str]:
    """Thinking as a collapsed block whose summary previews the first line."""
    body = text.strip()
    if opts.thinking_max_chars and len(body) > opts.thinking_max_chars:
        body = body[: opts.thinking_max_chars].rstrip() + "\n\n_… (truncated)_"
    first = next((ln.strip() for ln in body.split("\n") if ln.strip()), "")
    preview = first if len(first) <= 80 else first[:79].rstrip() + "…"
    budget = RICH_CHAR_BUDGET - 200
    chunks = split_rich(escape_prose(body), budget)
    n = len(chunks)
    out = []
    for i, chunk in enumerate(chunks, 1):
        label = escape_inline(preview) or "Thinking"
        if n > 1:
            label = f"{label} ({i}/{n})"
        out.append(details(f"💭 {label}", chunk))
    return out


def render_user_text(text: str) -> list[str]:
    return split_rich("👤 " + escape_prose(text.strip()))


def render_notice(text: str) -> list[str]:
    """Errors / warnings / info lines (already carry their emoji)."""
    return split_rich(escape_prose(text.strip()))


def render_local_command(command: str, output: str) -> list[str]:
    head = f"❯ {inline_code(command)}" if command else "❯"
    output = output.strip("\n")
    if not output:
        return [head]
    if "\n" not in output and len(output) < 200:
        return [f"{head}\n{escape_prose(output)}"]
    return _numbered(split_rich(f"{head}\n{code_block(output, 'text')}"), "output")


# ── tools ───────────────────────────────────────────────────────────────


def _s(inp: dict[str, Any], key: str) -> str:
    v = inp.get(key)
    return v if isinstance(v, str) else ""


def _mcp_parts(name: str) -> tuple[str, str] | None:
    if not name.startswith("mcp__"):
        return None
    _, _, rest = name.partition("mcp__")
    server, _, tool = rest.partition("__")
    return server, tool or server


def _json_block(data: Any) -> str:
    try:
        text = json.dumps(data, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        text = str(data)
    return code_block(text, "json")


def _todo_list(todos: Any) -> str:
    if not isinstance(todos, list):
        return ""
    lines = []
    for t in todos:
        if not isinstance(t, dict):
            continue
        content = escape_inline(str(t.get("content", "")))
        status = t.get("status")
        if status == "completed":
            lines.append(f"- [x] ~~{content}~~")
        elif status == "in_progress":
            lines.append(f"- [ ] **{content}** ⏳")
        else:
            lines.append(f"- [ ] {content}")
    return "\n".join(lines)


def tool_header(call: ToolCall) -> str:
    """The tool_use message: what is being run, in full."""
    inp, name = call.input, call.name
    if name == "Bash":
        desc = _s(inp, "description")
        cmd = _s(inp, "command")
        title = f"⚙️ **{escape_inline(desc)}**" if desc else "⚙️ **Bash**"
        if inp.get("run_in_background"):
            title += " · background"
        n = _lines(cmd)
        if n > COMMAND_INLINE_LINES:
            first = cmd.split("\n", 1)[0]
            first = first if len(first) <= 60 else first[:59] + "…"
            body = details(
                f"{inline_code(first)} · {_plural(n, 'line')}", code_block(cmd, "bash")
            )
        else:
            body = code_block(cmd, "bash")
        return f"{title}\n{body}"
    if name == "Read":
        path = _s(inp, "file_path")
        extra = ""
        if isinstance(inp.get("offset"), int) or isinstance(inp.get("limit"), int):
            start = int(inp.get("offset") or 1)
            limit = inp.get("limit")
            extra = f" · lines {start}–{start + int(limit) - 1}" if limit else ""
        return f"📖 **Read** {inline_code(path)}{extra}"
    if name in ("Edit", "MultiEdit"):
        path = _s(inp, "file_path")
        extra = " · all occurrences" if inp.get("replace_all") else ""
        return f"✏️ **Edit** {inline_code(path)}{extra}"
    if name == "NotebookEdit":
        return f"✏️ **NotebookEdit** {inline_code(_s(inp, 'notebook_path'))}"
    if name == "Write":
        return f"📝 **Write** {inline_code(_s(inp, 'file_path'))}"
    if name in ("Grep", "Glob"):
        icon = "🔎" if name == "Grep" else "🗂"
        head = f"{icon} **{name}** {inline_code(_s(inp, 'pattern'))}"
        if _s(inp, "path"):
            head += f" in {inline_code(_s(inp, 'path'))}"
        if _s(inp, "glob"):
            head += f" · {inline_code(_s(inp, 'glob'))}"
        return head
    if name == "WebFetch":
        head = f"🌐 **Fetch** {escape_inline(_s(inp, 'url'))}"
        prompt = _s(inp, "prompt")
        return head + ("\n" + details("Prompt", escape_prose(prompt)) if prompt else "")
    if name == "WebSearch":
        return f"🔍 **Search** {escape_inline(_s(inp, 'query'))}"
    if name in ("Agent", "Task"):
        desc = _s(inp, "description") or "sub-agent"
        who = ", ".join(x for x in (_s(inp, "subagent_type"), _s(inp, "model")) if x)
        head = f"🤖 **Agent** · {escape_inline(desc)}"
        if who:
            head += f" ({escape_inline(who)})"
        if inp.get("run_in_background"):
            head += " · background"
        prompt = _s(inp, "prompt")
        return head + ("\n" + details("Prompt", escape_prose(prompt)) if prompt else "")
    if name == "TodoWrite":
        todos = _todo_list(inp.get("todos"))
        return "📋 **Todos**" + (f"\n{todos}" if todos else "")
    if name == "TaskCreate":
        head = f"📋 **New task** · {escape_inline(_s(inp, 'subject'))}"
        desc = _s(inp, "description")
        return head + ("\n" + details("Details", escape_prose(desc)) if desc else "")
    if name == "TaskUpdate":
        status = _s(inp, "status")
        task = escape_inline(str(inp.get("taskId", "")))
        return f"📋 **Task** {task}" + (f" → {escape_inline(status)}" if status else "")
    if name == "Skill":
        head = f"🧩 **Skill** {inline_code(_s(inp, 'skill'))}"
        args = _s(inp, "args")
        return head + (f" · {escape_inline(args)}" if args else "")
    if name == "AskUserQuestion":
        return _questions(inp)
    mcp = _mcp_parts(name)
    if mcp:
        server, tool = mcp
        head = f"🔌 **{escape_inline(server)}** · {escape_inline(tool)}"
    else:
        head = f"🔧 **{escape_inline(name)}**"
    return head + ("\n" + details("Arguments", _json_block(inp)) if inp else "")


def _questions(inp: dict[str, Any]) -> str:
    """AskUserQuestion as a readable list (fallback when no UI is drawn)."""
    out = []
    for q in inp.get("questions") or []:
        if not isinstance(q, dict):
            continue
        header = _s(q, "header")
        title = f"❓ **{escape_inline(header)}**\n" if header else "❓ "
        out.append(title + escape_prose(_s(q, "question")))
        for i, opt in enumerate(q.get("options") or [], 1):
            if isinstance(opt, dict):
                label = escape_inline(str(opt.get("label", "")))
                desc = str(opt.get("description", ""))
                out.append(
                    f"{i}. **{label}**" + (f" — {escape_prose(desc)}" if desc else "")
                )
    return "\n".join(out) or "❓ **Question**"


def render_tool_use(call: ToolCall) -> list[str]:
    return split_rich(tool_header(call))


def _edit_diff(call: ToolCall) -> str:
    meta = call.result_meta or {}
    patch = meta.get("structuredPatch")
    if isinstance(patch, list) and patch:
        lines: list[str] = []
        for hunk in patch:
            if not isinstance(hunk, dict):
                continue
            lines.append(
                f"@@ -{hunk.get('oldStart')},{hunk.get('oldLines')} "
                f"+{hunk.get('newStart')},{hunk.get('newLines')} @@"
            )
            lines.extend(str(ln) for ln in hunk.get("lines") or [])
        return "\n".join(lines)
    old, new = _s(call.input, "old_string"), _s(call.input, "new_string")
    if not old and not new:
        return ""
    diff = difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="")
    return "\n".join(ln for ln in diff if not ln.startswith(("---", "+++")))


def _status(icon: str, *bits: str | None) -> str:
    return "⎿ " + icon + " " + " · ".join(b for b in bits if b)


def tool_result_body(call: ToolCall, opts: RenderOptions) -> str | None:
    """What the result adds below the header (None: nothing worth adding)."""
    took = _duration(call)
    text = (call.result_text or "").rstrip("\n")
    meta = call.result_meta or {}
    name = call.name

    if call.interrupted:
        return _status("⏹", "Interrupted", took)
    if call.is_error:
        first = text.split("\n", 1)[0] if text else "Error"
        first = first if len(first) <= 200 else first[:199] + "…"
        body = _status("❌", escape_inline(first), took)
        if "\n" in text:
            body += "\n" + _output_block(text, "Error", opts)
        return body

    if name == "Bash":
        if meta.get("backgroundTaskId"):
            return _status("🌙", "Running in background", took)
        stdout = meta.get("stdout") if isinstance(meta.get("stdout"), str) else text
        stderr = meta.get("stderr") if isinstance(meta.get("stderr"), str) else ""
        n = _lines(stdout or "")
        parts = [_status("✅", _plural(n, "line") if n else "no output", took)]
        for block in (
            _output_block(stdout or "", "Output", opts),
            _output_block(stderr or "", "stderr", opts),
        ):
            if block:
                parts.append(block)
        return "\n".join(parts)
    if name == "Read":
        return _status("✅", _plural(_lines(text), "line"), took)
    if name in ("Edit", "MultiEdit", "NotebookEdit"):
        diff = _edit_diff(call)
        if not diff:
            return _status("✅", "Edited", took)
        added = sum(1 for ln in diff.split("\n") if ln.startswith("+"))
        removed = sum(1 for ln in diff.split("\n") if ln.startswith("-"))
        head = _status("✅", f"+{added} −{removed}", took)
        block = _output_block(diff, "Diff", opts, "diff")
        return head + ("\n" + block if block else "")
    if name == "Write":
        content = _s(call.input, "content")
        n = _lines(content)
        head = _status("✅", f"Wrote {_plural(n, 'line')}", took)
        block = _output_block(
            content, "Content", opts, _lang_for(_s(call.input, "file_path"))
        )
        return head + ("\n" + block if block else "")
    if name in ("Grep", "Glob"):
        hits = [ln for ln in text.split("\n") if ln.strip()]
        what = "match" if name == "Grep" else "file"
        count = (
            _plural(len(hits), what)
            if hits
            else f"no {what}es"
            if what == "match"
            else "no files"
        )
        head = _status("✅", count, took)
        block = _output_block("\n".join(hits), "Results", opts)
        return head + ("\n" + block if block else "")
    if name in ("Agent", "Task"):
        if meta.get("status") == "async_launched" or meta.get("isAsync"):
            return _status("🚀", "Started in background", took)
        head = _status("✅", _plural(_lines(text), "line"), took)
        if not text or opts.tool_output == "summary":
            return head
        return (
            head
            + "\n"
            + details("Result", escape_prose(text), open_=opts.expand_output)
        )
    if name in ("WebFetch", "WebSearch"):
        head = _status("✅", f"{len(text):,} chars", took)
        if not text or opts.tool_output == "summary":
            return head
        return (
            head
            + "\n"
            + details("Result", escape_prose(text), open_=opts.expand_output)
        )
    if name in ("TodoWrite", "TaskCreate", "TaskUpdate", "Skill", "ToolSearch"):
        return None  # the header says it all; results are boilerplate
    head = _status("✅", took)
    block = _output_block(text, "Result", opts)
    return head + ("\n" + block if block else "")


def render_tool_result(call: ToolCall, opts: RenderOptions) -> list[str]:
    """Replacement text for the tool_use message, plus continuation messages.

    Returns ``[]`` when the result adds nothing (keep the tool_use message).
    """
    body = tool_result_body(call, opts)
    if body is None:
        return []
    return _numbered(split_rich(f"{tool_header(call)}\n{body}"), "continued")
