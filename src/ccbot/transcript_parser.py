"""JSONL transcript parser for Claude Code session files.

Parses Claude Code session JSONL files and extracts structured messages.
Handles: text, thinking, tool_use, tool_result, local_command, and user messages,
plus `type: "system"` entries that matter to the user (local command output,
API error retries, usage-limit notices, compaction, model refusals, recaps).
Tool pairing: tool_use blocks in assistant messages are matched with
tool_result blocks in subsequent user messages via tool_use_id.

Shared by both session.py (history) and session_monitor.py (real-time).
Format reference: https://github.com/desis123/claude-code-viewer

Key classes: TranscriptParser (static methods), ParsedEntry (pre-formatted
``text`` plus ``raw`` / ``tool`` structured data), ToolCall, ParsedMessage,
PendingToolInfo.
"""

import base64
import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ParsedMessage:
    """Parsed message from a transcript."""

    message_type: str  # "user", "assistant", "tool_use", "tool_result", etc.
    text: str  # Extracted text content
    tool_name: str | None = None  # For tool_use messages


@dataclass
class ToolCall:
    """Structured data of one tool call, for renderers that lay it out
    themselves (rich messages) instead of using ``ParsedEntry.text``.

    On a tool_use entry only the call side is set; on the tool_result entry
    the result side is filled in as well.
    """

    name: str
    input: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None  # tool_use timestamp
    result_text: str | None = None
    is_error: bool = False
    interrupted: bool = False
    # Claude Code's structured result (JSONL "toolUseResult"): Bash
    # stdout/stderr, Edit structuredPatch, Agent status, …
    result_meta: dict[str, Any] | None = None
    finished_at: str | None = None  # tool_result timestamp


@dataclass
class ParsedEntry:
    """A single parsed message entry ready for display."""

    role: str  # "user" | "assistant"
    text: str  # Already formatted text
    content_type: (
        str  # "text" | "thinking" | "tool_use" | "tool_result" | "local_command"
        # | "error" | "warning" | "info"
    )
    tool_use_id: str | None = None
    timestamp: str | None = None  # ISO timestamp from JSONL
    tool_name: str | None = (
        None  # For tool_use entries, the tool name (e.g. "AskUserQuestion")
    )
    image_data: list[tuple[str, bytes]] | None = (
        None  # For tool_result entries with images: (media_type, raw_bytes)
    )
    # Assistant entries: the API message's stop_reason ("end_turn" marks the
    # last message of a turn) and id (shared by the thinking/text lines of
    # one API message)
    stop_reason: str | None = None
    api_message_id: str | None = None
    # Unformatted body of text / thinking entries (``text`` may carry
    # expandable-quote sentinels) and structured tool data (tool entries)
    raw: str | None = None
    tool: ToolCall | None = None


@dataclass
class PendingToolInfo:
    """Information about a pending tool_use waiting for its tool_result."""

    summary: str  # Formatted tool summary (e.g. "**Read**(file.py)")
    tool_name: str  # Tool name (e.g. "Read", "Edit")
    input_data: Any = None  # Tool input parameters (for Edit to generate diff)
    started_at: str | None = None  # tool_use timestamp
    full_input: dict[str, Any] | None = None  # every tool's input (rich rendering)


class TranscriptParser:
    """Parser for Claude Code JSONL session files.

    Expected JSONL entry structure:
    - type: "user" | "assistant" | "system" | "summary" | "file-history-snapshot" | ...
    - message.content: list of blocks (text, tool_use, tool_result, thinking)
    - system entries carry a `subtype` (local_command, api_error, informational,
      compact_boundary, ...) and a plain-string `content` instead of `message`
    - sessionId, cwd, timestamp, uuid: metadata fields

    Tool pairing model: tool_use blocks appear in assistant messages,
    matching tool_result blocks appear in the next user message (keyed by tool_use_id).
    """

    # Magic string constants
    _NO_CONTENT_PLACEHOLDER = "(no content)"
    _INTERRUPTED_TEXT = "[Request interrupted by user for tool use]"
    _MAX_SUMMARY_LENGTH = 200

    @staticmethod
    def parse_line(line: str) -> dict | None:
        """Parse a single JSONL line.

        Args:
            line: A single line from the JSONL file

        Returns:
            Parsed dict or None if line is empty/invalid
        """
        line = line.strip()
        if not line:
            return None

        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def get_message_type(data: dict) -> str | None:
        """Get the message type from parsed data.

        Returns:
            Message type: "user", "assistant", "file-history-snapshot", etc.
        """
        return data.get("type")

    @staticmethod
    def is_user_message(data: dict) -> bool:
        """Check if this is a user message."""
        return data.get("type") == "user"

    @staticmethod
    def extract_text_only(content_list: list[Any]) -> str:
        """Extract only text content from structured content.

        This is used for Telegram notifications where we only want
        the actual text response, not tool calls or thinking.

        Args:
            content_list: List of content blocks

        Returns:
            Combined text content only
        """
        if not isinstance(content_list, list):
            if isinstance(content_list, str):
                return content_list
            return ""

        texts = []
        for item in content_list:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        texts.append(text)

        return "\n".join(texts)

    # CSI sequences, OSC sequences (BEL or ST terminated), and stray lone ESC
    _RE_ANSI_ESCAPE = re.compile(
        r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b"
    )

    _RE_COMMAND_NAME = re.compile(r"<command-name>(.*?)</command-name>")
    _RE_LOCAL_STDOUT = re.compile(
        r"<local-command-stdout>(.*?)</local-command-stdout>", re.DOTALL
    )
    _RE_SYSTEM_TAGS = re.compile(
        r"<(bash-input|bash-stdout|bash-stderr|local-command-caveat|system-reminder)"
    )

    @staticmethod
    def _format_edit_diff(old_string: str, new_string: str) -> str:
        """Generate a compact unified diff between old_string and new_string."""
        old_lines = old_string.splitlines(keepends=True)
        new_lines = new_string.splitlines(keepends=True)
        diff = difflib.unified_diff(old_lines, new_lines, lineterm="")
        # Skip the --- / +++ header lines
        result_lines: list[str] = []
        for line in diff:
            if line.startswith("---") or line.startswith("+++"):
                continue
            # Strip trailing newline for clean display
            result_lines.append(line.rstrip("\n"))
        return "\n".join(result_lines)

    @classmethod
    def format_tool_use_summary(cls, name: str, input_data: dict | Any) -> str:
        """Format a tool_use block into a brief summary line.

        Args:
            name: Tool name (e.g. "Read", "Write", "Bash")
            input_data: The tool input dict

        Returns:
            Formatted string like "**Read**(file.py)"
        """
        if not isinstance(input_data, dict):
            return f"**{name}**"

        # Pick a meaningful short summary based on tool name
        summary = ""
        if name in ("Read", "Glob"):
            summary = input_data.get("file_path") or input_data.get("pattern", "")
        elif name == "Write":
            summary = input_data.get("file_path", "")
        elif name in ("Edit", "NotebookEdit"):
            summary = input_data.get("file_path") or input_data.get("notebook_path", "")
            # Note: Edit/Update diff and stats are generated in tool_result stage,
            # not here. We just show the tool name and file path.
        elif name == "Bash":
            summary = input_data.get("command", "")
        elif name == "Grep":
            summary = input_data.get("pattern", "")
        elif name in ("Task", "Agent"):
            summary = input_data.get("description", "")
        elif name == "WebFetch":
            summary = input_data.get("url", "")
        elif name == "WebSearch":
            summary = input_data.get("query", "")
        elif name == "TodoWrite":
            todos = input_data.get("todos", [])
            if isinstance(todos, list):
                summary = f"{len(todos)} item(s)"
        elif name == "TodoRead":
            summary = ""
        elif name == "AskUserQuestion":
            questions = input_data.get("questions", [])
            if isinstance(questions, list) and questions:
                q = questions[0]
                if isinstance(q, dict):
                    summary = q.get("question", "")
        elif name == "ExitPlanMode":
            summary = ""
        elif name == "Skill":
            summary = input_data.get("skill", "")
        else:
            # Generic: show first string value
            for v in input_data.values():
                if isinstance(v, str) and v:
                    summary = v
                    break

        if summary:
            if len(summary) > cls._MAX_SUMMARY_LENGTH:
                summary = summary[: cls._MAX_SUMMARY_LENGTH] + "…"
            return f"**{name}**({summary})"
        return f"**{name}**"

    @staticmethod
    def extract_tool_result_text(content: list | Any) -> str:
        """Extract text from a tool_result content block."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    t = item.get("text", "")
                    if t:
                        parts.append(t)
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return ""

    @staticmethod
    def extract_tool_result_images(
        content: list | Any,
    ) -> list[tuple[str, bytes]] | None:
        """Extract base64-encoded images from a tool_result content block.

        Returns list of (media_type, raw_bytes) tuples, or None if no images found.
        """
        if not isinstance(content, list):
            return None
        images: list[tuple[str, bytes]] = []
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            source = item.get("source")
            if not isinstance(source, dict) or source.get("type") != "base64":
                continue
            media_type = source.get("media_type", "image/png")
            data_str = source.get("data", "")
            if not data_str:
                continue
            try:
                raw_bytes = base64.b64decode(data_str)
                images.append((media_type, raw_bytes))
            except Exception:
                logger.debug("Failed to decode base64 image in tool_result")
        return images if images else None

    @classmethod
    def _parse_local_command(cls, text: str) -> ParsedMessage | None:
        """Recognise local-command XML (invoke or stdout) in message text."""
        stdout_match = cls._RE_LOCAL_STDOUT.search(text)
        if stdout_match:
            stdout = stdout_match.group(1).strip()
            cmd_match = cls._RE_COMMAND_NAME.search(text)
            cmd = cmd_match.group(1) if cmd_match else None
            return ParsedMessage(
                message_type="local_command",
                text=stdout,
                tool_name=cmd,  # reuse field for command name
            )
        # Pure command invocation (no stdout) — carry command name
        cmd_match = cls._RE_COMMAND_NAME.search(text)
        if cmd_match:
            return ParsedMessage(
                message_type="local_command_invoke",
                text="",
                tool_name=cmd_match.group(1),
            )
        return None

    @staticmethod
    def _format_local_command(cmd: str, text: str) -> str:
        """Format local command output: "❯ `/cmd`" + code block / inline code."""
        multiline = "\n" in text
        body = f"```\n{text}\n```" if multiline else f"`{text}`"
        return f"❯ `{cmd}`\n{body}" if cmd else body

    @staticmethod
    def _format_token_count(n: int | float) -> str:
        """Compact token count: 16162 -> "16.2k", 1_250_000 -> "1.2M"."""
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}k"
        return str(int(n))

    @classmethod
    def _parse_system_entry(cls, data: dict) -> ParsedEntry | None:
        """Turn a `type: "system"` entry into a display entry, or None.

        Only subtypes users need to see are surfaced; bookkeeping entries
        (turn_duration, stop_hook_summary, unknown subtypes) are ignored.
        local_command is handled by the caller (it needs invoke/stdout pairing).
        """
        subtype = data.get("subtype")
        raw = data.get("content")
        content = raw.strip() if isinstance(raw, str) else ""
        ts = cls.get_timestamp(data)

        def entry(text: str, content_type: str) -> ParsedEntry:
            return ParsedEntry(
                role="assistant", text=text, content_type=content_type, timestamp=ts
            )

        if subtype == "api_error":
            # Written once per retry attempt: one notice per failure streak
            attempt = data.get("retryAttempt")
            if attempt is not None and attempt != 1:
                return None
            err = data.get("error")
            err = err if isinstance(err, dict) else {}
            detail = err.get("formatted") or err.get("message") or "unknown error"
            text = f"⚠️ API error: {detail}"
            max_retries = data.get("maxRetries")
            if isinstance(max_retries, int) and not isinstance(max_retries, bool):
                text += f" — retrying (up to {max_retries}×)"
            return entry(text, "warning")

        if subtype == "informational":
            return entry(f"ℹ️ {content}", "info") if content else None

        if subtype == "compact_boundary":
            text = "🗜 Conversation compacted"
            meta = data.get("compactMetadata")
            if isinstance(meta, dict):
                pre, post = meta.get("preTokens"), meta.get("postTokens")
                if (
                    isinstance(pre, (int, float))
                    and isinstance(post, (int, float))
                    and not isinstance(pre, bool)
                    and not isinstance(post, bool)
                ):
                    text += (
                        f" ({cls._format_token_count(pre)} → "
                        f"{cls._format_token_count(post)} tokens)"
                    )
            return entry(text, "info")

        if subtype in ("model_refusal_fallback", "model_refusal_no_fallback"):
            return entry(f"⚠️ {content}", "warning") if content else None

        if subtype == "away_summary":
            return entry(f"📋 {content}", "info") if content else None

        return None

    @classmethod
    def parse_message(cls, data: dict) -> ParsedMessage | None:
        """Parse a message entry from the JSONL data.

        Args:
            data: Parsed JSON dict from a JSONL line

        Returns:
            ParsedMessage or None if not a parseable message
        """
        msg_type = cls.get_message_type(data)

        if msg_type not in ("user", "assistant"):
            return None

        message = data.get("message")
        if not isinstance(message, dict):
            return None
        content = message.get("content", "")

        if isinstance(content, list):
            text = cls.extract_text_only(content)
        else:
            text = str(content) if content else ""
        text = cls._RE_ANSI_ESCAPE.sub("", text)

        # Detect local command responses in user messages.
        # These are rendered as bot replies: "❯ /cmd\n  ⎿  output"
        if msg_type == "user" and text:
            local = cls._parse_local_command(text)
            if local:
                return local

        return ParsedMessage(
            message_type=msg_type,
            text=text,
        )

    @staticmethod
    def get_timestamp(data: dict) -> str | None:
        """Extract timestamp from message data."""
        return data.get("timestamp")

    EXPANDABLE_QUOTE_START = "\x02EXPQUOTE_START\x02"
    EXPANDABLE_QUOTE_END = "\x02EXPQUOTE_END\x02"

    @classmethod
    def _format_expandable_quote(cls, text: str) -> str:
        """Format text as a Telegram expandable blockquote.

        Wraps text with sentinel markers. The actual MarkdownV2 formatting
        (> prefix, || suffix, escaping) is done in convert_markdown() after
        telegramify processes the surrounding content.
        """
        return f"{cls.EXPANDABLE_QUOTE_START}{text}{cls.EXPANDABLE_QUOTE_END}"

    @classmethod
    def _format_tool_result_text(
        cls,
        text: str,
        tool_name: str | None = None,
        tool_input_data: dict | None = None,
    ) -> str:
        """Format tool result text with statistics summary.

        Shows relevant statistics for each tool type, with expandable quote for full content.

        No truncation here — per project principles, truncation is handled
        only at the send layer (split_message / _render_expandable_quote).
        """
        if not text:
            return ""

        line_count = text.count("\n") + 1 if text else 0

        # Tool-specific statistics
        if tool_name == "Read":
            # Read: show line count instead of full content
            return f"  ⎿  Read {line_count} lines"

        elif tool_name == "Write":
            # Write: line count comes from the input content, not the result
            # (result is usually just "File created successfully at: ...")
            written = tool_input_data.get("content", "") if tool_input_data else ""
            if not written:
                written_lines = 0
            else:
                written_lines = written.count("\n") + (
                    0 if written.endswith("\n") else 1
                )
            return f"  ⎿  Wrote {written_lines} lines"

        elif tool_name == "Bash":
            # Bash: show output line count
            if line_count > 0:
                stats = f"  ⎿  Output {line_count} lines"
                return stats + "\n" + cls._format_expandable_quote(text)
            return cls._format_expandable_quote(text)

        elif tool_name == "Grep":
            # Grep: show match count (count non-empty lines)
            matches = len([line for line in text.split("\n") if line.strip()])
            stats = f"  ⎿  Found {matches} matches"
            return stats + "\n" + cls._format_expandable_quote(text)

        elif tool_name == "Glob":
            # Glob: show file count
            files = len([line for line in text.split("\n") if line.strip()])
            stats = f"  ⎿  Found {files} files"
            return stats + "\n" + cls._format_expandable_quote(text)

        elif tool_name in ("Task", "Agent"):
            # Task/Agent (renamed in newer Claude Code): show output length
            if line_count > 0:
                stats = f"  ⎿  Agent output {line_count} lines"
                return stats + "\n" + cls._format_expandable_quote(text)
            return cls._format_expandable_quote(text)

        elif tool_name == "WebFetch":
            # WebFetch: show content length
            char_count = len(text)
            stats = f"  ⎿  Fetched {char_count} characters"
            return stats + "\n" + cls._format_expandable_quote(text)

        elif tool_name == "WebSearch":
            # WebSearch: show results count (estimate by sections)
            results = text.count("\n\n") + 1 if text else 0
            stats = f"  ⎿  {results} search results"
            return stats + "\n" + cls._format_expandable_quote(text)

        # Default: expandable quote without stats
        return cls._format_expandable_quote(text)

    @classmethod
    def parse_entries(
        cls,
        entries: list[dict],
        pending_tools: dict[str, PendingToolInfo] | None = None,
    ) -> tuple[list[ParsedEntry], dict[str, PendingToolInfo]]:
        """Parse a list of JSONL entries into a flat list of display-ready messages.

        This is the shared core logic used by both get_recent_messages (history)
        and check_for_updates (monitor).

        Args:
            entries: List of parsed JSONL dicts (already filtered through parse_line)
            pending_tools: Optional carry-over pending tool_use state from a
                previous call (tool_use_id -> formatted summary). Used by the
                monitor to handle tool_use and tool_result arriving in separate
                poll cycles.

        Returns:
            Tuple of (parsed entries, remaining pending_tools state)
        """
        result: list[ParsedEntry] = []
        last_cmd_name: str | None = None
        # Pending tool_use blocks keyed by id
        _carry_over = pending_tools is not None
        if pending_tools is None:
            pending_tools = {}
        else:
            pending_tools = dict(pending_tools)  # don't mutate caller's dict

        tag_from = 0  # entries appended since the previous line
        tag_with: dict | None = None

        def _tag() -> None:
            if tag_with is None:
                return
            for e in result[tag_from:]:
                e.stop_reason = tag_with.get("stop_reason")
                e.api_message_id = tag_with.get("id")

        for data in entries:
            _tag()
            tag_from, tag_with = len(result), None
            msg_type = cls.get_message_type(data)
            entry_timestamp = cls.get_timestamp(data)

            if msg_type == "system":
                # Deliberately leaves last_cmd_name alone: unrelated system
                # entries may sit between a command invoke and its stdout.
                if data.get("subtype") == "local_command":
                    raw = data.get("content")
                    text = (
                        cls._RE_ANSI_ESCAPE.sub("", raw) if isinstance(raw, str) else ""
                    )
                    local = cls._parse_local_command(text) if text else None
                    if local and local.message_type == "local_command_invoke":
                        last_cmd_name = local.tool_name
                    elif local:
                        cmd = local.tool_name or last_cmd_name or ""
                        result.append(
                            ParsedEntry(
                                role="assistant",
                                text=cls._format_local_command(cmd, local.text),
                                content_type="local_command",
                                timestamp=entry_timestamp,
                            )
                        )
                        last_cmd_name = None
                    continue
                sys_entry = cls._parse_system_entry(data)
                if sys_entry:
                    result.append(sys_entry)
                continue

            if msg_type not in ("user", "assistant"):
                continue

            message = data.get("message")
            if not isinstance(message, dict):
                continue
            if msg_type == "assistant":
                tag_with = message
            content = message.get("content", "")
            if not isinstance(content, list):
                content = [{"type": "text", "text": str(content)}] if content else []

            parsed = cls.parse_message(data)

            # Handle local command messages first
            if parsed:
                if parsed.message_type == "local_command_invoke":
                    last_cmd_name = parsed.tool_name
                    continue
                if parsed.message_type == "local_command":
                    cmd = parsed.tool_name or last_cmd_name or ""
                    result.append(
                        ParsedEntry(
                            role="assistant",
                            text=cls._format_local_command(cmd, parsed.text),
                            content_type="local_command",
                            timestamp=entry_timestamp,
                        )
                    )
                    last_cmd_name = None
                    continue
            last_cmd_name = None

            if msg_type == "assistant":
                # Process content blocks
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")

                    if btype == "text":
                        t = block.get("text", "").strip()
                        if t and t != cls._NO_CONTENT_PLACEHOLDER:
                            # Synthetic assistant messages carrying an API
                            # failure ("You've hit your session limit", 5xx)
                            is_error = bool(data.get("isApiErrorMessage"))
                            result.append(
                                ParsedEntry(
                                    role="assistant",
                                    text=f"🚨 {t}" if is_error else t,
                                    content_type="error" if is_error else "text",
                                    timestamp=entry_timestamp,
                                    raw=t,
                                )
                            )

                    elif btype == "tool_use":
                        tool_id = block.get("id", "")
                        name = block.get("name", "unknown")
                        inp = block.get("input", {})
                        summary = cls.format_tool_use_summary(name, inp)

                        # ExitPlanMode: emit plan content as text before tool_use entry
                        if name == "ExitPlanMode" and isinstance(inp, dict):
                            plan = inp.get("plan", "")
                            if plan:
                                result.append(
                                    ParsedEntry(
                                        role="assistant",
                                        text=plan,
                                        content_type="text",
                                        timestamp=entry_timestamp,
                                    )
                                )
                        if tool_id:
                            # Store tool info for later tool_result formatting
                            # Edit tool needs input_data to generate diff in tool_result stage
                            input_data = (
                                inp
                                if name in ("Edit", "NotebookEdit", "Write")
                                else None
                            )
                            pending_tools[tool_id] = PendingToolInfo(
                                summary=summary,
                                tool_name=name,
                                input_data=input_data,
                                started_at=entry_timestamp,
                                full_input=inp if isinstance(inp, dict) else None,
                            )
                            # Also emit tool_use entry with tool_name for immediate handling
                            result.append(
                                ParsedEntry(
                                    role="assistant",
                                    text=summary,
                                    content_type="tool_use",
                                    tool_use_id=tool_id,
                                    timestamp=entry_timestamp,
                                    tool_name=name,
                                    tool=ToolCall(
                                        name=name,
                                        input=inp if isinstance(inp, dict) else {},
                                        started_at=entry_timestamp,
                                    ),
                                )
                            )
                        else:
                            result.append(
                                ParsedEntry(
                                    role="assistant",
                                    text=summary,
                                    content_type="tool_use",
                                    tool_use_id=tool_id or None,
                                    timestamp=entry_timestamp,
                                    tool_name=name,
                                    tool=ToolCall(
                                        name=name,
                                        input=inp if isinstance(inp, dict) else {},
                                        started_at=entry_timestamp,
                                    ),
                                )
                            )

                    elif btype == "thinking":
                        # Empty thinking blocks (redacted / summary-only)
                        # are skipped: a "(thinking)" placeholder per turn
                        # is pure noise in the chat.
                        thinking_text = block.get("thinking", "")
                        if thinking_text:
                            quoted = cls._format_expandable_quote(thinking_text)
                            result.append(
                                ParsedEntry(
                                    role="assistant",
                                    text=quoted,
                                    content_type="thinking",
                                    timestamp=entry_timestamp,
                                    raw=thinking_text,
                                )
                            )

            elif msg_type == "user":
                # Check for tool_result blocks and merge with pending tools
                user_text_parts: list[str] = []

                for block in content:
                    if not isinstance(block, dict):
                        if isinstance(block, str) and block.strip():
                            user_text_parts.append(block.strip())
                        continue
                    btype = block.get("type", "")

                    if btype == "tool_result":
                        tool_use_id = block.get("tool_use_id", "")
                        result_content = block.get("content", "")
                        result_text = cls.extract_tool_result_text(result_content)
                        result_images = cls.extract_tool_result_images(result_content)
                        is_error = block.get("is_error", False)
                        is_interrupted = result_text == cls._INTERRUPTED_TEXT
                        tool_info = pending_tools.pop(tool_use_id, None)
                        _tuid = tool_use_id or None

                        # Extract tool info from PendingToolInfo object
                        if tool_info is None:
                            tool_summary = None
                            tool_name = None
                            tool_input_data = None
                        else:
                            tool_summary = tool_info.summary
                            tool_name = tool_info.tool_name
                            tool_input_data = tool_info.input_data
                        n_before = len(result)

                        if is_interrupted:
                            # Show interruption inline with tool summary
                            entry_text = tool_summary or ""
                            if entry_text:
                                entry_text += "\n⏹ Interrupted"
                            else:
                                entry_text = "⏹ Interrupted"
                            result.append(
                                ParsedEntry(
                                    role="assistant",
                                    text=entry_text,
                                    content_type="tool_result",
                                    tool_use_id=_tuid,
                                    timestamp=entry_timestamp,
                                )
                            )
                        elif is_error:
                            # Show error in stats line
                            if tool_summary:
                                entry_text = tool_summary
                            else:
                                entry_text = "**Error**"
                            # Add error message in stats format
                            if result_text:
                                # Take first line of error as summary
                                error_summary = result_text.split("\n")[0]
                                if len(error_summary) > 100:
                                    error_summary = error_summary[:100] + "…"
                                entry_text += f"\n  ⎿  Error: {error_summary}"
                                # If multi-line error, add expandable quote
                                if "\n" in result_text:
                                    entry_text += "\n" + cls._format_expandable_quote(
                                        result_text
                                    )
                            else:
                                entry_text += "\n  ⎿  Error"
                            result.append(
                                ParsedEntry(
                                    role="assistant",
                                    text=entry_text,
                                    content_type="tool_result",
                                    tool_use_id=_tuid,
                                    timestamp=entry_timestamp,
                                    image_data=result_images,
                                )
                            )
                        elif tool_summary:
                            entry_text = tool_summary
                            # For Edit tool, generate diff stats and expandable quote
                            if tool_name == "Edit" and tool_input_data and result_text:
                                old_s = tool_input_data.get("old_string", "")
                                new_s = tool_input_data.get("new_string", "")
                                if old_s and new_s:
                                    diff_text = cls._format_edit_diff(old_s, new_s)
                                    if diff_text:
                                        added = sum(
                                            1
                                            for line in diff_text.split("\n")
                                            if line.startswith("+")
                                            and not line.startswith("+++")
                                        )
                                        removed = sum(
                                            1
                                            for line in diff_text.split("\n")
                                            if line.startswith("-")
                                            and not line.startswith("---")
                                        )
                                        stats = f"  ⎿  Added {added} lines, removed {removed} lines"
                                        entry_text += (
                                            "\n"
                                            + stats
                                            + "\n"
                                            + cls._format_expandable_quote(diff_text)
                                        )
                            # For other tools, append formatted result text
                            elif (
                                result_text
                                and cls.EXPANDABLE_QUOTE_START not in tool_summary
                            ):
                                entry_text += "\n" + cls._format_tool_result_text(
                                    result_text, tool_name, tool_input_data
                                )
                            result.append(
                                ParsedEntry(
                                    role="assistant",
                                    text=entry_text,
                                    content_type="tool_result",
                                    tool_use_id=_tuid,
                                    timestamp=entry_timestamp,
                                    image_data=result_images,
                                )
                            )
                        elif result_text or result_images:
                            result.append(
                                ParsedEntry(
                                    role="assistant",
                                    text=cls._format_tool_result_text(
                                        result_text, tool_name, tool_input_data
                                    )
                                    if result_text
                                    else (tool_summary or ""),
                                    content_type="tool_result",
                                    tool_use_id=_tuid,
                                    timestamp=entry_timestamp,
                                    image_data=result_images,
                                )
                            )

                        meta = data.get("toolUseResult")
                        call = ToolCall(
                            name=tool_name or "",
                            input=(tool_info.full_input if tool_info else None) or {},
                            started_at=tool_info.started_at if tool_info else None,
                            result_text=result_text,
                            is_error=bool(is_error),
                            interrupted=is_interrupted,
                            result_meta=meta if isinstance(meta, dict) else None,
                            finished_at=entry_timestamp,
                        )
                        for emitted in result[n_before:]:
                            emitted.tool = call

                    elif btype == "text":
                        t = block.get("text", "").strip()
                        if t and not cls._RE_SYSTEM_TAGS.search(t):
                            user_text_parts.append(t)

                # Add user text if present (skip if message was only tool_results)
                if user_text_parts:
                    combined = "\n".join(user_text_parts)
                    # Skip if it looks like local command XML
                    if not cls._RE_LOCAL_STDOUT.search(
                        combined
                    ) and not cls._RE_COMMAND_NAME.search(combined):
                        result.append(
                            ParsedEntry(
                                role="user",
                                text=combined,
                                content_type="text",
                                timestamp=entry_timestamp,
                            )
                        )

        # Flush remaining pending tools at end.
        # In carry-over mode (monitor), keep them pending for the next call
        # without emitting entries. In one-shot mode (history), emit them.
        remaining_pending = dict(pending_tools)
        if not _carry_over:
            for tool_id, tool_info in pending_tools.items():
                result.append(
                    ParsedEntry(
                        role="assistant",
                        text=tool_info.summary,
                        content_type="tool_use",
                        tool_use_id=tool_id,
                    )
                )

        _tag()

        # Strip whitespace
        for entry in result:
            entry.text = entry.text.strip()

        return result, remaining_pending
