"""Response message building for Telegram delivery.

Builds paginated response messages from Claude Code output:
  - Handles different content types (text, thinking, tool_use, tool_result)
  - Splits long messages into pages within Telegram's 4096 char limit
  - Truncates thinking content to keep messages compact

Markdown conversion is NOT done here — the send layer (message_sender,
message_queue) handles convert_markdown() so each message is converted
exactly once.

Key function:
  - build_response_parts: Build paginated response messages
"""

from ..config import config
from ..markdown_v2 import convert_markdown_tables
from ..telegram_sender import split_message
from ..transcript_parser import TranscriptParser

# Raw chars per thinking part; escaping inflates this before the 3800-char
# render budget in markdown_v2._render_expandable_quote
THINKING_PART_CHARS = 2800
THINKING_PREFIX = "∴ Thinking…"


def _build_thinking_parts(text: str, max_chars: int) -> list[str]:
    start_tag = TranscriptParser.EXPANDABLE_QUOTE_START
    end_tag = TranscriptParser.EXPANDABLE_QUOTE_END
    if start_tag in text and end_tag in text:
        inner = text[text.index(start_tag) + len(start_tag) : text.index(end_tag)]
    else:
        inner = text
    inner = inner.strip()
    if max_chars > 0:
        if len(inner) > max_chars:
            inner = inner[:max_chars] + "\n\n… (thinking truncated)"
        return [f"{THINKING_PREFIX}\n{start_tag}{inner}{end_tag}"]
    chunks = split_message(inner, max_length=THINKING_PART_CHARS) or [""]
    total = len(chunks)
    if total == 1:
        return [f"{THINKING_PREFIX}\n{start_tag}{chunks[0]}{end_tag}"]
    return [
        f"{THINKING_PREFIX} [{i}/{total}]\n{start_tag}{chunk}{end_tag}"
        for i, chunk in enumerate(chunks, 1)
    ]


def build_response_parts(
    text: str,
    is_complete: bool,
    content_type: str = "text",
    role: str = "assistant",
) -> list[str]:
    """Build paginated response messages for Telegram.

    Returns a list of raw markdown strings, each within Telegram's 4096 char limit.
    Multi-part messages get a [1/N] suffix.
    Markdown-to-MarkdownV2 conversion is done by the send layer, not here.
    """
    text = text.strip()

    # User messages: add emoji prefix (no newline)
    if role == "user":
        prefix = "👤 "
        separator = ""
        # User messages are typically short, no special processing needed
        if len(text) > 3000:
            text = text[:3000] + "…"
        return [f"{prefix}{text}"]

    # Thinking: truncate to config.thinking_max_chars, or (0) send it all as
    # [i/N] parts, each its own collapsed quote
    if content_type == "thinking" and is_complete:
        return _build_thinking_parts(text, config.thinking_max_chars)

    # Format based on content type
    if content_type == "thinking":
        # Thinking: prefix with "∴ Thinking…" and single newline
        prefix = "∴ Thinking…"
        separator = "\n"
    else:
        # Plain text: no prefix
        prefix = ""
        separator = ""

    # If text contains expandable quote sentinels, don't split —
    # the quote must stay atomic. Truncation is handled by
    # _render_expandable_quote in markdown_v2.py.
    if TranscriptParser.EXPANDABLE_QUOTE_START in text:
        if prefix:
            return [f"{prefix}{separator}{text}"]
        return [text]

    # Convert tables to card-style before splitting so tables aren't broken
    # across messages. The send layer's convert_markdown() call is idempotent.
    text = convert_markdown_tables(text)

    # Split first, then assemble each chunk.
    # Use conservative max to leave room for MarkdownV2 expansion at send layer.
    max_text = 3000 - len(prefix) - len(separator)

    text_chunks = split_message(text, max_length=max_text)
    total = len(text_chunks)

    if total == 1:
        if prefix:
            return [f"{prefix}{separator}{text_chunks[0]}"]
        return [text_chunks[0]]

    parts = []
    for i, chunk in enumerate(text_chunks, 1):
        if prefix:
            parts.append(f"{prefix}{separator}{chunk}\n\n[{i}/{total}]")
        else:
            parts.append(f"{chunk}\n\n[{i}/{total}]")
    return parts
