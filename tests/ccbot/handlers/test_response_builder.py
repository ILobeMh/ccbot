"""Tests for response_builder.build_response_parts."""

from ccbot.config import config
from ccbot.handlers.response_builder import build_response_parts
from ccbot.markdown_v2 import convert_markdown
from ccbot.transcript_parser import TranscriptParser

EXP_START = TranscriptParser.EXPANDABLE_QUOTE_START
EXP_END = TranscriptParser.EXPANDABLE_QUOTE_END


class TestBuildResponseParts:
    def test_user_message_has_emoji_prefix(self):
        parts = build_response_parts("hello", is_complete=True, role="user")
        assert len(parts) == 1
        assert "\U0001f464" in parts[0]

    def test_user_message_truncated_at_3000_chars(self):
        long_text = "a" * 4000
        parts = build_response_parts(long_text, is_complete=True, role="user")
        assert len(parts) == 1
        short_parts = build_response_parts("b" * 100, is_complete=True, role="user")
        assert len(parts[0]) < len(long_text)
        assert len(short_parts[0]) < len(parts[0])

    def test_thinking_content_truncated_at_limit(self, monkeypatch):
        monkeypatch.setattr(config, "thinking_max_chars", 500)
        inner = "x" * 800
        text = f"{EXP_START}{inner}{EXP_END}"
        parts = build_response_parts(text, is_complete=True, content_type="thinking")
        assert len(parts) == 1
        assert "truncated" in parts[0].lower()
        assert parts[0].count("x") == 500

    def test_thinking_full_is_split_into_quoted_parts(self, monkeypatch):
        monkeypatch.setattr(config, "thinking_max_chars", 0)
        inner = "\n".join(f"thought line {i} " + "y" * 60 for i in range(120))
        text = f"{EXP_START}{inner}{EXP_END}"
        parts = build_response_parts(text, is_complete=True, content_type="thinking")
        assert len(parts) > 1
        for i, part in enumerate(parts, 1):
            assert part.startswith(f"∴ Thinking… [{i}/{len(parts)}]")
            assert part.count(EXP_START) == 1 and part.endswith(EXP_END)
        # every part must fit one Telegram message after rendering
        for part in parts:
            assert len(convert_markdown(part)) <= 4096
        assert "truncated" not in "".join(parts)
        assert "thought line 119" in parts[-1]

    def test_thinking_full_short_is_single_part_without_counter(self, monkeypatch):
        monkeypatch.setattr(config, "thinking_max_chars", 0)
        parts = build_response_parts(
            f"{EXP_START}brief{EXP_END}", is_complete=True, content_type="thinking"
        )
        assert parts == [f"∴ Thinking…\n{EXP_START}brief{EXP_END}"]

    def test_plain_text_single_part(self):
        parts = build_response_parts("short text", is_complete=True)
        assert len(parts) == 1

    def test_plain_text_multi_part_has_page_suffix(self):
        long_text = "\n".join(f"line {i} " + "padding" * 50 for i in range(200))
        parts = build_response_parts(long_text, is_complete=True)
        assert len(parts) > 1
        assert "1/" in parts[0]

    def test_expandable_quote_stays_atomic(self):
        inner = "thought " * 100
        text = f"{EXP_START}{inner}{EXP_END}"
        parts = build_response_parts(text, is_complete=False, content_type="thinking")
        assert len(parts) == 1

    def test_thinking_has_prefix(self):
        parts = build_response_parts(
            "some thought", is_complete=True, content_type="thinking"
        )
        assert len(parts) == 1
        assert "Thinking" in parts[0]

    def test_assistant_text_no_prefix(self):
        parts = build_response_parts(
            "hello world", is_complete=True, content_type="text", role="assistant"
        )
        assert len(parts) == 1
        assert "\U0001f464" not in parts[0]
        assert "Thinking" not in parts[0]
