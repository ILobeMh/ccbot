"""Tests for ccbot.transcript_parser — pure logic, no I/O."""

import pytest

from ccbot.transcript_parser import (
    ParsedMessage,
    TranscriptParser,
)

EXPQUOTE_START = TranscriptParser.EXPANDABLE_QUOTE_START
EXPQUOTE_END = TranscriptParser.EXPANDABLE_QUOTE_END


# ── parse_line ───────────────────────────────────────────────────────────


class TestParseLine:
    @pytest.mark.parametrize(
        "line, expected",
        [
            ('{"type": "user"}', {"type": "user"}),
            ("not-json", None),
            ("", None),
            ("   \t  ", None),
        ],
        ids=["valid_json", "invalid_json", "empty", "whitespace"],
    )
    def test_parse_line(self, line: str, expected: dict | None):
        assert TranscriptParser.parse_line(line) == expected


# ── extract_text_only ────────────────────────────────────────────────────


class TestExtractTextOnly:
    @pytest.mark.parametrize(
        "content, expected",
        [
            ("plain string", "plain string"),
            (
                [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}],
                "hello\nworld",
            ),
            (
                [
                    {"type": "text", "text": "keep"},
                    {"type": "tool_use", "name": "Read"},
                ],
                "keep",
            ),
            ([], ""),
            (42, ""),
        ],
        ids=["string", "text_blocks", "mixed", "empty_list", "non_list_non_string"],
    )
    def test_extract_text_only(self, content: list | str | int, expected: str):
        assert TranscriptParser.extract_text_only(content) == expected


# ── format_tool_use_summary ──────────────────────────────────────────────


class TestFormatToolUseSummary:
    @pytest.mark.parametrize(
        "name, input_data, expected",
        [
            ("Read", {"file_path": "src/main.py"}, "**Read**(src/main.py)"),
            ("Write", {"file_path": "out.txt"}, "**Write**(out.txt)"),
            ("Bash", {"command": "ls -la"}, "**Bash**(ls -la)"),
            ("Grep", {"pattern": "TODO"}, "**Grep**(TODO)"),
            ("Glob", {"pattern": "*.py"}, "**Glob**(*.py)"),
            ("Task", {"description": "analyze code"}, "**Task**(analyze code)"),
            (
                "Agent",
                {"description": "analyze code", "subagent_type": "Explore"},
                "**Agent**(analyze code)",
            ),
            (
                "WebFetch",
                {"url": "https://example.com"},
                "**WebFetch**(https://example.com)",
            ),
            ("WebSearch", {"query": "python async"}, "**WebSearch**(python async)"),
            ("TodoWrite", {"todos": [1, 2, 3]}, "**TodoWrite**(3 item(s))"),
            ("TodoRead", {}, "**TodoRead**"),
            (
                "AskUserQuestion",
                {"questions": [{"question": "Continue?"}]},
                "**AskUserQuestion**(Continue?)",
            ),
            ("ExitPlanMode", {}, "**ExitPlanMode**"),
            ("Skill", {"skill": "code-review"}, "**Skill**(code-review)"),
            (
                "CustomTool",
                {"first_key": "value1"},
                "**CustomTool**(value1)",
            ),
        ],
        ids=[
            "Read",
            "Write",
            "Bash",
            "Grep",
            "Glob",
            "Task",
            "Agent",
            "WebFetch",
            "WebSearch",
            "TodoWrite",
            "TodoRead",
            "AskUserQuestion",
            "ExitPlanMode",
            "Skill",
            "unknown_tool",
        ],
    )
    def test_tool_summary(self, name: str, input_data: dict, expected: str):
        assert TranscriptParser.format_tool_use_summary(name, input_data) == expected

    def test_non_dict_input(self):
        assert (
            TranscriptParser.format_tool_use_summary("Read", "not a dict") == "**Read**"
        )

    def test_truncation_at_200_chars(self):
        long_value = "x" * 250
        result = TranscriptParser.format_tool_use_summary(
            "Bash", {"command": long_value}
        )
        assert len(long_value) > 200
        assert result == f"**Bash**({'x' * 200}…)"


# ── extract_tool_result_text ─────────────────────────────────────────────


class TestExtractToolResultText:
    @pytest.mark.parametrize(
        "content, expected",
        [
            ("raw string", "raw string"),
            (
                [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}],
                "line1\nline2",
            ),
            (
                [{"type": "text", "text": "keep"}, {"type": "image", "data": "..."}],
                "keep",
            ),
            (None, ""),
        ],
        ids=["string", "text_blocks", "mixed", "none"],
    )
    def test_extract_tool_result_text(self, content: str | list | None, expected: str):
        assert TranscriptParser.extract_tool_result_text(content) == expected


# ── parse_message ────────────────────────────────────────────────────────


class TestParseMessage:
    def test_user_text(self):
        data = {
            "type": "user",
            "message": {"content": [{"type": "text", "text": "hello"}]},
        }
        result = TranscriptParser.parse_message(data)
        assert result == ParsedMessage(message_type="user", text="hello")

    def test_assistant_text(self):
        data = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "hi there"}]},
        }
        result = TranscriptParser.parse_message(data)
        assert result == ParsedMessage(message_type="assistant", text="hi there")

    def test_local_command_with_stdout(self):
        data = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "<command-name>/help</command-name>"
                            "<local-command-stdout>Available commands</local-command-stdout>"
                        ),
                    }
                ]
            },
        }
        result = TranscriptParser.parse_message(data)
        assert result is not None
        assert result.message_type == "local_command"
        assert result.text == "Available commands"
        assert result.tool_name == "/help"

    def test_local_command_invoke(self):
        data = {
            "type": "user",
            "message": {
                "content": [
                    {"type": "text", "text": "<command-name>/clear</command-name>"}
                ]
            },
        }
        result = TranscriptParser.parse_message(data)
        assert result is not None
        assert result.message_type == "local_command_invoke"
        assert result.text == ""
        assert result.tool_name == "/clear"

    def test_non_user_assistant_returns_none(self):
        data = {
            "type": "summary",
            "message": {"content": "summary text"},
        }
        assert TranscriptParser.parse_message(data) is None

    def test_string_content(self):
        data = {
            "type": "assistant",
            "message": {"content": "plain response"},
        }
        result = TranscriptParser.parse_message(data)
        assert result == ParsedMessage(message_type="assistant", text="plain response")


# ── _format_edit_diff ────────────────────────────────────────────────────


class TestFormatEditDiff:
    @pytest.mark.parametrize(
        "old, new, check",
        [
            (
                "hello",
                "world",
                lambda r: "-hello" in r and "+world" in r,
            ),
            (
                "line1\nline2\nline3",
                "line1\nchanged\nline3",
                lambda r: "-line2" in r and "+changed" in r,
            ),
            (
                "same",
                "same",
                lambda r: r == "",
            ),
        ],
        ids=["single_line", "multi_line", "identical"],
    )
    def test_format_edit_diff(self, old: str, new: str, check):
        result = TranscriptParser._format_edit_diff(old, new)
        assert check(result), f"Check failed for ({old!r}, {new!r}): {result!r}"


# ── _format_tool_result_text ─────────────────────────────────────────────


class TestFormatToolResultText:
    @pytest.mark.parametrize(
        "text, tool_name, check",
        [
            (
                "line1\nline2\nline3",
                "Read",
                lambda r: r == "  ⎿  Read 3 lines",
            ),
            (
                # Write line count comes from tool_input_data (the written
                # content), not the result text — without it, 0 lines
                "File created successfully at: out.txt",
                "Write",
                lambda r: r == "  ⎿  Wrote 0 lines",
            ),
            (
                "output line",
                "Bash",
                lambda r: (
                    r.startswith("  ⎿  Output 1 lines")
                    and EXPQUOTE_START in r
                    and EXPQUOTE_END in r
                ),
            ),
            (
                "file1.py\nfile2.py\n",
                "Grep",
                lambda r: "Found 2 matches" in r and EXPQUOTE_START in r,
            ),
            (
                "a.py\nb.py\nc.py",
                "Glob",
                lambda r: "Found 3 files" in r and EXPQUOTE_START in r,
            ),
            (
                "agent says hello",
                "Task",
                lambda r: "Agent output 1 lines" in r and EXPQUOTE_START in r,
            ),
            (
                "page content here",
                "WebFetch",
                lambda r: (
                    f"Fetched {len('page content here')} characters" in r
                    and EXPQUOTE_START in r
                ),
            ),
            (
                "",
                "Read",
                lambda r: r == "",
            ),
        ],
        ids=["Read", "Write", "Bash", "Grep", "Glob", "Task", "WebFetch", "empty"],
    )
    def test_format_tool_result_text(self, text: str, tool_name: str, check):
        result = TranscriptParser._format_tool_result_text(text, tool_name)
        assert check(result), f"Failed check for {tool_name!r}: {result!r}"

    def test_write_counts_lines_from_input_content(self):
        """Write derives its line count from the written content (f5ddd7f)."""
        result = TranscriptParser._format_tool_result_text(
            "File created successfully at: out.txt",
            "Write",
            tool_input_data={"content": "line1\nline2"},
        )
        assert result == "  ⎿  Wrote 2 lines"

    def test_write_trailing_newline_not_counted_extra(self):
        result = TranscriptParser._format_tool_result_text(
            "ok",
            "Write",
            tool_input_data={"content": "line1\nline2\n"},
        )
        assert result == "  ⎿  Wrote 2 lines"


# ── parse_entries ────────────────────────────────────────────────────────


class TestParseEntries:
    def test_assistant_text(self, make_jsonl_entry, make_text_block):
        entries = [make_jsonl_entry("assistant", [make_text_block("Hello!")])]
        result, pending = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].text == "Hello!"
        assert result[0].content_type == "text"

    def test_user_text(self, make_jsonl_entry, make_text_block):
        entries = [make_jsonl_entry("user", [make_text_block("Hi bot")])]
        result, pending = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].text == "Hi bot"

    def test_tool_use_and_result_pairing(
        self,
        make_jsonl_entry,
        make_text_block,
        make_tool_use_block,
        make_tool_result_block,
    ):
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Read", {"file_path": "app.py"})],
            ),
            make_jsonl_entry(
                "user",
                [make_tool_result_block("t1", "file contents line1\nline2\nline3")],
            ),
        ]
        result, pending = TranscriptParser.parse_entries(entries)
        tool_use_entries = [e for e in result if e.content_type == "tool_use"]
        tool_result_entries = [e for e in result if e.content_type == "tool_result"]
        assert len(tool_use_entries) == 1
        assert tool_use_entries[0].tool_use_id == "t1"
        assert "**Read**" in tool_use_entries[0].text
        assert len(tool_result_entries) == 1
        assert tool_result_entries[0].tool_use_id == "t1"
        assert not pending

    def test_thinking_block(self, make_jsonl_entry, make_thinking_block):
        entries = [
            make_jsonl_entry("assistant", [make_thinking_block("reasoning here")])
        ]
        result, pending = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        assert result[0].content_type == "thinking"
        assert EXPQUOTE_START in result[0].text
        assert EXPQUOTE_END in result[0].text
        assert "reasoning here" in result[0].text

    def test_empty_thinking_block_is_skipped(
        self, make_jsonl_entry, make_thinking_block
    ):
        entries = [make_jsonl_entry("assistant", [make_thinking_block("")])]
        result, pending = TranscriptParser.parse_entries(entries)
        assert result == []

    def test_local_command_with_stdout(self, make_jsonl_entry, make_text_block):
        xml = (
            "<command-name>/status</command-name>"
            "<local-command-stdout>all good</local-command-stdout>"
        )
        entries = [make_jsonl_entry("user", [make_text_block(xml)])]
        result, pending = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        assert result[0].content_type == "local_command"
        assert "/status" in result[0].text
        assert "all good" in result[0].text

    def test_exit_plan_mode_emits_plan(self, make_jsonl_entry, make_tool_use_block):
        block = make_tool_use_block(
            "t1", "ExitPlanMode", {"plan": "Step 1: do X\nStep 2: do Y"}
        )
        entries = [make_jsonl_entry("assistant", [block])]
        result, pending = TranscriptParser.parse_entries(entries)
        texts = [e for e in result if e.content_type == "text"]
        tool_uses = [e for e in result if e.content_type == "tool_use"]
        assert len(texts) == 1
        assert "Step 1: do X" in texts[0].text
        assert len(tool_uses) >= 1

    def test_edit_tool_diff_stats(
        self,
        make_jsonl_entry,
        make_tool_use_block,
        make_tool_result_block,
    ):
        edit_input = {
            "file_path": "main.py",
            "old_string": "old line",
            "new_string": "new line",
        }
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Edit", edit_input)],
            ),
            make_jsonl_entry(
                "user",
                [make_tool_result_block("t1", "OK")],
            ),
        ]
        result, pending = TranscriptParser.parse_entries(entries)
        tool_result_entries = [e for e in result if e.content_type == "tool_result"]
        assert len(tool_result_entries) == 1
        tr = tool_result_entries[0]
        assert "Added" in tr.text
        assert "removed" in tr.text
        assert EXPQUOTE_START in tr.text

    def test_error_tool_result(
        self,
        make_jsonl_entry,
        make_tool_use_block,
        make_tool_result_block,
    ):
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Bash", {"command": "rm -rf /"})],
            ),
            make_jsonl_entry(
                "user",
                [make_tool_result_block("t1", "Permission denied", is_error=True)],
            ),
        ]
        result, pending = TranscriptParser.parse_entries(entries)
        tool_result_entries = [e for e in result if e.content_type == "tool_result"]
        assert len(tool_result_entries) == 1
        assert "Error: Permission denied" in tool_result_entries[0].text

    def test_interrupted_tool_result(
        self,
        make_jsonl_entry,
        make_tool_use_block,
        make_tool_result_block,
    ):
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Read", {"file_path": "x.py"})],
            ),
            make_jsonl_entry(
                "user",
                [make_tool_result_block("t1", TranscriptParser._INTERRUPTED_TEXT)],
            ),
        ]
        result, pending = TranscriptParser.parse_entries(entries)
        tool_result_entries = [e for e in result if e.content_type == "tool_result"]
        assert len(tool_result_entries) == 1
        assert "Interrupted" in tool_result_entries[0].text

    def test_pending_tools_carry_over(self, make_jsonl_entry, make_tool_use_block):
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Read", {"file_path": "a.py"})],
            ),
        ]
        result, pending = TranscriptParser.parse_entries(entries, pending_tools={})
        assert "t1" in pending
        flushed = [
            e for e in result if e.content_type == "tool_use" and e.tool_use_id == "t1"
        ]
        assert len(flushed) == 1

    def test_pending_tools_flushed_without_carry_over(
        self, make_jsonl_entry, make_tool_use_block
    ):
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Read", {"file_path": "a.py"})],
            ),
        ]
        result, pending = TranscriptParser.parse_entries(entries, pending_tools=None)
        tool_entries = [e for e in result if e.tool_use_id == "t1"]
        assert len(tool_entries) == 2
        assert tool_entries[0].content_type == "tool_use"
        assert tool_entries[1].content_type == "tool_use"

    def test_system_tag_filtered(self, make_jsonl_entry, make_text_block):
        entries = [
            make_jsonl_entry(
                "user",
                [
                    make_text_block(
                        "<system-reminder>secret instructions</system-reminder>"
                    )
                ],
            ),
        ]
        result, pending = TranscriptParser.parse_entries(entries)
        user_entries = [e for e in result if e.role == "user"]
        assert len(user_entries) == 0


class TestTurnTagging:
    def test_end_turn_and_message_id_tagged(self, make_jsonl_entry, make_text_block):
        e1 = make_jsonl_entry("assistant", [make_text_block("working")])
        e1["message"]["stop_reason"] = "tool_use"
        e1["message"]["id"] = "msg_a"
        e2 = make_jsonl_entry("assistant", [make_text_block("final")])
        e2["message"]["stop_reason"] = "end_turn"
        e2["message"]["id"] = "msg_b"
        result, _ = TranscriptParser.parse_entries([e1, e2])
        assert [(r.stop_reason, r.api_message_id) for r in result] == [
            ("tool_use", "msg_a"),
            ("end_turn", "msg_b"),
        ]


# ── system entries ───────────────────────────────────────────────────────

TS = "2026-09-30T12:00:00.000Z"


def _sys(subtype: str, **fields) -> dict:
    return {"type": "system", "subtype": subtype, "timestamp": TS, **fields}


def _parse(*entries: dict):
    result, _ = TranscriptParser.parse_entries(list(entries))
    return result


class TestAgentToolAlias:
    def test_agent_result_formatted_like_task(self):
        text = "line1\nline2"
        assert TranscriptParser._format_tool_result_text(
            text, "Agent"
        ) == TranscriptParser._format_tool_result_text(text, "Task")
        assert "Agent output 2 lines" in TranscriptParser._format_tool_result_text(
            text, "Agent"
        )


class TestAnsiStripping:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("\x1b[1mbold\x1b[22m", "bold"),
            ("\x1b[38;2;136;136;136m⛁ \x1b[39m", "⛁ "),
            ("a\x1b[2Kb\x1b[?25lc\x1b[3;4H", "abc"),
            ("\x1b]0;title\x07text", "text"),
            ("\x1b]8;;http://x\x1b\\link\x1b]8;;\x1b\\", "link"),
            ("stray\x1bescape", "strayescape"),
        ],
        ids=["sgr", "24bit", "csi_general", "osc_bel", "osc_st", "lone_esc"],
    )
    def test_strip(self, raw: str, expected: str):
        assert TranscriptParser._RE_ANSI_ESCAPE.sub("", raw) == expected

    def test_user_local_command_stdout_stripped(
        self, make_jsonl_entry, make_text_block
    ):
        xml = (
            "<local-command-stdout>\x1b[38;2;1;2;3mhi\x1b[39m\x1b[2K"
            "</local-command-stdout>"
        )
        result = _parse(make_jsonl_entry("user", [make_text_block(xml)]))
        assert len(result) == 1
        assert "\x1b" not in result[0].text
        assert "hi" in result[0].text


class TestSystemLocalCommand:
    INVOKE = (
        "<command-name>/model</command-name>\n"
        "            <command-message>model</command-message>\n"
        "            <command-args></command-args>"
    )
    STDOUT = "<local-command-stdout>Kept model as `Opus 5`</local-command-stdout>"

    def test_system_invoke_and_stdout(self):
        result = _parse(
            _sys("local_command", content=self.INVOKE),
            _sys("local_command", content=self.STDOUT),
        )
        assert len(result) == 1
        assert result[0].content_type == "local_command"
        assert result[0].role == "assistant"
        assert result[0].timestamp == TS
        assert result[0].text == "❯ `/model`\n`Kept model as `Opus 5``"

    def test_user_invoke_system_stdout(self, make_jsonl_entry, make_text_block):
        result = _parse(
            make_jsonl_entry("user", [make_text_block(self.INVOKE)]),
            _sys("local_command", content=self.STDOUT),
        )
        assert [r.content_type for r in result] == ["local_command"]
        assert result[0].text.startswith("❯ `/model`")

    def test_system_invoke_user_stdout(self, make_jsonl_entry, make_text_block):
        result = _parse(
            _sys("local_command", content=self.INVOKE),
            make_jsonl_entry("user", [make_text_block(self.STDOUT)]),
        )
        assert [r.content_type for r in result] == ["local_command"]
        assert result[0].text.startswith("❯ `/model`")

    def test_user_invoke_user_stdout(self, make_jsonl_entry, make_text_block):
        result = _parse(
            make_jsonl_entry(
                "user", [make_text_block("<command-name>/effort</command-name>")]
            ),
            make_jsonl_entry(
                "user",
                [
                    make_text_block(
                        "<local-command-stdout>Cancelled</local-command-stdout>"
                    )
                ],
            ),
        )
        assert len(result) == 1
        assert result[0].text == "❯ `/effort`\n`Cancelled`"

    def test_unrelated_system_entries_do_not_reset_command(self):
        result = _parse(
            _sys("local_command", content=self.INVOKE),
            _sys("turn_duration", durationMs=5, messageCount=3),
            _sys("stop_hook_summary", hookCount=1),
            _sys("something_new", content="x"),
            _sys("local_command", content=self.STDOUT),
        )
        assert len(result) == 1
        assert result[0].text.startswith("❯ `/model`")

    def test_multiline_ansi_stdout(self):
        content = (
            "<local-command-stdout> \x1b[1mContext Usage\x1b[22m\n"
            "\x1b[38;2;136;136;136m⛁ \x1b[38;2;153;153;153m⛁ ⛁ \x1b[39m  Opus 5\n"
            " 140.6k/1m tokens (14%)\x1b[39m</local-command-stdout>"
        )
        result = _parse(
            _sys("local_command", content="<command-name>/context</command-name>"),
            _sys("local_command", content=content),
        )
        assert len(result) == 1
        text = result[0].text
        assert "\x1b" not in text and "[38;2" not in text
        assert text.startswith("❯ `/context`\n```\n")
        assert "Context Usage" in text
        assert "140.6k/1m tokens (14%)" in text
        assert text.endswith("\n```")

    def test_stdout_without_invoke(self):
        result = _parse(_sys("local_command", content=self.STDOUT))
        assert result[0].text == "`Kept model as `Opus 5``"


class TestSystemNotices:
    def test_api_error_first_attempt(self):
        result = _parse(
            _sys(
                "api_error",
                level="error",
                error={"message": "Connection error.", "formatted": "Proxy refused"},
                retryInMs=584,
                retryAttempt=1,
                maxRetries=10,
            )
        )
        assert len(result) == 1
        e = result[0]
        assert e.role == "assistant"
        assert e.content_type == "warning"
        assert e.timestamp == TS
        assert e.text == "⚠️ API error: Proxy refused — retrying (up to 10×)"

    def test_api_error_falls_back_to_message(self):
        result = _parse(
            _sys("api_error", error={"message": "Connection error."}, maxRetries=3)
        )
        assert result[0].text == (
            "⚠️ API error: Connection error. — retrying (up to 3×)"
        )

    def test_api_error_without_max_retries(self):
        result = _parse(_sys("api_error", error={"message": "Boom"}, retryAttempt=1))
        assert result[0].text == "⚠️ API error: Boom"

    def test_api_error_retry_streak_emits_once(self):
        entries = [
            _sys("api_error", error={"message": "x"}, retryAttempt=n, maxRetries=10)
            for n in range(1, 6)
        ]
        assert len(_parse(*entries)) == 1

    def test_api_error_later_attempt_alone_ignored(self):
        assert _parse(_sys("api_error", error={"message": "x"}, retryAttempt=4)) == []

    def test_api_error_then_final_error_message(
        self, make_jsonl_entry, make_text_block
    ):
        final = make_jsonl_entry("assistant", [make_text_block("Connection error.")])
        final["isApiErrorMessage"] = True
        result = _parse(
            _sys("api_error", error={"message": "x"}, retryAttempt=1, maxRetries=2),
            _sys("api_error", error={"message": "x"}, retryAttempt=2, maxRetries=2),
            final,
        )
        assert [r.content_type for r in result] == ["warning", "error"]

    def test_informational(self):
        result = _parse(
            _sys(
                "informational",
                content="Usage limit reached · continuing automatically at 5:20pm",
                level="notice",
            )
        )
        assert result[0].content_type == "info"
        assert result[0].role == "assistant"
        assert result[0].timestamp == TS
        assert result[0].text == (
            "ℹ️ Usage limit reached · continuing automatically at 5:20pm"
        )

    def test_compact_boundary_with_tokens(self):
        result = _parse(
            _sys(
                "compact_boundary",
                content="Conversation compacted",
                compactMetadata={
                    "trigger": "manual",
                    "preTokens": 670887,
                    "postTokens": 16162,
                },
            )
        )
        assert result[0].content_type == "info"
        assert result[0].timestamp == TS
        assert result[0].text == "🗜 Conversation compacted (670.9k → 16.2k tokens)"

    def test_compact_boundary_millions(self):
        result = _parse(
            _sys(
                "compact_boundary",
                compactMetadata={"preTokens": 1_250_000, "postTokens": 900},
            )
        )
        assert result[0].text == "🗜 Conversation compacted (1.2M → 900 tokens)"

    def test_compact_boundary_without_metadata(self):
        result = _parse(_sys("compact_boundary", content="Conversation compacted"))
        assert result[0].text == "🗜 Conversation compacted"

    @pytest.mark.parametrize(
        "subtype", ["model_refusal_fallback", "model_refusal_no_fallback"]
    )
    def test_model_refusal(self, subtype: str):
        result = _parse(
            _sys(subtype, content="Opus 5.5's safeguards flagged this session.")
        )
        assert result[0].content_type == "warning"
        assert result[0].text == "⚠️ Opus 5.5's safeguards flagged this session."
        assert result[0].timestamp == TS

    def test_model_refusal_empty_content_skipped(self):
        assert _parse(_sys("model_refusal_fallback", content="")) == []
        assert _parse(_sys("model_refusal_no_fallback")) == []

    def test_away_summary(self):
        result = _parse(
            _sys("away_summary", content="You're upgrading your ccbot fork.")
        )
        assert result[0].content_type == "info"
        assert result[0].text == "📋 You're upgrading your ccbot fork."
        assert result[0].timestamp == TS

    @pytest.mark.parametrize(
        "entry",
        [
            _sys("turn_duration", durationMs=27432, messageCount=927),
            _sys("stop_hook_summary", hookCount=1),
            _sys("brand_new_subtype", content="whatever"),
            {"type": "system"},
        ],
        ids=["turn_duration", "stop_hook_summary", "unknown", "no_subtype"],
    )
    def test_ignored_subtypes(self, entry: dict):
        assert _parse(entry) == []

    def test_system_entries_do_not_tag_turn_state(
        self, make_jsonl_entry, make_text_block
    ):
        a = make_jsonl_entry("assistant", [make_text_block("final")])
        a["message"]["stop_reason"] = "end_turn"
        a["message"]["id"] = "msg_x"
        result = _parse(a, _sys("informational", content="note"))
        assert [(r.content_type, r.stop_reason) for r in result] == [
            ("text", "end_turn"),
            ("info", None),
        ]


class TestStructuredToolData:
    """ParsedEntry.tool / .raw carry what rich rendering needs, untruncated."""

    def _bash_entries(self, command: str):
        return [
            {
                "type": "assistant",
                "timestamp": "2026-09-30T10:00:00.000Z",
                "message": {
                    "id": "m1",
                    "content": [
                        {"type": "thinking", "thinking": "let me test **all** configs"},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Bash",
                            "input": {
                                "command": command,
                                "description": "Testing every subscription config",
                            },
                        },
                    ],
                },
            },
            {
                "type": "user",
                "timestamp": "2026-09-30T10:06:19.000Z",
                "toolUseResult": {
                    "stdout": "ok\nok",
                    "stderr": "warn",
                    "interrupted": False,
                },
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok\nok",
                        }
                    ]
                },
            },
        ]

    def test_bash_call_and_result_are_structured(self):
        command = "python3 - <<'EOF'\n" + "x = 1\n" * 100 + "EOF"
        entries, pending = TranscriptParser.parse_entries(self._bash_entries(command))
        assert pending == {}
        thinking, use, res = entries
        assert thinking.raw == "let me test **all** configs"
        assert use.tool is not None
        assert use.tool.input["command"] == command  # full, not cut at 200
        assert use.tool.input["description"] == "Testing every subscription config"
        assert use.tool.started_at == "2026-09-30T10:00:00.000Z"
        assert res.tool is not None
        assert res.tool.name == "Bash"
        assert res.tool.input["command"] == command
        assert res.tool.result_text == "ok\nok"
        assert res.tool.result_meta == {
            "stdout": "ok\nok",
            "stderr": "warn",
            "interrupted": False,
        }
        assert res.tool.started_at == "2026-09-30T10:00:00.000Z"
        assert res.tool.finished_at == "2026-09-30T10:06:19.000Z"

    def test_carry_over_keeps_input_across_polls(self):
        first, second = self._bash_entries("ls")
        _, pending = TranscriptParser.parse_entries([first], pending_tools={})
        entries, _ = TranscriptParser.parse_entries([second], pending_tools=pending)
        (res,) = entries
        assert res.tool is not None and res.tool.input == {
            "command": "ls",
            "description": "Testing every subscription config",
        }

    def test_error_result_is_flagged(self):
        first, second = self._bash_entries("false")
        second["message"]["content"][0]["is_error"] = True
        second["toolUseResult"] = "Error: exit code 1"  # string, not a dict
        entries, _ = TranscriptParser.parse_entries([first, second])
        res = entries[-1]
        assert res.tool is not None and res.tool.is_error
        assert res.tool.result_meta is None
