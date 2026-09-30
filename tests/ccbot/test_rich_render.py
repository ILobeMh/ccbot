"""Tests for rich_render: escaping, splitting and per-entry rich markdown."""

import re

import pytest

from ccbot import rich_render as rr
from ccbot.transcript_parser import ToolCall


def _balanced(chunk: str) -> bool:
    """Fences and <details> blocks open/close in pairs within a chunk."""
    fence: str | None = None
    depth = 0
    for line in chunk.split("\n"):
        m = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence is None:
            if m:
                fence = m.group(1)
            elif line.startswith("<details"):
                depth += 1
            elif line == "</details>":
                depth -= 1
                if depth < 0:
                    return False
        elif m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
            fence = None
    return fence is None and depth == 0


# ── escaping ──


class TestEscapeProse:
    def test_rich_only_syntax_is_neutralised(self):
        out = rr.escape_prose("set $HOME and $PATH; a < b && c; ==x== ||y||")
        assert "\\$HOME" in out and "\\$PATH" in out
        assert "&lt;" in out and "&amp;&amp;" in out
        assert "\\=\\=x\\=\\=" in out
        assert "\\|\\|y\\|\\|" in out

    def test_code_is_left_alone(self):
        text = "run `echo $HOME < x`\n```bash\nfoo $BAR <baz> || true\n```\nafter $X"
        out = rr.escape_prose(text)
        assert "`echo $HOME < x`" in out
        assert "foo $BAR <baz> || true" in out
        assert "after \\$X" in out

    def test_table_pipes_kept(self):
        table = "| a | b |\n|---|---|\n| 1 || 2 |"
        assert rr.escape_prose(table) == table

    def test_markdown_kept(self):
        md = "## Title\n- **bold** item\n1. `code`\n> quote"
        assert rr.escape_prose(md) == md

    def test_escape_inline_escapes_everything(self):
        assert rr.escape_inline("a*b_c[d]") == "a\\*b\\_c\\[d\\]"


def test_code_block_fence_outlasts_content():
    block = rr.code_block("a ``` b", "bash")
    assert block.startswith("````bash\n") and block.endswith("\n````")


# ── splitting ──


class TestSplitRich:
    def test_short_text_untouched(self):
        assert rr.split_rich("hello") == ["hello"]

    def test_long_code_block_split_stays_balanced(self):
        text = "head\n" + rr.code_block(
            "\n".join(f"line {i}" for i in range(5000)), "py"
        )
        chunks = rr.split_rich(text, max_chars=4000)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 4000
            assert _balanced(c), c[:200]
        assert chunks[1].startswith("```py")

    def test_details_split_reopens_block(self):
        body = rr.code_block("\n".join(f"out {i}" for i in range(3000)))
        text = "⚙️ **title**\n" + rr.details("Output · 3000 lines", body)
        chunks = rr.split_rich(text, max_chars=5000)
        assert len(chunks) > 2
        for c in chunks:
            assert _balanced(c), c[:300]
            assert len(c) <= 5000
        assert chunks[1].startswith("<details><summary>Output")

    def test_line_budget(self):
        text = "\n".join(f"- item {i}" for i in range(1000))
        chunks = rr.split_rich(text, max_lines=400)
        assert len(chunks) == 3
        assert all(c.count("\n") < 400 for c in chunks)

    def test_giant_single_line(self):
        chunks = rr.split_rich("x" * 50000, max_chars=10000)
        assert all(len(c) <= 10000 for c in chunks)
        assert "".join(chunks) == "x" * 50000


# ── tools ──


def _bash(command: str, **res) -> ToolCall:
    return ToolCall(
        name="Bash",
        input={"command": command, "description": "Testing every config"},
        started_at="2026-09-30T10:00:00Z",
        **res,
    )


class TestBash:
    def test_header_has_description_and_full_command(self):
        cmd = "uv run pytest -q\nuv run pyright"
        (msg,) = rr.render_tool_use(_bash(cmd))
        assert msg.startswith("⚙️ **Testing every config**\n```bash\n")
        assert cmd in msg

    def test_long_command_is_folded(self):
        cmd = "\n".join(f"echo {i}" for i in range(60))
        (msg,) = rr.render_tool_use(_bash(cmd))
        assert "<details><summary>$ `echo 0` · 60 lines</summary>" in msg
        assert cmd in msg

    def test_described_command_folds_sooner(self):
        cmd = "cat > /opt/probe.py <<'EOF'\n" + "x = 1\n" * 4 + "EOF"
        (described,) = rr.render_tool_use(_bash(cmd))
        assert (
            "<details><summary>$ `cat > /opt/probe.py <<'EOF'` · 6 lines" in described
        )
        (bare,) = rr.render_tool_use(
            ToolCall(name="Bash", input={"command": cmd}, started_at=None)
        )
        assert bare.startswith("⚙️ **Bash**\n```bash\n")  # nothing else explains it

    def test_one_long_line_preview_drops_cd_and_session_tmp(self):
        tmp = "/tmp/claude-1000/-root-dev-x/533274d3-bf4b-4bc7-9b16-d3ba2e2d824b/"
        cmd = f"cd /root/dev/x && ./scripts/test-fast > {tmp}gate.log 2>&1; " * 4
        (msg,) = rr.render_tool_use(_bash(cmd))
        summary = msg.split("</summary>")[0]
        assert "$ `./scripts/test-fast > …/gate.log" in summary
        assert "lines" not in summary
        assert cmd.strip() in msg  # the full command is untouched

    def test_result_status_duration_and_output(self):
        call = _bash(
            "probe",
            result_text="a\nb",
            result_meta={"stdout": "a\nb", "stderr": "", "interrupted": False},
            finished_at="2026-09-30T10:06:19Z",
        )
        (msg,) = rr.render_tool_result(call, rr.RenderOptions())
        assert "⎿ ✅ 2 lines · 6m 19s" in msg
        assert "<details><summary>Output · 2 lines</summary>" in msg
        assert _balanced(msg)

    def test_preview_and_summary_modes(self):
        out = "\n".join(str(i) for i in range(40))
        call = _bash("seq 40", result_text=out, result_meta={"stdout": out})
        (prev,) = rr.render_tool_result(
            call, rr.RenderOptions(tool_output="preview", preview_lines=5)
        )
        assert "<details>" not in prev
        assert "\n4\n" in prev and "\n5\n" not in prev
        assert "35 more lines" in prev
        (summ,) = rr.render_tool_result(call, rr.RenderOptions(tool_output="summary"))
        assert "```" in summ  # only the command block
        assert summ.count("```") == 2

    def test_stderr_shown_separately(self):
        call = _bash(
            "x", result_text="o", result_meta={"stdout": "o", "stderr": "boom"}
        )
        (msg,) = rr.render_tool_result(call, rr.RenderOptions())
        assert "<summary>stderr · 1 line</summary>" in msg

    def test_error_and_interrupt(self):
        err = _bash("false", result_text="Exit code 1\nnope", is_error=True)
        (msg,) = rr.render_tool_result(err, rr.RenderOptions())
        assert "⎿ ❌ Exit code 1" in msg
        stop = _bash("sleep 9", interrupted=True)
        (msg,) = rr.render_tool_result(stop, rr.RenderOptions())
        assert "⏹ Interrupted" in msg

    def test_huge_output_continues_in_more_messages(self):
        out = "\n".join(f"row {i:06d} " + "x" * 40 for i in range(4000))
        call = _bash("gen", result_text=out, result_meta={"stdout": out})
        msgs = rr.render_tool_result(call, rr.RenderOptions())
        assert len(msgs) > 1
        assert msgs[0].startswith("⚙️ **Testing every config**")
        for i, m in enumerate(msgs):
            assert len(m) <= rr.RICH_CHAR_BUDGET + 100
            assert _balanced(m)
            if i:
                assert m.startswith(f"_continued ({i + 1}/{len(msgs)})_")
        assert sum(m.count("row ") for m in msgs) == 4000


def test_edit_uses_structured_patch():
    call = ToolCall(
        name="Edit",
        input={"file_path": "/p/geo.py", "old_string": "a", "new_string": "b"},
        result_text="ok",
        result_meta={
            "structuredPatch": [
                {
                    "oldStart": 65,
                    "oldLines": 3,
                    "newStart": 65,
                    "newLines": 2,
                    "lines": [" keep", "-gone", "-gone2", "+new"],
                }
            ]
        },
    )
    (msg,) = rr.render_tool_result(call, rr.RenderOptions())
    assert msg.startswith("✏️ **Edit** `/p/geo.py`")
    assert "⎿ ✅ +1 −2" in msg
    assert "```diff\n@@ -65,3 +65,2 @@\n keep\n-gone" in msg


def test_write_shows_content_with_language():
    call = ToolCall(
        name="Write",
        input={"file_path": "/p/x.go", "content": "package x\n"},
        result_text="File created",
    )
    (msg,) = rr.render_tool_result(call, rr.RenderOptions())
    assert "Wrote 1 line" in msg and "```go\npackage x\n```" in msg


def test_todo_checklist_and_no_result_edit():
    call = ToolCall(
        name="TodoWrite",
        input={
            "todos": [
                {"content": "done one", "status": "completed"},
                {"content": "doing", "status": "in_progress"},
                {"content": "later", "status": "pending"},
            ]
        },
        result_text="Todos have been modified",
    )
    (msg,) = rr.render_tool_use(call)
    assert "- [x] ~~done one~~" in msg
    assert "- [ ] **doing** ⏳" in msg
    assert "- [ ] later" in msg
    assert rr.render_tool_result(call, rr.RenderOptions()) == []


def test_mcp_tool_header():
    call = ToolCall(name="mcp__context7__query-docs", input={"query": "x"})
    (msg,) = rr.render_tool_use(call)
    assert msg.startswith("🔌 **context7** · query\\-docs")
    assert "```json" in msg


def test_agent_background():
    call = ToolCall(
        name="Agent",
        input={"description": "Review", "subagent_type": "reviewer", "prompt": "go"},
        result_meta={"status": "async_launched", "isAsync": True},
        result_text="launched",
    )
    (msg,) = rr.render_tool_result(call, rr.RenderOptions())
    assert "🤖 **Agent** · Review (reviewer)" in msg
    assert "🚀 Started in background" in msg


# ── thinking / text ──


class TestThinking:
    def test_collapsed_with_first_line_preview(self):
        (msg,) = rr.render_thinking(
            "Let me check $HOME first.\nThen more.", rr.RenderOptions()
        )
        assert msg.startswith(
            "<details><summary>💭 Let me check \\$HOME first\\.</summary>"
        )
        assert _balanced(msg)

    def test_long_thinking_split_into_numbered_blocks(self):
        text = "\n\n".join("thought " + "y" * 200 for _ in range(400))
        msgs = rr.render_thinking(text, rr.RenderOptions())
        assert len(msgs) > 1
        assert all(_balanced(m) for m in msgs)
        assert f"(1/{len(msgs)})" in msgs[0]

    def test_max_chars_truncates(self):
        (msg,) = rr.render_thinking(
            "z" * 5000, rr.RenderOptions(thinking_max_chars=100)
        )
        assert "_… (truncated)_" in msg and "z" * 101 not in msg


def test_text_tables_pass_through():
    md = "| Metric | Value |\n|---|---:|\n| speed | 42 |"
    assert rr.render_text(md) == [md]


def test_local_command_output_block():
    (msg,) = rr.render_local_command("/context", "Context Usage\n⛁ ⛁ 14%")
    assert msg == "❯ `/context`\n```text\nContext Usage\n⛁ ⛁ 14%\n```"


@pytest.mark.parametrize(
    ("text", "rtl"),
    [
        ("سلام، این یک تست است", True),
        ("Hello world", False),
        ("این `some code here that is long` تست", True),
        ("```\nکد\n```\nEnglish prose", False),
    ],
)
def test_is_rtl(text, rtl):
    assert rr.is_rtl(text) is rtl


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(5, "5s"), (60, "1m"), (379, "6m 19s"), (3600, "1h"), (3720, "1h 2m")],
)
def test_format_duration(seconds, expected):
    assert rr.format_duration(seconds) == expected
