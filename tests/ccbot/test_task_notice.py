"""Tests for <task-notification> prompts (Claude Code 2.1.285 samples).

A background command, agent or monitor reporting back is injected into the
transcript as a user turn; it must render as a notice, not as "👤 <xml>".
"""

from ccbot import rich_render as rr
from ccbot.handlers.response_builder import build_rich_parts
from ccbot.transcript_parser import TranscriptParser as TP

TMP = "/tmp/claude-1000/-root-dev-work-x/533274d3-bf4b-4bc7-9b16-d3ba2e2d824b"

COMMAND = f"""<task-notification>
<task-id>b43umusyb</task-id>
<tool-use-id>toolu_01H4</tool-use-id>
<output-file>{TMP}/tasks/b43umusyb.output</output-file>
<status>completed</status>
<summary>Background command "./scripts/test-fast &gt; {TMP}/scratchpad/gate.log 2&gt;&amp;1; echo exit=$?" completed (exit code 0)</summary>
</task-notification>"""

DESCRIBED = """<task-notification>
<task-id>b1</task-id>
<tool-use-id>toolu_02</tool-use-id>
<status>completed</status>
<summary>Background command "Run the bot suite with 2 workers" completed (exit code 1: No matches found)</summary>
</task-notification>"""

AGENT = """<task-notification>
<task-id>a3c5</task-id>
<tool-use-id>toolu_03</tool-use-id>
<status>completed</status>
<summary>Agent "Map Cloudflare DNS pipeline" finished</summary>
<note>A task-notification fires each time this agent stops.</note>
<result>Here is the full map.

# Map

1. **Key finding** with `proxied=False`.</result>
<usage><subagent_tokens>281061</subagent_tokens><tool_uses>115</tool_uses><duration_ms>632100</duration_ms></usage>
<worktree><worktreePath>/r/.claude/worktrees/agent-a3c5</worktreePath><worktreeBranch>worktree-agent-a3c5</worktreeBranch></worktree>
</task-notification>"""

AGENT_FAILED = """<task-notification>
<task-id>a9</task-id>
<status>failed</status>
<summary>Agent "3x-ui on PostgreSQL" failed: Agent terminated early due to an API error: You've hit your session limit</summary>
</task-notification>"""

MONITOR_EVENT = """<task-notification>
<task-id>bf2</task-id>
<summary>Monitor event: "Grok worker exits"</summary>
<event>[Monitor expired after 30m with no events delivered.]</event>
</task-notification>"""

STOPPED = """<task-notification>
<task-id>bg</task-id>
<status>killed</status>
<summary>Background command "until ! pgrep -f x; do sleep 5; done" was stopped after reaching its background time limit</summary>
<note>If the work in progress still needs it, start it again.</note>
</task-notification>"""


def test_background_command():
    n = TP.parse_task_notification(COMMAND)
    assert n is not None
    assert (n.kind, n.icon, n.headline, n.outcome) == (
        "command",
        "✅",
        "Background command finished",
        "exit 0",
    )
    assert n.subject.startswith("./scripts/test-fast > /tmp/")  # entities decoded
    assert "2>&1" in n.subject
    assert n.tool_use_id == "toolu_01H4"


def test_description_becomes_the_headline():
    n = TP.parse_task_notification(DESCRIBED)
    assert n is not None
    assert (n.headline, n.subject, n.icon) == (
        "Run the bot suite with 2 workers",
        "",
        "⚠️",  # completed, but not exit 0
    )
    assert n.outcome == "exit 1: No matches found"
    (msg,) = rr.render_task_notice(n)
    assert msg == (
        "⚠️ **Run the bot suite with 2 workers** · finished · exit 1: No matches found"
    )


def test_agent_with_result_usage_and_worktree():
    n = TP.parse_task_notification(AGENT)
    assert n is not None
    assert (n.kind, n.headline, n.subject) == (
        "agent",
        "Agent finished",
        "Map Cloudflare DNS pipeline",
    )
    assert n.stats == "10m 32s · 115 tool uses · 281.1k tokens"
    assert n.worktree == "worktree-agent-a3c5"
    (msg,) = rr.render_task_notice(n)
    assert msg.startswith("✅ **Agent finished** · Map Cloudflare DNS pipeline · 10m")
    assert "<details><summary>Result</summary>" in msg
    assert "# Map" in msg
    assert "fires each time" not in msg  # the note is for the model


def test_agent_failure_reason():
    n = TP.parse_task_notification(AGENT_FAILED)
    assert n is not None
    assert (n.icon, n.headline, n.subject) == (
        "❌",
        "Agent failed",
        "3x-ui on PostgreSQL",
    )
    assert n.outcome.startswith("Agent terminated early")


def test_monitor_event_and_stopped_command():
    ev = TP.parse_task_notification(MONITOR_EVENT)
    assert ev is not None
    (msg,) = rr.render_task_notice(ev)
    assert msg.startswith("📡 **Monitor** · Grok worker exits\n")
    assert "Monitor expired after 30m" in msg
    st = TP.parse_task_notification(STOPPED)
    assert st is not None
    assert (st.icon, st.headline) == ("⏹", "Background command stopped")
    assert st.outcome == "after reaching its background time limit"


def test_long_command_is_folded_in_the_notice():
    n = TP.parse_task_notification(COMMAND)
    assert n is not None
    (msg,) = rr.render_task_notice(n)
    head, rest = msg.split("\n", 1)
    assert head == "✅ **Background command finished** · exit 0"
    assert rest.startswith("<details><summary>$ `./scripts/test-fast > …/scratchpad/")
    assert n.subject in msg


def test_parser_emits_a_notice_not_a_user_message():
    entry = {
        "type": "user",
        "timestamp": "2026-09-30T16:35:55Z",
        "origin": {"kind": "task-notification"},
        "message": {"role": "user", "content": COMMAND},
    }
    (parsed,), _ = TP.parse_entries([entry])
    assert parsed.role == "assistant"
    assert parsed.content_type == "task_notification"
    assert parsed.tool_use_id == "toolu_01H4"
    assert "<task-notification>" not in parsed.text
    assert parsed.text.startswith("✅ **Background command finished** · exit 0")
    parts = build_rich_parts(
        parsed.text, parsed.content_type, parsed.role, raw=parsed.raw
    )
    assert parts is not None and parts[0].startswith("✅ **Background command")
    assert "👤" not in parts[0]


def test_ordinary_user_text_is_untouched():
    entry = {
        "type": "user",
        "message": {"role": "user", "content": "please mention <task-notification>"},
    }
    (parsed,), _ = TP.parse_entries([entry])
    assert (parsed.role, parsed.content_type) == ("user", "text")
