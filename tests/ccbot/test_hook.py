"""Tests for Claude Code session tracking hook."""

import io
import json
import subprocess
import sys

import pytest

import ccbot.hook as hook_mod
from ccbot.hook import (
    _UUID_RE,
    _is_hook_installed,
    _should_skip_nested,
    _valid_transcript_path,
    hook_main,
)


class TestUuidRegex:
    @pytest.mark.parametrize(
        "value",
        [
            "550e8400-e29b-41d4-a716-446655440000",
            "00000000-0000-0000-0000-000000000000",
            "abcdef01-2345-6789-abcd-ef0123456789",
        ],
        ids=["standard", "all-zeros", "all-hex"],
    )
    def test_valid_uuid_matches(self, value: str) -> None:
        assert _UUID_RE.match(value) is not None

    @pytest.mark.parametrize(
        "value",
        [
            "not-a-uuid",
            "550e8400-e29b-41d4-a716",
            "550e8400-e29b-41d4-a716-44665544000g",
            "",
        ],
        ids=["gibberish", "truncated", "invalid-hex-char", "empty"],
    )
    def test_invalid_uuid_no_match(self, value: str) -> None:
        assert _UUID_RE.match(value) is None


class TestIsHookInstalled:
    def test_hook_present(self) -> None:
        settings = {
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {"type": "command", "command": "ccbot hook", "timeout": 5}
                        ]
                    }
                ]
            }
        }
        assert _is_hook_installed(settings) is True

    def test_no_hooks_key(self) -> None:
        assert _is_hook_installed({}) is False

    def test_different_hook_command(self) -> None:
        settings = {
            "hooks": {
                "SessionStart": [
                    {"hooks": [{"type": "command", "command": "other-tool hook"}]}
                ]
            }
        }
        assert _is_hook_installed(settings) is False

    def test_full_path_matches(self) -> None:
        settings = {
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": "/usr/bin/ccbot hook",
                                "timeout": 5,
                            }
                        ]
                    }
                ]
            }
        }
        assert _is_hook_installed(settings) is True


class TestHookMainValidation:
    def _run_hook_main(
        self, monkeypatch: pytest.MonkeyPatch, payload: dict, *, tmux_pane: str = ""
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["ccbot", "hook"])
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        if tmux_pane:
            monkeypatch.setenv("TMUX_PANE", tmux_pane)
        else:
            monkeypatch.delenv("TMUX_PANE", raising=False)
        monkeypatch.setattr(hook_mod, "_process_chain", lambda pid: [])
        hook_main()

    def test_missing_session_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("CCBOT_DIR", str(tmp_path))
        self._run_hook_main(
            monkeypatch,
            {"cwd": "/tmp", "hook_event_name": "SessionStart"},
        )
        assert not (tmp_path / "session_map.json").exists()

    def test_invalid_uuid_format(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("CCBOT_DIR", str(tmp_path))
        self._run_hook_main(
            monkeypatch,
            {
                "session_id": "not-a-uuid",
                "cwd": "/tmp",
                "hook_event_name": "SessionStart",
            },
        )
        assert not (tmp_path / "session_map.json").exists()

    def test_relative_cwd(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("CCBOT_DIR", str(tmp_path))
        self._run_hook_main(
            monkeypatch,
            {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "cwd": "relative/path",
                "hook_event_name": "SessionStart",
            },
        )
        assert not (tmp_path / "session_map.json").exists()

    def test_non_session_start_event(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("CCBOT_DIR", str(tmp_path))
        self._run_hook_main(
            monkeypatch,
            {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "cwd": "/tmp",
                "hook_event_name": "Stop",
            },
        )
        assert not (tmp_path / "session_map.json").exists()


SID = "550e8400-e29b-41d4-a716-446655440000"


class TestNestedGuard:
    def test_plain_session_not_skipped(self) -> None:
        chain = ["ccbot hook", "claude --dangerously-skip-permissions", "-zsh"]
        assert _should_skip_nested({}, chain) is None

    def test_print_mode_child_skipped(self) -> None:
        chain = [
            "ccbot hook",
            "claude -p summarise this",
            "/bin/zsh -c claude -p summarise this",
            "claude --resume abc",
        ]
        assert _should_skip_nested({}, chain) is not None

    def test_two_claude_ancestors_skipped(self) -> None:
        chain = ["ccbot hook", "claude", "/bin/bash", "claude --permission-mode plan"]
        assert _should_skip_nested({}, chain) is not None

    def test_sdk_entrypoint_skipped(self) -> None:
        assert _should_skip_nested({"CLAUDE_CODE_ENTRYPOINT": "sdk-py"}, []) is not None

    def test_bg_worker_skipped(self) -> None:
        assert _should_skip_nested({}, ["ccbot hook", "claude --bg"]) is not None


class TestTranscriptPath:
    def test_valid(self) -> None:
        p = f"/home/u/.claude/projects/-x/{SID}.jsonl"
        assert _valid_transcript_path(p, SID) == p

    @pytest.mark.parametrize(
        "path",
        ["relative.jsonl", f"/x/other-{SID}.jsonl", ""],
    )
    def test_invalid(self, path: str) -> None:
        assert _valid_transcript_path(path, SID) == ""


class TestHookMainMapping:
    def _run(self, monkeypatch, payload, *, tmux_pane: str = "") -> None:
        monkeypatch.setattr(sys, "argv", ["ccbot", "hook"])
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        if tmux_pane:
            monkeypatch.setenv("TMUX_PANE", tmux_pane)
        else:
            monkeypatch.delenv("TMUX_PANE", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_ENTRYPOINT", raising=False)
        monkeypatch.setattr(hook_mod, "_process_chain", lambda pid: [])
        hook_main()

    def test_writes_entry_with_transcript_path(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("CCBOT_DIR", str(tmp_path))
        monkeypatch.setattr(
            hook_mod.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(
                a, 0, stdout="ccbot:@7:proj\n", stderr=""
            ),
        )
        transcript = f"/home/u/.claude/projects/-proj/{SID}.jsonl"
        self._run(
            monkeypatch,
            {
                "session_id": SID,
                "cwd": "/proj",
                "hook_event_name": "SessionStart",
                "source": "startup",
                "transcript_path": transcript,
            },
            tmux_pane="%3",
        )
        data = json.loads((tmp_path / "session_map.json").read_text())
        assert data["ccbot:@7"] == {
            "session_id": SID,
            "cwd": "/proj",
            "window_name": "proj",
            "transcript_path": transcript,
        }

    def test_no_tmux_pane_startup_ignored(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("CCBOT_DIR", str(tmp_path))
        self._run(
            monkeypatch,
            {
                "session_id": SID,
                "cwd": "/proj",
                "hook_event_name": "SessionStart",
                "source": "startup",
            },
        )
        assert not (tmp_path / "session_map.json").exists()

    def test_no_tmux_pane_compact_keeps_existing_window(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setenv("CCBOT_DIR", str(tmp_path))
        (tmp_path / "session_map.json").write_text(
            json.dumps(
                {"ccbot:@2": {"session_id": SID, "cwd": "/old", "window_name": "w"}}
            )
        )
        self._run(
            monkeypatch,
            {
                "session_id": SID,
                "cwd": "/proj/sub",
                "hook_event_name": "SessionStart",
                "source": "compact",
            },
        )
        data = json.loads((tmp_path / "session_map.json").read_text())
        assert list(data) == ["ccbot:@2"]
        assert data["ccbot:@2"]["cwd"] == "/proj/sub"

    def test_no_tmux_pane_clear_maps_by_unique_pane(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setenv("CCBOT_DIR", str(tmp_path))
        other = "11111111-2222-3333-4444-555555555555"
        (tmp_path / "session_map.json").write_text(
            json.dumps(
                {"ccbot:@2": {"session_id": other, "cwd": "/proj", "window_name": "w"}}
            )
        )
        panes = (
            "ccbot:@0\u241e__main__\u241e/home\u241ezsh\n"
            "ccbot:@2\u241ew\u241e/proj\u241eclaude\n"
            "ccbot:@5\u241ex\u241e/other\u241eclaude\n"
        )
        monkeypatch.setattr(
            hook_mod.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=panes, stderr=""),
        )
        self._run(
            monkeypatch,
            {
                "session_id": SID,
                "cwd": "/proj",
                "hook_event_name": "SessionStart",
                "source": "clear",
            },
        )
        data = json.loads((tmp_path / "session_map.json").read_text())
        assert data["ccbot:@2"]["session_id"] == SID


class TestPhantomResumeId:
    """Claude Code >= 2.1.27x reports a fresh id on --resume but keeps
    writing to the resumed transcript; the hook must keep the real one."""

    NEW = "01a0c1a6-8f96-7522-ad8f-727bf194639f"

    def _run(self, monkeypatch, tmp_path, source: str) -> dict:
        monkeypatch.setenv("CCBOT_DIR", str(tmp_path))
        projects = tmp_path / "projects"
        real = projects / "-proj" / f"{SID}.jsonl"
        real.parent.mkdir(parents=True)
        real.write_text("{}\n")
        monkeypatch.setenv("CCBOT_CLAUDE_PROJECTS_PATH", str(projects))
        (tmp_path / "session_map.json").write_text(
            json.dumps(
                {
                    "ccbot:@3": {
                        "session_id": SID,
                        "cwd": "/proj",
                        "window_name": "w",
                        "transcript_path": str(real),
                    }
                }
            )
        )
        monkeypatch.setattr(sys, "argv", ["ccbot", "hook"])
        monkeypatch.setattr(
            sys,
            "stdin",
            io.StringIO(
                json.dumps(
                    {
                        "session_id": self.NEW,
                        "cwd": "/proj",
                        "hook_event_name": "SessionStart",
                        "source": source,
                    }
                )
            ),
        )
        monkeypatch.setenv("TMUX_PANE", "%3")
        monkeypatch.delenv("CLAUDE_CODE_ENTRYPOINT", raising=False)
        monkeypatch.setattr(hook_mod, "_process_chain", lambda pid: [])
        monkeypatch.setattr(
            hook_mod.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(
                a, 0, stdout="ccbot:@3:w\n", stderr=""
            ),
        )
        hook_main()
        return json.loads((tmp_path / "session_map.json").read_text())["ccbot:@3"]

    def test_resume_keeps_real_session(self, monkeypatch, tmp_path):
        entry = self._run(monkeypatch, tmp_path, "resume")
        assert entry["session_id"] == SID
        assert entry["transcript_path"].endswith(f"{SID}.jsonl")

    def test_clear_accepts_new_session(self, monkeypatch, tmp_path):
        entry = self._run(monkeypatch, tmp_path, "clear")
        assert entry["session_id"] == self.NEW


class TestSubagentIgnored:
    def _run(self, monkeypatch, tmp_path, payload) -> bool:
        monkeypatch.setenv("CCBOT_DIR", str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["ccbot", "hook"])
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        monkeypatch.setenv("TMUX_PANE", "%3")
        monkeypatch.delenv("CLAUDE_CODE_ENTRYPOINT", raising=False)
        monkeypatch.setattr(hook_mod, "_process_chain", lambda pid: [])
        monkeypatch.setattr(
            hook_mod.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(
                a, 0, stdout="ccbot:@3:w\n", stderr=""
            ),
        )
        hook_main()
        return (tmp_path / "session_map.json").exists()

    def test_agent_id_payload_ignored(self, monkeypatch, tmp_path):
        assert not self._run(
            monkeypatch,
            tmp_path,
            {
                "session_id": SID,
                "cwd": "/proj/.claude/worktrees/x",
                "hook_event_name": "SessionStart",
                "source": "startup",
                "agent_id": "af5e50618d2c061f4",
                "agent_type": "general-purpose",
            },
        )

    def test_sidechain_transcript_ignored(self, monkeypatch, tmp_path):
        assert not self._run(
            monkeypatch,
            tmp_path,
            {
                "session_id": SID,
                "cwd": "/proj",
                "hook_event_name": "SessionStart",
                "source": "startup",
                "transcript_path": "/x/projects/-proj/other/subagents/agent-1.jsonl",
            },
        )

    def test_main_session_written(self, monkeypatch, tmp_path):
        assert self._run(
            monkeypatch,
            tmp_path,
            {
                "session_id": SID,
                "cwd": "/proj",
                "hook_event_name": "SessionStart",
                "source": "startup",
                "transcript_path": f"/x/projects/-proj/{SID}.jsonl",
            },
        )
