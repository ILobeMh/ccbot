"""Tests against real Claude Code 2.1.272 pane captures (tests/ccbot/fixtures/panes)."""

from pathlib import Path

import pytest

from ccbot.terminal_parser import (
    AUTO_ANSWER_DIALOGS,
    extract_interactive_content,
    find_menu_option,
    has_update_pending,
    is_blocking_dialog,
    is_interactive_ui,
    is_prompt_ready,
    is_working,
    parse_permission_mode,
    parse_status_line,
)

FIXTURES = Path(__file__).parent / "fixtures" / "panes"


def load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class TestTrustDialog:
    def test_detected(self):
        ui = extract_interactive_content(load("trust_dialog.txt"))
        assert ui is not None
        assert ui.name == "TrustDialog"
        assert "Yes, I trust this folder" in ui.content
        assert "Enter to confirm" in ui.content

    def test_is_blocking(self):
        assert is_blocking_dialog(load("trust_dialog.txt"))
        assert not is_prompt_ready(load("trust_dialog.txt"))

    def test_menu_option_offsets(self):
        # Cursor on "No, exit" (first option), target is one line below
        offs = find_menu_option(
            load("trust_dialog.txt"), AUTO_ANSWER_DIALOGS["TrustDialog"] or ""
        )
        assert offs is not None
        cursor, target = offs
        assert target - cursor == 1

    def test_menu_option_missing(self):
        assert find_menu_option(load("trust_dialog.txt"), "nope") is None


class TestReadyPrompt:
    @pytest.mark.parametrize(
        "name,mode",
        [
            ("ready_bypass.txt", "bypassPermissions"),
            ("ready_manual_mode.txt", "default"),
        ],
    )
    def test_prompt_ready_and_mode(self, name: str, mode: str):
        pane = load(name)
        assert is_prompt_ready(pane)
        assert parse_permission_mode(pane) == mode
        assert not is_interactive_ui(pane)
        assert not is_working(pane)

    def test_ready_with_ghost_suggestion(self):
        # "❯ check if it's done yet" is a suggestion, still idle
        pane = load("done_shell_running.txt")
        assert is_prompt_ready(pane)
        assert not is_working(pane)

    def test_done_status_strips_noise(self):
        assert parse_status_line(load("done_shell_running.txt")) == "Churned for 8s"
        assert parse_status_line(load("ready_manual_mode.txt")) == "Churned for 4s"

    def test_permission_mode_footer_variants(self):
        assert (
            parse_permission_mode("x\n  ⏸ plan mode on (shift+tab to cycle)") == "plan"
        )
        assert (
            parse_permission_mode(
                "x\n  ⏵⏵ accept edits on · 1 shell · esc to interrupt"
            )
            == "acceptEdits"
        )
        assert parse_permission_mode("no footer here") is None


class TestBusyWithTipBlock:
    def test_status_found_past_tip(self):
        pane = load("busy_tip_block.txt")
        assert parse_status_line(pane) == "Moseying… (7s · ↓ 337 tokens)"
        assert is_working(pane)
        assert parse_permission_mode(pane) == "bypassPermissions"
        assert not is_interactive_ui(pane)


class TestMisc:
    def test_update_pending(self):
        assert has_update_pending("  ✔ Update installed · Restart to update\n────")
        assert not has_update_pending(load("ready_bypass.txt"))

    def test_generic_modal_fallback(self):
        pane = "\n".join(
            [
                "Some new dialog we have never seen",
                " ❯ Option A",
                "   Option B",
                "",
                " ↑/↓ to navigate · Enter to confirm · Esc to cancel",
            ]
        )
        ui = extract_interactive_content(pane)
        assert ui is not None
        assert ui.name == "Modal"
        assert "Option A" in ui.content
        assert is_blocking_dialog(pane)

    def test_modal_fallback_not_triggered_by_idle_prompt(self):
        assert extract_interactive_content(load("ready_bypass.txt")) is None

    def test_labelled_separator_is_chrome(self):
        pane = "\n".join(
            [
                "✶ Thinking… (2s)",
                "──────────────── ultracode ─────────────────────",
                "❯ ",
                "────────────────────────────────────────────────",
                "  ⏵⏵ auto mode on",
            ]
        )
        assert parse_status_line(pane) == "Thinking… (2s)"
        assert is_prompt_ready(pane)
        assert parse_permission_mode(pane) == "auto"
