"""Tests for ui_choices: parsing real Claude Code 2.1.285 UI captures."""

from pathlib import Path

import pytest

from ccbot.handlers.interactive_ui import _choice_keyboard
from ccbot.terminal_parser import extract_interactive_content
from ccbot.ui_choices import label_hash, parse_choices, render_view

PANES = Path(__file__).parent / "fixtures" / "panes"


def _view(name: str, questions=None):
    pane = (PANES / f"{name}.txt").read_text()
    ui = extract_interactive_content(pane)
    assert ui is not None
    view = parse_choices(ui, pane, questions)
    assert view is not None
    return view


def test_single_question():
    v = _view("ask_single")
    assert v.title == "Deploy"
    assert v.question == "Which server should I deploy to?"
    assert [(c.number, c.label, c.description, c.kind) for c in v.choices] == [
        (1, "mh-de", "production, Germany", "option"),
        (2, "staging", "test box", "option"),
        (3, "both", "staging first, then prod", "option"),
        (4, "Type something.", "", "other"),
        (5, "Chat about this", "", "chat"),
    ]
    assert not v.multi


def test_multi_question_tabs_and_checkboxes():
    tab1 = _view("ask_multi_tab1")
    assert tab1.title == "Color"
    assert tab1.tabs == [
        ("pending", "Color"),
        ("pending", "Extras"),
        ("submit", "Submit"),
    ]
    tab2 = _view("ask_multi_tab2")
    assert tab2.title == "Extras" and tab2.multi
    assert [c.checked for c in tab2.choices[:3]] == [False, False, False]
    toggled = _view(
        "ask_multi_toggled",
        [
            {"header": "Color", "question": "Which color?"},
            {"header": "Extras", "question": "Which extras?"},
        ],
    )
    assert toggled.title == "Extras"  # from the tool input: every tab is ☒
    assert [c.checked for c in toggled.choices[:3]] == [True, False, True]
    assert tab1.signature != tab2.signature


def test_submit_tab_review():
    v = _view("ask_submit_tab")
    assert v.title == "Review your answers"
    assert v.review == [("Which color?", "green"), ("Which extras?", "cheese, onions")]
    assert [c.label for c in v.choices] == ["Submit answers", "Cancel"]


def test_permission_prompts_recover_their_header():
    bash = _view("permission_bash")
    assert bash.title == "Bash command"
    assert bash.context[0] == "echo hello > /tmp/ccbot_cap_test2.txt"
    assert bash.question == "Do you want to proceed?"
    assert [c.label for c in bash.choices] == [
        "Yes",
        "Yes, and always allow access to /tmp from this project",
        "No",
    ]
    write = _view("permission_write")
    assert write.title == "Create file" and write.context[0] == "perm_test2.md"
    # a wrapped option label is re-joined
    assert write.choices[1].label.endswith("for this session (shift+tab)")


def test_plan_approval():
    v = _view("exit_plan")
    assert v.ui_name == "ExitPlanMode" and v.title == "Plan ready"
    assert v.context == []  # the clipped plan preview is dropped
    assert [c.label for c in v.choices] == [
        "Yes, auto-accept edits",
        "Yes, manually approve edits",
        "Tell Claude what to change",
    ]


def test_model_picker_columns():
    v = _view("model_picker")
    assert v.title == "Select model"
    assert v.choices[0].label == "Default (recommended)"
    assert v.choices[3].label == "Sonnet"
    assert v.choices[3].description.startswith("Sonnet 5")
    assert "effort" not in v.choices[-1].label.lower()


@pytest.mark.parametrize(
    "name",
    [
        "ask_single",
        "ask_multi_tab2",
        "ask_submit_tab",
        "permission_bash",
        "exit_plan",
        "model_picker",
    ],
)
def test_render_and_keyboard(name):
    v = _view(name)
    text = render_view(v)
    assert text.startswith(("❓", "🔐", "📋", "❔"))
    kb = _choice_keyboard("@12", v)
    buttons = [b for row in kb.inline_keyboard for b in row]
    assert all(len(b.callback_data.encode()) <= 64 for b in buttons)
    option_cbs = [b.callback_data for b in buttons if b.callback_data.startswith("ch:")]
    for c in v.choices:
        assert f"ch:@12:{c.number}:{label_hash(c.label)}{v.fingerprint}" in option_cbs
    assert any(b.text == "⌨️ Keys" for b in buttons)


def test_permission_button_colors():
    kb = _choice_keyboard("@1", _view("permission_bash"))
    styles = {b.text: b.style for row in kb.inline_keyboard for b in row}
    assert styles["1 · Yes"] == "success"
    assert styles["3 · No"] == "danger"


def test_tabs_get_prev_next_and_other_buttons():
    kb = _choice_keyboard("@1", _view("ask_multi_tab2"))
    texts = [b.text for row in kb.inline_keyboard for b in row]
    assert "‹ Prev" in texts and "Next ›" in texts
    assert "✍️ Other…" in texts and "💬 Chat about this" in texts
    assert "☐ cheese" in texts


def test_numbered_lines_in_a_command_are_not_options():
    """Review finding: a heredoc with '1. Install' must not become buttons."""
    pane = (
        (PANES / "permission_bash.txt")
        .read_text()
        .replace(
            "   echo hello > /tmp/ccbot_cap_test2.txt",
            "   cat <<EOF\n   1. Install\n   2. Run\n   3. Deploy\n   EOF",
        )
    )
    ui = extract_interactive_content(pane)
    assert ui is not None
    v = parse_choices(ui, pane)
    assert v is not None
    assert [c.label for c in v.choices] == [
        "Yes",
        "Yes, and always allow access to /tmp from this project",
        "No",
    ]
    assert "1. Install" in v.context


def test_same_options_different_command_have_different_fingerprints():
    """Review finding: 'Yes' for one command must not approve another."""
    pane = (PANES / "permission_bash.txt").read_text()
    other = pane.replace("echo hello > /tmp/ccbot_cap_test2.txt", "git push --force")
    a = parse_choices(extract_interactive_content(pane), pane)
    b = parse_choices(extract_interactive_content(other), other)
    assert a is not None and b is not None
    assert [c.label for c in a.choices] == [c.label for c in b.choices]
    assert a.fingerprint != b.fingerprint


def test_scrolling_picker_marks_and_hidden_count():
    """2.1.286 draws "↓ 3." on the last visible row and "… +9 models"."""
    v = _view("model_picker_small")
    assert [c.number for c in v.choices] == [1, 2, 3]
    assert v.choices[2].label == "Sonnet 5.5"
    assert v.choices[2].description.startswith("Most efficient")
    assert v.hidden == 9
    assert "models" not in v.choices[-1].description  # not a continuation line
    assert "✔" in v.choices[0].label  # the current model


def test_enlarged_picker_shows_ten_models():
    v = _view("model_picker_scroll")
    assert [c.number for c in v.choices] == list(range(1, 11))
    assert v.choices[9].label == "Opus 4.7"
    assert v.hidden == 2
    kb = _choice_keyboard("@1", v)
    texts = [b.text for row in kb.inline_keyboard for b in row]
    assert "10 · Opus 4.7" in texts and "3 · Sonnet 5.5" in texts
    assert "more" in render_view(v)


def test_rendering_keeps_one_item_per_line():
    """A bare newline is a soft break in rich messages: options are list items
    and the tab strip is its own paragraph."""
    v = _view("ask_multi_tab2")
    text = render_view(v)
    assert "\n\n_" in text  # tab strip after a blank line
    assert all(line.startswith("- ") for line in text.split("\n\n")[-1].split("\n"))
    review = render_view(_view("ask_submit_tab"))
    assert "\n- " in review


def test_no_next_button_on_the_submit_tab():
    def texts(name):
        kb = _choice_keyboard("@1", _view(name))
        return [b.text for row in kb.inline_keyboard for b in row]

    assert "Next ›" not in texts("ask_submit_tab")
    assert "‹ Prev" in texts("ask_submit_tab")
    assert "Next ›" in texts("ask_multi_tab1")
