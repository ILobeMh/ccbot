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
        assert f"ch:@12:{c.number}:{label_hash(c.label)}" in option_cbs
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
