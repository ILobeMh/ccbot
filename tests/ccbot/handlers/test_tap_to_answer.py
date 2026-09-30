"""Tests for interactive_ui.answer_choice: tap-to-answer against the pane."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ccbot.handlers import interactive_ui as iu
from ccbot.ui_choices import label_hash

PANES = Path(__file__).parent.parent / "fixtures" / "panes"
IDLE = "done\n" + "─" * 40 + "\n❯ \n" + "─" * 40 + "\n  ⏸ manual mode on\n"


@pytest.fixture
def env(monkeypatch):
    """Scripted pane captures + recorded key presses and edits."""
    state = {"panes": [], "keys": []}

    async def capture(window_id, with_ansi=False):
        panes = state["panes"]
        return panes.pop(0) if len(panes) > 1 else panes[0]

    async def send_keys(window_id, text, enter=True, literal=True):
        state["keys"].append(text)
        return True

    tmux = MagicMock()
    tmux.capture_pane = capture
    tmux.send_keys = send_keys
    tmux.find_window_by_id = AsyncMock(return_value=MagicMock(window_id="@5"))
    sm = MagicMock()
    sm.resolve_chat_id.return_value = -100
    edits: list[tuple] = []

    async def edit(bot, chat_id, message_id, text, keyboard, markdown):
        edits.append((message_id, text, keyboard))
        return True

    monkeypatch.setattr(iu, "tmux_manager", tmux)
    monkeypatch.setattr(iu, "session_manager", sm)
    monkeypatch.setattr(iu, "_edit", edit)
    monkeypatch.setattr(iu, "mark_ui", AsyncMock())
    monkeypatch.setattr(iu.asyncio, "sleep", AsyncMock())
    iu._interactive_msgs.clear()
    iu._interactive_mode.clear()
    iu._keypad_mode.clear()
    state["edits"] = edits
    yield state
    iu._interactive_msgs.clear()
    iu._interactive_mode.clear()


def _pane(name: str) -> str:
    return (PANES / f"{name}.txt").read_text()


@pytest.mark.asyncio
async def test_tap_types_the_digit_and_records_the_answer(env):
    env["panes"] = [_pane("ask_single"), IDLE]
    iu._interactive_msgs[(1, 42)] = 77
    toast = await iu.answer_choice(AsyncMock(), 1, 42, "@5", 2, label_hash("staging"))
    assert env["keys"] == ["2"]
    assert toast == "✓ staging"
    # UI gone → the question message becomes a one-line record (no keyboard)
    msg_id, text, keyboard = env["edits"][-1]
    assert msg_id == 77 and keyboard is None
    assert "Deploy → staging" in text.replace("\\", "")
    assert (1, 42) not in iu._interactive_msgs


@pytest.mark.asyncio
async def test_stale_tap_is_refused(env):
    env["panes"] = [_pane("ask_multi_tab1")]  # a different question is showing
    iu._interactive_msgs[(1, 42)] = 77
    toast = await iu.answer_choice(AsyncMock(), 1, 42, "@5", 2, label_hash("staging"))
    assert env["keys"] == []
    assert "changed" in toast
    # …and the message is refreshed to the current question
    assert "Color" in env["edits"][-1][1]


@pytest.mark.asyncio
async def test_picker_digit_is_confirmed_with_enter(env):
    picker = _pane("model_picker")
    moved = picker.replace("   ❯ 6. Opus", "     6. Opus").replace(
        "     4. Sonnet", "   ❯ 4. Sonnet"
    )
    env["panes"] = [picker, moved, IDLE]
    await iu.answer_choice(AsyncMock(), 1, 42, "@5", 4, label_hash("Sonnet"))
    assert env["keys"] == ["4", "Enter"]


@pytest.mark.asyncio
async def test_question_digit_is_not_followed_by_enter(env):
    # answering tab 1 advances to tab 2, whose cursor is on option 1: an extra
    # Enter would answer the next question — must not happen
    env["panes"] = [_pane("ask_multi_tab1"), _pane("ask_multi_tab2")]
    iu._interactive_msgs[(1, 42)] = 77
    await iu.answer_choice(AsyncMock(), 1, 42, "@5", 1, label_hash("red"))
    assert env["keys"] == ["1"]
    assert "Extras" in env["edits"][-1][1]  # message now shows the next tab


@pytest.mark.asyncio
async def test_other_asks_to_type(env):
    env["panes"] = [_pane("ask_single")]
    iu._interactive_msgs[(1, 42)] = 77
    toast = await iu.answer_choice(
        AsyncMock(), 1, 42, "@5", 4, label_hash("Type something.")
    )
    assert env["keys"] == ["4"]
    assert "type your answer" in toast


def test_keypad_toggle():
    iu._keypad_mode.clear()
    assert iu.toggle_keypad(1, 42) is True
    assert iu.toggle_keypad(1, 42) is False
