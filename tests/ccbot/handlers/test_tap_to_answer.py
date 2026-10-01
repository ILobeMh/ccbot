"""Tests for interactive_ui.answer_choice: tap-to-answer against the pane."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ccbot.handlers import interactive_ui as iu

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

    async def capture_expanded(window_id, before=None, rows=60, cols=0):
        # the enlarged capture (scripted by tests that need it), else as is
        scripted = state.get("expanded")
        if scripted:
            return scripted.pop(0) if len(scripted) > 1 else scripted[0]
        return await capture(window_id)

    tmux = MagicMock()
    tmux.capture_pane = capture
    tmux.capture_pane_expanded = capture_expanded
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


def _token(pane: str, number: int) -> str:
    """The callback token a button for option ``number`` on ``pane`` carries."""
    from ccbot.terminal_parser import extract_interactive_content
    from ccbot.ui_choices import parse_choices

    view = parse_choices(extract_interactive_content(pane), pane)
    choice = next(c for c in view.choices if c.number == number)
    return iu.choice_token(view, choice)


@pytest.mark.asyncio
async def test_tap_types_the_digit_and_records_the_answer(env):
    env["panes"] = [_pane("ask_single"), IDLE]
    iu._interactive_msgs[(1, 42)] = 77
    toast = await iu.answer_choice(
        AsyncMock(), 1, 42, "@5", 2, _token(_pane("ask_single"), 2)
    )
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
    toast = await iu.answer_choice(
        AsyncMock(), 1, 42, "@5", 2, _token(_pane("ask_single"), 2)
    )
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
    await iu.answer_choice(AsyncMock(), 1, 42, "@5", 4, _token(picker, 4))
    assert env["keys"] == ["4", "Enter"]


@pytest.mark.asyncio
async def test_question_digit_is_not_followed_by_enter(env):
    # answering tab 1 advances to tab 2, whose cursor is on option 1: an extra
    # Enter would answer the next question — must not happen
    env["panes"] = [_pane("ask_multi_tab1"), _pane("ask_multi_tab2")]
    iu._interactive_msgs[(1, 42)] = 77
    await iu.answer_choice(
        AsyncMock(), 1, 42, "@5", 1, _token(_pane("ask_multi_tab1"), 1)
    )
    assert env["keys"] == ["1"]
    assert "Extras" in env["edits"][-1][1]  # message now shows the next tab


@pytest.mark.asyncio
async def test_other_asks_to_type(env):
    env["panes"] = [_pane("ask_single")]
    iu._interactive_msgs[(1, 42)] = 77
    toast = await iu.answer_choice(
        AsyncMock(), 1, 42, "@5", 4, _token(_pane("ask_single"), 4)
    )
    assert env["keys"] == ["4"]
    assert "type your answer" in toast


def test_keypad_toggle():
    iu._keypad_mode.clear()
    assert iu.toggle_keypad(1, 42) is True
    assert iu.toggle_keypad(1, 42) is False


@pytest.mark.asyncio
async def test_yes_for_one_command_never_approves_another(env):
    """Review finding: a late 'Yes' tap on an old prompt is refused."""
    first = _pane("permission_bash")
    second = first.replace("echo hello > /tmp/ccbot_cap_test2.txt", "git push --force")
    env["panes"] = [second]
    iu._interactive_msgs[(1, 42)] = 77
    toast = await iu.answer_choice(AsyncMock(), 1, 42, "@5", 1, _token(first, 1))
    assert env["keys"] == []
    assert "changed" in toast


@pytest.mark.asyncio
async def test_answering_flag_is_set_while_typing(env):
    seen = []

    async def send_keys(window_id, text, enter=True, literal=True):
        seen.append(iu.is_answering(1, 42))
        return True

    iu.tmux_manager.send_keys = send_keys
    env["panes"] = [_pane("ask_single"), IDLE]
    await iu.answer_choice(AsyncMock(), 1, 42, "@5", 1, _token(_pane("ask_single"), 1))
    assert seen == [True]
    assert not iu.is_answering(1, 42)


@pytest.mark.asyncio
async def test_option_ten_walks_the_cursor_and_never_types_digits(env):
    """Typing "10" would pick option 1 first (a digit selects at once)."""
    big = _pane("model_picker_scroll")
    at_ten = big.replace("   ❯ 1.  ", "     1.  ").replace(
        "   ↓ 10. Opus 4.7", "   ❯ 10. Opus 4.7"
    )
    env["panes"] = [big, at_ten, IDLE]
    # the small capture has hidden options: the enlarged one is used instead
    env["expanded"] = [big, at_ten]
    toast = await iu.answer_choice(AsyncMock(), 1, 42, "@5", 10, _token(big, 10))
    assert env["keys"] == ["Down"] * 9 + ["Enter"]
    assert toast == "✓ Opus 4.7"


@pytest.mark.asyncio
async def test_unreachable_option_is_reported(env):
    big = _pane("model_picker_scroll")
    env["panes"] = [big]
    env["expanded"] = [big, big]  # the cursor never moves
    toast = await iu.answer_choice(AsyncMock(), 1, 42, "@5", 10, _token(big, 10))
    assert "Keys" in toast and "Enter" not in env["keys"]


def _effort_at(pane: str, level: str) -> str:
    """The captured /effort slider with its ▲ under ``level``."""
    lines = pane.split("\n")
    row = next(i for i, ln in enumerate(lines) if "▲" in ln)
    labels = next(ln for ln in lines if "xhigh" in ln)
    col = labels.index(level) + len(level) // 2
    bar = lines[row].replace("▲", "─")
    lines[row] = bar[:col] + "▲" + bar[col + 1 :]
    return "\n".join(lines)


@pytest.mark.asyncio
async def test_effort_level_walks_the_slider_then_applies_for_the_session(env):
    base = _pane("effort_picker")  # medium
    xhigh = _effort_at(base, "xhigh")
    env["panes"] = [base, xhigh, IDLE]
    env["expanded"] = [base, xhigh]
    iu._interactive_msgs[(1, 42)] = 77
    toast = await iu.answer_choice(AsyncMock(), 1, 42, "@5", 4, _token(base, 4))
    assert env["keys"] == ["Right", "Right", "s"]  # medium → high → xhigh
    assert toast == "✓ xhigh"
    assert "Effort → xhigh" in env["edits"][-1][1].replace("\\", "")


@pytest.mark.asyncio
async def test_effort_not_applied_if_the_slider_did_not_move(env):
    base = _pane("effort_picker")
    env["panes"] = [base]
    env["expanded"] = [base, base]
    toast = await iu.answer_choice(AsyncMock(), 1, 42, "@5", 5, _token(base, 5))
    assert "Keys" in toast and "s" not in env["keys"]
