"""Tests for the mode picker UI builder."""

from ccbot.handlers.callback_data import CB_MODE_CANCEL, CB_MODE_SELECT
from ccbot.handlers.directory_browser import build_mode_picker


def _buttons(keyboard):
    return [b for row in keyboard.inline_keyboard for b in row]


class TestBuildModePicker:
    def test_last_mode_first_and_marked(self):
        text, kb = build_mode_picker("/proj", "plan")
        buttons = _buttons(kb)
        assert buttons[0].callback_data == f"{CB_MODE_SELECT}plan"
        assert buttons[0].text.startswith("• ")
        assert "/proj" in text
        assert "Start" in text

    def test_all_modes_present_once_plus_cancel(self):
        _, kb = build_mode_picker("/proj", "default")
        data = [b.callback_data for b in _buttons(kb)]
        assert data[-1] == CB_MODE_CANCEL
        modes = [d[len(CB_MODE_SELECT) :] for d in data[:-1]]
        assert sorted(modes) == ["acceptEdits", "bypassPermissions", "default", "plan"]

    def test_resume_wording(self):
        text, _ = build_mode_picker("/proj", "default", resume_session_id="abc")
        assert "Resume" in text
