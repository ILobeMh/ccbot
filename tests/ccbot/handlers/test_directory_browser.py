"""Tests for directory browser helpers and the mode picker UI builder."""

from pathlib import Path

import pytest

import ccbot.handlers.directory_browser as db
from ccbot.handlers.callback_data import CB_MODE_CANCEL, CB_MODE_SELECT
from ccbot.handlers.directory_browser import (
    as_directory,
    browser_start_path,
    build_mode_picker,
)


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


class TestAsDirectory:
    def test_absolute_existing_dir(self, tmp_path):
        assert as_directory(f"  {tmp_path}  ") == str(tmp_path.resolve())

    def test_tilde_expands(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "proj").mkdir()
        assert as_directory("~/proj") == str((tmp_path / "proj").resolve())

    def test_quoted_path(self, tmp_path):
        assert as_directory(f'"{tmp_path}"') == str(tmp_path.resolve())

    @pytest.mark.parametrize(
        "text",
        ["hello", "relative/path", "/definitely/not/here/xyz", "/tmp\nmore", ""],
    )
    def test_rejects(self, text):
        assert as_directory(text) is None

    def test_file_is_not_dir(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        assert as_directory(str(f)) is None


class TestBrowserStartPath:
    def test_default_dir_used_when_valid(self, tmp_path, monkeypatch):
        monkeypatch.setattr(db.config, "default_dir", str(tmp_path))
        assert browser_start_path() == str(tmp_path.resolve())

    def test_falls_back_to_cwd(self, tmp_path, monkeypatch):
        monkeypatch.setattr(db.config, "default_dir", str(tmp_path / "missing"))
        assert browser_start_path() == str(Path.cwd())
