"""Tests for directory browser helpers and the mode picker UI builder."""

from pathlib import Path

import pytest

import ccbot.handlers.directory_browser as db
from ccbot.handlers.callback_data import CB_MODE_CANCEL, CB_MODE_SELECT
from ccbot.handlers.directory_browser import (
    as_directory,
    browser_start_path,
    build_mode_picker,
    resolve_session_ref,
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


class TestResolveSessionRef:
    SID = "550e8400-e29b-41d4-a716-446655440000"

    @pytest.fixture
    def projects(self, tmp_path, monkeypatch):
        projects = tmp_path / "projects"
        pdir = projects / "-home-u-proj"
        pdir.mkdir(parents=True)
        work = tmp_path / "work"
        work.mkdir()
        (pdir / f"{self.SID}.jsonl").write_text(
            '{"type":"user","cwd":"%s","message":{"role":"user","content":"hi"}}\n'
            % work
        )
        monkeypatch.setattr(db.config, "claude_projects_path", projects)
        return work

    def test_bare_id_reads_cwd_from_transcript(self, projects):
        ref = resolve_session_ref(self.SID)
        assert ref is not None
        assert ref.cwd == str(projects) and ref.session_id == self.SID
        assert ref.mode is None

    def test_path_and_id_with_mode(self, projects, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        ref = resolve_session_ref(f"{other} {self.SID.upper()} bypass")
        assert ref == db.SessionRef(str(other.resolve()), self.SID, "bypassPermissions")

    @pytest.mark.parametrize("prefix", ["/resume ", "resume ", "/resume --resume "])
    def test_command_prefixes(self, projects, prefix):
        ref = resolve_session_ref(prefix + self.SID)
        assert ref is not None and ref.session_id == self.SID

    @pytest.mark.parametrize(
        "text",
        [
            "hello",
            "/resume",
            "/home/u/proj",  # path only
            "550e8400-e29b-41d4-a716-446655440001",  # no transcript
            "/nonexistent/dir 550e8400-e29b-41d4-a716-446655440000",
        ],
    )
    def test_rejects(self, projects, text):
        assert resolve_session_ref(text) is None
