"""Tests for runtime settings persistence and the settings dashboard."""

import json
from datetime import datetime

import pytest

import ccbot.settings as st
from ccbot.config import config
from ccbot.handlers.settings_topic import render_settings
from ccbot.markdown_v2 import convert_markdown


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", tmp_path)
    for s in st.SETTINGS:
        monkeypatch.setattr(config, s.key, getattr(config, s.key))  # restore after
    monkeypatch.setattr(st, "_forget_last_modes", lambda v: None)
    for s in st.SETTINGS:
        if s.on_change is not None:
            monkeypatch.setattr(s, "on_change", lambda v: None)
    return tmp_path / "settings.json"


class TestSetAndCycle:
    def test_bool_toggle_persists(self, cfg):
        config.show_thinking = True
        assert st.cycle("show_thinking") is False
        assert config.show_thinking is False
        assert json.loads(cfg.read_text())["show_thinking"] is False

    def test_choice_cycles_and_wraps(self, cfg):
        config.thinking_max_chars = 500
        assert st.cycle("thinking_max_chars") == 1500
        st.cycle("thinking_max_chars")
        assert st.cycle("thinking_max_chars") == 0
        assert st.cycle("thinking_max_chars") == 500

    def test_float_choice_matches_int_from_json(self, cfg):
        config.shell_timeout = 120.0
        assert st.cycle("shell_timeout") == 300.0

    def test_invalid_value_rejected(self, cfg):
        with pytest.raises(ValueError):
            st.set_value("thinking_max_chars", 42)


class TestLoad:
    def test_applies_valid_and_ignores_invalid(self, cfg):
        cfg.write_text(
            json.dumps(
                {
                    "show_tool_calls": False,
                    "shell_timeout": 300,
                    "thinking_max_chars": "nope",
                    "unknown_key": 1,
                    "claude_permission_mode": "plan",
                }
            )
        )
        config.show_tool_calls = True
        config.shell_timeout = 120.0
        config.thinking_max_chars = 500
        st.load()
        assert config.show_tool_calls is False
        assert config.shell_timeout == 300.0
        assert config.thinking_max_chars == 500
        assert config.claude_permission_mode == "plan"

    def test_missing_or_broken_file(self, cfg):
        st.load()  # no file
        cfg.write_text("{broken")
        st.load()  # must not raise


class TestQuietHours:
    @pytest.mark.parametrize(
        "window,hour,expected",
        [
            ("", 3, False),
            ("23-08", 23, True),
            ("23-08", 3, True),
            ("23-08", 8, False),
            ("23-08", 12, False),
            ("09-17", 12, True),
            ("09-17", 20, False),
            ("bad", 3, False),
        ],
    )
    def test_windows(self, cfg, window, hour, expected):
        config.quiet_hours = window
        now = datetime(2026, 1, 1, hour, 30)
        assert st.in_quiet_hours(now) is expected


class TestDashboard:
    def test_renders_all_settings(self, cfg):
        text, kb = render_settings()
        buttons = [b for row in kb.inline_keyboard for b in row]
        assert len(buttons) == len(st.SETTINGS)
        assert all(b.callback_data.startswith("st:") for b in buttons)
        assert all(len(b.callback_data) <= 64 for b in buttons)
        convert_markdown(text)
        for s in st.SETTINGS:
            assert s.label in text
