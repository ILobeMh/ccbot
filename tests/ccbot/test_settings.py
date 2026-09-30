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
    """Home → section → picker navigation (all within one edited message)."""

    def _buttons(self, kb):
        return [b for row in kb.inline_keyboard for b in row]

    def test_home_has_one_button_per_section(self, cfg):
        text, kb = render_settings()
        from ccbot.handlers.settings_topic import SECTIONS

        groups = [g for g in SECTIONS if any(s.group == g for s in st.SETTINGS)]
        assert set(groups) == {s.group for s in st.SETTINGS}  # nothing unlisted
        buttons = self._buttons(kb)
        assert [b.callback_data for b in buttons] == [f"sg:sec:{g}" for g in groups]
        for g in groups:
            assert g in text
        convert_markdown(text)

    def test_every_setting_reachable_and_callbacks_short(self, cfg):
        from ccbot.handlers.settings_topic import render_picker, render_section

        seen: set[str] = set()
        for group in dict.fromkeys(s.group for s in st.SETTINGS):
            text, kb = render_section(group)
            convert_markdown(text)
            for b in self._buttons(kb):
                assert b.callback_data and len(b.callback_data.encode()) <= 64
                if b.callback_data.startswith("st:"):
                    seen.add(b.callback_data[3:])
                elif b.callback_data.startswith("sg:pick:"):
                    key = b.callback_data[len("sg:pick:") :]
                    seen.add(key)
                    _, pkb = render_picker(key)
                    for pb in self._buttons(pkb):
                        assert len(pb.callback_data.encode()) <= 64
        assert seen == {s.key for s in st.SETTINGS}

    def test_picker_highlights_current_and_sets_value(self, cfg):
        from ccbot.handlers.settings_topic import SettingsTopic, render_picker

        config.tool_output = "full"
        _, kb = render_picker("tool_output")
        opts = self._buttons(kb)[:-1]  # last one is « Back
        assert [b.style for b in opts] == ["primary", None, None]
        assert opts[0].text.startswith("● ")

        text, kb, toast = SettingsTopic._navigate(opts[2].callback_data)
        assert config.tool_output == "summary"
        assert toast == "Tool output: status only"
        assert "**Tool output** — status only" in text

    def test_bool_toggles_in_place(self, cfg):
        from ccbot.handlers.settings_topic import SettingsTopic

        config.show_thinking = True
        _, kb, _ = SettingsTopic._navigate("st:show_thinking")
        assert config.show_thinking is False
        (btn,) = [b for b in self._buttons(kb) if b.callback_data == "st:show_thinking"]
        assert btn.text.startswith("⬜") and btn.style is None
        SettingsTopic._navigate("st:show_thinking")
        assert config.show_thinking is True

    def test_unknown_key_raises_for_alert(self, cfg):
        from ccbot.handlers.settings_topic import SettingsTopic

        with pytest.raises(KeyError):
            SettingsTopic._navigate("sg:pick:nope")
