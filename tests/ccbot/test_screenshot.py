"""Tests for screenshot font routing and rendering."""

import io

import pytest
from PIL import Image

from ccbot import screenshot as ss


class TestFontTier:
    @pytest.mark.parametrize(
        "ch,tier",
        [
            ("a", 0),  # Meslo
            ("─", 0),  # box drawing in Meslo
            ("✔", 0),  # Nerd Font glyph in Meslo
            ("س", 1),  # Arabic → Vazirmatn
            ("گ", 1),  # Persian-specific letter
            ("中", 2),  # CJK → Noto
            ("⏵", 3),  # only Symbola
            ("⎿", 2),  # Noto before Symbola
            (" ", 0),  # whitespace never splits runs
        ],
    )
    def test_default_chain(self, ch: str, tier: int, monkeypatch):
        monkeypatch.delenv("CCBOT_SCREENSHOT_FONT", raising=False)
        assert ss._font_tier(ch, tuple(ss._font_paths())) == tier

    def test_persian_words_form_single_runs(self):
        segs = ss._split_line_segments_plain("abc سلام دنیا xyz")
        tiers = [t for _, t in segs]
        assert tiers == [0, 1, 0, 1, 0]
        assert segs[1][0] == "سلام"

    def test_missing_override_is_ignored(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CCBOT_SCREENSHOT_FONT", str(tmp_path / "nope.ttf"))
        assert ss._font_paths()[0].name == "MesloLGSNF-Regular.ttf"

    def test_override_becomes_tier_zero(self, monkeypatch):
        custom = ss._FONTS_DIR / "Symbola.ttf"
        monkeypatch.setenv("CCBOT_SCREENSHOT_FONT", str(custom))
        assert ss._font_paths()[0] == custom


class TestRender:
    @pytest.mark.asyncio
    async def test_mixed_script_png(self):
        png = await ss.text_to_image(
            "┌──┐\n│ab│ سلام \x1b[31mred\x1b[0m 中文 ⏵\n└──┘", font_size=16
        )
        img = Image.open(io.BytesIO(png))
        assert img.format == "PNG"
        assert img.height == 16 * 1.4 // 1 * 3 + 32
        assert img.width > 32

    @pytest.mark.asyncio
    async def test_font_size_env(self, monkeypatch):
        monkeypatch.setenv("CCBOT_SCREENSHOT_FONT_SIZE", "12")
        assert ss._default_font_size() == 12
        monkeypatch.setenv("CCBOT_SCREENSHOT_FONT_SIZE", "999")
        assert ss._default_font_size() == 28
