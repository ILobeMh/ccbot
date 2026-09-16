"""Terminal text → PNG screenshot renderer.

Converts captured tmux pane text (with optional ANSI color codes) into a
dark-background PNG image. Supports full ANSI color parsing (16/256/RGB)
and a font fallback chain matched to the user's terminal setup:
  1. MesloLGS NF — Latin, box-drawing, powerline / Nerd Font icons
     (override with CCBOT_SCREENSHOT_FONT=/path/to/font.ttf)
  2. Vazirmatn — Arabic script (Persian); shaped by Pillow's raqm layout
  3. Noto Sans Mono CJK SC — CJK characters
  4. Symbola — remaining special symbols

Glyph coverage is read from each font's cmap (fontTools) so characters are
routed to the first font that actually has them. Text is drawn per run of
same-font characters, advanced in whole terminal cells so columns and box
drawing stay aligned.

Key function: text_to_image(text, font_size, with_ansi) → PNG bytes.
"""

import asyncio
import functools
import io
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

_FONTS_DIR = Path(__file__).parent / "fonts"

# Font fallback chain (highest priority first). Tier 0 can be replaced via
# CCBOT_SCREENSHOT_FONT; the rest stay as fallbacks.
_DEFAULT_FONT_PATHS: list[Path] = [
    _FONTS_DIR / "MesloLGSNF-Regular.ttf",  # Apache-2.0 + Nerd Fonts (MIT)
    _FONTS_DIR / "Vazirmatn-Variable.ttf",  # OFL-1.1
    _FONTS_DIR / "NotoSansMonoCJKsc-Regular.otf",  # OFL-1.1
    _FONTS_DIR / "Symbola.ttf",  # free license
]


def _font_paths() -> list[Path]:
    override = os.environ.get("CCBOT_SCREENSHOT_FONT", "").strip()
    paths = list(_DEFAULT_FONT_PATHS)
    if override:
        custom = Path(override).expanduser()
        if custom.is_file():
            paths.insert(0, custom)
        else:
            logger.warning("CCBOT_SCREENSHOT_FONT not found: %s", custom)
    return paths


def _default_font_size() -> int:
    # Deferred import: config is heavy and screenshot is imported early
    from .config import config

    return int(config.screenshot_font_size)


@functools.lru_cache(maxsize=8)
def _coverage(path: Path) -> frozenset[int]:
    """Codepoints a font file has glyphs for (from its cmap)."""
    try:
        with TTFont(str(path), lazy=True) as tt:
            cmap = tt.getBestCmap() or {}
            return frozenset(cmap.keys())
    except Exception as e:
        logger.warning("Cannot read cmap of %s: %s", path, e)
        return frozenset()


# Codepoint → tier cache (per font chain)
_tier_cache: dict[tuple[tuple[Path, ...], int], int] = {}


def _font_tier(ch: str, paths: tuple[Path, ...] | None = None) -> int:
    """Index of the first font in the chain that has a glyph for ``ch``.

    Whitespace and control characters stay in tier 0 so they don't split
    runs; unknown characters fall back to tier 0 (rendered as .notdef).
    """
    if paths is None:
        paths = tuple(_font_paths())
    cp = ord(ch)
    key = (paths, cp)
    cached = _tier_cache.get(key)
    if cached is not None:
        return cached
    tier = 0
    if cp > 0x20 and not ch.isspace():
        for i, path in enumerate(paths):
            if cp in _coverage(path):
                tier = i
                break
    _tier_cache[key] = tier
    return tier


# ANSI color mapping (basic 16 colors)
_ANSI_COLORS: dict[int, tuple[int, int, int]] = {
    # Standard colors (30-37, 40-47)
    0: (0, 0, 0),  # Black
    1: (205, 49, 49),  # Red
    2: (13, 188, 121),  # Green
    3: (229, 229, 16),  # Yellow
    4: (36, 114, 200),  # Blue
    5: (188, 63, 188),  # Magenta
    6: (17, 168, 205),  # Cyan
    7: (229, 229, 229),  # White
    # Bright colors (90-97, 100-107)
    8: (102, 102, 102),  # Bright Black
    9: (241, 76, 76),  # Bright Red
    10: (35, 209, 139),  # Bright Green
    11: (245, 245, 67),  # Bright Yellow
    12: (59, 142, 234),  # Bright Blue
    13: (214, 112, 214),  # Bright Magenta
    14: (41, 184, 219),  # Bright Cyan
    15: (255, 255, 255),  # Bright White
}

# Default colors for terminals
_DEFAULT_FG = (212, 212, 212)  # Light gray
_DEFAULT_BG = (30, 30, 30)  # Dark gray


@dataclass
class TextStyle:
    """Text styling information from ANSI codes."""

    fg_color: tuple[int, int, int] = _DEFAULT_FG
    bg_color: tuple[int, int, int] | None = None


@dataclass
class StyledSegment:
    """A text segment with its styling."""

    text: str
    style: TextStyle
    font_tier: int


def _load_font(path: Path, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a TrueType/OpenType font, falling back to Pillow default."""
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        logger.warning("Failed to load font %s, using Pillow default", path)
        return ImageFont.load_default()


def _parse_ansi_line(line: str) -> list[StyledSegment]:
    """Parse a line with ANSI escape codes into styled segments."""
    # ANSI escape sequence pattern
    ansi_pattern = re.compile(r"\x1b\[([0-9;]*)m")

    segments: list[StyledSegment] = []
    current_style = TextStyle()
    pos = 0

    for match in ansi_pattern.finditer(line):
        # Add text before this escape code
        text_before = line[pos : match.start()]
        if text_before:
            # Split by font tier
            for seg_text, tier in _split_line_segments_plain(text_before):
                if seg_text:
                    segments.append(StyledSegment(seg_text, current_style, tier))

        # Parse escape code
        codes = match.group(1)
        if codes:
            current_style = _apply_ansi_codes(current_style, codes)
        else:
            # Empty code means reset
            current_style = TextStyle()

        pos = match.end()

    # Add remaining text after last escape code
    text_after = line[pos:]
    if text_after:
        for seg_text, tier in _split_line_segments_plain(text_after):
            if seg_text:
                segments.append(StyledSegment(seg_text, current_style, tier))

    return segments if segments else [StyledSegment("", TextStyle(), 0)]


def _apply_ansi_codes(style: TextStyle, codes: str) -> TextStyle:
    """Apply ANSI color codes to a text style."""
    # Create a new style (copy current)
    new_style = TextStyle(
        fg_color=style.fg_color,
        bg_color=style.bg_color,
    )

    parts = [int(c) for c in codes.split(";") if c]
    i = 0
    while i < len(parts):
        code = parts[i]

        if code == 0:  # Reset
            new_style = TextStyle()
        elif 30 <= code <= 37:  # Foreground color
            new_style.fg_color = _ANSI_COLORS[code - 30]
        elif code == 38:  # Extended foreground color
            if i + 1 < len(parts) and parts[i + 1] == 5:  # 256 color
                if i + 2 < len(parts):
                    color_idx = parts[i + 2] % 256
                    if color_idx < 16:
                        new_style.fg_color = _ANSI_COLORS[color_idx]
                    else:
                        # Approximate 256 colors (simplified)
                        new_style.fg_color = _approximate_256_color(color_idx)
                    i += 2
            elif i + 1 < len(parts) and parts[i + 1] == 2:  # RGB color
                if i + 4 < len(parts):
                    new_style.fg_color = (parts[i + 2], parts[i + 3], parts[i + 4])
                    i += 4
        elif code == 39:  # Default foreground
            new_style.fg_color = _DEFAULT_FG
        elif 40 <= code <= 47:  # Background color
            new_style.bg_color = _ANSI_COLORS[code - 40]
        elif code == 48:  # Extended background color
            if i + 1 < len(parts) and parts[i + 1] == 5:  # 256 color
                if i + 2 < len(parts):
                    color_idx = parts[i + 2] % 256
                    if color_idx < 16:
                        new_style.bg_color = _ANSI_COLORS[color_idx]
                    else:
                        new_style.bg_color = _approximate_256_color(color_idx)
                    i += 2
            elif i + 1 < len(parts) and parts[i + 1] == 2:  # RGB color
                if i + 4 < len(parts):
                    new_style.bg_color = (parts[i + 2], parts[i + 3], parts[i + 4])
                    i += 4
        elif code == 49:  # Default background
            new_style.bg_color = None
        elif 90 <= code <= 97:  # Bright foreground color
            new_style.fg_color = _ANSI_COLORS[code - 90 + 8]
        elif 100 <= code <= 107:  # Bright background color
            new_style.bg_color = _ANSI_COLORS[code - 100 + 8]

        i += 1

    return new_style


def _approximate_256_color(idx: int) -> tuple[int, int, int]:
    """Approximate a 256-color palette index to RGB."""
    if idx < 16:
        return _ANSI_COLORS[idx]
    elif idx < 232:
        # 216 color cube: 16 + 36*r + 6*g + b
        idx -= 16
        r = (idx // 36) * 51
        g = ((idx % 36) // 6) * 51
        b = (idx % 6) * 51
        return (r, g, b)
    else:
        # Grayscale: 232-255
        gray = 8 + (idx - 232) * 10
        return (gray, gray, gray)


def _split_line_segments_plain(line: str) -> list[tuple[str, int]]:
    """Split a line into (text, font_tier) segments.

    Consecutive characters sharing the same tier are merged.
    """
    if not line:
        return [("", 0)]
    segments: list[tuple[str, int]] = []
    cur_tier = _font_tier(line[0])
    start = 0
    for i in range(1, len(line)):
        tier = _font_tier(line[i])
        if tier != cur_tier:
            segments.append((line[start:i], cur_tier))
            cur_tier = tier
            start = i
    segments.append((line[start:], cur_tier))
    return segments


async def text_to_image(
    text: str, font_size: int | None = None, with_ansi: bool = True
) -> bytes:
    """Render monospace text onto a dark-background image and return PNG bytes.

    Args:
        text: The text to render (may contain ANSI color codes)
        font_size: Font size in pixels (default: CCBOT_SCREENSHOT_FONT_SIZE or 28)
        with_ansi: If True, parse and render ANSI color codes

    Returns:
        PNG image bytes
    """
    size = font_size or _default_font_size()

    def _render_image() -> bytes:
        paths = _font_paths()
        fonts = [_load_font(p, size) for p in paths]

        lines = text.split("\n")
        padding = 16

        # Parse lines into styled segments
        if with_ansi:
            line_segments = [_parse_ansi_line(line) for line in lines]
        else:
            # Legacy plain text mode
            line_segments_plain = [_split_line_segments_plain(line) for line in lines]
            line_segments = [
                [
                    StyledSegment(seg_text, TextStyle(), tier)
                    for seg_text, tier in segments
                ]
                for segments in line_segments_plain
            ]

        # Terminal cell geometry from the primary (monospace) font
        primary = fonts[0]
        cell_w = max(1.0, float(primary.getlength("M")))
        line_height = int(size * 1.4)
        ascent = (
            primary.getmetrics()[0]
            if isinstance(primary, ImageFont.FreeTypeFont)
            else size
        )
        baseline_offset = (line_height - size) // 2 + ascent

        def advance(seg: StyledSegment) -> float:
            """Width a segment occupies, snapped to whole terminal cells."""
            if not seg.text:
                return 0.0
            f = fonts[seg.font_tier]
            if seg.font_tier == 0:
                return len(seg.text) * cell_w
            measured = float(f.getlength(seg.text))
            return max(1, math.ceil(measured / cell_w - 0.05)) * cell_w

        max_width = 0.0
        for segments in line_segments:
            max_width = max(max_width, sum(advance(seg) for seg in segments))

        img_width = int(max_width) + padding * 2
        img_height = line_height * len(lines) + padding * 2

        img = Image.new("RGB", (img_width, img_height), _DEFAULT_BG)
        draw = ImageDraw.Draw(img)

        y = padding
        for segments in line_segments:
            x = float(padding)
            for seg in segments:
                if not seg.text:
                    continue
                f = fonts[seg.font_tier]
                width = advance(seg)

                # Draw background if specified
                if seg.style.bg_color:
                    draw.rectangle(
                        [x, y, x + width, y + line_height], fill=seg.style.bg_color
                    )

                # Baseline-anchored so fallback fonts line up with the primary
                draw.text(
                    (x, y + baseline_offset),
                    seg.text,
                    fill=seg.style.fg_color,
                    font=f,
                    anchor="ls",
                )
                x += width
            y += line_height

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # Run CPU-intensive image rendering in thread pool
    return await asyncio.to_thread(_render_image)
