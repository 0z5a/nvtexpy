# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import zipfile
from pathlib import Path

from pdftex import get_font_name


class GlyphMap:
    GLYPHMAPS_PATH = Path(__file__).parent / "fonts_glyphmap.zip"
    _re_letterspace = re.compile(r"^(.*)[+-]\d+ls$")
    _re_expand = re.compile(r"^(.*)[+-]\d+$")

    def __init__(self) -> None:
        self.glyph_to_unicode: dict[str, dict[int, str] | None] = {}
        self.fontid_to_unicode: dict[int, dict[int, str] | None] = {}
        self.zipfile = zipfile.ZipFile(self.GLYPHMAPS_PATH)
        self.available_fonts = set(
            name[:-4] for name in self.zipfile.namelist() if name.endswith(".txt")
        )

    def _load_font(self, font: str) -> dict[int, str] | None:
        if font not in self.available_fonts:
            print(f"Font {font!r} not available")
            self.glyph_to_unicode[font] = None
            return None
        print(f"Loading glyphmap for {font!r} from {self.GLYPHMAPS_PATH}/{font}.txt")
        return {
            int(code, 16): "".join(chr(int(c, 16)) for c in unicode.split())
            for code, unicode in (
                line.split(";", maxsplit=1)
                for line in self.zipfile.read(f"{font}.txt").decode().splitlines()
            )
        }

    def _get_load_font(self, font_name: str) -> dict[int, str] | None:
        assert isinstance(font_name, str)
        if font_name in self.glyph_to_unicode:
            return self.glyph_to_unicode[font_name]
        else:
            fontdef = self._load_font(font_name)
            if fontdef is not None:
                return fontdef
            # Try to remove -20ls
            m = self._re_letterspace.match(font_name)
            if m:
                inner_font_name = m.group(1)
                fontdef = self.glyph_to_unicode[font_name] = self._get_load_font(inner_font_name)
            if fontdef is not None:
                return fontdef
            # Try to remove -20
            m = self._re_expand.match(font_name)
            if m:
                inner_font_name = m.group(1)
                fontdef = self.glyph_to_unicode[font_name] = self._get_load_font(inner_font_name)
        return fontdef

    def get(self, font: str | int, glyph: int) -> str | None:
        if isinstance(font, str):
            fontdef = self._get_load_font(font)
        else:
            if font in self.fontid_to_unicode:
                fontdef = self.fontid_to_unicode[font]
            else:
                font_name = get_font_name(font).tobytes().decode()
                assert isinstance(font_name, str)
                fontdef = self._get_load_font(font_name)
                self.fontid_to_unicode[font] = fontdef

        if fontdef is None:
            return None
        return fontdef.get(glyph)
