# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Set

from pdftex import get_family_font, get_font_name

from nvtexpy.engine.callstack import CallStack
from nvtexpy.engine.common import BoxStackItem, FontProperties, Node, SaveLevel
from nvtexpy.engine.debug import DEBUG_FONT, DEBUG_INPUT_FORMATTER
from nvtexpy.engine.glyphmap import GlyphMap
from nvtexpy.engine.mathdef import DEFAULT_MATH_SYMBOLS
from nvtexpy.engine.stack_formatter import FormatterElement


class CharConverter:
    glyphmap: GlyphMap

    def __init__(self, glyphmap: GlyphMap):
        self.glyphmap = glyphmap

    def get_font(self, font: int) -> str:
        return get_font_name(font).tobytes().decode()

    def get_char(self, font: int, character: int, debug: bool = False) -> str:
        glyphmap_char = self.glyphmap.get(font, character)
        if glyphmap_char is not None:
            return glyphmap_char
        font_name = self.get_font(font)
        if not debug:
            print(f"WARNING: Unrecognized font/char fnt={font_name}({font}) chr={character}")
        return f"\\unknown{{{font_name}}}{{{character:04X}}}"

    def get_math_cs(self, family: int, character: int) -> str:
        font = get_family_font(family)
        res = self.get_char(font, character)
        if res in DEFAULT_MATH_SYMBOLS:
            if DEBUG_FONT:
                font_name = self.get_font(font)
                print(
                    f"FNT: font={font}={font_name} char={character} mapped={DEFAULT_MATH_SYMBOLS[res]}"
                )
            return DEFAULT_MATH_SYMBOLS[res]
        font_name = self.get_font(font)
        return f"\\unknown{{{font_name}}}{{{character:04X}}}"


class TexState:
    callstack: CallStack
    save_levels: List[SaveLevel]
    box_stack: List[BoxStackItem]
    _input_formatter_stack: List[FormatterElement]
    _input_formatter_stack_unordered: Set[FormatterElement]
    input_formatter_whatsit: int = 0
    input_formatter_store_next: bool = False
    char_converter: CharConverter
    current_page_failed: Optional[str]
    font_properties: Dict[int, FontProperties]
    family_properties: Dict[int, FontProperties]
    current_text_font: FontProperties
    addr_to_node: Dict[int, Node | None]

    def __init__(self):
        self.save_levels = [SaveLevel(addr=0x0, level=0, group_code=0, env_name="global")]
        self.box_stack = []
        self._input_formatter_stack = []
        self._input_formatter_stack_unordered = set()
        self.current_page_failed = None
        self.addr_to_node = {}
        self.font_properties = {}
        self.family_properties = {}
        self.current_text_font = FontProperties(0)

    def clear_input_formatter_stack(self):
        for fmt in self._input_formatter_stack:
            fmt.is_on_stack = False
        self._input_formatter_stack = []
        for fmt in self._input_formatter_stack_unordered:
            fmt.is_on_stack = False
        self._input_formatter_stack_unordered = set()
        self.input_formatter_whatsit = 0
        self.input_formatter_store_next = False

    def get_input_formatter_stack(self) -> List[FormatterElement]:
        if self._input_formatter_stack_unordered:
            return self._input_formatter_stack + list(self._input_formatter_stack_unordered)
        else:
            return self._input_formatter_stack.copy()

    def push_input_formatter_stack(self, formatter: FormatterElement):
        if DEBUG_INPUT_FORMATTER:
            print(f"TexState: Push formatter: {formatter}")
        assert not formatter.is_on_stack
        if formatter.stack_unordered:
            self._input_formatter_stack_unordered.add(formatter)
        else:
            self._input_formatter_stack.append(formatter)
        formatter.is_on_stack = True
        if formatter.format_whatsit_nodes:
            self.input_formatter_whatsit += 1

    def pop_input_formatter_stack(self, check_formatter: FormatterElement):
        if DEBUG_INPUT_FORMATTER:
            print(f"TexState: Pop formatter: {check_formatter}")
        assert check_formatter.is_on_stack
        if check_formatter.stack_unordered:
            assert len(self._input_formatter_stack_unordered) > 0, (
                "Tried to pop from empty input formatter stack"
            )
            try:
                self._input_formatter_stack_unordered.remove(check_formatter)
                formatter = check_formatter
            except KeyError:
                assert False, (
                    f"Expected {check_formatter} in {self._input_formatter_stack_unordered}"
                )
        else:
            assert len(self._input_formatter_stack) > 0, (
                "Tried to pop from empty input formatter stack"
            )

            formatter = self._input_formatter_stack.pop()
            assert formatter is check_formatter, f"Expected {check_formatter}, got {formatter}"
        check_formatter.is_on_stack = False
        if formatter.format_whatsit_nodes:
            self.input_formatter_whatsit -= 1
        return formatter

    def pop_store_next(self) -> bool:
        """Returns if the next Node should store the formatter stack."""
        store_next = self.input_formatter_store_next
        self.input_formatter_store_next = False
        return store_next

    def check_input_formatter_stack(self):
        assert len(self._input_formatter_stack) == 0, (
            f"Input formatter stack not empty: {self._input_formatter_stack}"
        )
