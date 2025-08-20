# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from nvtexpy.engine.common import CommandCode, get_int_value
from nvtexpy.engine.debug import DEBUG_SECTION
from nvtexpy.engine.rules.base import (
    Rule,
    Rules,
    assert2pagefail,
    begin_macro_hook,
    capture_stack_element,
)
from nvtexpy.engine.stack_formatter import SectionFormatter


@Rules.register
class SectionRule(Rule):
    """For section headings."""

    last_section_level = 10

    in_frontmatter: bool = False

    in_maketitle = capture_stack_element(["@maketitle", "maketitle", "HyOrg@maketitle"])

    @begin_macro_hook("frontmatter")
    def on_frontmatter_start(self, *args):
        self.in_frontmatter = True

    @begin_macro_hook("endfrontmatter")
    def on_frontmatter_end(self, *args):
        self.in_frontmatter = False

    def _track_specific_heading(self, section_level: int):
        if self.active:
            # Don't nest section
            return None

        if DEBUG_SECTION:
            print(f"on_heading begin {section_level}")

        self.active = True
        self.font_rule.clear_format()
        self.last_section_level = section_level
        formatter = SectionFormatter(section_level)
        self.push_formatter(formatter)

        def end_section(*_):
            self.font_rule.clear_format()
            self.pop_formatter(formatter)
            if DEBUG_SECTION:
                print(f"on_heading end {section_level}")
            self.active = False
            self.state.input_formatter_store_next = True
            self.last_section_level = section_level

        return end_section

    @begin_macro_hook(["@ssect@ltx", "@sect@ltx", "@sect"], str_args=True)
    @assert2pagefail
    def on_sect(
        self,
        cs: str,
        cmd: CommandCode,
        args: Tuple[str | None, ...] | None,
        head_addr: int,
        tail_addr: int,
    ):
        if self.active:
            # Don't nest section
            return None

        if DEBUG_SECTION:
            print("on_sect begin")
        assert args is not None
        level_arg = args[1]
        assert level_arg is not None
        try:
            section_level = get_int_value(level_arg) + 1
        except ValueError:
            assert False, f"Cannot read section level {level_arg}"
        if section_level < 0 or section_level > 6:
            section_level = self.last_section_level

        return self._track_specific_heading(section_level)

    @begin_macro_hook("@title")
    @assert2pagefail
    def on_title(self, *args):
        if self.in_maketitle or self.in_frontmatter:
            return self._track_specific_heading(section_level=0)

    @begin_macro_hook("H@old@part")
    @assert2pagefail
    def on_part(self, *args):
        return self._track_specific_heading(section_level=1)

    @begin_macro_hook("@startsection", str_args=True)
    @assert2pagefail
    def on_start_sec(
        self,
        cs: str,
        cmd: CommandCode,
        args: Tuple[str | None, ...] | None,
        head_addr: int,
        tail_addr: int,
    ):
        assert args and len(args) > 2
        level_arg = args[1]
        assert level_arg is not None
        try:
            self.last_section_level = get_int_value(level_arg) + 1
            print(f"Store section level {self.last_section_level}")
        except ValueError:
            assert False, f"Cannot read section level {level_arg}"

    @begin_macro_hook(["@svsechd", "@ssect"])
    @assert2pagefail
    def on_sec_output(
        self,
        cs: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        if not self.active:
            self.active = True
            self.font_rule.clear_format()
            formatter = SectionFormatter(self.last_section_level)
            self.push_formatter(formatter)

            def end_section(*_):
                self.font_rule.clear_format()
                self.pop_formatter(formatter)
                if DEBUG_SECTION:
                    print("on_sec_output end")
                self.active = False
                self.state.input_formatter_store_next = True

            return end_section
