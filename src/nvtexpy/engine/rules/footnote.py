# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nvtexpy.engine.rules.base import Rule, Rules, begin_macro_hook
from nvtexpy.engine.stack_formatter import CallbackFormatter
from nvtexpy.output.structured_output import FootnoteElement, FootnoteMarkElement


class FootnoteFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(FootnoteElement()),
            name="footnote",
        )


class FootnoteMarkFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(FootnoteMarkElement()),
            name="footnote_mark",
        )


@Rules.register
class FootnoteRule(Rule):
    """For \\footnote{...}."""

    active = False

    fn_mark_active = False

    @begin_macro_hook(["@makefnmark", "@thefnmark"])
    def on_footnotemark_start(self, *args):
        if self.fn_mark_active:
            return
        self.fn_mark_active = True
        formatter = FootnoteMarkFormatter()
        self.push_formatter(formatter)

        # print("FNmark: begin")

        def end_footnotemark(*_):
            # print("FNmark: end")
            self.pop_formatter(formatter)
            self.state.input_formatter_store_next = True
            self.fn_mark_active = False

        # Ends when the currently active save group ends
        # self.state.save_levels[-1].end_hooks.append(end_float)
        return end_footnotemark

    # frontmatter@footnote@produce

    @begin_macro_hook(
        [
            "@makefntext",
            "frontmatter@makefntext",
        ]
    )
    def on_footnotetext_start(self, *args):
        if self.active:
            return
        self.font_rule.clear_format()
        formatter = FootnoteFormatter()
        self.push_formatter(formatter)
        self.active = True
        # print("FN: begin")

        def end_footnotetext(head: int, tail: int):
            # print("FN: end")
            self.pop_formatter(formatter)
            self.font_rule.clear_format()
            self.state.input_formatter_store_next = True
            self.active = False

        # Ends when the currently active save group ends
        self.state.save_levels[-1].end_hooks.append(end_footnotetext)
