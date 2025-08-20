# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nvtexpy.engine.rules.base import Rule, Rules, begin_macro_hook, end_macro_hook
from nvtexpy.engine.stack_formatter import CallbackFormatter, FormatterElement
from nvtexpy.output.structured_output import (
    OverlineElement,
    StrikethroughElement,
    UnderlineElement,
)


class StrikethroughFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(StrikethroughElement()),
            name="strikethrough",
        )


class UnderlineFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(UnderlineElement()),
            name="underline",
        )


class OverlineFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(OverlineElement()),
            name="overline",
        )


@Rules.register
class UlemRule(Rule):
    """For \\sout{...}."""

    active = False

    fn_mark_active = False

    @end_macro_hook(["sout", "xout"])
    def on_sout(self, *args):
        if self.fn_mark_active:
            return
        self.fn_mark_active = True
        formatter = StrikethroughFormatter()
        self.push_formatter(formatter)

        # print("FNmark: begin")

        def end_sout(*_):
            # print("FNmark: end")
            self.pop_formatter(formatter)
            self.state.input_formatter_store_next = True
            self.fn_mark_active = False

        self.state.save_levels[-1].end_hooks.append(end_sout)

    @end_macro_hook(["uline", "uuline", "uwave", "dashuline", "dotuline"])
    def on_uline(self, *args):
        if self.fn_mark_active:
            return
        self.fn_mark_active = True
        formatter = UnderlineFormatter()
        self.push_formatter(formatter)

        # print("FNmark: begin")

        def end_uline(*_):
            # print("FNmark: end")
            self.pop_formatter(formatter)
            self.state.input_formatter_store_next = True
            self.fn_mark_active = False

        self.state.save_levels[-1].end_hooks.append(end_uline)


@Rules.register
class SoulRule(Rule):
    """For \\st{...} \\ul{...}."""

    formatters: list[FormatterElement] = []
    next_formatter = None

    @end_macro_hook("SOUL@setup")
    def on_setup(self, *args):
        self.next_formatter = None

    @end_macro_hook("SOUL@stsetup")
    def on_stsetup(self, *args):
        self.next_formatter = StrikethroughFormatter()

    @end_macro_hook("SOUL@ulsetup")
    def on_ulsetup(self, *args):
        self.next_formatter = UnderlineFormatter()

    @end_macro_hook("SOUL@hlsetup")
    def on_hlsetup(self, *args):
        self.next_formatter = OverlineFormatter()

    @begin_macro_hook("SOUL@n")
    def on_soul_begin(self, *args):
        if self.next_formatter is not None:
            self.formatters.append(self.next_formatter)
            self.push_formatter(self.next_formatter)
            self.next_formatter = None

            def end_soul(*_):
                self.pop_formatter(self.formatters.pop())
                self.state.input_formatter_store_next = True

            return end_soul
