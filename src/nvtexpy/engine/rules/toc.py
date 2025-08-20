# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nvtexpy.engine.rules.base import Rule, Rules, assert2pagefail, begin_macro_hook
from nvtexpy.engine.stack_formatter import CallbackFormatter
from nvtexpy.output.structured_output import TableOfContentsElement


class TableOfContentsFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(TableOfContentsElement()),
            name="tableofcontents",
        )


@Rules.register
class TableOfContentsRule(Rule):
    """For \\tableofcontents."""

    active = False

    @begin_macro_hook("@starttoc")
    @assert2pagefail
    def on_starttoc_start(self, *args):
        assert self.active, "Cannot use @starttoc outside of tableofcontents"

    @begin_macro_hook("tableofcontents")
    @assert2pagefail
    def on_tableofcontents_start(self, *args):
        if self.active:
            return
        self.font_rule.clear_format()
        formatter = TableOfContentsFormatter()
        self.push_formatter(formatter)
        self.active = True

        def end_tableofcontents(*_):
            self.pop_formatter(formatter)
            self.font_rule.clear_format()
            self.state.input_formatter_store_next = True
            self.active = False

        return end_tableofcontents
