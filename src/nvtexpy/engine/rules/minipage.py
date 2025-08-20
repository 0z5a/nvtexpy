# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nvtexpy.engine.rules.base import Rule, Rules, assert2pagefail, begin_macro_hook
from nvtexpy.engine.stack_formatter import CallbackFormatter
from nvtexpy.output.structured_output import FloatElement


class MinipageFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(FloatElement("minipage")),
            name="minipage",
        )


@Rules.register
class MinipageRule(Rule):
    """For \\begin{minipage}."""

    @begin_macro_hook("minipage")
    @assert2pagefail
    def on_minipage_start(self, *args):
        if self.active:
            # Don't nest verbatim
            return None

        self.active = True

        formatter = MinipageFormatter()
        self.push_formatter(formatter)
        # self.register_node_action()

        def end_minipage(head_addr: int, tail_addr: int):
            self.pop_formatter(formatter)
            self.active = False
            # print(f"F: END FLOAT")

        # Ends when the currently active save group ends
        self.state.save_levels[-1].end_hooks.append(end_minipage)

        self.font_rule.clear_format()
        self.start_stack = self.dump_stack()
