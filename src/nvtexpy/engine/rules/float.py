# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

from nvtexpy.engine.common import CommandCode, get_node_from_box_register
from nvtexpy.engine.rules.base import Rule, Rules, assert2pagefail, begin_macro_hook, end_macro_hook
from nvtexpy.engine.stack_formatter import CallbackFormatter, FormatterElement
from nvtexpy.output.structured_output import FloatElement


class FloatFormatter(CallbackFormatter):
    def __init__(self, environment: str) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(FloatElement(environment)),
            name="float",
        )


@Rules.register
class FloatRule(Rule):
    """For \\begin{float}."""

    start_stack: List[FormatterElement]

    @end_macro_hook(
        [
            "@endfloatbox",
            # "color@endbox",
            # "color@endgroup",
        ]
    )
    @assert2pagefail
    def on_float_end(self, cs_name: str, cmd, args: None, head: int, tail: int):
        if not self.active:
            return
        self.active = False
        currbox = get_node_from_box_register(b"@currbox")
        if currbox is None:
            return
        self.font_rule.clear_format()
        self.register_node_action(
            currbox.addr,
            direct_formatter=self.formatter,
            begin_stack=self.start_stack,
            dump_stack=True,
        )
        # print(f"F: End FLOAT")

    @begin_macro_hook(
        [
            # float with "H" positioning
            # "@float@HH",
            # float with other positioning (@xfloat)
            # "@float@Hx",
            # float
            "@xfloat",
        ],
        str_args=True,
    )
    @assert2pagefail
    def on_float_start(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: Tuple[str | None, ...] | None,
        head_addr: int,
        tail_addr: int,
    ):
        # TODO: Kern nodes must be detected to save spaces properly
        if self.active:
            # Don't nest
            return None

        self.active = True

        # print(f"F: Start FLOAT {args!r}")
        assert args is not None and len(args) > 0 and isinstance(args[0], str), (
            "Float must have a type"
        )
        environment = args[0]

        self.formatter = FloatFormatter(environment)
        # self.push_formatter(formatter)

        # def end_float(head_addr: int, tail_addr: int):
        #     # self.pop_formatter(formatter)
        #     self.active = False
        #     # print(f"F: (End FLOAT savelevel)")

        # # Ends when the currently active save group ends
        # self.state.save_levels[-1].end_hooks.append(end_float)

        self.font_rule.clear_format()
        self.start_stack = self.dump_stack()


@Rules.register
class WrapfloatRule(Rule):
    """For \\begin{wrapfigure/wraptable}."""

    start_stack: List[FormatterElement]

    @begin_macro_hook("WF@startfloating")
    @assert2pagefail
    def on_wrapfloat_end(self, *args):
        # print(f"F: endwrapfloat")
        if not self.active:
            return
        self.active = False
        currbox = get_node_from_box_register(b"WF@box")
        # print(f"F: WF@box {currbox!r}")
        if currbox is None:
            return
        self.font_rule.clear_format()
        self.register_node_action(
            currbox.addr,
            direct_formatter=self.formatter,
            begin_stack=self.start_stack,
            dump_stack=True,
        )
        # print(f"F: End WRAPFLOAT")

    @begin_macro_hook(
        [
            # wrapfig
            # "WF@wr",
            "wrapfloat",
        ],
        str_args=True,
    )
    @assert2pagefail
    def on_float_start(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: Tuple[str | None, ...] | None,
        head_addr: int,
        tail_addr: int,
    ):
        # TODO: Kern nodes must be detected to save spaces properly
        if self.active:
            # Don't nest
            return None

        self.active = True

        # print(f"F: Start FLOAT {args!r}")
        assert args is not None and len(args) > 0 and isinstance(args[0], str), (
            "Float must have a type"
        )
        environment = args[0]

        self.formatter = FloatFormatter(environment)
        # self.push_formatter(formatter)

        self.font_rule.clear_format()
        self.start_stack = self.dump_stack()
