# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nvtexpy.engine.rules.base import Rule, Rules, begin_macro_hook
from nvtexpy.engine.stack_formatter import CallbackFormatter
from nvtexpy.output.structured_output import FooterElement, HeaderElement, PageNumberElement


class FooterFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(FooterElement()),
            name="footer",
        )


class HeaderFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(HeaderElement()),
            name="header",
        )


class PageNumberFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(PageNumberElement()),
            name="page_number",
        )


@Rules.register
class PageRule(Rule):
    """For page footer and header @thefoot @thehead."""

    @begin_macro_hook("@thefoot")
    def on_footer_start(self, *args):
        self.font_rule.clear_format()
        formatter = FooterFormatter()
        self.box_tracker.register_box_action(direct_formatter=formatter)
        print("PAGE: foot begin")

        def end_footer(head_addr: int, tail_addr: int):
            self.font_rule.clear_format()
            self.state.input_formatter_store_next = True
            print("PAGE: foot end")

        # Ends when the currently active save group ends
        self.state.save_levels[-1].end_hooks.append(end_footer)

    @begin_macro_hook("@thehead")
    def on_header_start(self, *args):
        self.font_rule.clear_format()
        formatter = HeaderFormatter()
        self.box_tracker.register_box_action(direct_formatter=formatter)
        print("PAGE: head begin")

        def end_header(head_addr: int, tail_addr: int):
            self.font_rule.clear_format()
            self.state.input_formatter_store_next = True
            print("PAGE: head end")

        # Ends when the currently active save group ends
        self.state.save_levels[-1].end_hooks.append(end_header)

    @begin_macro_hook("thepage")
    def on_page_number_start(self, *args):
        formatter = PageNumberFormatter()
        self.push_formatter(formatter)

        def end_page_number(*_):
            self.pop_formatter(formatter)
            self.state.input_formatter_store_next = True

        # Ends when the currently active save group ends
        return end_page_number
