# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nvtexpy.engine.rules.base import Rule, Rules, assert2pagefail, begin_macro_hook
from nvtexpy.engine.stack_formatter import CallbackFormatter
from nvtexpy.output.structured_output import BibliographyElement


class BibliographyFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(BibliographyElement()),
            name="bibliography",
        )


@Rules.register
class BibliographyRule(Rule):
    """For natbib."""

    @begin_macro_hook(["bibsection", "bibitem"])
    @assert2pagefail
    def on_bib(self, *args):
        print(f"BIB: Item {args[0]}")
        assert self.active, "Bib item without active bibliography formatter"

    @begin_macro_hook("thebibliography")
    @assert2pagefail
    def on_thebibliography_start(self, *args):
        # TODO: Kern nodes must be detected to save spaces properly
        if self.active:
            # Don't nest bibliography
            return None

        self.active = True

        print("BIB: Active")

        formatter = BibliographyFormatter()
        self.push_formatter(formatter)

        def end_bibliography(head_addr: int, tail_addr: int):
            self.pop_formatter(formatter)
            self.active = False
            self.state.input_formatter_store_next = True
            print("BIB: End")

        # Ends when the currently active save group ends
        self.state.save_levels[-1].end_hooks.append(end_bibliography)
