# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nvtexpy.engine.common import CommandCode, Node
from nvtexpy.engine.debug import DEBUG_PAR
from nvtexpy.engine.mem import CharNode, MemReader
from nvtexpy.engine.mem import Node as MemNode
from nvtexpy.engine.mem import RuleNode, WhatsitNode
from nvtexpy.engine.rules.base import Rule, Rules, assert2pagefail, begin_macro_hook
from nvtexpy.output.structured_output import (
    BeginOfParagraphElement,
    EndOfParagraphElement,
    StructuredOutput,
)


def get_first_last_visible_node(
    head_addr: int, tail_addr: int
) -> tuple[MemNode | None, MemNode | None]:
    first_node = None
    last_node = None
    for node in MemReader().iter_head_tail(head_addr, tail_addr):
        if any(
            isinstance(subnode, (CharNode, WhatsitNode))
            or (isinstance(subnode, RuleNode) and not subnode.is_empty)
            for subnode in node.walk()
        ):
            if first_node is None:
                first_node = node
            last_node = node
    return first_node, last_node


def begin_par_action(node, mem_node, cur_h, cur_v, struct_out: StructuredOutput):
    struct_out.push_pop(BeginOfParagraphElement())


def end_par_action(node, mem_node, struct_out: StructuredOutput):
    struct_out.push_pop(EndOfParagraphElement())


@Rules.register
class ParagraphRule(Rule):
    """For paragraph end."""

    last_node: Node | None = None

    @begin_macro_hook("par")
    @assert2pagefail
    def on_next_par(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        if self.active:
            if DEBUG_PAR:
                print("Ignore par")
            return
        if DEBUG_PAR:
            print("Activate par on:")
        first_node, last_node = get_first_last_visible_node(head_addr, tail_addr)
        if DEBUG_PAR:
            for node in MemReader().iter_head_tail(head_addr, tail_addr):
                print(f"  - {node}")
        if first_node is not None and last_node is not None:
            if DEBUG_PAR:
                print("PAR: First/Last par node:")
                print(f"  - {first_node}")
                print(f"  - {last_node}")
            self.register_node_action(
                first_node.addr,
                last_node.addr,
                begin_action=begin_par_action,
                end_action=end_par_action,
                # make_new=False,
            )
            self.active = True

        def end(*_):
            self.active = False

        return end
        # print(f"PAR: {self.last_node}")
        # if self.last_node is not None:
        #     if DEBUG_PAR:
        #         print(f"End of par for {self.last_node}")
        #     self.register_node_action(
        #         self.last_node.orig_addr,
        #         begin_action=begin_par_action,
        #         make_new=False,
        #     )
        #     self.last_node = None
