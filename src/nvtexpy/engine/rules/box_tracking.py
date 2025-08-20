# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Callable, List

from nvtexpy.engine.common import CommandCode, Node
from nvtexpy.engine.mem import GlueNode, MemReader
from nvtexpy.engine.mem import Node as MemNode
from nvtexpy.engine.mem import WhatsitNode
from nvtexpy.engine.rules.base import Rule, Rules, begin_macro_hook
from nvtexpy.engine.stack_formatter import (
    DirectBeginFormatterWrapper,
    DirectEndFormatterWrapper,
    FormatterElement,
)
from nvtexpy.engine.texstate import TexState
from nvtexpy.output.structured_output import StructuredOutput


@dataclass
class BoxTrackNode:
    # The formatting stack at the begin
    formatting_begin: List[FormatterElement] | None = None

    # These actions are triggered, when the node-level is entered. Only for vlists/hlists
    begin_actions: List[Callable[["Node", MemNode, int, int, StructuredOutput], None]] | None = None
    end_actions: List[Callable[["Node", MemNode, StructuredOutput], None]] | None = None
    # Direct formatting on node enter/exit
    direct_formatting: List[FormatterElement] | None = None


@Rules.register
class BoxTrackRule(Rule):
    """Stores all tracked color@hbox, color@vbox."""

    stack: List[BoxTrackNode]

    def __init__(self, state: TexState, rules: Rules) -> None:
        super().__init__(state, rules)
        self.stack = []

    def push_box(self):
        print("Pushed box")
        self.stack.append(
            BoxTrackNode(
                formatting_begin=self.state.get_input_formatter_stack(),
            )
        )

    @begin_macro_hook("color@hbox")
    def on_color_hbox(self, *args):
        self.push_box()

    @begin_macro_hook("color@vbox")
    def on_color_vbox(self, *args):
        self.push_box()

    # @end_macro_hook("color@endbox")
    @begin_macro_hook("color@endbox")
    def on_color_endbox(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        popped_stack = self.stack.pop()
        print(f"Popped box {popped_stack}")
        r = MemReader()
        first_node_addr = None
        last_node_addr = None
        for node in r.iter_head_tail(head_addr, tail_addr):
            print(f"  - Node: {node}")
            if not isinstance(node, (GlueNode, WhatsitNode)):
                if first_node_addr is None:
                    first_node_addr = node.addr
                last_node_addr = node.addr
        if (
            last_node_addr is not None
            and first_node_addr is not None
            and (
                popped_stack.end_actions
                or popped_stack.begin_actions
                or popped_stack.direct_formatting
            )
        ):
            # Apply the actions to the nodes
            node_begin = self._get_node(first_node_addr, make_new=False)
            print(f"  Store actions for node_begin={node_begin}")
            if last_node_addr != first_node_addr:
                node_end = self._get_node(last_node_addr, make_new=False)
                print(f"  Store actions for node_end={node_end}")
            else:
                node_end = node_begin
            node_begin.formatting_begin = popped_stack.formatting_begin
            if popped_stack.begin_actions:
                if node_begin.begin_actions is None:
                    node_begin.begin_actions = popped_stack.begin_actions
                else:
                    node_begin.begin_actions.extend(popped_stack.begin_actions)
            if popped_stack.end_actions:
                if node_end.end_actions is None:
                    node_end.end_actions = popped_stack.end_actions
                else:
                    node_end.end_actions.extend(popped_stack.end_actions)
            if popped_stack.direct_formatting:
                if node_begin == node_end:
                    if node_end.direct_formatting is None:
                        node_end.direct_formatting = popped_stack.direct_formatting
                    else:
                        node_end.direct_formatting.extend(popped_stack.direct_formatting)
                else:
                    begin_elements = (
                        DirectBeginFormatterWrapper(direct_formatter)
                        for direct_formatter in popped_stack.direct_formatting
                    )
                    end_elements = (
                        DirectEndFormatterWrapper(direct_formatter)
                        for direct_formatter in popped_stack.direct_formatting
                    )
                    if node_begin.direct_formatting is None:
                        node_begin.direct_formatting = list(begin_elements)
                    else:
                        node_begin.direct_formatting.extend(begin_elements)
                    if node_end.direct_formatting is None:
                        node_end.direct_formatting = list(end_elements)
                    else:
                        node_end.direct_formatting.extend(end_elements)
            # Store the formatter state at the end
            node_end.formatting_end = self.state.get_input_formatter_stack()

    def register_box_action(
        self,
        *,
        begin_action: Callable[[Node, MemNode, int, int, StructuredOutput], None] | None = None,
        end_action: Callable[[Node, MemNode, StructuredOutput], None] | None = None,
        direct_formatter: FormatterElement | None = None,
    ):
        node = self.stack[-1]
        if begin_action is not None:
            if node.begin_actions is None:
                node.begin_actions = [begin_action]
            else:
                node.begin_actions.append(begin_action)
        if end_action is not None:
            if node.end_actions is None:
                node.end_actions = [end_action]
            else:
                node.end_actions.append(end_action)
        if direct_formatter is not None:
            if node.direct_formatting is None:
                node.direct_formatting = [direct_formatter]
            else:
                node.direct_formatting.append(direct_formatter)
