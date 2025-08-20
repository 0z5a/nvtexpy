# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

from nvtexpy.engine.common import Node
from nvtexpy.engine.debug import DEBUG_OUTPUT_STACK
from nvtexpy.engine.mem import CharNode, GlueNode, HBoxNode
from nvtexpy.engine.mem import Node as MemNode
from nvtexpy.engine.mem import RuleNode, VBoxNode
from nvtexpy.engine.stack_formatter import FormatterElement
from nvtexpy.engine.texstate import TexState
from nvtexpy.output.structured_output import StructuredOutput


class FormatterOutputStack:
    stack: list[FormatterElement]
    struct_out: StructuredOutput | None
    indent: str

    _rules_on_char_out: list[Callable]
    _rules_on_glue_out: list[Callable]
    _rules_on_rule_out: list[Callable]
    _rules_on_hlist_out_start: list[Callable]
    _rules_on_hlist_out_end: list[Callable]
    _rules_on_hlist_out_node: list[Callable]
    _rules_on_vlist_out_start: list[Callable]
    _rules_on_vlist_out_end: list[Callable]
    _rules_on_vlist_out_node: list[Callable]

    def __init__(self, state: TexState):
        self.stack = []
        self.struct_out = None
        self.state = state
        self.indent = ""

        self._rules_on_char_out = []
        self._rules_on_glue_out = []
        self._rules_on_rule_out = []
        self._rules_on_hlist_out_start = []
        self._rules_on_hlist_out_end = []
        self._rules_on_hlist_out_node = []
        self._rules_on_vlist_out_start = []
        self._rules_on_vlist_out_end = []
        self._rules_on_vlist_out_node = []

    def pop(self):
        assert len(self.stack) > 0
        formatter = self.stack.pop()
        formatter.end(self.struct_out)
        if DEBUG_OUTPUT_STACK:
            print(
                f"{self.indent} -> FMT END {formatter}",
            )
        if type(formatter).on_char_out != FormatterElement.on_char_out:
            self._rules_on_char_out.pop()
        if type(formatter).on_glue_out != FormatterElement.on_glue_out:
            self._rules_on_glue_out.pop()
        if type(formatter).on_rule_out != FormatterElement.on_rule_out:
            self._rules_on_rule_out.pop()
        if type(formatter).on_hlist_out_start != FormatterElement.on_hlist_out_start:
            self._rules_on_hlist_out_start.pop()
        if type(formatter).on_hlist_out_end != FormatterElement.on_hlist_out_end:
            self._rules_on_hlist_out_end.pop()
        if type(formatter).on_hlist_out_node != FormatterElement.on_hlist_out_node:
            self._rules_on_hlist_out_node.pop()
        if type(formatter).on_vlist_out_start != FormatterElement.on_vlist_out_start:
            self._rules_on_vlist_out_start.pop()
        if type(formatter).on_vlist_out_end != FormatterElement.on_vlist_out_end:
            self._rules_on_vlist_out_end.pop()
        if type(formatter).on_vlist_out_node != FormatterElement.on_vlist_out_node:
            self._rules_on_vlist_out_node.pop()
        if DEBUG_OUTPUT_STACK:
            print(
                f"{self.indent} -> Stack: {self.stack}",
            )
        return formatter

    def push(self, formatter: FormatterElement):
        if DEBUG_OUTPUT_STACK:
            print(
                f"{self.indent} -> FMT BEGIN {formatter}",
            )
        formatter.state = self.state
        formatter.begin(self.struct_out)
        self.stack.append(formatter)
        if type(formatter).on_char_out != FormatterElement.on_char_out:
            self._rules_on_char_out.append(formatter.on_char_out)
        if type(formatter).on_glue_out != FormatterElement.on_glue_out:
            self._rules_on_glue_out.append(formatter.on_glue_out)
        if type(formatter).on_rule_out != FormatterElement.on_rule_out:
            self._rules_on_rule_out.append(formatter.on_rule_out)
        if type(formatter).on_hlist_out_node != FormatterElement.on_hlist_out_node:
            self._rules_on_hlist_out_node.append(formatter.on_hlist_out_node)
        if type(formatter).on_hlist_out_start != FormatterElement.on_hlist_out_start:
            self._rules_on_hlist_out_start.append(formatter.on_hlist_out_start)
        if type(formatter).on_hlist_out_end != FormatterElement.on_hlist_out_end:
            self._rules_on_hlist_out_end.append(formatter.on_hlist_out_end)
        if type(formatter).on_vlist_out_node != FormatterElement.on_vlist_out_node:
            self._rules_on_vlist_out_node.append(formatter.on_vlist_out_node)
        if type(formatter).on_vlist_out_start != FormatterElement.on_vlist_out_start:
            self._rules_on_vlist_out_start.append(formatter.on_vlist_out_start)
        if type(formatter).on_vlist_out_end != FormatterElement.on_vlist_out_end:
            self._rules_on_vlist_out_end.append(formatter.on_vlist_out_end)
        if DEBUG_OUTPUT_STACK:
            print(
                f"{self.indent} -> Stack: {self.stack}",
            )

    def call_char_out(
        self,
        node: Node,
        memnode: CharNode,
        cur_h: int,
        cur_v: int,
        width: int,
        height: int,
        depth: int,
    ):
        for rule in self._rules_on_char_out:
            rule(node, memnode, cur_h, cur_v, width, height, depth)

    def call_glue_out(
        self, node: Node, memnode: GlueNode, cur_h: int, cur_v: int, width: int, height: int
    ):
        for rule in self._rules_on_glue_out:
            rule(node, memnode, cur_h, cur_v, width, height)

    def call_rule_out(
        self, node: Node | None, memnode: RuleNode, cur_h: int, cur_v: int, width: int, height: int
    ):
        for rule in self._rules_on_rule_out:
            rule(node, memnode, cur_h, cur_v, width, height)

    def call_hlist_out_start(self, node: Node | None, memnode: HBoxNode, cur_h: int, cur_v: int):
        for rule in self._rules_on_hlist_out_start:
            rule(node, memnode, cur_h, cur_v)

    def call_hlist_out_end(self, node: Node | None, memnode: HBoxNode):
        for rule in self._rules_on_hlist_out_end:
            rule(node, memnode)

    def call_hlist_out_node(self, node: Node | None, memnode: MemNode, cur_h: int, cur_v: int):
        for rule in self._rules_on_hlist_out_node:
            rule(node, memnode, cur_h, cur_v)

    def call_vlist_out_start(self, node: Node | None, memnode: VBoxNode, cur_h: int, cur_v: int):
        for rule in self._rules_on_vlist_out_start:
            rule(node, memnode, cur_h, cur_v)

    def call_vlist_out_end(self, node: Node | None, memnode: VBoxNode):
        for rule in self._rules_on_vlist_out_end:
            rule(node, memnode)

    def call_vlist_out_node(self, node: Node | None, memnode: MemNode, cur_h: int, cur_v: int):
        for rule in self._rules_on_vlist_out_node:
            rule(node, memnode, cur_h, cur_v)

    def update_output_stack_pop(self, stack: list[FormatterElement] | None):
        """Update the output stack to match the formatting of the given node."""
        if stack is None:
            return
        # Remove the stack elements
        while len(self.stack) > 0 and len(self.stack) > len(stack):
            self.pop()
        # Remove additional non-matching elements
        while len(self.stack) > 0 and self.stack[-1] != stack[len(self.stack) - 1]:
            self.pop()

    def update_output_stack_push(self, stack: list[FormatterElement] | None):
        """Update the output stack to match the formatting of the given node."""
        if stack is None:
            return
        # Add all stack elements that are present in the current node
        for formatter in stack[len(self.stack) :]:
            self.push(formatter)
            if formatter.suppress_formats:
                break

    def clear_unordered_formatter_stack(self):
        """Clears the unordered formatter stack."""
        while len(self.stack) > 0 and self.stack[-1].stack_unordered:
            self.pop()

    def is_empty(self):
        return len(self.stack) == 0

    def clear(self):
        while len(self.stack) > 0:
            self.pop()
