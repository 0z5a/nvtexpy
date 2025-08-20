# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import List

from nvtexpy.engine.common import Node
from nvtexpy.engine.mem import GlueSubtype
from nvtexpy.engine.rules.base import Rule, Rules
from nvtexpy.engine.texstate import TexState


@dataclass
class FailStackElement:
    nodes: List[Node] = field(default_factory=list)
    fail_message: str | None = None


@Rules.register
class FailStackRule(Rule):
    """Stores the fail stack for page fails."""

    stack: List[FailStackElement]

    def __init__(self, state: TexState, rules: Rules) -> None:
        super().__init__(state, rules)
        self.stack = [FailStackElement()]

    def on_page_fail(self, fail_message: str):
        if self.stack[-1].fail_message is None:
            self.stack[-1].fail_message = fail_message
            for node in self.stack[-1].nodes:
                node.page_fail = fail_message

    def on_char_alloc(self, node: Node):
        self.stack[-1].nodes.append(node)
        if self.stack[-1].fail_message is not None:
            node.page_fail = self.stack[-1].fail_message

    def on_glue_alloc(self, node: Node, glue_type: GlueSubtype):
        self.stack[-1].nodes.append(node)
        if self.stack[-1].fail_message is not None:
            node.page_fail = self.stack[-1].fail_message
