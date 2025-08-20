# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nvtexpy.engine.common import Node
from nvtexpy.engine.mem import LigatureNode
from nvtexpy.engine.mem import Node as MemNode
from nvtexpy.engine.rules.base import Rule, Rules
from nvtexpy.engine.texstate import TexState


@Rules.register
class LigatureRule(Rule):
    """For special ligatures like double quotes and long hyphens."""

    replacements: dict[int, str | None]

    def __init__(self, state: TexState, rules: Rules) -> None:
        super().__init__(state, rules)

        self.replacements = {}

    def on_node_dealloc(self, addr: int, size: int):
        if addr in self.replacements:
            del self.replacements[addr]

    def on_hlist_out_node(self, memnode: MemNode, cur_h: int, cur_v: int):
        if isinstance(memnode, LigatureNode):
            lig_char_nodes = list(memnode.ligs)
            lig_str = "".join([chr(n.character) for n in lig_char_nodes])
            if lig_str == "``" or lig_str == "''":
                self.replacements[lig_char_nodes[0].addr] = '"'
                self.replacements[lig_char_nodes[1].addr] = ""
            elif lig_str == "--" or lig_str == "---":
                self.replacements[lig_char_nodes[0].addr] = "-"
                self.replacements[lig_char_nodes[1].addr] = ""

    def on_char_out(
        self,
        node: Node,
        memnode: MemNode,
        cur_h: int,
        cur_v: int,
        width: int,
        height: int,
        depth: int,
    ):
        if memnode.addr in self.replacements:
            # Replace the characters if needed
            node.replacement_character = self.replacements[memnode.addr]
            del self.replacements[memnode.addr]
