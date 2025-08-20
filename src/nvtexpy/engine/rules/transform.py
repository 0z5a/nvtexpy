# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nvtexpy.engine.mem import HBoxNode
from nvtexpy.engine.mem import Node as MemNode
from nvtexpy.engine.mem import (
    VBoxNode,
    WhatsitPdfRestoreNode,
    WhatsitPdfSaveNode,
    WhatsitPdfSetMatrixNode,
)
from nvtexpy.engine.rules.base import Rule, Rules, assert2pagefail
from nvtexpy.output.structured_output import StructuredOutput, Transform


@Rules.register
class TransformRule(Rule):
    """
    Uses pdf_save_node, pdf_restore_node and pdf_setmatrix_node.
    """

    struct_out: StructuredOutput | None

    def on_sheet_begin(
        self,
        memnode: VBoxNode | HBoxNode | None,
        page: int,
        page_width: int,
        page_height: int,
        struct_out: StructuredOutput,
    ):
        self.cur_page_height = page_height
        self.struct_out = struct_out

    def on_sheet_end(self):
        self.struct_out = None

    @assert2pagefail
    def on_node_out(self, memnode: MemNode, cur_h: int, cur_v: int):
        if self.struct_out is None:
            return
        if isinstance(memnode, WhatsitPdfSaveNode):
            self.struct_out.push_transform()
            # print(f"TR: push_transform")
        elif isinstance(memnode, WhatsitPdfRestoreNode):
            # print(f"TR: pop_transform")
            self.struct_out.pop_transform()
        elif isinstance(memnode, WhatsitPdfSetMatrixNode):
            plain_matrix_data = memnode.setmatrix_data.to_plain_str()
            # print(f"TR: set_matrix {plain_matrix_data} {cur_h} {cur_v} {self.cur_page_height}")
            assert plain_matrix_data is not None
            self.struct_out.transform(
                Transform.from_tex(plain_matrix_data, cur_h, cur_v, self.cur_page_height)
            )

    def on_hlist_out_node(self, memnode: MemNode, cur_h: int, cur_v: int):
        self.on_node_out(memnode, cur_h, cur_v)

    def on_vlist_out_node(self, memnode: MemNode, cur_h: int, cur_v: int):
        self.on_node_out(memnode, cur_h, cur_v)
