# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nvtexpy.engine.common import get_node_from_box_register
from nvtexpy.engine.mem import MemReader
from nvtexpy.engine.rules.base import Rule, Rules, assert2pagefail, begin_macro_hook
from nvtexpy.engine.rules.graphics import GraphicsFormatter
from nvtexpy.output.structured_output import StructuredOutput


class TikzFormatter(GraphicsFormatter):
    def begin(self, struct_out: StructuredOutput):
        super().begin(struct_out)
        # Find the first parent which has a width and height
        reader = MemReader()
        for parent_box in reversed(self.state.box_stack):
            parent = reader.get_node(parent_box.addr)
            if parent.width > 0 and parent.height > 0:
                self.on_track_box(parent, parent_box.h, parent_box.v)
                break
            # print(f"- PARENT: h={parent_box.h * unity}, v={parent_box.v * unity}, w={parent.width * unity}, h={parent.height * unity}+{parent.depth * unity}", str(parent))


@Rules.register
class TikzRule(Rule):
    """For \\begin{tikzfigure}."""

    @begin_macro_hook(
        [
            # "endtikzpicture",
            "pgfsys@typesetpicturebox",
            # "color@endbox",
            # "color@endgroup",
        ],
        str_args=True,
    )
    @assert2pagefail
    def on_typesetpicturebox_end(self, cs_name: str, cmd, args, head: int, tail: int):
        pgfpic = get_node_from_box_register(args[0][1:].encode())
        if pgfpic is None:
            return
        formatter = TikzFormatter()
        self.register_node_action(
            pgfpic.addr,
            direct_formatter=formatter,
            begin_action=lambda node, mem_node, cur_h, cur_v, struct_out: formatter.on_track_box(
                mem_node, cur_h, cur_v
            ),
            dump_stack=True,
        )
