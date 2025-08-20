# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import List, TextIO

from nvtexpy.engine.common import Node
from nvtexpy.engine.debug import DEBUG_WRITE_PAGE_SVG
from nvtexpy.engine.mem import UNITY, CharNode, GlueNode, HBoxNode
from nvtexpy.engine.mem import Node as MemNode
from nvtexpy.engine.mem import (
    RuleNode,
    VBoxNode,
    WhatsitPdfRestoreNode,
    WhatsitPdfSaveNode,
    WhatsitPdfSetMatrixNode,
)
from nvtexpy.engine.rules.base import Rule, Rules
from nvtexpy.output.structured_output import (
    IdentityTransform,
    StructuredOutput,
    Transform,
    TransformImpl,
)

if DEBUG_WRITE_PAGE_SVG:
    _svg_register = Rules.register
else:

    def _svg_register(x):
        return x


def _safe_attr(s: MemNode):
    return (
        s._str(already_seen=set(), max_depth=1)
        .replace("<", "(")
        .replace(">", ")")
        .replace('"', "'")
        .replace("&", "-")
    )


@_svg_register
class WriteSvg(Rule):
    """
    Not a real "rule", but can write to svg :)
    """

    transform_stack: List[Transform]

    svg_level = 1
    svg_writer_inset = 0
    svg_writer: TextIO | None = None
    animation_depth: int = 0

    nesting_prefix: str = ""

    def on_sheet_begin(
        self,
        memnode: VBoxNode | HBoxNode | None,
        page: int,
        page_width: int,
        page_height: int,
        struct_out: StructuredOutput,
    ):
        print(f"Writing SVG to nvpdftex_page_{page}.svg")
        self.svg_writer = open(f"nvpdftex_page_{page}.svg", "w")
        self.transform_stack = [Transform.identity()]
        self.cur_page_height = page_height

        self.svg_writer.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{math.ceil(page_width / UNITY):.0f}px" height="{math.ceil(page_height / UNITY):.0f}px">\n'
        )
        self.svg_writer.write(
            "\n".join(
                (
                    "  <style>",
                    "    <![CDATA[",
                    "      rect.char {",
                    "        stroke: #000000;",
                    "        stroke-width: 0.25;",
                    "        fill: rgba(0, 0, 0, 0.1);",
                    "      }",
                    "      rect.hnode {",
                    "        stroke: #00ff00;",
                    "        stroke-width: 0.25;",
                    "        fill: rgba(0, 128, 0, 0.05);",
                    "      }",
                    "      rect.vnode {",
                    "        stroke: #0000ff;",
                    "        stroke-width: 0.25;",
                    "        fill: rgba(0, 0, 128, 0.05);",
                    "      }",
                    "      circle.glue {",
                    "        stroke: none;",
                    "        fill: #ff0000;",
                    "      }",
                    "      line.glue {",
                    "        stroke: #ff0000;",
                    "        fill: none;",
                    "      }",
                    "      rect.rule {",
                    "        stroke: #808000;",
                    "        fill: none;",
                    "      }",
                    *[
                        "\n".join(
                            (
                                f"      .depth-{i} {{",
                                f"        animation: depth-{i} {self.animation_depth}s infinite normal;",
                                "      }",
                            )
                        )
                        for i in range(self.animation_depth)
                    ],
                    *[
                        "\n".join(
                            (
                                f"      @keyframes depth-{i} {{",
                                "        from { opacity: 0.0; }",
                                f"        {i * 100 // self.animation_depth:.0f}% {{ opacity: 0.0; }}",
                                f"        {(i + 0.5) * 100 // self.animation_depth:.0f}% {{ opacity: 1.0; }}",
                                "        to { opacity: 1.0; }",
                                "      }",
                            )
                        )
                        for i in range(self.animation_depth)
                    ],
                    "    ]]>",
                    "  </style>",
                    f'  <image href="nvpdftex_page_{page}.png" width="{page_width / UNITY}" height="{page_height / UNITY}" />\n',
                )
            )
        )
        self.nesting_prefix = "  "

    def on_sheet_end(self):
        if self.svg_writer is not None:
            self.svg_writer.write("</svg>\n")
            self.svg_writer.close()
            self.svg_writer = None

    def on_node_out(self, memnode: MemNode, cur_h: int, cur_v: int):
        # Note: This code must be in line with the structured output code
        if isinstance(memnode, WhatsitPdfSaveNode):
            self.transform_stack.append(self.transform_stack[-1])
        elif isinstance(memnode, WhatsitPdfRestoreNode):
            self.transform_stack.pop()
        elif isinstance(memnode, WhatsitPdfSetMatrixNode):
            plain_matrix_data = memnode.setmatrix_data.to_plain_str()
            assert plain_matrix_data is not None
            transform = Transform.from_tex(plain_matrix_data, cur_h, cur_v, self.cur_page_height)
            self.transform_stack[-1] = transform @ self.transform_stack[-1]

    def on_hlist_out_node(self, memnode: MemNode, cur_h: int, cur_v: int):
        self.on_node_out(memnode, cur_h, cur_v)

    def on_vlist_out_node(self, memnode: MemNode, cur_h: int, cur_v: int):
        self.on_node_out(memnode, cur_h, cur_v)

    def transform(self) -> str:
        # svg transform attribute: transform="matrix(3 1 -1 3 30 40)"
        t = self.transform_stack[-1]
        if isinstance(t, IdentityTransform):
            return ""
        elif isinstance(t, TransformImpl):
            return f' transform="matrix({t.a} {t.b} {t.c} {t.d} {t.e / UNITY} {t.f / UNITY})"'
        assert False, "Invalid transform type"

    def on_char_out(
        self,
        node: Node,
        memnode: CharNode,
        cur_h: int,
        cur_v: int,
        width: int,
        height: int,
        depth: int,
    ):
        if self.svg_writer is not None:
            # Write rectangle
            self.svg_writer.write(
                self.nesting_prefix
                + f'<rect x="{cur_h / UNITY}" y="{(cur_v - height) / UNITY}" width="{width / UNITY}" height="{height / UNITY + depth / UNITY}" class="char depth-{self.svg_level}" data-pdftex="{_safe_attr(memnode)}"{self.transform()} />\n'
            )

    def on_glue_out(
        self, node: Node, memnode: GlueNode, cur_h: int, cur_v: int, width: int, height: int
    ):
        if self.svg_writer is not None:
            self.svg_writer.write(
                self.nesting_prefix + f'<g class="glue" data-pdftex="{_safe_attr(memnode)}">\n'
            )
            self.nesting_prefix += "  "
            # Write dot
            self.svg_writer.write(
                self.nesting_prefix
                + f'<circle cx="{cur_h / UNITY}" cy="{cur_v / UNITY}" r="1" class="glue {memnode.subtype.name} depth-{self.svg_level}"{self.transform()} />\n'
            )
            # Write rectangle
            self.svg_writer.write(
                self.nesting_prefix
                + f'<line x1="{cur_h / UNITY}" y1="{cur_v / UNITY}" x2="{(cur_h + width) / UNITY}" y2="{(cur_v + height) / UNITY}" class="glue depth-{self.svg_level}"{self.transform()} />\n'
            )
            # Write dot
            self.svg_writer.write(
                self.nesting_prefix
                + f'<circle cx="{(cur_h + width) / UNITY}" cy="{(cur_v + height) / UNITY}" r="1" class="glue {memnode.subtype.name} depth-{self.svg_level}"{self.transform()} />\n'
            )
            self.nesting_prefix = self.nesting_prefix[:-2]
            self.svg_writer.write(self.nesting_prefix + "</g>\n")

    def on_rule_out(
        self, node: Node | None, memnode: RuleNode, cur_h: int, cur_v: int, width: int, height: int
    ):
        if self.svg_writer is not None:
            # Write rectangle
            self.svg_writer.write(
                self.nesting_prefix
                + f'<rect x="{cur_h / UNITY}" y="{(cur_v - height) / UNITY}" width="{width / UNITY}" height="{height / UNITY}" class="rule depth-{self.svg_level}" data-pdftex="{_safe_attr(memnode)}"{self.transform()} />\n'
            )

    def on_hlist_out_start(self, memnode: HBoxNode, cur_h: int, cur_v: int):
        if self.svg_writer is not None:
            # Write rectangle and create group names "hlist"
            self.svg_writer.write(
                self.nesting_prefix + f'<g class="hlist" data-pdftex="{_safe_attr(memnode)}">\n'
            )
            self.nesting_prefix += "  "
            self.svg_writer.write(
                self.nesting_prefix
                + f'<rect x="{cur_h / UNITY + self.svg_writer_inset}" y="{cur_v / UNITY - memnode.height + self.svg_writer_inset}" width="{memnode.width - 2 * self.svg_writer_inset}" height="{memnode.height + memnode.depth - 2 * self.svg_writer_inset}" class="hnode depth-{self.svg_level}"{self.transform()} />\n'
            )
            self.svg_level += 1
            # self.svg_writer_inset += 0.5

    def on_hlist_out_end(self, memnode: HBoxNode):
        if self.svg_writer is not None:
            self.nesting_prefix = self.nesting_prefix[:-2]
            self.svg_writer.write(self.nesting_prefix + "</g>\n")
            self.svg_level -= 1
            # self.svg_writer_inset -= 0.5

    def on_vlist_out_start(self, memnode: VBoxNode, cur_h: int, cur_v: int):
        if self.svg_writer is not None:
            # Write rectangle and create group names "vlist"
            self.svg_writer.write(
                self.nesting_prefix + f'<g class="vlist" data-pdftex="{_safe_attr(memnode)}">\n'
            )
            self.nesting_prefix += "  "
            self.svg_writer.write(
                self.nesting_prefix
                + f'<rect x="{cur_h / UNITY + self.svg_writer_inset}" y="{cur_v / UNITY - memnode.height + self.svg_writer_inset}" width="{memnode.width - 2 * self.svg_writer_inset}" height="{memnode.height + memnode.depth - 2 * self.svg_writer_inset}" class="vnode depth-{self.svg_level}"{self.transform()} />\n'
            )
            self.svg_level += 1
            # self.svg_writer_inset += 0.5

    def on_vlist_out_end(self, memnode: VBoxNode):
        if self.svg_writer is not None:
            self.nesting_prefix = self.nesting_prefix[:-2]
            self.svg_writer.write(self.nesting_prefix + "</g>\n")
            self.svg_level -= 1
            # self.svg_writer_inset -= 0.5
