# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from nvtexpy.engine.common import Node
from nvtexpy.engine.debug import DEBUG_GRAPHICS
from nvtexpy.engine.hooks import enable_callbacks
from nvtexpy.engine.mem import UNITY, CharNode, HBoxNode, MemReader, NoadFieldSubBox
from nvtexpy.engine.mem import Node as MemNode
from nvtexpy.engine.mem import OrdNoad, RuleNode, VBoxNode
from nvtexpy.engine.rules.base import Rule, Rules, assert2pagefail, begin_macro_hook, end_macro_hook
from nvtexpy.engine.stack_formatter import FormatterElement
from nvtexpy.output.structured_output import (
    BoundingBox,
    Element,
    GraphicsElement,
    StructuredOutput,
    SuppressElement,
)


@dataclass
class GraphicsFormatter(FormatterElement):
    format_whatsit = True
    suppress_formats = True
    _graphics_element: Element | None = None
    _element: Element | None = None
    _bounding_boxes: list[BoundingBox] = field(default_factory=list)

    def begin(self, struct_out: StructuredOutput):
        super().begin(struct_out)
        self._graphics_element = struct_out.push(GraphicsElement())
        self._element = struct_out.push(SuppressElement())
        if self._bounding_boxes:
            self._graphics_element.boxes.extend(self._bounding_boxes)

    def end(self, struct_out: StructuredOutput):
        super().end(struct_out)
        if self._element is not None:
            struct_out.pop(self._element)
            assert self._graphics_element
            struct_out.pop(self._graphics_element)
            self._element = None

    def _on_track_box(self, cur_h: int, cur_v: int, width: int, height: int, depth: int):
        box = BoundingBox.from_tex(
            cur_h,
            cur_v,
            width,
            height,
            depth,
            self.struct_out.current_transform,
        )
        if self._graphics_element is not None:
            self._graphics_element.boxes.append(box)
        else:
            self._bounding_boxes.append(box)

    def on_track_box(self, node: MemNode, cur_h: int, cur_v: int):
        if DEBUG_GRAPHICS:
            print(f"G: track_box {node} {cur_h} {cur_v} {node.width} {node.height} {node.depth}")
        if isinstance(node, (HBoxNode, VBoxNode)):
            self._on_track_box(
                cur_h,
                cur_v,
                int(node.width * UNITY),
                int(node.height * UNITY),
                int(node.depth * UNITY),
            )


class TrackingGraphicsFormatter(GraphicsFormatter):
    def begin(self, struct_out: StructuredOutput):
        super().begin(struct_out)

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
        if DEBUG_GRAPHICS:
            print(f"G: Char out {node}")
        self._on_track_box(cur_h, cur_v, width, height, depth)

    def on_rule_out(
        self, node: Node | None, memnode: RuleNode, cur_h: int, cur_v: int, width: int, height: int
    ):
        if DEBUG_GRAPHICS:
            print(f"G: Rule out {node}")
        self._on_track_box(cur_h, cur_v, width, height, 0)

    def on_hlist_out_start(self, node: Node | None, memnode: HBoxNode, cur_h: int, cur_v: int):
        if DEBUG_GRAPHICS:
            print(f"G: HList out {node}")
        self._on_track_box(
            cur_h,
            cur_v,
            int(memnode.width * UNITY),
            int(memnode.height * UNITY),
            int(memnode.depth * UNITY),
        )

    def on_vlist_out_start(self, node: Node | None, memnode: HBoxNode, cur_h: int, cur_v: int):
        if DEBUG_GRAPHICS:
            print(f"G: VList out {node}")
        self._on_track_box(
            cur_h,
            cur_v,
            int(memnode.width * UNITY),
            int(memnode.height * UNITY),
            int(memnode.depth * UNITY),
        )


@Rules.register
class GraphicsRule(Rule):
    """
    For graphics.
    """

    formatter_stack: list[GraphicsFormatter] = []

    @begin_macro_hook("Gin@page")
    @assert2pagefail
    def on_graphics_page_begin(self, *args):
        # This is for \includegraphics[page=...]{...}, storing the page
        # Also applies to \includepdf[pages=...]{...}, which uses this internally
        assert False, "Including a specific page from a source document is ntent, skipping page."

    @begin_macro_hook("@err@")
    @assert2pagefail
    def on_graphics_error(self, *args):
        # If a graphics could not be included, skip the page.
        assert False, "Error in graphics"

    @begin_macro_hook("Ginclude@graphics")
    @assert2pagefail
    def on_include_graphics(self, *args):
        if self.active:
            # Don't nest graphics
            return None

        if DEBUG_GRAPHICS:
            print("G: Begin")

        self.active = True
        self.font_rule.clear_format()
        formatter = GraphicsFormatter()
        self.formatter_stack.append(formatter)

        self.start_stack = self.dump_stack()

        def end(
            head_addr: int,
            tail_addr: int,
        ):
            if DEBUG_GRAPHICS:
                print("G: XXXX End 2")
            assert self.active
            formatter = self.formatter_stack.pop()

            head_node = None
            tail_node = None
            for node in MemReader().iter_head_tail(head_addr, tail_addr):
                if DEBUG_GRAPHICS:
                    print(f"G: next {node}")
                if head_node is None:
                    if DEBUG_GRAPHICS:
                        print(f"G: head {head_node}")
                    head_node = node
                tail_node = node
            if DEBUG_GRAPHICS:
                print(f"G: tail {tail_node}")
            if head_node is None or tail_node is None:
                return

            # TODO: Hack, somehow, sometimes it's the node after tail?!
            while tail_node.link is not None:
                tail_node = tail_node.link
                if DEBUG_GRAPHICS:
                    print(f"G: tail next {tail_node}")
                assert isinstance(tail_node, MemNode)

            if isinstance(tail_node, OrdNoad):
                # Graphics in math
                if (
                    not isinstance(tail_node.nucleus, NoadFieldSubBox)
                    or tail_node.subscr is not None
                    or tail_node.supscr is not None
                ):
                    self.active = False
                    if DEBUG_GRAPHICS:
                        print("G: Brokenend math")
                    return
                tail_node = tail_node.nucleus.box
                if DEBUG_GRAPHICS:
                    print(f"G: tail {tail_node}")

            if not isinstance(tail_node, HBoxNode):
                self.active = False
                if DEBUG_GRAPHICS:
                    print("G: Brokenend")
                return

            assert isinstance(tail_node, HBoxNode), tail_node

            if DEBUG_GRAPHICS:
                print(f"G: End, register node 0x{tail_node.addr:06X}")

            self.font_rule.clear_format()

            # This is delicate: Set the formatter here and assign it to the hbox
            self.register_node_action(
                tail_node.addr,
                begin_action=lambda node,
                mem_node,
                cur_h,
                cur_v,
                struct_out: formatter.on_track_box(mem_node, cur_h, cur_v),
                direct_formatter=formatter,
                begin_stack=self.start_stack,
                dump_stack=True,
            )

            self.active = False

        return end


@Rules.register
class XYMatrix(Rule):
    """For urls using \\url."""

    @begin_macro_hook("xymatrix")
    def on_xy_matrix(self, *args):
        if self.active:
            # Don't nest title
            return None

        self.active = True
        self.formatter = TrackingGraphicsFormatter()
        self.push_formatter(self.formatter)
        # self.math_rule.disabled += 1

        if DEBUG_GRAPHICS:
            print("BEGIN XY MATRIX")

    @end_macro_hook("endxy")
    def on_xy_end(self, *args):
        if self.active:
            if DEBUG_GRAPHICS:
                print("END XY MATRIX")
            self.active = False
            self.pop_formatter(self.formatter)
            # Store track next glue as well
            self.state.input_formatter_store_next = True
            # MathRule.disabled -= 1


@Rules.register
class PictureRule(Rule):
    """For \\begin{picture}."""

    @begin_macro_hook("picture")
    def on_picture(self, *args):
        if self.active:
            # Don't nest title
            return None

        self.active = True
        formatter = TrackingGraphicsFormatter()
        self.push_formatter(formatter)

        def end_picture(*args):
            self.pop_formatter(formatter)
            self.active = False
            # Store track next glue as well
            self.state.input_formatter_store_next = True

        self.state.save_levels[-1].end_hooks.append(end_picture)
        # return end_picture


@Rules.register
class TikzPictureRule(Rule):
    """For \\begin{tikzpicture}."""

    def on_node_alloc(self, addr: int, size: int):
        if self.active:
            # Store all nodes?
            self.register_node_action(addr, dump_stack=True)

    @begin_macro_hook("tikzpicture")
    def on_tikzpicture(self, *args):
        if self.active:
            # Don't nest title
            return None

        self.active = True
        formatter = TrackingGraphicsFormatter()
        self.push_formatter(formatter)

        print("G: Disable callbacks inside")
        enable_callbacks(False)

        def end_picture(*args):
            self.pop_formatter(formatter)
            print("G: Reenable callbacks")
            enable_callbacks(True)
            self.active = False
            # Store track next glue as well
            self.state.input_formatter_store_next = True

        self.state.save_levels[-1].end_hooks.append(end_picture)
