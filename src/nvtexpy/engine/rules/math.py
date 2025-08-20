# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Generator, Iterable

import pdftex

from nvtexpy.engine.common import (
    NULL,
    CommandCode,
    FontProperties,
    Mode,
    Node,
    get_arg_cs,
    get_arg_int,
    get_int_value,
    get_mode,
    get_radical_symbol,
    is_protected,
)
from nvtexpy.engine.debug import DEBUG_FONT, DEBUG_MATH, DEBUG_MATH_ALLOC, DebugHex
from nvtexpy.engine.math import (
    SKIP_EXCEPT_SUB_SUP,
    SKIP_NODE,
    T_SKIP_EXCEPT_SUB_SUP,
    T_SKIP_NODE,
    TexContext,
    noad_to_tex,
)
from nvtexpy.engine.mathdef import (
    DEFAULT_MATH_SYMBOLS,
    MATH_SAME_SPECIAL_SYMBOL,
    MATH_SAME_SYMBOL,
    MATH_SYMBOLS,
)
from nvtexpy.engine.mem import (
    AccentNoad,
    CharNode,
    ChoiceNode,
    FractionNoad,
    GlueNode,
    HBoxNode,
    InnerNoad,
    LeftNoad,
    MemReader,
    Noad,
    NoadField,
    NoadFieldMathChar,
    NoadFieldMathTextChar,
    NoadFieldSubBox,
    NoadFieldSubMList,
)
from nvtexpy.engine.mem import Node as MemNode
from nvtexpy.engine.mem import (
    OverNoad,
    RightNoad,
    RightNoadSubtype,
    RuleNode,
    StyleNode,
    StyleType,
    UnderNoad,
    VBoxNode,
    WhatsitNode,
    node_list,
)
from nvtexpy.engine.rules.base import (
    Rule,
    Rules,
    assert2pagefail,
    begin_macro_hook,
    capture_stack_element,
    end_macro_hook,
    rule_ref,
)
from nvtexpy.engine.rules.graphics import GraphicsFormatter, TrackingGraphicsFormatter
from nvtexpy.engine.rules.strikeout import StrikethroughFormatter
from nvtexpy.engine.stack_formatter import (
    CallbackFormatter,
    DirectBeginFormatterWrapper,
    DirectEndFormatterWrapper,
    FormatterElement,
    SuppressFormatter,
)
from nvtexpy.engine.texstate import CharConverter
from nvtexpy.output.math import (
    AccentStylingITNode,
    ConcatITNode,
    GraphicsITNode,
    ITNode,
    LeftITNode,
    MathCharCmdITNode,
    MathCharITNode,
    MathFunctionITNode,
    MidITNode,
    OrdITNode,
    RadicalITNode,
    RightITNode,
    TableITNode,
    _iter_all,
    compact_math_node,
    to_str,
)
from nvtexpy.output.structured_output import (
    BoundingBox,
    MathElement,
    OverlineElement,
    StructuredOutput,
    SubscriptElement,
    SuperscriptElement,
    UnderlineElement,
)


class SuperscriptFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(SuperscriptElement()),
            name="superscript",
        )


class SubscriptFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(SubscriptElement()),
            name="subscript",
        )


class UnderlineFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(UnderlineElement()),
            name="underline",
        )


class OverlineFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(OverlineElement()),
            name="overline",
        )


class MathFormatter(FormatterElement):
    id: DebugHex
    _math_element: MathElement | None = None

    is_display: bool
    is_eq_num: bool
    tex: ITNode | None

    def __init__(self, id: DebugHex, is_display: bool, is_eq_num: bool, tex: ITNode | None) -> None:
        super().__init__()
        self.id = id
        self.is_display = is_display
        self.is_eq_num = is_eq_num
        self.tex = tex

    def begin(self, struct_out: StructuredOutput):
        super().begin(struct_out)
        if self.tex is not None and not self.tex.is_empty():
            if self.is_display:
                self._math_element = MathElement(
                    code=self.tex,
                    is_display=True,
                    is_eq_num=self.is_eq_num,
                )
            else:
                self._math_element = MathElement(
                    code=self.tex,
                    is_display=False,
                    is_eq_num=self.is_eq_num,
                )

            struct_out.push(self._math_element)

    def end(self, struct_out: StructuredOutput):
        super().end(struct_out)
        if self._math_element is not None:
            struct_out.pop(self._math_element)
            self._math_element = None

    def track(self, cur_h: int, cur_v: int, width: int, height: int, depth: int = 0):
        if self._math_element is not None and width > 0 and height + depth > 0:
            self._math_element.add_box(
                BoundingBox.from_tex(
                    cur_h, cur_v, width, height, depth, self.struct_out.current_transform
                )
            )

    def on_glue_out(
        self, node: Node, memnode: GlueNode, cur_h: int, cur_v: int, width: int, height: int
    ):
        self.track(cur_h, cur_v, width, height)

    def on_rule_out(
        self, node: Node | None, memnode: RuleNode, cur_h: int, cur_v: int, width: int, height: int
    ):
        self.track(cur_h, cur_v, width, height)

    def __repr__(self):
        tex_info = to_str(compact_math_node(self.tex, in_place=False))
        return f"MathFormatter(id={self.id}, is_display={self.is_display}, tex={tex_info!r})"


@dataclass
class MathContext:
    addr: DebugHex | None = None
    hlist_addrs: list[DebugHex] | None = None

    # The formatter for each root node in the math formula
    formatters: dict[int, MathFormatter] | None = None

    is_eq_num: bool = False
    eq_num: "MathContext | None" = None

    table_data: "TableITNode | None" = None
    replace_math: ITNode | None = None

    begin_stack: list[FormatterElement] | None = None

    style: StyleType | None = None
    is_display: bool = False
    skip_for_display_math: bool = False

    # If set to a function, that function will be called on nested math preparations
    # and outer math will be skipped.
    call_stack: list["CallContext"] | None = None

    def combined_tex(self) -> ITNode:
        if self.replace_math is not None:
            return self.replace_math
        assert self.formatters is not None
        return ConcatITNode(
            children=[
                formatter.tex for formatter in self.formatters.values() if formatter.tex is not None
            ],
        )

    def __repr__(self):
        return f"MathContext(id=0x{id(self):08X} addr={self.addr!r}, hlist_addr={self.hlist_addrs!r}, formatters={self.formatters}, is_eq_num={self.is_eq_num}, eq_num={self.eq_num}, table_data={self.table_data})"


class MListTexContext(TexContext):
    unsafe_text_chars = {
        "#": "\\#",
        "$": "\\$",
        "%": "\\%",
        "&": "\\&",
        "\\": "\\textbackslash",
        "^": "\\textasciicircum",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde",
    }

    def __init__(
        self,
        char_converter: CharConverter,
        math_hlists: dict[int, MathContext | T_SKIP_NODE | T_SKIP_EXCEPT_SUB_SUP],
        style: StyleType,
        family_formats: dict[int, str],
        nodes: dict[int, Node | None],
        math_default_charmap: dict[tuple[int, int], str],
        math_accents: dict[tuple[int, int], str],
        math_sqrt_signs: set[tuple[int, int]],
        valid_math_chars: set[int],
        special_nodes_getter: Callable[[int], ITNode | None | T_SKIP_NODE | T_SKIP_EXCEPT_SUB_SUP],
    ):
        super().__init__()
        self.char_converter = char_converter
        self.math_hlists = math_hlists
        self.family_formats = family_formats
        self.nodes = nodes
        self.math_default_charmap = math_default_charmap
        self.math_accents = math_accents
        self.math_sqrt_signs = math_sqrt_signs
        self.valid_math_chars = valid_math_chars
        self.special_nodes_getter = special_nodes_getter
        self.style_stack = [style]

    def get_char(self, font: int, character: int) -> str:
        ch = self.char_converter.get_char(font, character)
        return self.unsafe_text_chars.get(ch, ch)

    def get_math_char_format(self, family: int, character: int) -> str | None:
        if (family, character) in self.math_default_charmap:
            if DEBUG_MATH:
                print(f"fam={family}, char={character} -> fmt=<default>")
            return None
        if character in self.valid_math_chars:
            if DEBUG_MATH:
                print(f"fam={family}, char={character} -> fmt={self.family_formats.get(family)}")
            return self.family_formats.get(family)
        if self.family_formats.get(family) == "mathbf":
            # Special handling for bm...
            if DEBUG_MATH:
                print(f"fam={family}, char={character} -> fmt=mathbf")
            return "mathbf"
        return None

    def get_math_char(self, family: int, character: int) -> str:
        cs = self.math_default_charmap.get((family, character))
        if cs is None:
            cs = self.char_converter.get_math_cs(family, character)
        if DEBUG_MATH:
            print(f"fam={family}, char={character} -> cs={cs!r}")
        return cs

    def get_math_code(self, node_addr: int) -> ITNode | None | T_SKIP_EXCEPT_SUB_SUP | T_SKIP_NODE:
        special_node = self.special_nodes_getter(node_addr)
        if special_node is not None:
            return special_node
        context = self.math_hlists.get(node_addr)
        if DEBUG_MATH:
            print(f"M: Lookup hlist 0x{node_addr:06X} -> {context}")
        if context is None:
            # There is no math context, continue processing as normal
            return None
        elif context is SKIP_NODE or context is SKIP_EXCEPT_SUB_SUP:
            # Node is disabled (except sub/sup) and should not be displayed
            return context
        elif context.table_data is not None:
            return context.table_data
        return context.combined_tex()

    def get_math_accent(self, family: int, character: int) -> str | None:
        return self.math_accents.get((family, character))

    def get_math_sqrtsign(self, family: int, character: int) -> bool:
        return (family, character) in self.math_sqrt_signs

    def get_node(self, addr: int) -> Node | None:
        return self.nodes[addr]


def get_first_visible_node(node: MemNode) -> CharNode | RuleNode | WhatsitNode | None:
    while node is not None:
        if any(
            isinstance(subnode, (CharNode, WhatsitNode))
            or (isinstance(subnode, RuleNode) and not subnode.is_empty)
            for subnode in node.walk()
        ):
            return node
        node = node.link
    return None


def get_last_visible_node(node: MemNode) -> CharNode | RuleNode | WhatsitNode | None:
    last_node = None
    while node is not None:
        if any(
            isinstance(subnode, (CharNode, WhatsitNode))
            or (isinstance(subnode, RuleNode) and not subnode.is_empty)
            for subnode in node.walk()
        ):
            last_node = node
        node = node.link
    return last_node


def get_last_node(head: MemNode) -> MemNode:
    while head.link is not None:
        head = head.link
    return head


def _noad_field_has_visible_child(node: NoadField | None) -> bool:
    if node is None:
        return False
    if isinstance(node, NoadFieldSubBox):
        return any(
            isinstance(subnode, CharNode)
            or (isinstance(subnode, RuleNode) and not subnode.is_empty)
            for subnode in node.box.walk()
        )
    elif isinstance(node, NoadFieldSubMList):
        return any(noad_has_visible_child(subnode) for subnode in node.mlist)
    elif isinstance(node, (NoadFieldMathChar, NoadFieldMathTextChar)):
        return True
    return False


def noad_has_visible_child(node: MemNode | None) -> bool:
    if node is None:
        return False
    elif isinstance(node, Noad):
        return (
            not node.__is_basic__
            or _noad_field_has_visible_child(node.nucleus)
            or _noad_field_has_visible_child(node.subscr)
            or _noad_field_has_visible_child(node.supscr)
        )
    elif isinstance(node, StyleNode):
        return False
    elif isinstance(node, ChoiceNode):
        return any(noad_has_visible_child(subnode) for subnode in node.display_mlist)
    elif isinstance(node, FractionNoad):
        return _noad_field_has_visible_child(node.denominator) or _noad_field_has_visible_child(
            node.numerator
        )
    elif isinstance(node, LeftNoad):
        return node.delimiter is not None
    elif isinstance(node, RightNoad):
        return node.delimiter is not None
    else:
        return any(
            isinstance(subnode, CharNode)
            or (isinstance(subnode, RuleNode) and not subnode.is_empty)
            for subnode in node.walk()
        )


def _iter_nodes_choice_rec(nodes: Iterable[MemNode]) -> Generator[MemNode, None, None]:
    for _subnode in nodes:
        if isinstance(_subnode, ChoiceNode):
            yield from _iter_nodes_choice_rec(_subnode.display_mlist)
            yield from _iter_nodes_choice_rec(_subnode.text_mlist)
            yield from _iter_nodes_choice_rec(_subnode.script_mlist)
            yield from _iter_nodes_choice_rec(_subnode.script_script_mlist)
        else:
            yield _subnode


def iter_nodes_choice(node: MemNode) -> Generator[MemNode, None, None]:
    yield from _iter_nodes_choice_rec(node_list(node))


@Rules.register
class MathRule(Rule):
    """
    For math.

    Order of calls:
    1. math_shift (when math starts)
    2. after_math (when math ends)
    3. fin_mlist (with an address to the mlist which was just finished)
    4. mlist_to_hlist_start (with the address to the mlist which was just finished and the style)
    5. mlist_to_hlist_end (with the address to the mlist which was just finished)

    """

    _mlists: dict[int, MathContext] = {}
    # Hlists for recursive math conversions.
    # Maps the address of the hlist to the MathContext | False=disable.
    # The address may also be a noad field. In that case, extra_formatters_fields contains the
    # mapping to the actual node for deletion.
    _hlists: dict[int, MathContext | T_SKIP_NODE | T_SKIP_EXCEPT_SUB_SUP] = {}
    # Contains noads which span over multiple math contexts which should be united.
    # Format is: {noad_addr: (is_begin, is_end, MathContext) | False=disable}
    _mathspan: dict[int, tuple[bool, bool, MathContext] | T_SKIP_NODE | T_SKIP_EXCEPT_SUB_SUP] = {}
    _math_stack: list[MathContext] = []
    _after_math: MathContext | None = None
    _default_math_charmap: dict[tuple[int, int], str] = {}
    _accents: dict[tuple[int, int], str] = {}
    _sqrtsigns: set[tuple[int, int]] = set()
    _valid_math_chars: set[int] = set()
    _extra_formatters: dict[int, list[FormatterElement]] = defaultdict(list)
    _extra_formatters_fields: dict[int, list[int]] = defaultdict(list)

    _extra_math_context: bool = False
    _math_stack_extra_context: list[str] = []

    disabled: int = 0

    def register_table(self, hlist_addr: int, table: TableITNode) -> None:
        ctx = self._hlists.get(hlist_addr)
        if isinstance(ctx, MathContext):
            if DEBUG_MATH:
                print(f" M: Register update table 0x{hlist_addr:06X} -> {table}")
            ctx.table_data = table
            node = self.state.addr_to_node.get(hlist_addr)
            # Update existing math formatters
            if node is not None and node.direct_formatting is not None:
                for fmt in node.direct_formatting:
                    if isinstance(fmt, MathFormatter):
                        fmt.tex = table
        else:
            if DEBUG_MATH:
                print(f" M: Register new table 0x{hlist_addr:06X} -> {table}")
            self._hlists[hlist_addr] = MathContext(
                addr=DebugHex(hlist_addr),
                table_data=table,
            )

    def push_math_extra_context(self, ctx: str):
        if DEBUG_MATH:
            print(f"M: Push extra context {ctx}")
        self._math_stack_extra_context.append(ctx)
        if len(self._math_stack) == 0:
            self._extra_math_context = True
            self._math_stack.append(MathContext())
            if DEBUG_MATH:
                print(
                    len(self._math_stack) * "  "
                    + f"M: math_stack.push(ctx=0x{id(self.active_context):08X})"
                )

    def pop_math_extra_context(
        self, ctx: str, begin_addrs: list[DebugHex], end_addrs: list[DebugHex], call: ITNode
    ):
        if DEBUG_MATH:
            print(f"M: Pop extra context {ctx}")
        assert self._math_stack_extra_context[-1] == ctx
        self._math_stack_extra_context.pop()
        if len(self._math_stack_extra_context) == 0 and self._extra_math_context:
            popped_ctx = self._math_stack.pop()
            if DEBUG_MATH:
                print(
                    len(self._math_stack) * "  " + f"M: math_stack.pop(ctx=0x{id(popped_ctx):08X})"
                )
            self._extra_math_context = False
            for begin_addr, end_addr in zip(begin_addrs, end_addrs):
                self.register_node_action(
                    begin_addr,
                    end_addr,
                    direct_formatter=MathFormatter(
                        id=begin_addr,
                        is_display=False,
                        is_eq_num=False,
                        tex=call,
                    ),
                )

    def _get_math_special(self, addr: int) -> ITNode | None | T_SKIP_NODE | T_SKIP_EXCEPT_SUB_SUP:
        graphics_fmt = self.get_node_formatter(addr, (GraphicsFormatter, TrackingGraphicsFormatter))
        if DEBUG_MATH:
            print(f"M: Check special node 0x{addr:06X} -> {graphics_fmt}")
        if graphics_fmt is not None:
            return GraphicsITNode()
        return None

    def get_context(self, char_converter: CharConverter, style: StyleType) -> MListTexContext:
        return MListTexContext(
            char_converter,
            self._hlists,
            style,
            self.math_formats,
            self.state.addr_to_node,
            self._default_math_charmap,
            self._accents,
            self._sqrtsigns,
            self._valid_math_chars,
            self._get_math_special,
        )

    @property
    def active_context(self) -> MathContext | None:
        if len(self._math_stack) == 0:
            return None
        return self._math_stack[-1]

    @assert2pagefail
    def math_shift(self, mode: int):
        # assert (
        #     len(self._math_stack) == 0 or len(self._mlists) != 0 or len(self._hlists) != 0
        # ), "Have math stack, but no mlist"
        # Display seems to have another level of math, where the outer has display=True
        self.font_rule.clear_format()
        context = MathContext(begin_stack=self.dump_stack())
        context.is_display = get_mode(mode) == Mode.DISPLAY_MATH
        self._math_stack.append(context)
        if DEBUG_MATH:
            print(
                len(self._math_stack) * "  "
                + f"M: math_shift (ctx=0x{id(self.active_context):08X}): {context}"
            )
            print(
                len(self._math_stack) * "  "
                + f"M: math_stack.push(ctx=0x{id(self.active_context):08X})"
            )

    @assert2pagefail
    def after_math(self, mlist_addr: int, is_eq_num: bool):
        # At this point, the mlist is finished, but not yet converted to hlist
        # (the call will follow).
        context = self._math_stack.pop()
        if DEBUG_MATH:
            print(len(self._math_stack) * "  " + f"M: math_stack.pop(ctx=0x{id(context):08X})")
        if len(self._math_stack) == 0:
            self._extra_math_context = False
            assert len(self._math_stack_extra_context) == 0
        if DEBUG_MATH:
            print(len(self._math_stack) * "  " + f"M: after_math (ctx=0x{id(context):08X})")
        if mlist_addr == NULL:
            # This is a special "end of align in math" case.
            # There is no actual math going on, there will not be a mlist_to_hlist call.
            pass
        else:
            context.addr = DebugHex(mlist_addr)
            self._mlists[mlist_addr] = context
            assert self._after_math is None
            self._after_math = context
            context.is_eq_num = bool(is_eq_num)
            # Iterate through the mlist and fill the subscr_addrs and supscr_addrs
            for node in node_list(MemReader().get_node(mlist_addr)):
                for subnode in node.walk():
                    if isinstance(subnode, Noad):
                        if subnode.subscr is not None:
                            self.register_field_formatting(
                                subnode.addr, subnode.subscr, SubscriptFormatter()
                            )
                        if subnode.supscr is not None:
                            self.register_field_formatting(
                                subnode.addr, subnode.supscr, SuperscriptFormatter()
                            )
                        if isinstance(subnode, UnderNoad) and subnode.nucleus is not None:
                            self.register_field_formatting(
                                subnode.addr, subnode.nucleus, UnderlineFormatter()
                            )
                        elif isinstance(subnode, OverNoad) and subnode.nucleus is not None:
                            self.register_field_formatting(
                                subnode.addr, subnode.nucleus, OverlineFormatter()
                            )
            if is_eq_num:
                assert len(self._math_stack) > 0
                self._math_stack[-1].eq_num = context
                # This is the whole eq num, thus keep the context
                context.skip_for_display_math = False
        if DEBUG_MATH:
            print(
                len(self._math_stack) * "  "
                + f"M: after_math @0x{mlist_addr:06X} is_eq_num={is_eq_num}",
                context,
            )

    def register_field_formatting(
        self, node_addr: int, field: NoadField | None, formatter: FormatterElement
    ):
        if isinstance(field, NoadFieldSubBox):
            # This is more complex. Must set the nodes of the subbox!
            if DEBUG_MATH:
                print(
                    len(self._math_stack) * "  "
                    + f"M: Inner box formatter node=0x{field.box.addr:06X} -> {formatter}"
                )
            self.register_node_action(field.box.addr, direct_formatter=formatter)
        elif field is not None:
            if DEBUG_MATH:
                print(
                    len(self._math_stack) * "  "
                    + f"M: Register extra formatter node=0x{node_addr:06X} field=0x{field.addr:06X} -> {formatter}"
                )
            self._extra_formatters[field.addr].append(formatter)
            self._extra_formatters_fields[node_addr].append(field.addr)

    # def fin_mlist(self, addr: int):
    #     if DEBUG_MATH:
    #         node = MemReader().get_node(addr)
    #         print(f"FIN MLIST: {node}")

    #     node = MemReader().get_node(addr)
    #     while node is not None:
    #         print(f"FIN MLIST: 0x{node.addr:06X}")
    #         self._pending_mlists.add(node.addr)
    #         node = node.link

    @assert2pagefail
    def on_node_dealloc(self, addr: int, size: int):
        # if addr in self._mlists:
        #     del self._mlists[addr]
        if addr in self._hlists:
            if DEBUG_MATH_ALLOC:
                print(f"M: Dealloc 0x{addr:06X} hlists +{size}b")
            del self._hlists[addr]
        if addr in self._mathspan:
            if DEBUG_MATH_ALLOC:
                print(f"M: Dealloc 0x{addr:06X} mathspan +{size}b")
            del self._mathspan[addr]
        if addr in self._extra_formatters_fields:
            if DEBUG_MATH_ALLOC:
                print(f"M: Dealloc 0x{addr:06X} extra formatters fields")
            for field_addrs in self._extra_formatters_fields.pop(addr):
                if DEBUG_MATH_ALLOC:
                    print(f"M: Dealloc 0x{field_addrs:06X} extra formatters")
                if field_addrs in self._extra_formatters:
                    del self._extra_formatters[field_addrs]
                if field_addrs in self._hlists:
                    del self._hlists[field_addrs]
        if addr in self._extra_formatters:
            if DEBUG_MATH_ALLOC:
                print(f"M: Dealloc 0x{addr:06X} extra formatters")
            del self._extra_formatters[addr]

    @assert2pagefail
    def on_copy(self, src_node_addr: int, dst_node_addr: int):
        # Note: Can't copy mlists (only resulting hlists)
        if src_node_addr in self._hlists:
            if DEBUG_MATH_ALLOC:
                print(f"M: Copy 0x{src_node_addr:06X} hlists -> 0x{dst_node_addr:06X}")
            self._hlists[dst_node_addr] = self._hlists[src_node_addr]
        if src_node_addr in self._mathspan:
            if DEBUG_MATH_ALLOC:
                print(f"M: Copy 0x{src_node_addr:06X} mathspan -> 0x{dst_node_addr:06X}")
            self._mathspan[dst_node_addr] = self._mathspan[src_node_addr]
        assert src_node_addr not in self._extra_formatters_fields
        if src_node_addr in self._extra_formatters:
            if DEBUG_MATH_ALLOC:
                print(f"M: Copy 0x{src_node_addr:06X} extra formatters -> 0x{dst_node_addr:06X}")
            self._extra_formatters[dst_node_addr] = self._extra_formatters[src_node_addr]

    @assert2pagefail
    def math_clean_box(self, field_addr: int, new_mlist_addr: int):
        """This is called, when a subbox is to be converted using mlist_to_hlist. This call ist just
        before the mlist_to_hlist call, and is supposed to give translation from the source field to the
        specifically constructed new mlist.

        Args:
            field_addr: The address of the field in the noad to be converted using mlist_to_hlist.
            new_mlist_addr: The address of the new mlist, which is actually passed to mlist_to_hlist.
        """
        outer_math = self.active_context
        if outer_math is not None:
            # Convert sub/super script addresses to the new mlist address
            extra_fmts = self._extra_formatters.get(field_addr)
            if extra_fmts is not None:
                mem = MemReader()
                first_node = None
                last_node = None
                node = mem.get_node(new_mlist_addr)
                while node is not None:
                    # Skip glue nodes, those will not be in the output
                    if noad_has_visible_child(node):
                        if first_node is None:
                            first_node = node
                        last_node = node
                    node = node.link
                if first_node is None:
                    if DEBUG_MATH:
                        print(
                            len(self._math_stack) * "  "
                            + f"M: Clean box 0x{field_addr:06X} -> 0x{new_mlist_addr:06X} None"
                        )
                elif first_node == last_node:
                    self._extra_formatters[first_node.addr] = extra_fmts
                    if DEBUG_MATH:
                        print(
                            len(self._math_stack) * "  "
                            + f"M: Clean box 0x{field_addr:06X} -> 0x{first_node.addr:06X} {extra_fmts}"
                        )
                else:
                    extra_fmts_begin: list[FormatterElement] = [
                        DirectBeginFormatterWrapper(fmt) for fmt in extra_fmts
                    ]
                    extra_fmts_end: list[FormatterElement] = [
                        DirectEndFormatterWrapper(fmt) for fmt in extra_fmts
                    ]
                    self._extra_formatters[first_node.addr] = extra_fmts_begin
                    self._extra_formatters[last_node.addr] = extra_fmts_end
                    if DEBUG_MATH:
                        print(
                            len(self._math_stack) * "  "
                            + f"M: Clean box first 0x{field_addr:06X} -> 0x{first_node.addr:06X} {extra_fmts_begin}"
                        )
                        print(
                            len(self._math_stack) * "  "
                            + f"M: Clean box last 0x{field_addr:06X} -> 0x{last_node.addr:06X} {extra_fmts_end}"
                        )
            elif DEBUG_MATH:
                print(
                    len(self._math_stack) * "  "
                    + f"M: Clean box 0x{field_addr:06X} -> 0x{new_mlist_addr:06X} None"
                )
        elif DEBUG_MATH:
            print(
                len(self._math_stack) * "  "
                + f"M: Clean box 0x{field_addr:06X} -> 0x{new_mlist_addr:06X} None"
            )

    @assert2pagefail
    def math_lig_merge(self, dst_noad_addr: int, src_noad_addr: int):
        # Called when a two ord noads are merged into a single noad (char points to ligature).
        r = MemReader()
        dst_noad = r.get_node(dst_noad_addr)
        src_noad = r.get_node(src_noad_addr)
        assert isinstance(dst_noad, Noad)
        assert isinstance(src_noad, Noad)
        if src_noad.addr in self._extra_formatters:
            if DEBUG_MATH:
                print(
                    len(self._math_stack) * "  "
                    + f"M: Lig merge 0x{dst_noad.addr:06X} <- 0x{src_noad.addr:06X}"
                )
            self._extra_formatters[dst_noad.addr].extend(self._extra_formatters[src_noad.addr])
            del self._extra_formatters[src_noad.addr]
        assert isinstance(src_noad.nucleus, NoadFieldMathChar)
        assert isinstance(dst_noad.nucleus, (NoadFieldMathChar, NoadFieldMathTextChar))
        if src_noad.nucleus.addr in self._extra_formatters:
            if DEBUG_MATH:
                print(
                    len(self._math_stack) * "  "
                    + f"M: Lig merge nucleus 0x{dst_noad.nucleus.addr:06X} <- 0x{src_noad.nucleus.addr:06X}"
                )
            self._extra_formatters[dst_noad.nucleus.addr].extend(
                self._extra_formatters[src_noad.nucleus.addr]
            )
            del self._extra_formatters[src_noad.nucleus.addr]
        if src_noad.subscr is not None and src_noad.subscr.addr in self._extra_formatters:
            assert dst_noad.subscr is not None
            if DEBUG_MATH:
                print(
                    len(self._math_stack) * "  "
                    + f"M: Lig merge subscr 0x{dst_noad.subscr.addr:06X} <- 0x{src_noad.subscr.addr:06X}"
                )
            self._extra_formatters[dst_noad.subscr.addr].extend(
                self._extra_formatters[src_noad.subscr.addr]
            )
            del self._extra_formatters[src_noad.subscr.addr]
        if src_noad.supscr is not None and src_noad.supscr.addr in self._extra_formatters:
            assert dst_noad.supscr is not None
            if DEBUG_MATH:
                print(
                    len(self._math_stack) * "  "
                    + f"M: Lig merge supscr 0x{dst_noad.supscr.addr:06X} <- 0x{src_noad.supscr.addr:06X}"
                )
            self._extra_formatters[dst_noad.supscr.addr].extend(
                self._extra_formatters[src_noad.supscr.addr]
            )
            del self._extra_formatters[src_noad.supscr.addr]

    @assert2pagefail
    def mlist_to_hlist_start(self, mlist_addr: int, style: int):
        if DEBUG_MATH:
            ind = len(self._math_stack) * "  "
            print(ind + f"MLIST_TO_HLIST(0x{mlist_addr:06X}, style={style}), Noads:")
            for print_node in node_list(MemReader().get_node(mlist_addr)):
                print(f"{ind}  {print_node.format_str(ind + '  ')}")
        if self._after_math is not None and self._after_math.addr == mlist_addr:
            # If this follows on after_math -> this is the important call.
            self._after_math = None
        else:
            assert self._after_math is None
            # Just a subformula, ignore
            return
        context = self._mlists[mlist_addr]
        if DEBUG_MATH:
            print(f"MLIST_TO_HLIST(0x{mlist_addr:06X}) {context}")
        assert context.addr == DebugHex(mlist_addr)
        # This math formula is a subformula, still store it, because it might be child of a
        # disabled formula.
        node = MemReader().get_node(mlist_addr)
        assert node is not None
        style_t = StyleType(style & 6)
        context.style = style_t

        # recursively iterate node, and get all sub mlists
        # if DEBUG_MATH:
        #     print(len(self._math_stack) * "  " + "M: Registered hlists:")
        #     for addr, ctx in self._hlists.items():
        #         if ctx is False:
        #             print(len(self._math_stack) * "  " + f"  0x{addr:06X} False")
        #         else:
        #             print(len(self._math_stack) * "  " + f"  0x{addr:06X} {ctx.formatter.tex}")
        context.formatters = {
            subnode.addr: MathFormatter(
                id=DebugHex(mlist_addr),
                is_display=context.is_display,
                tex=compact_math_node(
                    noad_to_tex(
                        subnode,
                        self.get_context(self.state.char_converter, style_t),
                    ),
                    in_place=True,
                ),
                is_eq_num=context.is_eq_num,
            )
            for subnode in iter_nodes_choice(node)
        }

        if context.skip_for_display_math:
            if DEBUG_MATH:
                print(len(self._math_stack) * "  " + "M: Skip for display math, nodes:")
            for subnode in node.walk():
                subctx = self._hlists.get(subnode.addr)
                if isinstance(subctx, MathContext):
                    if DEBUG_MATH:
                        print(len(self._math_stack) * "  " + f" - {subctx}")
                    subctx.is_display = True
                    assert subctx.formatters is not None
                    for subfmt in subctx.formatters.values():
                        subfmt.is_display = True

        if DEBUG_MATH:
            ind = len(self._math_stack) * "  "
            print(f"{ind}M: Math nodes:")
            for print_node in iter_nodes_choice(node):
                fmt = context.formatters[print_node.addr]
                if fmt.tex is None:
                    print(f"{ind}  - ITNodes: <empty>")
                else:
                    print(f"{ind}  - ITNodes: {fmt.tex.dbg_str(indent=ind + '    ')}")
                print(f"{ind}    Noads: {print_node.format_str(ind + '    ', max_depth=20)}")
        # print(f"Tex: {tex!r}")
        # self.print_node_list(mlist_addr)
        if DEBUG_MATH:
            print(len(self._math_stack) * "  " + f"M: mlist_to_hlist {context}")
        self._math_stack.append(context)
        if DEBUG_MATH:
            print(
                len(self._math_stack) * "  "
                + f"M: math_stack.push(ctx=0x{id(self.active_context):08X})"
            )

    def math_choice_converted(
        self, src_noad_addr: int, dst_noad_start_addr: int, dst_noad_end_addr: int
    ):
        # This is called after a choice node has chosen which nodes to take.
        # The src_noad_addr is the choice node address, and is only for conversion and should not be accessed any more.
        # The dst_noad_start_addr and dst_noad_end_addr are the noads which are now the new noads sublist (start/end are both included).
        mem = MemReader()
        start_noad = mem.get_node(dst_noad_start_addr)
        end_noad = mem.get_node(dst_noad_end_addr)
        assert start_noad is not None
        assert end_noad is not None
        if DEBUG_MATH:
            print(
                f"M: Choice converted src=0x{src_noad_addr:06X}, dst=0x{dst_noad_start_addr:06X} -> 0x{dst_noad_end_addr:06X}:"
            )
        first_node = None
        last_node = None
        for node in mem.iter_nodes(start_noad, end_noad, include_last=True):
            if DEBUG_MATH:
                print(f"  - {node.format_str('    ')}")
            if noad_has_visible_child(node):
                if first_node is None:
                    first_node = node
                last_node = node
        if first_node is None:
            return
        assert last_node is not None

        # Handle extra formatters
        extra_fmts = self._extra_formatters.pop(src_noad_addr, None)
        if extra_fmts is not None:
            if first_node.addr == last_node.addr:
                self._extra_formatters[first_node.addr].extend(extra_fmts)
                if DEBUG_MATH:
                    for extra_fmt in extra_fmts:
                        print(f"  - {first_node.addr:06X} {extra_fmt}")
            else:
                for extra_fmt in extra_fmts:
                    if isinstance(extra_fmt, DirectBeginFormatterWrapper):
                        self._extra_formatters[first_node.addr].append(extra_fmt)
                        if DEBUG_MATH:
                            print(f"  - {first_node.addr:06X} {extra_fmt}")
                    elif isinstance(extra_fmt, DirectEndFormatterWrapper):
                        self._extra_formatters[last_node.addr].append(extra_fmt)
                        if DEBUG_MATH:
                            print(f"  - {last_node.addr:06X} {extra_fmt}")
                    else:
                        self._extra_formatters[first_node.addr].append(
                            DirectBeginFormatterWrapper(extra_fmt)
                        )
                        self._extra_formatters[last_node.addr].append(
                            DirectEndFormatterWrapper(extra_fmt)
                        )
                        if DEBUG_MATH:
                            print(
                                f"  - {first_node.addr:06X} {self._extra_formatters[first_node.addr][-1]}"
                            )
                            print(
                                f"  - {last_node.addr:06X} {self._extra_formatters[last_node.addr][-1]}"
                            )

        # Handle mathspans
        src_mathspan = self._mathspan.pop(src_noad_addr, None)
        if src_mathspan is not None:
            # Just disable all, will be revived below
            node = None
            for node in mem.iter_nodes(start_noad, end_noad, include_last=True):
                self._mathspan[node.addr] = SKIP_NODE
            if node is not None and src_mathspan is SKIP_EXCEPT_SUB_SUP:
                self._mathspan[node.addr] = src_mathspan

            if src_mathspan not in (SKIP_NODE, SKIP_EXCEPT_SUB_SUP):
                # Revive begin/end
                mathspan_begin, mathspan_end, mathspan_ctx = src_mathspan
                if not mathspan_end:
                    self._mathspan[first_node.addr] = (True, False, mathspan_ctx)
                elif not mathspan_begin:
                    self._mathspan[last_node.addr] = (False, True, mathspan_ctx)
                elif first_node.addr == last_node.addr:
                    self._mathspan[first_node.addr] = src_mathspan
                else:
                    self._mathspan[first_node.addr] = (True, False, mathspan_ctx)
                    self._mathspan[last_node.addr] = (False, True, mathspan_ctx)

    @assert2pagefail
    def mlist_to_hlist_converted(self, mlist_addr: int, noad_addr: int):
        # This is called after the mlist noads have been converted to hlists. The inner lists
        # are contained in the noad.new_hlist field.
        context = self._mlists.get(mlist_addr)
        if DEBUG_MATH:
            print(
                f"MLIST_TO_HLIST_CONVERTED0(mlist=0x{mlist_addr:06X}, noad=0x{noad_addr:06X}) {context}"
            )
            for print_node in node_list(MemReader().get_node(mlist_addr)):
                print(
                    f"  - {'-> ' if print_node.addr == noad_addr else ''}{type(print_node).__name__}(addr=0x{print_node.addr:06X})"
                )
        # if self.active_context is not None and self.active_context.call_replacements is not None and noad_addr in self.active_context.call_replacements:
        #     node = MemReader().get_node(noad_addr)
        #     if Noad.is_noad(node):
        #         hlist_first_node = node.new_hlist_head
        #         if hlist_first_node is not None:
        #             print(len(self._math_stack) * "  " + f"M: replace by function new hlist: {hlist_first_node.format_str(len(self._math_stack) * '  ' + '  ')}")
        #             # TODO: Now the new_hlist contains the hlist for the noad
        #             first_visible_node = get_first_visible_node(hlist_first_node)
        #             last_visible_node = get_last_visible_node(hlist_first_node)
        #             for node in node_list(hlist_first_node.link):
        #                 if DEBUG_MATH:
        #                     print(
        #                         len(self._math_stack) * "  "
        #                         + f"M: Disable hlist node 0x{node.addr:06X}"
        #                     )
        #                 self.active_context.call_replacements[node.addr] = False
        #             if first_visible_node is not None:
        #                 print(len(self._math_stack) * "  " + f"M: replace by function first 0x{first_visible_node.addr:06X} = 0x{noad_addr:06X}")
        #                 self.active_context.call_replacements[first_visible_node.addr] = self.active_context.call_replacements[noad_addr]
        if (
            context is not None
            and len(self._math_stack) > 0
            and context == self._math_stack[-1]
            and not context.skip_for_display_math
        ):
            node = MemReader().get_node(noad_addr)
            if Noad.is_noad(node):
                assert node is not None
                if DEBUG_MATH:
                    print(
                        f"MLIST_TO_HLIST_CONVERTED(mlist=0x{mlist_addr:06X}, noad=0x{noad_addr:06X}) {context} -> new_hlist={list(node.new_hlist)}"
                    )
                assert context.formatters is not None
                formatter = context.formatters.get(node.addr)
                if formatter is None:
                    # TODO: Is this correct? This can happen e.g. for a display noad.
                    if DEBUG_MATH:
                        print(f"No formatter for node {self._hlists.get(node.addr)}")
                    return
                hlist_first_node = node.new_hlist_head
                if hlist_first_node is None:
                    if DEBUG_MATH:
                        print("M: No math created")
                    return
                # TODO: Now the new_hlist contains the hlist for the noad
                first_visible_node = get_first_visible_node(hlist_first_node)
                last_visible_node = get_last_visible_node(hlist_first_node)
                if DEBUG_MATH:
                    ind = len(self._math_stack) * "  "
                    print(ind + "M: Nodes:")
                    for node in node_list(hlist_first_node):
                        print(f"{ind}  - {node.format_str(ind + '  ')}")
                mathspan = self._mathspan.get(noad_addr)
                if mathspan is SKIP_NODE or mathspan is SKIP_EXCEPT_SUB_SUP:
                    for node in node_list(hlist_first_node):
                        if DEBUG_MATH:
                            print(len(self._math_stack) * "  " + f"M: Math disabled node {node}")
                        self.register_node_action(node.addr, direct_formatter=SuppressFormatter())
                else:
                    if mathspan is not None:
                        mathspan_begin, mathspan_end, mathspan_ctx = mathspan
                        assert mathspan_ctx.formatters is not None
                        if DEBUG_MATH:
                            print(
                                len(self._math_stack) * "  "
                                + f"M: Mathspan overwriting orig formatter {node!r} -> {formatter}"
                            )
                        orig_formatter = formatter
                        formatter = mathspan_ctx.formatters[noad_addr]
                        if (
                            mathspan_end
                            and isinstance(orig_formatter.tex, OrdITNode)
                            and (
                                orig_formatter.tex.subscr is not None
                                or orig_formatter.tex.supscr is not None
                            )
                        ):
                            # This is a special case, where a sub/supscript is not part of the mathspan.
                            formatter.tex = OrdITNode(
                                nucleus=formatter.tex,
                                subscr=orig_formatter.tex.subscr,
                                supscr=orig_formatter.tex.supscr,
                            )
                            if DEBUG_MATH:
                                print(
                                    len(self._math_stack) * "  "
                                    + f"M: Mathspan append subscr/supscr -> {formatter}"
                                )
                        if not mathspan_end:
                            if DEBUG_MATH:
                                print(
                                    len(self._math_stack) * "  "
                                    + f"M: Mathspan begin node {node!r} -> {formatter}"
                                )
                            formatter = DirectBeginFormatterWrapper(formatter)
                        elif not mathspan_begin:
                            if DEBUG_MATH:
                                print(
                                    len(self._math_stack) * "  "
                                    + f"M: Mathspan end node {node!r} -> {formatter}"
                                )
                            formatter = DirectEndFormatterWrapper(formatter)
                        elif DEBUG_MATH:
                            print(
                                len(self._math_stack) * "  "
                                + f"M: Mathspan node {node!r} -> {formatter}"
                            )

                    # Disable all other hlist elements
                    for node in node_list(hlist_first_node.link):
                        if DEBUG_MATH:
                            print(
                                len(self._math_stack) * "  "
                                + f"M: Disable hlist node 0x{node.addr:06X}"
                            )
                        if self._hlists.get(node.addr) != SKIP_EXCEPT_SUB_SUP:
                            self._hlists[node.addr] = SKIP_NODE
                    if first_visible_node is not None and last_visible_node is not None:
                        if not self.disabled:
                            if DEBUG_MATH:
                                print(
                                    len(self._math_stack) * "  " + f"M: First {first_visible_node}"
                                )
                                print(len(self._math_stack) * "  " + f"M: Last {last_visible_node}")
                            self.register_node_action(
                                first_visible_node.addr,
                                last_visible_node.addr,
                                direct_formatter=formatter,
                                begin_stack=context.begin_stack,
                                dump_stack=True,
                            )

            elif isinstance(node, (VBoxNode, HBoxNode)):
                # TODO: This can be a hlist directly. What cases are this?
                if DEBUG_MATH:
                    print(
                        f"MLIST_TO_HLIST_CONVERTED(mlist=0x{mlist_addr:06X}, noad=0x{noad_addr:06X}) {context} -> node={node} DIRECT HLIST"
                    )
                assert context.formatters is not None
                formatter = context.formatters.get(noad_addr)
                if formatter is None:
                    # TODO: Is this correct? This can happen e.g. for a display noad.
                    return
                if not self.disabled:
                    self.register_node_action(
                        node.addr,
                        node.addr,
                        direct_formatter=formatter,
                        begin_stack=context.begin_stack,
                        dump_stack=True,
                    )
            else:
                if DEBUG_MATH:
                    print(
                        f"MLIST_TO_HLIST_CONVERTED(mlist=0x{mlist_addr:06X}, noad=0x{noad_addr:06X}) {context} -> node={node} UNPROCESSED NODE, skipping"
                    )
        extra_formatters = self._extra_formatters.get(noad_addr)
        if extra_formatters is not None:
            if DEBUG_MATH:
                print(f"M: Extra formatters for 0x{mlist_addr:06X}: {extra_formatters}")
            node = MemReader().get_node(noad_addr)
            assert node is not None
            if Noad.is_noad(node):
                hlist_first_node = node.new_hlist_head
                if hlist_first_node is None:
                    return
                first_visible_node = get_first_visible_node(hlist_first_node)
                last_visible_node = get_last_visible_node(hlist_first_node)
                if first_visible_node is not None and last_visible_node is not None:
                    for fmt in extra_formatters:
                        self.register_node_action(
                            first_visible_node.addr,
                            last_visible_node.addr,
                            direct_formatter=fmt,
                        )

    @assert2pagefail
    def mlist_to_hlist_end(self, mlist_addr: int, hlist_addr: int):
        # Note: The mlist cannot be accessed anymore, it has been deallocated.
        context = self._mlists.pop(mlist_addr, None)
        if context is not None and len(self._math_stack) > 0 and context == self._math_stack[-1]:
            self._math_expecting_next = None
            popped_context = self._math_stack.pop()
            if DEBUG_MATH:
                print(
                    len(self._math_stack) * "  "
                    + f"M: math_stack.pop(ctx=0x{id(popped_context):08X})"
                )
            assert popped_context == context
            if not context.skip_for_display_math:
                self._hlists[hlist_addr] = context
            context.hlist_addrs = [
                DebugHex(node.addr) for node in node_list(MemReader().get_node(hlist_addr))
            ]
            for addr in context.hlist_addrs:
                if addr not in self._hlists:
                    self._hlists[addr] = SKIP_NODE
            self.font_rule.clear_format()
            if DEBUG_MATH:
                print(len(self._math_stack) * "  " + f"M: mlist_to_hlist_end {context}")
            if (
                len(self._math_stack) > 0
                and self._math_stack[-1].call_stack is not None
                and len(self._math_stack[-1].call_stack) > 0
            ):
                self._math_stack[-1].call_stack[-1](context.combined_tex())
            # print(f"Finalize Math @0x{mlist_addr:06X} -> @0x{hlist_addr:06X} {context.tex}")
        assert len(self._math_stack) != 0 or len(self._mlists) == 0, "Have math stack, but no mlist"

    math_formats: dict[int, str] = {}
    cur_math_format = None

    @begin_macro_hook("extract@alph@from@version", str_args=True)
    @assert2pagefail
    def on_extract_math_version(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: tuple[str | None, ...] | None,
        head_addr: int,
        tail_addr: int,
    ):
        # E.g.: extract@alph@from@version('\\mv@normal', '8', '\\mathit ')
        # This might include non-mentioned types thatn the direct typing from below.
        assert (
            args is not None
            and len(args) == 3
            and isinstance(args[1], str)
            and isinstance(args[2], str)
        )
        fam = get_int_value(args[1])
        dst_cmd = args[2][1:].rstrip()
        if self.math_formats.get(fam) != fam:
            if DEBUG_MATH:
                print(f"M: Assigning math format {dst_cmd} to fam={args[1]}={fam}")
            self.math_formats[fam] = dst_cmd

    @begin_macro_hook(
        [
            "mathbb ",
            "mathbf ",
            "mathcal ",
            "mathfrak ",
            "mathit ",
            "mathnormal ",
            "mathrm ",
            "mathsf ",
            "mathtt ",
            "mathscr ",
        ]
    )
    @assert2pagefail
    def on_begin_math_format(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        self.cur_math_format = cs_name[:-1]

        def end_math_format(*_):
            self.cur_math_format = None

        return end_math_format

    @end_macro_hook("check@mathfonts")
    @assert2pagefail
    def on_check_math_fonts(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        if len(self._default_math_charmap) == 0:
            if DEBUG_MATH:
                print("M: Setup default math charmap")
            # Setup the inverse map
            charmap = pdftex.get_math_charmap()
            self._valid_math_chars = {
                i
                for i, (char, fam) in enumerate(charmap)
                if char != -1 and fam != -1 and chr(i).isprintable()
            }
            self._default_math_charmap = {
                (fam, char): DEFAULT_MATH_SYMBOLS[chr(i)]
                for i, (char, fam) in enumerate(charmap)
                if char != -1 and fam != -1 and chr(i) in DEFAULT_MATH_SYMBOLS
            }
            if DEBUG_MATH:
                print("M: Inverse charmap:")
                for (family, char), cs in self._default_math_charmap.items():
                    print(f"  fam={family}, char={char} -> cs={cs!r}")

    @begin_macro_hook("use@mathgroup", str_args=True)
    @assert2pagefail
    def on_begin_math_group(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: tuple[str | None, ...] | None,
        head_addr: int,
        tail_addr: int,
    ):
        assert args is not None and len(args) >= 2 and isinstance(args[1], str)
        if self.cur_math_format is None:
            print(f"M: Unknown math version for fam={args[1]}")
        if self.cur_math_format is not None:
            try:
                fam = get_int_value(args[1])
            except ValueError:
                print(f"M: Cannot assigning math format {self.cur_math_format} to fam={args[1]}")
                return
            if fam not in self.math_formats:
                if DEBUG_MATH:
                    print(f"M: Assigning math format {self.cur_math_format} to fam={args[1]}={fam}")
                self.math_formats[fam] = self.cur_math_format
            elif DEBUG_MATH:
                print(
                    f"M: Already assigned math format {self.math_formats[fam]} (want to assign {self.cur_math_format}) to fam={args[1]}={fam}"
                )
            self.cur_math_format = None

    @begin_macro_hook("getanddefine@fonts", raw_args=True)
    def on_getanddefine_fonts(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: tuple[int, ...] | None,
        head_addr: int,
        tail_addr: int,
    ):
        # Special handling for bm package
        assert args is not None and len(args) == 2, args
        try:
            name = get_arg_cs(args[0])
        except Exception:
            name = None
            fam = get_arg_int(args[0])
        path = get_arg_cs(args[1])
        if not path:
            print(f"FNT: Can't get and define fonts fam={name!r} path={path!r}")
            return

        def end(*_):
            """
            Convert a string like 'OML/cmm/m/it' to a FontProperties object.

            _re_series = re.compile(r"(ul|el|l|sl|m|sb|b|eb|ub)(uc|ec|c|sc|m|sx|x|ex|ux)?")
            _shapes = ('n', 'it', 'sl', 'sc', 'scit', 'scsl', 'sw')
            """
            nonlocal fam
            if name is not None:
                try:
                    fam = get_int_value("\\" + name)
                except Exception as e:
                    print(f"FNT: Get and define fonts fam={name!r} path={path!r} -> fam=?: {e}")
                    return

            font = pdftex.get_family_font(fam)

            _, fam_val, series_val, shape_val, *_ = path.split("/")
            props = FontProperties(
                font=font,
                bold="b" in series_val,
                italic_or_slanted="it" in shape_val or "sl" in shape_val,
                monospace=fam_val in ["cmtt", "lmtt", "pcr", "qcr"],
            )

            if DEBUG_FONT:
                print(
                    f"FNT: Get and define fonts {name!r} path={path!r} -> fam={fam} font=0x{font:02X} {props}"
                )
            if name is not None and name.startswith("symbold"):
                # Bold, special package bm
                if self.math_formats.get(fam) != fam:
                    if DEBUG_FONT:
                        print(
                            f"FNT: Assigning math format 'boldsymbol' to fam={name}={fam} font=0x{font:02X} {props}"
                        )
                    self.math_formats[fam] = "boldsymbol"
            self.state.family_properties[fam] = props

        return end

    _size_code = ("text", "script", "scriptscript")

    @end_macro_hook("math@fonts")
    def on_math_fonts(self, *_):
        # Apparently, there is space for 16 families in 3 sizes (text, script, scriptscript)
        for fam in range(16):
            props = self.state.family_properties.get(fam)
            if props is None:
                if DEBUG_FONT:
                    print(f"FNT: Math font fam={fam} font=NULL")
                continue
            for size in range(3):
                font = pdftex.get_family_font(fam + size * 16)
                if font == 0:
                    if DEBUG_FONT:
                        print(f"FNT: Math font fam={fam} size={self._size_code[size]} font=NULL")
                    continue
                props.font = font
                if DEBUG_FONT:
                    print(
                        f"FNT: Math font fam={fam} size={self._size_code[size]} font=0x{font:02X} {props}"
                    )
                self.state.font_properties[font] = props

    # @begin_macro_hook("DeclareMathSymbol")
    # @assert2pagefail
    # def on_declare_math_symbol(
    #     self,
    #     cs_name: str,
    #     cmd: CommandCode,
    #     args: tuple[str | None, ...] | None,
    #     head_addr: int,
    #     tail_addr: int,
    # ):
    #     assert args is not None and len(args) >= 4 and isinstance(args[0], str) and isinstance(args[2], str) and isinstance(args[3], str)
    #     symbol_def, _, font, character_id = args[:4]
    #     assert symbol_def.startswith("\\")
    #     symbol_def = symbol_def[1:]
    #     assert character_id.startswith("\"")
    #     character_id = int(character_id[1:], base=16)
    #     if DEBUG_MATH:
    #         print(f"M: Declare math symbol {symbol_def} -> fam={font}, char=0x{character_id:02X}")

    _math_expecting_next = None

    @begin_macro_hook(MATH_SYMBOLS)
    def on_math_symbol(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        if self.active_context is not None:
            cmd_text = MATH_SAME_SYMBOL.get(cs_name, cs_name)
            cmd_text = MATH_SAME_SPECIAL_SYMBOL.get(cmd_text, f"\\{cmd_text}")
            if DEBUG_MATH:
                print(f"M: Math symbol {cmd_text}")
            if cmd == CommandCode.math_given:
                # Find the actual translation
                val = get_int_value(f"\\{cs_name}")
                char = val % 256
                fam = (val >> 8) % 16
                if DEBUG_MATH:
                    print(f"M: Math symbol {cmd_text} -> fam={fam}, char=0x{char:02X}")
                self._default_math_charmap[(fam, char)] = cmd_text
            else:
                # Try to replace the math symbol
                if DEBUG_MATH:
                    print(f"M: Math symbol {cmd_text}: {cmd!r} in {self.active_context}")
                self._begin_handle_command(
                    cs_name, head_addr, tail_addr, cmd_text, allow_text_mode=False
                )

                def end(head_addr: int, tail_addr: int):
                    if not self._end_handle_command(
                        cs_name, head_addr, tail_addr, is_char=True, allow_text_mode=False
                    ):
                        # Uhoh, this means, that the actual command did not produce the tokens
                        if DEBUG_MATH:
                            print("M: DID NOT PRODUCE TOKENS!")
                        # Try again with the current definition of next@
                        self._math_expecting_next = cmd_text

                return end

    @begin_macro_hook("sqrtsign")
    def on_sqrt_sign(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        sqrtsign = get_radical_symbol(b"sqrtsign")
        small = (sqrtsign >> 12) & 0xFFF
        large = sqrtsign & 0xFFF
        small_fam = (small >> 8) & 0xF
        small_char = small & 0xFF
        large_fam = (large >> 8) & 0xF
        large_char = large & 0xFF
        if DEBUG_MATH:
            print(
                f"M: {cs_name}: {cmd} -> 0x{sqrtsign:06X} -> small=(fam={small_fam}, char=0x{small_char:02X}), large=(fam={large_fam}, char=0x{large_char:02X})"
            )
        self._sqrtsigns = {
            (
                small_fam,
                small_char,
            ),
            (
                large_fam,
                large_char,
            ),
        }

    @begin_macro_hook("next@")
    def on_math_symbol_next(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        if self._math_expecting_next == "@sqrt":
            # Special case for sqrt
            if DEBUG_MATH:
                print(f"M: Math sqrt next@: {cs_name}")
            self._begin_handle_command(
                cs_name, head_addr, tail_addr, "\\sqrt", allow_text_mode=False
            )

            def end(head_addr: int, tail_addr: int):
                if DEBUG_MATH:
                    print(f"M: Math sqrt next@ END: {cs_name}")

                def find_radical_inner(node: ITNode | None) -> ITNode | None:
                    if isinstance(node, RadicalITNode):
                        return node.child
                    if isinstance(node, ConcatITNode):
                        for child in node.children:
                            if isinstance(child, RadicalITNode):
                                return child.child
                    return None

                def combine_args(ctx: CallContext) -> MathFunctionITNode:
                    return MathFunctionITNode(
                        function=ctx.function,
                        args=[
                            ctx.args[0],
                            find_radical_inner(ctx.args[1]),
                        ],
                        args_prefer_braced=["Bracket", True],
                    )

                self._end_handle_command(
                    cs_name,
                    head_addr,
                    tail_addr,
                    is_char=False,
                    allow_text_mode=False,
                    fn_args=combine_args,
                )

            return end
        elif self._math_expecting_next is not None:
            cs_name = self._math_expecting_next
            if DEBUG_MATH:
                print(f"M: Math Symbol next@: {cs_name}")
            self._begin_handle_command(
                cs_name, head_addr, tail_addr, cs_name, allow_text_mode=False
            )

            def end(head_addr: int, tail_addr: int):
                if not self._end_handle_command(
                    cs_name, head_addr, tail_addr, is_char=True, allow_text_mode=False
                ):
                    # Uhoh, this means, that the actual command did not produce the tokens
                    if DEBUG_MATH:
                        print("M: DID NOT PRODUCE TOKENS!")
                    # Try again with the current definition of next@
                    self._math_expecting_next = cs_name

            return end

    def _begin_handle_command(
        self, cs_name: str, head_addr: int, tail_addr: int, function: str, allow_text_mode: bool
    ):
        # Find the right call that was actually inserted
        if DEBUG_MATH:
            print(f"{cs_name} BEGIN (ctx=0x{id(self.active_context):08X})")
        if allow_text_mode:
            self.push_math_extra_context(cs_name)
        assert self.active_context is not None
        if self.active_context.call_stack is None:
            self.active_context.call_stack = []
        self.active_context.call_stack.append(
            CallContext(
                function=function,
                head_addr=head_addr,
                tail_addr=tail_addr,
            )
        )

    def end_right(self, head_addr: int, tail_addr: int):
        if DEBUG_MATH:
            print(f"RIGHT END (ctx=0x{id(self.active_context):08X}) for:")
            if head_addr != tail_addr:
                node = MemReader().get_node(tail_addr)
                if node:
                    print(f"  - {node.format_str('    ')}")
        if (
            self.active_context is not None
            and self.active_context.call_stack is not None
            and len(self.active_context.call_stack) > 0
        ):
            self.active_context.call_stack[-1].has_right_end = True

    def _node_diff(
        self, ctx: "CallContext", head_addr: int, tail_addr: int, is_char: bool
    ) -> tuple[list[DebugHex], list[DebugHex], list[DebugHex]]:
        # Iterate over head -> tail and find the added nodes at the end
        if DEBUG_MATH:
            print("M: NodeDiff:")
        if ctx.has_right_end and is_char:
            if DEBUG_MATH:
                print("  - Has right end -> nested in InnerNoad")
            if head_addr == tail_addr:
                if DEBUG_MATH:
                    print("M: NodeDiff EMPTY")
                return [], [], []
            tail_node = MemReader().get_node(tail_addr)
            if DEBUG_MATH:
                print(f"  - {tail_node.format_str(indent='  ')}")
            if isinstance(tail_node, InnerNoad) and isinstance(
                tail_node.nucleus, NoadFieldSubMList
            ):
                nodes = tail_node.nucleus.mlist
            else:
                nodes = MemReader().iter_head_tail(head_addr, tail_addr)
        else:
            nodes = MemReader().iter_head_tail(head_addr, tail_addr)
        added_nodes = []
        first_node = None
        last_node = None
        end_nodes = []
        next_nodes = []
        for node in nodes:
            if not noad_has_visible_child(node):
                continue
            if node.addr not in ctx._pre_nodes:
                if first_node is None:
                    first_node = node
                    if isinstance(node, ChoiceNode):
                        for choice_mlist in node.mlists():
                            choice_first_node = None
                            choice_last_node = None
                            for choice_node in choice_mlist:
                                if not noad_has_visible_child(choice_node):
                                    continue
                                if choice_first_node is None:
                                    choice_first_node = choice_node
                                else:
                                    if DEBUG_MATH:
                                        print(f"  - {choice_node}")
                                    next_nodes.append(choice_node.addr)
                                    choice_last_node = choice_node
                            if choice_first_node is not None:
                                if DEBUG_MATH:
                                    print(
                                        f"  - ADD(choice): {choice_first_node.format_str(indent='  ')}"
                                    )
                                added_nodes.append(choice_first_node.addr)
                            if choice_last_node is not None:
                                if DEBUG_MATH:
                                    print(
                                        f"  - END(choice): {choice_last_node.format_str(indent='  ')}"
                                    )
                                end_nodes.append(choice_last_node.addr)
                else:
                    next_nodes.append(node.addr)
                    if DEBUG_MATH:
                        print(f"  - {node.format_str(indent='  ')}")
                last_node = node
        if first_node is not None:
            added_nodes.insert(0, first_node.addr)
            if DEBUG_MATH:
                print(f"  - ADD(first): {first_node.format_str(indent='  ')}")
        if last_node is not None:
            if DEBUG_MATH:
                print(f"  - END(first): {last_node.format_str(indent='  ')}")
            end_nodes.insert(0, last_node.addr)
        if DEBUG_MATH:
            print("M: NodeDiff END")
        return added_nodes, next_nodes, end_nodes

    def _end_handle_command(
        self,
        cs_name: str,
        head_addr: int,
        tail_addr: int,
        allow_text_mode: bool,
        is_char: bool,
        fn_args: (
            Callable[["CallContext"], tuple[int, ...] | int | ITNode | None]
            | tuple[int, ...]
            | int
            | ITNode
            | None
        ) = None,
        text_formatter: FormatterElement | None = None,
    ):
        assert (
            self.active_context is not None
            and self.active_context.call_stack is not None
            and len(self.active_context.call_stack) > 0
        )

        mem = MemReader()

        call = self.active_context.call_stack.pop()
        replace_addrs, omit_addrs, end_addrs = self._node_diff(
            call, head_addr, tail_addr, is_char=is_char
        )

        if DEBUG_MATH:
            print("M: All args:")
            for arg in call.args:
                print(f"   - {arg!r}")
        if fn_args is None:
            fn_args = tuple()
        if callable(fn_args):
            fn_args = fn_args(call)
            if fn_args is None:
                return
        if isinstance(fn_args, int):
            fn_args = (fn_args,)

        if isinstance(fn_args, ITNode):
            math = fn_args
        elif is_char:
            assert len(fn_args) == 0
            format = None
            for arg in call.args:
                # Find any char and use format
                for node in _iter_all(arg):
                    if isinstance(node, MathCharITNode):
                        format = node.format
                        break
            math = MathCharCmdITNode(
                char_cmd=call.function,
                format=format,
            )

            # Check for left/right/mid
            if len(replace_addrs) == 1:
                noad = MemReader().get_node(replace_addrs[0])
                if isinstance(noad, LeftNoad):
                    math = LeftITNode(delimiter=math)
                    if DEBUG_MATH:
                        print("M: -> Left")
                elif isinstance(noad, RightNoad):
                    if noad.subtype == RightNoadSubtype.MIDDLE_NOAD:
                        math = MidITNode(delimiter=math)
                        if DEBUG_MATH:
                            print("M: -> Mid")
                    else:
                        math = RightITNode(delimiter=math)
                        if DEBUG_MATH:
                            print("M: -> Right")
        else:
            if DEBUG_MATH:
                print(f"Use args {fn_args}")
            try:
                math = MathFunctionITNode(
                    function=call.function,
                    args=[call.args[i] for i in fn_args],
                    args_prefer_braced=[True] * len(fn_args),
                )
            except IndexError:
                raise ValueError(
                    f"Invalid args {call.args} for {call.function}, expected {fn_args}"
                )

        if DEBUG_MATH:
            print(f"{cs_name} END (ctx=0x{id(self.active_context):08X}): {math.dbg_str()}")

        if is_char and len(replace_addrs) == 0:
            return False

        # Find the right call that was actually inserted
        math_formatter = MathFormatter(
            id=DebugHex(replace_addrs[0]),
            is_display=self.active_context.is_display,
            is_eq_num=False,
            tex=math,
        )
        new_ctx = MathContext(
            addr=DebugHex(replace_addrs[0]),
            is_display=self.active_context.is_display,
            hlist_addrs=replace_addrs + omit_addrs,
            replace_math=math,
            formatters={addr: math_formatter for addr in (replace_addrs + end_addrs)},
        )

        for omit_addr in omit_addrs:
            self._hlists[omit_addr] = SKIP_NODE
            self._mathspan[omit_addr] = SKIP_NODE
            if DEBUG_MATH:
                print(f" - Mathspan disable: 0x{omit_addr:06X}")
        if len(omit_addrs) > 0:
            self._hlists[omit_addrs[-1]] = SKIP_EXCEPT_SUB_SUP
        for start_addr, end_addr in zip(replace_addrs, end_addrs):
            node = mem.get_node(start_addr)
            if (
                start_addr == end_addr
                and isinstance(node, Noad)
                and node.__is_basic__
                and node.supscr is None
                and node.subscr is None
                and node.nucleus is not None
            ):
                self._hlists[node.nucleus.addr] = new_ctx
                self._extra_formatters_fields[start_addr].append(node.nucleus.addr)
                if DEBUG_MATH:
                    print(
                        f"M: Add formatter for node=0x{start_addr:06X} field=0x{node.nucleus.addr:06X}: {math_formatter}"
                    )
            else:
                self._hlists[start_addr] = new_ctx

                if start_addr == end_addr:
                    if DEBUG_MATH:
                        print(f" - Replace Mathspan: 0x{start_addr:06X}")
                    self._mathspan[start_addr] = (True, True, new_ctx)
                else:
                    self._mathspan[start_addr] = (True, False, new_ctx)
                    self._mathspan[end_addr] = (False, True, new_ctx)
                    if DEBUG_MATH:
                        print(f" - Start Replace Mathspan: 0x{start_addr:06X}")
                        print(f" - End Replace Mathspan: 0x{end_addr:06X}")

        if text_formatter is not None:
            for begin_addr, end_addr in zip(replace_addrs, end_addrs):
                if begin_addr == end_addr:
                    node = mem.get_node(begin_addr)
                    if (
                        isinstance(node, Noad)
                        and node.__is_basic__
                        and node.supscr is None
                        and node.subscr is None
                        and node.nucleus is not None
                    ):
                        if DEBUG_MATH:
                            print("  - use nucleus")
                        # (node_addr, field_addr)
                        self._extra_formatters[node.nucleus.addr].append(text_formatter)
                        self._extra_formatters_fields[begin_addr].append(node.nucleus.addr)
                        if DEBUG_MATH:
                            print(
                                f"M: Add specific formatter for node=0x{begin_addr:06X} field=0x{node.nucleus.addr:06X}: {text_formatter}"
                            )
                        continue
                    else:
                        if DEBUG_MATH:
                            print(
                                f"M: Add specific formatter for node=0x{begin_addr:06X}: {text_formatter}"
                            )
                        node = mem.get_node(begin_addr)
                        self._extra_formatters[begin_addr].append(text_formatter)
                else:
                    self._extra_formatters[begin_addr].append(
                        DirectBeginFormatterWrapper(text_formatter)
                    )
                    self._extra_formatters[end_addr].append(
                        DirectEndFormatterWrapper(text_formatter)
                    )
                    if DEBUG_MATH:
                        print(
                            f"M: Add specific formatter for node=0x{begin_addr:06X}: {text_formatter}"
                        )
                if DEBUG_MATH:
                    print(
                        f"M: Add formatter for node=0x{begin_addr:06X}-0x{end_addr:06X}: {text_formatter}"
                    )
                self.register_node_action(begin_addr, end_addr, direct_formatter=text_formatter)

        if allow_text_mode:
            self.pop_math_extra_context(cs_name, replace_addrs, end_addrs, math)
        return True

    # Specific math commands following...

    @end_macro_hook("@sqrt")
    def on_begin_sqrt(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        # Actually, process the next@ command.
        self._math_expecting_next = cs_name

    @begin_macro_hook(["widetilde ", "widehat ", "widetilde", "widehat"])
    def on_begin_wide_accent(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        if not self.active_context:
            return

        # Check if it's defined as normal math accent
        if cs_name in ("widetilde", "widehat"):
            if is_protected(f"\\{cs_name}"):
                return None
            cmd_name = cs_name
        else:
            cmd_name = cs_name[:-1]

        self._begin_handle_command(
            cs_name, head_addr, tail_addr, f"\\{cs_name}", allow_text_mode=False
        )

        def end(head_addr: int, tail_addr: int):
            def make_fn(ctx: CallContext):
                tail_node = MemReader().get_node(tail_addr)
                if isinstance(tail_node, AccentNoad):
                    # Apply default accent handling. Much simpler :)
                    self._accents[(tail_node.accent_family, tail_node.accent_character)] = cmd_name
                    if DEBUG_MATH:
                        print(
                            f"M: Accents for (fam={tail_node.accent_family}, char=0x{tail_node.accent_character:02X}) = {cmd_name}"
                        )
                    return None
                if len(ctx.args) > 0:
                    return 0

            self._end_handle_command(
                cs_name,
                head_addr,
                tail_addr,
                allow_text_mode=False,
                is_char=False,
                fn_args=make_fn,
            )

        return end

    @begin_macro_hook("mathaccentV", str_args=True)
    def on_begin_accent_def(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: tuple[str | None, ...] | None,
        head_addr: int,
        tail_addr: int,
    ):
        # TODO: This helps the correct AccentNoad identification.
        # It does not work for nested accents, because the innermost character will
        # appear after the AccentNoads.
        assert args is not None and len(args) >= 4
        cmd_name = args[0]
        if cmd_name not in (
            "hat",
            "check",
            "tilde",
            "acute",
            "grave",
            "dot",
            "ddot",
            "breve",
            "bar",
            "vec",
            "mathring",
        ):
            return
        assert isinstance(args[1], str) and isinstance(args[2], str) and isinstance(args[3], str)
        self._accents[(int(args[1], 16), int(args[2] + args[3], 16))] = cmd_name

    # _mathacc_stack: list = ()
    # _mathacc_nesting_stack: list[ITNode | None] | None = None
    _mathacc_stack: int | None = None
    _mathacc_nesting_level: int | None = None
    _mathacc_nesting_top: ITNode | None = None

    @begin_macro_hook("macc@nested", str_args=True)
    def on_begin_accent_nested(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: tuple[str | None, ...] | None,
        head_addr: int,
        tail_addr: int,
    ):
        assert self._mathacc_stack is not None and self._mathacc_stack > 1
        if DEBUG_MATH:
            print(f"macc@nested for {cs_name}({args}):")
        self._mathacc_nesting_level = 0

    @begin_macro_hook("mathaccentV", str_args=True)
    def on_begin_accent(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: tuple[str | None, ...] | None,
        head_addr: int,
        tail_addr: int,
    ):
        # TODO: This works more reliable for nested constructs, which will have the character at the
        # end and not actually nested :exploding_head:
        if args is None or len(args) <= 0 or not isinstance(args[0], str):
            return None
        cmd_name = args[0]
        if cmd_name not in (
            "hat",
            "check",
            "tilde",
            "acute",
            "grave",
            "dot",
            "ddot",
            "breve",
            "bar",
            "vec",
            "mathring",
        ):
            return

        if len(args) == 4:
            # This is a nested call! Handle separately!
            assert self._mathacc_stack is not None, "Nesting without root call"
            self._mathacc_stack += 1
            print(f"M: macc nesting {self._mathacc_stack}")
            return
        elif self._mathacc_stack is None:
            self._mathacc_stack = 1
            self._mathacc_nesting_level = None
            self._mathacc_nesting_top = None
            if DEBUG_MATH:
                print("M: macc top level")

        if self._mathacc_nesting_level is not None:
            self._mathacc_nesting_level += 1
            self._mathacc_nesting_top = None
            if DEBUG_MATH:
                print(f"M: macc nesting at {self._mathacc_nesting_level}/{self._mathacc_stack}")

        self._begin_handle_command(
            cs_name, head_addr, tail_addr, f"\\{cmd_name}", allow_text_mode=False
        )

        def end(head_addr: int, tail_addr: int):
            if DEBUG_MATH:
                if self._mathacc_nesting_level is not None:
                    print(f"M: macc nesting at {self._mathacc_nesting_level}/{self._mathacc_stack}")

            def _make_fn(call: CallContext):
                if self._mathacc_nesting_level is not None:
                    self._mathacc_nesting_level -= 1
                    arg = self._mathacc_nesting_top
                    if arg is None:
                        arg = call.args[-1]
                else:
                    tail_node = MemReader().get_node(tail_addr)
                    if isinstance(tail_node, AccentNoad):
                        # Apply default accent handling. Much simpler and less error prone :)
                        self._accents[(tail_node.accent_family, tail_node.accent_character)] = (
                            cmd_name
                        )
                        if DEBUG_MATH:
                            print(
                                f"M: Accents for (fam={tail_node.accent_family}, char=0x{tail_node.accent_character:02X}) = {cmd_name}"
                            )
                        return None
                    arg = call.args[-1]
                math = AccentStylingITNode(
                    child=arg,
                    function=call.function,
                )
                if DEBUG_MATH:
                    print(f"M: accent: --> {math}")
                if self._mathacc_nesting_level is not None:
                    self._mathacc_nesting_top = math
                    if self._mathacc_nesting_level > 0:
                        return None
                return math

            self._end_handle_command(
                cs_name,
                head_addr,
                tail_addr,
                allow_text_mode=False,
                is_char=False,
                fn_args=_make_fn,
            )

            # if self._mathacc_nesting_level == 0:
            #     # Cleanup
            #     self._mathacc_nesting_top = None
            #     self._mathacc_nesting_level = None
            #     self._mathacc_stack = None

        return end

    @begin_macro_hook(["cancel ", "bcancel ", "xcancel "])
    def on_begin_cancel(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        # The created inner math modes are:
        # 1. mathchoice textstyle
        #   a. Measurement box (discarded)
        #   b. Real box
        #   c. cancel box (/ or \)
        #   d. (additional cancel box for xcancel)
        # 2. mathchoice scriptstyle
        # 3. mathchoice scriptscriptstyle
        # 4. mathchoice displaystyle
        self._begin_handle_command(
            cs_name, head_addr, tail_addr, f"\\{cs_name[:-1]}", allow_text_mode=True
        )

        def end(head_addr: int, tail_addr: int):
            self._end_handle_command(
                cs_name,
                head_addr,
                tail_addr,
                allow_text_mode=True,
                is_char=False,
                fn_args=1,
                text_formatter=StrikethroughFormatter(),
            )

        return end

    @begin_macro_hook("@can@slash")
    def on_begin_cancel_slash(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        # Just suppress the cancellation characters
        fmt = SuppressFormatter()
        self.push_formatter(fmt)

        def end(head_addr: int, tail_addr: int):
            self.pop_formatter(fmt)

        return end

    @begin_macro_hook(["fbox "])
    def on_begin_boxed(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        self._begin_handle_command(cs_name, head_addr, tail_addr, "\\boxed", allow_text_mode=True)

        def end(head_addr: int, tail_addr: int):
            self._end_handle_command(
                cs_name, head_addr, tail_addr, allow_text_mode=True, is_char=False, fn_args=0
            )

        return end

    _re_whatarrow = re.compile(r"\\([a-z]+)arrowfill@")

    @begin_macro_hook(["overarrow@", "underarrow@"], str_args=True)
    def on_begin_overarrow(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: tuple[str | None, ...] | None,
        head_addr: int,
        tail_addr: int,
    ):
        if cs_name == "overarrow@":
            where = "over"
        elif cs_name == "underarrow@":
            where = "under"
        else:
            assert False, f"Unknown arrow type {cs_name}"
        assert args is not None and len(args) >= 3 and isinstance(args[0], str)
        what_match = self._re_whatarrow.fullmatch(args[0])
        assert what_match is not None
        which = what_match.group(1)
        self._begin_handle_command(
            cs_name, head_addr, tail_addr, f"\\{where}{which}arrow", allow_text_mode=False
        )

        def end(head_addr: int, tail_addr: int):
            self._end_handle_command(
                cs_name,
                head_addr,
                tail_addr,
                allow_text_mode=False,
                is_char=False,
                fn_args=lambda call: (1 if call.function.startswith("\\over") else 0),
            )

        return end

    @begin_macro_hook(["underbrace ", "overbrace ", "underbrace", "overbrace"])
    def on_begin_overbrace(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        if cs_name in ("underbrace", "overbrace"):
            if is_protected(f"\\{cs_name}"):
                return None
            cmd_name = cs_name
        else:
            cmd_name = cs_name[:-1]

        self._begin_handle_command(
            cs_name, head_addr, tail_addr, f"\\{cmd_name}", allow_text_mode=False
        )

        def end(head_addr: int, tail_addr: int):
            self._end_handle_command(
                cs_name,
                head_addr,
                tail_addr,
                allow_text_mode=False,
                is_char=False,
                fn_args=lambda call: (1 if call.function.startswith("\\over") else 0),
            )

        return end

    @begin_macro_hook(["MT_underbracket_II:w", "MT_overbracket_II:w"])
    def on_begin_bracket(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        if cs_name == "MT_underbracket_II:w":
            inner_name = "underbracket"
        elif cs_name in "MT_overbracket_II:w":
            inner_name = "overbracket"
        else:
            assert False, f"Unknown bracket type {cs_name}"
        self._begin_handle_command(
            cs_name, head_addr, tail_addr, f"\\{inner_name}", allow_text_mode=False
        )

        def end(head_addr: int, tail_addr: int):
            self._end_handle_command(
                cs_name,
                head_addr,
                tail_addr,
                allow_text_mode=False,
                is_char=False,
                fn_args=lambda call: (1 if call.function.startswith("\\over") else 0),
            )

        return end

    in_endmathdisplay = capture_stack_element("endmathdisplay")

    @begin_macro_hook("df@tag")
    def on_separate_eq_num(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        if not self.in_endmathdisplay:
            return
        print(f"M: Eq num for ctx {self.active_context}")
        assert self.active_context is not None
        # The current context is the outer math display, which will also include the eq num

        def end(head_addr: int, tail_addr: int):
            print(f"M: Eq num end for ctx {self.active_context}:")
            for node in MemReader().iter_head_tail(head_addr, tail_addr):
                print(f"  - {node.format_str('    ')}")
                if not noad_has_visible_child(node):
                    continue
                # Find the right call that was actually inserted
                math_formatter = MathFormatter(
                    id=DebugHex(node.addr),
                    is_display=True,
                    is_eq_num=True,
                    tex=compact_math_node(
                        noad_to_tex(
                            node,
                            self.get_context(self.state.char_converter, StyleType.text),
                        ),
                        in_place=True,
                    ),
                )
                new_ctx = MathContext(
                    addr=DebugHex(node.addr),
                    is_display=True,
                    hlist_addrs=[DebugHex(node.addr)],
                    formatters={node.addr: math_formatter},
                )
                if DEBUG_MATH:
                    print(f"M: Eq num 0x{node.addr:06X} ctx {new_ctx}")
                self._hlists[node.addr] = new_ctx
                self.register_node_action(node.addr, direct_formatter=math_formatter)
            assert self.active_context is not None
            self.active_context.skip_for_display_math = True
            self.active_context.is_eq_num = True
            self.active_context.is_display = True

        return end


# Post:
# - \int\int\int\int -> \iiiint
# - \cdot\cdot\cdot -> \cdots


@dataclass
class CallContext:
    _: dataclasses.KW_ONLY

    function: str
    args: list[ITNode | None] = field(default_factory=list)
    _n_arg: int = 0

    head_addr: int
    tail_addr: int

    has_right_end: bool = False

    _pre_nodes: set[int] = field(init=False)

    def __post_init__(self):
        self._pre_nodes = set(
            DebugHex(node.addr)
            for node in MemReader().iter_head_tail(self.head_addr, self.tail_addr)
        )

    def __call__(self, tex: ITNode) -> ITNode:
        self.args.append(tex)
        return tex


@Rules.register
class URLRule(Rule):
    """For urls using \\url."""

    math_rule: MathRule = rule_ref(MathRule)

    @begin_macro_hook("Url@FormatString")
    def on_url_format_string(self, *args):
        if self.active:
            # Don't nest title
            return None

        self.active = True

        # URL starts, disable math rule
        self.math_rule.disabled += 1

        def end_url(head_addr: int, tail_addr: int):
            # URL ends, re-enable math rule
            self.math_rule.disabled -= 1
            self.active = False

        self.state.save_levels[-1].end_hooks.append(end_url)
