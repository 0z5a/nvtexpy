# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Callable, Dict, Iterable, Literal, Type, TypeVar

from nvtexpy.engine.common import Node as NodeRef
from nvtexpy.engine.mem import (
    AccentNoad,
    AdjustNode,
    BinNoad,
    CharNode,
    ChoiceNode,
    CloseNoad,
    Default,
    DiscNode,
    FractionNoad,
    GlueNode,
    GlueSubtype,
    HBoxNode,
    InnerNoad,
    InsNode,
    KernNode,
    LeftNoad,
    LigatureNode,
    MarginalKern,
    MarkNode,
    MathNode,
    Noad,
    NoadField,
    NoadFieldMathChar,
    NoadFieldMathTextChar,
    NoadFieldSubBox,
    NoadFieldSubMList,
    Node,
    OpenNoad,
    OpNoad,
    OrdNoad,
    OverNoad,
    PenaltyNode,
    PunctNoad,
    RadicalNoad,
    RelNoad,
    RightNoad,
    RightNoadSubtype,
    RuleNode,
    StyleNode,
    StyleType,
    UnderNoad,
    UnsetNode,
    VBoxNode,
    VCenterNoad,
    WhatsitNode,
    WhatsitPdfRestoreNode,
    WhatsitPdfSaveNode,
    WhatsitPdfSetMatrixNode,
)
from nvtexpy.output.math import (
    AccentStylingITNode,
    ConcatITNode,
    FractionITNode,
    GlueITNode,
    ITNode,
    LeftITNode,
    MathCharCmdITNode,
    MathCharITNode,
    MathUnknownCharITNode,
    MidITNode,
    OrdITNode,
    OverlineStylingITNode,
    RadicalITNode,
    RightITNode,
    TextCharCmdITNode,
    TextCharITNode,
    TextFontInfo,
    TextRuleITNode,
    TextUnknownCharITNode,
    UnderlineStylingITNode,
)

SKIP_EXCEPT_SUB_SUP = True
SKIP_NODE = False
T_SKIP_EXCEPT_SUB_SUP = Literal[True]
T_SKIP_NODE = Literal[False]


class TexContext(ABC):
    style_stack: list[StyleType]

    cur_font: str | None = None
    last_was_hyphen_or_disc: bool = False

    @abstractmethod
    def get_char(self, font: int, character: int) -> str: ...

    @abstractmethod
    def get_math_char_format(self, family: int, character: int) -> str | None: ...

    @abstractmethod
    def get_math_char(self, family: int, character: int) -> str: ...

    @abstractmethod
    def get_math_code(
        self, node_addr: int
    ) -> ITNode | None | T_SKIP_NODE | T_SKIP_EXCEPT_SUB_SUP: ...

    def get_math_char_code(
        self, family: int, character: int
    ) -> MathCharITNode | MathCharCmdITNode | MathUnknownCharITNode:
        math_char = self.get_math_char(family, character)
        math_format = self.get_math_char_format(family, character) or None
        if math_char.startswith("\\unknown{"):
            font, char = _split_unknown(math_char)
            return MathUnknownCharITNode(font=font, char=int(char), format=math_format)
        if math_char.startswith("\\"):
            return MathCharCmdITNode(char_cmd=math_char, format=math_format)
        return MathCharITNode(
            char=math_char,
            format=math_format,
        )

    @abstractmethod
    def get_math_accent(self, family: int, character: int) -> str | None: ...

    @abstractmethod
    def get_math_sqrtsign(self, family: int, character: int) -> bool: ...

    @abstractmethod
    def get_node(self, addr: int) -> NodeRef | None: ...

    @contextmanager
    def level_ctx(self):
        self.push_level()
        yield
        self.pop_level()

    @property
    def style(self) -> StyleType:
        return self.style_stack[-1]

    def set_style(self, style: StyleType):
        self.style_stack[-1] = style

    def push_level(self):
        self.style_stack.append(self.style_stack[-1])

    def pop_level(self):
        self.style_stack.pop()


TNode = TypeVar("TNode", bound=Node, covariant=True)
TexProcessor = Callable[[TNode, TexContext], ITNode | None]

_processors: Dict[Type[Node], TexProcessor] = {}


def _register_processor(type: Type[TNode]):
    def register_processor(fn: TexProcessor[TNode]):
        _processors[type] = fn
        return fn

    return register_processor


def _to_tex(node: Node, context: TexContext) -> ITNode | None:
    code = context.get_math_code(node.addr)
    if code is SKIP_NODE:
        # The node is disabled
        return None
    elif code is SKIP_EXCEPT_SUB_SUP:
        inner = _processors[type(node)](node, context)
        print(f"XXX: Checking {inner!r}")
        if isinstance(inner, OrdITNode):
            inner.nucleus = None
            return inner
        return None
    if code is not None:
        return code
    return _processors[type(node)](node, context)


def _to_tex_list(nodes: Iterable[Node], context: TexContext) -> list[ITNode]:
    with context.level_ctx():
        converted = (_to_tex(node, context) for node in nodes)
        return [cnode for cnode in converted if cnode]


@_register_processor(type=CharNode)
def char_node_to_tex(node: CharNode, context: TexContext) -> ITNode | None:
    node_data = context.get_node(node.addr)
    assert node_data is not None, "Missing node for char"
    if node_data.is_hyphen:
        context.last_was_hyphen_or_disc = True
        return None
    context.last_was_hyphen_or_disc = False
    if node_data.font_properties is not None:
        fmt = TextFontInfo(
            bold=node_data.font_properties.bold,
            italic_or_slanted=node_data.font_properties.italic_or_slanted,
            monospace=node_data.font_properties.monospace,
        )
    else:
        fmt = None
    if node_data.replacement_character is not None:
        if not node_data.replacement_character:
            return None
        return TextCharITNode(char=node_data.replacement_character, format=fmt)
    ch = context.get_char(node.font, node.character)
    if ch.startswith("\\unknown{"):
        font, char = _split_unknown(ch)
        return TextUnknownCharITNode(font=font, char=int(char), format=fmt)
    elif ch.startswith("\\"):
        return TextCharCmdITNode(char_cmd=ch, format=fmt)
    return TextCharITNode(char=ch, format=fmt)


@_register_processor(type=HBoxNode)
def hbox_node_to_tex(node: HBoxNode, context: TexContext) -> ITNode | None:
    return ConcatITNode(children=_to_tex_list(node.list, context))


@_register_processor(type=VBoxNode)
def vbox_node_to_tex(node: VBoxNode, context: TexContext) -> ITNode | None:
    return ConcatITNode(children=_to_tex_list(node.list, context))


@_register_processor(type=RuleNode)
def rule_node_to_tex(node: RuleNode, context: TexContext) -> ITNode | None:
    if node.is_empty:
        return None
    return TextRuleITNode(hrule=node.height is not None)


@_register_processor(type=InsNode)
def ins_node_to_tex(node: InsNode, context: TexContext) -> ITNode | None:
    if node.ins_ptr is not None:
        return _to_tex(node.ins_ptr, context)
    return None


@_register_processor(type=MarkNode)
def mark_node_to_tex(node: MarkNode, context: TexContext) -> ITNode | None:
    return None


@_register_processor(type=AdjustNode)
def adjust_node_to_tex(node: AdjustNode, context: TexContext) -> ITNode | None:
    # This should actually move the adjust node out of this context. Let's hope it doesn't happen.
    return ConcatITNode(children=_to_tex_list(node.adjust, context))


@_register_processor(type=LigatureNode)
def ligature_node_to_tex(node: LigatureNode, context: TexContext) -> ITNode | None:
    return ConcatITNode(children=_to_tex_list(node.ligs, context))


@_register_processor(type=DiscNode)
def disc_node_to_tex(node: DiscNode, context: TexContext) -> ITNode | None:
    context.last_was_hyphen_or_disc = True
    return None


@_register_processor(type=WhatsitNode)
@_register_processor(type=WhatsitPdfSaveNode)
@_register_processor(type=WhatsitPdfRestoreNode)
@_register_processor(type=WhatsitPdfSetMatrixNode)
def whatsit_node_to_tex(node: WhatsitNode, context: TexContext) -> ITNode | None:
    return None


@_register_processor(type=MathNode)
def math_node_to_tex(node: MathNode, context: TexContext) -> ITNode | None:
    # This should actually not happen, but be replaced by the inner code.
    return None
    # assert False, "Cannot handle math node"


@_register_processor(type=GlueNode)
def glue_node_to_tex(node: GlueNode, context: TexContext) -> ITNode | None:
    if node.subtype in (
        GlueSubtype.normal,
        GlueSubtype.space_skip_code,
        GlueSubtype.xspace_skip_code,
    ) and (node.glue.width > 0 or (node.glue.stretch > 0 and node.glue.stretch_order > 0)):
        if context.last_was_hyphen_or_disc:
            return None
        return GlueITNode(char=" ")
    elif node.subtype in (
        GlueSubtype.par_skip_code,
        GlueSubtype.par_fill_skip_code,
        GlueSubtype.line_skip_code,
        GlueSubtype.above_display_short_skip_code,
        GlueSubtype.above_display_skip_code,
        GlueSubtype.below_display_short_skip_code,
        GlueSubtype.below_display_skip_code,
    ):
        return GlueITNode(char="\n")
    elif node.subtype == GlueSubtype.baseline_skip_code:
        if context.last_was_hyphen_or_disc:
            context.last_was_hyphen_or_disc = False
            return None
        return GlueITNode(char=" ")
    elif node.subtype in (
        GlueSubtype.split_top_skip_code,
        GlueSubtype.top_skip_code,
    ):
        assert False, f"Unexpected glue in math {node}"
    return None


@_register_processor(type=KernNode)
def kern_node_to_tex(node: KernNode, context: TexContext) -> ITNode | None:
    # return TexCode(f'\\kern{node.width}', is_singular=True)
    return None


@_register_processor(type=MarginalKern)
def marginal_kern_to_tex(node: MarginalKern, context: TexContext) -> ITNode | None:
    # return TexCode(f"\\mkern{node.width}", ...)
    return None


@_register_processor(type=PenaltyNode)
def penalty_node_to_tex(node: PenaltyNode, context: TexContext) -> ITNode | None:
    return None


@_register_processor(type=UnsetNode)
def unset_node_to_tex(node: UnsetNode, context: TexContext) -> ITNode | None:
    return None


@_register_processor(type=StyleNode)
def style_node_to_tex(node: StyleNode, context: TexContext) -> ITNode | None:
    context.set_style(node.style)
    return None


@_register_processor(type=ChoiceNode)
def choice_node_to_tex(node: ChoiceNode, context: TexContext) -> ITNode | None:
    if context.style == StyleType.display:
        mlist = node.display_mlist
    elif context.style == StyleType.text:
        mlist = node.text_mlist
    elif context.style == StyleType.script:
        mlist = node.script_mlist
    elif context.style == StyleType.script_script:
        mlist = node.script_script_mlist
    else:
        raise NotImplementedError(context.style)
    return ConcatITNode(children=_to_tex_list(mlist, context))


def __inner_noad_field_to_tex(field: NoadField, context: TexContext) -> ITNode | None:
    if isinstance(field, (NoadFieldMathChar, NoadFieldMathTextChar)):
        return context.get_math_char_code(field.family, field.character)
    elif isinstance(field, NoadFieldSubBox):
        return _to_tex(field.box, context)
    elif isinstance(field, NoadFieldSubMList):
        return ConcatITNode(children=_to_tex_list(field.mlist, context))
    else:
        raise NotImplementedError(type(field))


def _noad_field_to_tex(field: NoadField | None, context: TexContext) -> ITNode | None:
    if field is None:
        return None
    replacement = context.get_math_code(field.addr)
    if replacement is SKIP_NODE:
        return None
    elif replacement is SKIP_EXCEPT_SUB_SUP:
        inner = __inner_noad_field_to_tex(field, context)
        print(f"XXX: Checking {inner!r}")
        if isinstance(inner, OrdITNode):
            inner.nucleus = None
            return inner
        return None
    elif replacement is not None:
        return replacement
    return __inner_noad_field_to_tex(field, context)


def ord_noad_to_tex(node: Noad, context: TexContext, force_ord_node: bool = False) -> ITNode | None:
    context.last_was_hyphen_or_disc = False
    if node.nucleus is not None:
        nucleus = _noad_field_to_tex(node.nucleus, context)
    else:
        nucleus = None
    if node.subscr is not None:
        subscr = _noad_field_to_tex(node.subscr, context)
    else:
        subscr = None
    if node.supscr is not None:
        # Special case: x'' -> x^{\prime\prime}, undo this case
        supscr = _noad_field_to_tex(node.supscr, context)
    else:
        supscr = None
    if subscr is not None or supscr is not None or force_ord_node:
        return OrdITNode(nucleus=nucleus, subscr=subscr, supscr=supscr)
    else:
        return nucleus


_register_processor(type=OrdNoad)(ord_noad_to_tex)
_register_processor(type=OpNoad)(ord_noad_to_tex)
_register_processor(type=BinNoad)(ord_noad_to_tex)
_register_processor(type=RelNoad)(ord_noad_to_tex)
_register_processor(type=OpenNoad)(ord_noad_to_tex)
_register_processor(type=CloseNoad)(ord_noad_to_tex)
_register_processor(type=PunctNoad)(ord_noad_to_tex)
_register_processor(type=InnerNoad)(ord_noad_to_tex)


@_register_processor(type=RadicalNoad)
def radical_noad_to_tex(node: RadicalNoad, context: TexContext) -> ITNode | None:
    nucleus = _noad_field_to_tex(node.nucleus, context)
    assert node.supscr is None
    assert node.subscr is None
    if node.left_delimiter is None:
        left_delimiter = None
    else:
        print(
            f"Checking {node.left_delimiter.small_fam}/{node.left_delimiter.small_char} and {node.left_delimiter.large_fam}/{node.left_delimiter.large_char} against {context.math_sqrt_signs}"
        )
        if context.get_math_sqrtsign(
            node.left_delimiter.small_fam, node.left_delimiter.small_char
        ) or context.get_math_sqrtsign(
            node.left_delimiter.large_fam, node.left_delimiter.large_char
        ):
            left_delimiter = MathCharCmdITNode(char_cmd="radical", format=None)
        else:
            left_delimiter = context.get_math_char_code(
                node.left_delimiter.small_fam, node.left_delimiter.small_char
            )
    return RadicalITNode(child=nucleus, left_delimiter=left_delimiter)


@_register_processor(type=FractionNoad)
def fraction_noad_to_tex(node: FractionNoad, context: TexContext) -> ITNode | None:
    numerator = _noad_field_to_tex(node.numerator, context)
    denominator = _noad_field_to_tex(node.denominator, context)
    if node.left_delimiter is not None:
        left_delimiter = context.get_math_char_code(
            node.left_delimiter.small_fam, node.left_delimiter.small_char
        )
    else:
        left_delimiter = None
    if node.right_delimiter is not None:
        right_delimiter = context.get_math_char_code(
            node.right_delimiter.small_fam, node.right_delimiter.small_char
        )
    else:
        right_delimiter = None
    return FractionITNode(
        thickness=None if isinstance(node.thickness, Default) else node.thickness,
        left_delimiter=left_delimiter,
        right_delimiter=right_delimiter,
        numerator=numerator,
        denominator=denominator,
    )


@_register_processor(type=UnderNoad)
def under_noad_to_tex(node: UnderNoad, context: TexContext) -> ITNode | None:
    code = ord_noad_to_tex(node, context)
    return UnderlineStylingITNode(child=code)


@_register_processor(type=OverNoad)
def over_noad_to_tex(node: OverNoad, context: TexContext) -> ITNode | None:
    code = ord_noad_to_tex(node, context)
    return OverlineStylingITNode(child=code)


@_register_processor(type=AccentNoad)
def accent_noad_to_tex(node: AccentNoad, context: TexContext) -> ITNode | None:
    code = ord_noad_to_tex(node, context, force_ord_node=True)
    # There is special handling for sub/sup: Accent only relates to the nucleus
    # I.e. \hat{x_1^2} -> \hat{x}_1^2
    assert isinstance(code, OrdITNode)
    accent_fn = context.get_math_accent(node.accent_family, node.accent_character)
    if accent_fn is not None:
        inner_code = AccentStylingITNode(function=f"\\{accent_fn}", child=code.nucleus)
        if code.subscr is not None or code.supscr is not None:
            # Wrap only the nucleus
            code.nucleus = inner_code
            return code
        # Otherwise there only is the nucleus
        return inner_code
    accent_char = context.get_math_char_code(node.accent_family, node.accent_character)
    # print(f"accent: {accent_char!r} from fam={node.accent_family}, char=0x{node.accent_character:02x}, inner={code!r}")
    inner_code = AccentStylingITNode(function="\\mathaccent", char=accent_char, child=code.nucleus)

    if code.subscr is not None or code.supscr is not None:
        code.nucleus = inner_code
        return code
    # There only is the nucleus
    return inner_code


@_register_processor(type=VCenterNoad)
def vcenter_noad_to_tex(node: VCenterNoad, context: TexContext) -> ITNode | None:
    return ord_noad_to_tex(node, context)


@_register_processor(type=LeftNoad)
def left_noad_to_tex(node: LeftNoad, context: TexContext) -> ITNode | None:
    if node.delimiter is None:
        delimiter = None
    else:
        delimiter = context.get_math_char_code(node.delimiter.small_fam, node.delimiter.small_char)
    return LeftITNode(delimiter=delimiter)


@_register_processor(type=RightNoad)
def right_noad_to_tex(node: RightNoad, context: TexContext) -> ITNode | None:
    if node.delimiter is None:
        delimiter = None
    else:
        delimiter = context.get_math_char_code(node.delimiter.small_fam, node.delimiter.small_char)
    if node.subtype == RightNoadSubtype.MIDDLE_NOAD:
        return MidITNode(delimiter=delimiter)
    else:
        return RightITNode(delimiter=delimiter)


def noad_to_tex(node: Node, context: TexContext) -> ITNode | None:
    return _to_tex(node, context)


def noads_to_tex(node: Iterable[Node], context: TexContext) -> ITNode | None:
    return ConcatITNode(children=_to_tex_list(node, context))


def _split_unknown(unknown_char: str) -> tuple[str, int]:
    sep = unknown_char.index("}{")
    assert unknown_char.endswith("}")
    font = unknown_char[len("\\unknown{") : sep]
    char = int(unknown_char[sep + 2 : -1], 16)
    return font, char
