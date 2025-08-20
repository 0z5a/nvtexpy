# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generator, Iterable, Literal, TypeVar

try:
    from nvtexpy.engine.debug import DEBUG_MATH_SIMPLIFY
except ImportError:
    DEBUG_MATH_SIMPLIFY = False


def size_to_pt(size: float) -> str:
    return f"{size:.5f}".rstrip("0.") + "pt"


@dataclass(frozen=True)
class TextFontInfo:
    _: dataclasses.KW_ONLY

    bold: bool
    italic_or_slanted: bool
    monospace: bool


DEFAULT_FONT = TextFontInfo(bold=False, italic_or_slanted=False, monospace=False)

T = TypeVar("T", covariant=True)


def assert_instance(obj: Any, cls: type[T] | tuple[type[T], ...]) -> T:
    assert isinstance(obj, cls), f"Expected {cls!r}, got {type(obj)!r}"
    return obj


def assert_none_or_instance(obj: Any, cls: type[T] | tuple[type[T], ...]) -> T | None:
    assert obj is None or isinstance(obj, cls), f"Expected {cls!r} or None, got {type(obj)!r}"
    return obj


def assert_not_none(obj: T | None) -> T:
    assert obj is not None, "Expected not None"
    return obj


@dataclass
class ITNode:
    _: dataclasses.KW_ONLY

    @abstractmethod
    def inner(self) -> Iterable["ITNode"]: ...

    @abstractmethod
    def is_empty(self) -> bool: ...

    def dbg_str(self, indent: str = "") -> str:
        raise NotImplementedError(f"Cannot convert {type(self)!r} to string")

    def __str__(self) -> str:
        return f"math(code={to_str(self)!r})"

    def __repr__(self) -> str:
        return self.dbg_str()


@dataclass
class ConcatITNode(ITNode):
    _: dataclasses.KW_ONLY

    children: list[ITNode]

    def inner(self) -> Iterable[ITNode]:
        return self.children

    def is_empty(self) -> bool:
        return all(child.is_empty() for child in self.children)

    def dbg_str(self, indent: str = "") -> str:
        if len(self.children) == 0:
            return "Concat: <empty>"
        return "Concat:" + "".join(
            f"\n{indent}  - {child.dbg_str(indent + '  ')}" for child in self.children
        )


@dataclass
class TextITNode(ITNode):
    _: dataclasses.KW_ONLY


@dataclass
class TextCharITNode(TextITNode):
    _: dataclasses.KW_ONLY

    char: str
    format: TextFontInfo | None

    def inner(self) -> Iterable["ITNode"]:
        return ()

    def is_empty(self) -> bool:
        return False

    def dbg_str(self, indent: str = "") -> str:
        fmt = []
        if self.format is None:
            fmt.append("<unformatted>")
        else:
            if self.format.bold:
                fmt.append("bf")
            if self.format.italic_or_slanted:
                fmt.append("it")
            if self.format.monospace:
                fmt.append("tt")
        fmt_str = ",".join(fmt)
        return f"TextChar: {self.char!r}" + (f" fmt={fmt_str}" if fmt_str else "")


@dataclass
class TextCharCmdITNode(TextITNode):
    _: dataclasses.KW_ONLY

    char_cmd: str
    format: TextFontInfo | None

    def inner(self) -> Iterable["ITNode"]:
        return ()

    def is_empty(self) -> bool:
        return False

    def dbg_str(self, indent: str = "") -> str:
        fmt = []
        if self.format is None:
            fmt.append("<unformatted>")
        else:
            if self.format.bold:
                fmt.append("bf")
            if self.format.italic_or_slanted:
                fmt.append("it")
            if self.format.monospace:
                fmt.append("tt")
        fmt_str = ", ".join(fmt)
        return f"TextCharCmd: {self.char_cmd!r}" + (f" fmt={fmt_str}" if fmt_str else "")


@dataclass
class TextUnknownCharITNode(TextITNode):
    _: dataclasses.KW_ONLY

    font: str
    char: int
    format: TextFontInfo | None

    def inner(self) -> Iterable["ITNode"]:
        return ()

    def is_empty(self) -> bool:
        return False

    def dbg_str(self, indent: str = "") -> str:
        fmt = []
        if self.format.bold:
            fmt.append("bf")
        if self.format.italic_or_slanted:
            fmt.append("it")
        if self.format.monospace:
            fmt.append("tt")
        fmt_str = ", ".join(fmt)
        return f"TextUnknownChar: \\unknown{{{self.font}}}{{{self.char:04X}}}" + (
            f" fmt={fmt_str}" if fmt_str else ""
        )


@dataclass
class GlueITNode(ITNode):
    _: dataclasses.KW_ONLY

    char: str

    def inner(self) -> Iterable["ITNode"]:
        return ()

    def is_empty(self) -> bool:
        return False

    def dbg_str(self, indent: str = "") -> str:
        return f"Glue: {self.char!r}"


@dataclass
class TextRuleITNode(TextITNode):
    _: dataclasses.KW_ONLY

    hrule: bool

    def inner(self) -> Iterable["ITNode"]:
        return ()

    def is_empty(self) -> bool:
        return False

    def dbg_str(self, indent: str = "") -> str:
        return f"TextRule: {'H' if self.hrule else 'V'}"


@dataclass
class InnerMathITNode(TextITNode):
    """Switch from text to math"""

    _: dataclasses.KW_ONLY

    child: ITNode

    def inner(self) -> Iterable[ITNode]:
        return (self.child,)

    def is_empty(self) -> bool:
        return self.child.is_empty()

    def dbg_str(self, indent: str = "") -> str:
        if self.child is None:
            return "InnerMath: <empty>"
        return f"InnerMath:\n{indent}  {self.child.dbg_str(indent + '  ')}"


@dataclass
class TextFormatITNode(TextITNode):
    """Will be inserted during simplification."""

    _: dataclasses.KW_ONLY

    format: str
    child: ITNode

    def is_empty(self) -> bool:
        return self.child.is_empty()

    def inner(self) -> Iterable[ITNode]:
        return (self.child,)

    def dbg_str(self, indent: str = "") -> str:
        return f"TextFormat: {self.format!r}\n{indent}  {self.child.dbg_str(indent + '  ')}"


@dataclass
class MathITNode(ITNode):
    _: dataclasses.KW_ONLY


@dataclass
class MathFunctionITNode(MathITNode):
    _: dataclasses.KW_ONLY

    function: str
    args: list[ITNode | str | None] = field(default_factory=list)
    args_prefer_braced: list[bool | Literal["Bracket"]] = field(default_factory=list)

    def inner(self) -> Iterable[ITNode]:
        for arg in self.args:
            if isinstance(arg, ITNode):
                yield arg

    def is_empty(self) -> bool:
        return False

    def dbg_str(self, indent: str = "") -> str:
        return f"MathFunction: {self.function!r}" + "".join(
            f"\n{indent}  - {(repr(arg) if isinstance(arg, str) else ('<empty>' if arg is None else arg.dbg_str(indent + '  ')))}"
            for arg in self.args
        )


@dataclass
class InnerTextITNode(MathITNode):
    """Switch from math to text"""

    _: dataclasses.KW_ONLY

    child: ITNode

    def inner(self) -> Iterable[ITNode]:
        return (self.child,)

    def is_empty(self) -> bool:
        return self.child.is_empty()

    def dbg_str(self, indent: str = "") -> str:
        if self.child is None:
            return "InnerText: <empty>"
        return f"InnerText:\n{indent}  {self.child.dbg_str(indent + '  ')}"


@dataclass
class MathCharITNode(MathITNode):
    _: dataclasses.KW_ONLY

    char: str
    format: str | None

    def inner(self) -> Iterable["ITNode"]:
        return ()

    def is_empty(self) -> bool:
        return False

    def dbg_str(self, indent: str = "") -> str:
        return f"MathChar: {self.char!r}" + (f" fmt={self.format!r}" if self.format else "")


@dataclass
class MathCharCmdITNode(MathITNode):
    _: dataclasses.KW_ONLY

    char_cmd: str
    format: str | None

    def inner(self) -> Iterable["ITNode"]:
        return ()

    def is_empty(self) -> bool:
        return False

    def dbg_str(self, indent: str = "") -> str:
        return f"MathCharCmd: {self.char_cmd!r}" + (f" fmt={self.format!r}" if self.format else "")


@dataclass
class MathUnknownCharITNode(MathITNode):
    _: dataclasses.KW_ONLY

    font: str
    char: int
    format: str | None

    def inner(self) -> Iterable["ITNode"]:
        return ()

    def is_empty(self) -> bool:
        return False

    def dbg_str(self, indent: str = "") -> str:
        return f"MathUnknownChar: \\unknown{{{self.font}}}{{{self.char:04X}}}"


@dataclass
class OrdITNode(MathITNode):
    _: dataclasses.KW_ONLY

    nucleus: ITNode | None
    subscr: ITNode | None
    supscr: ITNode | None

    def inner(self) -> Iterable[ITNode]:
        if self.nucleus:
            yield self.nucleus
        if self.subscr:
            yield self.subscr
        if self.supscr:
            yield self.supscr

    def is_empty(self) -> bool:
        return (
            (not self.nucleus or self.nucleus.is_empty())
            and (not self.subscr or self.subscr.is_empty())
            and (not self.supscr or self.supscr.is_empty())
        )

    def dbg_str(self, indent: str = "") -> str:
        return "Ord:" + "".join(
            f"\n{indent}  {key}:\n{indent}    {child.dbg_str(indent + '    ')}"
            for key, child in (
                ("Nucleus", self.nucleus),
                ("Subscr", self.subscr),
                ("Supscr", self.supscr),
            )
            if child is not None
        )


@dataclass
class StylingITNode(MathITNode):
    _: dataclasses.KW_ONLY

    child: ITNode | None

    def inner(self) -> Iterable[ITNode]:
        if self.child is not None:
            return (self.child,)
        return ()

    def is_empty(self) -> bool:
        return self.child is None or self.child.is_empty()


@dataclass
class OverlineStylingITNode(StylingITNode):
    _: dataclasses.KW_ONLY

    def dbg_str(self, indent: str = "") -> str:
        return "OverlineStyling:" + (
            " None" if self.child is None else f"\n{indent}  {self.child.dbg_str(indent + '  ')}"
        )


@dataclass
class UnderlineStylingITNode(StylingITNode):
    _: dataclasses.KW_ONLY

    def dbg_str(self, indent: str = "") -> str:
        return "UnderlineStyling:" + (
            " None" if self.child is None else f"\n{indent}  {self.child.dbg_str(indent + '  ')}"
        )


@dataclass
class AccentStylingITNode(StylingITNode):
    _: dataclasses.KW_ONLY

    function: str
    char: MathCharITNode | MathUnknownCharITNode | MathCharCmdITNode | None = None

    def dbg_str(self, indent: str = "") -> str:
        return f"AccentStyling: accent={self.function!r}" + (
            " None" if self.child is None else f"\n{indent}  {self.child.dbg_str(indent + '  ')}"
        )


@dataclass
class RadicalITNode(MathITNode):
    _: dataclasses.KW_ONLY

    child: ITNode | None
    left_delimiter: MathCharITNode | MathUnknownCharITNode | MathCharCmdITNode | None

    def inner(self) -> Iterable[ITNode]:
        if self.child is not None:
            yield self.child
        if self.left_delimiter is not None:
            yield self.left_delimiter

    def is_sqrt(self) -> bool:
        return (
            isinstance(self.left_delimiter, MathCharITNode)
            and self.left_delimiter.char == "√"
            or isinstance(self.left_delimiter, MathCharCmdITNode)
            and self.left_delimiter.char_cmd == "radical"
        )

    def is_empty(self) -> bool:
        return False

    def dbg_str(self, indent: str = "") -> str:
        return (
            "Radical: "
            + (
                "delim=None"
                if self.left_delimiter is None
                else "delim=" + self.left_delimiter.dbg_str(indent + "  ")
            )
            + (
                " None"
                if self.child is None
                else f"\n{indent}  {self.child.dbg_str(indent + '  ')}"
            )
        )


@dataclass
class FractionITNode(MathITNode):
    _: dataclasses.KW_ONLY

    # None -> Default
    thickness: float | None
    left_delimiter: MathCharITNode | MathUnknownCharITNode | MathCharCmdITNode | None
    right_delimiter: MathCharITNode | MathUnknownCharITNode | MathCharCmdITNode | None

    numerator: ITNode | None
    denominator: ITNode | None

    def inner(self) -> Iterable[ITNode]:
        if self.numerator is not None:
            yield self.numerator
        if self.denominator is not None:
            yield self.denominator
        if self.left_delimiter is not None:
            yield self.left_delimiter
        if self.right_delimiter is not None:
            yield self.right_delimiter

    def is_empty(self) -> bool:
        return False

    def dbg_str(self, indent: str = "") -> str:
        return (
            "Fraction:"
            + ("" if self.thickness is None else f" thickness={size_to_pt(self.thickness)}")
            + (
                ""
                if self.left_delimiter is None
                else " left=" + self.left_delimiter.dbg_str(indent + "  ")
            )
            + (
                ""
                if self.right_delimiter is None
                else " right=" + self.right_delimiter.dbg_str(indent + "  ")
            )
            + (
                " numerator=<empty>"
                if self.numerator is None
                else f"\n{indent}  Numerator:\n{indent}    {self.numerator.dbg_str(indent + '    ')}"
            )
            + (
                " denominator=<empty>"
                if self.denominator is None
                else f"\n{indent}  Denominator:\n{indent}    {self.denominator.dbg_str(indent + '    ')}"
            )
        )


@dataclass
class LeftMidRightITNode(MathITNode):
    _: dataclasses.KW_ONLY

    delimiter: MathCharITNode | MathUnknownCharITNode | MathCharCmdITNode | None

    def inner(self) -> Iterable[ITNode]:
        if self.delimiter is not None:
            yield self.delimiter

    def is_empty(self) -> bool:
        return False


@dataclass
class LeftITNode(LeftMidRightITNode):
    _: dataclasses.KW_ONLY

    def dbg_str(self, indent: str = "") -> str:
        return "Left: " + (
            "del=None" if self.delimiter is None else "del=" + self.delimiter.dbg_str(indent + "  ")
        )


@dataclass
class MidITNode(LeftMidRightITNode):
    _: dataclasses.KW_ONLY

    def dbg_str(self, indent: str = "") -> str:
        return "Mid: " + (
            "del=None" if self.delimiter is None else "del=" + self.delimiter.dbg_str(indent + "  ")
        )


@dataclass
class RightITNode(LeftMidRightITNode):
    _: dataclasses.KW_ONLY

    def dbg_str(self, indent: str = "") -> str:
        return "Right: " + (
            "del=None" if self.delimiter is None else "del=" + self.delimiter.dbg_str(indent + "  ")
        )


@dataclass
class TableCellITNode(ITNode):
    _: dataclasses.KW_ONLY

    child: ITNode | None
    colspan: int = 1
    rowspan: int = 1
    borders: tuple[bool, bool, bool, bool]

    def inner(self) -> Iterable[ITNode]:
        if self.child is not None:
            yield self.child

    def is_empty(self) -> bool:
        return self.child is None or self.child.is_empty()

    def dbg_str(self, indent: str = "") -> str:
        span = ""
        if self.colspan != 1:
            span += f" colspan={self.colspan}"
        if self.rowspan != 1:
            span += f" rowspan={self.rowspan}"
        return f"TableCell:{span}" + (
            " <empty>" if self.child is None else f"\n{indent}  {self.child.dbg_str(indent + '  ')}"
        )


@dataclass
class TableRowITNode(ITNode):
    _: dataclasses.KW_ONLY

    cells: list[TableCellITNode]
    borders: tuple[bool, bool]

    def inner(self) -> Iterable[ITNode]:
        return self.cells

    def is_empty(self) -> bool:
        return all(cell.is_empty() for cell in self.cells)

    def dbg_str(self, indent: str = "") -> str:
        return "TableRow:" + "".join(
            f"\n{indent}  - {cell.dbg_str(indent + '  ')}" for cell in self.cells
        )


@dataclass
class TableITNode(ITNode):
    _: dataclasses.KW_ONLY

    data: list[TableRowITNode]

    def inner(self) -> Iterable[ITNode]:
        return self.data

    def is_empty(self) -> bool:
        return all(row.is_empty() for row in self.data)

    def dbg_str(self, indent: str = "") -> str:
        return "Table:" + "".join(
            f"\n{indent}  - " + row.dbg_str(indent + "  ") for row in self.data
        )


@dataclass
class GraphicsITNode(ITNode):
    _: dataclasses.KW_ONLY

    def inner(self) -> Iterable[ITNode]:
        return ()

    def is_empty(self) -> bool:
        return False

    def dbg_str(self, indent: str = "") -> str:
        return "Graphics"


class ITNodeTransformer(ABC):
    """Helper class for transforming the tree. Will recursively visit elements."""

    def __call__(self, node: ITNode | None) -> ITNode | None:
        return self._node(node)

    def _node(self, node: ITNode | None) -> ITNode | None:
        if node is None:
            return None
        if isinstance(node, ConcatITNode):
            return self._concat(node)
        if isinstance(node, MathITNode):
            return self._math(node)
        if isinstance(node, TextITNode):
            return self._text(node)
        if isinstance(node, GlueITNode):
            return self._glue(node)
        if isinstance(node, TableITNode):
            return self._table(node)
        if isinstance(node, GraphicsITNode):
            return self._graphics(node)
        raise NotImplementedError(type(node))

    @abstractmethod
    def _concat(self, node: ConcatITNode) -> ITNode | None: ...

    def _math(self, node: MathITNode) -> ITNode | None:
        if isinstance(node, MathCharITNode):
            return self._math_char(node)
        if isinstance(node, MathUnknownCharITNode):
            return self._math_unknown_char(node)
        if isinstance(node, MathCharCmdITNode):
            return self._math_char_cmd(node)
        if isinstance(node, MathFunctionITNode):
            return self._math_function(node)
        if isinstance(node, OrdITNode):
            return self._ord(node)
        if isinstance(node, RadicalITNode):
            return self._radical(node)
        if isinstance(node, FractionITNode):
            return self._fraction(node)
        if isinstance(node, StylingITNode):
            return self._styling(node)
        if isinstance(node, LeftITNode):
            return self._left(node)
        if isinstance(node, MidITNode):
            return self._mid(node)
        if isinstance(node, RightITNode):
            return self._right(node)
        if isinstance(node, InnerTextITNode):
            return self._inner_text(node)
        raise NotImplementedError(type(node))

    def _text(self, node: TextITNode) -> ITNode | None:
        if isinstance(node, TextCharITNode):
            return self._text_char(node)
        if isinstance(node, TextCharCmdITNode):
            return self._text_char_cmd(node)
        if isinstance(node, TextUnknownCharITNode):
            return self._text_unknown_char(node)
        if isinstance(node, TextRuleITNode):
            return self._text_rule(node)
        if isinstance(node, InnerMathITNode):
            return self._inner_math(node)
        if isinstance(node, TextFormatITNode):
            return self._text_format(node)
        raise NotImplementedError(type(node))

    @abstractmethod
    def _glue(self, node: GlueITNode) -> ITNode | None: ...

    @abstractmethod
    def _table_cell(self, node: TableCellITNode) -> TableCellITNode: ...

    @abstractmethod
    def _table_row(self, node: TableRowITNode) -> TableRowITNode | None: ...

    @abstractmethod
    def _table(self, node: TableITNode) -> ITNode | None: ...

    @abstractmethod
    def _graphics(self, node: GraphicsITNode) -> ITNode | None: ...

    @abstractmethod
    def _math_char(self, node: MathCharITNode) -> ITNode | None: ...

    @abstractmethod
    def _math_unknown_char(self, node: MathUnknownCharITNode) -> ITNode | None: ...

    @abstractmethod
    def _math_char_cmd(self, node: MathCharCmdITNode) -> ITNode | None: ...

    @abstractmethod
    def _math_function(self, node: MathFunctionITNode) -> ITNode | None: ...

    @abstractmethod
    def _ord(self, node: OrdITNode) -> ITNode | None: ...

    @abstractmethod
    def _radical(self, node: RadicalITNode) -> ITNode | None: ...

    @abstractmethod
    def _fraction(self, node: FractionITNode) -> ITNode | None: ...

    def _styling(self, node: StylingITNode) -> ITNode | None:
        if isinstance(node, OverlineStylingITNode):
            return self._overline_styling(node)
        if isinstance(node, UnderlineStylingITNode):
            return self._underline_styling(node)
        if isinstance(node, AccentStylingITNode):
            return self._accent_styling(node)
        raise NotImplementedError(type(node))

    @abstractmethod
    def _overline_styling(self, node: OverlineStylingITNode) -> ITNode | None: ...

    @abstractmethod
    def _underline_styling(self, node: UnderlineStylingITNode) -> ITNode | None: ...

    @abstractmethod
    def _accent_styling(self, node: AccentStylingITNode) -> ITNode | None: ...

    @abstractmethod
    def _left(self, node: LeftITNode) -> ITNode | None: ...

    @abstractmethod
    def _mid(self, node: MidITNode) -> ITNode | None: ...

    @abstractmethod
    def _right(self, node: RightITNode) -> ITNode | None: ...

    @abstractmethod
    def _inner_text(self, node: InnerTextITNode) -> ITNode | None: ...

    @abstractmethod
    def _text_char(self, node: TextCharITNode) -> ITNode | None: ...

    @abstractmethod
    def _text_char_cmd(self, node: TextCharCmdITNode) -> ITNode | None: ...

    @abstractmethod
    def _text_unknown_char(self, node: TextUnknownCharITNode) -> ITNode | None: ...

    @abstractmethod
    def _text_rule(self, node: TextRuleITNode) -> ITNode | None: ...

    @abstractmethod
    def _inner_math(self, node: InnerMathITNode) -> ITNode | None: ...

    @abstractmethod
    def _text_format(self, node: TextFormatITNode) -> ITNode | None: ...


class CopyITNodeTransformer(ITNodeTransformer):
    """Copies the tree."""

    def _concat(self, node: ConcatITNode) -> ITNode | None:
        new_children = []
        for child in node.children:
            new_child = self._node(child)
            if new_child is not None:
                new_children.append(new_child)
        if len(new_children) == 0:
            return None
        if len(new_children) == 1:
            return new_children[0]
        return ConcatITNode(children=new_children)

    def _glue(self, node: GlueITNode) -> ITNode | None:
        return GlueITNode(char=node.char)

    def _math_char(self, node: MathCharITNode) -> ITNode | None:
        return MathCharITNode(char=node.char, format=node.format)

    def _math_unknown_char(self, node: MathUnknownCharITNode) -> ITNode | None:
        return MathUnknownCharITNode(font=node.font, char=node.char, format=node.format)

    def _math_char_cmd(self, node: MathCharCmdITNode) -> ITNode | None:
        return MathCharCmdITNode(char_cmd=node.char_cmd, format=node.format)

    def _math_function(self, node: MathFunctionITNode) -> ITNode | None:
        new_args = [
            arg if isinstance(arg, str) or arg is None else self._node(arg) for arg in node.args
        ]
        return MathFunctionITNode(
            function=node.function, args=new_args, args_prefer_braced=node.args_prefer_braced
        )

    def _ord(self, node: OrdITNode) -> ITNode | None:
        nucleus = self._node(node.nucleus)
        subscr = self._node(node.subscr)
        supscr = self._node(node.supscr)
        if subscr is None and supscr is None:
            return nucleus
        return OrdITNode(nucleus=nucleus, subscr=subscr, supscr=supscr)

    def _radical(self, node: RadicalITNode) -> ITNode | None:
        child = self._node(node.child)
        left_delimiter = assert_none_or_instance(
            self._node(node.left_delimiter),
            (MathCharITNode, MathUnknownCharITNode, MathCharCmdITNode),
        )
        if left_delimiter is None:
            return child
        return RadicalITNode(child=child, left_delimiter=left_delimiter)

    def _fraction(self, node: FractionITNode) -> ITNode | None:
        numerator = self._node(node.numerator)
        denominator = self._node(node.denominator)
        left_delimiter = self._node(node.left_delimiter)
        right_delimiter = self._node(node.right_delimiter)
        assert left_delimiter is None or isinstance(
            left_delimiter, (MathCharITNode, MathUnknownCharITNode, MathCharCmdITNode)
        )
        assert right_delimiter is None or isinstance(
            right_delimiter, (MathCharITNode, MathUnknownCharITNode, MathCharCmdITNode)
        )
        return FractionITNode(
            thickness=node.thickness,
            left_delimiter=left_delimiter,
            right_delimiter=right_delimiter,
            numerator=numerator,
            denominator=denominator,
        )

    def _overline_styling(self, node: OverlineStylingITNode) -> ITNode | None:
        child = self._node(node.child)
        if child is None:
            return None
        return OverlineStylingITNode(child=child)

    def _underline_styling(self, node: UnderlineStylingITNode) -> ITNode | None:
        child = self._node(node.child)
        if child is None:
            return None
        return UnderlineStylingITNode(child=child)

    def _accent_styling(self, node: AccentStylingITNode) -> ITNode | None:
        child = self._node(node.child)
        if node.char is not None:
            new_char = assert_none_or_instance(
                self._node(node.char), (MathCharITNode, MathUnknownCharITNode, MathCharCmdITNode)
            )
            if new_char is None:
                return child
        else:
            new_char = None
        return AccentStylingITNode(function=node.function, char=new_char, child=child)

    def _left(self, node: LeftITNode) -> ITNode | None:
        return LeftITNode(
            delimiter=assert_none_or_instance(
                self._node(node.delimiter),
                (MathCharITNode, MathUnknownCharITNode, MathCharCmdITNode),
            )
        )

    def _mid(self, node: MidITNode) -> ITNode | None:
        delimiter = assert_none_or_instance(
            self._node(node.delimiter), (MathCharITNode, MathUnknownCharITNode, MathCharCmdITNode)
        )
        if delimiter is None:
            return None
        return MidITNode(delimiter=delimiter)

    def _right(self, node: RightITNode) -> ITNode | None:
        return RightITNode(
            delimiter=assert_none_or_instance(
                self._node(node.delimiter),
                (MathCharITNode, MathUnknownCharITNode, MathCharCmdITNode),
            )
        )

    def _table_cell(self, node: TableCellITNode) -> TableCellITNode:
        return TableCellITNode(
            child=self._node(node.child),
            colspan=node.colspan,
            rowspan=node.rowspan,
            borders=node.borders,
        )

    def _table_row(self, node: TableRowITNode) -> TableRowITNode | None:
        cells: list[TableCellITNode] = [self._table_cell(cell) for cell in node.cells]
        if all(cell.child is None for cell in cells):
            return None
        return TableRowITNode(cells=cells, borders=node.borders)

    def _table(self, node: TableITNode) -> ITNode | None:
        new_rows = [self._table_row(row) for row in node.data]
        rows = [row for row in new_rows if row is not None]
        if len(rows) == 0 or len(rows[0].cells) == 0:
            return None
        if len(rows) == 1:
            return ConcatITNode(children=[c.child for c in rows[0].cells if c.child is not None])
        return TableITNode(data=rows)

    def _graphics(self, node: GraphicsITNode) -> ITNode | None:
        return GraphicsITNode()

    def _inner_text(self, node: InnerTextITNode) -> ITNode | None:
        child = self._node(node.child)
        if child is None:
            return None
        return InnerTextITNode(child=child)

    def _text_char(self, node: TextCharITNode) -> ITNode | None:
        return TextCharITNode(char=node.char, format=node.format)

    def _text_char_cmd(self, node: TextCharCmdITNode) -> ITNode | None:
        return TextCharCmdITNode(char_cmd=node.char_cmd, format=node.format)

    def _text_unknown_char(self, node: TextUnknownCharITNode) -> ITNode | None:
        return TextUnknownCharITNode(font=node.font, char=node.char, format=node.format)

    def _text_rule(self, node: TextRuleITNode) -> ITNode | None:
        return TextRuleITNode(hrule=node.hrule)

    def _inner_math(self, node: InnerMathITNode) -> ITNode | None:
        child = self._node(node.child)
        if child is None:
            return None
        return InnerMathITNode(child=child)

    def _text_format(self, node: TextFormatITNode) -> ITNode | None:
        inner = self._node(node.child)
        if inner is None:
            return None
        return TextFormatITNode(format=node.format, child=inner)


class InplaceITNodeTransformer(ITNodeTransformer):
    """Helper class for transforming the tree. Will recursively visit elements. Does avoid allocations
    of new nodes if possible by transforming the nodes in-place."""

    def _concat(self, node: ConcatITNode) -> ITNode | None:
        i = 0
        while i < len(node.children):
            child = node.children[i]
            child = self._node(child)
            if child is None:
                # Node was empty
                del node.children[i]
                continue
            if isinstance(child, ConcatITNode):
                node.children[i : i + 1] = child.children
            else:
                node.children[i] = child
            i += 1
        if len(node.children) == 0:
            return None
        if len(node.children) == 1:
            return node.children[0]
        return node

    def _glue(self, node: GlueITNode) -> ITNode | None:
        return node

    def _math_char(self, node: MathCharITNode) -> ITNode | None:
        return node

    def _math_unknown_char(self, node: MathUnknownCharITNode) -> ITNode | None:
        return node

    def _math_char_cmd(self, node: MathCharCmdITNode) -> ITNode | None:
        return node

    def _math_function(self, node: MathFunctionITNode) -> ITNode | None:
        i = 0
        while i < len(node.args):
            arg = node.args[i]
            if isinstance(arg, ITNode):
                node.args[i] = self._node(arg)
            i += 1
        return node

    def _ord(self, node: OrdITNode) -> ITNode | None:
        node.nucleus = self._node(node.nucleus)
        node.subscr = self._node(node.subscr)
        node.supscr = self._node(node.supscr)
        if node.subscr is None and node.supscr is None:
            return node.nucleus
        return node

    def _radical(self, node: RadicalITNode) -> ITNode | None:
        node.child = self._node(node.child)
        node.left_delimiter = assert_none_or_instance(
            self._node(node.left_delimiter),
            (MathCharITNode, MathUnknownCharITNode, MathCharCmdITNode),
        )
        if node.left_delimiter is None:
            return node.child
        return node

    def _fraction(self, node: FractionITNode) -> ITNode | None:
        node.numerator = self._node(node.numerator)
        node.denominator = self._node(node.denominator)
        node.left_delimiter = assert_none_or_instance(
            self._node(node.left_delimiter),
            (MathCharITNode, MathUnknownCharITNode, MathCharCmdITNode),
        )
        node.right_delimiter = assert_none_or_instance(
            self._node(node.right_delimiter),
            (MathCharITNode, MathUnknownCharITNode, MathCharCmdITNode),
        )
        return node

    def _overline_styling(self, node: OverlineStylingITNode) -> ITNode | None:
        node.child = self._node(node.child)
        if node.child is None:
            return None
        return node

    def _underline_styling(self, node: UnderlineStylingITNode) -> ITNode | None:
        node.child = self._node(node.child)
        if node.child is None:
            return None
        return node

    def _accent_styling(self, node: AccentStylingITNode) -> ITNode | None:
        node.child = self._node(node.child)
        if node.char is not None:
            node.char = assert_none_or_instance(
                self._node(node.char), (MathCharITNode, MathUnknownCharITNode, MathCharCmdITNode)
            )
            if node.char is None:
                return node.child
        return node

    def _left(self, node: LeftITNode) -> ITNode | None:
        node.delimiter = assert_none_or_instance(
            self._node(node.delimiter), (MathCharITNode, MathUnknownCharITNode, MathCharCmdITNode)
        )
        return node

    def _mid(self, node: MidITNode) -> ITNode | None:
        node.delimiter = assert_none_or_instance(
            self._node(node.delimiter), (MathCharITNode, MathUnknownCharITNode, MathCharCmdITNode)
        )
        if node.delimiter is None:
            return None
        return node

    def _right(self, node: RightITNode) -> ITNode | None:
        node.delimiter = assert_none_or_instance(
            self._node(node.delimiter), (MathCharITNode, MathUnknownCharITNode, MathCharCmdITNode)
        )
        return node

    def _table_cell(self, node: TableCellITNode) -> TableCellITNode:
        node.child = self._node(node.child)
        return node

    def _table_row(self, node: TableRowITNode) -> TableRowITNode | None:
        c = 0
        while c < len(node.cells):
            cell = node.cells[c]
            node.cells[c] = self._table_cell(cell)
            c += 1
        return node

    def _table(self, node: TableITNode) -> ITNode | None:
        r = 0
        while r < len(node.data):
            row = node.data[r]
            row = self._table_row(row)
            if row is None:
                del node.data[r]
                continue
            node.data[r] = row
            r += 1
        if len(node.data) == 0 or len(node.data[0].cells) == 0:
            return None
        if len(node.data) == 1:
            return ConcatITNode(children=[c.child for c in node.data[0].cells if c is not None])
        return node

    def _graphics(self, node: GraphicsITNode) -> ITNode | None:
        return node

    def _inner_text(self, node: InnerTextITNode) -> ITNode | None:
        new_child = self._node(node.child)
        if new_child is None:
            return None
        node.child = new_child
        return node

    def _text_char(self, node: TextCharITNode) -> ITNode | None:
        return node

    def _text_char_cmd(self, node: TextCharCmdITNode) -> ITNode | None:
        return node

    def _text_unknown_char(self, node: TextUnknownCharITNode) -> ITNode | None:
        return node

    def _text_rule(self, node: TextRuleITNode) -> ITNode | None:
        return node

    def _inner_math(self, node: InnerMathITNode) -> ITNode | None:
        new_child = self._node(node.child)
        if new_child is None:
            return None
        node.child = new_child
        return node

    def _text_format(self, node: TextFormatITNode) -> ITNode | None:
        new_child = self._node(node.child)
        if new_child is None:
            return None
        node.child = new_child
        return node


class _SimplifyFlattenLists(InplaceITNodeTransformer):
    def _concat(self, node: ConcatITNode) -> ITNode | None:
        i = 0
        while i < len(node.children):
            child = node.children[i]
            child = self._node(child)
            if child is None:
                # Node was empty
                del node.children[i]
                continue
            if isinstance(child, ConcatITNode):
                node.children[i : i + 1] = child.children
            else:
                node.children[i] = child
            i += 1
        if len(node.children) == 1:
            return node.children[0]
        elif len(node.children) == 0:
            return None
        return node


class _MathToFunctions(InplaceITNodeTransformer):
    """Converts specific nodes to function nodes to simplify further processing."""

    # \\accent^{o}{a},+ -> ['\\accent', '^', 'o', 'a', ',', '+']
    # \\{ -> ['\\{']
    re_split_simple_args = re.compile(
        r"(\{[^\{\}]*\})|(\\[a-zA-Z0-9]+)|(\\[^a-zA-z0-9])|([^0-9a-zA-Z\\])"
    )

    def _simple_fn_to_function(self, call: str) -> MathFunctionITNode:
        if "{" in call:
            args_details = self.re_split_simple_args.findall(call)
            name = args_details[0][1] or args_details[0][2]
            assert name, f"Cannot find function name in {call!r}"
            assert len(call) == sum(
                len(arg[0] or arg[1] or arg[2] or arg[3]) for arg in args_details
            ), f"Cannot match all arguments in {call!r}: {args_details!r}"
            args_braced = [bool(arg[0]) for arg in args_details[1:]]
            args = [
                arg[0][1:-1] if arg[0] else (arg[1] or arg[2] or arg[3]) for arg in args_details[1:]
            ]
            return MathFunctionITNode(function=name, args=args, args_prefer_braced=args_braced)
        return MathFunctionITNode(function=call)

    def _graphics(self, node: GraphicsITNode) -> ITNode | None:
        return MathFunctionITNode(
            function="\\includegraphics", args=[None], args_prefer_braced=[True]
        )

    def _text_rule(self, node: TextRuleITNode) -> ITNode | None:
        # return MathFunctionITNode(function="\\hline" if node.hrule else "\\vline")
        # Remove for now.
        return None

    def _math_unknown_char(self, node: MathUnknownCharITNode) -> ITNode | None:
        return MathFunctionITNode(
            function="\\unknown",
            args=[node.font, f"{node.char:04X}"],
            args_prefer_braced=[True, True],
        )

    def _math_char_cmd(self, node: MathCharCmdITNode) -> ITNode | None:
        return self._simple_fn_to_function(node.char_cmd)

    def _text_unknown_char(self, node: TextUnknownCharITNode) -> ITNode | None:
        return MathFunctionITNode(
            function="\\unknown",
            args=[node.font, f"{node.char:04X}"],
            args_prefer_braced=[True, True],
        )

    def _text_char_cmd(self, node: TextCharCmdITNode) -> ITNode | None:
        return self._simple_fn_to_function(node.char_cmd)

    def _inner_text(self, node: InnerTextITNode) -> ITNode | None:
        return MathFunctionITNode(
            function="\\text", args=[self._node(node.child)], args_prefer_braced=[True]
        )

    def _overline_styling(self, node: OverlineStylingITNode) -> ITNode | None:
        return MathFunctionITNode(
            function="\\overline", args=[self._node(node.child)], args_prefer_braced=[True]
        )

    def _underline_styling(self, node: UnderlineStylingITNode) -> ITNode | None:
        return MathFunctionITNode(
            function="\\underline", args=[self._node(node.child)], args_prefer_braced=[True]
        )

    def _accent_styling(self, node: AccentStylingITNode) -> ITNode | None:
        inner = self._node(node.child)
        if node.char is not None:
            accent_char = self._node(node.char)
            assert accent_char is not None
            return MathFunctionITNode(
                function=node.function, args=[accent_char, inner], args_prefer_braced=[False, True]
            )
        return MathFunctionITNode(function=node.function, args=[inner], args_prefer_braced=[True])

    def _radical(self, node: RadicalITNode) -> ITNode | None:
        if node.is_sqrt():
            return MathFunctionITNode(
                function="\\sqrt", args=[self._node(node.child)], args_prefer_braced=[True]
            )
        return MathFunctionITNode(
            function="\\radical",
            args=[self._node(node.left_delimiter), self._node(node.child)],
            args_prefer_braced=[False, True],
        )

    def _fraction(self, node: FractionITNode) -> ITNode | None:
        numerator = self._node(node.numerator)
        denominator = self._node(node.denominator)
        left_delimiter = self._node(node.left_delimiter)
        right_delimiter = self._node(node.right_delimiter)
        if node.thickness is None and left_delimiter is None and right_delimiter is None:
            return MathFunctionITNode(
                function="\\frac",
                args=[numerator, denominator],
                args_prefer_braced=[True, True],
            )
        if node.thickness is None:
            thickness = "1pt"
        else:
            thickness = size_to_pt(node.thickness)
        return MathFunctionITNode(
            function="\\genfrac",
            args=[
                left_delimiter,
                right_delimiter,
                thickness,
                "2",
                numerator,
                denominator,
            ],
            args_prefer_braced=[False, False, True, True, True, True],
        )

    def _left(self, node: LeftITNode) -> ITNode | None:
        return MathFunctionITNode(
            function="\\left", args=[self._node(node.delimiter)], args_prefer_braced=[False]
        )

    def _mid(self, node: MidITNode) -> ITNode | None:
        return MathFunctionITNode(
            function="\\middle", args=[self._node(node.delimiter)], args_prefer_braced=[False]
        )

    def _right(self, node: RightITNode) -> ITNode | None:
        return MathFunctionITNode(
            function="\\right", args=[self._node(node.delimiter)], args_prefer_braced=[False]
        )


class InplaceMathstackITNodeTransformer(InplaceITNodeTransformer):
    _in_math: bool

    def __init__(self, in_math: bool = False) -> None:
        super().__init__()
        self._in_math = in_math

    def _inner_text(self, node: InnerTextITNode) -> ITNode | None:
        prev_in_math = self._in_math
        self._in_math = False
        new_child = self._node(node.child)
        if new_child is None:
            return None
        node.child = new_child
        self._in_math = prev_in_math
        return node

    def _inner_math(self, node: InnerMathITNode) -> ITNode | None:
        prev_in_math = self._in_math
        self._in_math = True
        new_child = self._node(node.child)
        if new_child is None:
            return None
        node.child = new_child
        self._in_math = prev_in_math
        return node


class _CombineMathTextModes(InplaceITNodeTransformer):
    """
    Depending on `outer_math` this function does the following:
    * Combines text nodes into a text wrapper node.
    * Combines math nodes into a math wrapper node.
    """

    _in_math: bool | None

    def __init__(self, outer_math: bool) -> None:
        super().__init__()
        self._in_math = outer_math

    def _concat(self, node: ConcatITNode) -> ITNode | None:
        is_outer_math = self._in_math
        new_children = []
        collect_children = []
        collect_is_math = is_outer_math

        def _flush():
            if len(collect_children) > 0:
                if collect_is_math == is_outer_math:
                    new_children.extend(collect_children)
                elif collect_is_math is True:
                    new_children.append(
                        InnerMathITNode(child=ConcatITNode(children=collect_children))
                    )
                elif collect_is_math is False:
                    new_children.append(
                        InnerTextITNode(child=ConcatITNode(children=collect_children))
                    )

        # Top-level objects: MathITNode, TextITNode, (ConcatITNode), GlueITNode, TableITNode
        # Concat should be flattened, thus this cannot be a child, thus only Glue and Table are possible.
        for child in node.children:
            if isinstance(child, MathITNode):
                is_math = True
            elif isinstance(child, TextITNode):
                is_math = False
            elif isinstance(child, (GlueITNode, GraphicsITNode, TableITNode)):
                # Just keep the math state
                is_math = collect_is_math
            else:
                assert False, f"Unexpected node type {type(child)}"

            self._in_math = is_math
            child = self._node(child)
            self._in_math = is_outer_math
            if child is None:
                continue
            if collect_is_math != is_math:
                # Flush the node
                _flush()
                if len(collect_children):
                    # Create a sub node wrapping the mode
                    collect_children = []
                collect_is_math = is_math
            collect_children.append(child)
        _flush()
        if len(new_children) == 0:
            return None
        if len(new_children) == 1:
            return new_children[0]
        return ConcatITNode(children=new_children)

    def _inner_math(self, node: InnerMathITNode) -> ITNode | None:
        if self._in_math:
            return node.child
        return node

    def _inner_text(self, node: InnerTextITNode) -> ITNode | None:
        if not self._in_math:
            return node.child
        return node

    def _math(self, node: MathITNode) -> ITNode | None:
        in_math = self._in_math
        self._in_math = True
        new_node = super()._math(node)
        self._in_math = in_math
        if isinstance(new_node, MathITNode) and not in_math:
            new_node = InnerMathITNode(child=new_node)
        return new_node

    def _text(self, node: TextITNode) -> ITNode | None:
        in_math = self._in_math
        self._in_math = False
        new_node = super()._text(node)
        self._in_math = in_math
        if isinstance(new_node, TextITNode) and in_math:
            new_node = InnerTextITNode(child=new_node)
        return new_node


class _CombineSpecialSymbols(InplaceMathstackITNodeTransformer):
    """
    Combines special symbols into a single node. For example three dots will be \\dots.
    """

    # Key is the first detection character
    # Value is a tuple of the expected symbols, and the replacement command.
    CHAR_SEQ_REPLACE = {
        ".": [((".", ".", "."), "\\dots")],
    }
    SYMBOL_SEQ_REPLACE = {
        "\\cdot": [
            (
                (
                    "\\cdot",
                    "\\cdot",
                    "\\cdot",
                ),
                "\\cdots",
            )
        ],
        "\\int": [
            (("\\int", "\\int", "\\int", "\\int"), "\\iiiint"),
            (
                (
                    "\\int",
                    "\\int",
                    "\\int",
                ),
                "\\iiint",
            ),
            (
                (
                    "\\int",
                    "\\int",
                ),
                "\\iint",
            ),
        ],
    }

    def _concat(self, node: ConcatITNode) -> ITNode | None:
        node = super()._concat(node)
        if isinstance(node, ConcatITNode) and self._in_math:
            idx = 0
            while idx < len(node.children):
                subnode = self._node(node.children[idx])
                if isinstance(subnode, MathCharITNode):
                    if subnode.char in self.CHAR_SEQ_REPLACE:
                        for symbols, new_cmd in self.CHAR_SEQ_REPLACE[subnode.char]:
                            if idx + len(symbols) <= len(node.children) and all(
                                isinstance(cmp_symbol, MathCharITNode)
                                and cmp_symbol.char == src_symbol
                                and cmp_symbol.format == subnode.format
                                for src_symbol, cmp_symbol in zip(
                                    symbols, node.children[idx : idx + len(symbols)]
                                )
                            ):
                                node.children[idx : idx + len(symbols)] = [
                                    MathCharCmdITNode(char_cmd=new_cmd, format=subnode.format)
                                ]
                elif isinstance(subnode, MathCharCmdITNode):
                    if subnode.char_cmd in self.SYMBOL_SEQ_REPLACE:
                        for symbols, new_cmd in self.SYMBOL_SEQ_REPLACE[subnode.char_cmd]:
                            # print(f"XX: Checking {new_cmd!r}: {symbols!r} VS {node.children[idx : idx + len(symbols)]!r}")
                            if idx + len(symbols) <= len(node.children) and all(
                                isinstance(cmp_symbol, MathCharCmdITNode)
                                and cmp_symbol.char_cmd == src_symbol
                                and cmp_symbol.format == subnode.format
                                for src_symbol, cmp_symbol in zip(
                                    symbols, node.children[idx : idx + len(symbols)]
                                )
                            ):
                                node.children[idx : idx + len(symbols)] = [
                                    MathCharCmdITNode(char_cmd=new_cmd, format=subnode.format)
                                ]
                elif isinstance(subnode, LeftITNode):
                    # Match: \left<char>\right. -> \big
                    if idx + 1 < len(node.children):
                        right = node.children[idx + 1]
                        if isinstance(right, RightITNode) and right.delimiter is None:
                            node.children[idx : idx + 2] = [
                                MathFunctionITNode(
                                    function="\\big",
                                    args=[subnode.delimiter],
                                    args_prefer_braced=[False],
                                )
                            ]
                idx += 1
            return node
        return node


class _SimplifyCleanGlues(InplaceMathstackITNodeTransformer):
    """
    Remove adjacent glues, and glues in math context.
    """

    def _concat(self, node: ConcatITNode) -> ITNode | None:
        i = 0
        last_glue = False
        while i < len(node.children):
            child = node.children[i]
            child = self._node(child)
            if child is None:
                del node.children[i]
                continue
            if isinstance(child, GlueITNode):
                if self._in_math:
                    del node.children[i]
                    continue
                elif last_glue:
                    del node.children[i]
                    continue
                last_glue = True
            else:
                last_glue = False
            i += 1
        while len(node.children) > 1 and isinstance(node.children[0], GlueITNode):
            del node.children[0]
        while len(node.children) > 1 and isinstance(node.children[-1], GlueITNode):
            del node.children[-1]
        return node

    def _text_rule(self, node: TextRuleITNode) -> ITNode | None:
        # Purge all rules
        return None


class _SimplifyUnknown(InplaceMathstackITNodeTransformer):
    """
    Convert \\unknown{...}{...} -> \\unknown
    """

    def _concat(self, node: ConcatITNode) -> ITNode | None:
        i = 0
        last_math_unknown = None
        last_text_unknown = None
        while i < len(node.children):
            child = node.children[i]
            if isinstance(child, MathFunctionITNode):
                child = self._math_function(child)
            if isinstance(child, MathUnknownCharITNode):
                last_text_unknown = None
                if last_math_unknown is None:
                    last_math_unknown = node.children[i] = self._math_unknown_char(child)
                else:
                    del node.children[i]
                    continue
            elif isinstance(child, TextUnknownCharITNode):
                last_math_unknown = None
                if last_text_unknown is None:
                    last_text_unknown = node.children[i] = self._text_unknown_char(child)
                else:
                    del node.children[i]
                    continue
            else:
                new_child = self._node(child)
                if new_child is None:
                    del node.children[i]
                    continue
                else:
                    node.children[i] = new_child
                last_text_unknown = None
                last_math_unknown = None
            i += 1
        return node

    def _math_function(self, node: MathFunctionITNode) -> ITNode | None:
        if node.function == "\\includegraphics":
            return MathUnknownCharITNode(font="includegraphics", char=0, format=None)
        return node

    def _math_unknown_char(self, node: MathUnknownCharITNode) -> ITNode | None:
        return MathCharCmdITNode(char_cmd="\\unknown", format=node.format)

    def _text_unknown_char(self, node: TextUnknownCharITNode) -> ITNode | None:
        return TextCharCmdITNode(char_cmd="\\unknown", format=node.format)


class _FixPrime(InplaceITNodeTransformer):
    """
    Convert ^{\\prime} -> '
    """

    def _ord(self, node: OrdITNode) -> ITNode | None:
        if node.supscr is not None:
            if isinstance(node.supscr, ConcatITNode):
                primes = []
                while (
                    len(node.supscr.children) > 0
                    and isinstance(node.supscr.children[0], MathCharCmdITNode)
                    and node.supscr.children[0].char_cmd == "\\prime"
                ):
                    primes.append(MathCharITNode(char="'", format=node.supscr.children[0].format))
                    del node.supscr.children[0]
                if len(primes) > 0:
                    if len(node.supscr.children) == 0:
                        node.supscr = None
                    return self._node(
                        ConcatITNode(
                            children=[
                                *([node.nucleus] if node.nucleus is not None else []),
                                *primes[:-1],
                                OrdITNode(
                                    nucleus=primes[-1], supscr=node.supscr, subscr=node.subscr
                                ),
                            ]
                        )
                    )
            elif isinstance(node.supscr, MathCharCmdITNode) and node.supscr.char_cmd == "\\prime":
                return self._node(
                    ConcatITNode(
                        children=[
                            *([node.nucleus] if node.nucleus is not None else []),
                            OrdITNode(
                                nucleus=MathCharITNode(char="'", format=node.supscr.format),
                                supscr=None,
                                subscr=node.subscr,
                            ),
                        ]
                    )
                )
        return node


class _CleanSymbolFormats(InplaceITNodeTransformer):
    FORMATTING_SYMBOLS = re.compile("[a-zA-Z0-9]")

    def _math_char_cmd(self, node: MathCharCmdITNode) -> ITNode | None:
        if node.format != "boldsymbol":
            # Only allow normal letters and numbers, which are not commands.
            node.format = None
        return super()._math_char_cmd(node)

    def _math_char(self, node: MathCharITNode) -> ITNode | None:
        if node.format in ("varmathbb", "mathbbm", "mathbbmss", "mathbbmtt", "mathds", "mathbbb"):
            node.format = "mathbb"
        if node.format == "boldsymbol":
            if self.FORMATTING_SYMBOLS.fullmatch(node.char):
                # Convert to mathbf
                node.format = "mathbf"
            # Keep boldsymbol for symbols
        elif not self.FORMATTING_SYMBOLS.fullmatch(node.char):
            # Otherwise only allow normal letters and numbers
            node.format = None
        return super()._math_char(node)


class _SimplifyCombineFormats(InplaceMathstackITNodeTransformer):
    """
    Depending on `outer_format` this function does the following:
    * Combines same formats into a single format wrapper node.
    """

    def _concat(self, node: ConcatITNode) -> ITNode | None:
        # Combine adjacent nodes with same format and wrap in formatting
        if self._in_math:
            new_children = []
            collect_children = []
            current_format = None

            def _flush_math():
                if len(collect_children) > 0:
                    if current_format is None:
                        new_children.extend(collect_children)
                    else:
                        if len(collect_children) > 1:
                            new_child = ConcatITNode(children=collect_children)
                        else:
                            new_child = collect_children[0]
                        if current_format == "mathbf" and any(
                            isinstance(node, MathCharCmdITNode) for node in collect_children
                        ):
                            # Special handling for bm package
                            fmt = "boldsymbol"
                        else:
                            fmt = current_format

                        new_children.append(
                            MathFunctionITNode(
                                function=f"\\{fmt}",
                                args=[new_child],
                                args_prefer_braced=[True],
                            )
                        )

            for child in node.children:
                if isinstance(child, (MathCharITNode, MathUnknownCharITNode, MathCharCmdITNode)):
                    child_format = child.format
                    child.format = None
                else:
                    child = self._node(child)
                    if child is None:
                        continue
                    assert isinstance(child, (MathITNode, TableITNode, GraphicsITNode)), str(child)
                    child_format = None
                if child_format != current_format:
                    # Flush the node
                    _flush_math()
                    current_format = child_format
                    if len(collect_children) > 0:
                        collect_children = []
                if child_format is not None:
                    collect_children.append(child)
                else:
                    new_children.append(child)
            _flush_math()
            node.children = new_children
        else:  # In text
            new_children = []
            collect_children = []
            current_font = DEFAULT_FONT
            last_char_node_idx = None
            last_bold: TextFormatITNode | None = None
            last_italic: TextFormatITNode | None = None

            def _flush_text():
                nonlocal last_bold, last_italic
                if len(collect_children) == 0:
                    return
                if current_font == DEFAULT_FONT:
                    new_children.extend(collect_children)
                else:
                    # Only take as many items as possible for the formatting
                    font_content = collect_children[:last_char_node_idx]
                    if len(font_content) > 0:
                        if len(font_content) > 1:
                            new_child = ConcatITNode(children=font_content)
                        else:
                            new_child = font_content[0]
                        if current_font.bold:
                            if last_bold is not None:
                                inner_joins = new_children[new_children.index(last_bold) + 1 :]
                                # Merge with previous bold
                                if isinstance(last_bold.child, ConcatITNode):
                                    last_bold.child.children.extend(inner_joins)
                                    last_bold.child.children.append(new_child)
                                else:
                                    last_bold.child = ConcatITNode(
                                        children=[last_bold.child, *inner_joins, new_child]
                                    )
                            else:
                                new_child = last_bold = TextFormatITNode(
                                    format="\\textbf", child=new_child
                                )
                        else:
                            last_bold = last_italic = None
                        if current_font.italic_or_slanted:
                            if last_italic is not None:
                                assert last_bold is not None
                                if isinstance(last_bold.child, ConcatITNode):
                                    inner_joins = last_bold.child.children[
                                        last_bold.child.children.index(last_italic) + 1 :
                                    ]
                                else:
                                    inner_joins = ()
                                # Merge with previous italic
                                if isinstance(last_italic.child, ConcatITNode):
                                    last_italic.child.children.extend(inner_joins)
                                    last_italic.child.children.append(new_child)
                                else:
                                    last_italic.child = ConcatITNode(
                                        children=[last_italic.child, *inner_joins, new_child]
                                    )
                            new_child = TextFormatITNode(format="\\textit", child=new_child)
                            last_italic = new_child
                        else:
                            last_italic = None
                        if current_font.monospace:
                            new_child = TextFormatITNode(format="\\texttt", child=new_child)
                        new_children.append(new_child)
                    new_children.extend(collect_children[last_char_node_idx:])

            for child in node.children:
                if isinstance(child, (TextCharITNode, TextUnknownCharITNode, TextCharCmdITNode)):
                    if child.format is not None and child.format != current_font:
                        # Flush the node
                        _flush_text()
                        current_font = child.format
                        collect_children = []
                    child.format = DEFAULT_FONT
                    collect_children.append(child)
                    last_char_node_idx = len(collect_children)
                elif isinstance(child, (GlueITNode, TextRuleITNode, GraphicsITNode)):
                    collect_children.append(child)
                else:
                    assert isinstance(child, (InnerMathITNode, TableITNode)), str(child)
                    last_bold = last_italic = None
                    child = self._node(child)
                    if child is not None:
                        collect_children.append(child)
            _flush_text()
            node.children = new_children
        return node

    def _fraction(self, node: FractionITNode) -> ITNode | None:
        # No recursion for delimiters
        node.numerator = self._node(node.numerator)
        node.denominator = self._node(node.denominator)
        return node

    def _radical(self, node: RadicalITNode) -> ITNode | None:
        # No recursion for radical symbol, only for children
        node.child = self._node(node.child)
        return node

    def _accent_styling(self, node: AccentStylingITNode) -> ITNode | None:
        # No recursion for accent symbol, only for children
        node.child = self._node(node.child)
        return node

    def _left(self, node: LeftITNode) -> ITNode | None:
        if node.delimiter is not None and node.delimiter.format is not None:
            # Move format outside of node
            fmt = node.delimiter.format
            node.delimiter.format = None
            return MathFunctionITNode(function=f"\\{fmt}", args=[node], args_prefer_braced=[True])
        return node

    def _mid(self, node: MidITNode) -> ITNode | None:
        if node.delimiter is not None and node.delimiter.format is not None:
            # Move format outside of node
            fmt = node.delimiter.format
            node.delimiter.format = None
            return MathFunctionITNode(function=f"\\{fmt}", args=[node], args_prefer_braced=[True])
        return node

    def _right(self, node: RightITNode) -> ITNode | None:
        if node.delimiter is not None and node.delimiter.format is not None:
            # Move format outside of node
            fmt = node.delimiter.format
            node.delimiter.format = None
            return MathFunctionITNode(function=f"\\{fmt}", args=[node], args_prefer_braced=[True])
        return node

    def _math_char(self, node: MathCharITNode) -> ITNode | None:
        if node.format is not None:
            fmt = node.format
            node.format = None
            return MathFunctionITNode(function=f"\\{fmt}", args=[node], args_prefer_braced=[True])
        return node

    def _math_unknown_char(self, node: MathUnknownCharITNode) -> ITNode | None:
        if node.format is not None:
            fmt = node.format
            node.format = None
            return MathFunctionITNode(function=f"\\{fmt}", args=[node], args_prefer_braced=[True])
        return node

    def _text_char(self, node: TextCharITNode) -> ITNode | None:
        fmt = node.format
        node.format = None
        res_node = node
        if fmt is not None:
            if fmt.bold:
                res_node = TextFormatITNode(format="\\textbf", child=res_node)
            if fmt.italic_or_slanted:
                res_node = TextFormatITNode(format="\\textit", child=res_node)
            if fmt.monospace:
                res_node = TextFormatITNode(format="\\texttt", child=res_node)
        return res_node

    def _text_unknown_char(self, node: TextUnknownCharITNode) -> ITNode | None:
        # fmt = node.format
        node.format = None
        res_node = node
        # if fmt is not None:
        #     if fmt.bold:
        #         res_node = TextFormatITNode(format="\\textbf", child=res_node)
        #     if fmt.italic_or_slanted:
        #         res_node = TextFormatITNode(format="\\textit", child=res_node)
        #     if fmt.monospace:
        #         res_node = TextFormatITNode(format="\\texttt", child=res_node)
        return res_node


class _ToStr(InplaceITNodeTransformer):
    components: list[str]
    last_is_escaped: bool

    def __init__(self) -> None:
        super().__init__()
        self.components = []
        self.last_is_escaped = False

    def _text_char(self, node: TextCharITNode) -> ITNode | None:
        if not self.last_is_escaped and node.char.isalnum():
            self.components.append(" ")
        self.components.append(node.char)
        self.last_is_escaped = True
        return node

    def _text_unknown_char(self, node: TextUnknownCharITNode) -> ITNode | None:
        self.components.append(f"\\unknown{{{node.font}}}{{{node.char}}}")
        self.last_is_escaped = True
        return node

    def _math_char(self, node: MathCharITNode) -> ITNode | None:
        if not self.last_is_escaped and node.char.isalnum():
            self.components.append(" ")
        self.components.append(node.char)
        self.last_is_escaped = True
        return node

    def _inner_math(self, node: InnerMathITNode) -> ITNode | None:
        self.components.append("\\(")
        self.last_is_escaped = True
        self._node(node.child)
        self.components.append("\\)")
        self.last_is_escaped = True
        return node

    def _math_function(self, node: MathFunctionITNode) -> ITNode | None:
        self.components.append(node.function)
        self.last_is_escaped = not node.function[-1].isalnum()
        for arg, braced in zip(node.args, node.args_prefer_braced):
            if braced or isinstance(arg, ConcatITNode):
                if braced == "Bracket":
                    self.components.append("[")
                else:
                    self.components.append("{")
                self.last_is_escaped = True
            if arg is None or arg == "":
                if not braced:
                    self.components.append(".")
                    self.last_is_escaped = True
            elif isinstance(arg, str):
                self.components.append(arg)
                self.last_is_escaped = not arg[-1].isalnum() or (len(arg) == 1 and not braced)
            else:
                self._node(arg)
            if braced or isinstance(arg, ConcatITNode):
                if braced == "Bracket":
                    self.components.append("]")
                else:
                    self.components.append("}")
                self.last_is_escaped = True
        return node

    def _glue(self, node: GlueITNode) -> ITNode | None:
        self.components.append(node.char)
        self.last_is_escaped = True
        return node

    def _ord(self, node: OrdITNode) -> ITNode | None:
        if node.nucleus is None:
            # self.components.append("{}")
            pass
        else:
            self._node(node.nucleus)
        if node.subscr is not None:
            self.components.append("_")
            if isinstance(node.subscr, (ConcatITNode, OrdITNode)):
                self.components.append("{")
            self.last_is_escaped = True
            self._node(node.subscr)
            if isinstance(node.subscr, (ConcatITNode, OrdITNode)):
                self.components.append("}")
        if node.supscr is not None:
            self.components.append("^")
            if isinstance(node.supscr, (ConcatITNode, OrdITNode)):
                self.components.append("{")
            self.last_is_escaped = True
            self._node(node.supscr)
            if isinstance(node.supscr, (ConcatITNode, OrdITNode)):
                self.components.append("}")
        return node

    def _text_format(self, node: TextFormatITNode) -> ITNode | None:
        self.components.append(node.format)
        self.components.append("{")
        self.last_is_escaped = True
        self._node(node.child)
        self.components.append("}")
        self.last_is_escaped = True
        return node

    def _table_cell(self, node: TableCellITNode) -> TableCellITNode:
        if node.rowspan != 1:
            self.components.append("\\multirow{")
            self.components.append(str(node.rowspan))
            self.components.append("}{*}{")
        if node.colspan != 1:
            self.components.append("\\multicolumn{")
            self.components.append(str(node.colspan))
            self.components.append("}{c}{")
        self._node(node.child)
        if node.colspan != 1:
            self.components.append("}")
        if node.rowspan != 1:
            self.components.append("}")
        return node

    def _table_row(self, node: TableRowITNode) -> TableRowITNode | None:
        for cell in node.cells:
            self._table_cell(cell)
            self.components.append(" & ")
            self.last_is_escaped = True
        return node

    def _table(self, node: TableITNode) -> ITNode | None:
        self.components.append("\\begin{array}{")
        self.last_is_escaped = True
        assert len(node.data) > 0 and len(node.data[0].cells) > 0
        for _ in node.data[0].cells:
            self.components.append(" c")
        self.components.append(" }\n")
        self.last_is_escaped = True
        for row in node.data:
            self._table_row(row)
            # Override the &
            self.components[-1] = " \\\\\n"
            self.last_is_escaped = True
        # Override the \\
        self.components[-1] = "\n\\end{array}"

    def _text_rule(self, node: TextRuleITNode) -> ITNode | None:
        assert False, "Should be function"

    def _math_unkown_char(self, node: MathUnknownCharITNode) -> ITNode | None:
        assert False, "Should be function"

    def _math_char_cmd(self, node: MathCharCmdITNode) -> ITNode | None:
        assert False, "Should be function"

    def _text_char_cmd(self, node: TextCharCmdITNode) -> ITNode | None:
        assert False, "Should be function"

    def _inner_text(self, node: InnerTextITNode) -> ITNode | None:
        assert False, "Should be function"

    def _overline_styling(self, node: OverlineStylingITNode) -> ITNode | None:
        assert False, "Should be function"

    def _underline_styling(self, node: UnderlineStylingITNode) -> ITNode | None:
        assert False, "Should be function"

    def _accent_styling(self, node: AccentStylingITNode) -> ITNode | None:
        assert False, "Should be function"

    def _radical(self, node: RadicalITNode) -> ITNode | None:
        assert False, "Should be function"

    def _fraction(self, node: FractionITNode) -> ITNode | None:
        assert False, "Should be function"

    def _left(self, node: LeftITNode) -> ITNode | None:
        assert False, "Should be function"

    def _mid(self, node: MidITNode) -> ITNode | None:
        assert False, "Should be function"

    def _right(self, node: RightITNode) -> ITNode | None:
        assert False, "Should be function"


def _iter_all(root: ITNode | None) -> Generator[ITNode, None, None]:
    if root is None:
        return
    yield root
    for child in root.inner():
        yield from _iter_all(child)


def compact_math(nodes: list[ITNode], in_place: bool) -> list[ITNode]:
    if not in_place:
        temp = CopyITNodeTransformer()(ConcatITNode(children=nodes))
        if temp is None:
            return []
        if isinstance(temp, ConcatITNode):
            nodes = temp.children
        else:
            nodes = [temp]
    s = _SimplifyFlattenLists()
    res = []
    for node in nodes:
        node = s(node)
        if node is None:
            continue
        res.append(node)
    return res


def compact_math_node(node: ITNode | None, in_place: bool) -> ITNode | None:
    """Compact a single math node. Can be called multiple times without effect."""
    if not in_place:
        node = CopyITNodeTransformer()(node)
    return _SimplifyFlattenLists()(node)


def _dbg_str(node: ITNode | None, indent: str = "") -> str:
    if node is None:
        return "None"
    return node.dbg_str(indent)


def _simplify_math(
    node: ITNode | None, outer_math: bool, in_place: bool, merge_unknown: bool
) -> ITNode | None:
    """Simplify a list of math nodes. Can only be called once, afterwards, math was transformed to functions,
    which cannot be combined any more."""
    if not in_place:
        node = CopyITNodeTransformer()(node)
    if DEBUG_MATH_SIMPLIFY:
        print("MS: Initial: " + _dbg_str(node))
        print("MS: Fix prime")
    node = _FixPrime()(node)
    if DEBUG_MATH_SIMPLIFY:
        print("MS: Fixed prime: " + _dbg_str(node))
        print("MS: Simplify")
    node = _SimplifyFlattenLists()(node)
    if DEBUG_MATH_SIMPLIFY:
        print("MS: Simplified: " + _dbg_str(node))
    if merge_unknown:
        if DEBUG_MATH_SIMPLIFY:
            print("MS: Merge Unknown")
        node = _SimplifyUnknown(outer_math)(node)
        if DEBUG_MATH_SIMPLIFY:
            print("MS: Unknown Merging: " + _dbg_str(node))
    if DEBUG_MATH_SIMPLIFY:
        print("MS: Combine Modes")
    node = _CombineMathTextModes(outer_math)(node)
    if DEBUG_MATH_SIMPLIFY:
        print("MS: Modes Combined: " + _dbg_str(node))
        print("MS: Clean Glues")
    if outer_math and node is not None:
        node = InnerMathITNode(child=node)
    node = _SimplifyCleanGlues()(node)
    if DEBUG_MATH_SIMPLIFY:
        print("MS: Glues Cleaned: " + _dbg_str(node))
        print("MS: Clean formats")
    node = _CleanSymbolFormats()(node)
    if DEBUG_MATH_SIMPLIFY:
        print("MS: Formats Cleaned: " + _dbg_str(node))
        print("MS: Combine formats")
    node = _SimplifyCombineFormats()(node)
    if DEBUG_MATH_SIMPLIFY:
        print("MS: Formats Combined: " + _dbg_str(node))
        print("MS: Combine special symbols")
    node = _CombineSpecialSymbols()(node)
    if DEBUG_MATH_SIMPLIFY:
        print("MS: Special Symbols combined: " + _dbg_str(node))
        print("MS: Final")
    return node


def to_str(
    node: ITNode | None,
    force_math: bool = True,
    modify_in_place: bool = False,
    merge_unknown: bool = True,
) -> str:
    """Convert a math node to a string."""
    node = _simplify_math(
        node, outer_math=force_math, in_place=modify_in_place, merge_unknown=merge_unknown
    )
    if node is None:
        return ""
    node = _MathToFunctions()(node)
    if DEBUG_MATH_SIMPLIFY:
        print("MS: To Functions: " + _dbg_str(node))
    t = _ToStr()
    t(node)
    if DEBUG_MATH_SIMPLIFY:
        print("MS: As String", t.components)
    return "".join(t.components)
