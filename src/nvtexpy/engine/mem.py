# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Callable, ClassVar, Dict, Generator, Set, Tuple, Type, TypeVar, Union, cast

import numpy as np
from pdftex import get_cs, get_mem

UNITY = 0o200000
DEFAULT_CODE = 0o10000000000
NULL_FLAG = -0o10000000000


# Sentinel class
class Default:
    def __repr__(self) -> str:
        return "default"


default = Default()


T = TypeVar("T")
TMemObj = TypeVar("TMemObj", bound="MemObj")


@staticmethod
def _hh_b0(r: "MemReader", addr: int) -> int:
    return r._int16[(addr << 2) + 1]


@staticmethod
def _hh_b1(r: "MemReader", addr: int) -> int:
    return r._int16[addr << 2]


@staticmethod
def _hh_lh(r: "MemReader", addr: int) -> int:
    return r._uint32[addr << 1]


@staticmethod
def _hh_rh(r: "MemReader", addr: int) -> int:
    return r._uint32[(addr << 1) + 1]


@staticmethod
def _glue(r: "MemReader", addr: int) -> float:
    return r._float64[addr]


@staticmethod
def _cint0(r: "MemReader", addr: int) -> int:
    return r._int32[(addr << 1) + 1]


@staticmethod
def _cint1(r: "MemReader", addr: int) -> int:
    return r._int32[(addr << 1) + 0]


@staticmethod
def _v_b0(r: "MemReader", addr: int) -> int:
    return r._uint8[(addr << 3) + 7]


@staticmethod
def _v_b1(r: "MemReader", addr: int) -> int:
    return r._uint8[(addr << 3) + 6]


@staticmethod
def _v_b2(r: "MemReader", addr: int) -> int:
    return r._uint8[(addr << 3) + 5]


@staticmethod
def _v_b3(r: "MemReader", addr: int) -> int:
    return r._uint8[(addr << 3) + 4]


@staticmethod
def _hh_lh_b0(r: "MemReader", addr: int) -> int:
    return r._uint8[(addr << 3)]


@staticmethod
def _hh_lh_b1(r: "MemReader", addr: int) -> int:
    return r._uint8[(addr << 3) + 1]


_int = _cint0
_gr = _glue
_link = _hh_rh
_info = _hh_lh
_type = _hh_b0
_subtype = _hh_b1
_font = _type
_character = _subtype
_math_type = _link
_fam = _font


def _sc(r: "MemReader", addr: int) -> float:
    scint = _int(r, addr)
    return scint / UNITY


def _sc_with_default(r: "MemReader", addr: int) -> Union[float, Default]:
    res = _int(r, addr)
    return default if res == DEFAULT_CODE else res / UNITY


def _sc_with_null(r: "MemReader", addr: int) -> float | None:
    res = _int(r, addr)
    return None if res == NULL_FLAG else res / UNITY


def _llink(self, addr: int) -> int:
    return _info(self, addr + 1)


def _rlink(r: "MemReader", addr: int) -> int:
    return _link(r, addr + 1)


def _is_char_node(r: "MemReader", addr: int) -> bool:
    return addr >= r.hi_mem_min and addr < r.mem_max


def _node_ref(
    inner_reader: Callable[["MemReader", int], int],
) -> Callable[["MemReader", int], "Node"]:
    def reader(r: "MemReader", addr: int) -> Any:
        addr = inner_reader(r, addr)
        if _is_char_node(r, addr):
            return r.get_type(addr, CharNode)
        return r.get_node(addr)

    return reader


def size_to_pt(size: float) -> str:
    return f"{size:.5f}".rstrip("0.") + "pt"


TNode = TypeVar("TNode", bound="Node", covariant=True)


def _node_type_ref(
    inner_reader: Callable[["MemReader", int], int],
    inner_types: Union[Type[TNode], Tuple[Type[TNode], ...]],
) -> Callable[["MemReader", int], TNode | None]:
    node_reader = _node_ref(inner_reader)

    def reader(r: "MemReader", addr: int) -> TNode | None:
        node = node_reader(r, addr)
        if node is None:
            return None
        assert isinstance(node, inner_types), f"Expected {inner_types}, got {type(node)}"
        return node

    return reader


def _token_ref(
    inner_reader: Callable[["MemReader", int], int],
) -> Callable[["MemReader", int], "Token"]:
    def reader(r: "MemReader", addr: int) -> Any:
        addr = inner_reader(r, addr)
        return r.get_token(addr)

    return reader


def _type_ref(
    inner_reader: Callable[["MemReader", int], int],
    data_type: Union[Type[TMemObj], str],
) -> Callable[["MemReader", int], TMemObj]:
    def reader(r: "MemReader", addr: int) -> TMemObj:
        nonlocal data_type
        if isinstance(data_type, str):
            data_type = globals()[data_type]
        assert isinstance(data_type, type) and issubclass(data_type, MemObj)
        return r.get_type(inner_reader(r, addr), data_type)

    return reader


def _list_ref(
    first_reader: Callable[["MemReader", int], TMemObj | None],
    link: Callable[[TMemObj], TMemObj | None] = lambda x: x.link,
) -> Callable[["MemReader", int], Iterable[TMemObj]]:
    def reader(r: "MemReader", addr: int) -> Iterable[TMemObj]:
        obj = first_reader(r, addr)
        while obj is not None:
            # print(f"Yield {obj}")
            yield obj
            nextobj = link(obj)
            if nextobj == obj:
                print(f"Found recursive link: {obj}")
                break
            obj = nextobj

    return reader


class PrintAs(Enum):
    DIRECT = 1
    HEX = 2
    CHAR = 3
    OBJECT = 5
    OBJECT_LIST = 6


@dataclass(frozen=True)
class _MemAccess:
    offset: int
    field: Callable[["MemReader", int], Any]
    visible: bool = True
    print_as: PrintAs = PrintAs.DIRECT


# This is just for typing: The field's return type is actually asserted.
def _mem_access(
    *,
    offset: int = 0,
    field: Callable[["MemReader", int], T],
    str: bool = True,
    print_as: PrintAs = PrintAs.DIRECT,
) -> T:
    # Anonymize the type, it'll only be used as default value, but then removed by the meta class
    # `MemObjMeta`
    return cast(
        Any,
        _MemAccess(offset=offset, field=field, visible=str, print_as=print_as),
    )


class MemObjMeta(type):
    def __new__(cls, name, bases, attrs):
        access = {key: field for key, field in attrs.items() if isinstance(field, _MemAccess)}
        attrs = {key: field for key, field in attrs.items() if not isinstance(field, _MemAccess)}
        if "MemObj" in globals():
            for base in bases:
                if issubclass(base, MemObj) and hasattr(base, "__access__"):
                    access = {**base.__access__, **access}
                    break
        attrs["__access__"] = access
        return super().__new__(cls, name, bases, attrs)


class MemObj(metaclass=MemObjMeta):
    __size__: ClassVar[int] = 1
    __access__: ClassVar[Dict[str, _MemAccess]]
    __slots__ = ("_mem", "addr")
    _mem: "MemReader"
    addr: int

    def __init__(self, mem: "MemReader", addr: int) -> None:
        self._mem = mem
        self.addr = addr

    def __getattr__(self, name: str) -> Any:
        try:
            acc = type(self).__access__[name]
        except BaseException:
            raise AttributeError(name)
        return acc.field(self._mem, self.addr + acc.offset)

    def __hasattr__(self, name: str) -> bool:
        return name in type(self).__access__

    def __dir__(self) -> Iterable[str]:
        return tuple(super().__dir__()) + tuple(self.__access__.keys())

    def _data_to_str(
        self, data: Any, print_as: PrintAs, already_seen: Set[int], max_depth: int
    ) -> str:
        if print_as == PrintAs.DIRECT:
            return f"{data!r}"
        elif print_as == PrintAs.HEX:
            return f"0x{data:06X}"
        elif print_as == PrintAs.CHAR:
            if 0x21 < data < 0x7F:
                return f"{chr(data)!r} (0x{data:02x})"
            return f"0x{data:02x}"
        elif print_as == PrintAs.OBJECT:
            if isinstance(data, MemObj):
                if data.addr in already_seen:
                    return f"({data.__class__.__name__} @0x{data.addr:06X} !recursion!)"
                return data._str(already_seen, max_depth)
            return f"{data}"
        elif print_as == PrintAs.OBJECT_LIST:

            def _field_to_str(idx, subdata):
                try:
                    return subdata._str(already_seen, max_depth)
                except BaseException as e:
                    raise ValueError(f"Error in [{idx}]") from e

            return (
                "["
                + ", ".join(
                    (
                        (
                            f"({subdata.__class__.__name__} @0x{subdata.addr:06X} !recursion!)"
                            if subdata.addr in already_seen
                            else _field_to_str(idx, subdata)
                        )
                        if isinstance(subdata, MemObj)
                        else str(subdata)
                    )
                    for idx, subdata in enumerate(data)
                )
                + "]"
            )
        raise NotImplementedError(f"Unknown print_as: {print_as}")

    def _field_to_str(self, access: _MemAccess, already_seen: Set[int], max_depth: int) -> str:
        data = access.field(self._mem, self.addr + access.offset)
        return self._data_to_str(data, access.print_as, already_seen, max_depth)

    def format_str(
        self, indent: str, already_seen: Set[int] | None = None, max_depth: int = 8
    ) -> str:
        if already_seen is None:
            already_seen = set()
        if max_depth == 0:
            return f"{self.__class__.__name__}(@0x{self.addr:06X}...)"
        max_depth -= 1

        def get_field(key: str, access: _MemAccess) -> Any:
            try:
                return access.field(self._mem, self.addr + access.offset)
            except BaseException as e:
                raise ValueError(f"Error in {self.__class__.__name__}.{key}") from e

        node_data = [
            (key, access, get_field(key, access))
            for key, access in self.__access__.items()
            if access.visible
        ]

        def field_to_str(key: str, access: _MemAccess, data: Any) -> str:
            try:
                return f"{key}={self._data_to_str(data, access.print_as, already_seen, max_depth)}"
            except BaseException as e:
                raise ValueError(f"Error in {self.__class__.__name__}.{key}") from e

        params = ", ".join(
            field_to_str(key, access, data)
            for key, access, data in node_data
            if access.visible
            and (access.print_as not in (PrintAs.OBJECT, PrintAs.OBJECT_LIST) or data is None)
        )
        if params:
            params = ", " + params
        has_multiline = any(
            access.print_as in (PrintAs.OBJECT, PrintAs.OBJECT_LIST) and data is not None
            for key, access, data in node_data
        )
        inner_indent = indent + "  "
        inner_inner_indent = indent + "    "
        for key, access, data in node_data:
            if data is not None:
                try:
                    if access.print_as == PrintAs.OBJECT:
                        params += f",\n{inner_indent}{key}={data.format_str(inner_indent, already_seen, max_depth)}"
                    elif access.print_as == PrintAs.OBJECT_LIST:
                        params += f",\n{inner_indent}{key}=[\n"
                        for subdata in data:
                            params += f"{inner_inner_indent}{subdata.format_str(inner_inner_indent, already_seen, max_depth)},\n"
                        params += f"{inner_indent}]"
                except BaseException as e:
                    raise ValueError(f"Error in {self.__class__.__name__}.{key}") from e
        if has_multiline:
            params += ",\n" + indent
        return f"{self.__class__.__name__}(@0x{self.addr:06X}{params})"

    def _str(self, already_seen: Set[int], max_depth: int) -> str:
        if max_depth == 0:
            return f"{self.__class__.__name__}(@0x{self.addr:06X}...)"
        max_depth -= 1
        already_seen.add(self.addr)

        def field_to_str(key: str, access: _MemAccess) -> str:
            try:
                return f"{key}={self._field_to_str(access, already_seen, max_depth)}"
            except BaseException as e:
                raise ValueError(f"Error in {self.__class__.__name__}.{key}") from e

        params = ", ".join(
            field_to_str(key, access) for key, access in self.__access__.items() if access.visible
        )
        if params:
            params = f", {params}"
        return f"{self.__class__.__name__}(@0x{self.addr:06X}{params})"

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, MemObj):
            return NotImplemented
        return self.addr == __value.addr

    def __hash__(self) -> int:
        return self.addr

    def __str__(self) -> str:
        return self._str(already_seen=set(), max_depth=8)

    def __int__(self) -> int:
        return self.addr

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(addr=0x{self.addr:06X})"


class Node(MemObj):
    __type__: ClassVar[int]
    type: int = _mem_access(field=_type, str=False)
    subtype: int = _mem_access(field=_subtype)
    link: "Node | None" = _mem_access(field=_node_ref(_link), str=False)

    def _recursive_walk(self, nodes: Iterable["Node"], depth: int) -> Iterable["Node"]:
        for node in nodes:
            yield from node.walk()

    def walk(self, depth: int = 0) -> Iterable["Node"]:
        yield self


def node_list(node: Node | None) -> Iterable[Node]:
    while node is not None:
        yield node
        node = node.link


GLUE_NORMAL = 0
GLUE_STRETCHING = 1
GLUE_SHRINKING = 2


class CharNode(Node):
    __size__ = 1
    # Type is implicit by address
    __type__ = -1
    # Override these not to be displayed
    type: int = _mem_access(field=_type, str=False)
    subtype: int = _mem_access(field=_subtype, str=False)
    # The actual new names
    font: int = _mem_access(field=_type)
    character: int = _mem_access(field=_subtype, print_as=PrintAs.CHAR)


class HBoxNode(Node):
    __type__ = 0
    __size__ = 7
    width: float = _mem_access(offset=1, field=_sc)
    depth: float = _mem_access(offset=2, field=_sc)
    height: float = _mem_access(offset=3, field=_sc)
    shift_amount: float = _mem_access(offset=4, field=_sc)
    list: Iterable[Node] = _mem_access(
        offset=5,
        field=_list_ref(_node_ref(_link)),
        print_as=PrintAs.OBJECT_LIST,
    )
    glue_order: int = _mem_access(offset=5, field=_subtype)
    glue_sign: int = _mem_access(offset=5, field=_type)
    glue_set: float = _mem_access(offset=6, field=_gr)

    def walk(self, depth: int = 0) -> Iterable[Node]:
        yield self
        yield from self._recursive_walk(self.list, depth + 1)


class VBoxNode(Node):
    __type__ = 1
    __size__ = 7
    width: float = _mem_access(offset=1, field=_sc)
    depth: float = _mem_access(offset=2, field=_sc)
    height: float = _mem_access(offset=3, field=_sc)
    shift_amount: float = _mem_access(offset=4, field=_sc)
    list: Iterable[Node] = _mem_access(
        offset=5,
        field=_list_ref(_node_ref(_link)),
        print_as=PrintAs.OBJECT_LIST,
    )
    glue_order: int = _mem_access(offset=5, field=_subtype)
    glue_sign: int = _mem_access(offset=5, field=_type)
    glue_set: float = _mem_access(offset=6, field=_gr)

    def walk(self, depth: int = 0) -> Iterable[Node]:
        yield self
        yield from self._recursive_walk(self.list, depth + 1)


class RuleNode(Node):
    __type__ = 2
    __size__ = 4
    width: float | None = _mem_access(offset=1, field=_sc_with_null)
    depth: float | None = _mem_access(offset=2, field=_sc_with_null)
    height: float | None = _mem_access(offset=3, field=_sc_with_null)

    @property
    def is_empty(self) -> bool:
        return (
            (self.width is not None and self.width <= 0)
            or (
                self.depth is not None and self.height is not None and self.depth + self.height <= 0
            )
            or (self.depth is not None and self.height is None and self.depth <= 0)
            or (self.depth is None and self.height is not None and self.height <= 0)
        )


class GlueSpec(MemObj):
    __size__ = 4
    glue_ref_count: int = _mem_access(offset=0, field=_link)
    stretch_order = _mem_access(offset=0, field=_type)
    shrink_order = _mem_access(offset=0, field=_subtype)
    width: float = _mem_access(offset=1, field=_sc)
    stretch: float = _mem_access(offset=2, field=_sc)
    shrink: float = _mem_access(offset=3, field=_sc)


class InsNode(Node):
    __type__ = 3
    __size__ = 5
    float_cost: int = _mem_access(offset=1, field=_int)
    # holds the |split_max_depth| to be used in case this insertion is split, and the
    # |split_top_ptr| points to the corresponding |split_top_skip|
    depth: float = _mem_access(offset=2, field=_sc)
    # Holds the natural height plus depth of the vertical list being inserted.
    height: float = _mem_access(offset=3, field=_sc)
    # The vertical list to be inserted
    ins_ptr: Node | None = _mem_access(offset=4, field=_node_ref(_info), str=False)
    split_top_ptr: GlueSpec = _mem_access(
        offset=4, field=_type_ref(_link, GlueSpec), print_as=PrintAs.OBJECT, str=False
    )

    def walk(self, depth: int = 0) -> Iterable[Node]:
        yield self
        if self.ins_ptr is not None:
            yield from self.ins_ptr.walk(depth + 1)


class TokenCategory(IntEnum):
    bad = -1
    relax = 0
    left_brace = 1
    right_brace = 2
    math_shift = 3
    tab_mark = 4
    out_param = 5
    car_ret = 5
    mac_param = 6
    sup_mark = 7
    sub_mark = 8
    ignore = 9
    endv = 9
    spacer = 10
    letter = 11
    other_char = 12
    active_char = 13
    par_end = 13
    match = 13
    comment = 14
    end_match = 14
    stop = 14
    invalid_char = 15
    delim_num = 15
    max_char_code = 15
    cs_token = 16

    cs_token_flag = 4095

    @staticmethod
    def _from_cat(cat: int) -> "TokenCategory":
        if cat < 0:
            return TokenCategory.bad
        if cat >= 16:
            return TokenCategory.cs_token
        return TokenCategory(cat)


class Token(MemObj):
    __size__ = 1
    link: "Token | None" = _mem_access(field=_token_ref(_link), str=False)
    token: int = _mem_access(field=_info, str=False)
    # Token is split up into two bytes, the category and the character
    char: int = _mem_access(field=_hh_lh_b0, print_as=PrintAs.CHAR)
    category: TokenCategory = _mem_access(
        field=lambda r, addr: TokenCategory._from_cat(_hh_lh_b1(r, addr))
    )
    cs: int = _mem_access(field=lambda r, addr: _info(r, addr) - 4095, str=False)

    # @property
    # def token_category(self) -> TokenCategory:
    #     return TokenCategory(self.token >> 8)

    def __str__(self) -> str:
        """
        if info(p)>=cs_token_flag then print_cs(info(p)-cs_token_flag)
        else  begin m:=info(p) div @'400; c:=info(p) mod @'400;
        if info(p)<0 then print_esc("BAD.")
        @.BAD@>
        else @<Display the token $(|m|,|c|)$@>;
        """
        cat = self.category
        if cat == TokenCategory.cs_token:
            cs = get_cs(self.cs)
            if cs is None:
                return "\\null"
            else:
                return "\\" + cs.tobytes().decode(errors="backslashreplace")
        elif cat == TokenCategory.bad:
            return "BAD"
        elif cat == TokenCategory.out_param:
            # digit (out param)
            return "#" + chr(self.char + 48)
        elif cat.value <= 12:
            # Printable single character
            return chr(self.char)
        elif cat == TokenCategory.match:
            # Matched single character
            return "#" + chr(self.char)
        elif cat == TokenCategory.end_match:
            # End match
            return "->"
        else:
            return "BAD"


class TokenList(MemObj):
    __size__ = 1
    # Token list is preceded by a ref_count token
    ref_count: int = _mem_access(field=_info)
    # Ignore the first refcount token!
    tokens: Iterable[Token] = _mem_access(
        field=_list_ref(_token_ref(_link)),
        print_as=PrintAs.OBJECT_LIST,
    )

    def __str__(self) -> str:
        """
        if info(p)>=cs_token_flag then print_cs(info(p)-cs_token_flag)
        else  begin m:=info(p) div @'400; c:=info(p) mod @'400;
        if info(p)<0 then print_esc("BAD.")
        @.BAD@>
        else @<Display the token $(|m|,|c|)$@>;
        """
        token_strs = []
        n = 48
        match_chr = "#"
        for token in self.tokens:
            cat = token.category
            if cat == TokenCategory.cs_token:
                cs = get_cs(token.cs)
                if cs is None:
                    token_strs.append("\\null")
                else:
                    token_strs.append("\\" + cs.tobytes().decode(errors="backslashreplace"))
            elif cat == TokenCategory.bad:
                token_strs.append("BAD")
            elif cat == TokenCategory.out_param:
                # digit (out param)
                token_strs.append(match_chr + chr(token.char + 48))
            elif cat.value <= 12:
                # Printable single character
                token_strs.append(chr(token.char).replace("\n", "\\n"))
            elif cat == TokenCategory.match:
                # Matched single character
                match_chr = chr(token.char).replace("\n", "\\n")
                n += 1
                token_strs.append(chr(token.char) + chr(n))
            elif cat == TokenCategory.end_match:
                # End match
                token_strs.append("->")
            else:
                token_strs.append("BAD")
        return ", ".join(token_strs)

    def to_plain_str(self, resolve_cs: bool = False, evaluate_cs: bool = False) -> str | None:
        token_strs = []
        for token in self.tokens:
            cat = token.category
            if cat == TokenCategory.cs_token:
                if resolve_cs:
                    token_strs.append(
                        "\\" + get_cs(token.cs).tobytes().decode(errors="backslashreplace")
                    )
                elif evaluate_cs:
                    from nvtexpy.engine.common import CommandCode, get_cs_value

                    print(f"Resolving {get_cs(token.cs).tobytes()}")
                    token_type, token_value = get_cs_value(token.cs)
                    token_type = CommandCode(token_type)
                    print(f"-> {token_type.name}")
                    if token_type == CommandCode.char_given:
                        token_strs.append(chr(token_value))
                    elif token_type == CommandCode.call:
                        inner = self._mem.get_token_list(token_value).to_plain_str(evaluate_cs=True)
                        if inner is None:
                            return None
                        token_strs.append(inner)
                    elif token_type == CommandCode.ex_space:
                        token_strs.append(" ")
                    else:
                        token_strs.append(
                            "\\" + get_cs(token.cs).tobytes().decode(errors="backslashreplace")
                        )
                else:
                    return None
            elif cat == TokenCategory.spacer:
                token_strs.append(" ")
            elif 1 <= cat.value <= 12:
                # Printable single character
                token_strs.append(chr(token.char))
            elif cat == TokenCategory.end_match:
                if len(token_strs) != 0:
                    return None
            else:
                return None
        return "".join(token_strs)


class EmptyTokenList:
    tokens: Iterable[Token] = ()

    def __str__(self) -> str:
        return ""

    def to_plain_str(self, resolve_cs: bool = False) -> str | None:
        return ""


EMPTY_TOKEN_LIST = EmptyTokenList()


class NoRefcountTokenList(MemObj):
    tokens: Iterable[Token] = _mem_access(
        field=_list_ref(_token_ref(lambda r, addr: addr)),
        print_as=PrintAs.OBJECT_LIST,
    )

    __str__ = TokenList.__str__
    to_plain_str = TokenList.to_plain_str


class MarkNode(Node):
    __type__ = 4
    __size__ = 2
    mark_ptr: TokenList = _mem_access(offset=1, field=_type_ref(_link, TokenList))
    mark_class: int = _mem_access(offset=1, field=_info)


class AdjustNode(Node):
    __type__ = 5
    __size__ = 2
    adjust: Iterable[Node] = _mem_access(
        offset=1, field=_list_ref(_node_ref(_int)), print_as=PrintAs.OBJECT_LIST
    )

    def walk(self, depth: int = 0) -> Iterable[Node]:
        yield self
        yield from self._recursive_walk(self.adjust, depth + 1)


class LigatureNode(Node):
    __type__ = 6
    __size__ = 2
    # ligs: Iterable[CharNode] = _mem_access(
    #     offset=1,
    #     field=_list_ref(_node_type_ref(_link, CharNode)),
    #     print_as=PrintAs.OBJECT_LIST,
    # )
    ligs: Iterable[Node] = _mem_access(
        offset=1,
        field=_list_ref(_node_ref(_link)),
        print_as=PrintAs.OBJECT_LIST,
    )

    def walk(self, depth: int = 0) -> Iterable[Node]:
        yield self
        yield from self._recursive_walk(self.ligs, depth + 1)


class DiscNode(Node):
    __type__ = 7
    __size__ = 2
    replace_count: int = _mem_access(field=_subtype)
    pre_break: Node | None = _mem_access(field=_node_ref(_llink), print_as=PrintAs.OBJECT)
    post_break: Node | None = _mem_access(field=_node_ref(_rlink), print_as=PrintAs.OBJECT)

    def walk(self, depth: int = 0) -> Iterable[Node]:
        yield self
        if self.pre_break is not None:
            yield from self.pre_break.walk(depth + 1)
        if self.post_break is not None:
            yield from self.post_break.walk(depth + 1)


class WhatsitSubtype(IntEnum):
    open_node = 0
    write_node = 1
    close_node = 2
    special_node = 3
    latespecial_node = 4
    immediate_code = 5
    set_language_code = 6
    pdf_literal_node = 7 + 0
    pdf_lateliteral_node = 7 + 1
    pdf_obj_code = 7 + 2
    pdf_refobj_node = 7 + 3
    pdf_xform_code = 7 + 4
    pdf_refxform_node = 7 + 5
    pdf_ximage_code = 7 + 6
    pdf_refximage_node = 7 + 7
    pdf_annot_node = 7 + 8
    pdf_start_link_node = 7 + 9
    pdf_end_link_node = 7 + 10
    pdf_outline_code = 7 + 11
    pdf_dest_node = 7 + 12
    pdf_thread_node = 7 + 13
    pdf_start_thread_node = 7 + 14
    pdf_end_thread_node = 7 + 15
    pdf_save_pos_node = 7 + 16
    pdf_info_code = 7 + 17
    pdf_catalog_code = 7 + 18
    pdf_names_code = 7 + 19
    pdf_font_attr_code = 7 + 20
    pdf_include_chars_code = 7 + 21
    pdf_map_file_code = 7 + 22
    pdf_map_line_code = 7 + 23
    pdf_trailer_code = 7 + 24
    pdf_trailer_id_code = 7 + 25
    reset_timer_code = 7 + 26
    pdf_font_expand_code = 7 + 27
    set_random_seed_code = 7 + 28
    pdf_snap_ref_point_node = 7 + 29
    pdf_snapy_node = 7 + 30
    pdf_snapy_comp_node = 7 + 31
    pdf_glyph_to_unicode_code = 7 + 32
    pdf_colorstack_node = 7 + 33
    pdf_setmatrix_node = 7 + 34
    pdf_save_node = 7 + 35
    pdf_restore_node = 7 + 36
    pdf_nobuiltin_tounicode_code = 7 + 37
    pdf_interword_space_on_node = 7 + 38
    pdf_interword_space_off_node = 7 + 39
    pdf_fake_space_node = 7 + 40
    pdf_running_link_off_node = 7 + 41
    pdf_running_link_on_node = 7 + 42
    pdf_space_font_code = 7 + 43
    pdftex_last_extension_code = 7 + 43

    unknown = -1


def _whatsit_subtype(r: "MemReader", addr: int) -> WhatsitSubtype:
    try:
        return WhatsitSubtype(_subtype(r, addr))
    except ValueError:
        return WhatsitSubtype.unknown


class WhatsitNode(Node):
    __type__ = 8
    __size__ = 1

    __subtype__: ClassVar[int] = -1

    subtype: WhatsitSubtype = _mem_access(field=_whatsit_subtype)


class WhatsitPdfSaveNode(WhatsitNode):
    __subtype__ = WhatsitSubtype.pdf_save_node.value
    __size__ = 2


class WhatsitPdfRestoreNode(WhatsitNode):
    __subtype__ = WhatsitSubtype.pdf_restore_node.value
    __size__ = 2


class WhatsitPdfSetMatrixNode(WhatsitNode):
    __subtype__ = WhatsitSubtype.pdf_setmatrix_node.value
    __size__ = 2

    setmatrix_data: TokenList = _mem_access(offset=1, field=_type_ref(_link, TokenList))


class MathNode(Node):
    __type__ = 9
    __size__ = 2


class GlueSubtype(IntEnum):
    other = -1
    normal = 0
    # Note: The following are +1 (see new_skip_param, which adds one to subtype)
    # interline glue if |baseline_skip| is infeasible
    line_skip_code = 1
    # desired glue between baselines
    baseline_skip_code = 2
    # extra glue just above a paragraph
    par_skip_code = 3
    # extra glue just above displayed math
    above_display_skip_code = 4
    # extra glue just below displayed math
    below_display_skip_code = 5
    # glue above displayed math following short lines
    above_display_short_skip_code = 6
    # glue below displayed math following short lines
    below_display_short_skip_code = 7
    # glue at left of justified lines
    left_skip_code = 8
    # glue at right of justified lines
    right_skip_code = 9
    # glue at top of main pages
    top_skip_code = 10
    # glue at top of split pages
    split_top_skip_code = 11
    # glue between aligned entries
    tab_skip_code = 12
    # glue between words (if not |zero_glue|)
    space_skip_code = 13
    # glue after sentences (if not |zero_glue|)
    xspace_skip_code = 14
    # glue on last line of paragraph
    par_fill_skip_code = 15
    # thin space in math formula
    thin_mu_skip_code = 16
    # medium space in math formula
    med_mu_skip_code = 17
    # thick space in math formula
    thick_mu_skip_code = 18


def _glue_subtype(r: "MemReader", addr: int) -> GlueSubtype:
    try:
        return GlueSubtype(_subtype(r, addr))
    except ValueError:
        # Phil: I believe this can happen for skip and muskip glues. Need to check
        return GlueSubtype.other


class GlueNode(Node):
    __type__ = 10
    __size__ = 2
    subtype: GlueSubtype = _mem_access(field=_glue_subtype)
    glue: GlueSpec = _mem_access(field=_type_ref(_llink, GlueSpec), print_as=PrintAs.OBJECT)
    leader: RuleNode | HBoxNode | VBoxNode | None = _mem_access(
        field=_node_type_ref(_rlink, (RuleNode, HBoxNode, VBoxNode)),
        print_as=PrintAs.OBJECT,
    )

    def walk(self, depth: int = 0) -> Iterable[Node]:
        yield self
        if self.leader is not None:
            yield from self.leader.walk(depth + 1)


class KernSubtype(IntEnum):
    INVALID = -1
    normal = 0
    explicit = 1
    acc_kern = 2
    auto_kern = 3
    mu_glue = 99

    @classmethod
    def _missing_(cls, value):
        return cls.INVALID


class KernNode(Node):
    __type__ = 11
    __size__ = 2
    subtype: KernSubtype = _mem_access(field=lambda r, a: KernSubtype(_subtype(r, a)))
    width: float = _mem_access(offset=1, field=_sc)

    raw_width: int = _mem_access(offset=1, field=_int, str=False, print_as=PrintAs.HEX)


class MarginalKernSubtype(IntEnum):
    left_side = 0
    right_side = 1


class MarginalKern(Node):
    __type__ = 40
    __size__ = 3
    subtype: MarginalKernSubtype = _mem_access(
        field=lambda r, a: MarginalKernSubtype(_subtype(r, a))
    )
    width: float = _mem_access(offset=1, field=_sc)
    margin_char: int = _mem_access(offset=2, field=_info)


class PenaltyNode(Node):
    __type__ = 12
    __size__ = 2
    penalty: int = _mem_access(offset=1, field=_int)


class UnsetNode(Node):
    __type__ = 13
    __size__ = 7
    span_count: int = _mem_access(offset=0, field=_hh_b1)
    width: float = _mem_access(offset=1, field=_sc)
    depth: float = _mem_access(offset=2, field=_sc)
    height: float = _mem_access(offset=3, field=_sc)
    glue_shrink: float = _mem_access(offset=4, field=_sc)
    list: Iterable[Node] = _mem_access(
        offset=5,
        field=_list_ref(_node_ref(_link)),
        print_as=PrintAs.OBJECT_LIST,
    )
    glue_order: int = _mem_access(offset=5, field=_subtype)
    glue_sign: int = _mem_access(offset=5, field=_type)
    glue_stretch: float = _mem_access(offset=7, field=_sc)

    def walk(self, depth: int = 0) -> Iterable[Node]:
        yield self
        yield from self._recursive_walk(self.list, depth + 1)


class StyleType(IntEnum):
    display = 0
    text = 2
    script = 4
    script_script = 6


class StyleNode(Node):
    # Flag
    SUBTYPE_CRAMPED = 1

    __type__ = UnsetNode.__type__ + 1
    __size__ = 3
    # Width + Depth are unused!

    @property
    def style(self) -> StyleType:
        return StyleType(self.subtype & 6)

    @property
    def cramped(self) -> bool:
        return bool(self.subtype & self.SUBTYPE_CRAMPED)

    # In the nvtexpy_mlist_to_hlist_converted, this holds the converted hlist.
    # The hlist resulting from each noad is recorded in that noad's |new_hlist|
    # field, an integer field that replaces the |nucleus| or |thickness|.
    new_hlist: Iterable[Node] = _mem_access(offset=1, field=_list_ref(_node_ref(_int)), str=False)
    new_hlist_head: Node | None = _mem_access(offset=1, field=_node_ref(_int), str=False)


class NoadField(MemObj):
    math_type: int = _mem_access(field=_math_type, str=False)

    def walk(self, depth: int = 0) -> Iterable[Node]:
        return ()


class NoadFieldMathChar(NoadField):
    __type__ = 1
    family: int = _mem_access(field=_fam)
    character: int = _mem_access(field=_character, print_as=PrintAs.CHAR)


class NoadFieldSubBox(NoadField):
    __type__ = 2
    box: Node = _mem_access(field=_node_ref(_info), print_as=PrintAs.OBJECT)

    def walk(self, depth: int = 0):
        yield from self.box.walk(depth)


class NoadFieldSubMList(NoadField):
    __type__ = 3
    mlist: Iterable[Node] = _mem_access(
        field=_list_ref(_node_ref(_info)), print_as=PrintAs.OBJECT_LIST
    )
    mlist_head: Node | None = _mem_access(field=_node_ref(_info), str=False)

    def walk(self, depth: int = 0):
        for node in self.mlist:
            yield from node.walk(depth)


class NoadFieldMathTextChar(NoadField):
    __type__ = 4
    family: int = _mem_access(field=_fam)
    character: int = _mem_access(field=_character, print_as=PrintAs.CHAR)


_noad_field_types: Dict[int, Type[NoadField]] = {
    field_type.__type__: field_type
    for field_type in [
        NoadFieldMathChar,
        NoadFieldSubBox,
        NoadFieldSubMList,
        NoadFieldMathTextChar,
    ]
}


class NoadFieldError(NoadField):
    def format_str(
        self, indent: str, already_seen: Set[int] | None = None, max_depth: int = 8
    ) -> str:
        return f"Error: {self._mem.str_memoryword(self.addr)}"

    def __str__(self) -> str:
        return "Error: " + self._mem.str_memoryword(self.addr)

    def __repr__(self) -> str:
        return f"NoadFieldError(addr=0x{self.addr:06X}, data={self._mem.str_memoryword(self.addr)})"


def _noad_field_ref(r: "MemReader", addr: int) -> NoadField | None:
    math_type = _math_type(r, addr)
    if math_type == 0:
        # A normal Empty field
        return None
    # print(f"  - _noad_field_ref: {math_type} @0x{addr:06X}:{r._uint8[addr<<3:(addr<<3) +16]}")
    # lh = 0, rh = _null -> empty!
    # if math_type > 4:
    #     print(f"WARNING: Invalid noad field type: 0x{math_type:06X}")
    #     return NoadFieldError(r, addr)
    return r.get_type(addr, _noad_field_types[math_type])


class Noad(Node):
    # If this noad could be reduced e.g. to just the nucleus
    # (not the case for radicals, etc.)
    __is_basic__ = True
    __size__ = 4
    nucleus: NoadField | None = _mem_access(
        offset=1, field=_noad_field_ref, print_as=PrintAs.OBJECT
    )
    supscr: NoadField | None = _mem_access(offset=2, field=_noad_field_ref, print_as=PrintAs.OBJECT)
    subscr: NoadField | None = _mem_access(offset=3, field=_noad_field_ref, print_as=PrintAs.OBJECT)

    # In the nvtexpy_mlist_to_hlist_converted, this holds the converted hlist.
    # The hlist resulting from each noad is recorded in that noad's |new_hlist|
    # field, an integer field that replaces the |nucleus| or |thickness|.
    new_hlist: Iterable[Node] = _mem_access(offset=1, field=_list_ref(_node_ref(_int)), str=False)
    new_hlist_head: Node | None = _mem_access(offset=1, field=_node_ref(_int), str=False)

    def walk(self, depth: int = 0) -> Iterable[Node]:
        yield self
        if self.nucleus is not None:
            yield from self.nucleus.walk(depth + 1)
        if self.supscr is not None:
            yield from self.supscr.walk(depth + 1)
        if self.subscr is not None:
            yield from self.subscr.walk(depth + 1)

    @staticmethod
    def is_noad(node: Node | None) -> bool:
        return isinstance(node, (Noad, StyleNode, ChoiceNode, FractionNoad, LeftNoad, RightNoad))


class ChoiceNode(Node):
    __type__ = UnsetNode.__type__ + 2
    __size__ = 3

    # Note: Many things can appear in a mlist, so just allow all nodes
    display_mlist: Iterable[Node] = _mem_access(
        offset=1,
        field=_list_ref(_node_ref(_info)),
        print_as=PrintAs.OBJECT_LIST,
    )
    display_mlist_head: Node | None = _mem_access(
        offset=1,
        field=_node_ref(_info),
        str=False,
    )
    text_mlist: Iterable[Node] = _mem_access(
        offset=1,
        field=_list_ref(_node_ref(_link)),
        print_as=PrintAs.OBJECT_LIST,
    )
    text_mlist_head: Node | None = _mem_access(
        offset=1,
        field=_node_ref(_link),
        str=False,
    )
    script_mlist: Iterable[Node] = _mem_access(
        offset=2,
        field=_list_ref(_node_ref(_info)),
        print_as=PrintAs.OBJECT_LIST,
    )
    script_mlist_head: Node | None = _mem_access(
        offset=2,
        field=_node_ref(_info),
        str=False,
    )
    script_script_mlist: Iterable[Node] = _mem_access(
        offset=2,
        field=_list_ref(_node_ref(_link)),
        print_as=PrintAs.OBJECT_LIST,
    )
    script_script_mlist_head: Node | None = _mem_access(
        offset=2,
        field=_node_ref(_link),
        str=False,
    )

    def walk(self, depth: int = 0) -> Iterable[Node]:
        yield self
        yield from self._recursive_walk(self.display_mlist, depth + 1)
        yield from self._recursive_walk(self.text_mlist, depth + 1)
        yield from self._recursive_walk(self.script_mlist, depth + 1)
        yield from self._recursive_walk(self.script_script_mlist, depth + 1)

    def mlist(self, style: StyleType) -> Iterable[Node]:
        if style == StyleType.display:
            return self.display_mlist
        elif style == StyleType.text:
            return self.text_mlist
        elif style == StyleType.script:
            return self.script_mlist
        elif style == StyleType.script_script:
            return self.script_script_mlist
        else:
            assert False, f"Invalid style: {style}"

    def mlist_head(self, style: StyleType) -> Node | None:
        if style == StyleType.display:
            return self.display_mlist_head
        elif style == StyleType.text:
            return self.text_mlist_head
        elif style == StyleType.script:
            return self.script_mlist_head
        elif style == StyleType.script_script:
            return self.script_script_mlist_head
        else:
            assert False, f"Invalid style: {style}"

    def mlist_heads(self) -> Iterable[Node | None]:
        return (
            self.display_mlist_head,
            self.text_mlist_head,
            self.script_mlist_head,
            self.script_script_mlist_head,
        )

    def mlists(self) -> Iterable[Iterable[Node]]:
        return (
            self.display_mlist,
            self.text_mlist,
            self.script_mlist,
            self.script_script_mlist,
        )

    # In the nvtexpy_mlist_to_hlist_converted, this holds the converted hlist.
    # The hlist resulting from each noad is recorded in that noad's |new_hlist|
    # field, an integer field that replaces the |nucleus| or |thickness|.
    new_hlist: Iterable[Node] = _mem_access(offset=1, field=_list_ref(_node_ref(_int)), str=False)
    new_hlist_head: Node | None = _mem_access(offset=1, field=_node_ref(_int), str=False)


class OrdNoad(Noad):
    __type__ = UnsetNode.__type__ + 3


class OpNoad(Noad):
    __type__ = OrdNoad.__type__ + 1


class BinNoad(Noad):
    __type__ = OrdNoad.__type__ + 2


class RelNoad(Noad):
    __type__ = OrdNoad.__type__ + 3


class OpenNoad(Noad):
    __type__ = OrdNoad.__type__ + 4


class CloseNoad(Noad):
    __type__ = OrdNoad.__type__ + 5


class PunctNoad(Noad):
    __type__ = OrdNoad.__type__ + 6


class InnerNoad(Noad):
    __type__ = OrdNoad.__type__ + 7


class Delimiter(MemObj):
    __size__ = 4
    small_fam: int = _mem_access(field=_v_b0)
    small_char: int = _mem_access(field=_v_b1)
    large_fam: int = _mem_access(field=_v_b2)
    large_char: int = _mem_access(field=_v_b3)


def _ref_delimiter(r: "MemReader", addr: int) -> Delimiter | None:
    if _int(r, addr) == 0:
        return None
    return r.get_type(addr, Delimiter)


class RadicalNoad(Noad):
    __is_basic__ = False
    __type__ = InnerNoad.__type__ + 1
    __size__ = 5
    left_delimiter: Delimiter | None = _mem_access(
        offset=4, field=_ref_delimiter, print_as=PrintAs.OBJECT
    )


class FractionNoad(Node):
    # |fraction_noad| is able to represent all of \TeX's operators \.{\\over},
    # \.{\\atop}, \.{\\above}, \.{\\overwithdelims}, \.{\\atopwithdelims}, and
    # \.{\\abovewithdelims}.
    __type__ = RadicalNoad.__type__ + 1
    __size__ = 6
    thickness: Union[float, Default] = _mem_access(offset=1, field=_sc_with_default)
    numerator: NoadField | None = _mem_access(
        offset=2, field=_noad_field_ref, print_as=PrintAs.OBJECT
    )
    denominator: NoadField | None = _mem_access(
        offset=3, field=_noad_field_ref, print_as=PrintAs.OBJECT
    )
    left_delimiter: Delimiter | None = _mem_access(
        offset=4, field=_ref_delimiter, print_as=PrintAs.OBJECT
    )
    right_delimiter: Delimiter | None = _mem_access(
        offset=5, field=_ref_delimiter, print_as=PrintAs.OBJECT
    )

    # In the nvtexpy_mlist_to_hlist_converted, this holds the converted hlist.
    # The hlist resulting from each noad is recorded in that noad's |new_hlist|
    # field, an integer field that replaces the |nucleus| or |thickness|.
    new_hlist: Iterable[Node] = _mem_access(offset=1, field=_list_ref(_node_ref(_int)), str=False)
    new_hlist_head: Node | None = _mem_access(offset=1, field=_node_ref(_int), str=False)

    def walk(self, depth: int = 0) -> Iterable[Node]:
        yield self
        if self.numerator is not None:
            yield from self.numerator.walk(depth + 1)
        if self.denominator is not None:
            yield from self.denominator.walk(depth + 1)


class UnderNoad(Noad):
    __is_basic__ = False
    __type__ = FractionNoad.__type__ + 1


class OverNoad(Noad):
    __is_basic__ = False
    __type__ = UnderNoad.__type__ + 1


class AccentNoad(Noad):
    __is_basic__ = False
    __type__ = OverNoad.__type__ + 1
    __size__ = 5

    # The accent character
    accent_family: int = _mem_access(offset=4, field=_fam)
    accent_character: int = _mem_access(offset=4, field=_character, print_as=PrintAs.CHAR)


class VCenterNoad(Noad):
    __type__ = AccentNoad.__type__ + 1


class LeftNoad(Node):
    __type__ = VCenterNoad.__type__ + 1
    __size__ = 4

    # Has no nucleus, no sub/supscr
    delimiter: Delimiter | None = _mem_access(
        offset=1, field=_ref_delimiter, print_as=PrintAs.OBJECT
    )

    # Disable these
    # nucleus: NoadField | None = _mem_access(offset=1, field=_disabled, str=False)
    # supscr: NoadField | None = _mem_access(offset=2, field=_disabled, str=False)
    # subscr: NoadField | None = _mem_access(offset=3, field=_disabled, str=False)

    # In the nvtexpy_mlist_to_hlist_converted, this holds the converted hlist.
    # The hlist resulting from each noad is recorded in that noad's |new_hlist|
    # field, an integer field that replaces the |nucleus| or |thickness|.
    new_hlist: Iterable[Node] = _mem_access(offset=1, field=_list_ref(_node_ref(_int)), str=False)
    new_hlist_head: Node | None = _mem_access(offset=1, field=_node_ref(_int), str=False)


class RightNoadSubtype(IntEnum):
    OTHER = -1
    RIGHT_NOAD = 0
    MIDDLE_NOAD = 1


def _right_noad_subtype(r: "MemReader", addr: int) -> RightNoadSubtype:
    try:
        return RightNoadSubtype(_subtype(r, addr))
    except ValueError:
        return RightNoadSubtype.OTHER


class RightNoad(Node):
    __type__ = LeftNoad.__type__ + 1

    subtype: RightNoadSubtype = _mem_access(field=_right_noad_subtype)

    # Has no nucleus, no sub/supscr
    delimiter: Delimiter | None = _mem_access(
        offset=1, field=_ref_delimiter, print_as=PrintAs.OBJECT
    )

    # Disable these
    # nucleus: NoadField | None = _mem_access(offset=1, field=_disabled, str=False)
    # supscr: NoadField | None = _mem_access(offset=2, field=_disabled, str=False)
    # subscr: NoadField | None = _mem_access(offset=3, field=_disabled, str=False)

    # In the nvtexpy_mlist_to_hlist_converted, this holds the converted hlist.
    # The hlist resulting from each noad is recorded in that noad's |new_hlist|
    # field, an integer field that replaces the |nucleus| or |thickness|.
    new_hlist: Iterable[Node] = _mem_access(offset=1, field=_list_ref(_node_ref(_int)), str=False)
    new_hlist_head: Node | None = _mem_access(offset=1, field=_node_ref(_int), str=False)


nodes_by_type: Dict[int, Type[Node]] = {
    node_type.__type__: node_type
    for node_type in [
        HBoxNode,
        VBoxNode,
        RuleNode,
        InsNode,
        MarkNode,
        AdjustNode,
        LigatureNode,
        DiscNode,
        MathNode,
        GlueNode,
        KernNode,
        PenaltyNode,
        UnsetNode,
        StyleNode,
        ChoiceNode,
        OrdNoad,
        OpNoad,
        BinNoad,
        RelNoad,
        OpenNoad,
        CloseNoad,
        PunctNoad,
        InnerNoad,
        RadicalNoad,
        FractionNoad,
        UnderNoad,
        OverNoad,
        AccentNoad,
        VCenterNoad,
        LeftNoad,
        RightNoad,
    ]
}
whatsit_nodes_by_subtype: Dict[int, Type[WhatsitNode]] = {
    node_type.__subtype__: node_type
    for node_type in [
        WhatsitPdfRestoreNode,
        WhatsitPdfSaveNode,
        WhatsitPdfSetMatrixNode,
    ]
}


TMemObj = TypeVar("TMemObj", bound=MemObj)


class MemReader:
    def __init__(self) -> None:
        (
            self.mem,
            self.mem_min,
            self.mem_max,
            self.lo_mem_max,
            self.hi_mem_min,
        ) = get_mem()
        self._uint8 = np.frombuffer(self.mem, dtype=np.uint8)
        self._int16 = np.frombuffer(self.mem, dtype=np.int16)
        self._uint32 = np.frombuffer(self.mem, dtype=np.uint32)
        self._int32 = np.frombuffer(self.mem, dtype=np.int32)
        self._float64 = np.frombuffer(self.mem, dtype=np.float64)

    def str_memoryword(self, addr: int) -> str:
        return (
            f"0x{addr:06X}: "
            f"int={_int(self, addr)}, "
            f"lh,rh=({_hh_lh(self, addr)}, {_hh_rh(self, addr)}), "
            f"b0,b1=({_hh_b0(self, addr)}, {_hh_b1(self, addr)}), "
            f"sc={_sc(self, addr) / UNITY}, "
            f"gr={_gr(self, addr)}, "
            f"link=0x{_link(self, addr):06X}, "
            f"info=0x{_info(self, addr):06X}, "
            f"type={_type(self, addr)}, "
            f"subtype={_subtype(self, addr)}"
        )

    def get_type(self, addr: int, data_type: Type[TMemObj]) -> TMemObj:
        assert issubclass(data_type, MemObj)
        return data_type(self, addr)

    def get_node(self, addr: int) -> Node | None:
        if _is_char_node(self, addr):
            return self.get_type(addr, CharNode)
        if addr == 0 or addr < self.mem_min or addr > self.mem_max:
            return None
        node_type_int = _type(self, addr)
        if node_type_int in nodes_by_type:
            node_type = nodes_by_type[node_type_int]
        elif node_type_int == WhatsitNode.__type__:
            subtype_int = _subtype(self, addr)
            node_type = whatsit_nodes_by_subtype.get(subtype_int, WhatsitNode)
        else:
            return None

        # print("get_node", node_type)
        # for i in range(node_type.__size__):
        #     print(f"  - @offset={i}: {self.str_memoryword(addr + i)}")
        return self.get_type(addr, node_type)

    def get_token(self, addr: int) -> Token | None:
        if addr == 0 or addr < self.mem_min or addr > self.mem_max:
            return None
        return self.get_type(addr, Token)

    def get_token_list(self, addr: int) -> TokenList:
        return self.get_type(addr, TokenList)

    def get_norefcount_token_list(self, addr: int) -> Union[NoRefcountTokenList, EmptyTokenList]:
        if addr == 0 or addr < self.mem_min or addr > self.mem_max:
            return EMPTY_TOKEN_LIST
        return self.get_type(addr, NoRefcountTokenList)

    def iter_head_tail(self, head_addr: int, tail_addr: int) -> Generator[Node, None, None]:
        node = self.get_node(head_addr)
        while True:
            if node.addr == tail_addr:
                break
            node = node.link
            if node is None:
                break
            yield node

    def iter_nodes(
        self, first_node: Node, last_node: Node, include_last: bool = False
    ) -> Generator[Node, None, None]:
        while True:
            yield first_node
            first_node = first_node.link
            if first_node is None:
                break
            if first_node.addr == last_node.addr:
                if include_last:
                    yield last_node
                break
