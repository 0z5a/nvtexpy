# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Container, Dict, List

from pdftex import get_box, get_cs, get_cs_value, get_cur_font, get_font_properties, id_lookup

from nvtexpy.engine.debug import DEBUG_CS_RESOLVE, DebugHex
from nvtexpy.engine.mem import UNITY, MemReader
from nvtexpy.engine.mem import Node as MemNode
from nvtexpy.engine.mem import Token, TokenCategory, TokenList
from nvtexpy.engine.stack_formatter import FormatterElement
from nvtexpy.output.structured_output import StructuredOutput

NULL = -0xFFFFFFF


@dataclass
class SaveLevel:
    addr: int
    level: int
    group_code: int
    env_name: (
        str  # If available, the contents of @currenvir at the time of the save_level start event
    )
    end_hooks: List[Callable] = field(default_factory=list)

    debug_local_registers: Dict[int, str] | None = None


@dataclass
class FontProperties:
    font: int
    bold: bool = False
    italic_or_slanted: bool = False
    monospace: bool = False

    @staticmethod
    def from_tex(font: int, series: str, shape: str, fam: str) -> "FontProperties":
        return FontProperties(
            font=font,
            bold="b" in series,
            italic_or_slanted="it" in shape or "sl" in shape,
            monospace=fam in ["cmtt", "lmtt", "pcr", "qcr"],
        )

    @staticmethod
    def from_tex_name(font: int, name: str) -> "FontProperties":
        encoding_val, fam_val, series_val, shape_val = name.split("/")
        return FontProperties.from_tex(font, series_val, shape_val, fam_val)


@dataclass
class Node:
    orig_addr: DebugHex
    dbg_char: str
    rand_id: int
    character: int
    font: int
    env_name: str
    mode: int
    formatting: List[FormatterElement] | None
    font_properties: FontProperties | None
    save_levels: List[SaveLevel] = field(repr=False)
    # The formatting at the begin of the node. Only applies for boxes.
    formatting_begin: List[FormatterElement] | None = None
    # The formatting at the end of the node. Only applies for boxes.
    formatting_end: List[FormatterElement] | None = None
    replacement_character: str | None = None
    # These actions are triggered, when the node-level is entered. Only for vlists/hlists
    begin_actions: List[Callable[["Node", MemNode, int, int, StructuredOutput], None]] | None = None
    end_actions: List[Callable[["Node", MemNode, StructuredOutput], None]] | None = None
    # Direct formatting on node enter/exit
    direct_formatting: List[FormatterElement] | None = None
    is_hyphen: bool = False
    # If set, this node shall fail the page.
    page_fail: str | None = None


@dataclass
class BoxStackItem:
    type: str
    addr: int
    h: int
    v: int
    end_hooks: List[Callable] = field(default_factory=list)


class Mode(IntEnum):
    MATH = -3
    RESTRICTED_HORIZONTAL = -2
    INTERNAL_VERTICAL = -1
    NO_MODE = 0
    VERTICAL = 1
    HORIZONTAL = 2
    DISPLAY_MATH = 3


class GroupType(IntEnum):
    SIMPLE_GROUP = 1
    HBOX_GROUP = 2
    ADJUSTED_HBOX_GROUP = 3
    VBOX_GROUP = 4
    VTOP_GROUP = 5
    ALIGN_GROUP = 6
    NO_ALIGN_GROUP = 7
    OUTPUT_GROUP = 8
    MATH_GROUP = 9
    DISC_GROUP = 10
    INSERT_GROUP = 11
    VCENTER_GROUP = 12
    MATH_CHOICE_GROUP = 13
    SEMI_SIMPLE_GROUP = 14
    MATH_SHIFT_GROUP = 15
    MATH_LEFT_GROUP = 16


class CommandCode(IntEnum):
    relax = 0
    left_brace = 1
    right_brace = 2
    math_shift = 3
    tab_mark = 4
    car_ret = 5
    out_param = 5
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
    char_num = 16
    math_char_num = 17
    mark = 18
    xray = 19
    make_box = 20
    hmove = 21
    vmove = 22
    un_hbox = 23
    un_vbox = 24
    remove_item = 25
    hskip = 26
    vskip = 27
    mskip = 28
    kern = 29
    mkern = 30
    leader_ship = 31
    halign = 32
    valign = 33
    no_align = 34
    vrule = 35
    hrule = 36
    insert = 37
    vadjust = 38
    ignore_spaces = 39
    after_assignment = 40
    after_group = 41
    break_penalty = 42
    start_par = 43
    ital_corr = 44
    accent = 45
    math_accent = 46
    discretionary = 47
    eq_no = 48
    left_right = 49
    math_comp = 50
    limit_switch = 51
    above = 52
    math_style = 53
    math_choice = 54
    non_script = 55
    vcenter = 56
    case_shift = 57
    message = 58
    extension = 59
    in_stream = 60
    begin_group = 61
    end_group = 62
    omit = 63
    ex_space = 64
    no_boundary = 65
    radical = 66
    end_cs_name = 67
    char_given = 68
    min_internal = 68
    math_given = 69
    last_item = 70
    max_non_prefixed_command = 70
    toks_register = 71
    assign_toks = 72
    assign_int = 73
    assign_dimen = 74
    assign_glue = 75
    assign_mu_glue = 76
    assign_font_dimen = 77
    assign_font_int = 78
    set_aux = 79
    set_prev_graf = 80
    set_page_dimen = 81
    set_page_int = 82
    set_box_dimen = 83
    set_shape = 84
    def_code = 85
    def_family = 86
    set_font = 87
    def_font = 88
    register = 89
    max_internal = 89
    advance = 90
    multiply = 91
    divide = 92
    prefix = 93
    let = 94
    shorthand_def = 95
    read_to_cs = 96
    def_ = 97
    set_box = 98
    hyph_data = 99
    set_interaction = 100
    letterspace_font = 101
    pdf_copy_font = 102
    partoken_name = 103
    max_command = 103
    undefined_cs = max_command + 1
    expand_after = max_command + 2
    no_expand = max_command + 3
    input = max_command + 4
    if_test = max_command + 5
    fi_or_else = max_command + 6
    cs_name = max_command + 7
    convert = max_command + 8
    the = max_command + 9
    top_bot_mark = max_command + 10
    call = max_command + 11
    long_call = max_command + 12
    outer_call = max_command + 13
    long_outer_call = max_command + 14
    end_template = max_command + 15
    dont_expand = max_command + 16
    glue_ref = max_command + 17
    shape_ref = max_command + 18
    box_ref = max_command + 19
    data = max_command + 20


def get_mode(mode):
    if mode > 0:
        if mode // 104 == 0:
            return Mode.VERTICAL
        elif mode // 104 == 1:
            return Mode.HORIZONTAL
        elif mode // 104 == 2:
            return Mode.DISPLAY_MATH
    elif mode < 0:
        if -mode // 104 == 0:
            return Mode.INTERNAL_VERTICAL
        elif -mode // 104 == 1:
            return Mode.RESTRICTED_HORIZONTAL
        elif -mode // 104 == 2:
            return Mode.MATH
    return Mode.NO_MODE


def char(charcode: int):
    # Convert character code to a displayable string
    try:
        c = chr(charcode)
    except ValueError:
        return f"\\x{charcode:02X}"

    if c.isprintable():
        return c
    else:
        return f"\\x{charcode:02X}"


def cur_env() -> str:
    try:
        id, envir = id_lookup(b"@currenvir", True)
        if envir is not None:
            return str(envir)
        else:
            return "unknown"
    except NotImplementedError:
        return "unknown"


def get_node_from_box_register(box_register_name: bytes) -> MemNode | None:
    """Gets a node from a named box register (i.e. a macro retuning the register id)."""
    mem_reader = MemReader()

    cmd, currbox_register = id_lookup(box_register_name, True)
    while CommandCode(cmd) in (
        CommandCode.call,
        CommandCode.long_call,
        CommandCode.outer_call,
        CommandCode.long_outer_call,
    ):
        assert isinstance(currbox_register, int)
        tokens = list(mem_reader.get_token_list(currbox_register).tokens)
        assert len(tokens) == 2, ", ".join(str(t) for t in tokens)
        assert tokens[0].category == TokenCategory.end_match, (
            f"Expected end-match, got {str(tokens[0])}"
        )
        token = tokens[1]
        # print(f"  F: -> token: {token}")
        assert token.category == TokenCategory.cs_token, f"Expected cs name, got {str(token)}"
        cmd, currbox_register = get_cs_value(token.cs)
    assert CommandCode(cmd) == CommandCode.char_given, f"{CommandCode(cmd)!r} {currbox_register}"
    # print(f"F: currbox_register: 0x{currbox_register:06X}")
    assert isinstance(currbox_register, int)

    currbox_addr = get_box(currbox_register)
    # print(f"F: currbox_addr: 0x{currbox_addr:06X}")

    return mem_reader.get_node(currbox_addr)


_dimen_re = re.compile(
    rb"^(?P<value>\-?(?:(?:\d+)|(?:\d+[\.,]\d*)|(?:\d*[\.,]\d+)|(?:\"[A-Fa-f0-9]+)|(?:\'[0-8]+)|(?:`.)))(?P<unit>pt|em|ex|px)?$"
)


def _parse_number(text: bytes, allow_dimension: bool) -> int:
    m = _dimen_re.match(text)
    assert m is not None, f"Expected number, got {text!r}"
    value_str, unit_str = m.group("value", "unit")
    print(f"M: Parse {text} -> {value_str}, {unit_str}")
    assert unit_str is None or allow_dimension, f"Expected int, got {text!r}"
    if value_str.startswith(b"`"):
        assert len(value_str) == 2, f"Require 2 symbols for an alphanumeric character {value_str}"
        i_value = value_str[1]
        if unit_str is None:
            return i_value
        f_value = i_value
    elif b"," in value_str:
        value_str = value_str.replace(b",", b".")
        f_value = float(value_str)
        # expect dimension, must return int!
        assert unit_str is not None, f"Expected int, got {text!r}"
    elif b"." in value_str:
        f_value = float(value_str)
        # expect dimension, must return int!
        assert unit_str is not None, f"Expected int, got {text!r}"
    else:
        if value_str.startswith(b"'"):
            base = 8
            value_str = value_str[1:]
        elif value_str.startswith(b'"'):
            base = 16
            value_str = value_str[1:]
        else:
            base = 10
        i_value = int(value_str, base=base)
        if unit_str is None:
            return i_value
        f_value = i_value
    if unit_str == b"pt":
        v = UNITY
    elif unit_str == b"em":
        v = get_font_properties(get_cur_font())[5]
    elif unit_str == b"ex":
        v = get_font_properties(get_cur_font())[4]
    elif unit_str == b"px":
        v = get_font_properties(get_cur_font())[7]
    else:
        assert False, f"Unknown unit {unit_str!r}"

    return int(f_value * v)


def _get_int_value(
    tokens: list[Token], allow_dimension: bool = False, expected_code: CommandCode | None = None
) -> int:
    while True:
        if DEBUG_CS_RESOLVE:
            print(f"  F: -> tokens: {','.join(str(t) for t in tokens)}")
        if (
            len(tokens) == 2
            and tokens[0].category == TokenCategory.end_match
            and tokens[1].category == TokenCategory.cs_token
        ) or (len(tokens) == 1 and tokens[0].category == TokenCategory.cs_token):
            token = tokens[-1]
            if DEBUG_CS_RESOLVE:
                print(f"  F: -> token: {token}")
            cmd, value = get_cs_value(token.cs)
        elif (
            len(tokens) > 2
            and tokens[0].category == TokenCategory.cs_token
            and tokens[-1].category == TokenCategory.cs_token
            and get_cs(tokens[0].cs).tobytes() == b"csname"
            and get_cs(tokens[-1].cs).tobytes() == b"endcsname"
            and not any(
                t.category == TokenCategory.cs_token and get_cs(t.cs).tobytes() == b"endcsname"
                for t in tokens[1:-1]
            )
        ):
            name = b"".join(_get_token_text(tokens[1:-1]))
            if DEBUG_CS_RESOLVE:
                print(f"  F: -> cs: \\{name}")
            cmd, value = id_lookup(name, True)
        else:
            value = b"".join(_get_token_text(tokens))
            if DEBUG_CS_RESOLVE:
                print(f"  F: -> value: {value}")
            return _parse_number(value, allow_dimension=allow_dimension)
        if CommandCode(cmd) in (
            CommandCode.call,
            CommandCode.long_call,
            CommandCode.outer_call,
            CommandCode.long_outer_call,
        ):
            assert isinstance(value, int)
            tokens = [
                t
                for t in MemReader().get_token_list(value).tokens
                if t.category != TokenCategory.comment
            ]
            continue
        assert CommandCode(cmd) in (
            CommandCode.char_given,
            CommandCode.math_given,
            CommandCode.register,
            CommandCode.assign_dimen,
            CommandCode.assign_glue,
            CommandCode.assign_mu_glue,
            CommandCode.assign_int,
            CommandCode.set_font,
        ), f"cmd={CommandCode(cmd)!r} value={value}"
        assert expected_code is None or CommandCode(cmd) == expected_code, (
            f"cmd={CommandCode(cmd)!r} expected={expected_code!r} value={value}"
        )
        assert isinstance(value, int)
        return value


def get_int_value(
    value: str,
    mem_reader: MemReader | None = None,
    expected_code: CommandCode | None = None,
    allow_dimension: bool = True,
) -> int:
    """Gets a node from a named box register (i.e. a macro retuning the register id)."""
    if mem_reader is None:
        mem_reader = MemReader()

    if value.startswith("\\"):
        try:
            cmd, value = id_lookup(value[1:].encode(), True)
            # print(f"  F: -> cmd: {CommandCode(cmd)!r} value: {value}")
        except KeyError as e:
            raise ValueError(f"Unknown control sequence {value!r}") from e
        if CommandCode(cmd) in (
            CommandCode.call,
            CommandCode.long_call,
            CommandCode.outer_call,
            CommandCode.long_outer_call,
        ):
            assert isinstance(value, int)
            tokens = [
                t
                for t in MemReader().get_token_list(value).tokens
                if t.category != TokenCategory.comment
            ]
            return _get_int_value(
                tokens, allow_dimension=allow_dimension, expected_code=expected_code
            )

        assert CommandCode(cmd) in (
            CommandCode.char_given,
            CommandCode.math_given,
            CommandCode.register,
            CommandCode.assign_dimen,
            CommandCode.assign_glue,
            CommandCode.assign_mu_glue,
            CommandCode.assign_int,
            CommandCode.set_font,
        ), f"cmd={CommandCode(cmd)!r} value={value}"
        assert expected_code is None or CommandCode(cmd) == expected_code, (
            f"cmd={CommandCode(cmd)!r} expected={expected_code!r} value={value}"
        )
        assert isinstance(value, int)
        return value
    else:
        if DEBUG_CS_RESOLVE:
            print(f"  F: -> value: {value}")
        try:
            return int(value)
        except ValueError:
            assert False, f"Expected int, got {value!r}"


def _get_token_text(call: int | list[Token], keep_cs: Container[bytes] = ()) -> list[bytes]:
    if isinstance(call, int):
        tokens = [
            t
            for t in MemReader().get_token_list(call).tokens
            if t.category != TokenCategory.comment
        ]
    else:
        tokens = call
    # print(f"  F: -> tokens: " + ", ".join(f"{t} [{t.category!r}]" for t in tokens))
    text = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.category in (TokenCategory.other_char, TokenCategory.letter):
            text.append(bytes((token.char,)))
        elif token.category == TokenCategory.cs_token:
            cs = get_cs(token.cs).tobytes()
            if cs == b"csname":
                start_i = i
                while i < len(tokens):
                    if (
                        tokens[i].category == TokenCategory.cs_token
                        and get_cs(tokens[i].cs).tobytes() == b"endcsname"
                    ):
                        break
                    i += 1
                else:
                    assert False, "Missing endcsname"
                cs = b"".join(_get_token_text(tokens[start_i + 1 : i]))
                # print(f"  F: -> cs: {cs!r}")
            if cs in keep_cs:
                text.append(b"\\" + cs)
            else:
                cmd, value = id_lookup(cs, True)
                assert CommandCode(cmd) in (
                    CommandCode.call,
                    CommandCode.long_call,
                    CommandCode.outer_call,
                    CommandCode.long_outer_call,
                ), f"{CommandCode(cmd)!r} {value} from {cs}"
                assert isinstance(value, int)
                text.extend(_get_token_text(value))
        i += 1
    # print(f"  F: -> text: " + repr(b"".join(text)))
    return text


def get_tokens(value: str, mem_reader: MemReader | None = None) -> TokenList:
    """Gets a token list named register."""
    if mem_reader is None:
        mem_reader = MemReader()
    assert value.startswith("\\"), f"Expected control sequence, got {value!r}"
    try:
        cmd, value = id_lookup(value[1:].encode(), True)
    except KeyError as e:
        raise ValueError(f"Unknown control sequence {value!r}") from e
    assert CommandCode(cmd) == CommandCode.assign_toks, f"{CommandCode(cmd)!r}"
    assert isinstance(value, int)
    return mem_reader.get_token_list(value)


def get_text(value: str, keep_cs: Container[bytes] = ()) -> str:
    """Get the text value of the given control sequence."""
    if value.startswith("\\"):
        try:
            cmd, value = id_lookup(value[1:].encode(), True)
        except KeyError as e:
            raise ValueError(f"Unknown control sequence {value!r}") from e
        assert CommandCode(cmd) in (
            CommandCode.call,
            CommandCode.long_call,
            CommandCode.outer_call,
            CommandCode.long_outer_call,
        ), f"{CommandCode(cmd)!r}"
        assert isinstance(value, int)
        return b"".join(_get_token_text(value, keep_cs)).decode(errors="backslashreplace")
    return value


def get_arg_text(arg: int, keep_cs: Container[bytes] = ()) -> str:
    """Get the text value of the given cs call argument."""
    value = _get_token_text(list(MemReader().get_norefcount_token_list(arg).tokens), keep_cs)
    return b"".join(value).decode(errors="backslashreplace")


def get_arg_cs(arg: int) -> str:
    """Get the control sequence name of the given cs call argument. Expects that the argument is a
    single cs."""
    tokens = [
        t
        for t in MemReader().get_norefcount_token_list(arg).tokens
        if t.category != TokenCategory.comment
    ]
    assert len(tokens) > 0, f"Expected cs, got {tokens}"
    if len(tokens) == 1:
        assert tokens[0].category == TokenCategory.cs_token, f"Expected cs, got {tokens[0]}"
        return get_cs(tokens[0].cs).tobytes().decode(errors="backslashreplace")
    assert tokens[0].category == TokenCategory.cs_token, f"Expected cs, got {tokens}"
    assert tokens[-1].category == TokenCategory.cs_token, f"Expected cs, got {tokens}"
    assert get_cs(tokens[0].cs).tobytes() == b"csname", f"Expected csname, got {tokens}"
    assert get_cs(tokens[-1].cs).tobytes() == b"endcsname", f"Expected endcsname, got {tokens}"
    return b"".join(_get_token_text(tokens[1:-1])).decode(errors="backslashreplace")


def get_arg_int(
    arg: int, allow_dimension: bool = True, expected_code: CommandCode | None = None
) -> int:
    """Get the integer argument of the given cs call argument."""
    tokens = [
        t
        for t in MemReader().get_norefcount_token_list(arg).tokens
        if t.category != TokenCategory.comment
    ]
    return _get_int_value(tokens, allow_dimension=allow_dimension, expected_code=expected_code)


def get_radical_symbol(name: bytes) -> int:
    """Resolves DeclareMathRadical{\\name}"""
    cmd, value = id_lookup(name, True)
    assert CommandCode(cmd) in (
        CommandCode.call,
        CommandCode.long_call,
        CommandCode.outer_call,
        CommandCode.long_outer_call,
    )
    assert isinstance(value, int)
    tokens = [
        t for t in MemReader().get_token_list(value).tokens if t.category != TokenCategory.comment
    ]
    tokens = _get_token_text(tokens, keep_cs=(b"radical", b"relax"))
    assert tokens[0] == b"\\radical"
    while b"\\relax" in tokens:
        tokens.remove(b"\\relax")
    code_b = b"".join(tokens[1:])
    return _parse_number(code_b, allow_dimension=False)


# Fetch the id of the \protect command
_protect = id_lookup("protect", False)


def get_command_tokens(value: str) -> TokenList:
    """Get the value tokens of the given control sequence."""
    assert value.startswith("\\"), f"Expected control sequence, got {value!r}"
    cmd, value = id_lookup(value[1:].encode(), True)
    assert cmd in (
        CommandCode.call,
        CommandCode.long_call,
        CommandCode.outer_call,
        CommandCode.long_outer_call,
    ), f"{CommandCode(cmd)!r}"
    assert isinstance(value, int)
    return MemReader().get_token_list(value)


def is_protected(value: str) -> bool:
    """Check if a control sequence is protected."""
    # TODO: This is not super efficient. Could restrict to the first few tokens?
    tokens = list(get_command_tokens(value).tokens)
    return any(
        token.category == TokenCategory.cs_token and token.cs == _protect for token in tokens
    )
