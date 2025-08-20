# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import random
import traceback
from abc import ABC
from dataclasses import dataclass
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    List,
    Literal,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

if TYPE_CHECKING:
    from nvtexpy.engine import rules

from pdftex import get_cur_mode

from nvtexpy.engine.common import CommandCode, Node, cur_env
from nvtexpy.engine.debug import DEBUG_ASSERT_FAIL, DebugHex
from nvtexpy.engine.hooks import register_macro_call
from nvtexpy.engine.mem import CharNode, GlueNode, GlueSubtype, HBoxNode
from nvtexpy.engine.mem import Node as MemNode
from nvtexpy.engine.mem import RuleNode, VBoxNode
from nvtexpy.engine.stack_formatter import (
    DirectBeginFormatterWrapper,
    DirectEndFormatterWrapper,
    FormatterElement,
)
from nvtexpy.engine.texstate import TexState
from nvtexpy.output.structured_output import StructuredOutput


def assert2pagefail(method):
    assert not isinstance(method, _RuleDecoratorMacroHook), (
        "assert2pagefail must be applied before *_macro_hook"
    )

    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        try:
            return method(self, *method_args, **method_kwargs)
        except AssertionError:
            if DEBUG_ASSERT_FAIL:
                raise
            # Get the strack trace as string
            self.mark_page_as_failed(traceback.format_exc())

    return _impl


@dataclass
class _RuleDecoratorMacroHook:
    """Helper struct for @begin_macro_hook and @end_macro_hook."""

    macro_names: List[str]
    is_begin: bool
    method: Callable
    raw_args: bool
    str_args: bool

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.method is not None:
            return self.method(*args, **kwargs)


TSelf = TypeVar("TSelf", bound="Rule")


@overload
def begin_macro_hook(
    macro_name: str | list[str],
    *,
    str_args: Literal[False] = False,
    raw_args: Literal[False] = False,
) -> Callable[
    [
        Callable[
            [TSelf, str, CommandCode, None, int, int],
            Callable[[int, int], None] | None,
        ]
    ],
    Callable[
        [TSelf, str, CommandCode, None, int, int],
        Callable[[int, int], None] | None,
    ],
]: ...


@overload
def begin_macro_hook(
    macro_name: str | list[str], *, str_args: Literal[True], raw_args: Literal[False] = False
) -> Callable[
    [
        Callable[
            [TSelf, str, CommandCode, Tuple[str | None, ...] | None, int, int],
            Callable[[int, int], None] | None,
        ]
    ],
    Callable[
        [TSelf, str, CommandCode, Tuple[str | None, ...] | None, int, int],
        Callable[[int, int], None] | None,
    ],
]: ...


@overload
def begin_macro_hook(
    macro_name: str | list[str], *, str_args: Literal[False] = False, raw_args: Literal[True]
) -> Callable[
    [
        Callable[
            [TSelf, str, CommandCode, Tuple[int, ...] | None, int, int],
            Callable[[int, int], None] | None,
        ]
    ],
    Callable[
        [TSelf, str, CommandCode, Tuple[int, ...] | None, int, int],
        Callable[[int, int], None] | None,
    ],
]: ...


# Invalid case. To show the user some typing error.
@overload
def begin_macro_hook(
    macro_name: str | list[str], *, str_args: Literal[True] = True, raw_args: Literal[True]
) -> None: ...


def begin_macro_hook(
    macro_name: str | list[str], *, str_args: bool = False, raw_args: bool = False
):
    """Decorator to register a macro hook for a rule."""
    if isinstance(macro_name, str):
        macro_name = [macro_name]
    assert isinstance(macro_name, list)

    def decorator(
        func: (
            Callable[
                [TSelf, str, CommandCode, Tuple[str | None, ...] | None, int, int],
                Callable[[int, int], None] | None,
            ]
            | Callable[
                [TSelf, str, CommandCode, Tuple[int, ...] | None, int, int],
                Callable[[int, int], None] | None,
            ]
            | Callable[
                [TSelf, str, CommandCode, None, int, int],
                Callable[[int, int], None] | None,
            ]
        ),
    ):
        if isinstance(func, _RuleDecoratorMacroHook):
            # This is a multi-decorated function
            func.macro_names.extend(macro_name)
            return func
        return functools.wraps(func)(
            _RuleDecoratorMacroHook(
                macro_names=macro_name,
                is_begin=True,
                method=func,
                str_args=str_args,
                raw_args=raw_args,
            )
        )

    return decorator


@overload
def end_macro_hook(
    macro_name: str | list[str], *, str_args: Literal[False] = False
) -> Callable[
    [
        Callable[
            [TSelf, str, CommandCode, None, int, int],
            Callable[[int, int], None] | None,
        ]
    ],
    Callable[
        [TSelf, str, CommandCode, None, int, int],
        Callable[[int, int], None] | None,
    ],
]: ...


@overload
def end_macro_hook(
    macro_name: str | list[str], *, str_args: Literal[True]
) -> Callable[
    [
        Callable[
            [TSelf, str, CommandCode, Tuple[str | None, ...] | None, int, int],
            Callable[[int, int], None] | None,
        ]
    ],
    Callable[
        [TSelf, str, CommandCode, Tuple[str | None, ...] | None, int, int],
        Callable[[int, int], None] | None,
    ],
]: ...


def end_macro_hook(macro_name: str | list[str], *, str_args: bool = False):
    """Decorator to register a macro hook for a rule."""
    if isinstance(macro_name, str):
        macro_name = [macro_name]
    assert isinstance(macro_name, list)

    def decorator(
        func: (
            Callable[[TSelf, str, CommandCode, Tuple[str | None, ...] | None, int, int], None]
            | Callable[[TSelf, str, CommandCode, None, int, int], None]
        ),
    ):
        if isinstance(func, _RuleDecoratorMacroHook):
            # This is a multi-decorated function
            func.macro_names.extend(macro_name)
            return func
        return functools.wraps(func)(
            _RuleDecoratorMacroHook(
                macro_names=macro_name,
                is_begin=False,
                method=func,
                raw_args=False,
                str_args=str_args,
            )
        )

    return decorator


TRule = TypeVar("TRule", bound="Rule")
TFormatterElement = TypeVar("TFormatterElement", bound=FormatterElement)


def rule_ref(rule_type: Type[TRule] | str) -> TRule:
    """Decorator to register a macro hook for a rule."""
    if isinstance(rule_type, str):
        prop_name = f"_rule_{rule_type}"
    else:
        prop_name = f"_rule_{rule_type.__name__}"

    def _get(self):
        if not hasattr(self, prop_name):
            setattr(self, prop_name, self.rules.get_rule(rule_type))
        return getattr(self, prop_name)

    return cast(TRule, property(_get))


def capture_stack_element(macro_name: str | list[str]) -> property:
    """Decorator to capture a stack element for a rule."""
    _macro_state = 0

    def begin(*args):
        nonlocal _macro_state
        _macro_state += 1

    def end(*args):
        nonlocal _macro_state
        _macro_state -= 1

    if isinstance(macro_name, str):
        register_macro_call(macro_name, begin, end)
    else:
        assert isinstance(macro_name, list)
        for name in macro_name:
            register_macro_call(name, begin, end)

    return property(lambda self: _macro_state)


class Rule(ABC):
    """Abtract base class for a rule that is triggered by pdftex events and
    describes how to generate markdown elements.
    """

    active: bool = False  # Whether this rule is currently active

    font_rule: "rules.FontRule" = rule_ref("FontRule")
    box_tracker: "rules.BoxTrackRule" = rule_ref("BoxTrackRule")

    def __init__(self, state: TexState, rules: "Rules") -> None:
        self.state = state
        self.active = False
        self.formatter = None
        self.rules = rules

        # Register all rule hooks
        for key in dir(type(self)):
            val = getattr(type(self), key)
            if isinstance(val, _RuleDecoratorMacroHook):
                print("Register macro", key, val.macro_names, val)
                for macro_name in val.macro_names:
                    if val.is_begin:
                        register_macro_call(
                            macro_name,
                            functools.partial(val, self),
                            None,
                            val.str_args,
                            val.raw_args,
                        )
                    else:
                        register_macro_call(
                            macro_name,
                            None,
                            functools.partial(val, self),
                            val.str_args,
                            val.raw_args,
                        )

    def mark_page_as_failed(self, error_msg: str):
        if self.state.current_page_failed is None:
            self.state.current_page_failed = ""

        self.state.current_page_failed += "=" * 20 + "\n\n" + error_msg

    def push_formatter(self, formatter: FormatterElement):
        self.state.push_input_formatter_stack(formatter)

    def _get_node(self, node_addr: int, make_new: bool) -> Node:
        node = self.state.addr_to_node.get(node_addr)
        if node is None:
            self.state.addr_to_node[node_addr] = node = Node(
                rand_id=random.randint(0, 1000000),
                orig_addr=DebugHex(node_addr),
                dbg_char="",
                character=0,
                font=0,
                font_properties=None,
                env_name=cur_env(),
                save_levels=list(self.state.save_levels),
                mode=get_cur_mode(),
                formatting=None,
            )
        elif make_new and node.orig_addr != node_addr:
            self.state.addr_to_node[node_addr] = node = Node(
                rand_id=random.randint(0, 1000000),
                orig_addr=DebugHex(node_addr),
                dbg_char=node.dbg_char,
                character=node.character,
                font=node.font,
                font_properties=node.font_properties,
                env_name=node.env_name,
                save_levels=node.save_levels,
                mode=node.mode,
                formatting=[],
            )
        return node

    def get_node_formatter(
        self,
        node_addr: int,
        format_type: Union[Type[TFormatterElement], Tuple[Type[TFormatterElement], ...]],
    ) -> TFormatterElement | None:
        node = self.state.addr_to_node.get(node_addr)
        if node is None:
            return None
        if node.formatting is not None:
            # Check node stack formatting
            for formatter in node.formatting:
                if isinstance(formatter, format_type):
                    return formatter
        if node.direct_formatting is not None:
            # Check direct node formatting
            for formatter in node.direct_formatting:
                if isinstance(formatter, (DirectBeginFormatterWrapper, DirectEndFormatterWrapper)):
                    formatter = formatter.actual_formatter
                if isinstance(formatter, format_type):
                    return formatter
        if node.formatting_begin is not None:
            # Also check the formatting stack, if stored
            for formatter in reversed(node.formatting_begin):
                if isinstance(formatter, (DirectBeginFormatterWrapper, DirectEndFormatterWrapper)):
                    formatter = formatter.actual_formatter
                if isinstance(formatter, format_type):
                    return formatter
        if node.formatting_end is not None:
            # Also check the formatting stack, if stored
            for formatter in reversed(node.formatting_end):
                if isinstance(formatter, (DirectBeginFormatterWrapper, DirectEndFormatterWrapper)):
                    formatter = formatter.actual_formatter
                if isinstance(formatter, format_type):
                    return formatter
        return None

    def dump_stack(self) -> List[FormatterElement]:
        return self.state.get_input_formatter_stack()

    def register_node_action(
        self,
        node_addr: int,
        node_end_addr: int | None = None,
        *,
        begin_action: Callable[[Node, MemNode, int, int, StructuredOutput], None] | None = None,
        end_action: Callable[[Node, MemNode, StructuredOutput], None] | None = None,
        direct_formatter: FormatterElement | None = None,
        make_new: bool = True,
        dump_stack: bool = False,
        begin_stack: List[FormatterElement] | None = None,
    ):
        node_begin = self._get_node(node_addr, make_new)
        if node_end_addr is not None and node_end_addr != node_addr:
            node_end = self._get_node(node_end_addr, make_new)
        else:
            node_end = node_begin
        if begin_action is not None:
            if node_begin.begin_actions is None:
                node_begin.begin_actions = [begin_action]
            else:
                node_begin.begin_actions.append(begin_action)
        if end_action is not None:
            if node_end.end_actions is None:
                node_end.end_actions = [end_action]
            else:
                node_end.end_actions.append(end_action)
        if direct_formatter is not None:
            if node_end == node_begin or isinstance(direct_formatter, DirectBeginFormatterWrapper):
                if node_begin.direct_formatting is None:
                    node_begin.direct_formatting = [direct_formatter]
                else:
                    node_begin.direct_formatting.append(direct_formatter)
            elif isinstance(direct_formatter, DirectEndFormatterWrapper):
                if node_end.direct_formatting is None:
                    node_end.direct_formatting = [direct_formatter]
                else:
                    node_end.direct_formatting.append(direct_formatter)
            else:
                if node_begin.direct_formatting is None:
                    node_begin.direct_formatting = [DirectBeginFormatterWrapper(direct_formatter)]
                else:
                    node_begin.direct_formatting.append(
                        DirectBeginFormatterWrapper(direct_formatter)
                    )
                if node_end.direct_formatting is None:
                    node_end.direct_formatting = [DirectEndFormatterWrapper(direct_formatter)]
                else:
                    node_end.direct_formatting.append(DirectEndFormatterWrapper(direct_formatter))
        if begin_stack is not None:
            if node_begin.formatting_begin is not None:
                if len(node_begin.formatting_begin) < len(begin_stack):
                    assert (
                        node_begin.formatting_begin
                        == begin_stack[: len(node_begin.formatting_begin)]
                    ), (
                        f"Formatter stack mismatch: {node_begin.formatting_begin} is not a subset of {begin_stack}"
                    )
                    node_begin.formatting_begin = begin_stack
                else:
                    assert node_begin.formatting_begin[: len(begin_stack)] == begin_stack, (
                        f"Formatter stack mismatch: {node_begin.formatting_begin} is not a superset of {begin_stack}"
                    )
            node_begin.formatting_begin = begin_stack
        if dump_stack:
            node_begin.formatting_end = self.state.get_input_formatter_stack()
            if node_end != node_begin:
                node_end.formatting_end = self.state.get_input_formatter_stack()

    @assert2pagefail
    def pop_formatter(self, formatter: FormatterElement):
        return self.state.pop_input_formatter_stack(formatter)

    def on_char_alloc(self, node: Node):
        """Override to trigger on char_alloc events."""
        raise NotImplementedError("Only overridden methods should be called")

    def on_glue_alloc(self, node: Node, glue_type: GlueSubtype):
        """Override to trigger on glue_alloc events."""
        raise NotImplementedError("Only overridden methods should be called")

    def on_node_alloc(self, addr: int, size: int):
        """Override to trigger on node_alloc events."""
        raise NotImplementedError("Only overridden methods should be called")

    def on_node_dealloc(self, addr: int, size: int):
        """Override to trigger on node_alloc events."""
        raise NotImplementedError("Only overridden methods should be called")

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
        """Override to trigger on char_out events."""
        raise NotImplementedError("Only overridden methods should be called")

    def on_glue_out(
        self, node: Node, memnode: GlueNode, cur_h: int, cur_v: int, width: int, height: int
    ):
        """Override to trigger on char_out events."""
        raise NotImplementedError("Only overridden methods should be called")

    def on_rule_out(
        self, node: Node | None, memnode: RuleNode, cur_h: int, cur_v: int, width: int, height: int
    ):
        """Override to trigger on char_out events."""
        raise NotImplementedError("Only overridden methods should be called")

    def on_hlist_out_start(self, memnode: HBoxNode, cur_h: int, cur_v: int):
        """Override to trigger on hlist_out_start events."""
        raise NotImplementedError("Only overridden methods should be called")

    def on_hlist_out_end(self, memnode: HBoxNode):
        """Override to trigger on hlist_out_end events."""
        raise NotImplementedError("Only overridden methods should be called")

    def on_hlist_out_node(self, memnode: MemNode, cur_h: int, cur_v: int):
        """Override to trigger on hlist_out_node events."""
        raise NotImplementedError("Only overridden methods should be called")

    def on_vlist_out_start(self, memnode: VBoxNode, cur_h: int, cur_v: int):
        """Override to trigger on vlist_out_start events."""
        raise NotImplementedError("Only overridden methods should be called")

    def on_vlist_out_end(self, memnode: VBoxNode):
        """Override to trigger on vlist_out_end events."""
        raise NotImplementedError("Only overridden methods should be called")

    def on_vlist_out_node(self, memnode: MemNode, cur_h: int, cur_v: int):
        """Override to trigger on vlist_out_node events."""
        raise NotImplementedError("Only overridden methods should be called")

    def on_copy(self, src_node_addr: int, dst_node_addr: int):
        """Override to trigger on copy events."""
        raise NotImplementedError("Only overridden methods should be called")

    def on_sheet_begin(
        self,
        memnode: VBoxNode | HBoxNode | None,
        page: int,
        page_width: int,
        page_height: int,
        struct_out: StructuredOutput,
    ):
        """Override to trigger on sheet_begin events."""
        raise NotImplementedError("Only overridden methods should be called")

    def on_sheet_end(self):
        """Override to trigger on sheet_end events."""
        raise NotImplementedError("Only overridden methods should be called")


class Rules:
    _subclasses: ClassVar[List[Type["Rule"]]] = []  # All rules that have been registered

    rules: List["Rule"]

    @staticmethod
    def register(subclass: Type["TRule"]) -> Type["TRule"]:
        """Decorator to register a subclass of Rule."""
        Rules._subclasses.append(subclass)
        return subclass

    def __init__(self, state: TexState) -> None:
        self.rules = [cls(state, self) for cls in Rules._subclasses]

        self._rules_on_char_alloc = [
            rule for rule in self.rules if type(rule).on_char_alloc != Rule.on_char_alloc
        ]
        self._rules_on_glue_alloc = [
            rule for rule in self.rules if type(rule).on_glue_alloc != Rule.on_glue_alloc
        ]
        self._rules_on_node_alloc = [
            rule for rule in self.rules if type(rule).on_node_alloc != Rule.on_node_alloc
        ]
        self._rules_on_node_dealloc = [
            rule for rule in self.rules if type(rule).on_node_dealloc != Rule.on_node_dealloc
        ]
        self._rules_on_char_out = [
            rule for rule in self.rules if type(rule).on_char_out != Rule.on_char_out
        ]
        self._rules_on_glue_out = [
            rule for rule in self.rules if type(rule).on_glue_out != Rule.on_glue_out
        ]
        self._rules_on_rule_out = [
            rule for rule in self.rules if type(rule).on_rule_out != Rule.on_rule_out
        ]
        self._rules_on_hlist_out_node = [
            rule for rule in self.rules if type(rule).on_hlist_out_node != Rule.on_hlist_out_node
        ]
        self._rules_on_hlist_out_start = [
            rule for rule in self.rules if type(rule).on_hlist_out_start != Rule.on_hlist_out_start
        ]
        self._rules_on_hlist_out_end = [
            rule for rule in self.rules if type(rule).on_hlist_out_end != Rule.on_hlist_out_end
        ]
        self._rules_on_vlist_out_node = [
            rule for rule in self.rules if type(rule).on_vlist_out_node != Rule.on_vlist_out_node
        ]
        self._rules_on_vlist_out_start = [
            rule for rule in self.rules if type(rule).on_vlist_out_start != Rule.on_vlist_out_start
        ]
        self._rules_on_vlist_out_end = [
            rule for rule in self.rules if type(rule).on_vlist_out_end != Rule.on_vlist_out_end
        ]
        self._rules_on_copy = [rule for rule in self.rules if type(rule).on_copy != Rule.on_copy]
        self._rules_on_sheet_begin = [
            rule for rule in self.rules if type(rule).on_sheet_begin != Rule.on_sheet_begin
        ]
        self._rules_on_sheet_end = [
            rule for rule in self.rules if type(rule).on_sheet_end != Rule.on_sheet_end
        ]

    def call_char_alloc(self, node: Node):
        for subclass in self._rules_on_char_alloc:
            subclass.on_char_alloc(node)

    def call_glue_alloc(self, node: Node, glue_type: GlueSubtype):
        for subclass in self._rules_on_glue_alloc:
            subclass.on_glue_alloc(node, glue_type)

    def call_node_alloc(self, addr: int, size: int):
        for subclass in self._rules_on_node_alloc:
            subclass.on_node_alloc(addr, size)

    def call_node_dealloc(self, addr: int, size: int):
        for subclass in self._rules_on_node_dealloc:
            subclass.on_node_dealloc(addr, size)

    def call_char_out(
        self,
        node: Node,
        memnode: CharNode,
        cur_h: int,
        cur_v: int,
        width: int,
        height: int,
        depth: int,
    ):
        for subclass in self._rules_on_char_out:
            subclass.on_char_out(node, memnode, cur_h, cur_v, width, height, depth)

    def call_glue_out(
        self, node: Node, memnode: GlueNode, cur_h: int, cur_v: int, width: int, height: int
    ):
        for subclass in self._rules_on_glue_out:
            subclass.on_glue_out(node, memnode, cur_h, cur_v, width, height)

    def call_rule_out(
        self, node: Node | None, memnode: RuleNode, cur_h: int, cur_v: int, width: int, height: int
    ):
        for subclass in self._rules_on_rule_out:
            subclass.on_rule_out(node, memnode, cur_h, cur_v, width, height)

    def call_hlist_out_node(self, memnode: MemNode, cur_h: int, cur_v: int):
        for subclass in self._rules_on_hlist_out_node:
            subclass.on_hlist_out_node(memnode, cur_h, cur_v)

    def call_hlist_out_start(self, memnode: HBoxNode, cur_h: int, cur_v: int):
        for subclass in self._rules_on_hlist_out_start:
            subclass.on_hlist_out_start(memnode, cur_h, cur_v)

    def call_hlist_out_end(self, memnode: HBoxNode):
        for subclass in self._rules_on_hlist_out_end:
            subclass.on_hlist_out_end(memnode)

    def call_vlist_out_node(self, memnode: MemNode, cur_h: int, cur_v: int):
        for subclass in self._rules_on_vlist_out_node:
            subclass.on_vlist_out_node(memnode, cur_h, cur_v)

    def call_vlist_out_start(self, memnode: VBoxNode, cur_h: int, cur_v: int):
        for subclass in self._rules_on_vlist_out_start:
            subclass.on_vlist_out_start(memnode, cur_h, cur_v)

    def call_vlist_out_end(self, memnode: VBoxNode):
        for subclass in self._rules_on_vlist_out_end:
            subclass.on_vlist_out_end(memnode)

    def call_copy(
        self,
        src_node_addr: int,
        dst_node_addr: int,
    ):
        for subclass in self._rules_on_copy:
            subclass.on_copy(src_node_addr, dst_node_addr)

    def call_sheet_begin(
        self,
        memnode: VBoxNode | HBoxNode | None,
        page: int,
        page_width: int,
        page_height: int,
        struct_out: StructuredOutput,
    ):
        for subclass in self._rules_on_sheet_begin:
            subclass.on_sheet_begin(memnode, page, page_width, page_height, struct_out)

    def call_sheet_end(self):
        for subclass in self._rules_on_sheet_end:
            subclass.on_sheet_end()

    def get_rule(self, rule_type: Type[TRule] | str) -> TRule:
        if isinstance(rule_type, str):
            for rule in self.rules:
                if type(rule).__name__ == rule_type:
                    return rule
        else:
            for rule in self.rules:
                if isinstance(rule, rule_type):
                    return rule
        raise ValueError(f"Rule {rule_type} not found")
