# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from nvtexpy.engine import common
    from nvtexpy.engine.texstate import TexState

from nvtexpy.engine.mem import CharNode, GlueNode, HBoxNode
from nvtexpy.engine.mem import Node as MemNode
from nvtexpy.engine.mem import RuleNode, VBoxNode
from nvtexpy.output.structured_output import (
    Element,
    HeadingElement,
    StructuredOutput,
    SuppressElement,
)


class FormatterElement(ABC):
    state: "TexState"
    suppress_formats: bool = False
    format_whatsit_nodes: bool = False
    stack_unordered: bool = False

    is_on_stack: bool = False

    struct_out: StructuredOutput

    def begin(self, struct_out: StructuredOutput):
        self.struct_out = struct_out

    def end(self, struct_out: StructuredOutput):
        pass

    def on_char_out(
        self,
        node: "common.Node",
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
        self,
        node: "common.Node",
        memnode: GlueNode,
        cur_h: int,
        cur_v: int,
        width: int,
        height: int,
    ):
        """Override to trigger on glue_out events."""
        raise NotImplementedError("Only overridden methods should be called")

    def on_rule_out(
        self,
        node: "common.Node | None",
        memnode: RuleNode,
        cur_h: int,
        cur_v: int,
        width: int,
        height: int,
    ):
        """Override to trigger on rule_out events."""
        raise NotImplementedError("Only overridden methods should be called")

    def on_hlist_out_start(
        self, node: "common.Node | None", memnode: HBoxNode, cur_h: int, cur_v: int
    ):
        """Override to trigger on hlist_out_start events."""
        raise NotImplementedError("Only overridden methods should be called")

    def on_hlist_out_end(self, node: "common.Node | None", memnode: HBoxNode):
        """Override to trigger on hlist_out_end events."""
        raise NotImplementedError("Only overridden methods should be called")

    def on_hlist_out_node(
        self, node: "common.Node | None", memnode: MemNode, cur_h: int, cur_v: int
    ):
        """Override to trigger on hlist_out_node events."""
        raise NotImplementedError("Only overridden methods should be called")

    def on_vlist_out_start(
        self, node: "common.Node | None", memnode: VBoxNode, cur_h: int, cur_v: int
    ):
        """Override to trigger on vlist_out_start events."""
        raise NotImplementedError("Only overridden methods should be called")

    def on_vlist_out_end(self, node: "common.Node | None", memnode: VBoxNode):
        """Override to trigger on vlist_out_end events."""
        raise NotImplementedError("Only overridden methods should be called")

    def on_vlist_out_node(
        self, node: "common.Node | None", memnode: MemNode, cur_h: int, cur_v: int
    ):
        """Override to trigger on vlist_out_node events."""
        raise NotImplementedError("Only overridden methods should be called")


class DirectBeginFormatterWrapper(FormatterElement):
    def __init__(self, actual_formatter: FormatterElement) -> None:
        super().__init__()
        self.actual_formatter = actual_formatter

    def begin(self, struct_out: StructuredOutput):
        super().begin(struct_out)
        self.actual_formatter.begin(struct_out)

    def __repr__(self) -> str:
        return f"DirectBeginFormatterWrapper({self.actual_formatter})"


class DirectEndFormatterWrapper(FormatterElement):
    def __init__(self, actual_formatter: FormatterElement) -> None:
        super().__init__()
        self.actual_formatter = actual_formatter

    def end(self, struct_out: StructuredOutput):
        super().end(struct_out)
        self.actual_formatter.end(struct_out)

    def __repr__(self) -> str:
        return f"DirectEndFormatterWrapper({self.actual_formatter})"


class SectionFormatter(FormatterElement):
    def __init__(self, level: int) -> None:
        self.level = level
        self.my_heading = None

    def begin(self, struct_out: StructuredOutput):
        super().begin(struct_out)
        self.my_heading = struct_out.push(HeadingElement(level=self.level))

    def end(self, struct_out: StructuredOutput):
        super().end(struct_out)
        assert self.my_heading is not None
        struct_out.pop(self.my_heading)

    def __repr__(self) -> str:
        return f"SectionFormatter(level={self.level})"


class CallbackFormatter(FormatterElement):
    _element: Element | None = None

    def __init__(
        self,
        begin: Callable[[StructuredOutput], Element],
        end: Callable[[StructuredOutput, Element], Any] = lambda stack, el: stack.pop(el),
        name: Optional[str] = None,
    ) -> None:
        self._begin = begin
        self._end = end
        self.name = name

    def begin(self, struct_out: StructuredOutput):
        super().begin(struct_out)
        assert self._element is None
        self._element = self._begin(struct_out)
        self._struct_out = struct_out

    def end(self, struct_out: StructuredOutput):
        super().end(struct_out)
        assert self._element is not None
        self._end(struct_out, self._element)
        self._element = None
        self._struct_out = None

    def __repr__(self) -> str:
        if self.name is None:
            return f"{type(self).__name__}(begin={self._begin.__qualname__}, end={self._end.__qualname__})"
        else:
            return f"{type(self).__name__}(name={self.name}, begin={self._begin.__qualname__}, end={self._end.__qualname__})"


class SuppressFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            name="suppress", begin=lambda struct_out: struct_out.push(SuppressElement())
        )


class PagefailFormatter(FormatterElement):
    def __init__(self, error_message: str) -> None:
        self.error_message = error_message

    def begin(self, struct_out: StructuredOutput):
        if self.state.current_page_failed is None:
            self.state.current_page_failed = self.error_message
        elif self.error_message not in self.state.current_page_failed:
            self.state.current_page_failed += f"\n{self.error_message}"

    def end(self, struct_out: StructuredOutput):
        super().end(struct_out)

    def __repr__(self) -> str:
        return f"PagefailFormatter(error_message={self.error_message})"
