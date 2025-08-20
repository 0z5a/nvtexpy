# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import List

from nvtexpy.engine.common import Node, SaveLevel
from nvtexpy.engine.debug import DEBUG_LIST
from nvtexpy.engine.rules.base import (
    Rule,
    Rules,
    assert2pagefail,
    begin_macro_hook,
    capture_stack_element,
)
from nvtexpy.engine.stack_formatter import FormatterElement
from nvtexpy.output.structured_output import (
    Element,
    ListItemElement,
    ListItemLabelElement,
    OrderedListElement,
    StructuredOutput,
    SuppressElement,
    UnorderedListElement,
)


class ListFormatter(FormatterElement):
    _element: Element | None = None

    ordered: bool
    enabled: bool = True

    def __init__(self, ordered: bool):
        self.ordered = ordered

    @property
    def is_active(self) -> bool:
        return self._element is not None

    def begin(self, struct_out: StructuredOutput):
        super().begin(struct_out)
        if self.enabled:
            if self.ordered:
                self._element = struct_out.push(OrderedListElement())
            else:
                self._element = struct_out.push(UnorderedListElement())

    def end(self, struct_out: StructuredOutput):
        super().end(struct_out)
        if self._element is not None:
            struct_out.pop(self._element)
            self._element = None


class ListItemFormatter(FormatterElement):
    _element: Element | None = None

    parent_list: ListFormatter

    enabled: bool = True
    merge_prev: "ListItemFormatter | None" = None
    merge_next: bool = False

    def __init__(self, parent_list_formatter: ListFormatter):
        self.parent_list = parent_list_formatter

    def begin(self, struct_out: StructuredOutput):
        super().begin(struct_out)
        if self.parent_list.is_active:
            if self.merge_prev is not None:
                self._element = self.merge_prev._element
            elif self.enabled:
                self._element = struct_out.push(ListItemElement())
            else:
                self._element = struct_out.push(SuppressElement())

    def end(self, struct_out: StructuredOutput):
        super().end(struct_out)
        if self._element is not None and not self.merge_next:
            struct_out.pop(self._element)
            self._element = None


class ListItemLabelFormatter(FormatterElement):
    _element: Element | None = None

    enabled: bool = True

    def begin(self, struct_out: StructuredOutput):
        super().begin(struct_out)
        if self.enabled:
            self._element = struct_out.push(ListItemLabelElement())

    def end(self, struct_out: StructuredOutput):
        super().end(struct_out)
        if self._element is not None:
            struct_out.pop(self._element)
            self._element = None


@dataclass
class ListItemState:
    format: ListItemFormatter
    has_char: bool = False
    has_char_label: bool = False
    in_label: bool = True
    label_format: ListItemLabelFormatter = field(default_factory=ListItemLabelFormatter)


@dataclass
class ListState:
    save_level: SaveLevel
    ordered: bool
    items: List[ListItemState] = field(default_factory=list)
    format: ListFormatter | None = None

    @property
    def active_item(self) -> ListItemState | None:
        if len(self.items) == 0:
            return None
        return self.items[-1]


@Rules.register
class ListRule(Rule):
    """For itemize and enumerate."""

    in_package = capture_stack_element("@fileswith@pti@ns")

    list_stack: List[ListState] = []

    @property
    def active_list(self) -> ListState | None:
        if len(self.list_stack) == 0:
            return None
        return self.list_stack[-1]

    @staticmethod
    def _roman_to_int(roman: str) -> int:
        return ("i", "ii", "iii", "iv", "v", "vi").index(roman) + 1

    def _disable_list(self, list: ListState):
        if list.format is not None:
            list.format.enabled = False
        for item in list.items:
            item.format.enabled = False
            item.label_format.enabled = False
            item.label_format.suppress_formats = False

    def _merge_items(self, first_idx: int):
        state = self.active_list
        assert state is not None
        state.items[first_idx].format.merge_next = True
        state.items[first_idx + 1].format.merge_prev = state.items[first_idx].format

    def _start_list(self, *_):
        state = self.active_list
        assert state is not None
        assert state.format is None
        self.font_rule.clear_format()
        state.format = ListFormatter(ordered=state.ordered)
        self.push_formatter(state.format)
        if DEBUG_LIST:
            print("L: Start list", state)

    def _end_list(self, head_addr: int, tail_addr: int):
        state = self.list_stack.pop()
        if state.format is None:
            # Disable the list
            self._disable_list(state)
        else:
            # Pop last item formatter
            assert state.active_item is not None
            self.font_rule.clear_format()
            self.pop_formatter(state.active_item.format)
            if DEBUG_LIST:
                print("L: End item", state.items[-1])

            self.pop_formatter(state.format)
            if DEBUG_LIST:
                print("L: End list", state.items[-1])

            if not any(item.has_char for item in state.items):
                self._disable_list(state)

        if DEBUG_LIST:
            print("L: itemize end", state)

        if state.format is not None and state.format.enabled:
            self.state.input_formatter_store_next = True

    def _start_item(self, *_):
        assert self.active_list is not None
        assert self.active_list.format is not None
        self.font_rule.clear_format()
        item_state = ListItemState(format=ListItemFormatter(self.active_list.format))
        self.active_list.items.append(item_state)
        self.push_formatter(item_state.format)
        self.push_formatter(item_state.label_format)

        if DEBUG_LIST:
            print("L: Start item", item_state)

    def _end_item_label(self, *_):
        assert self.active_list is not None
        assert self.active_list.active_item is not None
        self.font_rule.clear_format()
        self.pop_formatter(self.active_list.active_item.label_format)
        self.active_list.active_item.in_label = False
        if not self.active_list.active_item.has_char_label:
            # No visible character for the list separator
            self._merge_items(len(self.active_list.items) - 2)
        # Now the actual item begins
        if DEBUG_LIST:
            print("L: End item label, begin item text", self.active_list.active_item)

    def _end_item(self, *_):
        list = self.active_list
        assert list is not None
        item = list.active_item
        assert item is not None
        self.font_rule.clear_format()
        item.format.enabled = item.has_char
        item.label_format.enabled = item.has_char
        item.label_format.suppress_formats = item.has_char
        # Pop last item formatter
        self.pop_formatter(item.format)
        if DEBUG_LIST:
            print("L: End item", item)

    @begin_macro_hook(
        [
            "itemize",
            "enumerate",
        ]
    )
    @assert2pagefail
    def on_itemize_begin(
        self,
        macro_name: str,
        *_args,
    ):
        if self.in_package:
            if DEBUG_LIST:
                print("L: in package")
            return

        if len(self.state.save_levels) == 0:
            return None

        state = ListState(save_level=self.state.save_levels[-1], ordered=macro_name == "enumerate")
        self.list_stack.append(state)

        if DEBUG_LIST:
            print("L: itemize begin", state)

        state.save_level.end_hooks.append(self._end_list)

        return self._start_list

    @begin_macro_hook("@item")
    @assert2pagefail
    def on_list_item_begin(
        self,
        *args,
    ):
        if self.in_package or self.active_list is None:
            return

        state = self.active_list
        assert state is not None

        if state.active_item is not None:
            self._end_item()

        assert state.format is not None
        self._start_item()

        return self._end_item_label

    def on_char_alloc(self, node: Node):
        for list in self.list_stack:
            # Activate the current item, if not empty
            if list.active_item is not None:
                if list.active_item.in_label:
                    list.active_item.has_char_label = True
                else:
                    list.active_item.has_char = True

    # @begin_macro_hook(
    #     [
    #         "@listi",
    #         "@listii",
    #         "@listiii",
    #         "@listiv",
    #         "@listv",
    #         "@listvi",
    #     ]
    # )
    # def on_list_begin_verify(
    #     self,
    #     macro_name: str,
    #     *_args,
    # ):
    #     if self.in_package:
    #         return

    #     level = self._roman_to_int(macro_name.removeprefix("@list"))

    #     print(f"List level: {level} {macro_name}")

    #     # assert len(self.list_stack) == level, f"List level mismatch: {len(self.list_stack)} != {level}"

    #     def listi_end():
    #         print(f"List level end {level} {macro_name}")
    #     return listi_end

    @begin_macro_hook(
        [
            "labelitemi",
            "labelitemii",
            "labelitemiii",
            "labelitemiv",
            "labelitemv",
            "labelitemvi",
        ]
    )
    @assert2pagefail
    def on_list_item_item_begin_verify(
        self,
        macro_name: str,
        *_args,
    ):
        if self.in_package or self.active_list is None:
            return

        level = self._roman_to_int(macro_name.removeprefix("labelitem"))

        assert sum(0 if s.ordered else 1 for s in self.list_stack) == level, (
            f"List level mismatch: {sum(0 if s.ordered else 1 for s in self.list_stack)} != {level}"
        )

    @begin_macro_hook(
        [
            "labelenumi",
            "labelenumii",
            "labelenumiii",
            "labelenumiv",
            "labelenumv",
            "labelenumvi",
        ]
    )
    @assert2pagefail
    def on_list_item_enum_begin_verify(
        self,
        macro_name: str,
        *_args,
    ):
        if self.in_package or self.active_list is None:
            return

        level = self._roman_to_int(macro_name.removeprefix("labelenum"))

        assert sum(1 if s.ordered else 0 for s in self.list_stack) == level, (
            f"List level mismatch: {sum(1 if s.ordered else 0 for s in self.list_stack)} != {level}"
        )
