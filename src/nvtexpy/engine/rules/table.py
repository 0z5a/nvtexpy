# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from nvtexpy.engine.common import CommandCode, Node
from nvtexpy.engine.debug import DEBUG_TABLE, DebugHex
from nvtexpy.engine.math import noads_to_tex
from nvtexpy.engine.mem import (
    UNITY,
    GlueNode,
    GlueSubtype,
    HBoxNode,
    MathNode,
    MemReader,
    Noad,
    NoadFieldSubBox,
)
from nvtexpy.engine.mem import Node as MemNode
from nvtexpy.engine.mem import PenaltyNode, RuleNode, StyleType, UnsetNode, VBoxNode
from nvtexpy.engine.rules.base import (
    Rule,
    Rules,
    assert2pagefail,
    begin_macro_hook,
    end_macro_hook,
    rule_ref,
)
from nvtexpy.engine.rules.math import MathRule
from nvtexpy.engine.stack_formatter import CallbackFormatter, FormatterElement
from nvtexpy.output.math import ITNode, TableCellITNode, TableITNode, TableRowITNode
from nvtexpy.output.structured_output import (
    BoundingBox,
    Element,
    StructuredOutput,
    SuppressElement,
    TableCellElement,
    TableRowElement,
)


class TableCellRowspanFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(SuppressElement()),
            name="table_cell_rowspan",
        )


class TableCellFormatter(FormatterElement):
    _element: Element | None = None

    enabled: bool = True

    rowspan: int = 1
    colspan: int = 1

    top_border: bool = False
    right_border: bool = False
    left_border: bool = False
    bottom_border: bool = False

    bounding_box: BoundingBox | None = None

    tid: DebugHex = DebugHex(0)
    rid: DebugHex = DebugHex(0)

    def begin(self, struct_out: StructuredOutput):
        super().begin(struct_out)
        if self.enabled:
            if DEBUG_TABLE:
                print(f"T[{self.tid}, r={self.rid}, c=0x{id(self):06X}]: cell begin")
            self._element = self._bb_element = struct_out.push(
                TableCellElement(
                    rowspan=self.rowspan,
                    colspan=self.colspan,
                    border_left=self.left_border,
                    border_top=self.top_border,
                    border_right=self.right_border,
                    border_bottom=self.bottom_border,
                )
            )
            if self.bounding_box is not None:
                self._element.boxes.append(self.bounding_box)

    def end(self, struct_out: StructuredOutput):
        super().end(struct_out)
        if self._element is not None:
            struct_out.pop(self._element)
            if DEBUG_TABLE:
                print(f"T[{self.tid}, r={self.rid}, c=0x{id(self):06X}]: cell end")
            self._element = None

    def on_track_box(self, node: MemNode, cur_h: int, cur_v: int):
        if not self.enabled:
            return
        if DEBUG_TABLE:
            print(
                f"T[{self.tid}, r={self.rid}, c=0x{id(self):06X}]: track cell bounding box {node}"
            )
        bounding_box = None
        if isinstance(node, (HBoxNode, VBoxNode, UnsetNode, RuleNode)):
            bounding_box = BoundingBox.from_tex(
                cur_h,
                cur_v,
                int(node.width * UNITY) if node.width is not None else 0,
                int(node.height * UNITY) if node.height is not None else 0,
                int(node.depth * UNITY) if node.depth is not None else 0,
                self.struct_out.current_transform,
            )
        elif isinstance(node, GlueNode):
            bounding_box = BoundingBox.from_tex(
                cur_h, cur_v, 0, 0, 0, self.struct_out.current_transform
            )
        if bounding_box is not None:
            if self.bounding_box is None:
                self.bounding_box = bounding_box
                if self._element is not None:
                    self._element.boxes.append(bounding_box)
            if self.bounding_box is not None:
                self.bounding_box.update_union(bounding_box)

    def __repr__(self) -> str:
        return f"TableCellFormatter(@0x{id(self):06X}, tid={self.tid}, rid={self.rid}, enabled={self.enabled}, rowspan={self.rowspan}, colspan={self.colspan})"


class TableRowFormatter(FormatterElement):
    _element: Element | None = None

    is_first: bool = False
    is_last: bool = False
    enabled: bool = True

    top_border: bool = False
    bottom_border: bool = False

    bounding_box: BoundingBox | None = None

    tid: DebugHex = DebugHex(0)

    def begin(self, struct_out: StructuredOutput):
        super().begin(struct_out)
        if self.enabled:
            if DEBUG_TABLE:
                print(f"T[{self.tid}, r=0x{id(self):06X}]: row begin")
            self._element = struct_out.push(
                TableRowElement(
                    border_top=self.top_border,
                    border_bottom=self.bottom_border,
                    is_first=self.is_first,
                    is_last=self.is_last,
                )
            )
            if self.bounding_box is not None:
                self._element.boxes.append(self.bounding_box)

    def end(self, struct_out: StructuredOutput):
        super().end(struct_out)
        if self._element is not None:
            struct_out.pop(self._element)
            self._element = None
            if DEBUG_TABLE:
                print(f"T[{self.tid}, r=0x{id(self):06X}]: row end")

    def on_track_box(self, node: MemNode, cur_h: int, cur_v: int):
        if not self.enabled:
            return
        if DEBUG_TABLE:
            print(f"T[{self.tid}, r=0x{id(self):06X}]: track row bounding box {node}")
        if isinstance(node, (HBoxNode, VBoxNode, UnsetNode, RuleNode)):
            assert self.bounding_box is None
            self.bounding_box = BoundingBox.from_tex(
                cur_h,
                cur_v,
                int(node.width * UNITY) if node.width is not None else 0,
                int(node.height * UNITY) if node.height is not None else 0,
                int(node.depth * UNITY) if node.depth is not None else 0,
                self.struct_out.current_transform,
            )
            if self._element is not None:
                self._element.boxes.append(self.bounding_box)

    def __repr__(self) -> str:
        return f"TableRowFormatter(@0x{id(self):06X}, tid={self.tid}, enabled={self.enabled})"


@dataclass
class TableCellState:
    # Raw states (will be converted to "border" later)
    has_char: bool = False
    has_left_vrule: bool = False
    has_right_vrule: bool = False
    has_hrule: bool = False

    format: TableCellFormatter = field(default_factory=TableCellFormatter)
    begin_stack: List[FormatterElement] | None = None
    rowspan_format: TableCellRowspanFormatter | None = None

    debug_content: str = ""

    tid: DebugHex = DebugHex(0)
    rid: DebugHex = DebugHex(0)
    id: DebugHex = DebugHex(0)

    ridx: int = 0
    cidx: int = 0
    # The rowspan in real rows (include ignored rows)
    ridx_end: int = 1

    math_cell: ITNode | None = None

    def __post_init__(self):
        self.id = DebugHex(id(self.format))
        self.format.tid = self.tid
        self.format.rid = self.rid

    @property
    def enabled(self) -> bool:
        return self.format.enabled

    def disable(self):
        self.format.enabled = False
        self.format.colspan -= 1


@dataclass
class TableRowState:
    # Contains the same cell multiple times for colspan
    cells: List[TableCellState] = field(default_factory=list)

    has_hrule: bool = False

    format: TableRowFormatter = field(default_factory=TableRowFormatter)
    begin_stack: List[FormatterElement] | None = None

    tid: DebugHex = DebugHex(0)
    id: DebugHex = DebugHex(0)

    ridx: int = 0

    active_rowspans: Dict[int, TableCellState] = field(default_factory=dict)
    active_hborder: bool = False
    active_hborders: Set[int] = field(default_factory=set)

    def __post_init__(self):
        self.id = DebugHex(id(self.format))
        self.format.tid = self.tid

    @property
    def is_noalign(self) -> bool:
        return len(self.cells) == 1 and self.cells[0].format.colspan == -1

    @property
    def enabled(self) -> bool:
        return self.format.enabled

    @property
    def has_active_cell(self) -> bool:
        return len(self.cells) != 0

    @property
    def active_cell(self) -> TableCellState:
        assert self.has_active_cell
        return self.cells[-1]

    def append_cell(
        self, no_align: bool = False, begin_stack: List[FormatterElement] | None = None
    ) -> TableCellState:
        cell = self.active_rowspans.get(len(self.cells), None)
        if cell is None:
            cell = TableCellState(
                tid=self.tid,
                rid=self.id,
                ridx=self.ridx,
                ridx_end=self.ridx,
                cidx=len(self.cells),
                begin_stack=begin_stack,
            )
        self.cells.append(cell)
        if no_align:
            cell.format.colspan = -1
        if self.active_hborder or cell.cidx in self.active_hborders:
            cell.format.top_border = True
        return cell

    def disable(self):
        self.format.enabled = False
        for cell in self.cells:
            if cell.ridx == self.ridx:
                cell.disable()

    def is_empty(self) -> bool:
        return len(self.cells) == 0 or all(not cell.has_char for cell in self.cells)

    def set_border_top(self, cell_idx: int | None = None):
        if cell_idx is None:
            self.format.top_border = True
            for cell in self.cells:
                if cell.ridx == self.ridx:
                    # Not in the middle of a multirow cell
                    cell.format.top_border = True
        elif cell_idx >= 0 and cell_idx < len(self.cells):
            if self.cells[cell_idx].ridx == self.ridx:
                # Not in the middle of a multirow cell
                self.cells[cell_idx].format.top_border = True

    def set_border_bottom(self, cell_idx: int | None = None):
        if cell_idx is None:
            self.format.bottom_border = True
            for cell in self.cells:
                if cell.ridx + cell.format.rowspan - 1 == self.ridx:
                    # Not in the middle of a multirow cell
                    cell.format.bottom_border = True
        elif cell_idx >= 0 and cell_idx < len(self.cells):
            if self.cells[cell_idx].ridx_end == self.ridx:
                self.cells[cell_idx].format.bottom_border = True

    def set_border_left(self, cell_idx: int):
        if cell_idx >= 0 and cell_idx < len(self.cells):
            self.cells[cell_idx].format.left_border = True

    def set_border_right(self, cell_idx: int):
        if cell_idx >= 0 and cell_idx < len(self.cells):
            self.cells[cell_idx].format.right_border = True

    def add_colspan(self, is_rowspanning: bool = False):
        assert len(self.cells) > 0
        if not is_rowspanning:
            self.cells[-1].format.colspan += 1
        self.cells.append(self.cells[-1])


@dataclass
class TableState:
    enabled: bool = True
    rows: List[TableRowState] = field(default_factory=list)

    begin_stack: List[FormatterElement] | None = None

    in_row: bool = False
    in_cell: bool = False

    id: DebugHex = DebugHex(0)

    @property
    def math_table(self) -> TableITNode:
        return TableITNode(
            data=[
                TableRowITNode(
                    cells=[
                        TableCellITNode(
                            child=cell.math_cell,
                            colspan=int(cell.format.colspan),
                            rowspan=int(cell.format.rowspan),
                            borders=(
                                cell.format.left_border,
                                cell.format.top_border,
                                cell.format.right_border,
                                cell.format.bottom_border,
                            ),
                        )
                        for cell in row.cells
                    ],
                    borders=(row.format.top_border, row.format.bottom_border),
                )
                for row in self.rows
            ]
        )

    active_hborder: bool = False
    active_hborders: Set[int] = field(default_factory=set)

    # outer list contains the columns with rowspan for the coming rows
    # Inner list maps column index to the data
    active_rowspans: List[Dict[int, TableCellState]] = field(default_factory=list)

    def __post_init__(self):
        self.id = DebugHex(random.randint(0, 0xFFFFFF))

    def append_row(
        self, no_align: bool = False, begin_stack: List[FormatterElement] | None = None
    ) -> TableRowState:
        row = TableRowState(
            tid=self.id,
            ridx=len(self.rows),
            active_rowspans=(
                {}
                if no_align
                else ({} if len(self.active_rowspans) == 0 else self.active_rowspans.pop(0))
            ),
            active_hborder=self.active_hborder,
            active_hborders=self.active_hborders,
            begin_stack=begin_stack,
        )
        self.rows.append(row)
        if self.active_hborder:
            row.format.top_border = True
        return row

    def pop_row(self) -> None:
        row = self.rows.pop()
        assert row is not None
        if row.active_rowspans:
            self.active_rowspans.insert(0, row.active_rowspans)
        # Store border state from that row
        if row.has_hrule:
            self.active_hborder = True
        for cell in row.cells:
            if cell.has_hrule:
                self.active_hborders.add(cell.cidx)
        row.disable()

    def next_row(self) -> None:
        self.active_hborder = False
        self.active_hborders.clear()

    @property
    def has_active_row(self) -> bool:
        return len(self.rows) > 0

    @property
    def has_active_cell(self) -> bool:
        return self.has_active_row and self.active_row.has_active_cell

    @property
    def active_row(self) -> TableRowState:
        assert self.has_active_row
        return self.rows[-1]

    @property
    def active_cell(self) -> TableCellState:
        return self.active_row.active_cell

    @property
    def active_cell_rowspanning(self) -> bool:
        return self.active_cell.ridx < len(self.rows) - 1

    def is_valid(self) -> bool:
        return (
            sum(0 if row.is_empty() else 1 for row in self.rows) > 1
            and max(len(row.cells) for row in self.rows) > 0
        )

    def disable(self):
        if DEBUG_TABLE:
            print("T: Table disable")
        self.enabled = False
        for row in self.rows:
            row.format.enabled = False
            for cell in row.cells:
                cell.format.enabled = False

    def set_rowspan(self, rowspan: int):
        assert self.active_cell is not None
        assert self.active_cell.format.rowspan == 1, "Multiple rowspans defined"
        cell = self.active_cell
        cell.format.rowspan = rowspan
        cell.rowspan_format = TableCellRowspanFormatter()
        cell.ridx_end += rowspan - 1
        for row_offset in range(1, rowspan):
            if len(self.active_rowspans) < row_offset:
                active_rowspan = {}
                self.active_rowspans.append(active_rowspan)
            else:
                active_rowspan = self.active_rowspans[row_offset - 1]
            for cell_offset in range(cell.format.colspan):
                active_rowspan[cell.cidx + cell_offset] = cell

    def add_colspan(self):
        assert self.active_row is not None
        assert self.active_cell is not None
        cell = self.active_cell
        self.active_row.add_colspan(is_rowspanning=self.active_cell_rowspanning)
        if self.active_cell_rowspanning:
            # This cell is from a rowspan above. Do not add the colspan
            return
        if cell.format.rowspan > 1:
            cell_offset = cell.format.colspan - 1
            for row_offset in range(0, cell.format.rowspan - 1):
                self.active_rowspans[row_offset][cell.cidx + cell_offset] = cell

    def set_hborder(self, cidx: int | None = None):
        if cidx is None:
            self.active_hborder = True
        else:
            self.active_hborders.add(cidx)
        if len(self.rows) >= 2:
            self.rows[-2].set_border_bottom(cidx)


@Rules.register
class TableRule(Rule):
    """
    Trigger this when align_init is called.
    Capture all align_* calls.
    """

    # state of each cell [row][col]
    active_table_stack: List[TableState] = []
    last_table: TableState | None = None

    math_rule: MathRule = rule_ref(MathRule)

    @property
    def has_active_table(self) -> bool:
        return len(self.active_table_stack) > 0

    @property
    def active_table(self) -> TableState:
        assert self.has_active_table
        return self.active_table_stack[-1]

    @property
    def active_row(self) -> TableRowState:
        return self.active_table.active_row

    @property
    def active_cell(self) -> TableCellState:
        return self.active_table.active_cell

    @assert2pagefail
    def align_init(self):
        # Start align
        self.font_rule.clear_format()
        self.active_table_stack.append(TableState(begin_stack=self.dump_stack()))
        assert self.has_active_table
        self.last_table = None

        if DEBUG_TABLE:
            print(
                f"T[{self.active_table.id}]: table start (level {len(self.active_table_stack)})",
                self.state.get_input_formatter_stack(),
            )

    @assert2pagefail
    def align_end(self, head_addr: int, tail_addr: int):
        assert self.has_active_table

        self.font_rule.clear_format()
        table = self.active_table_stack.pop()
        self.last_table = table
        if DEBUG_TABLE:
            print(f"T[{table.id}]: table end {self.state._input_formatter_stack}")
            print(f"T[{table.id}]: math table data:")
            print(table.math_table)

        if DEBUG_TABLE:
            print(f"T[{table.id}]: debug data:")
            for row in table.rows:
                tr = []
                last_cell = None
                for cell in row.cells:
                    if last_cell != cell:
                        if last_cell is not None:
                            if cell.format.rowspan != 1 and cell.format.colspan != 1:
                                tr.append(
                                    f" <colspan={cell.format.colspan},rowspan={cell.format.rowspan}> "
                                )
                            elif cell.format.colspan != 1:
                                tr.append(f" <colspan={cell.format.colspan}> ")
                            elif cell.format.colspan != 1:
                                tr.append(f" <rowspan={cell.format.rowspan}> ")
                            else:
                                tr.append(" ")
                        else:
                            if cell.format.rowspan != 1 and cell.format.colspan != 1:
                                tr.append(
                                    f"<colspan={cell.format.colspan},rowspan={cell.format.rowspan}> "
                                )
                            elif cell.format.colspan != 1:
                                tr.append(f"<colspan={cell.format.colspan}> ")
                            elif cell.format.rowspan != 1:
                                tr.append(f"<rowspan={cell.format.rowspan}> ")
                        tr.append(cell.debug_content)
                        last_cell = cell
                print(f"  [r={row.id}] " + "".join(tr))

        if DEBUG_TABLE:
            print(f"T[{table.id}]: debug data 2:")
            for row in table.rows:
                tr = []
                last_cell = None
                for cell in row.cells:
                    tr.append(
                        f"<colspan={cell.format.colspan},rowspan={cell.format.rowspan}>[c={cell.id}] {cell.debug_content}"
                    )
                print(f"  [r={row.id}] " + " ".join(tr))

        if DEBUG_TABLE:
            print(f"T[{table.id}]: debug data 3:")
            for row in table.rows:
                print(f"  [r={row.id}] {row.format}")
                for cell in row.cells:
                    print(f"    [c={cell.id}] {cell}")

        assert not table.active_rowspans, "Rowspan outside table"

        if not table.is_valid():
            if DEBUG_TABLE:
                print(
                    f"T[{table.id}]: Table invalid. Rows: {len(table.rows)}, nonempty-rows: {sum(0 if row.is_empty() else 1 for row in table.rows)}. Max cols: {max((len(row.cells) for row in table.rows), default=0)}, non-empty cols: {max((sum(1 if cell.has_char else 0 for cell in row.cells) for row in table.rows), default=0)}"
                )

            table.disable()
            return
        # Go over all cells and prune empty cells.
        # If a pruned cell is vruled, assign border to neighbors
        # If a row has contains a hrule, and has no content, remove it and assign borders to neighbors.
        table_col_states = [False] * max(
            sum(col.format.colspan for col in row.cells) for row in table.rows
        )
        if DEBUG_TABLE:
            print(f"T[{table.id}]: postproc:")
        for row_idx, row in enumerate(table.rows):
            assert not row.is_empty()
            if row.is_noalign:
                row.cells[0].format.colspan = 1
                # This is a row covering all cols
                for _ in range(len(table_col_states) - 1):
                    row.add_colspan()
                if DEBUG_TABLE:
                    print(f"  [{row_idx},*]: Row covering all columns ({len(row.cells)})")
                continue
            last_cell = None
            for cell_idx, cell in enumerate(row.cells):
                # Do not process again if colspanning
                if cell == last_cell:
                    if DEBUG_TABLE:
                        print(f"  [{row_idx},{cell_idx}]: Skipping colspanning cell")
                    continue
                last_cell = cell
                if cell.has_char:
                    table_col_states[cell_idx] = True
                    if DEBUG_TABLE:
                        print(f"  [{row_idx},{cell_idx}]: Has char, set col state True")

        # Sync borders
        for row_idx, row in enumerate(table.rows):
            assert not row.is_empty()
            last_cell = None
            last_border = False
            last_cell_idx = -1
            for cell_idx, cell in enumerate(row.cells):
                # Do not process again if colspanning
                if cell == last_cell:
                    continue
                last_cell = cell
                # If cell as a border, store this
                if cell.has_left_vrule or last_border:
                    last_border = True
                    row.set_border_left(cell_idx)
                    row.set_border_right(last_cell_idx)
                if table_col_states[cell_idx]:
                    # This is a visible cell, thus reset the border for the right side
                    last_border = False
                    last_cell_idx = cell_idx
                if cell.has_right_vrule:
                    row.set_border_right(cell_idx)
                    last_border = True
        if DEBUG_TABLE:
            print(
                f"T[{table.id}]: table_col_states: {table_col_states}, row_states: {[not row.is_empty() for row in table.rows]}"
            )

        # Last stage: Disable the cols which have no content
        for row_idx, row in enumerate(table.rows):
            for cell_idx in reversed(range(len(row.cells))):
                if not table_col_states[cell_idx]:
                    if DEBUG_TABLE:
                        print(
                            f"T[{table.id}]: postproc[{row_idx},{cell_idx}]: Disabling colspanned cell"
                        )

                    # Disable for colspanned cells
                    row.cells[cell_idx].disable()
                    del row.cells[cell_idx]

        if table.enabled:
            # Skip initial and last glue nodes
            first_nonglue = None
            last_nonglue = None
            if DEBUG_TABLE:
                print(f"T[{table.id}]: all table elements:")
            for node in MemReader().iter_head_tail(head_addr, tail_addr):
                if DEBUG_TABLE:
                    print(f"  - {node}")
                if isinstance(node, (GlueNode, PenaltyNode)):
                    continue
                if first_nonglue is None:
                    first_nonglue = node
                last_nonglue = node

            if DEBUG_TABLE:
                print(f"  T[{table.id}]: first non-glue table element {first_nonglue}")
                print(f"  T[{table.id}]: last non-glue table element {last_nonglue}")

            if first_nonglue is None or last_nonglue is None:
                table.disable()
                return

            self.state.input_formatter_store_next = True

            table.rows[0].format.is_first = True
            table.rows[-1].format.is_last = True

            # self.register_node_action(
            #     first_nonglue.addr,
            #     last_nonglue.addr,
            #     direct_formatter=table.format,
            #     begin_stack=table.begin_stack,
            #     dump_stack=True,
            # )
            # for node in MemReader().iter_nodes(first_nonglue, last_nonglue, include_last=True):
            #     if DEBUG_TABLE:
            #         print(f"T[{table.id}]: tracking table element {node}")
            #     self.register_node_action(
            #         node.addr,
            #         begin_action=lambda node, mem_node, cur_h, cur_v, struct_out: table.format.on_track_box(
            #             mem_node, cur_h, cur_v
            #         ),
            #     )

    @assert2pagefail
    def align_omit(self):
        # This means, that the next col will not use the template. We can ignore this?
        if DEBUG_TABLE:
            print(f"T[{self.active_table.id}]: omit", self.state._input_formatter_stack)

    @assert2pagefail
    def align_span(self):
        # Merge this col with the next. Instead of col_end, col_init, we can get this.
        self.active_table.add_colspan()
        if DEBUG_TABLE:
            print(
                f"T[{self.active_table.id}, r={self.active_table.active_row.id}]: colspan",
                self.state._input_formatter_stack,
            )

    @assert2pagefail
    def align_row_init(self):
        # A new row starts
        self.font_rule.clear_format()
        row = self.active_table.append_row(begin_stack=self.dump_stack())
        self.active_table.in_row = True
        if DEBUG_TABLE:
            print(
                f"T[{self.active_table.id}, r={row.id}]: row start [{row.ridx}]",
                self.state._input_formatter_stack,
            )

    @assert2pagefail
    def align_row_end(self, box_addr: int):
        self.font_rule.clear_format()
        row = self.active_table.active_row
        if DEBUG_TABLE:
            print(
                f"T[{self.active_table.id}, r={row.id}]: row end [{row.ridx}]",
                self.state._input_formatter_stack,
            )
        self.active_table.in_row = False

        box = MemReader().get_node(box_addr)
        assert isinstance(box, UnsetNode)
        if row.is_empty() or (box.height == 0 and box.depth == 0) or box.width == 0:
            # This box is collapsed. Do not add it to the table
            if DEBUG_TABLE:
                print(f"T[{self.active_table.id}, r={row.id}]: pop empty row {box}")
            self.active_table.pop_row()
            return
        if DEBUG_TABLE:
            print(f"T[{self.active_table.id}, r={row.id}]: finish row")

        self.active_table.next_row()

        if DEBUG_TABLE:
            print(f"T[{self.active_table.id}, r={row.id}]: tracking row {box}")
        self.register_node_action(
            box_addr,
            begin_action=lambda node, mem_node, cur_h, cur_v, struct_out: row.format.on_track_box(
                mem_node, cur_h, cur_v
            ),
            direct_formatter=row.format,
            begin_stack=row.begin_stack,
            dump_stack=True,
        )

        # Ensure rowspans are added
        max_rowspan_col = max(row.active_rowspans.keys(), default=-1)
        while len(row.cells) <= max_rowspan_col:
            row.append_cell(begin_stack=self.dump_stack())

    @assert2pagefail
    def align_col_init(self):
        # A new column starts
        assert self.active_table.in_row
        self.font_rule.clear_format()
        cell = self.active_row.append_cell(begin_stack=self.dump_stack())
        self.active_table.in_cell = True
        self.state.input_formatter_store_next = True  # To store stack in next glue
        if DEBUG_TABLE:
            print(
                f"T[{self.active_table.id}, r={self.active_row.id}, c={cell.id}]: cell start [{cell.ridx}, {cell.cidx}]",
                self.state._input_formatter_stack,
            )

    @assert2pagefail
    def align_col_end(self, box_addr: int, tabskip_addr: int):
        self.font_rule.clear_format()
        cell = self.active_cell
        if DEBUG_TABLE:
            print(
                f"T[{self.active_table.id}, r={self.active_row.id}, c={cell.id}]: cell end [{cell.ridx}, {cell.cidx}]",
                self.state._input_formatter_stack,
            )
        self.active_table.in_cell = False

        if DEBUG_TABLE:
            print(
                f"T[{self.active_table.id}, r={self.active_row.id}, c={cell.id}]: tracking cell {MemReader().get_node(box_addr)}"
            )
            print(
                f"T[{self.active_table.id}, r={self.active_row.id}, c={cell.id}]: tracking cell {MemReader().get_node(tabskip_addr)}"
            )

        if self.active_table.active_cell_rowspanning:
            self.register_node_action(
                box_addr,
                begin_action=lambda node,
                mem_node,
                cur_h,
                cur_v,
                struct_out: cell.format.on_track_box(mem_node, cur_h, cur_v),
                direct_formatter=cell.rowspan_format,
                begin_stack=cell.begin_stack,
                dump_stack=True,
            )
        else:
            box = MemReader().get_node(box_addr)
            assert isinstance(box, UnsetNode)
            cell.math_cell = noads_to_tex(
                [el for el in box.list if not isinstance(el, GlueNode)],
                self.math_rule.get_context(self.state.char_converter, StyleType.text),
            )

            self.register_node_action(
                box_addr,
                begin_action=lambda node,
                mem_node,
                cur_h,
                cur_v,
                struct_out: cell.format.on_track_box(mem_node, cur_h, cur_v),
                direct_formatter=self.active_cell.format,
                begin_stack=self.active_cell.begin_stack,
                dump_stack=True,
            )
        self.register_node_action(
            tabskip_addr,
            begin_action=lambda node, mem_node, cur_h, cur_v, struct_out: cell.format.on_track_box(
                mem_node, cur_h, cur_v
            ),
        )

    @assert2pagefail
    def align_no_align_begin(self):
        # Covers a whole line without template
        assert not self.active_table.in_row
        self.font_rule.clear_format()
        # This is a row without cells
        row = self.active_table.append_row(no_align=True)
        self.push_formatter(row.format)
        # Create a dummy cell
        cell = row.append_cell(no_align=True)
        self.push_formatter(cell.format)
        if DEBUG_TABLE:
            print(
                f"T[{self.active_table.id}, r={row.id}, c={cell.id}]: noalign begin [{row.ridx}]",
                self.state._input_formatter_stack,
            )
            cell.debug_content += "\\noalign"

    @assert2pagefail
    def align_no_align_end(self):
        assert not self.active_table.in_row
        assert self.active_table.has_active_cell
        self.font_rule.clear_format()
        if DEBUG_TABLE:
            print(
                f"T[{self.active_table.id}, r={self.active_row.id}, c={self.active_cell.id}]: noalign end [{self.active_row.ridx}]",
                self.state._input_formatter_stack,
            )
            self.active_cell.debug_content += "\\\\"
        self.pop_formatter(self.active_cell.format)
        self.pop_formatter(self.active_row.format)
        if self.active_row.is_empty():
            if DEBUG_TABLE:
                print(
                    f"T[{self.active_table.id}, r={self.active_table.active_row.id}]: pop empty noalign row"
                )
            self.active_table.pop_row()
        else:
            if DEBUG_TABLE:
                print(
                    f"T[{self.active_table.id}, r={self.active_table.active_row.id}]: finish noalign row"
                )
            self.active_table.next_row()

    @begin_macro_hook("vrule")
    @assert2pagefail
    def vrule(self, *args):
        if not self.has_active_table or not self.active_table.has_active_cell:
            return
        assert self.active_table is not None
        if self.active_cell is not None:
            if not self.active_cell.has_char:
                self.active_cell.has_left_vrule = True
            else:
                self.active_cell.has_right_vrule = True
        if DEBUG_TABLE:
            print(
                f"T[{self.active_table.id}, r={self.active_row.id}, c={self.active_cell.id}]: vrule"
            )
            if self.active_cell is not None:
                self.active_cell.debug_content += "\\vrule"

    @begin_macro_hook("hrule")
    @assert2pagefail
    def hrule(self, *args):
        if not self.has_active_table or not self.active_table.has_active_cell:
            return
        if self.active_table.in_cell:
            assert self.active_row.active_cell is not None
            self.active_cell.has_hrule = True
            self.active_table.set_hborder(self.active_cell.cidx)
        else:
            self.active_row.has_hrule = True
            self.active_table.set_hborder()
        if DEBUG_TABLE:
            print(
                f"T[{self.active_table.id}, r={self.active_row.id}, c={self.active_cell.id}]: hrule"
            )
            if self.active_cell is not None:
                self.active_cell.debug_content += "\\hrule"

    if DEBUG_TABLE:

        def on_glue_alloc(self, node: Node, glue_type: GlueSubtype):
            if not self.has_active_table:
                return
            if not self.active_table.has_active_cell:
                return
            if glue_type == GlueSubtype.tab_skip_code:
                self.active_table.active_cell.debug_content += "&"
            elif glue_type == GlueSubtype.normal:
                self.active_table.active_cell.debug_content += " "
            elif (
                glue_type == GlueSubtype.baseline_skip_code
                or glue_type == GlueSubtype.line_skip_code
            ):
                self.active_table.active_cell.debug_content += "\\\\"
            else:
                self.active_table.active_cell.debug_content += f"\\{glue_type.name}"

    def on_char_alloc(
        self,
        node: Node,
    ):
        if not self.has_active_table:
            return
        if not self.active_table.has_active_cell:
            return
        self.active_table.active_cell.has_char = True
        self.active_table.active_cell.has_right_vrule = False
        if DEBUG_TABLE:
            self.active_table.active_cell.debug_content += self.state.char_converter.get_char(
                node.font, node.character, debug=True
            )

    @end_macro_hook("@multirow", str_args=True)
    @assert2pagefail
    def on_multirow_end(
        self,
        cs: str,
        cmd: CommandCode,
        args: Tuple[str | None, ...] | None,
        head_addr: int,
        tail_addr: int,
    ):
        if not self.has_active_table:
            return
        try:
            rowspan = int(args[1])
        except ValueError:
            assert False, f"Non-integer rowspan {args[1]!r}"
        assert rowspan > 0, "negative rowspan not supported"
        self.active_table.set_rowspan(rowspan)
        if DEBUG_TABLE:
            print(
                f"T[{self.active_table.id}, r={self.active_table.active_row.id}]: set rowspan {rowspan}"
            )

    @end_macro_hook("endarray")
    @assert2pagefail
    def on_math_table_end(
        self,
        cs: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        if self.last_table is None:
            # There is no last table?
            return
        # Tail addr -> Points to the noad which contains the table!
        if DEBUG_TABLE:
            print(f"M: endarray {self.last_table}")
            print("M: endarray node:")
            for node in MemReader().iter_head_tail(head_addr, tail_addr):
                print(f"  - {node}")
        tail = None
        for node in MemReader().iter_head_tail(head_addr, tail_addr):
            if isinstance(node, (GlueNode, PenaltyNode, MathNode)):
                continue
            tail = node
        if DEBUG_TABLE:
            print(f"  tail: {tail}")
        if isinstance(tail, (HBoxNode, VBoxNode)):
            self.math_rule.register_table(tail.addr, self.last_table.math_table)
        elif isinstance(tail, Noad):
            assert isinstance(tail, Noad), f"{tail}"
            assert isinstance(tail.nucleus, NoadFieldSubBox), f"{tail.nucleus}"
            assert self.last_table is not None
            assert self.last_table.math_table is not None
            self.math_rule.register_table(tail.nucleus.box.addr, self.last_table.math_table)
        else:
            assert False, f"{tail}"
        self.last_table = None

    def on_sheet_begin(
        self,
        memnode: VBoxNode | HBoxNode | None,
        page: int,
        page_width: int,
        page_height: int,
        struct_out: StructuredOutput,
    ):
        self.active_table_stack.clear()
        self.last_table = None

    def on_sheet_end(self):
        self.active_table_stack.clear()
        self.last_table = None


@Rules.register
class UnsupportedTables(Rule):
    @begin_macro_hook("deluxetable")
    @assert2pagefail
    def deluxetable(self, *args):
        assert False, "deluxetable is not supported"
