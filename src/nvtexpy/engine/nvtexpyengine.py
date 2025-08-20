# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import math
import random
import traceback
from typing import Dict, List, Tuple

from pdftex import (
    get_cs,
    get_cur_group,
    get_cur_mode,
    get_font_name,
    get_font_properties,
    get_real_font,
    id_lookup,
)

from nvtexpy.engine.callstack import CallStack
from nvtexpy.engine.common import (
    NULL,
    BoxStackItem,
    CommandCode,
    GroupType,
    Node,
    SaveLevel,
    cur_env,
)
from nvtexpy.engine.debug import (
    DEBUG_ALL_CALLS,
    DEBUG_ASSERT_FAIL,
    DEBUG_BOX,
    DEBUG_CHAR_ALLOCS,
    DEBUG_COPY,
    DEBUG_NODE_ALLOCS,
    DEBUG_NODE_ID,
    DEBUG_NODE_OUT,
    DEBUG_OUTPUT_STACK,
    DEBUG_SAVELEVELS,
    DEBUG_WATCHDOG,
    DEBUG_WRITE_COMPACT,
    DEBUG_WRITE_FLAT_HTML,
    DEBUG_WRITE_FLAT_MD,
    DEBUG_WRITE_TEXT,
    DebugHex,
)
from nvtexpy.engine.glyphmap import GlyphMap
from nvtexpy.engine.hooks import (
    cs_begin_peek,
    cs_call_end,
    cs_end_peek,
    cs_pop_input,
    cs_push_input,
    debug_cs_call,
)
from nvtexpy.engine.hooks import debug_prefix as cs_debug_prefix
from nvtexpy.engine.hooks import (
    init_hooks,
    record_in_command,
    register_global_macro_call,
    register_macro_call,
)
from nvtexpy.engine.mem import (
    UNITY,
    CharNode,
    DiscNode,
    GlueNode,
    GlueSubtype,
    HBoxNode,
    KernNode,
    KernSubtype,
    LigatureNode,
    MemReader,
)
from nvtexpy.engine.mem import Node as MemNode
from nvtexpy.engine.mem import RuleNode, VBoxNode
from nvtexpy.engine.output_stack import FormatterOutputStack
from nvtexpy.engine.rules import Rules
from nvtexpy.engine.rules.base import assert2pagefail
from nvtexpy.engine.rules.math import MathRule
from nvtexpy.engine.rules.table import TableRule
from nvtexpy.engine.stack_formatter import (
    DirectBeginFormatterWrapper,
    DirectEndFormatterWrapper,
    FormatterElement,
)
from nvtexpy.engine.texstate import CharConverter, TexState
from nvtexpy.output import nested_bbox
from nvtexpy.output.pickle_output import save_pickle
from nvtexpy.output.structured_output import (
    BoundingBox,
    CharacterElement,
    HyphenElement,
    LayoutbreakElement,
    LinebreakElement,
    ParagraphbreakElement,
    SpaceElement,
    StructuredOutput,
)


class NVTexPyEngine:
    """The main class that is instantiated by the PdfTeX engine"""

    # Increase when interface for calls from engine to C code changes
    PDFTEX_INTERFACE_VERSION: int = 7
    # Increase when interface for calls from pdftex to engine
    ENGINE_INTERFACE_VERSION: int = 6

    struct_out: StructuredOutput | None

    # If true, do not trigger on out_ hooks.
    shipout_skip = False
    last_output_font: int = 0
    last_output_kern: float = 0
    _last_spacing_font: int = -1
    _last_font_spacing: float = 0
    _last_font_spacing_large: float = 0
    _last_stretch_glue: bool = False

    output_stack: FormatterOutputStack
    direct_output_stack: FormatterOutputStack

    state: TexState

    def __init__(self):
        print("Python NVTexPyEngine.__init__()")
        random.seed(0)

        self.state = TexState()
        self.state.callstack = CallStack()

        self.struct_out = None
        self.indent = ""
        self.last_was_hyphen_or_disc = False

        # Create rules and hook up the macro callbacks
        self.rules = Rules(self.state)

        self._str_cache = {}
        self._cs_cache = {}
        self._cur_hu: Dict[int, Node | None] = {}
        self._cur_hu_hyph_stack: List[Node | None] = []
        self._save_level_hyph = 0

        self.output_stack = FormatterOutputStack(self.state)
        self.direct_output_stack = FormatterOutputStack(self.state)

        init_hooks(DEBUG_ALL_CALLS)
        if DEBUG_ALL_CALLS:
            self.cs_call = debug_cs_call
            register_global_macro_call(lambda *x: self._register_check(*x), None)

        self.glyph_map = GlyphMap()
        self.state.char_converter = CharConverter(self.glyph_map)

        # Assign directly forwarded hooks
        table_rule = self.rules.get_rule(TableRule)
        self.align_init = table_rule.align_init
        self.align_end = table_rule.align_end
        self.align_row_init = table_rule.align_row_init
        self.align_row_end = table_rule.align_row_end
        self.align_col_init = table_rule.align_col_init
        self.align_col_end = table_rule.align_col_end
        self.align_omit = table_rule.align_omit
        self.align_span = table_rule.align_span
        self.align_no_align_begin = table_rule.align_no_align_begin
        self.align_no_align_end = table_rule.align_no_align_end
        self.math_rule = self.rules.get_rule(MathRule)
        self.mlist_to_hlist_end = self.math_rule.mlist_to_hlist_end
        self.mlist_to_hlist_converted = self.math_rule.mlist_to_hlist_converted
        self.mlist_to_hlist_start = self.math_rule.mlist_to_hlist_start
        self.after_math = self.math_rule.after_math
        # self.fin_mlist = self.math_rule.fin_mlist
        self.math_shift = self.math_rule.math_shift
        self.math_clean_box = self.math_rule.math_clean_box
        self.math_lig_merge = self.math_rule.math_lig_merge
        self.math_choice_converted = self.math_rule.math_choice_converted
        self.cs_call_end = cs_call_end
        self.push_input = cs_push_input
        self.begin_peek = cs_begin_peek
        self.end_peek = cs_end_peek
        self.pop_input = cs_pop_input

        # Extra hooks to skip content in these sections.
        register_macro_call(
            "__hook shipout/before",
            begin_callback=self._on_shipout_skip_begin,
            end_callback=self._on_shipout_skip_end,
        )
        register_macro_call(
            "__hook shipout/background",
            begin_callback=self._on_shipout_skip_begin,
            end_callback=self._on_shipout_skip_end,
        )
        register_macro_call(
            "__hook shipout/after",
            begin_callback=self._on_shipout_skip_begin,
            end_callback=self._on_shipout_skip_end,
        )

        if DEBUG_WATCHDOG:
            from nvtexpy.engine.dbg_watchdog import watchdog_thread

            watchdog_thread(DEBUG_WATCHDOG)

    def _end_sheet(self):
        self.rules.call_sheet_end()

        if self.struct_out is not None:
            self.output_stack.clear()
            if not self.direct_output_stack.is_empty():
                if DEBUG_ASSERT_FAIL:
                    raise AssertionError(
                        f"Direct output stack not empty: {self.direct_output_stack.stack}"
                    )
                self.direct_output_stack.clear()
                self.state.current_page_failed = "Direct output stack not empty"

            if not self.state.current_page_failed:
                try:
                    self.struct_out.save()

                    save_pickle(self.struct_out)

                    if DEBUG_WRITE_COMPACT:
                        compact_root = copy.deepcopy(self.struct_out._root)
                        nested_bbox.compact_in_place(compact_root)
                        save_pickle(
                            self.struct_out,
                            override_filename=f"{self.struct_out.page_name}.compact.pkl",
                            override_root=compact_root,
                        )
                        if DEBUG_WRITE_TEXT:
                            with open(self.struct_out.page_name + ".compact.txt", "w") as f:
                                f.write(compact_root.to_str_compact())

                    if DEBUG_WRITE_FLAT_MD or DEBUG_WRITE_FLAT_HTML:
                        print("Writing out flat layout")

                        from nvtexpy.output.flat_layout import compact_in_place_flat_layout
                        from nvtexpy.output.flat_layout import (
                            convert_from_pkl as convert_flat_layout,
                        )

                        compact_root = copy.deepcopy(self.struct_out._root)
                        compact_in_place_flat_layout(compact_root)
                        if DEBUG_WRITE_TEXT:
                            with open(self.struct_out.page_name + ".dlncompact.txt", "w") as f:
                                f.write(compact_root.to_str_compact())

                        if DEBUG_WRITE_FLAT_MD:
                            result = convert_flat_layout(f"{self.struct_out.page_name}.pkl")
                            # assert isinstance(result, dict)
                            assert isinstance(result, str)

                            # Write to file with same name but flat_layout extension
                            with open(f"{self.struct_out.page_name}.flat_layout.json", "w") as wf:
                                wf.write(result)
                            with open(f"{self.struct_out.page_name}.flat_layout.txt", "w") as wf:
                                import json

                                result = json.loads(result)
                                for ann in result["ann"]:
                                    code = ann.get("content", "<no_contents>")
                                    if ann.get("bbox"):
                                        code = f"<x_{ann['bbox'][0]}><y_{ann['bbox'][1]}>{code}<x_{ann['bbox'][2]}><y_{ann['bbox'][3]}>"
                                    if ann.get("category_id"):
                                        code = f"{code}<class_{ann['category_id']}>"
                                    wf.write(code + "\n\n")
                            print("Wrote out flat layout md")
                        if DEBUG_WRITE_FLAT_HTML:
                            result = convert_flat_layout(
                                f"{self.struct_out.page_name}.pkl",
                                table_format="html",
                                style_format="html",
                            )
                            # assert isinstance(result, dict)
                            assert isinstance(result, str)

                            # Write to file with same name but flat_layout extension
                            with open(
                                f"{self.struct_out.page_name}.flat_layout.html.json", "w"
                            ) as wf:
                                wf.write(result)
                            with open(f"{self.struct_out.page_name}.flat_layout.html", "w") as wf:
                                import json

                                result = json.loads(result)
                                for ann in result["ann"]:
                                    code = ann.get("content", "<no_contents>")
                                    if ann.get("bbox"):
                                        code = f"<x_{ann['bbox'][0]}><y_{ann['bbox'][1]}>{code}<x_{ann['bbox'][2]}><y_{ann['bbox'][3]}>"
                                    if ann.get("category_id"):
                                        code = f"{code}<class_{ann['category_id']} />"
                                    wf.write(code + "\n\n")
                            print("Wrote out flat layout html")
                except AssertionError:
                    if DEBUG_ASSERT_FAIL:
                        raise
                    self.state.current_page_failed = traceback.format_exc()
            if self.state.current_page_failed:
                # This page failed

                with open(self.struct_out.page_name + ".failed", "w") as f:
                    f.write(str(self.state.current_page_failed))

    def sheet(self, mag: int, total_pages: int, node_addr: int, page_width: int, page_height: int):
        print(f"SHEET: {mag} {total_pages} 0x{node_addr:06X}")
        # self.md.write(f"# Sheet {total_pages+1}\n")
        self._end_sheet()

        self.state.current_page_failed = None
        self.struct_out = StructuredOutput(
            page_number=total_pages + 1, page_size=(page_width, page_height)
        )
        FormatterElement.struct_out = self.struct_out
        self.output_stack.struct_out = self.struct_out
        self.direct_output_stack.struct_out = self.struct_out

        memnode = MemReader().get_node(node_addr)
        if not isinstance(memnode, (VBoxNode, HBoxNode)):
            raise ValueError("Invalid page node", memnode)

        self.rules.call_sheet_begin(
            memnode, total_pages + 1, page_width, page_height, self.struct_out
        )

    def _on_shipout_skip_begin(self, cmd, *_):
        def _recording_callback(cmds):
            pass

        record_in_command(_recording_callback)
        self.shipout_skip = True

    def _on_shipout_skip_end(self, cmd, *_):
        self.shipout_skip = False

    def mark_page_as_failed(self, traceback: str):
        self.state.current_page_failed = traceback

    def copy(self, src_addr: int, dest_addr: int):
        if src_addr in self.state.addr_to_node:
            self.state.addr_to_node[dest_addr] = self.state.addr_to_node[src_addr]
            if DEBUG_COPY:
                print(
                    f"{cs_debug_prefix()}COPY: 0x{src_addr:06X} -> 0x{dest_addr:06X} {MemReader().get_node(src_addr)} from {self.state.addr_to_node[src_addr]}"
                )
        elif DEBUG_COPY:
            print(
                f"{cs_debug_prefix()}NOCOPY: 0x{src_addr:06X} -> 0x{dest_addr:06X} {MemReader().get_node(src_addr)}"
            )
            assert not isinstance(MemReader().get_node(src_addr), (CharNode, GlueNode)), (
                f"Unexpected copy of {MemReader().get_node(src_addr)} at 0x{src_addr:06X}"
            )
        # Note: Can't copy mlists (only resulting hlists)
        self.rules.call_copy(src_addr, dest_addr)

    def char_alloc_hyph(
        self,
        addr: int,
    ):
        # A hyphen was allocated explicitly (by "\-" command)
        if DEBUG_COPY:
            node = MemReader().get_node(addr)
            assert isinstance(node, CharNode)
            char_str = self.state.char_converter.get_char(node.font, node.character, debug=True)
            print(
                cs_debug_prefix()
                + f"CHAR_ALLOC_HYPH {char_str} @0x{addr:06X} {node} from {self.state.addr_to_node.get(addr)}"
            )
        node = self.state.addr_to_node.get(addr)
        assert node is not None
        node.is_hyphen = True
        # Clear the stack formatting
        node.formatting = None
        assert node.direct_formatting is None

    def set_hu(self, char_node_ptr: int, hu_idx: int):
        # A character is put into the hu list from a char node
        if DEBUG_COPY:
            print(
                f"{cs_debug_prefix()}SET_HU hu_idx={hu_idx} @0x{char_node_ptr:06X} {MemReader().get_node(char_node_ptr)} from {self.state.addr_to_node.get(char_node_ptr)}"
            )
        self._cur_hu[hu_idx] = self.state.addr_to_node.get(char_node_ptr)

    def set_hu_hyph(self, char_node_ptr: int, hu_idx: int):
        # A new hyphen character is put into the hu list. If char_node_ptr is None, the hyphen
        # has been processed and the hu entry should be restored.
        if DEBUG_COPY:
            if char_node_ptr == NULL:
                print(
                    f"{cs_debug_prefix()}SETHU_HYPH restore hu_idx={hu_idx} from {self._cur_hu_hyph_stack[-1]}"
                )
            else:
                print(
                    f"{cs_debug_prefix()}SETHU_HYPH hu_idx={hu_idx} @0x{char_node_ptr:06X} {MemReader().get_node(char_node_ptr)} from {self.state.addr_to_node.get(char_node_ptr)}"
                )
        if char_node_ptr == NULL:
            # Restore the previous character to the hu list.
            self._cur_hu[hu_idx] = self._cur_hu_hyph_stack.pop()
        else:
            # Store the allocated hyphen character to the hu list
            char_node = self.state.addr_to_node.get(char_node_ptr)
            assert char_node is not None
            char_node.is_hyphen = True
            char_node.formatting = None
            self._cur_hu_hyph_stack.append(self._cur_hu[hu_idx])
            self._cur_hu[hu_idx] = char_node

    def char_alloc_hyph_copy(
        self,
        addr: int,
        src_node_ptr: int,
    ):
        # A char is copied during hyphenation
        if DEBUG_COPY:
            node = MemReader().get_node(addr)
            assert isinstance(node, CharNode)
            char_str = self.state.char_converter.get_char(node.font, node.character, debug=True)
            print(
                cs_debug_prefix()
                + f"CHAR_ALLOC_HYPH_COPY {char_str} @0x{addr:06X} from 0x{src_node_ptr:06X} {node} from {self.state.addr_to_node[src_node_ptr]}"
            )
        assert src_node_ptr in self.state.addr_to_node

        self.state.addr_to_node[addr] = self.state.addr_to_node[src_node_ptr]

    def char_alloc_hyph_hu(
        self,
        addr: int,
        hu_idx: int,
    ):
        # A char is allocated from the hu list during hyphenation
        if DEBUG_COPY:
            node = MemReader().get_node(addr)
            assert isinstance(node, CharNode)
            char_str = self.state.char_converter.get_char(node.font, node.character, debug=True)
            print(
                cs_debug_prefix()
                + f"CHAR_ALLOC_HYPH_HU hu_idx={hu_idx} {char_str} @0x{addr:06X} {MemReader().get_node(addr)} from {self._cur_hu[hu_idx]}"
            )
        self.state.addr_to_node[addr] = self._cur_hu[hu_idx]

    def char_alloc(self, addr: int):
        rand_id = random.randint(0, 0xFFFFFF) if DEBUG_NODE_ID else 0

        if (
            len(self.state.save_levels) > 0
            and self.state.save_levels[-1].group_code == GroupType.DISC_GROUP
        ):
            assert self._save_level_hyph == 0, "Only simple \\discretionary{...}{}{} is supported"
            is_hyphen = True
        else:
            is_hyphen = False

        # This method is also called when a ligature is allocated. then character and font are
        # invalid.
        mem_node = MemReader().get_node(addr)
        if isinstance(mem_node, CharNode):
            realfont = get_real_font(mem_node.font)
            self.state.pop_store_next()
            self.state.addr_to_node[addr] = node = Node(
                orig_addr=DebugHex(addr),
                dbg_char=self.state.char_converter.get_char(mem_node.font, mem_node.character),
                character=mem_node.character,
                font=realfont,
                font_properties=self.state.font_properties.get(
                    mem_node.font, self.state.font_properties.get(realfont)
                ),
                env_name=cur_env(),
                save_levels=list(self.state.save_levels),
                mode=get_cur_mode(),
                formatting=None if is_hyphen else self.state.get_input_formatter_stack(),
                rand_id=rand_id,
                is_hyphen=is_hyphen,
            )
            if DEBUG_CHAR_ALLOCS:
                char_str = self.state.char_converter.get_char(
                    mem_node.font, mem_node.character, debug=True
                )
                print(
                    cs_debug_prefix()
                    + f"CHARALLOC {char_str} @0x{addr:06X} #{rand_id:06X} char=0x{mem_node.character:02X} font=0x{mem_node.font:02X}={get_font_name(mem_node.font).tobytes().decode()}=0x{realfont:02X}={get_font_name(realfont).tobytes().decode()} {node}"
                )
            self.rules.call_char_alloc(node)
        elif isinstance(mem_node, LigatureNode):
            if DEBUG_CHAR_ALLOCS:
                print(cs_debug_prefix() + f"LIGALLOC @0x{addr:06X} #{rand_id:06X} {mem_node}")
        else:
            raise ValueError(f"Invalid char_alloc: {mem_node}")

    def hyph_alloc(
        self,
        disc_addr: int,
        hyph_addr: int,
    ):
        reader = MemReader()
        disc_node = reader.get_node(disc_addr)
        hyph_node = reader.get_node(hyph_addr)
        if hyph_node is not None:
            char_str = self.state.char_converter.get_char(
                hyph_node.font, hyph_node.character, debug=True
            )
        else:
            char_str = "?"
        if DEBUG_COPY:
            print(
                cs_debug_prefix()
                + f"HYPH_ALLOC {disc_addr:06X} {hyph_addr:06X} {disc_node} {hyph_node} {char_str}"
            )

    def glue_alloc(self, addr: int):
        store_formatting = self.state.pop_store_next()
        glue_node = MemReader().get_node(addr)

        rand_id = random.randint(0, 0xFFFFFF) if DEBUG_NODE_ID else 0

        assert glue_node is not None
        if glue_node.subtype in (
            GlueSubtype.above_display_short_skip_code,
            GlueSubtype.below_display_short_skip_code,
            GlueSubtype.above_display_skip_code,
            GlueSubtype.below_display_skip_code,
            # GlueSubtype.par_skip_code,
            # GlueSubtype.line_skip_code,
            # GlueSubtype.baseline_skip_code,
        ):
            store_formatting = True
            store_str = "STORE_ALWAYS"
        else:
            store_str = "STORE_YES" if store_formatting else "STORE_NO"

        if DEBUG_CHAR_ALLOCS:
            print(
                cs_debug_prefix()
                + f"GLUEALLOC @0x{addr:06X} #{rand_id:06X} {store_str} {glue_node}"
            )

        self.state.addr_to_node[addr] = node = Node(
            orig_addr=DebugHex(addr),
            dbg_char="",
            character=0,
            font=0,
            font_properties=None,
            env_name=cur_env(),
            save_levels=list(self.state.save_levels),
            mode=get_cur_mode(),
            formatting=self.state.get_input_formatter_stack() if store_formatting else None,
            rand_id=rand_id,
        )
        self.rules.call_glue_alloc(node, glue_node.subtype)

    def node_alloc(self, addr: int, size: int):
        if DEBUG_NODE_ALLOCS:
            print(cs_debug_prefix() + f"NODEALLOC @0x{addr:06X}+{size}b")
        # It's important to reset addrs of other nodes.
        self.rules.call_node_alloc(addr, size)

    def node_dealloc(self, addr: int, size: int):
        if DEBUG_NODE_ALLOCS:
            print(cs_debug_prefix() + f"NODEDEALLOC @0x{addr:06X}+{size}b")
        # It's important to reset addrs of other nodes.
        if addr in self.state.addr_to_node:
            del self.state.addr_to_node[addr]
        self.rules.call_node_dealloc(addr, size)

    if DEBUG_NODE_ALLOCS:
        # Only define if enabled.

        def after_node_alloc(self, addr: int):
            print(
                cs_debug_prefix() + f"AFTERNODEALLOC @0x{addr:06X}", str(MemReader().get_node(addr))
            )

    def print_node_list(self, addr: int):
        node = MemReader().get_node(addr)
        while node:
            print(f" - {node.format_str('   ')}")
            node = node.link

    @assert2pagefail
    def node_enter(self, node: Node | None, mem_node: MemNode, cur_h: int, cur_v: int):
        if node is None:
            return
        if node.formatting is not None:
            if DEBUG_OUTPUT_STACK:
                print(f"{self.indent}FMT POP {node.formatting}")
            self.output_stack.update_output_stack_pop(node.formatting)
        if node.formatting_begin is not None:
            if DEBUG_OUTPUT_STACK:
                print(f"{self.indent}FMT_BEGIN {node.formatting}")
            self.output_stack.update_output_stack_pop(node.formatting_begin)
            self.output_stack.update_output_stack_push(node.formatting_begin)
        if node.direct_formatting is not None:
            if DEBUG_OUTPUT_STACK:
                print(f"{self.indent}DIRECT_FORMATTING {node.direct_formatting}")
            # Reverse this, so that the last formatter is on top. Typically parsing is from inside
            # to outside (hlists are created at the very end, thus the action is then).
            for formatter in reversed(node.direct_formatting):
                if isinstance(formatter, DirectBeginFormatterWrapper):
                    self.direct_output_stack.push(formatter.actual_formatter)
                elif isinstance(formatter, DirectEndFormatterWrapper):
                    pass
                else:
                    self.direct_output_stack.push(formatter)
        if node.begin_actions is not None:
            for action in node.begin_actions:
                if DEBUG_OUTPUT_STACK:
                    print(f"{self.indent}BEGIN_ACTION {action}")
                action(node, mem_node, cur_h, cur_v, self.struct_out)
        if node.formatting is not None:
            if DEBUG_OUTPUT_STACK:
                print(f"{self.indent}FMT PUSH {node.formatting}")
            self.output_stack.update_output_stack_push(node.formatting)

    @assert2pagefail
    def node_exit(self, node: Node | None, mem_node: MemNode):
        if node is None:
            return
        if node.formatting_end is not None:
            if DEBUG_OUTPUT_STACK:
                print(f"{self.indent}FMT_END {node.formatting_end}")
            self.output_stack.update_output_stack_pop(node.formatting_end)
            self.output_stack.update_output_stack_push(node.formatting_end)
        if node.end_actions is not None:
            if DEBUG_OUTPUT_STACK:
                print(f"{self.indent}END_ACTIONS {node.end_actions}")
            for action in node.end_actions:
                action(node, mem_node, self.struct_out)
        if node.direct_formatting is not None:
            if DEBUG_OUTPUT_STACK:
                print(f"{self.indent}DIRECT_FMT {node.direct_formatting}")
            end_fmts = dict(
                (
                    (id(formatter.actual_formatter), formatter)
                    if isinstance(formatter, DirectEndFormatterWrapper)
                    else (id(formatter), formatter)
                )
                for formatter in reversed(node.direct_formatting)
                if not isinstance(
                    formatter,
                    DirectBeginFormatterWrapper,
                )
            )
            while len(end_fmts) > 0:
                popped_fmt = self.direct_output_stack.pop()
                try:
                    end_fmts.pop(id(popped_fmt))
                except KeyError:
                    assert False, f"Expected {end_fmts.values()} but got {popped_fmt}"

    def _get_font_spacing(self) -> Tuple[float, float]:
        if self.last_output_font == 0:
            # Spacing is inf for null font
            return math.inf, math.inf
        if self._last_spacing_font != self.last_output_font:
            slant, space, space_stretch, space_shrink, x_height, quad, extra_space, pdf_px_dimen = (
                get_font_properties(self.last_output_font)
            )
            if space == 0:
                # Likely a math font -> 1 mu = em_width / 18
                # Medium space: 4/18 em_width
                self._last_font_spacing = 4 * quad / 18
                self._last_font_spacing_large = 5 * quad / 18
            else:
                self._last_font_spacing = (space - space_shrink) / UNITY
                self._last_font_spacing_large = space / UNITY
            if DEBUG_NODE_OUT:
                print(
                    f"{self.indent}compute font spacing font={self.last_output_font} spacing={space / UNITY} stretch={space_stretch / UNITY} shrink={space_shrink / UNITY}, em_width={quad / UNITY}",
                )
            self._last_spacing_font = self.last_output_font
        return self._last_font_spacing, self._last_font_spacing_large

    def _flush_out_kern(self, clear: bool):
        self._last_stretch_glue = False
        if self.last_output_kern > 0 and self.last_output_kern >= self._get_font_spacing()[0]:
            if DEBUG_NODE_OUT:
                print(
                    f"{self.indent}FLUSH KERN {self.last_output_kern} font=0x{self.last_output_font:02X}",
                )
            self.struct_out.push_pop(SpaceElement())
            # Disable more kern to space conversion
            self.last_output_kern = -math.inf
        if clear:
            self.last_output_kern = 0

    def char_out(
        self,
        addr: int,
        cur_h: int,
        cur_v: int,
        width: int,
        height: int,
        depth: int,
        real_font: int,
    ):
        if self.shipout_skip:
            return
        assert self.struct_out is not None
        self._last_stretch_glue = False

        node = MemReader().get_node(addr)
        # This is a character node
        assert node is not None and isinstance(node, CharNode), "Not a char node"
        assert addr in self.state.addr_to_node, f"All char nodes are tracked 0x{addr:06X}"
        node_ref = self.state.addr_to_node[addr]
        if DEBUG_NODE_OUT:
            font_name = self.state.char_converter.get_font(node.font)
            char_str = self.state.char_converter.get_char(node.font, node.character, debug=True)
            print(
                f"{self.indent}CHAROUT {char_str} (chr=0x{node.character:02X}, font=0x{node.font:02X}={font_name}) @0x{addr:06X} #{node_ref.rand_id:06X} from @{node_ref.orig_addr}{' hyphen' if node_ref.is_hyphen else ''} fmt={node_ref.formatting} direct_fmt={node_ref.direct_formatting} act_beg={node_ref.begin_actions} act_end={node_ref.end_actions}",
            )

        # Replace the font with the real (non-virtual) font
        self.last_output_font = node_ref.font
        node_ref.font = node.font

        self._flush_out_kern(clear=True)

        self.rules.call_char_out(
            node_ref,
            node,
            cur_h,
            cur_v,
            width,
            height,
            depth,
        )

        self.node_enter(node_ref, node, cur_h, cur_v)

        self.output_stack.call_char_out(node_ref, node, cur_h, cur_v, width, height, depth)
        self.direct_output_stack.call_char_out(node_ref, node, cur_h, cur_v, width, height, depth)

        assert self.struct_out is not None
        if node_ref.replacement_character is not None:
            final_char = node_ref.replacement_character
            if DEBUG_NODE_OUT:
                print(
                    f"{self.indent}CHANGED_CHAROUT {final_char} @0x{addr:06X} #{node_ref.rand_id:06X} from @{node_ref.orig_addr} {node_ref.formatting}",
                )
        else:
            final_char = self.state.char_converter.get_char(node_ref.font, node_ref.character)

        if final_char:
            self.last_was_hyphen_or_disc = node_ref.is_hyphen
            if node_ref.is_hyphen:
                self.struct_out.push_pop(
                    HyphenElement(
                        final_char,
                        box=BoundingBox.from_tex(
                            cur_h, cur_v, width, height, depth, self.struct_out.current_transform
                        ),
                    )
                )
            else:
                self.struct_out.push_pop(
                    CharacterElement(
                        final_char,
                        box=BoundingBox.from_tex(
                            cur_h, cur_v, width, height, depth, self.struct_out.current_transform
                        ),
                    )
                )

        self.node_exit(node_ref, node)

    def vlist_out_start(self, addr: int, cur_h: int, cur_v: int):
        if self.shipout_skip:
            return
        # self.md.write(f"{self.indent}  - **vlist**\n")
        self._last_stretch_glue = False
        box = BoxStackItem("vlist", addr=addr, h=cur_h, v=cur_v)
        self.state.box_stack.append(box)
        node = self.state.addr_to_node.get(addr)
        if DEBUG_NODE_OUT:
            if node is not None:
                fmt_str = f" fmt={node.formatting} direct_fmt={node.direct_formatting} act_beg={node.begin_actions} act_end={node.end_actions}"
            else:
                fmt_str = ""
            print(
                f"{self.indent}VLISTOUTSTART @0x{addr:06X} {self.output_stack.stack} node={self.state.addr_to_node.get(addr)}{fmt_str}"
            )
            self.indent = self.indent + "  "
            self.output_stack.indent = self.indent
        self.last_output_kern = 0

        vbox = MemReader().get_node(addr)
        assert vbox is not None and isinstance(vbox, VBoxNode)
        self.node_enter(node, vbox, cur_h, cur_v)
        self.rules.call_vlist_out_start(vbox, cur_h, cur_v)
        self.output_stack.call_vlist_out_start(node, vbox, cur_h, cur_v)
        self.direct_output_stack.call_vlist_out_start(node, vbox, cur_h, cur_v)

    def vlist_out_end(self, addr: int):
        if self.shipout_skip:
            return
        box = self.state.box_stack.pop()
        node = self.state.addr_to_node.get(addr)
        if DEBUG_NODE_OUT:
            if node is not None:
                fmt_str = f" fmt={node.formatting} direct_fmt={node.direct_formatting} act_beg={node.begin_actions} act_end={node.end_actions}"
            else:
                fmt_str = ""
            self.indent = self.indent[:-2]
            self.output_stack.indent = self.indent
            print(f"{self.indent}VLISTOUTEND @0x{addr:06X}{fmt_str}")
        self.last_output_kern = 0

        for hook in box.end_hooks:
            hook()

        # parent_box = self.state.box_stack[-1] if len(self.state.box_stack) > 0 else None
        # if parent_box is not None and parent_box.node is not None:
        #     # Use the formatting from the parent node
        #     self._update_output_stack(parent_box.node)

        assert box.type == "vlist" and box.addr == addr

        vbox = MemReader().get_node(addr)
        assert vbox is not None and isinstance(vbox, VBoxNode)
        self.output_stack.call_vlist_out_end(node, vbox)
        self.direct_output_stack.call_vlist_out_end(node, vbox)
        self.rules.call_vlist_out_end(vbox)
        self.node_exit(node, vbox)

    def hlist_out_start(self, addr: int, cur_h: int, cur_v: int):
        if self.shipout_skip:
            return
        assert self.struct_out is not None

        self._last_stretch_glue = False

        node = self.state.addr_to_node.get(addr)

        # self.md.write(f"{self.indent}  - **hlist**\n")
        box = BoxStackItem("hlist", addr=addr, h=cur_h, v=cur_v)
        self.state.box_stack.append(box)
        if DEBUG_NODE_OUT:
            if node is not None:
                fmt_str = f"fmt={node.formatting} direct_fmt={node.direct_formatting} act_beg={node.begin_actions} act_end={node.end_actions} "
            else:
                fmt_str = ""
            print(
                f"{self.indent}HLISTOUTSTART @0x{addr:06X} {self.output_stack.stack} {fmt_str}node={node}"
            )
            self.indent = self.indent + "  "
            self.output_stack.indent = self.indent

        hbox = MemReader().get_node(addr)
        assert hbox is not None and isinstance(hbox, HBoxNode)
        self.node_enter(node, hbox, cur_h, cur_v)
        self.rules.call_hlist_out_start(hbox, cur_h, cur_v)
        self.output_stack.call_hlist_out_start(node, hbox, cur_h, cur_v)
        self.direct_output_stack.call_hlist_out_start(node, hbox, cur_h, cur_v)

    def hlist_out_end(self, addr: int):
        if self.shipout_skip:
            return
        node = self.state.addr_to_node.get(addr)
        if DEBUG_NODE_OUT:
            if node is not None:
                fmt_str = f" fmt={node.formatting} direct_fmt={node.direct_formatting} act_beg={node.begin_actions} act_end={node.end_actions}"
            else:
                fmt_str = ""
            self.indent = self.indent[:-2]
            self.output_stack.indent = self.indent
            print(f"{self.indent}HLISTOUTEND @0x{addr:06X} {fmt_str}")
        box = self.state.box_stack.pop()

        for hook in box.end_hooks:
            hook()

        hbox = MemReader().get_node(addr)
        assert hbox is not None and isinstance(hbox, HBoxNode)
        self.output_stack.call_hlist_out_end(node, hbox)
        self.direct_output_stack.call_hlist_out_end(node, hbox)
        self.rules.call_hlist_out_end(hbox)
        self.node_exit(node, hbox)
        # parent_box = self.state.box_stack[-1] if len(self.state.box_stack) > 0 else None
        # if parent_box is not None and parent_box.node is not None:
        #     # Use the formatting from the parent node
        #     self._update_output_stack(parent_box.node)

        assert box.type == "hlist" and box.addr == addr

    def _glue_out(
        self, mem_node: GlueNode, node_ref: Node, cur_h: int, cur_v: int, width: int, height: int
    ):
        assert node_ref is not None, "All glue nodes are tracked"
        assert self.struct_out is not None
        if DEBUG_NODE_OUT:
            print(
                f"{self.indent}GLUEOUT {mem_node} from @{node_ref.orig_addr} #{node_ref.rand_id:06X} w={width / UNITY} node={node_ref}"
            )
        flt_width = width / UNITY
        self.node_enter(node_ref, mem_node, cur_h, cur_v)
        self.rules.call_glue_out(node_ref, mem_node, cur_h, cur_v, width, height)
        self.output_stack.call_glue_out(node_ref, mem_node, cur_h, cur_v, width, height)
        self.direct_output_stack.call_glue_out(node_ref, mem_node, cur_h, cur_v, width, height)
        if node_ref.replacement_character is not None:
            if len(node_ref.replacement_character) > 0:
                self.struct_out.push_pop(
                    CharacterElement(
                        node_ref.replacement_character,
                        box=BoundingBox.from_tex(
                            cur_h, cur_v, width, height, 0, self.struct_out.current_transform
                        ),
                    )
                )
        elif (
            mem_node.subtype
            in (
                GlueSubtype.space_skip_code,
                GlueSubtype.xspace_skip_code,
            )
            and (
                mem_node.glue.width > 0
                or (mem_node.glue.stretch > 0 and mem_node.glue.stretch_order > 0)
                or flt_width > self._get_font_spacing()[1]
            )
        ) or (
            mem_node.subtype == GlueSubtype.normal
            and (mem_node.glue.width > 0 or flt_width > self._get_font_spacing()[1])
        ):
            if self.last_was_hyphen_or_disc:
                pass
            else:
                self.struct_out.push_pop(SpaceElement())
                self.last_output_kern = -math.inf
            if mem_node.glue.stretch > 0 and mem_node.glue.stretch_order > 0:
                self._last_stretch_glue = True
        elif mem_node.subtype == GlueSubtype.tab_skip_code:
            self._last_stretch_glue = False
            self.last_output_kern = -math.inf
            if flt_width > self._get_font_spacing()[1]:
                self.struct_out.push_pop(SpaceElement())
            # self.md.write_format_start("|")
        elif (
            mem_node.subtype == GlueSubtype.par_skip_code
            or mem_node.subtype == GlueSubtype.par_fill_skip_code
        ):
            self._last_stretch_glue = False
            self.output_stack.clear_unordered_formatter_stack()
            self.struct_out.push_pop(ParagraphbreakElement())
        elif mem_node.subtype == GlueSubtype.line_skip_code:
            self.output_stack.clear_unordered_formatter_stack()
            if self._last_stretch_glue:
                self._last_stretch_glue = True
                self.struct_out.push_pop(LinebreakElement())
            else:
                self._last_stretch_glue = True
                self.struct_out.push_pop(SpaceElement())
        elif mem_node.subtype == GlueSubtype.baseline_skip_code:
            if self.last_was_hyphen_or_disc:
                self.last_was_hyphen_or_disc = False
            else:
                # TODO:
                # Baseline_skip is inserted if the current paragraph line is broken to the next
                # line. But it is also used in some special cases to mark a newline :-/
                if self._last_stretch_glue:
                    self._last_stretch_glue = True
                    self.struct_out.push_pop(LinebreakElement())
                else:
                    self._last_stretch_glue = True
                    self.struct_out.push_pop(SpaceElement())
                    self.last_output_kern = -math.inf
        elif mem_node.subtype in (
            GlueSubtype.above_display_short_skip_code,
            GlueSubtype.above_display_skip_code,
            GlueSubtype.below_display_short_skip_code,
            GlueSubtype.below_display_skip_code,
        ):
            self._last_stretch_glue = False
            self.output_stack.clear_unordered_formatter_stack()
            self.struct_out.push_pop(ParagraphbreakElement())
        elif (
            mem_node.subtype == GlueSubtype.split_top_skip_code
            or mem_node.subtype == GlueSubtype.top_skip_code
        ):
            self._last_stretch_glue = False
            # This is a special case for the glue at top of split pages
            self.struct_out.push_pop(LayoutbreakElement())

        self.node_exit(node_ref, mem_node)

    def _rule_out(
        self,
        mem_node: RuleNode,
        node_ref: Node | None,
        cur_h: int,
        cur_v: int,
        width: int,
        height: int,
    ):
        assert self.struct_out is not None
        if DEBUG_NODE_OUT:
            if node_ref is not None:
                fmt_str = f" {node_ref.formatting} fmt={node_ref.formatting} direct_fmt={node_ref.direct_formatting} act_beg={node_ref.begin_actions} act_end={node_ref.end_actions}"
            else:
                fmt_str = ""
            rand_id = node_ref.rand_id if node_ref else 0
            orig_addr = node_ref.orig_addr if node_ref else 0
            print(f"{self.indent}RULEOUT {mem_node} from @{orig_addr} #{rand_id:06X}{fmt_str}")
        self.last_output_kern = 0
        self.node_enter(node_ref, mem_node, cur_h, cur_v)
        self.rules.call_rule_out(node_ref, mem_node, cur_h, cur_v, width, height)
        self.output_stack.call_rule_out(node_ref, mem_node, cur_h, cur_v, width, height)
        self.direct_output_stack.call_rule_out(node_ref, mem_node, cur_h, cur_v, width, height)
        self.node_exit(node_ref, mem_node)

    def _kern_out(
        self,
        mem_node: KernNode,
        node_ref: Node | None,
        cur_h: int,
        cur_v: int,
    ):
        if node_ref is not None and node_ref.replacement_character is not None:
            if DEBUG_NODE_OUT:
                print(f"{self.indent}KERNOUT flush as {node_ref.replacement_character!r}")
            if len(node_ref.replacement_character) > 0:
                # This kern should be written directly
                self.struct_out.push_pop(
                    CharacterElement(
                        node_ref.replacement_character,
                        box=BoundingBox.from_tex(
                            cur_h,
                            cur_v,
                            mem_node.raw_width,
                            0,
                            0,
                            self.struct_out.current_transform,
                        ),
                    )
                )
            self.last_output_kern = 0
            return
        if mem_node.subtype == KernSubtype.acc_kern:
            # The current char is actually an accent -> apply to next character
            # TODO
            pass
        self.last_output_kern += mem_node.width

    def hlist_out_node(self, addr: int, cur_h: int, cur_v: int):
        if self.shipout_skip:
            return
        mem_node = MemReader().get_node(addr)
        if mem_node is None or isinstance(mem_node, (HBoxNode, VBoxNode, GlueNode, RuleNode)):
            return
        node = self.state.addr_to_node.get(addr)
        if DEBUG_NODE_OUT:
            if node is not None:
                fmt_str = f" {node!r}"
            else:
                fmt_str = " (no Node)"
            print(f"{self.indent}HLISTOUTNODE {mem_node}{fmt_str}")

        self.node_enter(node, mem_node, cur_h, cur_v)

        self.rules.call_hlist_out_node(mem_node, cur_h, cur_v)
        self.output_stack.call_hlist_out_node(node, mem_node, cur_h, cur_v)
        self.direct_output_stack.call_hlist_out_node(node, mem_node, cur_h, cur_v)

        if isinstance(mem_node, KernNode):
            self._kern_out(mem_node, self.state.addr_to_node.get(addr), cur_h, cur_v)
        if isinstance(mem_node, DiscNode) and (
            mem_node.replace_count > 0
            or mem_node.pre_break is not None
            or mem_node.post_break is not None
        ):
            self.last_was_hyphen_or_disc = True

        self.node_exit(node, mem_node)

    def vlist_out_node(self, addr: int, cur_h: int, cur_v: int):
        if self.shipout_skip:
            return
        mem_node = MemReader().get_node(addr)
        if mem_node is None or isinstance(mem_node, (HBoxNode, VBoxNode, GlueNode, RuleNode)):
            return
        if DEBUG_NODE_OUT:
            print(f"{self.indent}VLISTOUTNODE {mem_node}")
        self.last_output_kern = 0

        node = self.state.addr_to_node.get(addr)
        self.node_enter(node, mem_node, cur_h, cur_v)

        self.rules.call_vlist_out_node(mem_node, cur_h, cur_v)
        self.output_stack.call_vlist_out_node(node, mem_node, cur_h, cur_v)
        self.direct_output_stack.call_vlist_out_node(node, mem_node, cur_h, cur_v)

        self.node_exit(node, mem_node)

    def hlist_out_glue(self, addr: int, cur_h: int, cur_v: int, width: int, height: int):
        if self.shipout_skip:
            return
        mem_node = MemReader().get_node(addr)
        if mem_node is None or not isinstance(mem_node, GlueNode):
            return
        node_ref = self.state.addr_to_node.get(addr)
        if node_ref is None:
            return
        self._glue_out(mem_node, node_ref, cur_h, cur_v, width, height)

    def vlist_out_glue(self, addr: int, cur_h: int, cur_v: int, width: int, height: int):
        if self.shipout_skip:
            return
        mem_node = MemReader().get_node(addr)
        if mem_node is None or not isinstance(mem_node, GlueNode):
            return
        node_ref = self.state.addr_to_node.get(addr)
        if node_ref is None:
            return
        self._glue_out(mem_node, node_ref, cur_h, cur_v, width, height)

    def hlist_out_rule(self, addr: int, cur_h: int, cur_v: int, width: int, height: int):
        if self.shipout_skip:
            return
        mem_node = MemReader().get_node(addr)
        if mem_node is None or not isinstance(mem_node, RuleNode):
            return
        node_ref = self.state.addr_to_node.get(addr)
        if node_ref is None:
            return
        self._rule_out(mem_node, node_ref, cur_h, cur_v, width, height)

    def vlist_out_rule(self, addr: int, cur_h: int, cur_v: int, width: int, height: int):
        if self.shipout_skip:
            return
        mem_node = MemReader().get_node(addr)
        if mem_node is None or not isinstance(mem_node, RuleNode):
            return
        node_ref = self.state.addr_to_node.get(addr)
        if node_ref is None:
            return
        self._rule_out(mem_node, node_ref, cur_h, cur_v, width, height)

    def save_level_start(self, save_ptr: int, cur_level: int, group_code: int):
        if DEBUG_SAVELEVELS:
            print(
                f"{cs_debug_prefix()}{'X' * cur_level} SAVELEVELSTART @0x{save_ptr:06X} {cur_level} {GroupType(group_code).name}"
            )
        cur_group, _cur_level = get_cur_group()
        assert _cur_level == cur_level
        # assert group_code != GroupType.DISC_GROUP, "Manual discretionary not supported"
        self.state.save_levels.append(
            SaveLevel(
                addr=save_ptr,
                level=cur_level,
                group_code=cur_group,
                env_name=cur_env(),
            )
        )
        # self.md.write(f"{self.indent}  - **save_level** (group={cur_group})\n")

        if len(self.state.save_levels) != cur_level:
            print(
                f"WARNING: save_levels list length ({len(self.state.save_levels)}) != cur_level ({cur_level})"
            )
            raise ValueError()

    def save_level_end(self, save_ptr: int, cur_level: int, head_addr: int, tail_addr: int):
        popped_level = self.state.save_levels.pop()
        if DEBUG_SAVELEVELS:
            print(
                f"{cs_debug_prefix()}{'X' * (cur_level + 1)} SAVELEVELEND: 0x{save_ptr + 1:06X} {cur_level + 1} {GroupType(popped_level.group_code).name}"
            )
        assert popped_level.addr == save_ptr + 1 and popped_level.level == cur_level + 1

        if popped_level.group_code == GroupType.DISC_GROUP:
            # Discretionary groups always appear in pairs of three
            self._save_level_hyph = (self._save_level_hyph + 1) % 3

        if popped_level.group_code == GroupType.MATH_LEFT_GROUP:
            self.math_rule.end_right(head_addr, tail_addr)

        for hook in popped_level.end_hooks:
            hook(head_addr, tail_addr)

        if len(self.state.save_levels) != cur_level:
            print(
                f"WARNING: save_levels list length ({len(self.state.save_levels)}) != cur_level ({cur_level})"
            )
            raise ValueError()

    if DEBUG_BOX:
        box_stack: list[int] = []

        def _box_context_str(self, box_context: int) -> str:
            box_flag = 0o10000000000  # {context code for `\.{\\setbox0}'}
            global_box_flag = 0o10000100000  # {context code for `\.{\\global\\setbox0}'}
            ship_out_flag = 0o10000200000  # {context code for `\.{\\shipout}'}
            leader_flag = 0o10000200001  # {context code for `\.{\\leaders}'}

            if box_context < box_flag:
                return f"(write to current list shifted by {box_context / UNITY})"
            elif box_context < global_box_flag:
                return f"\\global\\setbox{box_context - box_flag}"
            elif box_context < ship_out_flag:
                return f"\\setbox{box_context - global_box_flag}"
            elif box_context < leader_flag:
                return "\\shipout"
            else:
                return "\\leaders"

        def begin_box(self, cmd: int, chr_code: int, box_context: int, box_addr: int):
            BOX_CODE = 0
            COPY_CODE = 1
            LAST_BOX_CODE = 2
            VSPLIT_CODE = 3
            VTOP_CODE = 4
            VTOP_VMODE_CODE = 5
            if CommandCode(cmd) == CommandCode.make_box:
                if chr_code == BOX_CODE:
                    cmd_str = "\\box"
                elif chr_code == COPY_CODE:
                    cmd_str = "\\copy"
                elif chr_code == LAST_BOX_CODE:
                    cmd_str = "\\lastbox"
                elif chr_code == VSPLIT_CODE:
                    cmd_str = "\\vsplit"
                elif chr_code == VTOP_CODE:
                    cmd_str = "\\vtop"
                elif chr_code == VTOP_VMODE_CODE:
                    cmd_str = "\\vbox"
                else:
                    cmd_str = "\\hbox"
            else:
                cmd_str = "\\" + CommandCode(cmd).name

            context_str = self._box_context_str(box_context)

            box = MemReader().get_node(box_addr)
            print(
                f"{cs_debug_prefix()}BOX BEGINBOX {context_str} {cmd_str} {box} (0x{box_addr:06X})"
            )
            self.box_stack.append(box_context)

        def end_box(self, box_context: int, box_addr: int):
            if len(self.box_stack) > 0 and self.box_stack[-1] == box_context:
                _ref_ctx = self.box_stack.pop()
            else:
                print("WARNING: Mismatched box context")
            context_str = self._box_context_str(box_context)

            box = MemReader().get_node(box_addr)
            print(
                f"{cs_debug_prefix()}BOX ENDBOX @0x{box_addr:06X} {context_str} {box} (0x{box_addr:06X})"
            )

    def hook(self):
        # Called when \nvtexpyhook is encountered in TeX
        # Mainly for debugging purposes
        print("=== ADLR HOOK ===")
        print(f"CUR ENV: {cur_env()}")

        id_type, code = id_lookup(b"f@family", True)
        print(f"family: {id_type}, {code}")
        id_type, code = id_lookup(b"f@series", True)
        print(f"series: {id_type}, {code}")
        id_type, code = id_lookup(b"f@shape", True)
        print(f"shape: {id_type}, {code}")
        print()

        # prims = get_primitives()
        # print(f"Primitives: ({len(prims)})")
        # for prim in prims:
        #     if prim["text"] != 0:
        #         prim_text = get_str(prim["text"] - 1).tobytes().decode()
        #         print(f"  {prim_text}")

    if DEBUG_ALL_CALLS:

        def define(self, eqtb_addr: int, type: int, value: int, is_global: bool):
            cmd = CommandCode(type)
            if cmd == CommandCode.call:
                value = ", ".join(
                    f"{token}[{token.category!r}]"
                    for token in MemReader().get_token_list(value).tokens
                )
            cs = get_cs(eqtb_addr)
            if cs is not None:
                dst_node = cs.tobytes().decode()
            else:
                dst_node = "None"
            print(
                cs_debug_prefix()
                + f"{'GLOBAL ' if is_global else ''}DEFINE {dst_node}[0x{eqtb_addr:06X}]=({CommandCode(type).name}) {value}"
            )

        def word_define(self, eqtb_addr: int, value: int, is_global: bool):
            cs = get_cs(eqtb_addr)
            if cs is not None:
                dst_node = cs.tobytes().decode()
            else:
                dst_node = "None"
            print(
                cs_debug_prefix()
                + f"{'GLOBAL ' if is_global else ''}WORD_DEFINE {dst_node}[0x{eqtb_addr:06X}]={value}"
            )

        _cur_register_name: str | None = None
        _local_register_names: List[Dict[int, str]] = []
        _global_register_names: Dict[int, str] = {}

        def _register_check(
            self,
            cs_name: str,
            cmd: CommandCode,
            args: None,
            head_addr: int,
            tail_addr: int,
        ):
            if cmd == CommandCode.register:
                self._cur_register_name = cs_name

        def _get_register(self, addr: int, is_global: bool) -> str:
            if self._cur_register_name is not None:
                register_name = self._cur_register_name
                self._cur_register_name = None
                if is_global:
                    self._global_register_names[addr] = register_name
                else:
                    if self.state.save_levels[-1].debug_local_registers is None:
                        self.state.save_levels[-1].debug_local_registers = {}
                    self.state.save_levels[-1].debug_local_registers[addr] = register_name
            else:
                if is_global:
                    register_name = self._global_register_names.get(addr)
                else:
                    register_name = None
                    for save_level in reversed(self.state.save_levels):
                        if save_level.debug_local_registers is not None:
                            register_name = save_level.debug_local_registers.get(addr, None)
                            if register_name is not None:
                                break
            if register_name is None:
                register_name = f"0x{addr:06X}"
            else:
                register_name = f"{register_name}(0x{addr:06X})"
            return register_name

        def sa_define(self, sa_addr: int, value: int, is_global: bool):
            print(
                cs_debug_prefix()
                + f"{'GLOBAL ' if is_global else ''}SA DEFINE 0x{self._get_register(sa_addr, is_global)}={value}"
            )

        def sa_word_define(self, sa_addr: int, value: int, is_global: bool):
            print(
                cs_debug_prefix()
                + f"{'GLOBAL ' if is_global else ''}SA WORD_DEFINE {self._get_register(sa_addr, is_global)}={value}"
            )

    def exit(self):
        print("Python NVTexPyEngine.exit()")
        self._end_sheet()
        self.state.check_input_formatter_stack()

        # Must flush here.
        print("Python NVTexPyEngine.exit() done", flush=True)

    # def glyph_to_unicode(self, idx: int, glyph_name: str, tfmname: str, code: int):
    #     print(f"GLYPH_TO_UNICODE {tfmname}[0x{idx:02X}] = {glyph_name} {code:04X}")
