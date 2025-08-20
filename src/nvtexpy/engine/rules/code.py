# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

from pdftex import get_cur_box, id_lookup

from nvtexpy.engine.common import CommandCode, Node, get_int_value
from nvtexpy.engine.debug import DEBUG_LSTLISTINGS
from nvtexpy.engine.mem import GlueNode, GlueSubtype, HBoxNode, KernNode, MemReader
from nvtexpy.engine.rules.base import (
    Rule,
    Rules,
    assert2pagefail,
    begin_macro_hook,
    capture_stack_element,
    end_macro_hook,
)
from nvtexpy.engine.stack_formatter import CallbackFormatter
from nvtexpy.output.structured_output import (
    AlgorithmElement,
    AlgorithmIndentElement,
    CodeBlockElement,
    LinebreakElement,
    LineNumberElement,
)


class VerbatimFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(CodeBlockElement()),
            name="block_code",
        )

    def on_glue_out(
        self, node: Node, memnode: GlueNode, cur_h: int, cur_v: int, width: int, height: int
    ):
        if width < 0 and node.replacement_character == " ":
            node.replacement_character = ""


@Rules.register
class VerbatimBlockRule(Rule):
    """For \\begin{verbatim}."""

    line_active: bool = False
    first_line: bool = False

    @begin_macro_hook("verbatim@start")
    @assert2pagefail
    def on_verbatim_start(self, *args):
        # TODO: Kern nodes must be detected to save spaces properly
        if self.active:
            # Don't nest verbatim
            return None

        self.active = True
        self.line_active = False
        self.first_line = True

        formatter = VerbatimFormatter()
        self.push_formatter(formatter)

        def end_verbatim(head_addr: int, tail_addr: int):
            self.pop_formatter(formatter)
            self.active = False
            self.line_active = False
            self.first_line = True
            self.state.input_formatter_store_next = True

        # Ends when the currently active save group ends
        self.state.save_levels[-1].end_hooks.append(end_verbatim)

    @begin_macro_hook("verbatim@processline")
    @assert2pagefail
    def on_verbatim_processline(self, *args):
        # TODO: Kern nodes must be detected to save spaces properly
        if not self.active:
            return None

        self.line_active = True
        if self.first_line:
            self.begin_stack = self.dump_stack()
        if DEBUG_LSTLISTINGS:
            print("VERBATIM: Begin process line")

        def end_line(*_):
            self.line_active = False
            if DEBUG_LSTLISTINGS:
                print("VERBATIM: End process line")

        # Ends when the currently active save group ends
        return end_line

    def on_char_alloc(self, node: Node):
        if self.active:
            self.first_line = False
            self.begin_stack = None

    def on_glue_alloc(self, node: Node, glue_type: GlueSubtype):
        if self.active:
            if self.line_active:
                if glue_type == GlueSubtype.normal:
                    self.register_node_action(
                        node.orig_addr, begin_stack=self.begin_stack, dump_stack=True
                    )
                    node.replacement_character = " "
                    if DEBUG_LSTLISTINGS:
                        print("VERBATIM: Save glue")
            else:
                if glue_type == GlueSubtype.normal:
                    node.replacement_character = ""
                    if DEBUG_LSTLISTINGS:
                        print("VERBATIM: Skip glue")


class LstListingFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(CodeBlockElement()),
            name="block_code",
        )


class LstListingLineNumberFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(LineNumberElement()),
            name="line_number",
        )


@Rules.register
class LstListingBlockRule(Rule):
    """For \\begin{lstlisting}."""

    cur_text_node: List[List[Node]]

    _last_added_line: List[Node]
    _last_added_buf: List[Node]
    _last_glue: Node | None = None
    _last_nl_glue: Node | None = None

    _space_visible: bool = False
    _space_character: int = 0
    _last_space: int = 0

    @begin_macro_hook("\\lstlisting@", str_args=True)
    @assert2pagefail
    def on_lstlisting_start(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: Tuple[str | None, ...] | None,
        head_addr: int,
        tail_addr: int,
    ):
        # TODO: Kern nodes must be detected to save spaces properly
        assert not self.active, "Nested code?"

        if DEBUG_LSTLISTINGS:
            self._last_added_line = []
        self._last_added_buf = []
        self._last_space = 0

        formatter = LstListingFormatter()

        assert args is not None and len(args) > 0

        start_glue = None

        def start_code(*_):
            if DEBUG_LSTLISTINGS:
                print("LSTLISTING: Start Code")
            assert not self.active, "Nested code?"
            self.active = True
            assert self._last_nl_glue is not None
            nonlocal start_glue
            start_glue = self._last_nl_glue.orig_addr
            if DEBUG_LSTLISTINGS:
                self.cur_text_node = [[]]
                self._last_added_line.clear()
            self._last_added_buf.clear()
            self._last_space = 0

            # Extract if space visible
            if DEBUG_LSTLISTINGS:
                self._space_visible = id_lookup(b"lst@outputspace", True) == id_lookup(
                    b"lst@visiblespace", True
                )
                self._space_character = id_lookup(b"lst@visiblespace", True)
                print(f"LST: Space visible={self._space_visible} character={self._space_character}")

        def end_lstlisting(head_addr: int, tail_addr: int):
            if DEBUG_LSTLISTINGS:
                print("LSTLISTING: End Code:")
                print(self.cur_text_node)
                print("\n".join(self._dbg_line_to_str(t) for t in self.cur_text_node))
            assert self._last_glue is not None
            self.register_node_action(
                start_glue, self._last_glue.orig_addr, direct_formatter=formatter
            )
            self.active = False
            self.state.input_formatter_store_next = True

        # Ends when the currently active save group ends
        self.state.save_levels[-1].end_hooks.append(end_lstlisting)

        return start_code

    @begin_macro_hook("lst@OnceAtEOL", str_args=True)
    @assert2pagefail
    def on_lst_newline(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: Tuple[str | None, ...] | None,
        head_addr: int,
        tail_addr: int,
    ):
        if self.active:
            if DEBUG_LSTLISTINGS:
                pos = get_int_value("\\lst@pos")
                length = get_int_value("\\lst@length")
                print(f"LST lst@OnceAtEOL({args}): lst@pos={pos} lst@length={length}")
                print("Resolved -> '\\n'")
                self._dbg_last_added()
                self.cur_text_node.append(list(self._last_added_line))
                self._last_added_line.clear()
            self._last_space = 0
            self._last_glue = None
            self._last_added_buf.clear()

    def _dbg_line_to_str(self, line: List[Node]) -> str:
        line_text = []
        for node in line:
            if node.replacement_character is not None:
                line_text.append(node.replacement_character)
            elif node.dbg_char:
                line_text.append(node.dbg_char)
        return "".join(line_text)

    def _dbg_last_added(self):
        print("cur line:", repr("".join(self._dbg_line_to_str(self._last_added_line))))

    @end_macro_hook("lst@PlaceNumber", str_args=True)
    @assert2pagefail
    def on_lst_numbersep(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: Tuple[str | None, ...] | None,
        head_addr: int,
        tail_addr: int,
    ):
        if DEBUG_LSTLISTINGS:
            print(f"LST lst@PlaceNumber({args}):")
        node = None
        for node in MemReader().iter_head_tail(head_addr, tail_addr):
            pass
            if DEBUG_LSTLISTINGS:
                print(f" - Node: {node}")
        if node is not None and isinstance(node, HBoxNode):
            last_child = None
            for last_child in node.list:
                pass
            if isinstance(last_child, KernNode):
                self._get_node(last_child.addr, make_new=False).replacement_character = ""

    @begin_macro_hook("thelstnumber", str_args=True)
    @assert2pagefail
    def on_lst_number(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: Tuple[str | None, ...] | None,
        head_addr: int,
        tail_addr: int,
    ):
        if self.active:
            if DEBUG_LSTLISTINGS:
                print(f"LST thelstnumber({args})")
                lst_number = get_int_value("\\c@lstnumber")
                print(f"Resolved -> {lst_number}")
            self._last_added_buf.clear()

            def end_lst_newline(*_):
                if len(self._last_added_buf) > 0:
                    self.register_node_action(
                        self._last_added_buf[0].orig_addr,
                        self._last_added_buf[-1].orig_addr,
                        direct_formatter=LstListingLineNumberFormatter(),
                        dump_stack=True,
                    )
                    if DEBUG_LSTLISTINGS:
                        print(f"Line number -> {self._last_added_buf}")
                        for el in reversed(self._last_added_buf):
                            removed_el = self._last_added_line.pop()
                            assert el == removed_el
                    self._last_added_buf.clear()

            return end_lst_newline

    @begin_macro_hook("lst@OutputToken", str_args=True)
    @assert2pagefail
    def on_lst_output_token(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: Tuple[str | None, ...] | None,
        head_addr: int,
        tail_addr: int,
    ):
        if self.active:
            # lst@pos: The current writing position, which is decreased by 1 for every character
            #   writte, including spaces.
            # lst@length: The length of the text still to be appended (i.e. in tokens)
            # lst@column: The current column in the output box(?)
            # tokens = list(get_tokens('\\lst@token').tokens)

            if DEBUG_LSTLISTINGS:
                pos = get_int_value("\\lst@pos")
                column = get_int_value("\\lst@column")
                length = get_int_value("\\lst@length")
                print(
                    f"LST lst@OutputToken({args}): lst@column={column} lst@pos={pos} lst@length={length}, last_space={self._last_space}"
                )
                self._dbg_last_added()

            # self._last_added_line.clear()

            def end_box(*_):
                pos = get_int_value("\\lst@pos")
                length = get_int_value("\\lst@length")
                column = get_int_value("\\lst@column")
                self._last_space = column - pos + length
                self._last_glue = None
                if DEBUG_LSTLISTINGS:
                    print(
                        f"LST lst@OutputToken() END pos={pos} length={length} column={column}, last_space={self._last_space}"
                    )
                    self._dbg_last_added()

            return end_box

    _in_outputspace = False

    @begin_macro_hook("lst@InsertLostSpace")
    @assert2pagefail
    def on_lst_insert_lost_space(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        if self.active and not self._in_outputspace:
            self._last_added_buf.clear()
            if DEBUG_LSTLISTINGS:
                print(f"LST lst@InsertLostSpace() start: last_space={self._last_space}")
                self._dbg_last_added()

            def end(*_):
                pos = get_int_value("\\lst@pos")
                column = get_int_value("\\lst@column")
                next_space = column - pos
                if DEBUG_LSTLISTINGS:
                    print(
                        f"LST lst@InsertLostSpace() end: lst@column={column} lst@pos={pos}, space_pos={next_space}, last_space={self._last_space}"
                    )
                if len(self._last_added_buf) > 0:
                    if DEBUG_LSTLISTINGS:
                        print(f"lostspace set space (lastbuf) to {next_space - self._last_space}")
                    self._last_added_buf[-1].replacement_character = " " * (
                        next_space - self._last_space
                    )
                    self._last_added_buf.clear()
                elif self._last_glue is not None:
                    if DEBUG_LSTLISTINGS:
                        print(f"lostspace set space (lastglue) to {next_space - self._last_space}")
                    self._last_glue.replacement_character = " " * (next_space - self._last_space)
                self._last_glue = None
                self._last_space = next_space
                if DEBUG_LSTLISTINGS:
                    self._dbg_last_added()

            return end

    @begin_macro_hook("lst@outputspace")
    @assert2pagefail
    def on_lst_space(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        if self.active:
            if DEBUG_LSTLISTINGS:
                print("LST lst@outputspace start")
                self._dbg_last_added()
            self._last_added_buf.clear()
            self._in_outputspace = True

            def end(*_):
                self._in_outputspace = False

                pos = get_int_value("\\lst@pos")
                column = get_int_value("\\lst@column")
                length = get_int_value("\\lst@length")

                next_space = column - pos + length
                if DEBUG_LSTLISTINGS:
                    print(
                        f"LST lst@outputspace() end: lst@column={column} lst@pos={pos} lst@length={length}, space_pos={next_space}, last_space={self._last_space}"
                    )
                if len(self._last_added_buf) > 0:
                    if DEBUG_LSTLISTINGS:
                        print(f"outputspace set space (lastbuf) to {next_space - self._last_space}")
                    self._last_added_buf[-1].replacement_character = " " * (
                        next_space - self._last_space
                    )
                    self._last_added_buf.clear()
                elif self._last_glue is not None:
                    if DEBUG_LSTLISTINGS:
                        print(
                            f"outputspace set space (lastglue) to {next_space - self._last_space}"
                        )
                    self._last_glue.replacement_character = " " * (next_space - self._last_space)
                self._last_glue = None
                self._last_space = next_space
                # self._dbg_last_added()

            return end

    @end_macro_hook("lst@Kern", str_args=True)
    @assert2pagefail
    def on_lst_kern(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: Tuple[str | None, ...] | None,
        head_addr: int,
        tail_addr: int,
    ):
        if self.active:
            # Fetch the created kern box container (last in current head/tail)
            kern_container = None
            for kern_container in MemReader().iter_head_tail(head_addr, tail_addr):
                pass
            assert isinstance(kern_container, HBoxNode)
            # Fetch the actual kern
            container_list = list(kern_container.list)
            assert len(container_list) == 1
            kern_memnode = container_list[0]
            assert isinstance(kern_memnode, KernNode)
            kern_node = self._get_node(kern_memnode.addr, make_new=False)
            if DEBUG_LSTLISTINGS:
                print(f"LST lst@Kern({args}) End: {kern_memnode} {kern_node}")
            kern_node.replacement_character = ""

            self._last_glue = kern_node
            self._last_added_buf.append(kern_node)
            if DEBUG_LSTLISTINGS:
                self._last_added_line.append(kern_node)

    def on_glue_alloc(self, node: Node, glue_type: GlueSubtype):
        if self.active:
            if glue_type == GlueSubtype.normal:
                if DEBUG_LSTLISTINGS:
                    print("LST: append", node)
                    self._last_added_line.append(node)
                self._last_added_buf.append(node)
                # Kill all intermediate glues. If appropriate, they'll be readded by space processing.
                node.replacement_character = ""
                self._last_glue = node
            elif glue_type in (
                GlueSubtype.tab_skip_code,
                GlueSubtype.space_skip_code,
                GlueSubtype.xspace_skip_code,
                GlueSubtype.baseline_skip_code,
            ):
                # Make sure it does not output a space
                node.replacement_character = ""
            else:
                if DEBUG_LSTLISTINGS:
                    print("LST: append", node)
                    self._last_added_line.append(node)
                self._last_added_buf.append(node)
                self._last_glue = None
        elif glue_type == GlueSubtype.baseline_skip_code:
            self._last_nl_glue = node

    def on_char_alloc(self, node: Node):
        if self.active:
            if DEBUG_LSTLISTINGS:
                print("LST: append", node)
                self._last_added_line.append(node)
            self._last_added_buf.append(node)


class AlgorithmFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(AlgorithmElement()),
            name="algorithm",
        )


class AlgorithmLineNumberFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(LineNumberElement()),
            name="line_number",
        )


class AlgorithmIndentFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(AlgorithmIndentElement()),
            name="indent",
        )


@Rules.register
class Algorithm2eRule(Rule):
    """For \\begin{algorithm} for package algorithm2e."""

    is_in_par = capture_stack_element("par")
    indent: int = 0

    @begin_macro_hook("\\algorithm")
    @assert2pagefail
    def on_algorithm_start(self, *args):
        if self.active:
            # Don't nest
            return None

        if not hasattr(self, "package_loaded"):
            try:
                id_lookup(b"ver@algorithm2e.sty", False)
                self.package_loaded = True
            except KeyError:
                self.package_loaded = False
        if not self.package_loaded:
            return None

        self.active = True
        self.indent = 0

        self.formatter = AlgorithmFormatter()

        def end_algorithm(head_addr: int, tail_addr: int):
            if DEBUG_LSTLISTINGS:
                print("ALGORITHM: End Code")
            self.pop_formatter(self.formatter)
            self.active = False
            self.state.input_formatter_store_next = True

        def start_code(*_):
            if DEBUG_LSTLISTINGS:
                print("ALGORITHM: Start Code")
            self.push_formatter(self.formatter)
            # Ends when the currently active save group ends
            self.state.save_levels[-1].end_hooks.append(end_algorithm)

        return start_code

    @begin_macro_hook("algocf@latexcaption")
    @assert2pagefail
    def on_caption(self, *args):
        # Skip caption
        if not self.active:
            return

        if DEBUG_LSTLISTINGS:
            print("ALGOCF: caption")
        self.pop_formatter(self.formatter)

        def end_caption(*_):
            if DEBUG_LSTLISTINGS:
                print("ALGOCF: caption end")
            self.push_formatter(self.formatter)

        return end_caption

    @begin_macro_hook("algocf@push", str_args=True)
    @assert2pagefail
    def on_push(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: Tuple[str | None, ...] | None,
        head_addr: int,
        tail_addr: int,
    ):
        if not self.active:
            return

        if DEBUG_LSTLISTINGS:
            print(f"ALGOCF: push {args}")

        for node in MemReader().iter_head_tail(head_addr, tail_addr):
            print(f" - Node: {node}")

        def end_push(head_addr: int, tail_addr: int):
            if DEBUG_LSTLISTINGS:
                print("ALGOCF: push end")
                for node in MemReader().iter_head_tail(head_addr, tail_addr):
                    print(f" - Node: {node}")
                box = MemReader().get_node(get_cur_box())
                print(f" - Box: {box} -> indent={self.indent}")
            self.indent += 1

        return end_push

    @begin_macro_hook("algocf@pop", str_args=True)
    @assert2pagefail
    def on_pop(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: Tuple[str | None, ...] | None,
        head_addr: int,
        tail_addr: int,
    ):
        if not self.active:
            return

        self.indent -= 1

        if DEBUG_LSTLISTINGS:
            print(f"ALGOCF: pop {args}")

        for node in MemReader().iter_head_tail(head_addr, tail_addr):
            print(f" - Node: {node}")

        def end_pop(head_addr: int, tail_addr: int):
            if DEBUG_LSTLISTINGS:
                print("ALGOCF: pop end")
            box = MemReader().get_node(get_cur_box())
            self.register_node_action(
                get_cur_box(),
                direct_formatter=AlgorithmIndentFormatter(),
            )
            for node in MemReader().iter_head_tail(head_addr, tail_addr):
                print(f" - Node: {node}")
                print(f" - Box: {box} -> indent={self.indent}")

        return end_pop

    @begin_macro_hook("algocf@everypar")
    @assert2pagefail
    def on_par_begin(self, *args):
        if not self.active:
            return

        fmt = AlgorithmLineNumberFormatter()
        self.push_formatter(fmt)
        if DEBUG_LSTLISTINGS:
            print("ALGOCF: line number begin")

        def end_par_begin(*_):
            if DEBUG_LSTLISTINGS:
                print("ALGOCF: line number end")
            self.pop_formatter(fmt)

        return end_par_begin

    def on_glue_alloc(self, node: Node, glue_type: GlueSubtype):
        if self.active and self.is_in_par and glue_type == GlueSubtype.baseline_skip_code:
            if DEBUG_LSTLISTINGS:
                print("ALGOCF: linebreak")
            self.register_node_action(
                node.orig_addr,
                begin_action=lambda node, mem_node, cur_h, cur_v, struct_out: struct_out.push_pop(
                    LinebreakElement()
                ),
            )


@Rules.register
class AlgorithmicxBlockRule(Rule):
    """For \\begin{algorithmic} for algorithmicx package."""

    indent_value: int = 0
    line_prefix_active: bool = False
    nesting: int = 0

    indentation: list[AlgorithmIndentFormatter] = []

    @begin_macro_hook("ALG@beginalgorithmic")
    @assert2pagefail
    def on_algorithm_start(self, *args):
        # TODO: Kern nodes must be detected to save spaces properly
        if self.active:
            # Don't nest
            return None

        if not hasattr(self, "package_loaded"):
            try:
                id_lookup(b"ver@algorithmicx.sty", False)
                self.package_loaded = True
            except KeyError:
                self.package_loaded = False
        if not self.package_loaded:
            if DEBUG_LSTLISTINGS:
                print("package algorithmicx not loaded")
            return None

        self.active = True
        self.nesting = 0
        self.indentation.clear()

        formatter = AlgorithmFormatter()

        def end_algorithm(head_addr: int, tail_addr: int):
            while self.nesting > 0:
                self.nesting -= 1
                ind = self.indentation.pop()
                self.pop_formatter(ind)
            if DEBUG_LSTLISTINGS:
                print("ALGORITHMICX: End Code")
            self.pop_formatter(formatter)
            self.active = False
            self.state.input_formatter_store_next = True

        def start_code(*_):
            if DEBUG_LSTLISTINGS:
                print("ALGORITHMICX: Start Code")
            self.push_formatter(formatter)
            # Ends when the currently active save group ends
            self.state.save_levels[-1].end_hooks.append(end_algorithm)

            if DEBUG_LSTLISTINGS:
                print(f"Indent: {self.indent_value}")

        return start_code

    @begin_macro_hook("ALG@step")
    @assert2pagefail
    def on_lineno(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        if not self.active:
            return

        nesting = get_int_value("\\c@ALG@nested")
        while nesting > self.nesting:
            self.nesting += 1
            ind = AlgorithmIndentFormatter()
            self.indentation.append(ind)
            self.push_formatter(ind)
        while nesting < self.nesting:
            self.nesting -= 1
            ind = self.indentation.pop()
            self.pop_formatter(ind)

        fmt = AlgorithmLineNumberFormatter()
        self.push_formatter(fmt)

        if DEBUG_LSTLISTINGS:
            print(f"ALG@step begin, nesting={nesting}")

        def end_par_begin(*_):
            self.pop_formatter(fmt)
            if DEBUG_LSTLISTINGS:
                print("ALG@step end")
                print(f" - Box: {MemReader().get_node(get_cur_box())}")
                for node in MemReader().iter_head_tail(head_addr, tail_addr):
                    print(f" - Node: {node}")

        return end_par_begin


@Rules.register
class ProgramBlockRule(Rule):
    """For \\begin{program} for program package."""

    indent_value: int = 0
    line_prefix_active: bool = False
    nesting: int = 0

    indentation: list[AlgorithmIndentFormatter] = []

    @begin_macro_hook("program")
    @assert2pagefail
    def on_algorithm_start(self, *args):
        # TODO: Kern nodes must be detected to save spaces properly
        if self.active:
            # Don't nest
            return None

        if not hasattr(self, "package_loaded"):
            try:
                id_lookup(b"ver@program.sty", False)
                self.package_loaded = True
            except KeyError:
                self.package_loaded = False
        if not self.package_loaded:
            if DEBUG_LSTLISTINGS:
                print("package program not loaded")
            return None

        assert False, "Package 'program' not supported."

        self.active = True
        self.nesting = 0
        self.indentation.clear()

        self.formatter = AlgorithmFormatter()
        self.push_formatter(self.formatter)

    @begin_macro_hook("endprogram")
    @assert2pagefail
    def on_algo_end(self, *args):
        if not self.active:
            return

        while self.nesting > 0:
            self.nesting -= 1
            ind = self.indentation.pop()
            self.pop_formatter(ind)
        if DEBUG_LSTLISTINGS:
            print("PROGRAM: End Code")
        self.pop_formatter(self.formatter)
        self.active = False
        self.state.input_formatter_store_next = True

    @begin_macro_hook(["tab", "qtab"])
    @assert2pagefail
    def on_tab(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        if not self.active:
            return

        ind = AlgorithmIndentFormatter()
        self.indentation.append(ind)
        self.push_formatter(ind)

    @begin_macro_hook("untab")
    @assert2pagefail
    def on_untab(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        if not self.active:
            return

        self.nesting -= 1
        ind = self.indentation.pop()
        self.pop_formatter(ind)
