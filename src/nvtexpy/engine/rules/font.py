# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Type

from pdftex import get_cur_font, get_real_font

from nvtexpy.engine.common import CommandCode, FontProperties, Node, get_int_value, get_text
from nvtexpy.engine.debug import DEBUG_FONT
from nvtexpy.engine.rules.base import Rule, Rules, begin_macro_hook, end_macro_hook
from nvtexpy.engine.stack_formatter import CallbackFormatter, SuppressFormatter
from nvtexpy.output.structured_output import CodeElement, EmphasisElement, StrongElement


class StrongFormatter(CallbackFormatter):
    stack_unordered = True

    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(StrongElement()),
            name="strong",
        )


class EmphasisFormatter(CallbackFormatter):
    stack_unordered = True

    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(EmphasisElement()),
            name="emphasis",
        )


class CodeFormatter(CallbackFormatter):
    stack_unordered = True

    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(CodeElement()),
            name="code",
        )


@Rules.register
class FontRule(Rule):
    """For different char fonts."""

    strong = None
    emphasis = None
    code = None

    replaced_fonts = {
        "C70/gbsn/bx/n": "C70/gbsn/m/n",
    }
    fonts_by_name = {}

    current_font: int = 0

    def clear_format(self):
        if self.strong is not None:
            self.pop_formatter(self.strong)
            self.strong = None
        if self.emphasis is not None:
            self.pop_formatter(self.emphasis)
            self.emphasis = None
        if self.code is not None:
            self.pop_formatter(self.code)
            self.code = None

    def _update_format(
        self,
        node: Node,
        is_set: bool,
        formatter: CallbackFormatter | None,
        type: Type[StrongFormatter] | Type[EmphasisFormatter] | Type[CodeFormatter],
    ):
        was_set = formatter is not None
        if is_set and not was_set:
            formatter = type()
            self.push_formatter(formatter)
            if node.formatting is not None:
                node.formatting.append(formatter)
        elif not is_set and was_set:
            self.pop_formatter(formatter)
            if node.formatting is not None:
                node.formatting.remove(formatter)
            formatter = None
        return formatter

    def _update_font(self, node: Node, font_properties: FontProperties):
        self.strong = self._update_format(node, font_properties.bold, self.strong, StrongFormatter)
        self.emphasis = self._update_format(
            node, font_properties.italic_or_slanted, self.emphasis, EmphasisFormatter
        )
        self.code = self._update_format(node, font_properties.monospace, self.code, CodeFormatter)

    def on_char_alloc(self, node: Node):
        if node.font_properties is not None:
            self._update_font(node, node.font_properties)

    @begin_macro_hook("wrong@fontshape")
    def on_wrong_font_shape(self, *args):
        unsupported_fontshape = get_text("\\curr@fontshape")

        def end(*_):
            replacement_fontshape = get_text("\\curr@fontshape")
            print(f"wrong@fontshape: replace {unsupported_fontshape} by {replacement_fontshape}")
            self.replaced_fonts[unsupported_fontshape] = replacement_fontshape

        return end

    @end_macro_hook("selectfont")
    def on_select_font(self, *args):
        try:
            font = get_int_value(
                "\\font@name", expected_code=CommandCode.set_font, allow_dimension=False
            )
        except Exception:
            font = get_cur_font()
        realfont = get_real_font(font)
        if font in self.state.font_properties:
            props = self.state.font_properties[font]
        elif realfont != font and realfont in self.state.font_properties:
            props = self.state.font_properties[realfont]
            self.state.font_properties[font] = props
        else:
            encoding_val = get_text("\\f@encoding")
            fam_val = get_text("\\f@family")
            series_val = get_text("\\f@series")
            shape_val = get_text("\\f@shape")
            full_name = f"{encoding_val}/{fam_val}/{series_val}/{shape_val}"

            if full_name in self.replaced_fonts:
                real_full_name = self.replaced_fonts[full_name]
                if real_full_name not in self.fonts_by_name:
                    encoding_val, fam_val, series_val, shape_val = real_full_name.split("/")
                    props = FontProperties.from_tex_name(font, real_full_name)
                else:
                    props = self.fonts_by_name[real_full_name]
                    self.state.font_properties[font] = props

                if font != realfont:
                    self.state.font_properties[realfont] = props

                if DEBUG_FONT:
                    print(
                        f"Load font=0x{font:02X}=0x{realfont:02X} from {full_name} (using {real_full_name}): {props}"
                    )
            else:
                props = FontProperties.from_tex(font, series_val, shape_val, fam_val)
                self.state.font_properties[font] = props
                self.fonts_by_name[full_name] = props
                if font != realfont:
                    self.state.font_properties[realfont] = props
                if DEBUG_FONT:
                    print(f"Load font=0x{font:02X}=0x{realfont:02X} from {full_name}: {props}")
        if DEBUG_FONT:
            print(f"Activate font=0x{font:02X}=0x{realfont:02X}: {props}")
        self.state.current_text_font = props


@Rules.register
class CJKBoldRule(Rule):
    """For CJK bold font, for deduplicating bold-making characters."""

    # Somehow, this can be stacked...
    cjk_symbol_count = []
    cjk_primary_char = []
    in_symbol = False

    @begin_macro_hook("CJKsymbol ")
    def on_cjk_symbol(self, *args):
        # Enable within symbol
        self.cjk_symbol_count.append(0)
        self.cjk_primary_char.append([])
        print("CJK symbol begin")

        def end(*_):
            if self.cjk_symbol_count[-1] > 1:
                # Make symbol bold
                node: Node
                for node in self.cjk_primary_char[-1]:
                    if node.formatting is None:
                        node.formatting = [StrongFormatter()]
                    else:
                        # This is a case, where the strong will be removed by the formatting
                        for idx, fmt in enumerate(node.formatting):
                            if isinstance(fmt, (StrongFormatter, EmphasisFormatter, CodeFormatter)):
                                node.formatting.insert(idx, StrongFormatter())
                                break
                        else:
                            node.formatting.append(StrongFormatter())
                    print(f"CJK bold: {node}")
                return
            print("CJK symbol end")
            # Disable again
            self.cjk_symbol_count.pop()
            self.cjk_primary_char.pop()

        return end

    @begin_macro_hook("symbol ")
    def on_symbol(self, *args):
        # Next symbol is emitted
        if len(self.cjk_symbol_count) > 0:
            self.cjk_symbol_count[-1] += 1
            print(f"CJK next symbol: {self.cjk_symbol_count}")
        self.in_symbol = True

        def end(*_):
            self.in_symbol = False

        return end

    def on_char_alloc(self, node: Node):
        if self.in_symbol and len(self.cjk_symbol_count) > 0:
            if self.cjk_symbol_count[-1] < 2:
                # Store in case this'll be bold
                self.cjk_primary_char[-1].append(node)
            else:
                # This is a bold duplication, skip it.
                self.register_node_action(node.orig_addr, direct_formatter=SuppressFormatter())
                print(f"CJK bold skip: {node}")
                # node.replacement_character = ""
