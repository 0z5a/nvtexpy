# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Callable, Generator, List, Optional, Tuple, Type, TypeVar, Union

from nvtexpy.output.math import (
    ConcatITNode,
    GlueITNode,
    GraphicsITNode,
    InnerMathITNode,
    InnerTextITNode,
    ITNode,
    LeftITNode,
    MathCharCmdITNode,
    MathCharITNode,
    MathFunctionITNode,
    MathUnknownCharITNode,
    MidITNode,
    OrdITNode,
    OverlineStylingITNode,
    RightITNode,
    TableITNode,
    TextCharCmdITNode,
    TextCharITNode,
    TextRuleITNode,
    TextUnknownCharITNode,
    UnderlineStylingITNode,
    compact_math_node,
)
from nvtexpy.output.structured_output import (
    AlgorithmElement,
    AlgorithmIndentElement,
    AlignBlockElement,
    AlignBlockEqNumElement,
    BeginOfParagraphElement,
    BibliographyElement,
    BoundingBox,
    CharacterElement,
    CodeBlockElement,
    CodeElement,
    Element,
    EmphasisElement,
    EndOfParagraphElement,
    FloatElement,
    FooterElement,
    FootnoteElement,
    FootnoteMarkElement,
    GraphicsElement,
    HeaderElement,
    HeadingElement,
    HighlightElement,
    HyphenElement,
    LayoutBlockElement,
    LayoutbreakElement,
    LinebreakElement,
    LineNumberElement,
    ListItemElement,
    ListItemLabelElement,
    MathElement,
    OrderedListElement,
    OverlineElement,
    PageNumberElement,
    ParagraphbreakElement,
    ParagraphElement,
    RootElement,
    SpaceElement,
    StrikethroughElement,
    StrongElement,
    SubscriptElement,
    SuperscriptElement,
    SuppressElement,
    TableCellElement,
    TableElement,
    TableOfContentsElement,
    TableRowElement,
    UnderlineElement,
    UnorderedListElement,
    WordElement,
)

try:
    from nvtexpy.engine.debug import DEBUG_RESTRUCTURING
except ImportError:
    DEBUG_RESTRUCTURING = False


TElement = TypeVar("TElement", bound=Element)

######
# Base


element_order = {
    SpaceElement: 0,
    LinebreakElement: 1,
    LayoutbreakElement: 2,
    ParagraphbreakElement: 3,
}


def _clean_space_elements(node: Element) -> None:
    for child in node.children:
        _clean_space_elements(child)
    # In inline elements, extract leading/ending spaces
    i = 0
    while i < len(node.children):
        child = node.children[i]
        if child.is_inline:
            # Extract leading spaces
            while len(child.children) > 0:
                grandchild = child.children[0]
                if isinstance(
                    grandchild,
                    (SpaceElement, LinebreakElement, LayoutbreakElement, ParagraphbreakElement),
                ):
                    # print(f"Extracting leading space {grandchild!r} from {child!r}")
                    node.children.insert(i, child.children.pop(0))
                    i += 1
                else:
                    break
            i += 1
            # Extract trailing spaces
            while len(child.children) > 0:
                grandchild = child.children[-1]
                if isinstance(
                    grandchild,
                    (SpaceElement, LinebreakElement, LayoutbreakElement, ParagraphbreakElement),
                ):
                    # print(f"Extracting trailing space {grandchild!r} from {child!r}")
                    node.children.insert(i, child.children.pop(-1))
                    i += 1
                else:
                    break
        else:
            i += 1
    if not node.is_inline:
        # In block elements, strip start/end
        while len(node.children) > 0:
            child = node.children[0]
            if isinstance(
                child, (SpaceElement, LinebreakElement, LayoutbreakElement, ParagraphbreakElement)
            ):
                node.children.pop(0)
                continue
            else:
                break
        while len(node.children) > 0:
            child = node.children[-1]
            if isinstance(
                child, (SpaceElement, LinebreakElement, LayoutbreakElement, ParagraphbreakElement)
            ):
                node.children.pop()
                continue
            else:
                break
    # In all elements, compact spaces
    if len(node.children) > 1:
        i = 1
        while i < len(node.children):
            child = node.children[i]
            if isinstance(
                child,
                (SpaceElement, LinebreakElement, LayoutbreakElement, ParagraphbreakElement),
            ):
                last_child = node.children[i - 1]
                if isinstance(
                    last_child,
                    (SpaceElement, LinebreakElement, LayoutbreakElement, ParagraphbreakElement),
                ):
                    if element_order[type(child)] > element_order[type(last_child)]:
                        # Replace last child with this child, this has higher order
                        node.children.pop(i - 1)
                    else:
                        # Remove this child, it has lower/equal order
                        node.children.pop(i)
                    continue
            i += 1


def _paragraphbreak_to_paragraph(node: Element) -> None:
    if len(node.children) == 0:
        return
    if any(isinstance(child, ParagraphbreakElement) for child in node.children):
        new_children = []
        child_collector = []

        def _flush():
            # Extract last eop
            if len(child_collector) > 0 and isinstance(child_collector[-1], EndOfParagraphElement):
                eop = child_collector.pop()
            else:
                eop = None
            # Clean inner bop/eop
            i = 0
            while i < len(child_collector):
                if isinstance(child_collector[i], (BeginOfParagraphElement, EndOfParagraphElement)):
                    child_collector.pop(i)
                else:
                    i += 1
            # Create new paragraph
            if len(child_collector) > 0:
                new_children.append(ParagraphElement(children=child_collector))
            # Add eop
            if eop is not None:
                new_children.append(eop)

        for child in node.children:
            if isinstance(child, ParagraphbreakElement):
                # Strip start/end at paragraphs
                _flush()
                child_collector = []
            elif isinstance(child, BeginOfParagraphElement):
                # Add bop if at start, otherwise clean the bop
                if len(child_collector) == 0:
                    new_children.append(child)
            else:
                child_collector.append(child)
        if len(child_collector) > 0:
            _flush()
        node.children = new_children

    for child in node.children:
        _paragraphbreak_to_paragraph(child)


def _compact_extract_layout_blocks(node: Element) -> None:
    new_children = []
    for child in node.children:
        # if isinstance(child, ParagraphElement):
        if isinstance(child, (ParagraphElement, ListItemElement)):
            grandchildren = child.children
            if all(isinstance(grandchild, LayoutBlockElement) for grandchild in grandchildren):
                # Convert layoutblock elements to layoutbreaks
                grandchildren = []
                for grandchild in child.children:
                    grandchildren.extend(grandchild.children)
                    grandchildren.append(LayoutbreakElement())
                # Remove last layoutbreak
                if len(grandchildren) > 0:
                    grandchildren.pop()
            else:
                # Search for layout break, i.e. when the baseline of contained characters
                # is not same or increasing, register a layout break
                # TODO: Remove this, when the detection in the raw output is fixed
                last_baseline = 0
                idx = 0
                while idx < len(grandchildren):
                    grandchild = grandchildren[idx]
                    if isinstance(grandchild, WordElement):
                        letter_idx = 0
                        while letter_idx < len(grandchild.children):
                            charelement = grandchild.children[letter_idx]
                            assert len(charelement.boxes) == 1, (
                                f"Expected single box, got {charelement!r}"
                            )
                            box = charelement.boxes[0]
                            if box.y_baseline >= last_baseline:
                                last_baseline = box.y_baseline
                            else:
                                # Split word
                                if letter_idx > 0:
                                    grandchildren.insert(idx + 1, LayoutbreakElement())
                                    # Insert second part of word after layoutbreak
                                    new_word = WordElement(
                                        word="".join(
                                            c.character for c in grandchild.children[letter_idx:]
                                        )
                                    )
                                    new_word.children = grandchild.children[letter_idx:]
                                    grandchildren.insert(idx + 2, new_word)
                                    # First part of word stays
                                    grandchild.children = grandchild.children[:letter_idx]
                                    grandchild.word = "".join(
                                        c.character for c in grandchild.children
                                    )
                                    # Skip to inserted word
                                    idx += 1
                                else:
                                    grandchildren.insert(idx, LayoutbreakElement())
                                    # Redo the word
                                    idx -= 1
                                last_baseline = box.y_baseline
                                break
                            letter_idx += 1
                    if isinstance(grandchild, CharacterElement):
                        for box in grandchild.boxes:
                            if box.y_baseline > last_baseline:
                                last_baseline = box.y_baseline
                            elif box.y_baseline < last_baseline:
                                grandchildren.insert(idx, LayoutbreakElement())
                                break
                    idx += 1
            if isinstance(child, ParagraphElement):
                if all(
                    not grandchild.is_inline
                    or isinstance(grandchild, (LinebreakElement, LayoutbreakElement, SpaceElement))
                    for grandchild in grandchildren
                ):
                    # Extract children, all are block elements
                    new_children.extend(
                        grandchild
                        for grandchild in child.children
                        if not isinstance(
                            grandchild, (LayoutbreakElement, LinebreakElement, SpaceElement)
                        )
                    )
                elif any(
                    isinstance(grandchild, LayoutbreakElement) for grandchild in grandchildren
                ):
                    # There are mixed block elements and non-block elements in the middle of the
                    # paragraph, and layout breaks
                    layoutblock_collector = []
                    grandchild_collector = []

                    def flush_layoutblock():
                        nonlocal grandchild_collector
                        if grandchild_collector:
                            # Flush collected inline elements
                            layoutblock_collector.append(
                                LayoutBlockElement(children=grandchild_collector)
                            )
                            grandchild_collector = []

                    def flush_child():
                        nonlocal layoutblock_collector
                        if layoutblock_collector:
                            while len(layoutblock_collector[-1].children) > 0 and isinstance(
                                layoutblock_collector[-1].children[-1],
                                (SpaceElement, LinebreakElement),
                            ):
                                layoutblock_collector[-1].children.pop()
                            if len(layoutblock_collector) == 1:
                                new_children.append(
                                    ParagraphElement(children=layoutblock_collector[0].children)
                                )
                            elif len(layoutblock_collector) > 1:
                                new_children.append(
                                    ParagraphElement(children=layoutblock_collector)
                                )
                            layoutblock_collector = []

                    for grandchild in grandchildren:
                        if isinstance(grandchild, LayoutbreakElement):
                            flush_layoutblock()
                        elif grandchild.is_inline or isinstance(grandchild, ListItemLabelElement):
                            grandchild_collector.append(grandchild)
                        else:
                            flush_layoutblock()
                            flush_child()
                            new_children.append(grandchild)
                    flush_layoutblock()
                    flush_child()
                elif any(not grandchild.is_inline for grandchild in grandchildren):
                    # There are mixed block elements and non-block elements in the middle of the
                    # paragraph
                    grandchild_collector = []

                    def flush_child():
                        nonlocal grandchild_collector
                        if grandchild_collector:
                            if isinstance(
                                grandchild_collector[-1], (SpaceElement, LinebreakElement)
                            ):
                                grandchild_collector.pop()
                            # Flush collected inline elements
                            new_children.append(ParagraphElement(children=grandchild_collector))
                            grandchild_collector = []

                    for grandchild in grandchildren:
                        if grandchild.is_inline:
                            grandchild_collector.append(grandchild)
                        else:
                            flush_child()
                            new_children.append(grandchild)
                    flush_child()
                else:
                    # Keep paragraph as it is, it's just inline elements
                    new_children.append(child)
            else:
                assert isinstance(child, ListItemElement)
                if any(isinstance(grandchild, LayoutbreakElement) for grandchild in grandchildren):
                    # There are mixed block elements and non-block elements in the middle of the
                    # paragraph, and layout breaks
                    layoutblock_collector = []
                    grandchild_collector = []

                    def flush_layoutblock():
                        nonlocal grandchild_collector
                        if grandchild_collector:
                            # Flush collected inline elements
                            layoutblock_collector.append(
                                LayoutBlockElement(children=grandchild_collector)
                            )
                            grandchild_collector = []

                    def flush_child():
                        nonlocal layoutblock_collector
                        if layoutblock_collector:
                            while len(layoutblock_collector[-1].children) > 0 and isinstance(
                                layoutblock_collector[-1].children[-1],
                                (SpaceElement, LinebreakElement),
                            ):
                                layoutblock_collector[-1].children.pop()
                            if len(layoutblock_collector) == 1:
                                new_children.append(
                                    ListItemElement(children=layoutblock_collector[0].children)
                                )
                            elif len(layoutblock_collector) > 1:
                                new_children.append(ListItemElement(children=layoutblock_collector))
                            layoutblock_collector = []

                    for grandchild in grandchildren:
                        if isinstance(grandchild, LayoutbreakElement):
                            flush_layoutblock()
                        elif grandchild.is_inline or isinstance(grandchild, ListItemLabelElement):
                            grandchild_collector.append(grandchild)
                        else:
                            flush_layoutblock()
                            flush_child()
                            new_children.append(grandchild)
                    flush_layoutblock()
                    flush_child()
                else:
                    # Keep list item as it is, it's just inline elements
                    new_children.append(child)
        else:
            new_children.append(child)
    node.children = new_children


def _align_block_extract_rows(
    element: AlignBlockElement,
) -> Generator[List[Element], None, None]:
    if len(element.children) == 0:
        return
    if isinstance(element.children[0], TableElement):
        assert len(element.children) == 1
        for row in element.children[0].children:
            if isinstance(row, TableRowElement):
                assert all(
                    isinstance(
                        cell,
                        (
                            TableCellElement,
                            SpaceElement,
                            BeginOfParagraphElement,
                            EndOfParagraphElement,
                        ),
                    )
                    for cell in row.children
                )
                # Extract cells, or take space elements as they are
                yield [
                    inner_element
                    for cell in row.children
                    for inner_element in (
                        cell.children if isinstance(cell, TableCellElement) else (cell,)
                    )
                ]
            else:
                assert isinstance(
                    row,
                    (
                        SpaceElement,
                        LinebreakElement,
                        BeginOfParagraphElement,
                        EndOfParagraphElement,
                    ),
                ), f"Unexpected element {row}"
    elif isinstance(element.children[0], TableRowElement):
        assert all(
            isinstance(
                row,
                (
                    TableRowElement,
                    SpaceElement,
                    LinebreakElement,
                    BeginOfParagraphElement,
                    EndOfParagraphElement,
                ),
            )
            for row in element.children
        )
        for row in element.children:
            if isinstance(row, TableRowElement):
                assert all(
                    isinstance(
                        cell,
                        (
                            TableCellElement,
                            SpaceElement,
                            BeginOfParagraphElement,
                            EndOfParagraphElement,
                        ),
                    )
                    for cell in row.children
                )
                # Extract cells, or take space elements as they are
                yield [
                    inner_element
                    for cell in row.children
                    for inner_element in (
                        cell.children if isinstance(cell, TableCellElement) else (cell,)
                    )
                ]
            else:
                assert isinstance(
                    row,
                    (
                        SpaceElement,
                        LinebreakElement,
                        BeginOfParagraphElement,
                        EndOfParagraphElement,
                    ),
                ), f"Unexpected element {row}"
    elif isinstance(element.children[0], ParagraphElement):
        for paragraph_element in element.children:
            if isinstance(paragraph_element.children[0], LayoutBlockElement):
                for layout_block in paragraph_element.children:
                    yield layout_block.children
            else:
                yield paragraph_element.children
    else:
        yield element.children


def _clean_eq_num(node: Element) -> bool:
    """Remove AlignBlockEqNumElement from the tree by adding children directly. Returns if there was any"""
    res = False
    i = 0
    while i < len(node.children):
        child = node.children[i]
        if isinstance(child, AlignBlockEqNumElement):
            node.children.pop(i)
            node.children.extend(child.children)
            res = True
        else:
            i += 1
    return res


def _compact_extract_align_block(node: Element) -> None:
    # print(f"AlignBlock from: \n - " + "\n   ".join(str(c) for c in node.children))
    new_children = []
    for child in node.children:
        if isinstance(child, AlignBlockElement):
            # Make all display math!
            for el in _find_all_descendants(child, MathElement):
                el.is_display = True
            new_align_children = []
            for row_elements in _align_block_extract_rows(child):
                math_par = DisplayMathElement(
                    children=[BeginOfParagraphElement()] + row_elements + [EndOfParagraphElement()],
                    has_eq_num=False,
                )
                # Additionally merge space
                _compact_merge_space(math_par)
                _compact_extract_merge_math(math_par)
                math_par.has_eq_num = (
                    _clean_eq_num(math_par)
                    or any(mel.is_eq_num for mel in _find_all_descendants(math_par, MathElement))
                    or any(
                        mel.has_eq_num
                        for mel in _find_all_descendants(math_par, DisplayMathElement)
                    )
                )
                new_align_children.append(math_par)
            child.children = new_align_children

        new_children.append(child)
    node.children = new_children
    # print(f"AlignBlock: \n - " + "\n   ".join(str(c) for c in node.children))


def __math_can_extract(node: ITNode | None) -> bool:
    if isinstance(node, ConcatITNode):
        return all(__math_can_extract(child) for child in node.children)
    else:
        return isinstance(
            node, (InnerTextITNode, TextCharITNode, TextUnknownCharITNode, TextCharCmdITNode)
        )


def _math_can_extract(node: MathElement) -> bool:
    """
    Check if math element can be extracted, i.e. if no information will be lost if extracted.
    This is the case if the math element only contains characters or text, and no commands.
    """
    return __math_can_extract(node.code)


__math_whitelist_simple_char_cmds = {
    "\\#",
    "\\$",
    "\\%",
    "\\&",
    "\\_",
    "\\{",
    "\\}",
    "\\sim",
    "\\approx",
    "\\dagger",
    "\\ddagger",
    "\\S",
    "\\textbackslash",
    "\\rbrack",
    "\\lbrack",
    "\\backslash",
    "\\dots",
    # '\\prime',
    "\\star",
    "\\mathparagraph",
}


def __math_is_simple(node: ITNode | None) -> bool:
    """
    Check if math element is simple, i.e. no special structures are used, like fractions.
    Expects already compressed math.
    """
    if node is None:
        # print(f"{ind}XX: None -> True")
        return True
    if isinstance(node, ConcatITNode):
        # print(f"{ind}XX: Check concat")
        return all(__math_is_simple(child) for child in node.children)
    elif isinstance(node, InnerMathITNode):
        # print(f"{ind}XX: Check inner math")
        return __math_is_simple(node.child)
    elif isinstance(node, OrdITNode):
        if not __math_is_simple(node.nucleus):
            # print(f"{ind}XX: Check ord nucleus -> False")
            return False
        if node.subscr is not None:
            if node.supscr is not None:
                # Both won't work, but each individually is ok
                # print(f"{ind}XX: Check ord -> False")
                return False
            # print(f"{ind}XX: Check ord")
            return __math_is_simple(node.subscr)
        elif node.supscr is not None:
            # print(f"{ind}XX: Check ord")
            return __math_is_simple(node.supscr)
        # print(f"{ind}XX: Check ord -> True")
        return True
    elif isinstance(node, MathFunctionITNode) and node.function in (
        "\\cancel",
        "\\xcancel",
        "\\bcancel",
        "\\boxed",
    ):
        # print(f"{ind}XX: Check simple element types {type(node)}")
        # Special case: Cancel and boxed have a simple representation
        assert len(node.args) == 1 and not isinstance(node.args[0], str)
        return __math_is_simple(node.args[0])
    elif isinstance(node, (UnderlineStylingITNode, OverlineStylingITNode)):
        # print(f"{ind}XX: Check simple element types for {type(node)}")
        return __math_is_simple(node.child)
    else:
        # print(f"{ind}XX: Check simple element types for {node}")
        return isinstance(
            node,
            (
                GraphicsITNode,
                InnerTextITNode,
                TextCharITNode,
                TextCharCmdITNode,
                MathCharITNode,
                # MathCharCmdITNode,
                GlueITNode,
                LeftITNode,
                MidITNode,
                RightITNode,
                TextRuleITNode,
            ),
        ) or (
            isinstance(node, MathCharCmdITNode)
            and node.char_cmd in __math_whitelist_simple_char_cmds
        )
        # and not isinstance(node, (TextUnknownCharITNode, MathUnknownCharITNode))


def __math_is_graphics(node: ITNode | None) -> bool:
    """
    Check if math element is graphics, i.e. only consists of graphics commands.
    """
    if node is None:
        return False
    if isinstance(node, ConcatITNode):
        return len(node.children) > 0 and all(__math_is_graphics(child) for child in node.children)
    elif isinstance(node, (InnerTextITNode, InnerMathITNode)):
        return __math_is_graphics(node.child)
    else:
        return isinstance(node, GraphicsITNode)


def __math_is_table(node: ITNode | None) -> TableITNode | None:
    """
    Check if math element is a table, i.e. only consists of table commands.
    """
    if node is None:
        return None
    if isinstance(node, ConcatITNode):
        if len(node.children) != 1:
            return None
        return __math_is_table(node.children[0])
    elif isinstance(node, (InnerTextITNode, InnerMathITNode)):
        return __math_is_table(node.child)
    elif isinstance(node, TableITNode):
        return node
    return None


def __compact_extract_float_tables(node: Element) -> None:
    idx = 0
    while idx < len(node.children):
        child = node.children[idx]
        if isinstance(child, MathElement):
            table_node = __math_is_table(child.code)
            if table_node is not None:
                # Extract the table from the math element
                node.children[idx : idx + 1] = child.children
                idx += len(child.children)
                continue
        else:
            __compact_extract_float_tables(child)
        idx += 1


def _compact_extract_float_tables(node: Element) -> None:
    for child in node.children:
        if isinstance(child, FloatElement):
            if child.environment == "table":
                __compact_extract_float_tables(child)


def _compact_extract_merge_math(
    node: Element, ambiguous_function: Callable[[MathElement], bool] = _math_can_extract
) -> None:
    # print(f"XXX: Math Compacting: {node}")
    # Math mask contains: None for non-math, True for non-abiguous math (i.e. math for sure),
    # False for ambiguous math (i.e. this may be text or math, depending on neighbors)
    # [Text, Math(amb), Math(Non-amb), Math(amb), Math(Non-amb), Math(amb), Text]
    # -> [None, False, True, False, True, False, None]
    math_mask: List[Optional[bool]] = [None] * len(node.children)
    # Propagate mask from left
    # -> [None, False, True, True, True, True, None]
    last_was_ambiguous = True
    last_was_eq_num = False
    last_was_math = True
    idx = 0
    while idx < len(node.children):
        child = node.children[idx]
        if isinstance(child, MathElement):
            if not child.is_display and not child.is_eq_num and ambiguous_function(child):
                math_mask[idx] = not last_was_ambiguous
            else:
                math_mask[idx] = True
                last_was_ambiguous = False
            if child.is_eq_num:
                # If the last element was not an equation number, but this is, we need to merge
                # the math elements
                if not last_was_eq_num and idx > 0:
                    if isinstance(
                        node.children[idx - 1],
                        (
                            SpaceElement,
                            LinebreakElement,
                            BeginOfParagraphElement,
                            EndOfParagraphElement,
                        ),
                    ):
                        # Make last (which should be space or so) not math, such that this is separated
                        math_mask[idx - 1] = None
                    else:
                        # Insert a space node.
                        node.children.insert(idx, SpaceElement())
                        math_mask.insert(idx, None)
                        idx += 1
                # Keep the eq num.
                math_mask[idx] = True
            elif last_was_eq_num and idx > 0:
                # Switch from eq num to normal math
                # Make last (which should be space or so) not math, such that this is separated
                if isinstance(
                    node.children[idx - 1],
                    (
                        SpaceElement,
                        LinebreakElement,
                        BeginOfParagraphElement,
                        EndOfParagraphElement,
                    ),
                ):
                    math_mask[idx - 1] = None
                else:
                    # Insert a space node.
                    node.children.insert(idx, SpaceElement())
                    math_mask.insert(idx, None)
                    idx += 1

            last_was_eq_num = child.is_eq_num
            last_was_math = True
        elif (
            isinstance(
                child,
                (
                    SpaceElement,
                    LinebreakElement,
                    BeginOfParagraphElement,
                    EndOfParagraphElement,
                ),
            )
            and last_was_math
        ):
            # Space is ambiguous
            math_mask[idx] = not last_was_ambiguous
        else:
            last_was_math = False
            last_was_ambiguous = True
        idx += 1

    # print(f"XXX: Math mask ltr: {math_mask}")

    # Remove mask for space from right
    last_was_math = False
    last_was_eq_num = False
    for idx in range(len(node.children) - 1, -1, -1):
        child = node.children[idx]
        if isinstance(child, MathElement):
            last_was_math = True
            last_was_eq_num = child.is_eq_num
        elif (
            isinstance(
                child,
                (
                    SpaceElement,
                    LinebreakElement,
                    BeginOfParagraphElement,
                    EndOfParagraphElement,
                ),
            )
            and math_mask[idx] is not None
            and (not last_was_math or last_was_eq_num)
        ):
            # Space is not between math
            math_mask[idx] = None
        else:
            last_was_math = False

    # print(f"XXX: Math mask space rtl: {math_mask}")

    # Propagate mask from right
    # -> [None, True, True, True, True, True, None]
    last_mask = None
    for idx in range(len(node.children) - 1, -1, -1):
        if math_mask[idx] is False:
            math_mask[idx] = True if last_mask is True else None
        last_mask = math_mask[idx]
    # print(f"XXX: Math mask rtl: {math_mask}")
    # print(f"XXX: for \n  - " + "\n  - ".join(c.to_str(indent='    ') for c in node.children))

    # Now merge those math nodes
    new_children = []
    last_math = None
    for child, need_math in zip(node.children, math_mask):
        if isinstance(child, MathElement):
            if need_math is True:
                if last_math is None:
                    last_math = child
                    new_children.append(last_math)
                else:
                    # Merge sibling math elements
                    if isinstance(last_math.code, ConcatITNode):
                        last_math.code.children.append(child.code)
                    else:
                        last_math.code = ConcatITNode(
                            children=[
                                last_math.code,
                                child.code,
                            ]
                        )
                    last_math.children.extend(child.children)
                    last_math.is_display = last_math.is_display or child.is_display
                    last_math.boxes.extend(child.boxes)
            else:
                new_children.extend(child.children)
        elif (
            isinstance(
                child,
                (
                    SpaceElement,
                    LinebreakElement,
                    BeginOfParagraphElement,
                    EndOfParagraphElement,
                ),
            )
            and need_math is True
        ):
            # Skip this space, it's between math elements
            pass
        else:
            new_children.append(child)
            if last_math is not None:
                # Don't forget to simplify nested concats.
                last_math.code = compact_math_node(last_math.code, in_place=False)
                assert last_math.code is not None
            last_math = None
    if last_math is not None:
        # Don't forget to simplify nested concats.
        last_math.code = compact_math_node(last_math.code, in_place=False)
        assert last_math.code is not None
    if (
        isinstance(node.children[0], BeginOfParagraphElement)
        and new_children[0] != node.children[0]
    ):
        new_children.insert(0, node.children[0])
    if (
        isinstance(node.children[-1], EndOfParagraphElement)
        and new_children[-1] != node.children[-1]
    ):
        new_children.append(node.children[-1])
    node.children = new_children


def _compact_merge_space(node: Element) -> None:
    last_space = False
    last_linebreak = False

    # print(f"Compacting space: \n - " + "\n   ".join(str(c) for c in node.children))
    new_children = []
    # Collect all characters in this node
    for child in node.children:
        if isinstance(child, SpaceElement):
            if not last_space and not last_linebreak:
                last_space = True
                new_children.append(child)
        elif isinstance(child, LinebreakElement):
            if not last_linebreak:
                if last_space:
                    new_children.pop()
                    last_space = False
                last_linebreak = True
                new_children.append(child)
        else:
            last_space = False
            last_linebreak = False
            new_children.append(child)
    node.children = new_children
    # print(f"Compacted space: \n - " + "\n   ".join(str(c) for c in node.children))


def _compact_generate_words(node: Element) -> None:
    character_collector: List[Union[CharacterElement, HyphenElement]] = []
    new_children = []

    def flush_word():
        nonlocal character_collector
        if len(character_collector) > 0:
            # Create a word element from the collected characters, without hyphens
            word_element = WordElement(
                word="".join(
                    c.character for c in character_collector if not isinstance(c, HyphenElement)
                )
            )
            word_element.children = character_collector
            # Insert the word element before the current node
            new_children.append(word_element)

            # Remove the characters from the current node
            character_collector = []

    # Collect all characters in this node
    for child in node.children:
        if isinstance(child, (CharacterElement, HyphenElement)):
            character_collector.append(child)
        else:
            flush_word()
            new_children.append(child)
    flush_word()
    node.children = new_children


def _compact_merge_graphics_over_paragraphs(node: Element) -> None:
    new_children = []
    prev_graphics_child = None
    prev_graphics_child_index = None
    for child in node.children:
        if isinstance(child, ParagraphElement):
            # If a paragraph, merge starting graphics with the previous paragraph (if that ended
            # with a graphics element)
            if prev_graphics_child is not None and prev_graphics_child_index is not None:
                offset = 0
                # Skip spaces
                while offset < len(child.children) and isinstance(
                    child.children[offset], (LinebreakElement, SpaceElement)
                ):
                    offset += 1
                # While first (non-space) element is graphics, merge it with previous graphic
                while offset < len(child.children) and isinstance(
                    child.children[offset], GraphicsElement
                ):
                    # Merge this graphic with previous graphic
                    prev_graphics_child.children[prev_graphics_child_index].boxes.extend(
                        child.children[offset].boxes
                    )
                    # Remove intermediate spaces
                    del prev_graphics_child.children[prev_graphics_child_index + 1 :]
                    # Remove the graphic from this paragraph and all preceding spaces
                    del child.children[: offset + 1]
                    offset = 0
                    # Skip spaces
                    while offset < len(child.children) and isinstance(
                        child.children[offset], (LinebreakElement, SpaceElement)
                    ):
                        offset += 1
            # Search from end, get last graphics element (ignoring spaces)
            offset = len(child.children) - 1
            # Skip spaces
            while offset >= 0 and isinstance(
                child.children[offset], (LinebreakElement, SpaceElement)
            ):
                offset -= 1
            # Check if last (non-space) element is graphic and store it for the next paragraph
            if offset >= 0 and isinstance(child.children[offset], GraphicsElement):
                prev_graphics_child = child
                prev_graphics_child_index = offset
            elif len(child.children) > 0:
                prev_graphics_child = None
                prev_graphics_child_index = None
            # Else keep the prev_graphics_child, because this element will now be skipped.

            if len(child.children) > 0:
                new_children.append(child)
        else:
            new_children.append(child)
            prev_graphics_child = None
            prev_graphics_child_index = None
    node.children = new_children


def _compact_merge_graphics(node: Element):
    new_children = []
    last_graphics_idx = None
    for child in node.children:
        if isinstance(child, (LinebreakElement, SpaceElement)):
            new_children.append(child)
            continue
        if isinstance(child, GraphicsElement):
            if last_graphics_idx is not None:
                node.children[last_graphics_idx].boxes.extend(child.boxes)
                # Delete intermediate space elements
                del new_children[last_graphics_idx + 1 :]
            else:
                last_graphics_idx = len(new_children)
                new_children.append(child)
        else:
            new_children.append(child)
            last_graphics_idx = None
    node.children = new_children


def _compact_merge_paragraphs(node: HeadingElement) -> None:
    """Transform consecutive child paragraphs into one.
    For example, if the article title is two or more lines, each line arrives
    as a separate paragraph, which we want to combine into one.
    """

    def _clean_eop(_para: ParagraphElement) -> ParagraphElement:
        _para.children = [
            _clean_eop(c) for c in _para.children if not isinstance(c, EndOfParagraphElement)
        ]
        return _para

    new_children = []
    for child in node.children:
        if isinstance(child, ParagraphElement):
            if new_children and isinstance(new_children[-1], ParagraphElement):
                prev_para = new_children[-1]
                if not isinstance(prev_para.children[-1], EndOfParagraphElement):
                    # sometimes EndOfParagraphElement is nested deep within ParagraphElement,
                    # so propagate it back to top level, to be able to append before it
                    _clean_eop(prev_para)
                    prev_para.children.append(EndOfParagraphElement())
                for parachild in child.children:
                    if isinstance(parachild, BeginOfParagraphElement):
                        prev_para.children.insert(-1, SpaceElement())
                    elif isinstance(parachild, EndOfParagraphElement):
                        pass
                    else:
                        prev_para.children.insert(-1, parachild)
            else:
                new_children.append(child)
        else:
            new_children.append(child)
    node.children = new_children


def _compact_simplify_heading(node: HeadingElement) -> None:
    if (
        all(
            isinstance(child, (StrongElement, MathElement, LinebreakElement, SpaceElement))
            for child in node.children
        )
        or all(
            isinstance(child, (EmphasisElement, MathElement, LinebreakElement, SpaceElement))
            for child in node.children
        )
        or all(
            isinstance(child, (CodeElement, MathElement, LinebreakElement, SpaceElement))
            for child in node.children
        )
    ):
        # If the heading only contains a single formatting like bold, italic, etc. extract that
        new_children = []
        for child in node.children:
            if isinstance(child, (StrongElement, EmphasisElement, CodeElement)):
                for grandchild in child.children:
                    new_children.append(grandchild)
            else:
                new_children.append(child)
        node.children = new_children


def _remove_certain_empty_children(node: Element) -> None:
    new_children = []
    for child in node.children:
        if not isinstance(
            child,
            (
                RootElement,
                CharacterElement,
                HyphenElement,
                WordElement,
                SpaceElement,
                LinebreakElement,
                LayoutbreakElement,
                BeginOfParagraphElement,
                EndOfParagraphElement,
                GraphicsElement,
                TableRowElement,
                TableCellElement,
            ),
        ):
            if len(child.children) == 0 or all(
                isinstance(child, (SpaceElement, LinebreakElement, LayoutbreakElement))
                for child in child.children
            ):
                continue
        new_children.append(child)
    node.children = new_children


def _compact_split_bib_toc(node: Element) -> None:
    """
    Splits bibliography and table of contents into separate block-elements.
    E.g. if a bibliography contains multiple block-elements, make a bibliography-entry of each
    block-element.
    """
    new_children = []
    cur_collect_element_parent = None
    cur_collect_element = None

    def _flush():
        nonlocal cur_collect_element_parent, cur_collect_element
        if cur_collect_element is not None:
            new_children.append(type(cur_collect_element_parent)(children=cur_collect_element))
            cur_collect_element_parent = None
            cur_collect_element = None

    for child in node.children:
        if isinstance(child, (BibliographyElement, TableOfContentsElement)):
            for grandchild in child.children:
                if grandchild.is_inline:
                    if cur_collect_element_parent != child:
                        # Flush previous
                        _flush()
                    if cur_collect_element_parent is None or cur_collect_element_parent != child:
                        cur_collect_element_parent = child
                        cur_collect_element = [grandchild]
                    else:
                        assert cur_collect_element is not None
                        cur_collect_element.append(grandchild)
                else:
                    _flush()
                    new_children.append(type(child)(children=[grandchild]))
        else:
            _flush()
            new_children.append(child)
    _flush()
    node.children = new_children


def _compact_clean_table_row(row: TableRowElement) -> None:
    if not all(isinstance(child, TableCellElement) for child in row.children):
        new_children = []
        for child in row.children:
            if isinstance(child, TableCellElement):
                new_children.append(child)
            else:
                assert isinstance(child, (SpaceElement, LinebreakElement)), (
                    f"Unexpected {child} in table row"
                )
        row.children = new_children


def _compact_clean_table(table: TableElement) -> Element:
    i = 0
    while i < len(table.children):
        if isinstance(table.children[i], TableRowElement):
            i += 1
        else:
            assert isinstance(table.children[i], (SpaceElement, LinebreakElement)), (
                f"Unexpected {table.children[i]} in table"
            )
            del table.children[i]
    if len(table.children[0].children) == 1 and table.children[0].children[0].colspan == 1:
        # It's a one-column table
        new_children = []
        for row in table.children:
            assert isinstance(row, TableRowElement)
            assert len(row.children) == 1
            assert isinstance(row.children[0], TableCellElement)
            if len(new_children) > 0:
                new_children.append(LinebreakElement())
            new_children.extend(row.children[0].children)
        return ParagraphElement(children=new_children)
    return table


def _compact_concat_tables(node: Element) -> None:
    """Takes table rows and builds tables from them."""
    new_children = []
    cur_table = None
    num_cols = None
    rowspan_carryover_cells = {}
    for child in node.children:
        if isinstance(child, TableRowElement):
            _compact_clean_table_row(child)
            if cur_table is None:
                cur_table = TableElement(children=[child])
                num_cols = sum(
                    cell.colspan for cell in child.children if isinstance(cell, TableCellElement)
                )
                cur_col = 0
                rowspan_carryover_cells = {}
                for cell in child.children:
                    assert isinstance(cell, TableCellElement)
                    if cell.rowspan > 1:
                        for col in range(cur_col, cur_col + cell.colspan):
                            rowspan_carryover_cells[col] = cell.rowspan - 1
                    cur_col += cell.colspan
            else:
                cur_num_cols = 0
                for cell in child.children:
                    assert isinstance(cell, TableCellElement)
                    while cur_num_cols in rowspan_carryover_cells:
                        rowspan_carryover_cells[cur_num_cols] -= 1
                        if rowspan_carryover_cells[cur_num_cols] == 0:
                            del rowspan_carryover_cells[cur_num_cols]
                        cur_num_cols += 1
                    for col in range(cur_num_cols, cur_num_cols + cell.colspan):
                        assert col not in rowspan_carryover_cells, (
                            f"Rowspan carryover in table: {child}"
                        )
                    if cell.rowspan > 1:
                        for col in range(cur_num_cols, cur_num_cols + cell.colspan):
                            rowspan_carryover_cells[col] = cell.rowspan - 1
                    cur_num_cols += cell.colspan
                while cur_num_cols in rowspan_carryover_cells:
                    rowspan_carryover_cells[cur_num_cols] -= 1
                    if rowspan_carryover_cells[cur_num_cols] == 0:
                        del rowspan_carryover_cells[cur_num_cols]
                    cur_num_cols += 1
                assert cur_num_cols == num_cols, (
                    f"Table with different number of columns {cur_num_cols}!={num_cols}: {child}"
                )
                cur_table.children.append(child)
            if child.is_last:
                new_children.append(_compact_clean_table(cur_table))
                cur_table = None
        elif (
            isinstance(child, (SpaceElement, LinebreakElement, ParagraphbreakElement))
            and cur_table is not None
        ):
            # Well, we just suck up all the spacing.
            cur_table.children.append(child)
        else:
            new_children.append(child)
    if cur_table is not None:
        new_children.append(_compact_clean_table(cur_table))
    node.children = new_children


def _compact_in_place(node: Element):
    # Reformat AlignBlockElements: Extract math to lines of math
    if any(isinstance(child, AlignBlockElement) for child in node.children):
        _compact_extract_align_block(node)

    for child in node.children:
        _compact_in_place(child)

    if isinstance(node, MathElement):
        node.is_inline = True

    # Extract block elements from paragraphs (might split paragraphs into multiple paragraphs)
    if any(isinstance(child, (ParagraphElement, ListItemElement)) for child in node.children):
        _compact_extract_layout_blocks(node)

    # Merge table rows to tables
    if not isinstance(node, TableElement) and any(
        isinstance(child, TableRowElement) for child in node.children
    ):
        _compact_concat_tables(node)

    # A tabular withing a "table" float should be extracted for sure
    if any(isinstance(child, FloatElement) for child in node.children):
        _compact_extract_float_tables(node)

    # Merge math elements
    if any(isinstance(child, MathElement) for child in node.children):
        # Extract ambiguous math and merge sibling math elements
        _compact_extract_merge_math(node)

    # Compact bib and toc
    if any(
        isinstance(child, (BibliographyElement, TableOfContentsElement)) for child in node.children
    ):
        _compact_split_bib_toc(node)

    # Compact spaces and linebreaks
    if any(isinstance(child, (SpaceElement, LinebreakElement)) for child in node.children):
        _compact_merge_space(node)

    # If there are characters in this node, we need to compact them
    if any(isinstance(child, CharacterElement) for child in node.children):
        _compact_generate_words(node)

    if any(isinstance(child, ParagraphElement) for child in node.children):
        # Merge graphics elements over paragraphs
        _compact_merge_graphics_over_paragraphs(node)

    # If there are (multiple) graphics in this node, we need to compact them
    # by joining them (and maybe also join spaces between)
    if sum(1 if isinstance(child, GraphicsElement) else 0 for child in node.children) >= 2:
        _compact_merge_graphics(node)

    if isinstance(node, HeadingElement):
        _compact_merge_paragraphs(node)
        _compact_simplify_heading(node)

    _remove_certain_empty_children(node)


def compact_in_place(root: RootElement) -> None:
    """Compact the root element in place"""
    assert isinstance(root, RootElement)

    # Compact the heading levels to reflect a consistent leveling from h2 onwards
    def _find_headings(node: Element) -> Generator[HeadingElement, None, None]:
        for child in node.children:
            if isinstance(child, HeadingElement):
                yield child
            else:
                yield from _find_headings(child)

    # Level 1 is always kept
    heading_levels = {1}
    for heading in _find_headings(root):
        heading_levels.add(heading.level)
    heading_mapping = {
        heading_level: new_level
        for new_level, heading_level in enumerate(sorted(heading_levels), 1)
    }

    for heading in _find_headings(root):
        heading.level = heading_mapping[heading.level]

    # Pull out headers and footers and place all in the beginning and the end of the root element
    headers = []
    footers = []

    # First recursively search for all HeaderElement, remove them from the current place and add
    # them to the headers list
    def _extract_headers_and_footers(node: Element) -> None:
        nonlocal headers, footers
        i = 0
        while i < len(node.children):
            child = node.children[i]
            if isinstance(child, HeaderElement):
                headers.append(child)
                del node.children[i]
            elif isinstance(child, FooterElement):
                footers.append(child)
                del node.children[i]
            else:
                _extract_headers_and_footers(child)
                i += 1

    _extract_headers_and_footers(root)

    # Now merge the content of the headers and footers into one header and one footer element.
    # But drop the header or footer if it's empty
    new_root_children = []

    if headers:
        all_header_children = []
        for header in headers:
            all_header_children.extend(header.children)
        new_root_children.append(HeaderElement(children=all_header_children))

    new_root_children.extend(root.children)

    if footers:
        all_footer_children = []
        for footer in footers:
            if len(all_footer_children) > 0:
                all_footer_children.append(SpaceElement())
            all_footer_children.extend(footer.children)
        new_root_children.append(FooterElement(children=all_footer_children))

    root.children = new_root_children

    _extract_begin_end_of_paragraph(root)

    # Clean space elements
    _clean_space_elements(root)
    # Paragraphbreaks to paragraphs
    _paragraphbreak_to_paragraph(root)

    _compact_in_place(root)

    i = 0
    while i < len(root.children):
        if isinstance(root.children[i], (LinebreakElement, LayoutbreakElement)):
            del root.children[i]
        else:
            i += 1


######
## Flat structuring

UNK_TOKEN = "<|unk|>"


class CaptionElement(Element):
    """An element representing a caption, i.e. text in a float."""

    pass


class ToBeContinuedElement(Element):
    """Inserted during post-processing if a paragraph is continued on the next column or page."""

    is_inline = True


class DisplayMathElement(Element):
    """Inserted during post-processing if a paragraph is just math and equation number."""

    has_eq_num: bool

    def __init__(self, *args, has_eq_num: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_eq_num = has_eq_num


def _count_math_unknown_characters(node: ITNode | None) -> tuple[int, int]:
    if node is None:
        return 0, 0
    known_chars = 0
    unknown_chars = 0
    if isinstance(node, (MathUnknownCharITNode, TextUnknownCharITNode)):
        unknown_chars += 1
    elif isinstance(node, (MathCharITNode, TextCharITNode, MathCharCmdITNode, TextCharCmdITNode)):
        known_chars += 1
    elif isinstance(node, ConcatITNode):
        for child in node.children:
            inner_known_chars, inner_unknown_chars = _count_math_unknown_characters(child)
            known_chars += inner_known_chars
            unknown_chars += inner_unknown_chars
    else:
        for child in node.inner():
            inner_known_chars, inner_unknown_chars = _count_math_unknown_characters(child)
            known_chars += inner_known_chars
            unknown_chars += inner_unknown_chars
    return known_chars, unknown_chars


def _merge_unknown_characters(node: Element) -> tuple[int, int]:
    known_chars = 0
    unknown_chars = 0

    for child in node.children:
        inner_known_chars, inner_unknown_chars = _merge_unknown_characters(child)
        if isinstance(child, MathElement):
            inner_known_chars, inner_unknown_chars = _count_math_unknown_characters(child.code)
        known_chars += inner_known_chars
        unknown_chars += inner_unknown_chars
    if any(
        (isinstance(child, CharacterElement) and child.character.startswith("\\unknown"))
        for child in node.children
    ):
        new_children = []
        unknown_chars_collector = []

        def flush():
            nonlocal unknown_chars
            new_children.append(
                CharacterElement(
                    character=UNK_TOKEN,
                    box=BoundingBox.bboxes_union(
                        [box for c in unknown_chars_collector for box in c.boxes]
                    ),
                )
            )
            unknown_chars += len(unknown_chars_collector)

        for child in node.children:
            if isinstance(child, CharacterElement) and child.character.startswith("\\unknown"):
                unknown_chars_collector.append(child)
            else:
                if unknown_chars_collector:
                    flush()
                    unknown_chars_collector.clear()
                if isinstance(child, CharacterElement):
                    known_chars += 1
                new_children.append(child)
        if unknown_chars_collector:
            flush()

        if isinstance(node, WordElement):
            node.word = "".join(c.character for c in new_children)

        node.children = new_children
    else:
        known_chars += sum(1 for child in node.children if isinstance(child, CharacterElement))
    return known_chars, unknown_chars


def _is_all_math(node: Element):
    if isinstance(node, MathElement):
        return True
    if isinstance(node, (CharacterElement, WordElement, GraphicsElement)):
        return False
    return all(_is_all_math(child) for child in node.children)


def _has_math(node: Element):
    if isinstance(node, MathElement):
        return True
    return any(_has_math(child) for child in node.children)


def _math_node_has_unknown(node: Element) -> bool:
    if isinstance(node, CharacterElement) and node.character.startswith("\\unknown{"):
        return True
    if isinstance(node, WordElement) and ("\\unknown{" in node.word or UNK_TOKEN in node.word):
        return True
    # Do not recurse inner math elements
    return any(
        _math_node_has_unknown(child)
        for child in node.children
        if not isinstance(child, MathElement)
    )


def _has_displaymath(node: Element):
    if isinstance(node, DisplayMathElement):
        return True
    return any(_has_displaymath(child) for child in node.children)


def _dln_join_math(node: Element):
    for child in node.children:
        _dln_join_math(child)
    if any(isinstance(child, MathElement) for child in node.children):
        _compact_extract_merge_math(node, _should_extract_math)


def _unpack_math(node: Element) -> Element:
    """Converts all-math nodes to a single MathElement."""
    if isinstance(node, RootElement):
        for idx in range(len(node.children)):
            node.children[idx] = _unpack_math(node.children[idx])
        return node
    elif isinstance(node, DisplayMathElement):
        # This is already converted to block
        # Maybe extract text from eq num, if it's extractable.
        if node.has_eq_num:
            idx = 0
            while idx < len(node.children):
                child = node.children[idx]
                if (
                    isinstance(child, MathElement)
                    and child.is_eq_num
                    and __math_can_extract(child.code)
                ):
                    node.children[idx : idx + 1] = child.children
                else:
                    idx += 1
        return node
    elif (
        not isinstance(node, MathElement)
        and _is_all_math(node)
        and _has_math(node)
        and not _has_displaymath(node)
    ):
        cur_math = []
        cur_children = []
        is_display = False
        is_eq_num = False
        has_eq_num = False
        new_children = []
        pending_space = []

        def _flush():
            nonlocal cur_math, cur_children, is_display, is_eq_num
            if len(cur_math) > 0:
                code = compact_math_node(ConcatITNode(children=cur_math), in_place=False)
                if (
                    __math_is_simple(code)
                    and not is_display
                    and not any(_math_node_has_unknown(child) for child in cur_children)
                ) or (__math_can_extract(code) and is_eq_num):
                    # print("XX: Extracting math", code, cur_children)
                    new_children.extend(cur_children)
                    if pending_space:
                        new_children.extend(pending_space)
                else:
                    if code is not None:
                        math_el = MathElement(
                            code=code,
                            is_display=is_display,
                            is_eq_num=is_eq_num,
                        )
                        math_el.children = cur_children
                        new_children.append(math_el)
                    if pending_space:
                        new_children.extend(pending_space)
                cur_math = []
                cur_children = []
                is_display = False

        for child in _find_all_descendants(
            node,
            (MathElement, SpaceElement, LinebreakElement),
            stop_at=(MathElement, DisplayMathElement),
        ):
            if isinstance(child, MathElement):
                if is_eq_num != child.is_eq_num:
                    _flush()
                    is_eq_num = child.is_eq_num
                cur_children.extend(child.children)
                cur_math.append(child.code)
                is_display |= child.is_display
                has_eq_num |= child.is_eq_num
                pending_space.clear()
            else:
                pending_space.append(child)
        _flush()
        combined_math = DisplayMathElement(children=new_children, has_eq_num=has_eq_num)

        if isinstance(node, (ListItemElement, TableCellElement, CaptionElement)):
            # If the math is in a list item or table cell, we need to wrap it in a paragraph
            node.children = [combined_math]
            return node

        # Replace current element by math element
        return combined_math
    elif isinstance(
        node,
        (
            ParagraphElement,
            LayoutBlockElement,
            CaptionElement,
            ListItemElement,
            TableCellElement,
            BibliographyElement,
            TableOfContentsElement,
            AlignBlockElement,
        ),
    ):
        for idx in range(len(node.children)):
            node.children[idx] = _unpack_math(node.children[idx])
        return node
    return node


def _should_extract_math(node: MathElement) -> bool:
    """Check if a MathElement should be extracted from its parent."""
    if _math_node_has_unknown(node):
        return False
    if __math_is_simple(node.code) and not node.is_display:
        return True
    if __math_is_graphics(node.code) and not node.is_display:
        return True
    table = __math_is_table(node.code)
    if table is not None:
        if DEBUG_RESTRUCTURING:
            print(f"RST: Table in math node {node}")
        non_math = 0
        math = 0
        for row in table.data:
            for cell in row.cells:
                if cell.child is None or cell.child.is_empty():
                    if DEBUG_RESTRUCTURING:
                        print(f"RST: Table Cell in math is empty: {cell.child}")
                    continue
                if __math_is_simple(cell.child):
                    if DEBUG_RESTRUCTURING:
                        print(f"RST: Table Cell in math is simple: {cell.child}")
                    non_math += 1
                else:
                    if DEBUG_RESTRUCTURING:
                        print(f"RST: Table Cell in math is non-simple: {cell.child}")
                    math += 1
        if non_math > math:
            if DEBUG_RESTRUCTURING:
                print(
                    f"RST: Table in math is mostly non math ({non_math} non-math VS {math} math cells) -> extract"
                )
            return True
        if non_math == 0:
            if DEBUG_RESTRUCTURING:
                print(f"RST: Table in math is only math ({math} math cells) -> merge")
            # Merge the table into a single math element and extract it! There is only math in the
            # table and only one table in the math element.
            # Well, it might also be an actual alignment, but rather unlikely?
            node.children = [
                child
                for cell in _find_all_descendants(
                    node.children[0], TableCellElement, stop_at=TableCellElement
                )
                for child in cell.children
            ]
            return False
        if DEBUG_RESTRUCTURING:
            print(f"RST: Table in math is mostly math ({non_math} non-math VS {math} math) -> keep")
    return False


def _treat_math_simple_nonsimple(node: Element, is_in_table_float: bool = False):
    """MathElements which are simple, shall be unpacked and replaced
    by their children. Non-simple MathElements need to have their children removed."""

    new_children = []
    for child in node.children:
        if isinstance(child, MathElement):
            if _should_extract_math(child):
                if DEBUG_RESTRUCTURING:
                    print("RST: Extract math node", child)
                # Extract this node
                for grandchild in child.children:
                    _treat_math_simple_nonsimple(grandchild, is_in_table_float=is_in_table_float)
                new_children.extend(child.children)
            else:
                if DEBUG_RESTRUCTURING:
                    print("RST: Keep math node", child)
                # Remove the children of non-simple math elements
                # but keep bbox of those elements
                box = child.compute_bounding_box()
                if box is None:
                    # No box -> Element non-existent, empty (might be rule or so)
                    continue
                child.boxes = [box]
                child.children = []

                _treat_math_simple_nonsimple(child, is_in_table_float)
                new_children.append(child)
        else:
            if isinstance(child, FloatElement) and child.environment == "table":
                next_is_in_table_float = True
            else:
                next_is_in_table_float = is_in_table_float
            _treat_math_simple_nonsimple(child, next_is_in_table_float)
            new_children.append(child)

    node.children = new_children


def _convert_layoutblocks(node: Element):
    """Convert a Paragraph that contains multiple LayoutBlockElements to multiple ParagraphElements."""

    for child in node.children:
        _convert_layoutblocks(child)

    if any(isinstance(child, LayoutBlockElement) for child in node.children):
        # Convert LayoutBlockElement to ParagraphElement
        new_children = []
        for child in node.children:
            if isinstance(child, LayoutBlockElement):
                new_children.append(ParagraphElement(child.children))
            else:
                new_children.append(child)
        node.children = new_children


def _check_graphics_in_table_float(node: Element):
    """Check if a GraphicsElement is inside a FloatElement."""
    if isinstance(node, FloatElement) and node.environment == "table":
        if DEBUG_RESTRUCTURING:
            if not any(c for c in _find_all_descendants(node, TableElement)):
                print("\\begin{table} without tabular, likely an image table, skipping page")
                print(node)
        assert any(c for c in _find_all_descendants(node, TableElement)), (
            "\\begin{table} without tabular, likely an image table, skipping page"
        )

    for child in node.children:
        _check_graphics_in_table_float(child)


def _extract_elements_inplace(
    node: Element,
    tps: Union[Tuple[Type[Element], ...], Type[Element]],
    permeable_element: Callable[[Element], bool] = lambda x: True,
):
    """
    Split out elements of the given type and add them to the parent node.
    May duplicate parents in order to perform the split.
    Example:
    <other 1>
        <other 1.1>
        <tp 1.2>
            <other 1.2.1>
                <tp 1.2.1.1>
        <other 1.3>

    becomes:
    <other 1>
        <other 1.1>
    <tp 1.2>
        <other 1.2.1>
    <tp 1.2.1.1>
    <other 1>
        <other 1.3>

    starting at the innermost element first, recursively.

    Args:
        node: The node to search in
        tps: The types of element to extract
        permeable_element: If true, then extract through this element, otherwise stop at this
            element, do not extract further inside.
    """
    for child in node.children:
        _extract_elements_inplace(child, tps, permeable_element)

    # dbg_cnt = 0

    if any(isinstance(grandchild, tps) for child in node.children for grandchild in child.children):
        # print(f"Extracting children for {node} of type {tps}", flush=True)
        new_children = []
        for child in node.children:
            if not permeable_element(child):
                # Cannot recurse.
                new_children.append(child)
                # print(f"Skip child {child}", flush=True)
            elif any(isinstance(grandchild, tps) for grandchild in child.children):
                # This child may need splitting (or just from start/end without being split)
                had_non_tp = False
                gc_idx = 0
                while gc_idx < len(child.children):
                    # dbg_cnt += 1
                    # assert dbg_cnt < 1000, "Infinite loop?"
                    grandchild = child.children[gc_idx]
                    if isinstance(grandchild, tps):
                        if had_non_tp:
                            # Uhoh, split the child into two halfes, ouch! Both keep their
                            # properties, except the grandchildren.
                            # Shallow copy, keeping the original child as first half
                            next_child = copy.copy(child)
                            # First half
                            child.children = child.children[:gc_idx]
                            new_children.append(child)
                            # Add current grandchild to parent
                            new_children.append(grandchild)
                            # Second half continues
                            del next_child.children[: gc_idx + 1]
                            # print(f"Split and extract {grandchild} from {child} and {next_child}", flush=True)
                            # Continue to search
                            had_non_tp = False
                            gc_idx = 0
                            child = next_child
                            # gc_idx now points to first element of new child
                        else:
                            # Just remove that child and add to parent directly
                            child.children.pop(gc_idx)
                            new_children.append(grandchild)
                            # No change of gc_idx, next element is at the same index
                            # print(f"Extract {grandchild} from {child}[0]", flush=True)
                    else:
                        # Now, if another tp follows, we need to split
                        had_non_tp = True
                        gc_idx += 1
                # Add what remains of the child
                if len(child.children) > 0:
                    new_children.append(child)
            else:
                new_children.append(child)
        node.children = new_children


def _unpack_floats(node: Element):
    """Replace FloatElements with their children. Any text will become a CaptionElement."""
    # Replace by CaptionElements
    new_children = []
    for child in node.children:
        if isinstance(child, FloatElement):
            caption_collector = []
            for grandchild in child.children:
                if isinstance(grandchild, ParagraphElement):
                    if len(caption_collector) > 0:
                        new_children.append(CaptionElement(caption_collector))
                        caption_collector = []
                    # Convert the paragraph to a CaptionElement
                    new_children.append(CaptionElement(grandchild.children))
                else:
                    caption_collector.append(grandchild)
            if len(caption_collector) > 0:
                new_children.append(CaptionElement(caption_collector))
        else:
            new_children.append(child)

    node.children = new_children


def _unpack_list_items(node: Element):
    """Unpack lists and only keep the list items. Even for nested lists."""
    # Except inside Float or Table

    if any(isinstance(c, ListItemElement) for c in node.children):
        new_children = []
        for child in node.children:
            if isinstance(child, ListItemElement):
                label_container = child
                if len(child.children) > 0 and isinstance(
                    child.children[0], (LayoutBlockElement, ParagraphElement)
                ):
                    label_container = child.children[0]
                if len(label_container.children) > 0 and isinstance(
                    label_container.children[0], ListItemLabelElement
                ):
                    label_element = label_container.children.pop(0)
                    # If this child is a paragraph, extract it.
                    if len(label_element.children) == 1 and isinstance(
                        label_element.children[0], ParagraphElement
                    ):
                        label_element = label_element.children[0]
                else:
                    label_element = None
                if len(child.children) > 0 and isinstance(
                    child.children[0], (ParagraphElement, LayoutBlockElement)
                ):
                    # First par is list item content, other pars are separate pars
                    content_element = child.children.pop(0)
                    extract_elements = child.children
                else:
                    content_element = child
                    extract_elements = None
                if label_element:
                    # Merge label element with content element
                    new_element = ListItemElement(
                        [label_element, SpaceElement()] + content_element.children
                    )
                    new_element.boxes = content_element.boxes
                    new_children.append(new_element)
                else:
                    # Content element
                    new_element = ListItemElement(content_element.children)
                    new_element.boxes = content_element.boxes
                    new_children.append(new_element)
                if extract_elements is not None:
                    if any(grandchild.is_inline for grandchild in extract_elements):
                        if any(not grandchild.is_inline for grandchild in extract_elements):
                            # Create paragraphs for inline elements
                            grandchild_collector = []

                            def _flush():
                                nonlocal grandchild_collector
                                if grandchild_collector:
                                    new_children.append(ParagraphElement(grandchild_collector))
                                    grandchild_collector = []

                            for subchild in extract_elements:
                                if subchild.is_inline:
                                    grandchild_collector.append(subchild)
                                else:
                                    _flush()
                                    new_children.append(subchild)
                            _flush()
                        else:
                            new_children.append(ParagraphElement(extract_elements))
                    else:
                        new_children.extend(extract_elements)
            else:
                new_children.append(child)
        node.children = new_children


def _is_empty(node: Element):
    if isinstance(node, (CharacterElement, WordElement, GraphicsElement)):
        return False
    return all(_is_empty(child) for child in node.children)


def _purge_lists(node: Element):
    """Remove all list elements from the tree."""
    for child in node.children:
        _purge_lists(child)

    if any(
        isinstance(child, (OrderedListElement, UnorderedListElement)) for child in node.children
    ):
        new_children = []
        for child in node.children:
            if isinstance(child, (OrderedListElement, UnorderedListElement)):
                assert _is_empty(child), f"List element {child} is not empty"
                continue
            new_children.append(child)
        node.children = new_children


def _filter_heading_bold(element: Element):
    """Filter out bold elements from headings."""

    if isinstance(element, HeadingElement):
        new_children = []
        for child in element.children:
            if isinstance(child, StrongElement):
                new_children.extend(child.children)
            else:
                new_children.append(child)
        element.children = new_children

    for child in element.children:
        _filter_heading_bold(child)


def _remove_code_block_line_numbers(node: Element):
    """Remove all LineNumberElements from the tree."""

    for child in node.children:
        _remove_code_block_line_numbers(child)

    if any(isinstance(child, LineNumberElement) for child in node.children):
        new_children = []
        for child in node.children:
            if isinstance(child, LineNumberElement):
                continue
            new_children.append(child)
        node.children = new_children


def _to_plain(node: Element):
    """Remove all emphasis/strong/code formatting from the tree."""

    new_children = []
    for child in node.children:
        if isinstance(
            child,
            (
                EmphasisElement,
                StrongElement,
                CodeElement,
                StrikethroughElement,
                UnderlineElement,
                OverlineElement,
                HighlightElement,
            ),
        ):
            new_children.extend(child.children)
        else:
            _to_plain(child)
            new_children.append(child)

    node.children = new_children


def _update_code_blocks(root: RootElement):
    """Remove all LineNumberElements from the tree and clean from inner formatting."""

    for node in root.children:
        if isinstance(node, (CodeBlockElement, AlgorithmElement)):
            _remove_code_block_line_numbers(node)
            new_children = []
            for child in node.children:
                if isinstance(child, ParagraphElement):
                    _to_plain(child)
                    new_children.append(child)
                else:
                    new_children.append(ParagraphElement([child]))

            node.children = new_children


def _remove_empty(node: Element):
    """Remove certain elements that are empty or only contain empty elements."""

    removable_elements = (
        AlignBlockElement,
        ParagraphElement,
        ListItemLabelElement,
        ListItemElement,
        UnorderedListElement,
        OrderedListElement,
        BibliographyElement,
        TableOfContentsElement,
        HeadingElement,
    )

    for child in node.children:
        _remove_empty(child)

    new_children = []
    for child in node.children:
        if isinstance(child, removable_elements):
            if len(child.children) == 0 or all(
                isinstance(
                    child.children,
                    (
                        ToBeContinuedElement,
                        BeginOfParagraphElement,
                        EndOfParagraphElement,
                    ),
                )
                for child in child.children
            ):
                continue
        new_children.append(child)

    node.children = new_children


def _hierarchy_check(root_node: Element):
    """Checks that only certain elements are at a non-root level."""

    forbidden_elements = (
        CharacterElement,
        LayoutBlockElement,
        OrderedListElement,
        UnorderedListElement,
        SpaceElement,
        SuppressElement,
        WordElement,
        FloatElement,
    )

    root_elements = (
        BibliographyElement,
        CaptionElement,
        CodeBlockElement,
        AlgorithmElement,
        FooterElement,
        FootnoteElement,
        GraphicsElement,
        HeaderElement,
        # HeadingElement,
        ListItemElement,
        # ParagraphElement,
        TableElement,
        TableOfContentsElement,
    )

    non_root_elements = (
        CodeElement,
        EmphasisElement,
        UnderlineElement,
        OverlineElement,
        StrikethroughElement,
        BeginOfParagraphElement,
        EndOfParagraphElement,
        FootnoteMarkElement,
        HyphenElement,
        LinebreakElement,
        ListItemLabelElement,
        PageNumberElement,
        StrongElement,
        SubscriptElement,
        SuperscriptElement,
        TableCellElement,
        TableRowElement,
        LineNumberElement,
    )

    # Check that only certain elements are at the root level
    for child in root_node.children:
        if isinstance(child, forbidden_elements) or isinstance(child, non_root_elements):
            raise ValueError(f"Element {type(child)} is not allowed at the root level")

    # Check all non-root elements
    for root_child in root_node.children:
        for child in _find_all_descendants(root_child, Element):
            if isinstance(child, root_elements):
                raise ValueError(f"Element {type(child)} is not allowed at a non-root level")


def _inline_algorithm_indent(node: Element, indent_level: int = 0):
    i = 0
    while i < len(node.children):
        child = node.children[i]
        if isinstance(child, AlgorithmIndentElement):
            _inline_algorithm_indent(child, indent_level + 1)
            # Extract children
            node.children[i : i + 1] = child.children
            i += len(child.children)
            continue
        else:
            _inline_algorithm_indent(child, indent_level)
            if not child.is_inline and any(grandchild.is_inline for grandchild in child.children):
                if indent_level > 0:
                    child.children.insert(0, WordElement("  " * indent_level))
        i += 1


def _purge_begin_end_of_paragraph(element: Element):
    """Remove all BeginOfParagraphElements and EndOfParagraphElements from the tree."""

    for child in element.children:
        _purge_begin_end_of_paragraph(child)

    if any(
        isinstance(child, (BeginOfParagraphElement, EndOfParagraphElement))
        for child in element.children
    ):
        new_children = []
        for idx, child in enumerate(element.children):
            if isinstance(child, (BeginOfParagraphElement, EndOfParagraphElement)):
                continue
            new_children.append(child)
        element.children = new_children


def _extract_begin_end_of_paragraph(root: RootElement):
    """Move all Begin/EndOfParagraph to the outmost element if at start/end of it."""
    new_children = []
    for node in root.children:
        first_bop = node
        while first_bop is not None:
            if isinstance(
                first_bop,
                (
                    FooterElement,
                    HeaderElement,
                    FloatElement,
                    CaptionElement,
                    GraphicsElement,
                ),
            ):
                # Those floating elements shall be ignored
                first_bop = None
                break
            if len(first_bop.children) > 0:
                first_child = first_bop.children[0]
                if isinstance(first_child, BeginOfParagraphElement):
                    first_bop = first_bop.children.pop(0)
                    break
                else:
                    # Look inside, in case it's nested formatting
                    first_bop = first_child
            else:
                first_bop = None
                break
        if first_bop is not None:
            new_children.append(first_bop)

        new_children.append(node)

        last_eop = node
        while last_eop is not None:
            if isinstance(
                last_eop,
                (
                    FooterElement,
                    HeaderElement,
                    FloatElement,
                    CaptionElement,
                    GraphicsElement,
                ),
            ):
                # Those floating elements shall be ignored
                last_eop = None
                break
            if len(last_eop.children) > 0:
                last_child = last_eop.children[-1]
                if isinstance(last_child, EndOfParagraphElement):
                    last_eop = last_eop.children.pop()
                    break
                else:
                    # Look inside, in case it's nested formatting
                    last_eop = last_child
            else:
                last_eop = None

        if last_eop is not None:
            new_children.append(last_eop)

        # Remove all other Begin/EndOfParagraph
        _purge_begin_end_of_paragraph(node)
    root.children = new_children


def _update_begin_end_of_paragraph(root: RootElement):
    """Takes EOP/BOP at root level and moves TBC into elements where missing.
    Move outside EOP/BOP into siblings. Remove all others.
    """
    # print("-" * 20)
    # print("BOP/EOP mid")
    # print(root)

    # Compress BOP/EOP like this:
    # If there is EOP/BOP and a following element that cares, add EOP/BOP to that element.
    last_was_ended = False
    last_tbc_element = None
    idx = 0
    while idx < len(root.children):
        child = root.children[idx]
        if isinstance(child, (BeginOfParagraphElement, EndOfParagraphElement)):
            # Remove and store that this was here
            last_was_ended = True
            del root.children[idx]
            # No increment, next element is now at this index
        else:
            if isinstance(child, ParagraphElement) and not last_was_ended:
                child.children.insert(0, ToBeContinuedElement())

            if (
                last_tbc_element is not None
                and not last_was_ended
                and isinstance(
                    child,
                    (
                        ParagraphElement,
                        ListItemElement,
                        BibliographyElement,
                        TableOfContentsElement,
                    ),
                )
            ):
                # This is a text-flow element, add TBC if there was no interruption
                last_tbc_element.children.append(ToBeContinuedElement())

            if isinstance(child, HeadingElement):
                # These elements interrupt the flow of the document without the need of end
                last_was_ended = True
                last_tbc_element = None
            if isinstance(
                child,
                (
                    ParagraphElement,
                    ListItemElement,
                    BibliographyElement,
                    TableOfContentsElement,
                ),
            ):
                # This element may continue on the next page / column
                last_tbc_element = child
                last_was_ended = False
            idx += 1
    if not last_was_ended and last_tbc_element is not None:
        last_tbc_element.children.append(ToBeContinuedElement())


def _simple_merge_math(root: RootElement):
    """Merge sibling math blocks and insert a linebreak in between."""
    i = 0
    while i < len(root.children):
        child = root.children[i]
        if isinstance(child, DisplayMathElement) and not child.has_eq_num:
            j = i + 1
            while j < len(root.children) and isinstance(root.children[j], DisplayMathElement):
                dmath: DisplayMathElement = root.children.pop(j)
                child.children.append(LinebreakElement())
                child.children.extend(dmath.children)
                if dmath.has_eq_num:
                    child.has_eq_num = True
                    break
        i += 1


def _extract_inline_paragraphs(root: RootElement):
    """From all outer block elements, decides what to do with contained paragraphs. Either convert
    to plain or convert to separate blocks."""
    new_children = []
    for node in root.children:
        if isinstance(
            node,
            (
                FooterElement,
                HeaderElement,
            ),
        ):
            # These may contain sub-paragraphs, which shall be merged with a space
            if any(isinstance(grandchild, ParagraphElement) for grandchild in node.children):
                new_grandchildren = []
                last_was_paragraph = False
                for grandchild in node.children:
                    if isinstance(grandchild, ParagraphElement):
                        if len(new_grandchildren) > 0:
                            new_grandchildren.append(LinebreakElement())
                        new_grandchildren.extend(grandchild.children)
                        last_was_paragraph = True
                    else:
                        if last_was_paragraph:
                            last_was_paragraph = False
                            new_grandchildren.append(LinebreakElement())
                        new_grandchildren.append(grandchild)
                node.children = new_grandchildren
            new_children.append(node)
        elif isinstance(
            node,
            (
                # Tex-flow elements, shall be split
                HeadingElement,
                BibliographyElement,
                TableOfContentsElement,
                # Floating elements, shall be split
                CaptionElement,
                # CodeBlockElement, # CodeBlock shall contain paragraphs
                # AlgorithmElement, # Algorithm shall contain paragraphs
                FootnoteElement,
                # ListItemElement, # This shall already be cleaned up
                # TableElement,  # Shall already been cleaned up
            ),
        ):
            # They may contain sub-paragraphs. If so, split into multiple blocks.
            if any(isinstance(grandchild, ParagraphElement) for grandchild in node.children):
                node_collector = []
                for grandchild in node.children:
                    if isinstance(grandchild, ParagraphElement):
                        if node_collector:
                            new_node = copy.copy(node)
                            new_node.children = node_collector
                            new_children.append(new_node)
                            node_collector = []
                        new_node = copy.copy(node)
                        new_node.boxes.extend(grandchild.boxes)
                        new_node.children = grandchild.children
                        new_children.append(new_node)
                    else:
                        node_collector.append(grandchild)
                if node_collector:
                    new_node = copy.copy(node)
                    new_node.children = node_collector
                    new_children.append(new_node)
            else:
                new_children.append(node)
        else:
            new_children.append(node)
    root.children = new_children


def _collapse_graphics(node: Element, is_in_math=False):
    """Collapse all nieghboring graphics elements into one element."""

    if isinstance(node, MathElement):
        is_in_math = True

    for child in node.children:
        _collapse_graphics(child, is_in_math)

    # If among children there are consecutive graphics elements, replace them with one graphics element
    if len(node.children) >= 1:
        new_children = []
        cur_graphics = None
        for child in node.children:
            # Check if the child only contains a graphics element

            if (
                isinstance(child, ParagraphElement)
                and len(child.children) == 1
                and isinstance(child.children[0], GraphicsElement)
            ):
                child = child.children[0]

            if isinstance(child, GraphicsElement):
                # Fail on Graphics that do not have a bounding box. This is a bug in nvpdftex v8
                if is_in_math:
                    # Fail page if graphics is in math
                    raise ValueError("GraphicsElement inside MathElement")

                if len(child.boxes) == 0:
                    raise ValueError("GraphicsElement without bounding box")

                if cur_graphics is None:
                    cur_graphics = child
                else:
                    pass
            else:
                if cur_graphics is not None:
                    new_children.append(cur_graphics)
                    cur_graphics = None
                new_children.append(child)
        if cur_graphics is not None:
            new_children.append(cur_graphics)
        node.children = new_children


def _unpack_paragraphs(node: Element):
    """Unpack singular ParagraphElements in block elements."""
    for child in node.children:
        _unpack_paragraphs(child)

    if len(node.children) == 1 and isinstance(node.children[0], ParagraphElement):
        node.boxes.extend(node.children[0].boxes)
        node.children = node.children[0].children


def _pack_paragraphs(node: RootElement):
    """Pack root level inline elements into paragraphs."""
    i = 0
    while i < len(node.children):
        if node.children[i].is_inline and not isinstance(
            node.children[i], (BeginOfParagraphElement, EndOfParagraphElement, GraphicsElement)
        ):
            paragraph = ParagraphElement([node.children.pop(i)])
            while (
                i < len(node.children)
                and node.children[i].is_inline
                and not isinstance(
                    node.children[i],
                    (BeginOfParagraphElement, EndOfParagraphElement, GraphicsElement),
                )
            ):
                paragraph.children.append(node.children.pop(i))
            node.children.insert(i, paragraph)
        i += 1


def _compact_words(node: Element) -> None:
    """Compact all consecutive characters in a node to words.
    For each word, a new bounding box is created that covers all characters in the word.
    If the word is split across lines, multiple bounding boxes are created.
    """

    if not isinstance(node, WordElement):
        for child in node.children:
            _compact_words(child)

    if any(isinstance(n, SuppressElement) for n in node.children):
        idx = 0
        while idx < len(node.children):
            if isinstance(node.children[idx], SuppressElement):
                del node.children[idx]
            else:
                idx += 1

    # If there are characters in this node, we need to compact them to words
    if not isinstance(node, WordElement) and any(
        isinstance(n, CharacterElement) for n in node.children
    ):
        character_collector = []
        new_children = []

        def flush():
            # Convert the characters to a word element
            word_chars = "".join([c.character for c in character_collector])
            assert len(word_chars) > 0
            word_element = WordElement(word=word_chars)
            word_element.children = character_collector
            # Insert the word element before the current node
            new_children.append(word_element)

        # Collect all characters in this node
        for child in node.children + [None]:
            if child is not None and isinstance(child, CharacterElement):
                character_collector.append(child)
            else:
                if len(character_collector) > 0:
                    flush()

                    # Remove the characters from the current node
                    character_collector = []
                if child is not None:
                    new_children.append(child)
        if len(character_collector) > 0:
            flush()
        node.children = new_children


_inline_formatting_types = (
    EmphasisElement,
    StrongElement,
    CodeElement,
    StrikethroughElement,
    UnderlineElement,
    OverlineElement,
    HighlightElement,
)


def _extract_start_type(
    node: Element, element_type: Type[Element]
) -> tuple[bool, list[Element] | None, list[SpaceElement]]:
    """Extract the first element of a certain type from a node, keeping the outer formatting."""
    parents = []
    while len(node.children) > 0:
        if not isinstance(node, _inline_formatting_types):
            return False, None, None
        if isinstance(node, element_type):
            nodes = node.children
            for parent in reversed(parents):
                nodes = [type(parent)(nodes)]
            if len(parents) > 0:
                # Remove the innermost node
                parents[-1].children.pop(0)
            spaces = []
            is_clean = True
            while len(parents) > 0:
                last_parent = parents.pop()
                while len(last_parent.children) > 0 and isinstance(
                    last_parent.children[0], SpaceElement
                ):
                    spaces.append(last_parent.children.pop(0))
                if len(last_parent.children) > 0:
                    is_clean = False
                if is_clean and len(parents) > 0:
                    assert last_parent == parents[-1].children[0]
                    parents[-1].children.pop(0)
            return is_clean, nodes, spaces
        parents.append(node)
        node = node.children[0]
    return False, None, None


def _merge_inline_formatting(node: Element, ind: int = 0):
    """Merge sibling inline formatting elements. This routine will try to find the outermost
    formatting, and move that outside of the inner formatting. It is greedy, i.e. it will try to
    merge as much as possible first, then the remainder may be partitioned.

    A few examples:
    - <em><strong>text</strong></em> (no modifications)-> <em><strong>text</strong></em>
    - <em>text<strong>text</strong></em> (no modifications)-> <em>text<strong>text</strong></em>
    - <em>text</em><em><strong>text</strong></em> (merge ems)-> <em>text<strong>text</strong></em>
    - <em>text</em><strong><em>text</em></strong> (move inner em out)-> <em>text<strong>text</strong></em>
    - <em>text<strong>text</em></strong><em>text</em> (can't merge to the end)-> <em>text<strong>text</em></strong><em>text</em>
    - <em>text</em><strong><em>text</em>text</strong> (pull out first what can be pulled)-> <em>text<strong>text</em></strong><em>text</em>
    """

    if len(node.children) > 1:
        i = 0
        while i < len(node.children):
            child = node.children[i]

            # print("  " * ind + f"[{i}] Check child type: {child.to_str(indent='  ' * ind)}")

            if isinstance(child, _inline_formatting_types):
                if len(child.children) == 0:
                    # Empty formatting, may be resulting from previous operations
                    node.children.pop(i)
                    continue
                # Find the end of the formatting
                child_type = type(child)

                # print("  " * ind + f"Check Merge: {child_type}")

                # Find the next following elements of the same type and merge
                j = i + 1
                while j < len(node.children):
                    nextchild = node.children[j]
                    # print("  " * ind + f"[{j}] Merge check: {nextchild.to_str(indent='  ' * ind)}")
                    if isinstance(nextchild, SpaceElement):
                        # Add the space, might be moved out again at the end
                        child.children.append(node.children.pop(j))
                        # print("  " * ind + f"  -> Space Merge")
                    elif isinstance(nextchild, child_type):
                        # Easy, this is directly the correct type, merge
                        child.children.extend(node.children.pop(j).children)
                        # print("  " * ind + f"  -> Full Merge")
                    else:
                        # Try to extract the next element of the correct type from nested elements
                        is_complete, extracted_nodes, spaces = _extract_start_type(
                            node.children[j], child_type
                        )
                        if extracted_nodes is None:
                            # No element of the correct type found
                            # print("  " * ind + f"  -> No Merge")
                            break
                        # Merge the extracted element
                        child.children.extend(extracted_nodes)
                        # for extracted_node in extracted_nodes:
                        #     print("  " * ind + f"  -> Merge: complete={is_complete} {extracted_node.to_str(ind * '  ')}")
                        if not is_complete:
                            # The inner element still has other types, thus break
                            # But add the spaces after the current element, before the next
                            node.children[j:j] = spaces
                            break
                        # Add the spaces
                        child.children.extend(spaces)
                        j += 1
                # Remove ending spaces
                while len(child.children) > 0 and isinstance(child.children[-1], SpaceElement):
                    i += 1
                    node.children.insert(i, child.children.pop(-1))
            _merge_inline_formatting(child, ind + 1)
            i += 1
    else:
        for child in node.children:
            _merge_inline_formatting(child, ind + 1)


def _find_all_descendants(
    node: Element,
    element_type: Union[Type[TElement], Tuple[Type[TElement], ...]],
    stop_at: Union[Type[Element], Optional[Tuple[Type[Element], ...]]] = None,
) -> List[TElement]:
    """Find all descendants of a node that are of a certain type."""
    result = []

    for child in node.children:
        if isinstance(child, element_type):
            result.append(child)

        if stop_at is not None and isinstance(child, stop_at):
            continue

        result += _find_all_descendants(child, element_type)
    return result


def compact_in_place_flat_layout(root: RootElement, print_errors: bool = False):
    compact_in_place(root)
    if DEBUG_RESTRUCTURING:
        print("-" * 20)
        print("After compact_in_place:")
        print(root)
    if print_errors:
        _dbg_orig = str(root)
    try:
        known_chars, unknown_chars = _merge_unknown_characters(root)
        if DEBUG_RESTRUCTURING:
            print(
                f"Unknown characters: {unknown_chars} / {known_chars + unknown_chars} ({100 * unknown_chars / max(known_chars + unknown_chars, 1):.1f}%)"
            )
        assert (unknown_chars < 100 or unknown_chars * 10 < known_chars) and (
            known_chars > 10 or unknown_chars < 10
        ), "More than 10% unknown characters"

        _compact_words(root)

        # Convert AlgorithmIndentElements to spaces within inner blocks
        _inline_algorithm_indent(root)

        # Extract this to root before unpacking math, such that it can be taken from math.
        _extract_begin_end_of_paragraph(root)

        # First join math if needs joining.
        _dln_join_math(root)

        # Then unpack those not joined.
        _unpack_math(root)
        if DEBUG_RESTRUCTURING:
            print("-" * 20)
            print("After unpack_math:")
            print(root)
        # Extract simple elements
        _treat_math_simple_nonsimple(root)

        # Need it again since compact_in_place introduces additional ones
        _convert_layoutblocks(root)

        # print("Root:", self.root)

        # _collapse_graphics(root)

        _check_graphics_in_table_float(root)

        if DEBUG_RESTRUCTURING:
            print("-" * 20)
            print("Before extraction:")
            print(root)

        # Extract graphics, except if in tables (that will actually raise an exception later)
        # TODO: To remove this constraint, we'd need to decide what should happen to graphics in
        # tables.
        _extract_elements_inplace(
            root,
            GraphicsElement,
            permeable_element=lambda x: not isinstance(
                x,
                (
                    TableElement,
                    TableCellElement,
                ),
            ),
        )
        # Extract paragraphs to root, if not in another block element
        _extract_elements_inplace(
            root,
            ParagraphElement,
            permeable_element=lambda x: not isinstance(
                x,
                (
                    BibliographyElement,
                    TableOfContentsElement,
                    CodeBlockElement,
                    AlgorithmElement,
                    ListItemElement,
                    HeadingElement,
                    TableElement,
                    TableCellElement,
                    FloatElement,
                    FooterElement,
                    HeaderElement,
                    FootnoteElement,
                ),
            ),
        )
        # Extract floating blocks. These are quite safe
        _extract_elements_inplace(
            root,
            FloatElement,
            permeable_element=lambda x: not isinstance(
                x,
                (
                    FooterElement,
                    HeaderElement,
                    FootnoteElement,
                ),
            ),
        )
        # Extract other blocks from blocks, also from paragraphs to root
        _extract_elements_inplace(
            root,
            (
                ListItemElement,
                BibliographyElement,
                TableOfContentsElement,
                CodeBlockElement,
                AlgorithmElement,
                TableElement,
                DisplayMathElement,
            ),
            permeable_element=lambda x: not isinstance(
                x,
                (
                    BibliographyElement,
                    TableOfContentsElement,
                    CodeBlockElement,
                    AlgorithmElement,
                    TableElement,
                    TableCellElement,
                    FooterElement,
                    HeaderElement,
                    FootnoteElement,
                ),
            ),
        )
        # Extract other blocks from blocks, also from paragraphs to root
        _extract_elements_inplace(
            root,
            HeadingElement,
            permeable_element=lambda x: not isinstance(
                x,
                (
                    CodeBlockElement,
                    AlgorithmElement,
                    TableElement,
                    TableCellElement,
                    FooterElement,
                    HeaderElement,
                    FootnoteElement,
                ),
            ),
        )
        # Extract other blocks to root (except if nested)
        _extract_elements_inplace(
            root,
            (
                FooterElement,
                HeaderElement,
                FootnoteElement,
            ),
            permeable_element=lambda x: not isinstance(
                x,
                (
                    FooterElement,
                    HeaderElement,
                    FootnoteElement,
                ),
            ),
        )

        if DEBUG_RESTRUCTURING:
            print("-" * 20)
            print("After extractions:")
            print(root)

        # List items already at root level, this just removes empty list shells
        _purge_lists(root)

        # Merge inline formattings
        _remove_empty(root)
        _merge_inline_formatting(root)
        if DEBUG_RESTRUCTURING:
            print("-" * 20)
            print("After inline merge:")
            print(root)

        _compact_split_bib_toc(root)

        _unpack_floats(root)
        _unpack_list_items(root)
        _filter_heading_bold(root)
        _update_code_blocks(root)
        _remove_empty(root)
        _extract_inline_paragraphs(root)
        _unpack_paragraphs(root)
        _pack_paragraphs(root)
        _clean_space_elements(root)
        _update_begin_end_of_paragraph(root)
        _remove_empty(root)
        _simple_merge_math(root)
        # _split_paragraphs_at_graphics_or_tables(root)

        if DEBUG_RESTRUCTURING:
            print("-" * 20)
            print("End of restructuring:")
            print(root)

        # print("-" * 20)
        # print("End of restructuring")
        # print(root)

        _hierarchy_check(root)
    except Exception:
        if print_errors:
            print("Error during restructuring")
            print("From" + "-" * 20)
            print(_dbg_orig)
            print("To" + "-" * 20)
            print(root)
        raise


def _count_text_chars(node: Element, limit: int) -> int:
    """Count the number of characters in a node and its children.
    Ignores characters inside headers and footers."""

    char_count = 0
    for child in node.children:
        if isinstance(child, CharacterElement):
            char_count += 1
            if char_count >= limit:
                return char_count
        elif isinstance(child, WordElement):
            char_count += len(child.word)
            if char_count >= limit:
                return char_count
        elif isinstance(child, HeaderElement) or isinstance(child, FooterElement):
            # Ignore header and footer text
            continue

        char_count += _count_text_chars(child, limit)
        if char_count >= limit:
            return char_count
    return char_count


def check_page_almost_just_graphics(root: Element) -> bool:
    """Check if a page contains graphics and almost no text outside header and footer.

    Return True if the page is a full-page graphic, False otherwise."""

    if _count_text_chars(root, 15) < 15:
        if len(_find_all_descendants(root, GraphicsElement)) > 0:
            return True

    return False
