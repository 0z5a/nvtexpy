# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, Type, TypeVar, Union

from nvtexpy.output import math
from nvtexpy.output.restructuring import (
    CaptionElement,
    DisplayMathElement,
    ToBeContinuedElement,
    compact_in_place_flat_layout,
)
from nvtexpy.output.structured_output import (
    AlgorithmElement,
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
    HyphenElement,
    LayoutBlockElement,
    LinebreakElement,
    ListItemElement,
    ListItemLabelElement,
    MathElement,
    OrderedListElement,
    OverlineElement,
    PageNumberElement,
    ParagraphElement,
    RootElement,
    SpaceElement,
    StrikethroughElement,
    StrongElement,
    SubscriptElement,
    SuperscriptElement,
    TableCellElement,
    TableElement,
    TableOfContentsElement,
    TableRowElement,
    UnderlineElement,
    UnorderedListElement,
    WordElement,
)

_converters = {}

TElement = TypeVar("TElement", bound=Element, covariant=True)


@dataclass
class State:
    """Settings for the conversion"""

    page_center_x_px: float
    page_center_y_px: float

    # The scale factor for the coordinates
    coord_scale_factor: Tuple[float, float] = (1, 1)
    coord_transformer: Optional[Callable[[float, float], Tuple[float, float]]] = None
    bbox_transformer: Optional[
        Callable[[float, float, float, float], Tuple[float, float, float, float]]
    ] = None

    table_format: Literal["latex", "html"] = "latex"
    style_format: Literal["markdown", "html"] = "markdown"

    # Inner state: Want to convert to plaintext
    want_plaintext: bool = False
    # True, if the first character of a new paragraph was not printed yet
    is_start: bool = False
    # True if inside table, which requires different escaping
    is_in_table: bool = False

    def transform_bbox(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> Tuple[int, int, int, int]:
        if self.bbox_transformer is not None:
            x0, y0, x1, y1 = self.bbox_transformer(x0, y0, x1, y1)
            return round(x0), round(y0), round(x1), round(y1)
        elif self.coord_transformer is not None:
            x0, y0 = self.coord_transformer(x0, y0)
            x1, y1 = self.coord_transformer(x1, y1)
            x2, y2 = self.coord_transformer(x0, y1)
            x3, y3 = self.coord_transformer(x1, y0)
            return (
                round(min(x0, x1, x2, x3)),
                round(min(y0, y1, y2, y3)),
                round(max(x0, x1, x2, x3)),
                round(max(y0, y1, y2, y3)),
            )
        return round(x0), round(y0), round(x1), round(y1)


CATEGORY_INLINE_TEXT = "Inline-Text"
CATEGORY_TEXT = "Text"
CATEGORY_TITLE = "Title"
CATEGORY_SECTION_HEADER = "Section-header"
CATEGORY_LIST_ITEM = "List-item"
CATEGORY_CAPTION = "Caption"
CATEGORY_TABLE = "Table"
CATEGORY_PICTURE = "Picture"
CATEGORY_FORMULA = "Formula"
CATEGORY_CODE = "Code"
CATEGORY_FOOTNOTE = "Footnote"
CATEGORY_PAGE_HEADER = "Page-header"
CATEGORY_PAGE_FOOTER = "Page-footer"
CATEGORY_BIB = "Bibliography"
CATEGORY_TOC = "TOC"


@dataclass
class FlatLayoutNode:
    """A node in the flat layout tree"""

    bbox: Optional[BoundingBox] = None
    content: str = ""
    category_id: str = CATEGORY_INLINE_TEXT


def converts(
    cls: Type[TElement],
) -> Callable[
    [Callable[[TElement, State], FlatLayoutNode]], Callable[[TElement, State], FlatLayoutNode]
]:
    def decorator(func: Callable[[TElement, State], FlatLayoutNode]):
        _converters[cls] = func
        return func

    return decorator


def _convert(element: Element, settings: State) -> FlatLayoutNode:
    t = type(element)
    if t in _converters:
        return _converters[t](element, settings)
    for t in t.__bases__:
        if t in _converters:
            return _converters[t](element, settings)
    raise ValueError(f"No converter for element class {type(element)}")


def _is_inline_deep(element: Element) -> bool:
    return element.is_inline and _all_inline_deep(element)


def _all_inline_deep(element: Element) -> bool:
    return all(_is_inline_deep(child) for child in element.children)


def _clean_empty_nodes(nodes: List[FlatLayoutNode]) -> List[FlatLayoutNode]:
    """Remove empty nodes without a bounding box."""

    # stuff like this:
    # __
    # __<br>
    # __<br><tbc>
    # **``**
    # ## <br>
    # ### <br>
    # ``
    # ```\n\n\n```
    # ```\n\n```
    # ``<br><tbc>
    # <tbc>
    # <tbc> <tbc>
    # <tbc>__
    # <tbc>_****_
    # <tbc>_``_<tbc>
    # <tbc>**``**<tbc>
    # <tbc><br>
    # <tbc><br><tbc>
    def _is_empty(node):
        if node.bbox:
            return False
        content = (
            node.content.replace("_", "")
            .replace("**", "")
            .replace("`", "")
            .replace("#", "")
            .replace("<br>", "")
            .replace("<tbc>", "")
            .strip()
        )
        assert node.category_id == CATEGORY_PICTURE or content == "", (
            f"node has no bbox: '{node.content}'"
        )
        return True

    return [node for node in nodes if not _is_empty(node)]


def convert_join(
    element: Element, settings: State, category_id: str, as_plaintext: bool = True
) -> FlatLayoutNode:
    """Join multiple text elements into one from the children of `element`."""
    old_plaintext = settings.want_plaintext
    if as_plaintext:
        settings.want_plaintext = True
    bboxes = list(element.boxes)
    contents = []
    for c in element.children:
        b = _convert(c, settings)
        assert b.category_id in (
            CATEGORY_INLINE_TEXT,
            CATEGORY_FORMULA,
        ), f"Cannot join non-inline-text {element}"
        if b.bbox is not None:
            bboxes.append(b.bbox)
        contents.append(b.content)
    settings.want_plaintext = old_plaintext
    return FlatLayoutNode(
        bbox=BoundingBox.bboxes_union(bboxes),
        content="".join(contents),
        category_id=category_id,
    )


def convert(
    root: RootElement,
    page_width_px: int,
    page_height_px: int,
    coord_scale_factor: Tuple[float, float],
    *,
    table_format: Literal["latex", "html"] = "latex",
    style_format: Literal["markdown", "html"] = "markdown",
    print_errors: bool = False,
) -> str:
    compact_in_place_flat_layout(root, print_errors=print_errors)
    state = State(
        coord_scale_factor=coord_scale_factor,
        page_center_x_px=page_width_px / 2,
        page_center_y_px=page_height_px / 2,
        table_format=table_format,
        style_format=style_format,
    )
    nodes = convert_root(root, state)
    nodes = _clean_empty_nodes(nodes)

    return json.dumps(
        {
            "metadata": {
                "coord_scale_factor": list(coord_scale_factor),
                "page_width_px": page_width_px,
                "page_height_px": page_height_px,
            },
            "ann": [
                {
                    "category_id": node.category_id,
                    "bbox": state.transform_bbox(
                        *node.bbox.as_xywh_list_scaled(*state.coord_scale_factor)
                    ),
                    "content": node.content.strip(),
                }
                for node in nodes
                if node.bbox
            ],
        },
        ensure_ascii=False,
    )


def convert_from_pkl(
    pkl: Union[dict, str, Path],
    table_format: Literal["latex", "html"] = "latex",
    style_format: Literal["markdown", "html"] = "markdown",
    *,
    print_errors: bool = False,
) -> str:
    if isinstance(pkl, (str, Path)):
        with open(pkl, "rb") as f:
            pkl = pickle.load(f)
    return convert(
        pkl["root"],
        pkl["page_width_px"],
        pkl["page_height_px"],
        pkl["coord_scale_factor"],
        table_format=table_format,
        style_format=style_format,
        print_errors=print_errors,
    )


def convert_root(element: RootElement, state: State) -> List[FlatLayoutNode]:
    return [_convert(c, state) for c in element.children]


@converts(FloatElement)
def convert_float(element: FloatElement, state: State) -> FlatLayoutNode:
    raise Exception("Float elements should have been unpacked")


# Either (not preceded by backslash already) one of: \ ` * < > ~ _ ]( ][ ~~
_re_md_unsafe_chars = re.compile(r"([\\`*<>_]|\]\(|\]\[|~~)")
# one of: (start of line)# (start of line)(spaces*)-
_re_md_unsafe_chars_start_of_line = re.compile(r"^(#)")
_re_md_unsafe_chars_start_of_line_spaced = re.compile(r"^(\s*)([->\*+])")
_re_md_unsafe_chars_table = re.compile(r"(&)")


def _safe_md_chars(txt: str) -> str:
    txt = _re_md_unsafe_chars.sub(r"\\\1", txt)
    return txt


def _safe_md_chars_start_of_line(txt: str) -> str:
    txt = _re_md_unsafe_chars_start_of_line.sub(r"\\\1", txt)
    txt = _re_md_unsafe_chars_start_of_line_spaced.sub(r"\1\\\2", txt)
    return txt


def _safe_md_chars_table(txt: str) -> str:
    txt = _re_md_unsafe_chars_table.sub(r"\\\1", txt)
    return txt


_re_html_unsafe_chars = re.compile(r"([&<>])")


def _safe_html_chars(txt: str) -> str:
    txt = _re_html_unsafe_chars.sub(r"\\\1", txt)
    return txt


@converts(CharacterElement)
def convert_char(element: CharacterElement, state: State) -> FlatLayoutNode:
    if state.style_format == "html":
        cnt = _safe_html_chars(element.character)
    else:
        cnt = _safe_md_chars(element.character)
        if state.is_start:
            cnt = _safe_md_chars_start_of_line(cnt)
        if state.is_in_table:
            cnt = _safe_md_chars_table(cnt)
    state.is_start = False
    return FlatLayoutNode(
        bbox=element.compute_bounding_box(),
        content=cnt,
    )


@converts(WordElement)
def convert_word(element: WordElement, state: State) -> FlatLayoutNode:
    if state.style_format == "html":
        cnt = _safe_html_chars(element.word)
    else:
        cnt = _safe_md_chars(element.word)
        if state.is_start:
            cnt = _safe_md_chars_start_of_line(cnt)
        if state.is_in_table:
            cnt = _safe_md_chars_table(cnt)
    state.is_start = False
    return FlatLayoutNode(
        bbox=element.compute_bounding_box(),
        content=cnt,
    )


@converts(SpaceElement)
def convert_space(element: SpaceElement, state: State) -> FlatLayoutNode:
    return FlatLayoutNode(
        bbox=element.compute_bounding_box(),
        content=" ",
    )


@converts(LayoutBlockElement)
def convert_layout_block(element: LayoutBlockElement, state: State) -> FlatLayoutNode:
    # There are sometimes non-root-level layout blocks such as in Bilbiography
    # We ignore them for now
    raise Exception("LayoutBlockElements should have been converted to paragraphs")


@converts(ParagraphElement)
def convert_paragraph(element: ParagraphElement, state: State) -> FlatLayoutNode:
    state.is_start = True
    return convert_join(
        element, state, category_id=CATEGORY_FORMULA if element.is_math_formula else CATEGORY_TEXT
    )


@converts(CaptionElement)
def convert_caption(element: CaptionElement, state: State) -> FlatLayoutNode:
    assert not state.want_plaintext, "Parent requires plaintext, caption should be at root level"
    return convert_join(element, state, category_id=CATEGORY_CAPTION)


@converts(BeginOfParagraphElement)
def convert_begin_of_paragraph(element: BeginOfParagraphElement, state: State) -> FlatLayoutNode:
    """Let's ignore this for now. Later we can use it to detect paragraphs that are broken across pages."""
    return FlatLayoutNode()


@converts(EndOfParagraphElement)
def convert_end_of_paragraph(element: EndOfParagraphElement, state: State) -> FlatLayoutNode:
    """Let's ignore this for now. Later we can use it to detect paragraphs that are broken across pages."""
    return FlatLayoutNode()


@converts(ToBeContinuedElement)
def convert_to_be_continued(element: ToBeContinuedElement, state: State) -> FlatLayoutNode:
    return FlatLayoutNode(bbox=None, content="<tbc>")


@converts(HeadingElement)
def convert_heading(element: HeadingElement, state: State) -> FlatLayoutNode:
    if state.want_plaintext:
        return convert_join(element, state, category_id=CATEGORY_INLINE_TEXT)
    state.is_start = False
    res = convert_join(
        element,
        state,
        category_id=CATEGORY_TITLE if element.level == 1 else CATEGORY_SECTION_HEADER,
    )
    assert element.level >= 1 and element.level <= 6, f"invalid heading level: {element.level}"
    if state.style_format == "html":
        res.content = f"<h{element.level}>{res.content}</h{element.level}>"
    else:
        prefix = "#" * element.level
        res.content = prefix + " " + res.content.replace("\n", " ")
    return res


@converts(LinebreakElement)
def convert_line_break(element: LinebreakElement, state: State) -> FlatLayoutNode:
    if state.want_plaintext:
        state.is_start = True
        return FlatLayoutNode(content="<br>\n")
    return FlatLayoutNode()


@converts(MathElement)
def convert_math(element: MathElement, state: State) -> FlatLayoutNode:
    state.is_start = False
    code = math.to_str(element.code)
    return FlatLayoutNode(
        bbox=element.compute_bounding_box(),
        content=code,
        category_id=CATEGORY_FORMULA,
    )


@converts(DisplayMathElement)
def convert_display_math(element: DisplayMathElement, state: State) -> FlatLayoutNode:
    state.is_start = True
    return convert_join(element, state, category_id=CATEGORY_FORMULA)


@converts(AlignBlockEqNumElement)
def convert_align_block_eq_num(element: AlignBlockEqNumElement, state: State) -> FlatLayoutNode:
    state.is_start = False
    return convert_join(element, state, category_id=CATEGORY_INLINE_TEXT)


@converts(CodeElement)
def convert_code(element: CodeElement, state: State) -> FlatLayoutNode:
    state.is_start = False
    res = convert_join(
        element, state, category_id=CATEGORY_INLINE_TEXT if state.want_plaintext else CATEGORY_CODE
    )
    if state.style_format == "html":
        res.content = f"<code>{res.content}</code>"
    else:
        res.content = "`" + res.content + "`"
    return res


@converts(CodeBlockElement)
def convert_code_block(element: CodeBlockElement, state: State) -> FlatLayoutNode:
    # Remove any nested code elements from inside
    assert not state.want_plaintext, "Code block should be at root level"
    state.is_start = False
    join = []
    for child in element.children:
        join.append(convert_join(child, state, category_id=CATEGORY_CODE))
    if state.style_format == "html":
        content = "<pre>\n" + "\n".join([c.content.rstrip() for c in join]).rstrip() + "\n</pre>"
    else:
        content = "```\n" + "\n".join([c.content.rstrip() for c in join]).rstrip() + "\n```"
    return FlatLayoutNode(
        bbox=BoundingBox.bboxes_union([c.bbox for c in join]),
        content=content,
        category_id=CATEGORY_CODE,
    )


@converts(AlgorithmElement)
def convert_algorithm(element: AlgorithmElement, state: State) -> FlatLayoutNode:
    # Remove any nested code elements from inside
    assert not state.want_plaintext, "Algorithm should be at root level"
    join = []
    for child in element.children:
        state.is_start = True
        join.append(convert_join(child, state, category_id=CATEGORY_CODE))
    if state.style_format == "html":
        content = "<br>\n".join([c.content.rstrip() for c in join])
    else:
        content = "<br>\n".join([c.content.rstrip() for c in join])
    return FlatLayoutNode(
        bbox=BoundingBox.bboxes_union([c.bbox for c in join]),
        content=content,
        category_id=CATEGORY_CODE,
    )


@converts(StrongElement)
def convert_strong(element: StrongElement, state: State) -> FlatLayoutNode:
    state.is_start = False
    res = convert_join(element, state, category_id=CATEGORY_INLINE_TEXT)
    if state.style_format == "html":
        res.content = f"<strong>{res.content}</strong>"
    else:
        res.content = f"**{res.content}**"
    return res


@converts(EmphasisElement)
def convert_emphasis(element: EmphasisElement, state: State) -> FlatLayoutNode:
    state.is_start = False
    res = convert_join(element, state, category_id=CATEGORY_INLINE_TEXT)
    if state.style_format == "html":
        res.content = f"<em>{res.content}</em>"
    else:
        res.content = f"_{res.content}_"
    return res


@converts(SubscriptElement)
def convert_subscript(element: SubscriptElement, state: State) -> FlatLayoutNode:
    state.is_start = False
    res = convert_join(element, state, category_id=CATEGORY_INLINE_TEXT)
    res.content = f"<sub>{res.content}</sub>"
    return res


@converts(SuperscriptElement)
def convert_superscript(element: SuperscriptElement, state: State) -> FlatLayoutNode:
    state.is_start = False
    res = convert_join(element, state, category_id=CATEGORY_INLINE_TEXT)
    res.content = f"<sup>{res.content}</sup>"
    return res


@converts(StrikethroughElement)
def convert_strikethrough(element: StrikethroughElement, state: State) -> FlatLayoutNode:
    state.is_start = False
    res = convert_join(element, state, category_id=CATEGORY_INLINE_TEXT)
    res.content = f"<del>{res.content}</del>"
    return res


@converts(UnderlineElement)
def convert_underline(element: UnderlineElement, state: State) -> FlatLayoutNode:
    state.is_start = False
    res = convert_join(element, state, category_id=CATEGORY_INLINE_TEXT)
    res.content = f"<u>{res.content}</u>"
    return res


@converts(OverlineElement)
def convert_overline(element: OverlineElement, state: State) -> FlatLayoutNode:
    state.is_start = False
    res = convert_join(element, state, category_id=CATEGORY_INLINE_TEXT)
    res.content = f"<o>{res.content}</o>"
    return res


@converts(TableElement)
def convert_table(element: TableElement, state: State) -> FlatLayoutNode:
    assert not state.want_plaintext, "Table should be at root level"
    state.is_start = False
    state.is_in_table = True
    bboxes: list[BoundingBox | None] = list(element.boxes)  # shallow copy
    rows = []
    num_cols = -1
    rowspans = {}
    for row_el in element.children:
        if isinstance(row_el, (LinebreakElement, SpaceElement)):
            continue
        assert isinstance(row_el, TableRowElement), f"{row_el!r}"
        row_cols = 0
        cells = []
        for cell_el in row_el.children:
            if isinstance(cell_el, SpaceElement):
                continue
            assert isinstance(cell_el, TableCellElement), f"{cell_el!r}"
            while rowspans.get(row_cols, 0) > 0:
                rowspans[row_cols] -= 1
                if state.table_format == "latex":
                    bboxes.append(None)
                    cells.append("")
                row_cols += 1
            cell = convert_join(cell_el, state, category_id=CATEGORY_TABLE)
            if state.table_format == "html":
                if cell_el.colspan != 1 and cell_el.rowspan != 1:
                    cell.content = f'<td rowspan="{cell_el.rowspan}" colspan="{cell_el.colspan}" title="c={row_cols}">{cell.content}</td>'
                elif cell_el.rowspan != 1:
                    cell.content = (
                        f'<td rowspan="{cell_el.rowspan}" title="c={row_cols}">{cell.content}</td>'
                    )
                elif cell_el.colspan != 1:
                    cell.content = (
                        f'<td colspan="{cell_el.colspan}" title="c={row_cols}">{cell.content}</td>'
                    )
                else:
                    cell.content = f'<td title="c={row_cols}">{cell.content}</td>'
                if cell_el.rowspan != 1:
                    for span_col in range(row_cols, row_cols + cell_el.colspan):
                        rowspans[span_col] = cell_el.rowspan - 1
            else:
                if cell_el.rowspan != 1:
                    cell.content = f"\\multirow{{{cell_el.rowspan}}}{{*}}{{{cell.content}}}"
                    for span_col in range(row_cols, row_cols + cell_el.colspan):
                        rowspans[span_col] = cell_el.rowspan - 1
                if cell_el.colspan != 1:
                    cell.content = f"\\multicolumn{{{cell_el.colspan}}}{{c}}{{{cell.content}}}"
            row_cols += cell_el.colspan
            bboxes.append(cell.bbox)
            cells.append(cell.content)
        while rowspans.get(row_cols, 0) > 0:
            rowspans[row_cols] -= 1
            if state.table_format == "latex":
                # For latex, we need to add empty cells for the rowspans
                bboxes.append(None)
                cells.append("")
            row_cols += 1
        if num_cols == -1:
            num_cols = row_cols
        else:
            assert num_cols == row_cols, (
                f"Row {len(rows)} has {row_cols} columns, expected {num_cols}"
            )
        if state.table_format == "html":
            rows.append(f'<tr title="r={len(rows)}">' + "".join(cells) + "</tr>\n")
        else:
            rows.append(" & ".join(cells) + " \\\\\n")
    state.is_in_table = False
    if state.table_format == "html":
        content = "<table>\n" + "".join(rows) + "</table>\n"
    else:
        content = (
            "\\begin{tabular}{c" + " c" * (num_cols - 1) + "}\n" + "".join(rows) + "\\end{tabular}"
        )
    return FlatLayoutNode(
        bbox=BoundingBox.bboxes_union(bboxes),
        content=content,
        category_id=CATEGORY_TABLE,
    )


@converts(TableRowElement)
def convert_table_row(element: TableRowElement, state: State) -> FlatLayoutNode:
    raise Exception("Table row outside table")


@converts(TableCellElement)
def convert_table_cell(element: TableCellElement, state: State) -> FlatLayoutNode:
    raise Exception("Table cell outside table")


@converts(OrderedListElement)
def convert_ordered_list(element: OrderedListElement, state: State) -> FlatLayoutNode:
    raise Exception("Lists should have been unpacked")


@converts(UnorderedListElement)
def convert_unordered_list(element: UnorderedListElement, state: State) -> FlatLayoutNode:
    raise Exception("Lists should have been unpacked")


@converts(ListItemElement)
def convert_list_item(element: ListItemElement, state: State) -> FlatLayoutNode:
    assert not state.want_plaintext, "List item should be at root level"
    state.is_start = True
    return convert_join(element, state, category_id=CATEGORY_LIST_ITEM)


@converts(ListItemLabelElement)
def convert_list_item_label(element: ListItemLabelElement, state: State) -> FlatLayoutNode:
    state.is_start = False
    return convert_join(element, state, category_id=CATEGORY_INLINE_TEXT)


@converts(FootnoteElement)
def convert_footnote(element: FootnoteElement, state: State) -> FlatLayoutNode:
    assert not state.want_plaintext, "Footnote should be at root level"
    state.is_start = True
    return convert_join(element, state, category_id=CATEGORY_FOOTNOTE)


@converts(FootnoteMarkElement)
def convert_footnote_mark(element: FootnoteMarkElement, state: State) -> FlatLayoutNode:
    return convert_join(element, state, category_id=CATEGORY_INLINE_TEXT)


@converts(HeaderElement)
def convert_header(element: HeaderElement, state: State) -> FlatLayoutNode:
    assert not state.want_plaintext, "Header should be at root level"
    state.is_start = True
    return convert_join(element, state, category_id=CATEGORY_PAGE_HEADER)


@converts(FooterElement)
def convert_footer(element: FooterElement, state: State) -> FlatLayoutNode:
    assert not state.want_plaintext, "Footer should be at root level"
    state.is_start = True
    return convert_join(element, state, category_id=CATEGORY_PAGE_FOOTER)


@converts(PageNumberElement)
def convert_page_number(element: PageNumberElement, state: State) -> FlatLayoutNode:
    if state.want_plaintext:
        return convert_join(element, state, category_id=CATEGORY_INLINE_TEXT)
    else:
        state.is_start = True
        bbox = element.compute_bounding_box()
        assert bbox is not None
        if bbox.y < state.page_center_y_px:
            return convert_join(element, state, category_id=CATEGORY_PAGE_HEADER)
        else:
            return convert_join(element, state, category_id=CATEGORY_PAGE_FOOTER)


@converts(GraphicsElement)
def convert_graphics(element: GraphicsElement, state: State) -> FlatLayoutNode:
    assert not state.want_plaintext, "TOC should be at root level"
    state.is_start = False
    if state.style_format == "html":
        content = "<img />"
    else:
        content = "<graphics>"
    return FlatLayoutNode(
        bbox=element.compute_bounding_box(),
        content=content,
        category_id=CATEGORY_PICTURE,
    )


@converts(TableOfContentsElement)
def convert_table_of_contents(element: TableOfContentsElement, state: State) -> FlatLayoutNode:
    assert not state.want_plaintext, "TOC should be at root level"
    state.is_start = True
    return convert_join(element, state, category_id=CATEGORY_TOC)


@converts(BibliographyElement)
def convert_bibliography(element: BibliographyElement, state: State) -> FlatLayoutNode:
    assert not state.want_plaintext, "TOC should be at root level"
    state.is_start = True
    return convert_join(element, state, category_id=CATEGORY_BIB)


@converts(HyphenElement)
def convert_hyphen(element: HyphenElement, state: State) -> FlatLayoutNode:
    """Ignore hyphenation hyphens. This does not include word hyphens."""
    return FlatLayoutNode()


@converts(AlignBlockElement)
def convert_align_block(element: AlignBlockElement, state: State) -> FlatLayoutNode:
    """Converts AlignBlockElements to display math."""
    assert False, "AlignBlockElements should have been converted to math"


def load_pkl_file(filepath: Union[str, Path]) -> dict:
    """Load a pickle file"""
    with open(filepath, "rb") as f:
        return pickle.load(f)
