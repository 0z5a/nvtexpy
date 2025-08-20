# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
)

from nvtexpy.output.restructuring import compact_in_place
from nvtexpy.output.structured_output import (
    AlignBlockElement,
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
    MathElement,
    OrderedListElement,
    PageNumberElement,
    ParagraphElement,
    RootElement,
    SpaceElement,
    StrongElement,
    StructuredOutput,
    SubscriptElement,
    SuperscriptElement,
    TableCellElement,
    TableElement,
    TableOfContentsElement,
    TableRowElement,
    UnorderedListElement,
    WordElement,
)


@dataclass
class State:
    """Settings for the conversion"""

    # The scale factor for the coordinates
    coord_scale_factor: Tuple[float, float] = (1, 1)
    coord_transformer: Optional[Callable[[float, float], Tuple[float, float]]] = None
    bbox_transformer: Optional[
        Callable[[float, float, float, float], Tuple[float, float, float, float]]
    ] = None

    is_first_word: bool = False

    bbox_coordinates: bool = True
    math_coordinates: bool = False
    wordwise_coordinates: bool = False

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


_converters = {}

TElement = TypeVar("TElement", bound=Element, covariant=True)


def converts(
    cls: Type[TElement],
) -> Callable[[Callable[[TElement, State], str]], Callable[[TElement, State], str]]:
    def decorator(func: Callable[[TElement, State], str]):
        _converters[cls] = func
        return func

    return decorator


def _convert(element: Element, settings: State) -> str:
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


def _get_coords(element: Element, settings: State) -> Tuple[int, int, int, int]:
    bbox = element.compute_bounding_box()
    assert bbox is not None, f"Missing bounding box for {element}"
    return settings.transform_bbox(*bbox.as_xyxy_list_scaled(*settings.coord_scale_factor))


def _get_multi_coords(elements: Sequence[Element], settings: State) -> Tuple[int, int, int, int]:
    assert len(elements) > 0, f"Missing bounding box for {elements}"

    child_boxes = [element.compute_bounding_box() for element in elements]

    final_boxes = [b for b in child_boxes if b is not None]

    assert len(final_boxes) > 0, f"Missing bounding box for {elements}"
    bbox = BoundingBox.bboxes_union(final_boxes)
    return settings.transform_bbox(*bbox.as_xyxy_list_scaled(*settings.coord_scale_factor))


def convert_join(element: Element, settings: State, force_bbox: Optional[bool] = None) -> str:
    if settings.bbox_coordinates and (
        force_bbox is True
        or (
            force_bbox is not False
            and not element.is_inline
            and any(el.is_inline for el in element.children)
        )
    ):
        # This element contains inline element -> set a bounding box
        x0, y0, x1, y1 = _get_coords(element, settings)
        settings.is_first_word = True
        return (
            f"<x_{x0}/><y_{y0}/>"
            + "".join([_convert(c, settings) for c in element.children])
            + f"<x_{x1}/><y_{y1}/>"
        )
    else:
        return "".join([_convert(c, settings) for c in element.children])


@converts(RootElement)
def convert_root(element: RootElement, settings: State) -> str:
    return convert_join(element, settings)


@converts(FloatElement)
def convert_float(element: FloatElement, settings: State) -> str:
    return "<float>" + convert_join(element, settings, force_bbox=True) + "</float>\n"


@converts(WordElement)
def convert_word(element: WordElement, settings: State) -> str:
    if settings.is_first_word or not settings.wordwise_coordinates:
        settings.is_first_word = False
        return element.word
    else:
        x, y, _, _ = _get_coords(element, settings)
        return f"<x_{x}/><y_{y}/>{element.word}"


@converts(SpaceElement)
def convert_space(element: SpaceElement, settings: State) -> str:
    return " "


@converts(CharacterElement)
def convert_character(element: CharacterElement, settings: State) -> str:
    raise Exception("Character element outside word")


@converts(HyphenElement)
def convert_hyphen(element: HyphenElement, settings: State) -> str:
    raise Exception("Hyphen element outside word")


@converts(LayoutBlockElement)
def convert_layout_block(element: LayoutBlockElement, settings: State) -> str:
    raise Exception("Layout block element outside paragraph")


@converts(ParagraphElement)
def convert_paragraph(element: ParagraphElement, settings: State) -> str:
    while len(element.children) == 1 and isinstance(element.children[0], ParagraphElement):
        # If the paragraph only contains one paragraph,
        # remove the paragraph around it
        element = element.children[0]

    if len(element.children) >= 1 and isinstance(element.children[0], LayoutBlockElement):
        # This has multiple bounding boxes, because there is a layout break!
        # (But only if the layout block contains text)
        res = []
        for layout_block in element.children:
            assert isinstance(layout_block, LayoutBlockElement)
            res.append(convert_join(layout_block, settings))
        return "<p>" + "".join(res) + "</p>\n"
    else:
        return "<p>" + convert_join(element, settings) + "</p>\n"


@converts(HeadingElement)
def convert_heading(element: HeadingElement, settings: State) -> str:
    tag = f"h{element.level}"

    return f"<{tag}>" + convert_join(element, settings) + f"</{tag}>\n"


@converts(LinebreakElement)
def convert_line_break(element: LinebreakElement, settings: State) -> str:
    return "<br/>"


@converts(AlignBlockElement)
def convert_align_block(element: AlignBlockElement, settings: State) -> str:
    raise Exception("Align block element outside paragraph")


@converts(MathElement)
def convert_math(element: MathElement, settings: State) -> str:
    if element.is_simple:
        # This is effectively just text, output the inner elements instead
        return convert_join(element, settings)

    # If math is not delimited, it's complicated :see_no_evil: Math which contains \text only.
    # This cannot be consistently detected -> Fail
    if settings.math_coordinates:
        x0, y0, x1, y1 = _get_coords(element, settings)
        bbox_prefix = f"<x_{x0}/><y_{y0}/>"
        bbox_suffix = f"<x_{x1}/><y_{y1}/>"
    else:
        bbox_prefix = ""
        bbox_suffix = ""
    return f"<latexmath>{bbox_prefix}\\({element.code}\\){bbox_suffix}</latexmath>"


@converts(CodeElement)
def convert_code(element: CodeElement, settings: State) -> str:
    return "<code>" + convert_join(element, settings) + "</code>"


@converts(CodeBlockElement)
def convert_code_block(element: CodeBlockElement, settings: State) -> str:
    return "<code_block>" + convert_join(element, settings) + "</code_block>\n"


@converts(TableOfContentsElement)
def convert_toc(element: TableOfContentsElement, settings: State) -> str:
    return "<toc>" + convert_join(element, settings) + "</toc>\n"


@converts(BibliographyElement)
def convert_bibliography(element: BibliographyElement, settings: State) -> str:
    return "<bibliography>" + convert_join(element, settings) + "</bibliography>\n"


@converts(StrongElement)
def convert_strong(element: StrongElement, settings: State) -> str:
    return "<strong>" + convert_join(element, settings) + "</strong>"


@converts(EmphasisElement)
def convert_emphasis(element: EmphasisElement, settings: State) -> str:
    return "<em>" + convert_join(element, settings) + "</em>"


@converts(SubscriptElement)
def convert_subscript(element: SubscriptElement, settings: State) -> str:
    return "<sub>" + convert_join(element, settings) + "</sub>"


@converts(SuperscriptElement)
def convert_superscript(element: SuperscriptElement, settings: State) -> str:
    return "<sup>" + convert_join(element, settings) + "</sup>"


@converts(TableElement)
def convert_table(element: TableElement, settings: State) -> str:
    rows = []
    for row_el in element.children:
        if isinstance(row_el, (LinebreakElement, SpaceElement)):
            continue
        assert isinstance(row_el, TableRowElement), f"{row_el!r}"
        cells = []
        for cell_el in row_el.children:
            if isinstance(cell_el, SpaceElement):
                continue
            assert isinstance(cell_el, TableCellElement), f"{cell_el!r}"
            if cell_el.colspan != 1:
                colspan = "<colspan_{}/>".format(cell_el.colspan)
            else:
                colspan = ""
            if cell_el.rowspan != 1:
                rowspan = "<rowspan_{}/>".format(cell_el.rowspan)
            else:
                rowspan = ""
            if settings.bbox_coordinates:
                x_left, _, x_right, _ = _get_coords(cell_el, settings)
                cells.append(
                    f"<td><x_{x_left}/>{colspan}{rowspan}"
                    + convert_join(cell_el, settings, force_bbox=False)
                    + f"<x_{x_right}/></td>"
                )
            else:
                cells.append(
                    f"<td>{colspan}{rowspan}"
                    + convert_join(cell_el, settings, force_bbox=False)
                    + "</td>"
                )
        if settings.bbox_coordinates:
            _, y_top, _, y_bottom = _get_coords(row_el, settings)
            rows.append(f"<tr><y_{y_top}/>" + "".join(cells) + f"<y_{y_bottom}/></tr>\n")
        else:
            rows.append("<tr>" + "".join(cells) + "</tr>\n")
    return "<table>\n" + "".join(rows) + "</table>\n"


@converts(TableRowElement)
def convert_table_row(element: TableRowElement, settings: State) -> str:
    raise Exception("Table row outside table")


@converts(TableCellElement)
def convert_table_cell(element: TableCellElement, settings: State) -> str:
    raise Exception("Table cell outside table")


@converts(OrderedListElement)
@converts(UnorderedListElement)
def list_to_lines(element: Union[OrderedListElement, UnorderedListElement], settings: State) -> str:
    lines = []
    for item_el in element.children:
        if isinstance(item_el, (LinebreakElement, SpaceElement)):
            continue
        assert isinstance(item_el, ListItemElement), f"{item_el!r}"
        lines.append("<li>" + convert_join(item_el, settings) + "</li>\n")
    eltype = "ul" if isinstance(element, UnorderedListElement) else "ol"
    return f"<{eltype}>\n" + "".join(lines) + f"</{eltype}>\n"


@converts(EndOfParagraphElement)
def convert_eop(element: EndOfParagraphElement, settings: State) -> str:
    return ""


@converts(BeginOfParagraphElement)
def convert_bop(element: BeginOfParagraphElement, settings: State) -> str:
    return ""


@converts(ListItemElement)
def convert_list_item(element: ListItemElement, settings: State) -> str:
    raise Exception("List item outside list")


@converts(FootnoteElement)
def convert_footnote(element: FootnoteElement, settings: State) -> str:
    return "<footnote>" + convert_join(element, settings) + "</footnote>"


@converts(FootnoteMarkElement)
def convert_footnote_mark(element: FootnoteMarkElement, settings: State) -> str:
    return "<footnote_ref>" + convert_join(element, settings) + "</footnote_ref>"


@converts(HeaderElement)
def convert_header(element: HeaderElement, settings: State) -> str:
    return "<header>" + convert_join(element, settings) + "</header>\n"


@converts(FooterElement)
def convert_footer(element: FooterElement, settings: State) -> str:
    return "<footer>" + convert_join(element, settings) + "</footer>\n"


@converts(PageNumberElement)
def convert_page_number(element: PageNumberElement, settings: State) -> str:
    # return "<pagenumber>" + convert_join(element, settings) + "</pagenumber>"
    return convert_join(element, settings)


@converts(GraphicsElement)
def convert_graphics(element: GraphicsElement, settings: State) -> str:
    if settings.bbox_coordinates:
        x0, y0, x1, y1 = _get_coords(element, settings)
        return f"<graphics><x_{x0}/><y_{y0}/><x_{x1}/><y_{y1}/></graphics>"
    else:
        return "<graphics></graphics>"


class PklData(TypedDict):
    root: RootElement
    coord_scale_factor: Tuple[float, float]
    page_width_px: int
    page_height_px: int


def load_pkl_file(filepath: Union[str, Path]) -> PklData:
    """Load a pickle file"""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_pkl_file(filepath: Union[str, Path], obj: PklData) -> None:
    """Load a pickle file"""
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def convert(
    root: RootElement,
    coord_scale_factor: Tuple[float, float],
    wordwise_coordinates: bool,
    coord_transformer: Optional[Callable[[float, float], Tuple[float, float]]] = None,
    bbox_transformer: Optional[
        Callable[[float, float, float, float], Tuple[float, float, float, float]]
    ] = None,
    root_is_compact: bool = False,
) -> str:
    """Convert a root element to a string"""
    if not root_is_compact:
        compact_in_place(root)
    assert isinstance(root, RootElement)
    # TODO: Remove this in v6, when the issue is fixed in the generator
    assert not (
        len(root.children) == 1
        and isinstance(root.children[0], ParagraphElement)
        and len(root.children[0].children) == 1
        and isinstance(root.children[0].children[0], GraphicsElement)
    ), "Page only consists of a graphic"
    return _convert(
        root,
        State(
            coord_scale_factor=coord_scale_factor,
            wordwise_coordinates=wordwise_coordinates,
            coord_transformer=coord_transformer,
            bbox_transformer=bbox_transformer,
        ),
    )


def struct_to_bbox(struct_out: StructuredOutput, wordwise_coordinates: bool = False) -> None:
    root = copy.deepcopy(struct_out._root)
    compact_in_place(root)
    converted = convert(
        root,
        struct_out.coord_scale_factor,
        wordwise_coordinates=wordwise_coordinates,
        root_is_compact=True,
    )

    with open(f"{struct_out.page_name}.bbox.compact.txt", "w") as wf:
        wf.write(str(root))

    with open(f"{struct_out.page_name}.bbox", "w") as wf:
        wf.write(converted)


class NestedBboxNode:
    node_type: str
    text_content: Optional[str] = None
    children: Optional[List["NestedBboxNode"]] = None

    bbox: Optional[List[Optional[int]]] = None

    rowspan: Optional[int] = None
    colspan: Optional[int] = None

    _element_re = re.compile(
        r"<(?:x_(?P<x>\d+)|y_(?P<y>\d+)|colspan_(?P<colspan>\d+)|rowspan_(?P<rowspan>\d+)|br)/>|</?(?:float|p|h1|h2|h3|h4|h5|h6|h7|h8|h9|latexmath|code_block|code|strong|em|sub|sup|table|tr|td|ul|ol|li|footnote|footnote_ref|header|footer|graphics)>"
    )

    _newline_cleanup_re = re.compile(r"\n\n\n+")

    def __init__(
        self, node_type: str, has_children: bool = True, text_content: Optional[str] = None
    ) -> None:
        """
        Create a new NestedBboxNode

        Args:
            node_type: The type of the node.
            text_content: The text content of the node if node_type == '_text'.
        """
        self.node_type = node_type
        if text_content is None:
            if has_children:
                self.children = []
        else:
            self.text_content = text_content

    @staticmethod
    def text_node(text: str) -> "NestedBboxNode":
        return NestedBboxNode("_text", text_content=text)

    @staticmethod
    def parse(text: str) -> "NestedBboxNode":
        """Parse a text into a nested bbox tree"""
        root = NestedBboxNode("_root")
        stack = [root]
        current = root
        last_pos = 0
        try:
            for match in NestedBboxNode._element_re.finditer(text):
                # First, flush the text in between
                if match.start() > last_pos:
                    assert current.bbox is None or (
                        current.bbox[2] is None and current.bbox[3] is None
                    ), "Coordinates not boxed by element"
                    # Split the text into LayoutBlocks
                    current.children.append(
                        NestedBboxNode.text_node(text[last_pos : match.start()])
                    )
                # Process the element
                el = match.group(0)
                if el.startswith("</"):
                    # closing element, search for matching same?
                    if current.node_type == "layoutblock":
                        # In layoutblock mode, pop the layoutblock
                        stack.pop()
                        current = stack[-1]
                    assert current.node_type == el[2:-1], (
                        f"Unexpected closing tag {el}, expected {current.node_type}"
                    )
                    assert current.bbox is None or (
                        (current.bbox[0] is None) == (current.bbox[2] is None)
                        and (current.bbox[1] is None) == (current.bbox[3] is None)
                    ), f"Box not properly defined: {current.bbox}"
                    stack.pop()
                    current = stack[-1]
                elif el.endswith("/>"):
                    # self-closing element
                    if match.group("x"):
                        if current.bbox is None:
                            current.bbox = [None, None, None, None]
                        if len(current.children) == 0:
                            assert current.bbox[2] is None, "Overwriting x coordinates"
                            if current.bbox[0] is None:
                                current.bbox[0] = int(match.group("x"))
                            else:
                                current.bbox[2] = int(match.group("x"))
                        else:
                            if current.bbox[3] is not None:
                                # There is a layout break here
                                if current.node_type != "layoutblock":
                                    # Move everything to a layoutblock element
                                    layoutblock_element = NestedBboxNode("layoutblock")
                                    layoutblock_element.children = current.children
                                    layoutblock_element.bbox = current.bbox
                                    current.children = [layoutblock_element]
                                    current.bbox = None
                                else:
                                    # Already in layoutblock mode, pop the current block
                                    stack.pop()
                                    current = stack[-1]
                                # Next layoutblock
                                assert current.bbox is None, "Overlapping layoutblocks"
                                layoutblock_element = NestedBboxNode("layoutblock")
                                current.children.append(layoutblock_element)
                                stack.append(layoutblock_element)
                                current = layoutblock_element
                                current.bbox = [int(match.group("x")), None, None, None]
                            else:
                                assert current.bbox[0] is not None, (
                                    f"Missing bbox start for {current.node_type}"
                                )
                                assert current.bbox[2] is None, "Overwriting x coordinates"
                                current.bbox[2] = int(match.group("x"))
                    elif match.group("y"):
                        if current.bbox is None:
                            current.bbox = [None, None, None, None]
                        if len(current.children) == 0:
                            assert current.bbox[3] is None, "Overwriting y coordinates"
                            if current.bbox[1] is None:
                                current.bbox[1] = int(match.group("y"))
                            else:
                                current.bbox[3] = int(match.group("y"))
                        else:
                            assert current.bbox[1] is not None, (
                                f"Missing bbox start for {current.node_type}"
                            )
                            assert current.bbox[3] is None, "Overwriting y coordinates"
                            current.bbox[3] = int(match.group("y"))
                    elif match.group("colspan"):
                        assert current.colspan is None
                        assert current.node_type == "td", "colspan only allowed in table cells"
                        assert len(current.children) == 0, "colspan after content"
                        current.colspan = int(match.group("colspan"))
                    elif match.group("rowspan"):
                        assert current.rowspan is None
                        assert current.node_type == "td", "rowspan only allowed in table cells"
                        assert len(current.children) == 0, "rowspan after content"
                        current.rowspan = int(match.group("rowspan"))
                    else:
                        current.children.append(NestedBboxNode(el[1:-2], has_children=False))
                else:
                    assert current.bbox is None or (
                        current.bbox[2] is None and current.bbox[3] is None
                    ), "Coordinates not boxed by element"
                    # opening element
                    new_node = NestedBboxNode(el[1:-1])
                    current.children.append(new_node)
                    stack.append(new_node)
                    current = new_node
                last_pos = match.end()
            assert len(stack) == 1, f"Unclosed tags {stack}"
        except AssertionError as e:
            raise AssertionError(f"Tree so far: {root}") from e
        if last_pos < len(text):
            current.children.append(NestedBboxNode("_text", text_content=text[last_pos:]))
        return root

    def get_all_boxes(
        self,
    ) -> Generator[Tuple[int, int, int, int], None, None]:
        # Special case for table. Ignores other siblings.
        if self.node_type == "table":
            for row in self.children:
                if row.node_type == "tr":
                    for cell in row.children:
                        if cell.node_type == "td":
                            if (
                                row.bbox is not None
                                and row.bbox[1] is not None
                                and row.bbox[3] is not None
                                and cell.bbox is not None
                                and cell.bbox[0] is not None
                                and cell.bbox[2] is not None
                            ):
                                yield [cell.bbox[0], row.bbox[1], cell.bbox[2], row.bbox[3]]
                        yield from cell.get_all_boxes()
                else:
                    yield from row.get_all_boxes()
            return
        # Default bbox
        if self.bbox is not None and all(c is not None for c in self.bbox):
            yield self.bbox
        # Traverse children
        if self.children is not None:
            for child in self.children:
                yield from child.get_all_boxes()

    def get_math_only(self) -> Generator["NestedBboxNode", None, None]:
        """Get all math elements in the tree."""
        if self.node_type == "latexmath":
            yield self
        if self.children is not None:
            for child in self.children:
                yield from child.get_math_only()

    def get_table_only(self) -> Generator["NestedBboxNode", None, None]:
        """Get all table elements in the tree (not traversing into nested table elements)."""
        if self.node_type == "table":
            yield self
        elif self.children is not None:
            for child in self.children:
                yield from child.get_table_only()

    def get_float_only(self) -> Generator["NestedBboxNode", None, None]:
        """Get all float elements in the tree (not traversing into nested float elements)."""
        if self.node_type == "float":
            yield self
        elif self.children is not None:
            for child in self.children:
                yield from child.get_float_only()

    def get_text_only(
        self,
        include_float: bool = True,
        include_table: bool = False,
        include_footnote: bool = True,
        include_layout: bool = True,
        include_math: bool = True,
        paragraph_separator: str = "\n\n",
    ) -> str:
        """
        Returns only the inner text of the tree, without any bounding boxes or other elements.

        Args:
            include_float: If true, also output floats.
            float_last: If true, floats are output last.
            include_table: If true, also output tables.
            include_footnote: If true, also output footnotes.
            include_layout: If true, also output headers and footers.
            include_math: If true, also output math elements.
            paragraph_separator: The separator to use between paragraphs.

        Returns:
            The text content of the tree.
        """

        def _get_text_only(node: NestedBboxNode) -> Generator[str, None, None]:
            if node.node_type == "_text":
                yield node.text_content
            elif node.node_type == "br":
                yield "\n"
            elif node.node_type in (
                "float",
                "p",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "h7",
                "h8",
                "h9",
                "code_block",
                "ul",
                "ol",
                "li",
                "footnote",
                "header",
                "footer",
                "table",
                "tr",
            ):
                yield paragraph_separator
                if (
                    (node.node_type == "float" and not include_float)
                    or (node.node_type == "footnote" and not include_footnote)
                    or (node.node_type in ("header", "footer") and not include_layout)
                    or (node.node_type in ("table", "tr") and not include_table)
                ):
                    return
                assert node.children is not None
                for child in node.children:
                    yield from _get_text_only(child)
                yield paragraph_separator
            elif node.children is not None and (node.node_type != "latexmath" or include_math):
                for child in node.children:
                    yield from _get_text_only(child)

        return self._newline_cleanup_re.sub("\n\n", "".join(_get_text_only(self))).strip()

    def to_str(self, include_coords: bool = True) -> str:
        if self.node_type == "_root":
            return "".join(child.to_str(include_coords=include_coords) for child in self.children)
        if self.node_type == "_text":
            return self.text_content
        if self.children is None:
            return f"<{self.node_type}/>"
        prefix = ""
        suffix = ""
        if self.node_type != "layoutblock":
            prefix += f"<{self.node_type}>"
        if self.node_type != "layoutblock":
            suffix += f"</{self.node_type}>"
        if self.bbox is not None and include_coords:
            if self.bbox[0] is not None:
                prefix += f"<x_{self.bbox[0]}/>"
            if self.bbox[1] is not None:
                prefix += f"<y_{self.bbox[1]}/>"
            if self.bbox[3] is not None:
                suffix = f"<y_{self.bbox[3]}/>" + suffix
            if self.bbox[2] is not None:
                suffix = f"<x_{self.bbox[2]}/>" + suffix
        if self.colspan is not None:
            prefix += f"<colspan_{self.colspan}/>"
        if self.rowspan is not None:
            prefix += f"<rowspan_{self.rowspan}/>"
        return f"{prefix}{''.join(child.to_str(include_coords=include_coords) for child in self.children)}{suffix}"

    def __str__(self) -> str:
        return self.to_str()
