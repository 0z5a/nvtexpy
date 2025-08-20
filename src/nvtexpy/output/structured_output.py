# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type, TypeVar, Union

from nvtexpy.output import math


class Transform(ABC):
    @abstractmethod
    def __matmul__(self, other: "Transform") -> "Transform": ...

    @abstractmethod
    def transform_scaledint(self, x: int, y: int) -> Tuple[int, int]: ...

    @abstractmethod
    def transform_scaledint_rect(
        self, left: int, top: int, right: int, bottom: int
    ) -> Tuple[int, int, int, int]: ...

    @abstractmethod
    def inv_transform_scaledint(self, x: int, y: int) -> Tuple[int, int]: ...

    @abstractmethod
    def inv_transform_scaledint_rect(
        self, left: int, top: int, right: int, bottom: int
    ) -> Tuple[int, int, int, int]: ...

    @staticmethod
    def identity() -> "Transform":
        return IdentityTransform()

    @staticmethod
    def from_tex(text: str, cur_h: int, cur_v: int, page_height: int) -> "Transform":
        # pdfsetmatrix(str_start[str_ptr], cur_h, cur_page_height - cur_v)
        try:
            # sscanf((const char *) in, " %lf %lf %lf %lf ", &x.a, &x.b, &x.c, &x.d)
            x = [float(a) for a in text.split(" ")]
        except ValueError:
            assert False, f"Failed to parse matrix: {text}"
        assert len(x) == 4
        # No idea why, but empirically, these must be * -1.
        x[1] *= -1
        x[2] *= -1
        # x.e = (double) pos.h * (1.0 - x.a) - (double) pos.v * x.c;
        # x.f = (double) pos.v * (1.0 - x.d) - (double) pos.h * x.b;
        # This should actually be part of the code, but empirically must be omitted.
        # cur_v = page_height - cur_v
        x.append(int(cur_h) * (1.0 - x[0]) - (int(cur_v) * x[2]))
        x.append(int(cur_v) * (1.0 - x[3]) - (int(cur_h) * x[1]))
        return TransformImpl(*x)


class IdentityTransform(Transform):
    """Identity noop transformation"""

    def __matmul__(self, other: "Transform") -> "Transform":
        return other

    def transform_scaledint(self, x: int, y: int) -> Tuple[int, int]:
        return int(x), int(y)

    def transform_scaledint_rect(
        self, left: int, top: int, right: int, bottom: int
    ) -> Tuple[int, int, int, int]:
        return int(left), int(top), int(right), int(bottom)

    def inv_transform_scaledint(self, x: int, y: int) -> Tuple[int, int]:
        return int(x), int(y)

    def inv_transform_scaledint_rect(
        self, left: int, top: int, right: int, bottom: int
    ) -> Tuple[int, int, int, int]:
        return int(left), int(top), int(right), int(bottom)

    def __repr__(self) -> str:
        return "IdentityTransform()"


@dataclass(frozen=True, order=True)
class TransformImpl(Transform):
    """The actual pdf transformation"""

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    def __matmul__(self, other: "Transform") -> "Transform":
        if isinstance(other, TransformImpl):
            return TransformImpl(
                self.a * other.a + self.b * other.c,
                self.a * other.b + self.b * other.d,
                self.c * other.a + self.d * other.c,
                self.c * other.b + self.d * other.d,
                self.e * other.a + self.f * other.c + other.e,
                self.e * other.b + self.f * other.d + other.f,
            )
        return self

    def transform_scaledint(self, x: int, y: int) -> Tuple[int, int]:
        res = (
            round(float(x) * self.a + float(y) * self.c + self.e),
            round(float(x) * self.b + float(y) * self.d + self.f),
        )
        # print(f"Transform Scaledint {x, y} -> {res}")
        return res

    def transform_scaledint_rect(
        self, left: int, top: int, right: int, bottom: int
    ) -> Tuple[int, int, int, int]:
        x1, y1 = self.transform_scaledint(left, top)
        x2, y2 = self.transform_scaledint(right, top)
        x3, y3 = self.transform_scaledint(left, bottom)
        x4, y4 = self.transform_scaledint(right, bottom)
        return (
            min(x1, x2, x3, x4),
            min(y1, y2, y3, y4),
            max(x1, x2, x3, x4),
            max(y1, y2, y3, y4),
        )

    def inv_transform_scaledint(self, x: int, y: int) -> Tuple[int, int]:
        d = self.b * self.c - self.a * self.d
        assert d != 0
        return (
            round((self.c * float(y) - self.d * float(x) - self.c * self.f + self.d * self.e) / d),
            round((self.b * float(x) - self.a * float(y) - self.b * self.e + self.a * self.f) / d),
        )

    def inv_transform_scaledint_rect(
        self, left: int, top: int, right: int, bottom: int
    ) -> Tuple[int, int, int, int]:
        x1, y1 = self.inv_transform_scaledint(left, top)
        x2, y2 = self.inv_transform_scaledint(right, top)
        x3, y3 = self.inv_transform_scaledint(left, bottom)
        x4, y4 = self.inv_transform_scaledint(right, bottom)
        return (
            min(x1, x2, x3, x4),
            min(y1, y2, y3, y4),
            max(x1, x2, x3, x4),
            max(y1, y2, y3, y4),
        )


@dataclass
class BoundingBox:
    """A bounding box of a character or a string.
    The origin of the coordinate system is the upper left corner of the page.
    The (x, y) position of the bounding box should point to the upper left corner
    of the character or string."""

    x: int
    y: int
    y_baseline: int
    width: int
    height: int

    @staticmethod
    def from_tex(
        cur_h: int,
        cur_v: int,
        width: int,
        height: int,
        depth: int,
        transform: Union[Transform, IdentityTransform],
    ) -> "BoundingBox":
        cur_h = int(cur_h)
        cur_v = int(cur_v)
        width = int(width)
        height = int(height)
        depth = int(depth)
        x = cur_h
        y = cur_v - height
        height += depth
        if not isinstance(transform, IdentityTransform):
            x, y, width, height = transform.transform_scaledint_rect(x, y, x + width, y + height)
            width -= x
            height -= y
        return BoundingBox(x=x, y=y, y_baseline=cur_v, width=width, height=height)

    @staticmethod
    def bboxes_union(bboxes: List[Optional["BoundingBox"]]) -> Optional["BoundingBox"]:
        """Computes the union of a list of bounding boxes (might normalize transform space)."""
        bboxes = [b for b in bboxes if b is not None]
        if len(bboxes) == 0:
            return None
        elif len(bboxes) == 1:
            return bboxes[0]

        new_x = min(*[b.x for b in bboxes])
        new_y = min(*[b.y for b in bboxes])
        new_width = max(*[b.x + b.width for b in bboxes]) - new_x
        new_height = max(*[b.y + b.height for b in bboxes]) - new_y
        # assert all([b.y_baseline == bboxes[0].y_baseline for b in bboxes])

        return BoundingBox(new_x, new_y, bboxes[0].y_baseline, new_width, new_height)

    def __or__(self, other: "BoundingBox") -> "BoundingBox":
        """Returns the union of this bounding box and another one."""
        return BoundingBox.bboxes_union([self, other])

    def update_union(self, other: "BoundingBox"):
        """Returns the union of this bounding box and another one."""
        newbb = self | other
        self.x = newbb.x
        self.y = newbb.y
        self.y_baseline = newbb.y_baseline
        self.width = newbb.width
        self.height = newbb.height

    def as_xyxy_list_scaled(self, scale_x: float = 1.0, scale_y: float = 1.0) -> List[int]:
        # print(f"TR: as_list_scaled: {self.transform} {self.x, self.y, self.x + self.width, self.y + self.height} => {x0=}, {y0=}, {x1=}, {y1=}")
        return [
            round(self.x * scale_x),
            round(self.y * scale_y),
            round((self.x + self.width) * scale_x),
            round((self.y + self.height) * scale_y),
        ]

    def as_xywh_list_scaled(self, scale_x: float = 1.0, scale_y: float = 1.0) -> List[int]:
        # print(f"TR: as_list_scaled: {self.transform} {self.x, self.y, self.x + self.width, self.y + self.height} => {x0=}, {y0=}, {x1=}, {y1=}")
        return [
            round(self.x * scale_x),
            round(self.y * scale_y),
            round(self.width * scale_x),
            round(self.height * scale_y),
        ]

    def hocr_str(self, scale_x: float = 1.0, scale_y: float = 1.0) -> str:
        return " ".join(str(round(f)) for f in self.as_xyxy_list_scaled(scale_x, scale_y))


class Element(ABC):
    """One node / element in the structured output tree."""

    children: List["Element"]
    boxes: List[BoundingBox]  # The bounding boxes of this element.

    keep_empty: bool = True  # Whether to keep empty elements of this type
    # Is this an inline formatter?
    is_inline: bool = False

    def __init__(self, children: Optional[List["Element"]] = None) -> None:
        self.children = [] if children is None else children
        self.boxes = []

    def compute_bounding_box(self) -> Optional[BoundingBox]:
        """Computes the bounding box of this element by combining the bounding boxes of the children."""

        if len(self.children) == 0:
            if len(self.boxes) == 0:
                return None
            else:
                return BoundingBox.bboxes_union(self.boxes)

        child_boxes = [child.compute_bounding_box() for child in self.children]

        final_boxes = [b for b in child_boxes + self.boxes if b is not None]

        if len(final_boxes) == 0:
            return None
        else:
            return BoundingBox.bboxes_union(final_boxes)

    def to_str_compact(self, indent: str = "") -> str:
        clean_dict = self.__dict__.copy()
        clean_dict.pop("children")
        clean_dict.pop("boxes")

        if len(clean_dict) > 0:
            r = f"<{self.__class__.__name__}, " + ", ".join(
                f"{k}={v}" for k, v in clean_dict.items()
            )
        else:
            r = f"<{self.__class__.__name__}"
        if len(self.children) > 0:
            in_text = False
            for child in self.children:
                if isinstance(child, (CharacterElement, WordElement, SpaceElement, HyphenElement)):
                    if not in_text:
                        r += f'\n{indent}  "'
                        in_text = True
                    if isinstance(child, CharacterElement):
                        r += child.character
                    elif isinstance(child, WordElement):
                        for grandchild in child.children:
                            if isinstance(grandchild, HyphenElement):
                                r += f"<Hyphen {grandchild.character!r}>"
                            elif isinstance(grandchild, CharacterElement):
                                r += grandchild.character
                            else:
                                r += grandchild.to_str_compact(indent + "  ")
                    elif isinstance(child, HyphenElement):
                        r += f"<Hyphen {child.character!r}>"
                    else:
                        r += " "
                else:
                    if in_text:
                        r += '"'
                        in_text = False
                    r += f"\n{indent}  " + child.to_str_compact(indent + "  ")
            if in_text:
                r += '"'
        if "\n" in r:
            r += f"\n{indent}>"
        else:
            r += " />"

        return r

    def to_str(self, indent: str = "") -> str:
        dict_without_children_parents = self.__dict__.copy()
        dict_without_children_parents.pop("children")
        if len(dict_without_children_parents["boxes"]) == 0:
            dict_without_children_parents.pop("boxes")

        if len(dict_without_children_parents) > 0:
            r = f"<{self.__class__.__name__}, {dict_without_children_parents}"
        else:
            r = f"<{self.__class__.__name__}"
        if len(self.children) > 0:
            for child in self.children:
                r += f"\n{indent}  " + child.to_str(indent + "  ")

        if "\n" in r:
            r += f"\n{indent}>"
        else:
            r += " />"

        return r

    def __str__(self) -> str:
        """Displays the element and it's sub-tree recursively."""
        return self.to_str()


class RootElement(Element):
    """The one and only root element of the document page."""

    pass


class CharacterElement(Element):
    """An element representing a single character."""

    is_inline = True

    character: str  # The character

    def __init__(self, character: str, box: BoundingBox) -> None:
        super().__init__()
        self.character = character
        self.boxes = [box]

    @property
    def box(self) -> BoundingBox:
        return self.boxes[0]


class HyphenElement(Element):
    """An element representing a single character which is a hyphen."""

    is_inline = True

    character: str  # The character

    def __init__(self, character: str, box: BoundingBox) -> None:
        super().__init__()
        self.character = character
        self.boxes = [box]

    @property
    def box(self) -> BoundingBox:
        return self.boxes[0]


class WordElement(Element):
    """An element representing a word."""

    is_inline = True

    word: str

    def __init__(self, word: str, boxes: Optional[List[BoundingBox]] = None) -> None:
        super().__init__()
        self.word = word
        assert len(word) > 0
        if boxes is not None:
            self.boxes = boxes

    def to_str(self, indent: str = "", compact_text: bool = False) -> str:
        if compact_text:
            return f"{self.word}"
        else:
            return f"{self.word!r}"


class SpaceElement(Element):
    """An element representing word separation."""

    is_inline = True


class LinebreakElement(Element):
    """An element representing a line break."""

    is_inline = True


class ParagraphbreakElement(Element):
    """An element representing a paragraph break."""

    pass


class LayoutbreakElement(Element):
    """An element representing a layout break."""

    is_inline = True


class ParagraphElement(Element):
    """An element representing a paragraph."""

    # Sometimes ParagraphElement is used to represent a math equation instead of plain text
    is_math_formula: bool = False

    def __init__(self, *args, is_math_formula: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_math_formula = is_math_formula


class LayoutBlockElement(Element):
    """An element representing a layout block."""

    pass


class BeginOfParagraphElement(Element):
    """An element representing the begin of a paragraph."""

    is_inline = True


class EndOfParagraphElement(Element):
    """An element representing the end of a paragraph."""

    is_inline = True


class EmphasisElement(Element):
    keep_empty = False
    is_inline = True


class StrongElement(Element):
    keep_empty = False
    is_inline = True


class StrikethroughElement(Element):
    keep_empty = False
    is_inline = True


class UnderlineElement(Element):
    keep_empty = False
    is_inline = True


class OverlineElement(Element):
    keep_empty = False
    is_inline = True


class HighlightElement(Element):
    keep_empty = False
    is_inline = True


class CodeElement(Element):
    keep_empty = False
    is_inline = True


class SubscriptElement(Element):
    keep_empty = False
    is_inline = True


class SuperscriptElement(Element):
    keep_empty = False
    is_inline = True


class CodeBlockElement(Element):
    keep_empty = False


class AlgorithmElement(Element):
    keep_empty = False


class AlgorithmIndentElement(Element):
    keep_empty = False


class LineNumberElement(Element):
    keep_empty = False
    is_inline = True


class BibliographyElement(Element):
    keep_empty = False


class AlignBlockElement(Element):
    keep_empty = False
    starred: bool
    env_name: str

    def __init__(self, starred: bool, env_name: str) -> None:
        super().__init__()
        self.starred = starred
        self.env_name = env_name


class AlignBlockEqNumElement(Element):
    def __init__(self) -> None:
        super().__init__()


class HeadingElement(Element):
    level: int

    def __init__(self, level: int) -> None:
        super().__init__()
        self.level = int(level)


class SuppressElement(Element):
    is_inline = True


class FloatElement(Element):
    # The type of the float (e.g. "figure", "table")
    environment: str

    def __init__(self, environment: str, children: Optional[List[Element]] = None) -> None:
        super().__init__(children)
        self.environment = environment


class TableOfContentsElement(Element):
    pass


class FootnoteElement(Element):
    pass


class FootnoteMarkElement(Element):
    is_inline = True


class HeaderElement(Element):
    pass


class FooterElement(Element):
    pass


class PageNumberElement(Element):
    is_inline = True


class UnorderedListElement(Element):
    keep_empty = False


class OrderedListElement(Element):
    keep_empty = False


class ListItemElement(Element):
    pass


class ListItemLabelElement(Element):
    pass


class TableElement(Element):
    keep_empty = False


class TableRowElement(Element):
    border_top: bool
    border_bottom: bool

    is_first: bool
    is_last: bool

    def __init__(
        self, border_top: bool, border_bottom: bool, is_first: bool, is_last: bool
    ) -> None:
        super().__init__()
        self.border_top = border_top
        self.border_bottom = border_bottom
        self.is_first = is_first
        self.is_last = is_last


class TableCellElement(Element):
    colspan: int
    rowspan: int
    border_left: bool
    border_top: bool
    border_right: bool
    border_bottom: bool

    def __init__(
        self,
        colspan: int,
        rowspan: int,
        border_left: bool,
        border_top: bool,
        border_right: bool,
        border_bottom: bool,
    ) -> None:
        super().__init__()
        self.colspan = int(colspan)
        self.rowspan = int(rowspan)
        self.border_left = border_left
        self.border_top = border_top
        self.border_right = border_right
        self.border_bottom = border_bottom


class MathElement(Element):
    #: The code of the element (hierarchical)
    code: math.ITNode
    #: If this element is in displaymath context
    is_display: bool
    #: If this element is equation number
    is_eq_num: bool

    def __init__(self, code: math.ITNode, is_display: bool, is_eq_num: bool) -> None:
        super().__init__()
        self.code = code
        self.is_display = is_display
        self.is_inline = not is_display
        self.is_eq_num = is_eq_num

    def add_box(self, box: BoundingBox) -> None:
        self.boxes.append(box)


class GraphicsElement(Element):
    is_inline = True


TElement = TypeVar("TElement", bound=Element)


class StructuredOutput:
    """This is a tree of Element nodes.
    The root is a StructuredOutputRootElement. Since we are building the tree by pushing and popping elements,
    we need to keep track of the current path in the tree."""

    _root: RootElement
    _path_to_current: List[Element]
    page_name: str  # The name of the page

    # The current pdf transformation
    _transform_stack: List[Transform]

    def __init__(self, page_number: int, page_size: Tuple[int, int]) -> None:
        self._root = RootElement()
        self._path_to_current = [self._root]
        self.page_number = page_number
        self.page_name = f"nvpdftex_page_{page_number}"
        self.page_size = page_size

        # Some constants for the conversion from TeX points to pixels
        magic_factor = 1 / 65781.76
        points_per_inch = 72  # 1 inch = 72 pt
        pixels_per_inch = int(
            os.environ.get("HOCR_IMAGE_DPI", 300)
        )  # PPI of our PNG of the PDF (-r flag to ghostscript)
        pixels_per_point = pixels_per_inch / points_per_inch

        page_width_pt = page_size[0] * magic_factor
        page_height_pt = page_size[1] * magic_factor

        self.page_width_px = round(page_width_pt * pixels_per_point)
        self.page_height_px = round(page_height_pt * pixels_per_point)

        # Multiply the tex points by this factor to get the pixel coordinates
        self.coord_scale_factor = (pixels_per_point * magic_factor, pixels_per_point * magic_factor)

        self._transform_stack = [Transform.identity()]

    def save(self):
        with open(self.page_name + ".txt", "w") as f:
            f.write(str(self._root))

    def _pop(self, idx: int):
        # Potentially close the elmements. Needed?
        # for stack_element in reversed(self._path_to_current[idx:]):
        #     pass

        del self._path_to_current[idx:]

    def push(self, element: TElement) -> TElement:
        # element.start(self._writer, self._format_state)

        self._path_to_current[-1].children.append(element)
        self._path_to_current.append(element)

        return element

    def pop(self, element: Element) -> bool:
        # Remove the element from the stack, if it is in the stack, and close all elements, which were opened after the element
        try:
            idx = self._path_to_current.index(element)
        except ValueError:
            return False
        else:
            self._pop(idx)
            return True

    def push_pop(self, element: Element) -> None:
        self.pop(self.push(element))

    def pop_type(self, tp: Type[Element]) -> bool:
        for idx, stack_element in enumerate(reversed(self._path_to_current)):
            if isinstance(stack_element, tp):
                self._pop(len(self._path_to_current) - idx - 1)
                return True
        return False

    def has(self, tp: Type[Element]) -> bool:
        for stack_element in self._path_to_current:
            if isinstance(stack_element, tp):
                return True
        return False

    def push_transform(self) -> None:
        self._transform_stack.append(self._transform_stack[-1])

    def pop_transform(self) -> None:
        self._transform_stack.pop()
        assert len(self._transform_stack) > 0

    def transform(self, transformation: Transform) -> None:
        self._transform_stack[-1] = transformation @ self._transform_stack[-1]

    @property
    def current_transform(self) -> Transform:
        return self._transform_stack[-1]
