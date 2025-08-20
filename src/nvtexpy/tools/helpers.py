# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import re
from pathlib import Path
from typing import Literal
from abc import ABC, abstractmethod

from nvtexpy.output.flat_layout import convert_from_pkl


_re_page_num = re.compile(r"nvpdftex_page_(\d+)\.png")


def find_generated_pages(work_dir: Path) -> list[dict]:
    """
    Find all generated files in the work directory.

    Args:
        work_dir: Working directory to search

    Returns:
        List of pages
    """
    pages = []
    for image_file in work_dir.glob("nvpdftex_page_*.png"):
        m = _re_page_num.match(image_file.name)
        assert m is not None, f"Expected page number in {image_file.name}"
        page_num = int(m.group(1))
        pickle_file = work_dir / f"nvpdftex_page_{page_num}.pkl"
        if not pickle_file.exists():
            pickle_file = None
        pages.append(
            {
                "page_num": page_num,
                "image_file": image_file.name,
                "pickle_file": pickle_file.name if pickle_file else None,
            }
        )
    pages.sort(key=lambda x: x["page_num"])
    return pages


class StructuredConverter(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, pickle_file: Path) -> str: ...


class FlatLayoutConverter(StructuredConverter):
    def __init__(
        self,
        name: str = "flat_layout",
        table_format: Literal["latex", "html"] = "latex",
        style_format: Literal["markdown", "html"] = "markdown",
    ):
        super().__init__(name)
        self.table_format: Literal["latex", "html"] = table_format
        self.style_format: Literal["markdown", "html"] = style_format

    def __call__(self, pickle_file: Path) -> str:
        return convert_from_pkl(
            str(pickle_file),
            table_format=self.table_format,
            style_format=self.style_format,
        )
