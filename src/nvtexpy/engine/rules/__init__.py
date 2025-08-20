# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pyright: reportUnusedImport=false
import typing

import nvtexpy.engine.rules.align
import nvtexpy.engine.rules.bibliography
import nvtexpy.engine.rules.code
import nvtexpy.engine.rules.end_of_paragraph
import nvtexpy.engine.rules.float
import nvtexpy.engine.rules.footnote
import nvtexpy.engine.rules.graphics
import nvtexpy.engine.rules.heading
import nvtexpy.engine.rules.ligature
import nvtexpy.engine.rules.minipage
import nvtexpy.engine.rules.page
import nvtexpy.engine.rules.pdfstringdef
import nvtexpy.engine.rules.strikeout
import nvtexpy.engine.rules.svg_writer
import nvtexpy.engine.rules.tikz
import nvtexpy.engine.rules.toc
from nvtexpy.engine.rules.base import Rules
from nvtexpy.engine.rules.box_tracking import BoxTrackRule
from nvtexpy.engine.rules.font import FontRule
from nvtexpy.engine.rules.list import ListRule
from nvtexpy.engine.rules.math import MathRule
from nvtexpy.engine.rules.table import TableRule
from nvtexpy.engine.rules.transform import Transform

if typing.TYPE_CHECKING:
    # Use them to avoid cleaning up the imports
    nvtexpy.engine.rules.align
    nvtexpy.engine.rules.bibliography
    nvtexpy.engine.rules.code
    nvtexpy.engine.rules.end_of_paragraph
    nvtexpy.engine.rules.float
    nvtexpy.engine.rules.footnote
    nvtexpy.engine.rules.graphics
    nvtexpy.engine.rules.heading
    nvtexpy.engine.rules.ligature
    nvtexpy.engine.rules.minipage
    nvtexpy.engine.rules.page
    nvtexpy.engine.rules.pdfstringdef
    nvtexpy.engine.rules.strikeout
    nvtexpy.engine.rules.svg_writer
    nvtexpy.engine.rules.tikz
    nvtexpy.engine.rules.toc


__all__ = [
    "BoxTrackRule",
    "Rules",
    "MathRule",
    "TableRule",
    "FontRule",
    "ListRule",
    "Transform",
]
