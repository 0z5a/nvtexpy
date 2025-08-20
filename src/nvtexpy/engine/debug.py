# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING

DEBUG_CALLS = bool(os.environ.get("NVTEXPY_DEBUG_CALLS", ""))
DEBUG_ALL_CALLS = bool(os.environ.get("NVTEXPY_DEBUG_ALL_CALLS", ""))
DEBUG_CALL_RECORDING = bool(os.environ.get("NVTEXPY_DEBUG_CALL_RECORDING", ""))
DEBUG_CALL_ARGS = bool(os.environ.get("NVTEXPY_DEBUG_CALL_ARGS", ""))
DEBUG_CHAR_ALLOCS = bool(os.environ.get("NVTEXPY_DEBUG_CHAR_ALLOCS", ""))
DEBUG_NODE_ALLOCS = bool(os.environ.get("NVTEXPY_DEBUG_NODE_ALLOCS", ""))
# Assign a random id to the tracked nodes. No measurable speed impact.
DEBUG_NODE_ID = bool(os.environ.get("NVTEXPY_DEBUG_NODE_ID", DEBUG_CHAR_ALLOCS))
DEBUG_NODE_OUT = bool(os.environ.get("NVTEXPY_DEBUG_NODE_OUT", ""))
DEBUG_BOX = bool(os.environ.get("NVTEXPY_DEBUG_BOX", ""))
DEBUG_COPY = bool(os.environ.get("NVTEXPY_DEBUG_COPY", ""))
DEBUG_SAVELEVELS = bool(os.environ.get("NVTEXPY_DEBUG_SAVELEVELS", ""))
DEBUG_MD_STACK = bool(os.environ.get("NVTEXPY_DEBUG_MD_STACK", ""))
DEBUG_INPUT_FORMATTER = bool(os.environ.get("NVTEXPY_DEBUG_INPUT_FORMATTER", ""))
DEBUG_OUTPUT_STACK = bool(os.environ.get("NVTEXPY_DEBUG_OUTPUT_STACK", ""))
DEBUG_SECTION = bool(os.environ.get("NVTEXPY_DEBUG_SECTION", ""))
DEBUG_LIST = bool(os.environ.get("NVTEXPY_DEBUG_LIST", ""))
DEBUG_TABLE = bool(os.environ.get("NVTEXPY_DEBUG_TABLE", ""))
DEBUG_MATH = bool(os.environ.get("NVTEXPY_DEBUG_MATH", ""))
DEBUG_MATH_ALLOC = bool(os.environ.get("NVTEXPY_DEBUG_MATH_ALLOC", ""))
DEBUG_MATH_SIMPLIFY = bool(os.environ.get("NVTEXPY_DEBUG_MATH_SIMPLIFY", ""))
DEBUG_WRITE_PAGE_SVG = bool(os.environ.get("NVTEXPY_DEBUG_WRITE_PAGE_SVG", ""))
DEBUG_ASSERT_FAIL = bool(os.environ.get("NVTEXPY_DEBUG_ASSERT_FAIL", ""))
DEBUG_GRAPHICS = bool(os.environ.get("NVTEXPY_DEBUG_GRAPHICS", ""))
DEBUG_PAR = bool(os.environ.get("NVTEXPY_DEBUG_PAR", ""))
DEBUG_LSTLISTINGS = bool(os.environ.get("NVTEXPY_DEBUG_LSTLISTINGS", ""))
DEBUG_RESTRUCTURING = bool(os.environ.get("NVTEXPY_DEBUG_RESTRUCTURING", ""))
# After 10sec, kill the main process for debugging
DEBUG_WATCHDOG = bool(os.environ.get("NVTEXPY_DEBUG_WATCHDOG", ""))
DEBUG_FONT = bool(os.environ.get("NVTEXPY_DEBUG_FONT", ""))
DEBUG_CS_RESOLVE = bool(os.environ.get("NVTEXPY_DEBUG_CS_RESOLVE", ""))

DEBUG_WRITE_TEXT = bool(os.environ.get("NVTEXPY_DEBUG_WRITE_TEXT", ""))
DEBUG_WRITE_COMPACT = bool(os.environ.get("NVTEXPY_DEBUG_WRITE_COMPACT", DEBUG_WRITE_TEXT))
DEBUG_WRITE_FLAT_MD = bool(os.environ.get("NVTEXPY_DEBUG_WRITE_FLAT_MD", DEBUG_WRITE_TEXT))
DEBUG_WRITE_FLAT_HTML = bool(os.environ.get("NVTEXPY_DEBUG_WRITE_FLAT_HTML", DEBUG_WRITE_TEXT))


class DebugHex(int):
    def __repr__(self) -> str:
        return f"0x{self:06X}"


def repl():
    """This starts a REPL with the callers globals and locals available"""
    import code
    import inspect
    import readline  # to allow arrow keys to work

    if TYPE_CHECKING:
        readline  # silence unused import warning

    frame = inspect.currentframe()
    if not frame:
        raise RuntimeError("No caller frame")

    code.interact(local=dict(frame.f_back.f_globals, **frame.f_back.f_locals))
