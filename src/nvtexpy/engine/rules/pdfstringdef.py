# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nvtexpy.engine.hooks import record_in_command
from nvtexpy.engine.rules.base import Rule, Rules, assert2pagefail, begin_macro_hook


@Rules.register
class PdfStringDefRule(Rule):
    """For \\pdfstringdef."""

    @begin_macro_hook("pdfstringdef")
    @assert2pagefail
    def on_tableofcontents_start(self, *args):
        def _recording_callback(cmds):
            print("Discarding recorded commands within pdfstringdef")

        print("pdfstringdef START")

        record_in_command(_recording_callback)

        def _end(*_):
            print("pdfstringdef END")

        return _end
