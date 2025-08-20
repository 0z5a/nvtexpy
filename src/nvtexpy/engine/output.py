# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class MarkdownOutput:
    """Manages the markdown output file state"""

    def __init__(self, filename: str = "output2.md"):
        self.file = open(filename, "w")

    def write(self, s: str):
        if self.file:
            self.file.write(s)

    def close(self):
        if self.file:
            self.file.close()
            self.file = None

    def __del__(self):
        self.close()
