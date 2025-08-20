# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tools for working with nvtexpy."""

from pathlib import Path

from . import convert_single_document

ENGINE_PATH = Path(__file__).parent.parent / "engine"

__all__ = ["convert_single_document", "ENGINE_PATH"]
