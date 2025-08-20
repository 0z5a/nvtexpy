# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture(scope="session")
def pdflatex_binary() -> str:
    pdflatex_path = shutil.which("pdflatex")
    if pdflatex_path is None:
        pytest.skip("pdflatex not found in PATH")
        return ""
    return pdflatex_path


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    return Path(__file__).parent / "documents"


@pytest.fixture(scope="session")
def expected_dir() -> Path:
    return Path(__file__).parent / "expected"


@pytest.fixture(scope="session")
def results_dir() -> Path:
    return Path(__file__).parent / "results"


@pytest.fixture
def work_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
