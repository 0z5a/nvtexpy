# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import pickle
import shutil
import subprocess
from pathlib import Path
import traceback

from nvtexpy.output import nested_bbox
import pytest

from nvtexpy.tools.convert_single_document import (
    CompileError,
    FlatLayoutConverter,
    process_single_document,
)


def save_page_as_text(page: dict, work_dir: Path, results_output_dir: Path, converter_name: str):
    if page[f"{converter_name}_file"] is None:
        return
    print("Processing " + page[f"{converter_name}_file"])
    base_name = page["image_file"].rsplit(".", 1)[0]
    with open(work_dir / page[f"{converter_name}_file"], "r") as f:
        page_objects = json.load(f)
    page_texts = "\n\n".join(
        f'<box class="{obj["category_id"]}" bbox="{"_".join(str(b) for b in obj["bbox"])}">\n'
        + obj["content"]
        + "\n</box>"
        for obj in page_objects["ann"]
    )
    print(f"Saving {results_output_dir}/{base_name}.{converter_name}.txt")
    with open(results_output_dir / f"{base_name}.{converter_name}.txt", "w") as f:
        f.write(page_texts)


@pytest.mark.parametrize("document", ["document1", "document2"])
def test_regression(
    document: str,
    test_data_dir: Path,
    expected_dir: Path,
    results_dir: Path,
    pdflatex_binary: str,
    tmp_path: Path,
):
    """
    Test that pdflatex with nvtexpy produces expected outputs for regression test cases.
    """
    test_doc_dir = test_data_dir / document
    expected_output_dir = expected_dir / document
    results_output_dir = results_dir / document
    tex_file = test_doc_dir / f"{document}.tex"
    compile_dir = tmp_path / "compile"
    compile_dir.mkdir(parents=True)
    shutil.rmtree(results_output_dir, ignore_errors=True)
    results_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = process_single_document(
            tex_file_path=tex_file,
            work_dir=compile_dir,
            dpi=300,
            final_generators=[
                FlatLayoutConverter(
                    name="flat_layout_latex_markdown",
                    table_format="latex",
                    style_format="markdown",
                ),
                FlatLayoutConverter(
                    name="flat_layout_html",
                    table_format="html",
                    style_format="html",
                ),
            ],
        )
    except CompileError as e:
        pytest.fail(
            f"\nRegression test '{document}' failed - compilation failed:\n\n"
            f"{traceback.format_exc()}\n\n"
            f"Console log:\n{e.console_log}\n\n"
            f"TeX log:\n{e.tex_log}\n\n"
        )
        raise

    print(results)

    # Convert pages to text
    for page in results["pages"]:
        if page["image_file"] is None:
            continue
        save_page_as_text(page, compile_dir, results_output_dir, "flat_layout_latex_markdown")
        save_page_as_text(page, compile_dir, results_output_dir, "flat_layout_html")

    # Convert pkl to compact structure for regression diff
    for page in results["pages"]:
        if page.get("pickle_file"):
            pkl_path = compile_dir / page["pickle_file"]
            compact_txt_path = results_output_dir / (pkl_path.stem + ".compact.txt")
            # Assume convert_from_pkl has an argument for "compact" output
            # If not, this could instead call a function that runs the conversion to the desired structure
            with open(pkl_path, "rb") as f:
                raw_pkl = pickle.load(f)
            nested_bbox.compact_in_place(raw_pkl["root"])
            with open(compact_txt_path, "w") as f:
                f.write(raw_pkl["root"].to_str_compact())

    diff_result = subprocess.run(
        ["diff", "-r", "-u", str(expected_output_dir), str(results_output_dir)],
        capture_output=True,
        text=True,
    )

    if diff_result.returncode != 0:
        pytest.fail(
            f"\nRegression test '{document}' failed - outputs differ from expected:\n\n"
            f"{diff_result.stdout}\n"
        )
