# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""
Structured Pickle to Annotation Converter

This script processes structured pickle file(s) and generates machine learning
OCR annotations.

This tool is part of the nvtexpy package.
"""

from pathlib import Path
from typing import Literal

import click

from nvtexpy.tools.helpers import FlatLayoutConverter, find_generated_pages


@click.group()
def cli():
    """
    Convert structured pickle files to annotations.
    """


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--flat-layout-suffix",
    type=str,
    default="flat_layout",
    help="Flat layout file suffix",
)
@click.option(
    "--flat-layout-format",
    type=click.Choice(["latex", "html"], case_sensitive=False),
    default="latex",
    help="Flat layout table format",
)
@click.option(
    "--flat-layout-style",
    type=click.Choice(["markdown", "html"], case_sensitive=False),
    default="markdown",
    help="Flat layout style format",
)
def document(
    output_dir: Path,
    flat_layout_suffix: str,
    flat_layout_format: Literal["latex", "html"],
    flat_layout_style: Literal["markdown", "html"],
):
    """
    Convert a single document, which has already been converted to structured pickle format to machine learning annotations.
    OUTPUT_DIR is the directory containing the pickle files and will be used to store the generated annotations.
    """
    print(f"Processing document: {output_dir}")
    print(f"Flat layout format: {flat_layout_format}")
    print(f"Flat layout style: {flat_layout_style}")
    print("-" * 50)

    # Process the document
    # Find generated files
    pages = find_generated_pages(output_dir)

    converter = FlatLayoutConverter(
        name=flat_layout_suffix,
        table_format=flat_layout_format,
        style_format=flat_layout_style,
    )

    # Generate flat layout annotations if requested
    for page in pages:
        if page.get("pickle_file") is None:
            continue
        result = converter(output_dir / page["pickle_file"])
        converted_file = (output_dir / page["pickle_file"]).with_suffix(f".{converter.name}.json")
        page[f"{converter.name}_file"] = converted_file.name
        # Save flat layout output
        with open(converted_file, "w") as f:
            f.write(result)
        print(f"Generated flat layout annotations for page {page['page_num']}")
    print(f"Generated flat layout annotations for document {output_dir.name}")


@cli.command()
@click.argument("page_pickle_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_file", type=click.Path(file_okay=True, path_type=Path), default=None)
@click.option(
    "--flat-layout-suffix",
    type=str,
    default="flat_layout",
    help="Flat layout file suffix",
)
@click.option(
    "--flat-layout-format",
    type=click.Choice(["latex", "html"], case_sensitive=False),
    default="latex",
    help="Flat layout table format",
)
@click.option(
    "--flat-layout-style",
    type=click.Choice(["markdown", "html"], case_sensitive=False),
    default="markdown",
    help="Flat layout style format",
)
def page(
    page_pickle_file: Path,
    output_file: Path | None,
    flat_layout_suffix: str,
    flat_layout_format: Literal["latex", "html"],
    flat_layout_style: Literal["markdown", "html"],
):
    """
    Convert a single page, which has already been converted to structured pickle format to machine learning annotations.
    PAGE_PICKLE_FILE is the pickle file for a single page.
    OUTPUT_FILE is the file to write the annotations to. If not provided, the annotations will be written to a file with the same name as the pickle file but with the {flat_layout_suffix}.json extension.
    """
    print(f"Processing page: {page_pickle_file}")
    print(f"Flat layout format: {flat_layout_format}")
    print(f"Flat layout style: {flat_layout_style}")
    print("-" * 50)

    # Process the document
    res = FlatLayoutConverter(
        name=flat_layout_suffix,
        table_format=flat_layout_format,
        style_format=flat_layout_style,
    )(page_pickle_file)
    with open(output_file or page_pickle_file.with_suffix(f".{flat_layout_suffix}.json"), "w") as f:
        f.write(res)
    print(f"Generated flat layout annotations for page {page_pickle_file.name}")


if __name__ == "__main__":
    cli()
