# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""
Single Document LaTeX to Annotation Converter

This script processes a single LaTeX document and generates machine learning
OCR annotations in multiple formats including flat layout, HOCR, and markdown.

This tool is part of the nvtexpy package and uses the NVTexPyEngine for processing.
"""

from enum import IntFlag
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
import traceback
from typing import Literal, Optional, Sequence

import click

from nvtexpy.tools.helpers import find_generated_pages, FlatLayoutConverter, StructuredConverter

TEX_PREAMBLE_ID = b"% NV arXiv LaTeX Preamble v1"
TEX_PREAMBLE_CODE = rb"""% NV arXiv LaTeX Preamble v1
% NVPDFTEX: The following code is used to always use T1 encoding and support .EPS images
\makeatletter%
\newcommand{\adlrbegin}{%
  \usepackage[T1]{fontenc}%
  \RequirePackage{epsfig}%
  \RequirePackage{epstopdf}%
  \g@addto@macro\Gin@extensions{,.EPS,.ps}%
  \epstopdfDeclareGraphicsRule{.EPS}{pdf}{.pdf}{%
    \ETE@epstopdf{##1}%
  }%
  \epstopdfDeclareGraphicsRule{.ps}{pdf}{.pdf}{%
    \ETE@epstopdf{##1}%
  }%
  \nvtexpyhook%
}%
\makeatother%
\AtBeginDocument{\adlrbegin}%
% NVPDFTEX: End of custom preamble, original tex code follows
"""


class PdfTexFlags(IntFlag):
    ORIGINAL = 0
    DEBUGPY = 1
    NVTEXPY = 2
    NO_INPUT_STACK_OPTIMIZATION = 16


def run_command(
    command: list[str],
    workdir: Path,
    log_file: Optional[Path] = None,
    timeout: int = 300,
) -> None:
    """
    Run a command and capture output.

    Args:
        command: Command to run as list of strings
        workdir: Working directory for the command
        log_file: Optional log file to write output to
        timeout: Command timeout in seconds
    """
    if log_file:
        with open(log_file, "wb") as f:
            f.write(b"Command: " + b" ".join(c.encode("utf-8") for c in command) + b"\n")
            f.flush()
            subprocess.run(
                command,
                cwd=workdir,
                stdout=f,
                stderr=f,
                text=True,
                timeout=timeout,
                check=True,
            )
    else:
        subprocess.run(
            command,
            cwd=workdir,
            text=True,
            timeout=timeout,
            check=True,
        )


def add_tex_preamble(tex_file_path: Path) -> None:
    """
    Add standard preamble to LaTeX file if not already present.

    Args:
        tex_file_path: Path to the .tex file to modify
    """
    file_bytes = tex_file_path.read_bytes()
    if file_bytes.startswith(TEX_PREAMBLE_ID):
        print(f"Skipping {tex_file_path} - preamble already present")
        return

    with tex_file_path.open("wb") as texwf:
        texwf.write(TEX_PREAMBLE_CODE)
        texwf.write(file_bytes)
    print(f"Added preamble to {tex_file_path}")


def convert_pdf_to_images(
    pdf_file: Path,
    work_dir: Path,
    log_file: Optional[Path] = None,
    dpi: int = 300,
) -> None:
    """
    Convert PDF pages to PNG images using Ghostscript.

    Args:
        pdf_file: Path to the PDF file
        work_dir: Working directory for output images
        log_file: Optional log file to write output to
        dpi: Resolution for the output images
    """
    command = [
        "gs",
        "-dNOPAUSE",
        "-dBATCH",
        "-sDEVICE=png16m",
        "-dTextAlphaBits=4",
        "-dGraphicsAlphaBits=4",
        "-dMaxBitmap=500000000",
        "-dAlignToPixels=0",
        "-dGridFitTT=2",
        "-r" + str(dpi),
        "-o",
        str(work_dir / "nvpdftex_page_%d.png"),
        str(pdf_file),
    ]

    print(f"Converting PDF to PNG images at {dpi} DPI...")

    run_command(command, work_dir, log_file)

    # Check if images were generated
    image_files = list(work_dir.glob("nvpdftex_page_*.png"))
    assert image_files, "No PNG images were generated"

    print(f"Successfully generated {len(image_files)} PNG images")


def process_latex_with_nvtexpy(
    tex_file_path: Path,
    work_dir: Path,
    nvtexpy_flags: PdfTexFlags,
    log_file: Optional[Path] = None,
) -> None:
    """
    Process LaTeX file using the custom pdflatex with nvtexpy hooks.

    Args:
        tex_file_path: Path to the input .tex file
        work_dir: Working directory for processing
        nvtexpy_flags: Flags to pass to nvtexpy (-nvtexpy option)
        log_file: Optional log file to write output to
    """
    # Use the custom pdflatex binary if available
    pdflatex_binary = "/usr/bin/pdflatex"
    if not os.path.exists(pdflatex_binary):
        pdflatex_binary = "pdflatex"

    assert tex_file_path.resolve().parent == work_dir.resolve(), (
        "tex_file_path must be within work_dir"
    )

    # Run pdflatex with nvtexpy hooks
    command = [
        pdflatex_binary,
        "-nvtexpy",
        str(nvtexpy_flags.value),
        "-output-format",
        "pdf",
        "-interaction=batchmode",
        "-halt-on-error",
        tex_file_path.name,
    ]

    print(f"Running pdflatex with nvtexpy flags {nvtexpy_flags}...")

    run_command(command, work_dir, log_file)

    # Check if PDF was generated
    pdf_file = work_dir / f"{tex_file_path.stem}.pdf"
    assert pdf_file.exists(), "PDF file was not generated"

    print(f"Successfully generated PDF: {pdf_file}")


class CompileError(Exception):
    def __init__(self, message: str, console_log: str | None, tex_log: str | None):
        self.message = message
        self.console_log = console_log
        self.tex_log = tex_log


def process_single_document(
    tex_file_path: Path,
    work_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    num_pdftex_calls: int = 4,
    dpi: int = 300,
    final_generators: Sequence[StructuredConverter] = (),
) -> dict:
    """
    Process a single LaTeX document and generate annotations.

    Args:
        tex_file_path: Path to the input .tex file
        work_dir: Working directory for temporary files (optional)
        output_dir: Output directory for results (optional)
        dpi: Resolution for PNG images
        generate_flat_layout: Whether to generate flat layout annotations
        flat_layout_format: Flat layout table format
        flat_layout_style: Flat layout style format
    """
    start_time = time.perf_counter()

    results = {
        "document_path": str(tex_file_path),
        "work_directory": str(work_dir),
        "output_directory": str(output_dir) if output_dir else None,
        "dpi": dpi,
        "times": {},
    }

    tex_file_in_work_dir = None

    try:
        # Setup working directory
        if work_dir is None:
            work_dir = tex_file_path.parent
        else:
            work_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(tex_file_path.parent, work_dir, dirs_exist_ok=True)

        # Add preamble to tex file
        tex_file_in_work_dir = work_dir / tex_file_path.name
        add_tex_preamble(tex_file_in_work_dir)

        results["times"]["compile"] = []
        for i_pdftex_call in range(num_pdftex_calls):
            # Process LaTeX with nvtexpy
            compile_start = time.perf_counter()
            log_file = work_dir / f"nvtexpy_processing_{i_pdftex_call}.log"
            try:
                process_latex_with_nvtexpy(
                    tex_file_in_work_dir,
                    work_dir,
                    nvtexpy_flags=PdfTexFlags.ORIGINAL
                    if i_pdftex_call < num_pdftex_calls - 1
                    else PdfTexFlags.NVTEXPY,
                    log_file=log_file,
                )
            except subprocess.CalledProcessError:
                try:
                    results["console_log"] = log_file.read_text()
                except FileNotFoundError:
                    pass
                raise
            finally:
                results["times"]["compile"].append(time.perf_counter() - compile_start)

        pdf_file = work_dir / f"{tex_file_in_work_dir.stem}.pdf"
        assert pdf_file.exists(), "PDF file was not generated"
        results["pdf_file"] = pdf_file.name

        # Convert PDF to images
        images_start = time.perf_counter()
        log_file = work_dir / "ghostscript.log"
        try:
            convert_pdf_to_images(pdf_file, work_dir, log_file=log_file, dpi=dpi)
        except subprocess.CalledProcessError:
            try:
                results["console_log"] = log_file.read_text()
            except FileNotFoundError:
                pass
            raise
        finally:
            results["times"]["images"] = time.perf_counter() - images_start

        # Find generated files
        results["pages"] = find_generated_pages(work_dir)

        # Generate flat layout annotations if requested
        if final_generators:
            print("Generating final annotations...")
            for generator in final_generators:
                results["times"][generator.name] = []
                for page in results["pages"]:
                    if page.get("pickle_file") is None:
                        continue
                    flat_layout_start = time.perf_counter()
                    try:
                        result = generator(work_dir / page["pickle_file"])
                        converted_file = (work_dir / page["pickle_file"]).with_suffix(
                            f".{generator.name}.json"
                        )
                        page[f"{generator.name}_file"] = converted_file.name
                        # Save flat layout output
                        with open(converted_file, "w") as f:
                            f.write(result)
                        print(f"Generated flat layout annotations for page {page['page_num']}")
                    finally:
                        results["times"][generator.name].append(
                            time.perf_counter() - flat_layout_start
                        )

        # Copy results to output directory if specified
        if output_dir and output_dir != work_dir:
            copy_start = time.perf_counter()
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(pdf_file, output_dir / pdf_file.name)
                for page in results["pages"]:
                    for generator in final_generators:
                        generator_file = page.get(f"{generator.name}_file")
                        if generator_file is not None:
                            generator_file = work_dir / generator_file
                            shutil.copy2(generator_file, output_dir / generator_file.name)
                            print(f"Copied {generator_file.name} to output directory")
            finally:
                results["times"]["copy"] = time.perf_counter() - copy_start
        results["times"]["all"] = time.perf_counter() - start_time
        results["status"] = "success"
    except Exception as e:
        results["times"]["all"] = time.perf_counter() - start_time
        results["status"] = "failure"
        results["error"] = str(e)
        results["traceback"] = traceback.format_tb(e.__traceback__)

        # Save results summary
        if output_dir:
            results_file = output_dir / "processing_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {results_file}")
        if isinstance(e, subprocess.CalledProcessError):
            if tex_file_in_work_dir is not None:
                tex_log = tex_file_in_work_dir.with_name(f"{tex_file_in_work_dir.name[:-4]}.log")
                if tex_log.exists():
                    results["tex_log"] = tex_log.read_text()
            raise CompileError(
                "Compilation failed", results.get("console_log"), results.get("tex_log")
            )
        raise

    # Save results summary
    if output_dir:
        results_file = output_dir / "processing_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_file}")

    return results


@click.command()
@click.argument("tex_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument(
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
)
@click.option(
    "--work-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Working directory for temporary files (defaults to tex file directory)",
)
@click.option(
    "--dpi",
    type=int,
    default=300,
    help="Resolution for PNG images",
)
@click.option(
    "--flat-layout/--no-flat-layout",
    is_flag=True,
    default=True,
    help="Generate flat layout annotation",
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
@click.option(
    "--keep-work-dir",
    is_flag=True,
    help="Keep working directory after processing (for debugging)",
)
def main(
    tex_file: Path,
    output_dir: Optional[Path],
    work_dir: Optional[Path],
    dpi: int,
    flat_layout: bool,
    flat_layout_format: Literal["latex", "html"],
    flat_layout_style: Literal["markdown", "html"],
    keep_work_dir: bool,
):
    """
    \b
    Convert a single LaTeX document TEX_FILE to machine learning annotations using nvtexpy.
    OUTPUT_DIR is the output directory for the generated files.

    \b
    This tool processes a LaTeX file and generates:
    - PDF output using custom pdflatex with nvtexpy hooks
    - PNG page images
    - Pickle files with structured annotations
    - Flat layout format annotations if requested.

    \b
    Example:
        python convert_single_document.py document.tex ./output
    """
    print(f"Processing LaTeX document: {tex_file}")
    print(f"Output directory: {output_dir or 'same as input'}")
    print(f"DPI: {dpi}")
    print(f"Generate flat layout: {flat_layout}")
    if flat_layout:
        print(f"Flat layout format: {flat_layout_format}")
        print(f"Flat layout style: {flat_layout_style}")
    print("-" * 50)

    temp_work_dir = None
    try:
        # Create temporary work directory if needed
        if work_dir is None:
            temp_work_dir = Path(tempfile.mkdtemp())
            work_dir = temp_work_dir
            print(f"Created temporary work directory: {work_dir}")

        # Process the document
        process_single_document(
            tex_file_path=tex_file,
            work_dir=work_dir,
            output_dir=output_dir,
            dpi=dpi,
            final_generators=[
                FlatLayoutConverter(
                    name="flat_layout",
                    table_format=flat_layout_format,
                    style_format=flat_layout_style,
                ),
            ]
            if flat_layout
            else (),
        )

        # Cleanup
        if temp_work_dir and not keep_work_dir:
            shutil.rmtree(temp_work_dir)
            print("Cleaned up temporary work directory")
        elif keep_work_dir:
            print(f"Work directory preserved: {work_dir}")

    except Exception as e:
        traceback.print_exc()

        print(f"\nERROR: Processing failed: {e}")

        # Cleanup on error
        if temp_work_dir and not keep_work_dir:
            shutil.rmtree(temp_work_dir)
            print("Cleaned up temporary work directory")

        raise click.ClickException(f"Processing failed: {e}")


if __name__ == "__main__":
    main()
