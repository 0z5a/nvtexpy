# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pickle import HIGHEST_PROTOCOL, dumps

from nvtexpy.output import Output
from nvtexpy.output.structured_output import RootElement, StructuredOutput


def save_pickle(
    structured_output: StructuredOutput,
    override_filename: str | None = None,
    override_root: RootElement | None = None,
):
    with open(override_filename or f"{structured_output.page_name}.pkl", "wb") as f:
        f.write(
            dumps(
                {
                    "root": override_root or structured_output._root,
                    "coord_scale_factor": structured_output.coord_scale_factor,
                    "page_width_px": structured_output.page_width_px,
                    "page_height_px": structured_output.page_height_px,
                },
                HIGHEST_PROTOCOL,
            )
        )


class Pickle(Output):
    def generate(self, structured_output: StructuredOutput):
        save_pickle(structured_output)
