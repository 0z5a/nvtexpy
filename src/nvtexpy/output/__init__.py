# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from nvtexpy.output.structured_output import StructuredOutput


class Output(ABC):
    @abstractmethod
    def generate(self, structured_output: StructuredOutput, output_file: str):
        raise NotImplementedError("Output.generate() is not implemented")
