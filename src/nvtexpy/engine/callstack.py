# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

from nvtexpy.engine.common import CommandCode


@dataclass
class CallStackItem:
    macro_name: str
    macro_args: Tuple[str | None, ...] | None
    _end_hooks: List[Callable[[], None]] = field(default_factory=list, repr=False)


class CallStack:
    """Keeps track of macro calls that we are interested in.
    It only tracks the calls of macros that we hook specifically."""

    stack: List[CallStackItem]

    def __init__(self):
        self.stack = []

    def _hook_begin(
        self,
        name: str,
        cmd: CommandCode,
        args: Tuple[str | None, ...] | None,
        head: int,
        tail: int,
    ):
        assert args is not None, "Cannot hook primitive"

        self.stack.append(
            CallStackItem(
                macro_name=name,
                macro_args=args,
            )
        )

    def _hook_end(
        self,
        name: str,
        cmd: CommandCode,
        args: Tuple[str | None, ...] | None,
        head: int,
        tail: int,
    ):
        self.stack.pop()

    def __getitem__(self, item: int):
        return self.stack[item]

    def __len__(self):
        return len(self.stack)

    def __contains__(self, item: str):
        # print("CONTAINS:", item, self.to_names())
        return any(item == item.macro_name for item in self.stack)

    def __repr__(self) -> str:
        return self.stack.__repr__()

    def __str__(self) -> str:
        return self.stack.__str__()

    def to_names(self):
        return [item.macro_name for item in self.stack]
