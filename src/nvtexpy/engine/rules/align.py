# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from nvtexpy.engine.common import CommandCode
from nvtexpy.engine.rules.base import Rule, Rules, assert2pagefail, begin_macro_hook
from nvtexpy.engine.stack_formatter import CallbackFormatter
from nvtexpy.output.structured_output import AlignBlockElement, AlignBlockEqNumElement


class AlignFormatter(CallbackFormatter):
    starred: bool
    env_name: str

    def __init__(self, starred: bool, env_name: str) -> None:
        self.starred = starred
        self.env_name = env_name
        super().__init__(
            begin=lambda struct_out: struct_out.push(AlignBlockElement(starred, env_name)),
            name="align",
        )


class AlignEqNumFormatter(CallbackFormatter):
    def __init__(self) -> None:
        super().__init__(
            begin=lambda struct_out: struct_out.push(AlignBlockEqNumElement()),
            name="align_eq_num",
        )


@Rules.register
class AlignRule(Rule):
    """For \\begin{align} and similar functions."""

    @begin_macro_hook(["@eqnnum", "print@eqnum"])
    @assert2pagefail
    def on_eq_num(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        if not self.active:
            return None
        fmt = AlignEqNumFormatter()
        self.push_formatter(fmt)

        print("on_eq_num")

        def end(*_):
            self.pop_formatter(fmt)
            print("on_eq_num end")

        return end

    @begin_macro_hook(["eqnarray", "eqnarray*"])
    @assert2pagefail
    def on_eqnarray_start(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        if self.active:
            # Don't nest
            return None

        self.active = True

        formatter = AlignFormatter(starred=cs_name.endswith("*"), env_name="eqnarray")
        self.push_formatter(formatter)

        def end_env(*args):
            self.pop_formatter(formatter)
            self.active = False
            self.state.input_formatter_store_next = True

        # Ends when the currently active save group ends
        self.state.save_levels[-1].end_hooks.append(end_env)

    @begin_macro_hook("start@gather", str_args=True)
    @assert2pagefail
    def on_gather_start(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: Tuple[str | None, ...] | None,
        head_addr: int,
        tail_addr: int,
    ):
        if self.active:
            # Don't nest
            return None

        self.active = True
        formatter = AlignFormatter(
            starred=(args[0] == "\\st@rredtrue" if args else False), env_name="gather"
        )
        self.push_formatter(formatter)

        def end_env(*args):
            self.pop_formatter(formatter)
            self.active = False
            self.state.input_formatter_store_next = True

        # Ends when the currently active save group ends
        self.state.save_levels[-1].end_hooks.append(end_env)

    @begin_macro_hook("start@multline", str_args=True)
    @assert2pagefail
    def on_multline_start(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: Tuple[str | None, ...] | None,
        head_addr: int,
        tail_addr: int,
    ):
        if self.active:
            # Don't nest
            return None

        self.active = True

        formatter = AlignFormatter(
            starred=(args[0] == "\\st@rredtrue" if args else False), env_name="multline"
        )
        self.push_formatter(formatter)

        def end_env(*args):
            self.pop_formatter(formatter)
            self.active = False
            self.state.input_formatter_store_next = True

        # Ends when the currently active save group ends
        self.state.save_levels[-1].end_hooks.append(end_env)

    @begin_macro_hook("start@align", str_args=True)
    @assert2pagefail
    def on_align_start(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: Tuple[str | None, ...] | None,
        head_addr: int,
        tail_addr: int,
    ):
        if self.active:
            # Don't nest
            return None
        assert args is not None

        # print("start@align", args)

        if args[0] == "\\z@":
            env_name = "alignat"
        elif args[0] == "\\@ne":
            if args[2] == "\\m@ne":
                env_name = "align"
            else:
                env_name = "xalignat"
        elif args[0] == "\\tw@":
            if args[2] == "\\m@ne":
                env_name = "flalign"
            else:
                env_name = "xxalignat"
        else:
            assert False, f"Unknown align env: {args}"

        self.active = True

        formatter = AlignFormatter(starred=args[1] == "\\st@rredtrue", env_name=env_name)
        self.push_formatter(formatter)

        def end_env(*args):
            self.pop_formatter(formatter)
            self.active = False
            self.state.input_formatter_store_next = True
            # print("start@align end")

        # Ends when the currently active save group ends
        self.state.save_levels[-1].end_hooks.append(end_env)

    @begin_macro_hook("equation")
    @assert2pagefail
    def on_equation_start(
        self,
        cs_name: str,
        cmd: CommandCode,
        args: None,
        head_addr: int,
        tail_addr: int,
    ):
        if self.active:
            # Don't nest
            return None

        self.active = True

        # There is no starred version of equation without
        formatter = AlignFormatter(starred=False, env_name="equation")
        self.push_formatter(formatter)

        def end_env(*args):
            self.pop_formatter(formatter)
            self.active = False
            self.state.input_formatter_store_next = True
            # print("start@align end")

        # Ends when the currently active save group ends
        self.state.save_levels[-1].end_hooks.append(end_env)
