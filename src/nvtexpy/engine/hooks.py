# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
from dataclasses import dataclass, field
from typing import Callable, Sequence, TypeAlias, TypeVar

from pdftex import get_cs, hook_cs, id_lookup

from nvtexpy.engine.common import CommandCode
from nvtexpy.engine.debug import DEBUG_CALL_ARGS, DEBUG_CALL_RECORDING, DEBUG_CALLS
from nvtexpy.engine.mem import MemReader

BeginHookCallback: TypeAlias = Callable[
    [str, CommandCode, None, int, int], Callable[[int, int], None] | None
]
BeginStrHookCallback: TypeAlias = Callable[
    [str, CommandCode, tuple[str | None, ...] | None, int, int], Callable[[int, int], None] | None
]
BeginRawHookCallback: TypeAlias = Callable[
    [str, CommandCode, tuple[int, ...] | None, int, int], Callable[[int, int], None] | None
]
EndHookCallback: TypeAlias = Callable[[str, CommandCode, None, int, int], None]
EndStrHookCallback: TypeAlias = Callable[
    [str, CommandCode, tuple[str | None, ...] | None, int, int], None
]


def _id_lookup(macro_name: str) -> int:
    return id_lookup(macro_name.encode(), False, True)


@dataclass
class IdHook:
    id: int
    str_args: bool = False
    begin_callbacks: list[BeginHookCallback | BeginStrHookCallback | BeginRawHookCallback] = field(
        default_factory=list
    )
    end_callbacks: list[EndHookCallback | EndStrHookCallback] = field(default_factory=list)


hooks_by_addr: dict[int, IdHook] = {}
global_hook: IdHook | None = None


@dataclass
class CallStackEntryBegin:
    args: tuple[int, int, tuple[int, ...] | None, int, int, int, int]

    id_hook: IdHook | None = None


@dataclass
class CallStackEntryEnd:
    cs: int
    head_addr: int
    tail_addr: int


@dataclass
class CallStackEntry:
    cs: int
    callbacks: list[Callable[[int, int], None]]


def _partial_cb(
    callback: EndStrHookCallback | EndHookCallback,
    cs_name: str,
    cmd: CommandCode,
    args: tuple[str | None, ...] | tuple[int | None, ...] | None,
) -> Callable[[int, int], None]:
    @functools.wraps(callback)
    def _wrapper(head_addr: int, tail_addr: int):
        return callback(cs_name, cmd, args, head_addr, tail_addr)

    return _wrapper


def _str_args(args: tuple[int, ...]) -> tuple[str | None, ...]:
    mem = MemReader()
    return tuple(mem.get_norefcount_token_list(arg).to_plain_str(True) for arg in args)


def _cs_name(cs: int) -> str:
    return get_cs(cs).tobytes().decode(errors="backslashreplace")


def _call_info(
    cmd: CommandCode,
    cs: int,
    cs_name: str | None,
    args: tuple[int, ...] | None,
    str_args: tuple[str | None, ...] | None,
):
    if cs_name is None:
        cs_name = _cs_name(cs)
    info = f"[{cmd.name}[0x{cs:06x}]] {cs_name}"
    if args is not None:
        if DEBUG_CALL_ARGS:
            mem = MemReader()
            info_args = ", ".join(
                "["
                + ", ".join(
                    f"{token} [{token.category!r}]"
                    for token in mem.get_norefcount_token_list(arg).tokens
                )
                + f"] from 0x{arg:06X}"
                for arg in args
            )
        else:
            if str_args is None:
                str_args = _str_args(args)
            assert str_args is not None
            info_args = ", ".join(repr(arg) for arg in str_args)
        info += f"({info_args})"
    return info


class _Hooks:
    """
    Class for registering hooks.
    """

    hooks_by_id: dict[int, IdHook]
    callstack: list[CallStackEntry]
    global_hook: IdHook | None = None
    enabled: bool = True

    recording_hook: Callable[[Sequence[CallStackEntryBegin | CallStackEntryEnd]], None] | None = (
        None
    )
    record_callstack: list[CallStackEntryBegin]
    record_callstack_depth: int = 0
    recorder: list[list[CallStackEntryBegin | CallStackEntryEnd] | None]

    _is_replaying: bool = False

    def __init__(self, capture_all: bool = False):
        assert not global_hook or capture_all, (
            "Cannot register global callback if not capturing all calls"
        )
        self.capture_all = capture_all
        self.hooks_by_id = {}
        self.callstack = []
        self.record_callstack = []
        self.recorder = []
        self.global_hook = global_hook
        for hook in hooks_by_addr.values():
            self.register(hook)
        # print("DEBUG HOOKS: ", self.hooks_by_id.values())

    def begin_peek(self):
        if len(self.recorder) == 0:
            self.record_callstack_depth = 0
        self.recorder.append(None)

    def end_peek(self, was_restored: bool) -> list[CallStackEntryBegin | CallStackEntryEnd] | None:
        recording = self.recorder.pop()
        if was_restored and recording is not None:
            self.replay(recording)
            return None
        return recording

    def replay(self, recording: list[CallStackEntryBegin | CallStackEntryEnd]):
        self._is_replaying = True
        try:
            for entry in recording:
                if isinstance(entry, CallStackEntryBegin):
                    if entry.id_hook is None:
                        r = self.cs_call(*entry.args)
                    else:
                        r = self._cs_call(entry.id_hook, True, *entry.args)
                    assert entry.args[2] is None or r is True
                else:
                    self.cs_call_end(entry.cs, entry.head_addr, entry.tail_addr)
        finally:
            self._is_replaying = False

    def record_in_command(
        self,
        recording_hook: Callable[[Sequence[CallStackEntryBegin | CallStackEntryEnd]], None],
    ):
        assert len(self.recorder) == 0
        self.recording_hook = recording_hook
        self.record_callstack_depth = 0
        self.recorder.append(None)
        if DEBUG_CALL_RECORDING:
            self.record_callstack.clear()
            print(debug_prefix() + "RECORD BEGIN HOOK")

    def _record_cmd(
        self,
        id_hook: IdHook | None,
        cmd: int,
        cs: int,
        args: tuple[int, ...] | None,
        token_list: int,
        mode: int,
        head_addr: int,
        tail_addr: int,
    ):
        if DEBUG_CALL_RECORDING:
            info = _call_info(CommandCode(cmd), cs, None, args, None)
            print(debug_prefix() + f"CS_CALL RECORD {info}")
        if self.recorder[-1] is None:
            self.recorder[-1] = []
        entry = CallStackEntryBegin(
            (
                cmd,
                cs,
                args,
                token_list,
                mode,
                head_addr,
                tail_addr,
            ),
            id_hook,
        )
        self.recorder[-1].append(entry)
        if args is not None:
            if DEBUG_CALL_RECORDING:
                self.record_callstack.append(entry)
            self.record_callstack_depth += 1

    def _record_cmd_end(self, cs: int, head_addr: int, tail_addr: int):
        self.record_callstack_depth -= 1
        if DEBUG_CALL_RECORDING:
            if len(self.record_callstack) > 0:
                endcmd = self.record_callstack.pop()
                info = _call_info(
                    CommandCode(endcmd.args[0]), endcmd.args[1], None, endcmd.args[2], None
                )
            else:
                endcmd = None
                info = "<callstack empty>"
            print(debug_prefix() + f"CS_CALL_END RECORD {info}")
        if self.recorder[-1] is None:
            self.recorder[-1] = []
        if self.record_callstack_depth == -1 and self.recording_hook is not None:
            if DEBUG_CALL_RECORDING:
                print(debug_prefix() + "RECORD END HOOK")
            rec_hook = self.recording_hook
            self.recording_hook = None
            self.record_callstack_depth = 0
            recording = self.recorder.pop() or []
            assert len(self.recorder) == 0
            rec_hook(recording)
            # And replay the end command of the recording command
            self.replay([CallStackEntryEnd(cs, head_addr, tail_addr)])
        else:
            self.recorder[-1].append(CallStackEntryEnd(cs, head_addr, tail_addr))

    def _cs_call(
        self,
        hook: IdHook | None,
        force_end: bool,
        cmd: int,
        cs: int,
        args: tuple[int, ...] | None,
        token_list_addr: int,
        mode: int,
        head_addr: int,
        tail_addr: int,
    ) -> bool:
        if len(self.recorder) > 0:
            self._record_cmd(hook, cmd, cs, args, token_list_addr, mode, head_addr, tail_addr)
            return True

        command_code = CommandCode(cmd)
        cs_name = _cs_name(cs)

        if args is not None and ((hook is not None and hook.str_args) or DEBUG_CALLS):
            str_args = _str_args(args)
        else:
            str_args = None

        end_hooks = []

        if DEBUG_CALLS:
            info = _call_info(command_code, cs, cs_name, args, str_args)
            print(debug_prefix() + "CS_CALL" + (" REPLAY" if self._is_replaying else "") + info)

            def log(head_addr: int, tail_addr: int):
                print(
                    debug_prefix()
                    + "CS_CALL_END"
                    + (" REPLAY" if self._is_replaying else "")
                    + info
                )

            end_hooks.append(log)

        if self.global_hook is not None and self.enabled:
            for callback in self.global_hook.begin_callbacks:
                if getattr(callback, "raw_args", False):
                    _args = args
                elif getattr(callback, "str_args", False):
                    _args = str_args
                else:
                    _args = None
                end_hook = callback(cs_name, command_code, _args, head_addr, tail_addr)
                if end_hook is not None:
                    end_hooks.append(end_hook)

        if hook is not None and self.enabled:
            for callback in hook.begin_callbacks:
                if getattr(callback, "raw_args", False):
                    _args = args
                elif getattr(callback, "str_args", False):
                    _args = str_args
                else:
                    _args = None
                end_hook = callback(cs_name, command_code, _args, head_addr, tail_addr)
                if end_hook is not None:
                    end_hooks.append(end_hook)
            if args is None:
                # Its an atomic call, which has no end call, end directly.
                for callback in hook.end_callbacks:
                    if getattr(callback, "raw_args", False):
                        _args = args
                    elif getattr(callback, "str_args", False):
                        _args = str_args
                    else:
                        _args = None
                    callback(cs_name, command_code, _args, head_addr, tail_addr)
            else:
                for callback in hook.end_callbacks:
                    if getattr(callback, "raw_args", False):
                        _args = args
                    elif getattr(callback, "str_args", False):
                        _args = str_args
                    else:
                        _args = None
                    end_hooks.append(_partial_cb(callback, cs_name, command_code, str_args))
        if args is None:
            # Its an atomic call, which has no end call, end directly.
            for callback in end_hooks:
                callback(head_addr, tail_addr)
        elif len(end_hooks) > 0 or force_end:
            self.callstack.append(CallStackEntry(cs, end_hooks))
            return True
        return False

    def register(self, id_hook: IdHook):
        if not self.capture_all:
            hook_cs(id_hook.id, functools.partial(self._cs_call, id_hook, False))
        self.hooks_by_id[id_hook.id] = id_hook

    def cs_call(
        self,
        cmd: int,
        cs: int,
        args: tuple[int, ...] | None,
        token_list_addr: int,
        mode: int,
        head_addr: int,
        tail_addr: int,
    ):
        """Called when a macro is called, only if debugging is enabled."""
        return self._cs_call(
            self.hooks_by_id.get(cs),
            True,
            cmd,
            cs,
            args,
            token_list_addr,
            mode,
            head_addr,
            tail_addr,
        )

    def cs_call_end(self, cs: int, head_addr: int, tail_addr: int):
        """Called when a macro call ends."""
        if len(self.recorder) == 0:
            entry = self.callstack.pop()
            # if entry.cs != cs:
            #     print(debug_prefix() + f"Expected cs=0x{entry.cs:06X}({_cs_name(entry.cs)!r}), got 0x{cs:06X}({_cs_name(cs)!r})")
            assert entry.cs == cs, (
                f"Expected cs=0x{entry.cs:06X}({_cs_name(entry.cs)!r}), got 0x{cs:06X}({_cs_name(cs)!r})"
            )
            for callback in entry.callbacks:
                callback(head_addr, tail_addr)
        else:
            self._record_cmd_end(cs, head_addr, tail_addr)


# The internal global instance of the active hook tracker.
_hooks: _Hooks | None = None


def init_hooks(capture_all: bool = False):
    """
    Initializes the hooks.

    Args:
        capture_all: Whether to use capture all hooks, or only registered hooks.
    """
    global _hooks
    assert _hooks is None, "Hooks already initialized, reinitialization not supported"
    _hooks = _Hooks(capture_all)


def debug_cs_call(
    cmd: int,
    cs: int,
    args: tuple[int, ...] | None,
    token_list_addr: int,
    mode: int,
    head_addr: int,
    tail_addr: int,
):
    """Called when a macro is called. Only available after `debug_hooks()` was called."""
    return _hooks.cs_call(cmd, cs, args, token_list_addr, mode, head_addr, tail_addr)


def cs_call_end(cs: int, head_addr: int, tail_addr: int):
    """Called when a macro call ends."""
    _hooks.cs_call_end(cs, head_addr, tail_addr)


_capture_next_input_level: list[Callable[[], None]] = []
_captured_input_levels: dict[int, list[Callable[[], None]]] = {}


def cs_push_input(level: int):
    """Called when an input level is pushed."""
    assert _hooks is not None
    if DEBUG_CALL_RECORDING and (
        len(_captured_input_levels) > 0 or len(_capture_next_input_level) > 0
    ):
        print(debug_prefix() + f"PUSH_INPUT({level})")
    _captured_input_levels[level] = list(_capture_next_input_level)
    _capture_next_input_level.clear()


def cs_pop_input(level: int):
    """Called when an input level is popped."""
    assert _hooks is not None
    if DEBUG_CALL_RECORDING and (
        len(_captured_input_levels) > 0 or len(_capture_next_input_level) > 0
    ):
        print(debug_prefix() + f"POP_INPUT({level})")

    if level in _captured_input_levels:
        for cb in _captured_input_levels[level]:
            cb()
        del _captured_input_levels[level]


_peek_level = 0


def cs_begin_peek():
    """Called when an input peek begins."""
    global _peek_level
    assert _hooks is not None
    assert _peek_level >= 0, f"Peek level is {_peek_level}"
    if DEBUG_CALL_RECORDING:
        print(debug_prefix() + f"BEGIN_PEEK({_peek_level})")
    _peek_level += 1
    _hooks.begin_peek()


def cs_end_peek(was_restored: bool):
    """Called when an input peek ends."""
    global _peek_level
    assert _hooks is not None
    assert _peek_level >= 1, f"Peek level is {_peek_level}"
    _peek_level -= 1
    if DEBUG_CALL_RECORDING:
        print(debug_prefix() + f"END_PEEK({_peek_level}, was_restored={was_restored})")
    recording = _hooks.end_peek(was_restored)
    if recording is not None:
        # Need to register the next input level and replay what was saved in the peek call stream
        capture_next_input_level(lambda: _hooks.replay(recording))


def capture_next_input_level(cb: Callable[[], None]):
    """Captures the next input level."""
    _capture_next_input_level.append(cb)


def debug_prefix() -> str:
    """Returns the space prefix for debug output."""
    return "" if _hooks is None else (len(_hooks.callstack) + _hooks.record_callstack_depth) * "  "


TCall = TypeVar("TCall", bound=Callable)


def _set_args(fn: TCall, str_args: bool, raw_args: bool) -> TCall:
    if raw_args:
        try:
            setattr(fn, "raw_args", True)
        except AttributeError:

            @functools.wraps(fn)
            def _wrapper(*args):
                return fn(*args)

            setattr(_wrapper, "raw_args", True)
            return _wrapper
    elif str_args:
        try:
            setattr(fn, "str_args", True)
        except AttributeError:

            @functools.wraps(fn)
            def _wrapper(*args):
                return fn(*args)

            setattr(_wrapper, "str_args", True)
            return _wrapper
    return fn


def register_macro_call(
    macro_name: str,
    begin_callback: BeginHookCallback | BeginStrHookCallback | BeginRawHookCallback | None,
    end_callback: EndHookCallback | EndStrHookCallback | None,
    str_args: bool = False,
    raw_args: bool = False,
):
    """
    Registers a macro call for hooking.

    Args:
        macro_name: The macro call to register
        begin_callback: The callback to call when the macro call begins
        end_callback: The callback to call when the macro call ends
        str_args: If true, the arguments are stringified args.
        raw_args: If true, the arguments are raw integers.
    """
    id = _id_lookup(macro_name)
    id_hook = hooks_by_addr.get(id)
    if id_hook is None:
        hooks_by_addr[id] = id_hook = IdHook(id=id)
        if _hooks is not None:
            _hooks.register(id_hook)
    id_hook.str_args = id_hook.str_args or str_args
    if begin_callback is not None:
        id_hook.begin_callbacks.append(_set_args(begin_callback, str_args, raw_args))
    if end_callback is not None:
        assert not raw_args, "Cannot have raw args for end callback"
        id_hook.end_callbacks.append(_set_args(end_callback, str_args, raw_args))


def register_global_macro_call(
    begin_callback: BeginHookCallback | BeginStrHookCallback | BeginRawHookCallback | None,
    end_callback: EndHookCallback | EndStrHookCallback | None,
    str_args: bool = False,
    raw_args: bool = False,
):
    """
    Registers a macro call for hooking.

    Args:
        macro_name: The macro call to register
        begin_callback: The callback to call when the macro call begins
        end_callback: The callback to call when the macro call ends
        str_args: If true, the arguments are stringified args.
        raw_args: If true, the arguments are raw integers.
    """
    global global_hook
    if global_hook is None:
        global_hook = IdHook(id=-1)
        if _hooks is not None:
            assert _hooks.capture_all, "Cannot register global callback if not capturing all calls"
            _hooks.global_hook = global_hook
    global_hook.str_args = global_hook.str_args or str_args
    if begin_callback is not None:
        global_hook.begin_callbacks.append(_set_args(begin_callback, str_args, raw_args))
    if end_callback is not None:
        assert not raw_args, "Cannot have raw args for end callback"
        global_hook.end_callbacks.append(_set_args(end_callback, str_args, raw_args))


def record_in_command(
    recording_hook: Callable[[Sequence[CallStackEntryBegin | CallStackEntryEnd]], None],
):
    assert _hooks is not None
    _hooks.record_in_command(recording_hook)


def enable_callbacks(enable: bool):
    assert _hooks is not None
    _hooks.enabled = enable
