from typing import Callable, Literal, Tuple, Union, overload

import numpy as np

# This function returns the current input_stack array.
# It also returns the current line_stack array that can be used to find the line
# number of each level in the input stack (if it's a file input level not token input)
# NOTE: This does not include the current input which is not on the stack! Use get_cur_input() for that.
#
# Returns:
#   - A tuple (input_stack, line_stack).
#
# The shape of both arrays is one-dimensional.
#   - The input stack is an array of structs with the following fields:
#       * state: If state == 0, then it's a token input level (more details in pdftex.web)
#       * index: Not used by us.
#       * start: When reading a file, this is the byte offset of loc within the line buffer.
#       * loc: Current reading position. Either byte offset in line (if state > 0) or token pointer in token list.
#       * limit: The end of the line buffer. limit - start == line length
#       * name: The name of the input file.
#       * tag: Used by synctex.
#   - The line stack is an uint32 array of line numbers.
#       * line: The line number.
#
# For more information on the fields, search pdftex.web for "@!in_state_record".
def get_input_stack() -> Tuple[np.ndarray, np.ndarray]: ...

# This function returns the current input struct.
# It also returns the current line number.
#
# Returns:
#   - A tuple (input, line).
#
# The input ndarray will have only one element and can be concatenated with the result
# from get_input_stack(), if desired.
# The line number is an integer, it's the current line number of the current input
def get_cur_input() -> Tuple[np.ndarray, int]: ...

# This function returns the primitives list.
# It's a list of strings that contains the names of all primitives.
def get_primitives() -> np.ndarray: ...

# This function returns the string with the given index in the string pool str_pool.
# It returns a memoryview object that can be converted to a string using str().
# It can also be compared to a bytes object without copying it.
def get_str(index: int, /) -> memoryview: ...

# This function returns the control sequence with the given index in the control sequence name pool.
def get_cs(index: int, /) -> memoryview: ...

# Gets the control sequence value from the given control sequence index. Returns (cmd, value) tuple.
def get_cs_value(index: int, /) -> Tuple[int, Union[int, str]]: ...

# This function returns the index of the control sequence with the given name in the control sequence name pool.
# If get_value is True, it also returns the value of the control sequence.
#
# Returns:
#   - A str or int value if get_value is True.
#   - The index if get_value is False.
#
# NOTE: Only a few control sequence types are supported for get_value=True.
# If the control sequence is not supported, a NotImplementedError is raised.
@overload
def id_lookup(
    name: bytes, get_value: Literal[False], new_control_sequence: bool = False, /
) -> int: ...
@overload
def id_lookup(
    name: bytes, get_value: Literal[True], new_control_sequence: bool = False, /
) -> Tuple[int, Union[int, str]]: ...
def id_lookup(
    name: bytes, get_value: bool = False, new_control_sequence: bool = False, /
) -> Union[int, Tuple[int, Union[int, str]]]: ...

# This function returns the cur_group and cur_level globals.
# This is like the top of save_stack (search pdftex.web for "@!save_stack" and "@!cur_group")
#
# cur_group is the current group number. and is one of the following:
#
#  - 0: bottom_level
#  - 1: simple_group
#  - 2: hbox_group
#  - 3: adjusted_hbox_group
#  - 4: vbox_group
#  - 5: vtop_group
#  - 6: align_group
#  - 7: no_align_group
#  - 8: output_group
#  - 9: math_group
#  - 10: disc_group
#  - 11: insert_group
#  - 12: vcenter_group
#  - 13: math_choice_group
#  - 14: semi_simple_group
#  - 15: math_shift_group
#  - 16: math_left_group
#
# cur_level is the current group level and increases when groups are nested.
#
# Returns:
#   - A tuple (group, level).
#
def get_cur_group() -> Tuple[int, int]: ...

# This function returns the current mode.
# It's one of the following:
#
#  - 0: in no mode
#  - 1: vertical mode (vmode)
#  - 1+1*(max_command+1) == 105: horizontal mode (hmode)
#  - 1+2*(max_command+1) == 209: displayed forumla mode (mmode)
#  - -1: internal vertical mode
#  - -105: restricted horizontal mode
#  - -209: math mode
#
# NOTE: The above values assume that max_command is 103.
# Returns:
#   - The current mode as integer.
#
def get_cur_mode() -> int: ...

# Gets the current active text font index.
def get_cur_font() -> int: ...

# This function returns the name of a font with the given index.
def get_font_name(idx: int, /) -> memoryview: ...

# This function returns the number of fonts in the system.
def get_font_count() -> int: ...

# This function returns the spacing (slant, space, space_stretch, space_shrink, x_height, quad, extra_space, pdf_px_dimen)
def get_font_properties(idx: int, /) -> Tuple[int, int, int, int, int, int, int, int]: ...

# This function returns the font index for the given math font family
def get_family_font(family: int, /) -> int: ...

# This function gets the real font for a given font. Returns the same number if the font is not virtual.
def get_real_font(idx: int, /) -> int: ...

# This function returns the math charmap. The result is a 256-item list/tuple of (character, family)
# for each of the 256 inputs.
def get_math_charmap() -> Tuple[Tuple[int, int], ...]: ...

# Gets the memory array.
def get_mem() -> Tuple[memoryview, int, int, int, int]: ...

# Gets the current output state (mode, head_addr, tail_addr). The offset must be negative and
# specifies the save stack offset (0 = stack top).
def get_cur(offset: int = 0) -> Tuple[int, int, int]: ...

# Get the token list address of the param stack item with the given index.
def get_param_stack_item(idx: int, /) -> int: ...

# Returns the current param_ptr, i.e. the first unused slot in param_stack, i.e. the current length
def get_param_ptr() -> int: ...

# Gets the box register with the given number, returns the box address.
def get_box(register_num: int, /) -> int: ...

# Gets the box currently being built.
def get_cur_box() -> int: ...

# Create a hook for a control sequence
def hook_cs(
    cs_id: int, callback: Callable[[int, int, Tuple[int, ...] | None, int, int, int, int], bool], /
) -> None: ...
