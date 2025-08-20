# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from collections import defaultdict


_re_parse_cmd1 = re.compile(
    r"(?P<cmd>\\[a-zA-Z]+)[^;]*;[^;]*;(?P<tval>[^;]*);.*?&#x(?P<val>[0-9a-fA-F]+);"
)
_re_parse_cmd2 = re.compile(r"(?P<cmd>\\[a-zA-Z]+)[^;]*;[^;]*;(?P<tval>[^;]+);")

if __name__ == "__main__":
    kv = defaultdict(list)
    with open("tmp.csv", errors="ignore") as file:
        for line in file:
            m = _re_parse_cmd1.match(line)
            if m:
                cmd = m.group("cmd")
                val = m.group("val")
                kv[val].append(cmd)
                # print(m.groupdict())
            else:
                m = _re_parse_cmd2.match(line)
                if m:
                    cmd = m.group("cmd")
                    kv[m.group("tval")].append(cmd)
                    # print(m.groupdict())
    for key, value in kv.items():
        print(f"{key!r}: {value!r}")
