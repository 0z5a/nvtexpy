# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Append simplified Chinese fonts supported by LaTeX zhmetrics package to the fonts_glyphmap.zip file.
"""

import subprocess
import zipfile
from nvtexpy.tools import ENGINE_PATH


def get_charset(path, index=0) -> str:
    """Read the supported charset from the font file."""
    chars = []
    res = subprocess.run(
        ["fc-query", "--format=%{charset}\n", path, f"--index={index}"],
        check=True,
        capture_output=True,
        text=True,
    )
    for value in res.stdout.split():
        if "-" in value:
            start, end = value.split("-")
            for cp in range(int(start, 16), int(end, 16) + 1):
                chars.append(chr(cp))
        else:
            chars.append(chr(int(value, 16)))
    return "".join(sorted(chars))


fonts = {
    "unifs": "simfang.ttf",
    "unihei": "simhei.ttf",
    "unikai": "simkai.ttf",
    "unili": "simli.ttf",
    "unisong": "simsun.ttc",
    "uniyou": "simyou.ttf",
}

print("Reading character sets...")

charsets = {}
for fontname, fontpath in fonts.items():
    charsets[fontname] = get_charset(fontpath)

print(f"Writing glyphmaps for {len(charsets)} fonts...")

with zipfile.ZipFile(ENGINE_PATH / "fonts_glyphmap.zip", "a") as wf_all:
    for fontname, charset in charsets.items():
        pages = sorted(set([ord(c) >> 8 for c in charset]))
        for page in pages:
            page_codes = sorted([ord(c) for c in charset if ord(c) >> 8 == page])
            if not page_codes:
                continue
            wf_all.writestr(
                f"{fontname}{page:02x}.txt",
                "\n".join(f"{code & 0xFF:02X};{code:04X}" for code in page_codes),
            )
