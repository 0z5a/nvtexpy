# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import re
import zipfile
from collections import defaultdict
from pathlib import Path

from nvtexpy.tools import ENGINE_PATH
from vf_reader import get_vf_remapping

with open(ENGINE_PATH / "combined_glyphmap.txt", "r") as rf:
    main_glyphmap = {}
    for line in rf:
        codename, hexcode, unicode = line.split(";", maxsplit=2)
        main_glyphmap[codename] = hexcode


_re_uni = re.compile(r"^uni([0-9A-Fa-f]{4})+$")
_re_u = re.compile(r"^u([0-9A-Fa-f]{4,6})$")


def read_no_comments(f):
    for line in f:
        line = line.strip()
        if line.startswith("%"):
            continue
        elif "%" in line:
            yield line[: line.index("%")].strip()
        else:
            yield line


def read_no_comments_b(f):
    for line in f:
        line = line.strip()
        if line.startswith(b"%"):
            continue
        elif b"%" in line:
            yield line[: line.index(b"%")].strip()
        else:
            yield line


def _get_glyph_unicode(tfmname: str, code: str) -> str | None:
    if "." in code:
        # Try to load without the suffix first
        r = _get_glyph_unicode(tfmname, code.split(".", maxsplit=1)[0])
        if r is not None:
            return r
    if "_" in code:
        codes = [_get_glyph_unicode(tfmname, c) for c in code.split("_")]
        if any(code is None for code in codes):
            return None
        return " ".join(code for code in codes if code is not None)
    elif f"tfm:{tfmname}/{code}" in main_glyphmap:
        return main_glyphmap[f"tfm:{tfmname}/{code}"]
    elif code in main_glyphmap:
        return main_glyphmap[code]
    elif code.startswith("uni"):
        # Direct unicode hex of 4 digits, possibly multiple values
        _re_uni_match = _re_uni.match(code)
        if _re_uni_match is not None:
            return " ".join(code[i : i + 4] for i in range(3, len(code), 4))
    elif code.startswith("u"):
        u_match = _re_u.match(code)
        if u_match is not None:
            # Direct unicode hex, 4-6 digits
            return f"{u_match.group(1)}"
    elif code.lower() in main_glyphmap:
        return main_glyphmap[code.lower()]
    elif code.rstrip("0123456789") in main_glyphmap:
        return main_glyphmap[code.rstrip("0123456789")]
    return None


print("Loading font glyphs from .pfb...")
# font_glyphmap["fontname"][code] = "charname"
font_glyphmap: dict[str, dict[int, str]] = defaultdict(dict)

# read "dup 173 /harpoonleftright put"
_pfb_encoding_re = re.compile(
    rb"""
    /Encoding
        (?:
            (?:\s+(?P<standard>StandardEncoding)\s+def)
            |(?:\s*\[
                    \s*(?P<arrdef>.+?)
                \]\s+(?:readonly\s+)def
            )
            |(?:\s+256+\s+array
                (?:\s+0\s+1\s+255\s*\{\s*1\s+index\s+exch\s*/\.notdef\s+put\s*\}\s*for)?
                \s*(?P<dupdef>.*?)
                \s*(?:readonly\s+)def
            )
        )
    """,
    re.VERBOSE | re.MULTILINE | re.DOTALL,
)
_pfb_entry_dup_re = re.compile(
    rb"^dup\s+(?P<code>\d+)\s*/(?P<charname>[\.a-zA-Z0-9_\-\\\*]+)\s+put\s*",
    re.MULTILINE | re.DOTALL,
)
_pfb_entry_dup_exch_re = re.compile(
    rb"^dup\s+dup\s+(?P<to>\d+)\s+exch\s+(?P<from>\d+)\s+get\s+put\s*", re.MULTILINE | re.DOTALL
)
_pfb_entry_dup_interval_re = re.compile(
    rb"^dup\s+dup\s+(?P<from>\d+)\s+(?P<size>\d+)\s+getinterval\s+(?P<to>\d+)\s+exch\s+putinterval\s*",
    re.MULTILINE | re.DOTALL,
)
_pfb_arr_entry_re = re.compile(
    rb"^/(?P<charname>[\.a-zA-Z0-9_\-\\\*]+)\s*", re.MULTILINE | re.DOTALL
)

# From writet1.c
standard_glyph_names = [
    # 0x00
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    # 0x10
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    # 0x20
    "space",
    "exclam",
    "quotedbl",
    "numbersign",
    "dollar",
    "percent",
    "ampersand",
    "quoteright",
    "parenleft",
    "parenright",
    "asterisk",
    "plus",
    "comma",
    "hyphen",
    "period",
    "slash",
    # 0x30
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "colon",
    "semicolon",
    "less",
    "equal",
    "greater",
    "question",
    # 0x40
    "at",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    # 0x50
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "bracketleft",
    "backslash",
    "bracketright",
    "asciicircum",
    "underscore",
    # 0x60
    "quoteleft",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    # 0x70
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "braceleft",
    "bar",
    "braceright",
    "asciitilde",
    None,
    # 0x80
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    # 0x90
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    # 0xa0
    None,
    "exclamdown",
    "cent",
    "sterling",
    "fraction",
    "yen",
    "florin",
    "section",
    "currency",
    "quotesingle",
    "quotedblleft",
    "guillemotleft",
    "guilsinglleft",
    "guilsinglright",
    "fi",
    "fl",
    # 0xb0
    None,
    "endash",
    "dagger",
    "daggerdbl",
    "periodcentered",
    None,
    "paragraph",
    "bullet",
    "quotesinglbase",
    "quotedblbase",
    "quotedblright",
    "guillemotright",
    "ellipsis",
    "perthousand",
    None,
    "questiondown",
    # 0xc0
    None,
    "grave",
    "acute",
    "circumflex",
    "tilde",
    "macron",
    "breve",
    "dotaccent",
    "dieresis",
    None,
    "ring",
    "cedilla",
    None,
    "hungarumlaut",
    "ogonek",
    "caron",
    # 0xd0
    "emdash",
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    # 0xe0
    None,
    "AE",
    None,
    "ordfeminine",
    None,
    None,
    None,
    None,
    "Lslash",
    "Oslash",
    "OE",
    "ordmasculine",
    None,
    None,
    None,
    None,
    # 0xf0
    None,
    "ae",
    None,
    None,
    None,
    "dotlessi",
    None,
    None,
    "lslash",
    "oslash",
    "oe",
    "germandbls",
    None,
    None,
    None,
    None,
]


def parse_pfb(fnt_path: Path) -> dict[int, str]:
    glyphmap = {}
    with fnt_path.open("rb") as f:
        raw = f.read()
    if b"dict begin" in raw:
        raw = raw[raw.index(b"dict begin") :]
    elif b"dict dup begin" in raw:
        raw = raw[raw.index(b"dict dup begin") :]
    else:
        assert False, f"Invalid PFB file {fnt_path}"
    assert b"currentdict end" in raw, f"Invalid PFB file {fnt_path}"
    raw = raw[: raw.index(b"currentdict end")]
    raw = raw.replace(b"\r", b"\n")
    # Remove comments
    raw = b"\n".join(
        line[: line.index(b"%")] if b"%" in line else line
        for line in raw.split(b"\n")
        if line.strip() and not line.lstrip().startswith(b"%")
    )
    # Find the encoding
    match = _pfb_encoding_re.search(raw)
    if not match and b"/Encoding" in raw:
        print(f"Invalid encoding line in {fnt_path}: {raw}")
    assert match, f"No encoding defined in {fnt_path}"
    if match.group("standard"):
        for idx, charname in enumerate(standard_glyph_names):
            if charname is None:
                continue
            glyphmap[idx] = charname
    elif match.group("arrdef"):
        inner = match.group("arrdef")
        while inner:
            match = _pfb_arr_entry_re.match(inner)
            assert match, f"Invalid encoding entry in {fnt_path}: {inner}"
            charname = match.group("charname").decode()
            if charname != ".notdef":
                glyphmap[len(glyphmap)] = charname
            inner = inner[match.end() :]
    else:
        assert match.group("dupdef") is not None, f"Invalid encoding line in {fnt_path}: {raw}"
        inner = match.group("dupdef")
        while inner:
            match = _pfb_entry_dup_re.match(inner)
            if match:
                code = match.group("code").decode()
                charname = match.group("charname").decode()
                if charname != ".notdef":
                    glyphmap[int(code)] = charname
                inner = inner[match.end() :]
                continue
            match = _pfb_entry_dup_exch_re.match(inner)
            if match:
                to = int(match.group("to"))
                from_ = int(match.group("from"))
                charname = glyphmap.get(from_)
                if charname is not None:
                    glyphmap[to] = charname
                inner = inner[match.end() :]
                continue
            match = _pfb_entry_dup_interval_re.match(inner)
            if match:
                to = int(match.group("to"))
                from_ = int(match.group("from"))
                size = int(match.group("size"))
                for i in range(size):
                    charname = glyphmap.get(from_ + i)
                    if charname is not None:
                        glyphmap[to + i] = charname
                inner = inner[match.end() :]
                continue
            assert False, f"Invalid line in {fnt_path}: {inner}"
    if len(glyphmap) == 0:
        print(f"Empty glyphmap for {fnt_path}")
    return glyphmap


for fnt_path in Path("/usr/share/texmf-dist/fonts/type1").rglob("*.pfb"):
    font_name = fnt_path.name[:-4]
    print(f"Loading font {font_name}")
    font_glyphmap[font_name] = parse_pfb(fnt_path)


print("Loading font maps from .map...")


# enc_registry[encoding filename][code] = "charname"
enc_registry: dict[str, dict[int, str]] = {}

_enc_inner = re.compile(
    rb"^\s*/(?P<encoding>[a-zA-Z0-9\-_\.]+)\s*\[\s*(?P<chars>.+?)\s*\]\s*def\s*$",
    re.MULTILINE | re.DOTALL,
)
_enc_entry_re = re.compile(rb"^/(?P<charname>[a-zA-Z0-9\._\-\\\*]+)\s*", re.MULTILINE | re.DOTALL)

encoding_files = list(Path("/usr/share/texmf-dist/fonts/enc").rglob("*.enc"))
encoding_files_by_name = {file.name: file for file in encoding_files}
encoding_files_by_name_lower = {file.name.lower(): file for file in encoding_files}

# print("Enc Files:", encoding_files_by_name_lower)


def get_enc(enc_file: str) -> dict[int, str]:
    if enc_file in enc_registry:
        return enc_registry[enc_file]
    enc_registry[enc_file] = enc = {}
    # Find this file
    enc_path = encoding_files_by_name.get(enc_file)
    if enc_path is None:
        enc_path = encoding_files_by_name_lower.get(enc_file.lower())
    if enc_path is None:
        print(f"Missing encoding file {enc_file}")
        return enc
    with enc_path.open("rb") as f:
        data = f.read()
    data = data.replace(b"\r", b"\n")
    # Remove comments
    data = b"\n".join(
        line[: line.index(b"%")] if b"%" in line else line
        for line in data.split(b"\n")
        if line.strip() and not line.lstrip().startswith(b"%")
    )
    match = _enc_inner.match(data)
    assert match, f"Invalid encoding file {enc_file}"
    inner = match.group("chars")
    idx = 0
    while inner:
        match = _enc_entry_re.match(inner)
        assert match, f"Invalid encoding entry in {enc_file}: {inner}"
        charname = match.group("charname").decode()
        if charname != ".notdef":
            enc[idx] = charname
        inner = inner[match.end() :]
        idx += 1
    return enc


# read "ecrm1000 SFRM1000 "T1Encoding ReEncodeFont" <cm-super-t1.enc <sfrm1000.pfb"
_map_entry_re = re.compile(
    r"""^
    (?P<fontname>[0-9a-zA-Z_\-]+)\s+
    (?:(?:\"(?:[^\"]+)\"|\<?[\[\<]?[0-9a-zA-Z_\-\.]+)\s+)*
    \<?[\[\<]?(?P<encfile>[0-9a-zA-Z_\-\.]+\.enc)\s+
    """,
    re.MULTILINE | re.DOTALL | re.VERBOSE,
)
# ntx-Regular-tlf-t1 TeXGyreTermesX-Regular " encntx-ec-tlf ReEncodeFont " <[ntx-ec-tlf.enc <ztmr.pfb
for fnt_path in itertools.chain(
    Path("/usr/share/texmf-dist/fonts/map").rglob("**/*.map"),
    Path("/usr/share/texmf-var/fonts/map").rglob("**/*.map"),
):
    with fnt_path.open("r") as f:
        print(f"Processing {fnt_path}")
        for line in f:
            match = _map_entry_re.match(line)
            if match:
                font_name = match.group("fontname")
                # font_file = match.group("fontfile")
                # encoding = match.group("encoding")
                enc_file = match.group("encfile")
                # pfb_file = match.group("pfbfile")
                print(f"Registering font {font_name} from {enc_file}")
                font_enc = get_enc(enc_file)
                if font_name in font_glyphmap:
                    existing_map = font_glyphmap[font_name]
                    if existing_map != font_enc:
                        print(f"WARNING: Font {font_name} already loaded, overwriting")
                        for key, value in font_enc.items():
                            exval = existing_map.get(key)
                            if exval != value:
                                if exval is not None and (
                                    "." not in exval
                                    and (not exval.startswith("uni") or "." in value)
                                ):
                                    new_value = exval
                                else:
                                    new_value = value
                                print(
                                    f"  - {key} {existing_map.get(key)!r} <-> {value!r} -> {new_value!r}"
                                )
                                existing_map[key] = new_value
                        # font_glyphmap[font_name].update(font_enc)
                    # font_glyphmap[font_name] = font_enc
                else:
                    font_glyphmap[font_name] = font_enc


print("Reading virtual fonts...")


def read_vf_real_font(vf_file: Path) -> str | None:
    with vf_file.open("rb") as f:
        h = f.read(2)
        assert h == b"\xf7\xca", f"Invalid header {h}"
        cmdl = f.read(1)[0]
        f.read(cmdl)
        f.read(4 + 4)
        cmd = f.read(1)
        if cmd[0] < 243 or 246 < cmd[0]:
            return None
        f.read(cmd[0] - 243 + 1)
        f.read(4)
        f.read(4)
        f.read(4)
        fnt_path_len = f.read(1)[0]
        fnt_name_len = f.read(1)[0]
        f.read(fnt_path_len).decode()
        fnt_name = f.read(fnt_name_len).decode()
        return fnt_name


for vf_fnt_path in Path("/usr/share/texmf-dist/fonts/vf").rglob("**/*.vf"):
    # print(f"Reading {vf_fnt_path}")
    # real_font = read_vf_real_font(vf_fnt_path)
    real_font, remapping = get_vf_remapping(vf_fnt_path)
    print(f"Redirecting {vf_fnt_path.name[:-3]} -> {real_font}")
    if remapping and real_font:
        font_glyphmap[vf_fnt_path.name[:-3]] = vf_glyphmap = dict(font_glyphmap[real_font])
        for key, src_glyphs in remapping.items():
            dchr = "_".join(
                font_glyphmap[src_fnt][src_idx]
                for src_fnt, src_idx in src_glyphs
                if src_idx in font_glyphmap[src_fnt]
            )
            if dchr:
                vf_glyphmap[key] = dchr
                # print(f" - 0x{key:02X} -> {dchr} ({src_glyphs})")


print(f"Writing glyphmaps for {len(font_glyphmap)} fonts...")

unsupported_fonts = []

with zipfile.ZipFile(ENGINE_PATH / "fonts_glyphmap.zip", "w") as wf_all:
    for fontname, chars in font_glyphmap.items():
        translated_chars = {}
        for code, charname in chars.items():
            unicode = _get_glyph_unicode(fontname, charname)
            if unicode is not None:
                translated_chars[code] = unicode
            else:
                pass
                # print(f"Undef glyph: {fontname}/{code!r} {charname!r}")
        if len(translated_chars) == 0:
            continue

        wf_all.writestr(
            f"{fontname}.txt",
            "\n".join(f"{code:02X};{unicode}" for code, unicode in translated_chars.items()),
        )

with open(ENGINE_PATH / "fonts_glyphmap.txt", "w") as wf_all:
    for fontname, chars in font_glyphmap.items():
        translated_chars = {}
        for code, charname in chars.items():
            unicode = _get_glyph_unicode(fontname, charname)
            if unicode is not None:
                translated_chars[code] = unicode
            else:
                pass
                # print(f"Undef glyph: {fontname}/{code!r} {charname!r}")
        if len(translated_chars) == 0:
            unsupported_fonts.append(fontname)
            wf_all.write(f"{fontname};unsupported\n")
            # print(f"Unsupported font: {fontname}")
            continue
        else:
            wf_all.write(f"{fontname};supported\n")

# start = time.time()
# reconstructed_fontmap: dict[str, dict[str, str] | None] = defaultdict(dict)
# with open(ENGINE_PATH / "fonts_glyphmap.txt", "r") as wf_all:
#     for line in wf_all:
#         line = line.rstrip()
#         if line.endswith("unsupported"):
#             fontname, _ = line.split(";", maxsplit=1)
#             reconstructed_fontmap[fontname] = None
#             continue
#         fontname, code, unicode = line.split(";", maxsplit=2)
#         reconstructed_fontmap[fontname][int(code, 16)] = "".join(
#             chr(int(c, 16)) for c in unicode.split()
#         )
# print(f"Loading took {time.time() - start:.2f}s")
