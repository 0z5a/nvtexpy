# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025, NVIDIA CORPORATION.
import mmap
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Counter, Generator

from nvtexpy.tools import ENGINE_PATH

# Maps from glyphname to unicode (as hex), with alternative unicode representations
global_glyphmap: dict[str, list[list[str]]] = defaultdict(list)


def update_glyphmap(new_map: dict[str, str]):
    for key, value in new_map.items():
        new_alternatives = [
            a
            for a in (
                [
                    cc
                    for cc in a.split(" ")
                    if cc and unicodedata.category(chr(int(cc, 16))) not in ("Co", "Cs", "Cc", "Cf")
                ]
                for a in value.split(",")
            )
            if len(a) > 0 and not all(unicodedata.category(chr(int(cc, 16))) in ("Mn",) for cc in a)
        ]
        if len(new_alternatives) == 0:
            continue
        global_glyphmap[key].extend(new_alternatives)


def read_glyphmap_txt(file: Path) -> dict[str, str]:
    glyphmap = {}
    with file.open("r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            glyphname, unicode = line.strip().split(";")
            glyphmap[glyphname] = unicode
    return glyphmap


print("Loading glyphlists...")
# Find all /usr/share/texmf-dist/tex/latex/*.txt
for file in (
    Path("/usr/share/texmf-dist/fonts/map/glyphlist/glyphlist.txt"),
    Path("/usr/share/texmf-dist/fonts/map/glyphlist/texglyphlist.txt"),
    Path("/usr/share/texmf-dist/fonts/map/glyphlist/pdfglyphlist.txt"),
):
    print(f"Processing {file}")
    update_glyphmap(read_glyphmap_txt(file))
    print(f"Processed {file}")

print(f"Num Glyphmap entries: {len(global_glyphmap)}")


def search_files(root: Path, file_pattern: str, pattern: bytes) -> Generator[Path, None, None]:
    for file in root.rglob(file_pattern):
        if file.is_file() and file.stat().st_size > 10:
            with file.open("r") as f:
                m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                if m.find(pattern) != -1:
                    yield file


_re_glyph_to_unicode = re.compile(
    rb"""
        \\pdfglyphtounicode (?:\s*\#.*\n)*\s*
        \{ (?:\s*\#.*\n)*\s*
          (?P<glyph>[a-zA-Z0-9_:/\.\-]+) (?:\s*\#.*\n)*\s*
        \} (?:\s*\#.*\n)*\s*
        \{ (?:\s*\#.*\n)*\s*
          (?P<unicode>(?:[0-9a-fA-F]+) (?:\s*\#.*\n)*\s*)+
        \} (?:\s*\#.*\n)*\s*
    """,
    flags=re.VERBOSE,
)
_re_glyph_to_unicode_special = re.compile(
    rb"""
        \\p (?:\s*\#.*\n)*\s*
        \{ (?:\s*\#.*\n)*\s*
          (?P<glyph>[a-zA-Z0-9_:/\.\-]+) (?:\s*\#.*\n)*\s*
        \} (?:\s*\#.*\n)*\s*
        \{ (?:\s*\#.*\n)*\s*
          (?P<unicode>(?:[0-9a-fA-F]+) (?:\s*\#.*\n)*\s*)+
        \} (?:\s*\#.*\n)*\s*
    """,
    flags=re.VERBOSE,
)


def extract_glyph_to_unicode(file: Path) -> dict[str, str]:
    glyph_to_unicode = {}
    with open(file, "rb") as rf:
        content = rf.read()
        if b"\\let\\p\\pdfglyphtounicode" in content:
            for match in _re_glyph_to_unicode_special.finditer(content):
                glyph, unicode = match.group("glyph", "unicode")
                glyph_to_unicode[glyph.decode()] = unicode.decode()
        for match in _re_glyph_to_unicode.finditer(content):
            glyph, unicode = match.group("glyph", "unicode")
            glyph_to_unicode[glyph.decode()] = unicode.decode()
    return glyph_to_unicode


print("Processing pdfglyphtounicode...")
# Find files in /usr/share/texmf-dist containing "\\pdfglyphtounicode" as string
for file in search_files(Path("/usr/share/texmf-dist"), "*.*", b"\\pdfglyphtounicode"):
    print(f"Processing {file}")
    update_glyphmap(extract_glyph_to_unicode(file))

print(f"Num Glyphmap entries: {len(global_glyphmap)}")


default_translations = {
    #   "grave": "`",
    #   "acute": "´",
    "circumflex": "^",
    "tilde": "~",
    #   "dieresis": "¨",
    #   "hungarumlaut": "˝",
    #   "ring": "˚",
    #   "caron": "ˇ",
    #   "breve": "˘",
    #   "macron": "¯",
    #   "dotaccent": "˙",
    #   "cedilla": "¸",
    #   "ogonek": "˛",
    #   "quotesinglbase": "‚",
    #   "guilsinglleft": "‹",
    #   "guilsinglright": "›",
    #   "quotedblleft": "“",
    #   "quotedblright": "”",
    #   "quotedblbase": "„",
    #   "guillemotleft": "«",
    #   "guillemotright": "»",
    #   "endash": "–",
    #   "emdash": "—",
    #   "cwm": "\u200C",
    #   "perthousand": "‰",
    #   "dotlessi": "ı",
    #   "dotlessj": "ȷ",
    "ff": "ff",
    "fi": "fi",
    "fl": "fl",
    "ffi": "ffi",
    "ffl": "ffl",
    "space": " ",
    "exclam": "!",
    "quotedbl": '"',
    "numbersign": "#",
    "dollar": "$",
    "percent": "%",
    "ampersand": "&",
    "quoteright": "'",
    "parenleft": "(",
    "parenright": ")",
    "asterisk": "*",
    "plus": "+",
    "comma": ",",
    "hyphen": "-",
    "period": ".",
    "slash": "/",
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "zerooldstyle": "0",
    "oneoldstyle": "1",
    "twooldstyle": "2",
    "threeoldstyle": "3",
    "fouroldstyle": "4",
    "fiveoldstyle": "5",
    "sixoldstyle": "6",
    "sevenoldstyle": "7",
    "eightoldstyle": "8",
    "nineoldstyle": "9",
    "colon": ":",
    "semicolon": ";",
    "less": "<",
    "equal": "=",
    "greater": ">",
    "question": "?",
    "at": "@",
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
    "E": "E",
    "F": "F",
    "G": "G",
    "H": "H",
    "I": "I",
    "J": "J",
    "K": "K",
    "L": "L",
    "M": "M",
    "N": "N",
    "O": "O",
    "P": "P",
    "Q": "Q",
    "R": "R",
    "S": "S",
    "T": "T",
    "U": "U",
    "V": "V",
    "W": "W",
    "X": "X",
    "Y": "Y",
    "Z": "Z",
    "bracketleft": "[",
    "backslash": "\\",
    "bracketright": "]",
    "asciicircum": "^",
    "underscore": "_",
    #   "quoteleft": "‘",
    "a": "a",
    "b": "b",
    "c": "c",
    "d": "d",
    "e": "e",
    "f": "f",
    "g": "g",
    "h": "h",
    "i": "i",
    "j": "j",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "o": "o",
    "p": "p",
    "q": "q",
    "r": "r",
    "s": "s",
    "t": "t",
    "u": "u",
    "v": "v",
    "w": "w",
    "x": "x",
    "y": "y",
    "z": "z",
    "braceleft": "{",
    "bar": "|",
    "braceright": "}",
    "asciitilde": "~",
    # "angbracketleft": "\\langle",
    # "angbracketright": "\\rangle",
    #   "sfthyphen": "\u00AD",
    #   "Abreve": "Ă",
    #   "Aogonek": "Ą",
    #   "Cacute": "Ć",
    #   "Ccaron": "Č",
    #   "Dcaron": "Ď",
    #   "Ecaron": "Ě",
    #   "Eogonek": "Ę",
    #   "Gbreve": "Ğ",
    #   "Lacute": "Ĺ",
    #   "Lcaron": "Ľ",
    #   "Lslash": "Ł",
    #   "Nacute": "Ń",
    #   "Ncaron": "Ň",
    #   "Eng": "Ŋ",
    #   "Ohungarumlaut": "Ő",
    #   "Racute": "Ŕ",
    #   "Rcaron": "Ř",
    #   "Sacute": "Ś",
    #   "Scaron": "Š",
    #   "Scedilla": "Ş",
    #   "Tcaron": "Ť",
    #   "Tcedilla": "Ţ",
    #   "Uhungarumlaut": "Ű",
    #   "Uring": "Ů",
    #   "Ydieresis": "Ÿ",
    #   "Zacute": "Ź",
    #   "Zcaron": "Ž",
    #   "Zdotaccent": "Ż",
    #   "IJ": "Ĳ",
    #   "Idotaccent": "İ",
    #   "dcroat": "đ",
    #   "section": "§",
    #   "abreve": "ă",
    #   "aogonek": "ą",
    #   "cacute": "ć",
    #   "ccaron": "č",
    #   "dcaron": "ď",
    #   "ecaron": "ě",
    #   "eogonek": "ę",
    #   "gbreve": "ğ",
    #   "lacute": "ĺ",
    #   "lcaron": "ľ",
    #   "lslash": "ł",
    #   "nacute": "ń",
    #   "ncaron": "ň",
    #   "eng": "ŋ",
    #   "ohungarumlaut": "ő",
    #   "racute": "ŕ",
    #   "rcaron": "ř",
    #   "sacute": "ś",
    #   "scaron": "š",
    #   "scedilla": "ş",
    #   "tcaron": "ť",
    #   "tcedilla": "ţ",
    #   "uhungarumlaut": "ű",
    #   "uring": "ů",
    #   "ydieresis": "ÿ",
    #   "zacute": "ź",
    #   "zcaron": "ž",
    #   "zdotaccent": "ż",
    "ij": "ij",
    #   "exclamdown": "¡",
    #   "questiondown": "¿",
    #   "sterling": "£",
    #   "Agrave": "À",
    #   "Aacute": "Á",
    #   "Acircumflex": "Â",
    #   "Atilde": "Ã",
    #   "Adieresis": "Ä",
    #   "Aring": "Å",
    #   "AE": "Æ",
    #   "Ccedilla": "Ç",
    #   "Egrave": "È",
    #   "Eacute": "É",
    #   "Ecircumflex": "Ê",
    #   "Edieresis": "Ë",
    #   "Igrave": "Ì",
    #   "Iacute": "Í",
    #   "Icircumflex": "Î",
    #   "Idieresis": "Ï",
    #   "Eth": "Ð",
    #   "Ntilde": "Ñ",
    #   "Ograve": "Ò",
    #   "Oacute": "Ó",
    #   "Ocircumflex": "Ô",
    #   "Otilde": "Õ",
    #   "Odieresis": "Ö",
    #   "OE": "Œ",
    #   "Oslash": "Ø",
    #   "Ugrave": "Ù",
    #   "Uacute": "Ú",
    #   "Ucircumflex": "Û",
    #   "Udieresis": "Ü",
    #   "Yacute": "Ý",
    #   "Thorn": "Þ",
    #   "SS": "SS",
    #   "agrave": "à",
    #   "aacute": "á",
    #   "acircumflex": "â",
    #   "atilde": "ã",
    #   "adieresis": "ä",
    #   "aring": "å",
    #   "ae": "æ",
    #   "ccedilla": "ç",
    #   "egrave": "è",
    #   "eacute": "é",
    #   "ecircumflex": "ê",
    #   "edieresis": "ë",
    #   "igrave": "ì",
    #   "iacute": "í",
    #   "icircumflex": "î",
    #   "idieresis": "ï",
    #   "eth": "ð",
    #   "ntilde": "ñ",
    #   "ograve": "ò",
    #   "oacute": "ó",
    #   "ocircumflex": "ô",
    #   "otilde": "õ",
    #   "odieresis": "ö",
    #   "oe": "œ",
    #   "oslash": "ø",
    #   "ugrave": "ù",
    #   "uacute": "ú",
    #   "ucircumflex": "û",
    #   "udieresis": "ü",
    #   "yacute": "ý",
    #   "thorn": "þ",
    #   "germandbls": "ß",
    "minus": "-",
    "divide": "/",
}

update_glyphmap(
    dict((k, " ".join(f"{ord(c):04X}" for c in v)) for k, v in default_translations.items())
)


# glyphmap = {k: default_translations.get(v, "\\" + v) for k, v in glyphmap.items()}
# with (ENGINE_PATH / "fonts_glyphmap.json").open("w") as wf:
#     json.dump(glyphmap, wf, indent=2)


# with (ENGINE_PATH / "fonts_glyphmap_keys.txt").open("w") as wft:
#     for key in set(glyphmap.values()):
#         wft.write(f"{key!r},\n")

# import pickle

# with (ENGINE_PATH / "fonts_glyphmap.pkl").open("wb") as wf:
#     pickle.dump({
#         (fontname, char): default_translations[charname]
#         for fontname, chars in glyphmap.items()
#         for char, charname in chars.items()
#         if charname in default_translations
#     }, wf, pickle.HIGHEST_PROTOCOL)

# Clean the global_glyphmap
main_glyphmap = {}
for charname, unicode_alternatives in global_glyphmap.items():
    # Use the most frequent alternative
    main_glyphmap[charname] = Counter(tuple(x) for x in unicode_alternatives).most_common(1)[0][0]

# Write out the glyphmap to a file
print("Writing all_glyphmaps...")
with (ENGINE_PATH / "all_glyphmaps.txt").open("w") as wf:
    for charname, unicode_alternatives in global_glyphmap.items():
        unicode_alt_text = [
            "".join(chr(int(c, 16)) for c in glyphs) for glyphs in unicode_alternatives
        ]
        # unicode_alt_text = [unicodedata.normalize("NFKC", s) for s in unicode_alt_text]
        unicode_alt_text = [txt for txt, _cnt in Counter(unicode_alt_text).most_common()]
        unicode_alternatives = [[f"{ord(c):04X}" for c in alt] for alt in unicode_alt_text]
        # unicode_alternatives = [k for k, _cnt in Counter(tuple(a) for a in unicode_alternatives).most_common()]
        wf.write(
            f"{charname};{','.join(' '.join(alt) for alt in unicode_alternatives)};{','.join(unicode_alt_text)}\n"
        )


# Write out the glyphmap to a file
print("Writing all_glyphmaps...")
with (ENGINE_PATH / "main_glyphmaps.txt").open("w") as wf:
    for charname, glyph in main_glyphmap.items():
        wf.write(f"{charname};{' '.join(glyph)};{''.join(chr(int(c, 16)) for c in glyph)}\n")

print("Done")
