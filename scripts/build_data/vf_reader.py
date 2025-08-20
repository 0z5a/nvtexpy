# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Constants
from enum import IntEnum
from pathlib import Path

vf_id = 202
do_char = 70

vf_max_packet_length = 10000
vf_max_recursion = 10
vf_stack_size = 100


class DVICmd(IntEnum):
    """DVI commands"""

    """|set_char_0| 0. Typeset character number~0 from font~|f|
    such that the reference point of the character is at |(h,v)|. Then
    increase |h| by the width of that character. Note that a character may
    have zero or negative width, so one cannot be sure that |h| will advance
    after this command; but |h| usually does increase."""
    set_char_0 = 0
    """set_char_1 through set_char_127 (opcodes 1 to 127).
    Do the operations of |set_char_0|; but use the character whose number
    matches the opcode, instead of character~0."""
    set_char_1 = 1
    set_char_2 = 2
    set_char_3 = 3
    set_char_4 = 4
    set_char_5 = 5
    set_char_6 = 6
    set_char_7 = 7
    set_char_8 = 8
    set_char_9 = 9
    set_char_10 = 10
    set_char_11 = 11
    set_char_12 = 12
    set_char_13 = 13
    set_char_14 = 14
    set_char_15 = 15
    set_char_16 = 16
    set_char_17 = 17
    set_char_18 = 18
    set_char_19 = 19
    set_char_20 = 20
    set_char_21 = 21
    set_char_22 = 22
    set_char_23 = 23
    set_char_24 = 24
    set_char_25 = 25
    set_char_26 = 26
    set_char_27 = 27
    set_char_28 = 28
    set_char_29 = 29
    set_char_30 = 30
    set_char_31 = 31
    set_char_32 = 32
    set_char_33 = 33
    set_char_34 = 34
    set_char_35 = 35
    set_char_36 = 36
    set_char_37 = 37
    set_char_38 = 38
    set_char_39 = 39
    set_char_40 = 40
    set_char_41 = 41
    set_char_42 = 42
    set_char_43 = 43
    set_char_44 = 44
    set_char_45 = 45
    set_char_46 = 46
    set_char_47 = 47
    set_char_48 = 48
    set_char_49 = 49
    set_char_50 = 50
    set_char_51 = 51
    set_char_52 = 52
    set_char_53 = 53
    set_char_54 = 54
    set_char_55 = 55
    set_char_56 = 56
    set_char_57 = 57
    set_char_58 = 58
    set_char_59 = 59
    set_char_60 = 60
    set_char_61 = 61
    set_char_62 = 62
    set_char_63 = 63
    set_char_64 = 64
    set_char_65 = 65
    set_char_66 = 66
    set_char_67 = 67
    set_char_68 = 68
    set_char_69 = 69
    set_char_70 = 70
    set_char_71 = 71
    set_char_72 = 72
    set_char_73 = 73
    set_char_74 = 74
    set_char_75 = 75
    set_char_76 = 76
    set_char_77 = 77
    set_char_78 = 78
    set_char_79 = 79
    set_char_80 = 80
    set_char_81 = 81
    set_char_82 = 82
    set_char_83 = 83
    set_char_84 = 84
    set_char_85 = 85
    set_char_86 = 86
    set_char_87 = 87
    set_char_88 = 88
    set_char_89 = 89
    set_char_90 = 90
    set_char_91 = 91
    set_char_92 = 92
    set_char_93 = 93
    set_char_94 = 94
    set_char_95 = 95
    set_char_96 = 96
    set_char_97 = 97
    set_char_98 = 98
    set_char_99 = 99
    set_char_100 = 100
    set_char_101 = 101
    set_char_102 = 102
    set_char_103 = 103
    set_char_104 = 104
    set_char_105 = 105
    set_char_106 = 106
    set_char_107 = 107
    set_char_108 = 108
    set_char_109 = 109
    set_char_110 = 110
    set_char_111 = 111
    set_char_112 = 112
    set_char_113 = 113
    set_char_114 = 114
    set_char_115 = 115
    set_char_116 = 116
    set_char_117 = 117
    set_char_118 = 118
    set_char_119 = 119
    set_char_120 = 120
    set_char_121 = 121
    set_char_122 = 122
    set_char_123 = 123
    set_char_124 = 124
    set_char_125 = 125
    set_char_126 = 126

    """|set_char_127| 127. Do the operations of |set_char_0|; but use
    character number~127, instead of character~0."""
    set_char_127 = 127

    """|set1| 128 |c[1]|. Same as |set_char_0|, except that character
    number~|c| is typeset. TeX82 uses this command for characters in the
    range |128<=c<256|."""
    set1 = 128

    """|@!set2| 129 |c[2]|. Same as |set1|, except that |c|~is two
    bytes long, so it is in the range |0<=c<65536|. TeX82 never uses this
    command, but it should come in handy for extensions of TeX that deal
    with oriental languages."""
    set2 = 129

    """|@!set3| 130 |c[3]|. Same as |set1|, except that |c|~is three
    bytes long, so it can be as large as $2^{24}-1$. Not even the Chinese
    language has this many characters, but this command might prove useful
    in some yet unforeseen extension.
    """
    set3 = 130

    """|@!set4| 131 |c[4]|. Same as |set1|, except that |c|~is four
    bytes long. Imagine that."""
    set4 = 131

    """|set_rule| 132 |a[4]| |b[4]|. Typeset a solid black rectangle
    of height~|a| and width~|b|, with its bottom left corner at |(h,v)|. Then
    set |h:=h+b|. If either |a<=0| or |b<=0|, nothing should be typeset. Note
    that if |b<0|, the value of |h| will decrease even though nothing else happens.
    See below for details about how to typeset rules so that consistency with
    MF is guaranteed."""

    set_rule = 132
    """|@!put1| 133 |c[1]|. Typeset character number~|c| from font~|f|
    such that the reference point of the character is at |(h,v)|. (The `put'
    commands are exactly like the `set' commands, except that they simply put out a
    character or a rule without moving the reference point afterwards.)"""

    put1 = 133
    """|@!put2| 134 |c[2]|. Same as |set2|, except that |h| is not changed."""
    put2 = 134
    """|@!put3| 135 |c[3]|. Same as |set3|, except that |h| is not changed."""
    put3 = 135
    """|@!put4| 136 |c[4]|. Same as |set4|, except that |h| is not changed."""
    put4 = 136

    """|put_rule| 137 |a[4]| |b[4]|. Same as |set_rule|, except that
    |h| is not changed."""
    put_rule = 137

    """|nop| 138. No operation, do nothing. Any number of |nop|'s
    may occur between .DVI commands, but a |nop| cannot be inserted between
    a command and its parameters or between two parameters."""
    nop = 138

    """|bop| 139 $c_0[4]$ $c_1[4]$ $\\ldots$ $c_9[4]$ $p[4]$. Beginning
    of a page: Set |(h,v,w,x,y,z):=(0,0,0,0,0,0)| and set the stack empty. Set
    the current font |f| to an undefined value.  The ten $c_i$ parameters hold
    the values of \\count0 $\\ldots$ \\count9 in TeX at the time
    \\shipout was invoked for this page; they can be used to identify
    pages, if a user wants to print only part of a .DVI file. The parameter
    |p| points to the previous |bop| in the file; the first
    |bop| has $p=-1$."""
    bop = 139

    """|eop| 140.  End of page: Print what you have read since the
    previous |bop|. At this point the stack should be empty. (The .DVI-reading
    programs that drive most output devices will have kept a buffer of the
    material that appears on the page that has just ended. This material is
    largely, but not entirely, in order by |v| coordinate and (for fixed |v|) by
    |h|~coordinate; so it usually needs to be sorted into some order that is
    appropriate for the device in question.)"""
    eop = 140

    """|push| 141. Push the current values of |(h,v,w,x,y,z)| onto the
    top of the stack; do not change any of these values. Note that |f| is
    not pushed."""
    push = 141

    """|pop| 142. Pop the top six values off of the stack and assign
    them respectively to |(h,v,w,x,y,z)|. The number of pops should never
    exceed the number of pushes, since it would be highly embarrassing if the
    stack were empty at the time of a |pop| command."""
    pop = 142

    """|right1| 143 |b[1]|. Set |h:=h+b|, i.e., move right |b| units.
    The parameter is a signed number in two's complement notation, |-128<=b<128|;
    if |b<0|, the reference point moves left."""
    right1 = 143

    """|@!right2| 144 |b[2]|. Same as |right1|, except that |b| is a
    two-byte quantity in the range |-32768<=b<32768|."""
    right2 = 144

    """|@!right3| 145 |b[3]|. Same as |right1|, except that |b| is a
    three-byte quantity in the range |@t$-2^{23}$@><=b<@t$2^{23}$@>|."""
    right3 = 145

    """|@!right4| 146 |b[4]|. Same as |right1|, except that |b| is a
    four-byte quantity in the range |@t$-2^{31}$@><=b<@t$2^{31}$@>|."""
    right4 = 146

    """|w0| 147. Set |h:=h+w|; i.e., move right |w| units. With luck,
    this parameterless command will usually suffice, because the same kind of motion
    will occur several times in succession; the following commands explain how
    |w| gets particular values."""
    w0 = 147
    """|w1| 148 |b[1]|. Set |w:=b| and |h:=h+b|. The value of |b| is a
    signed quantity in two's complement notation, |-128<=b<128|. This command
    changes the current |w|~spacing and moves right by |b|."""
    w1 = 148
    """|@!w2| 149 |b[2]|. Same as |w1|, but |b| is two bytes long,
    |-32768<=b<32768|."""
    w2 = 149
    """|@!w3| 150 |b[3]|. Same as |w1|, but |b| is three bytes long,
    |@t$-2^{23}$@><=b<@t$2^{23}$@>|."""
    w3 = 150
    """|@!w4| 151 |b[4]|. Same as |w1|, but |b| is four bytes long,
    |@t$-2^{31}$@><=b<@t$2^{31}$@>|."""
    w4 = 151

    """|x0| 152. Set |h:=h+x|; i.e., move right |x| units. The `|x|'
    commands are like the `|w|' commands except that they involve |x| instead
    of |w|."""
    x0 = 152
    """|x1| 153 |b[1]|. Set |x:=b| and |h:=h+b|. The value of |b| is a
    signed quantity in two's complement notation, |-128<=b<128|. This command
    changes the current |x|~spacing and moves right by |b|."""
    x1 = 153
    """|@!x2| 154 |b[2]|. Same as |x1|, but |b| is two bytes long,
    |-32768<=b<32768|."""
    x2 = 154
    """|@!x3| 155 |b[3]|. Same as |x1|, but |b| is three bytes long,
    |@t$-2^{23}$@><=b<@t$2^{23}$@>|."""
    x3 = 155
    """|@!x4| 156 |b[4]|. Same as |x1|, but |b| is four bytes long,
    |@t$-2^{31}$@><=b<@t$2^{31}$@>|."""
    x4 = 156

    """|down1| 157 |a[1]|. Set |v:=v+a|, i.e., move down |a| units.
    The parameter is a signed number in two's complement notation, |-128<=a<128|;
    if |a<0|, the reference point moves up."""
    down1 = 157
    """|@!down2| 158 |a[2]|. Same as |down1|, except that |a| is a
    two-byte quantity in the range |-32768<=a<32768|."""
    down2 = 158
    """|@!down3| 159 |a[3]|. Same as |down1|, except that |a| is a
    three-byte quantity in the range |@t$-2^{23}$@><=a<@t$2^{23}$@>|."""
    down3 = 159
    """|@!down4| 160 |a[4]|. Same as |down1|, except that |a| is a
    four-byte quantity in the range |@t$-2^{31}$@><=a<@t$2^{31}$@>|."""
    down4 = 160

    """|y0| 161. Set |v:=v+y|; i.e., move down |y| units. With luck,
    this parameterless command will usually suffice, because the same kind of motion
    will occur several times in succession; the following commands explain how
    |y| gets particular values."""
    y0 = 161
    """|y1| 162 |a[1]|. Set |y:=a| and |v:=v+a|. The value of |a| is a
    signed quantity in two's complement notation, |-128<=a<128|. This command
    changes the current |y|~spacing and moves down by |a|."""
    y1 = 162
    """|@!y2| 163 |a[2]|. Same as |y1|, but |a| is two bytes long,
    |-32768<=a<32768|."""
    y2 = 163
    """|@!y3| 164 |a[3]|. Same as |y1|, but |a| is three bytes long,
    |@t$-2^{23}$@><=a<@t$2^{23}$@>|."""
    y3 = 164
    """|@!y4| 165 |a[4]|. Same as |y1|, but |a| is four bytes long,
    |@t$-2^{31}$@><=a<@t$2^{31}$@>|."""
    y4 = 165

    """|z0| 166. Set |v:=v+z|; i.e., move down |z| units. The `|z|' commands
    are like the `|y|' commands except that they involve |z| instead of |y|."""
    z0 = 166
    """|z1| 167 |a[1]|. Set |z:=a| and |v:=v+a|. The value of |a| is a
    signed quantity in two's complement notation, |-128<=a<128|. This command
    changes the current |z|~spacing and moves down by |a|."""
    z1 = 167
    """|@!z2| 168 |a[2]|. Same as |z1|, but |a| is two bytes long,
    |-32768<=a<32768|."""
    z2 = 168
    """|@!z3| 169 |a[3]|. Same as |z1|, but |a| is three bytes long,
    |@t$-2^{23}$@><=a<@t$2^{23}$@>|."""
    z3 = 169
    """|@!z4| 170 |a[4]|. Same as |z1|, but |a| is four bytes long,
    |@t$-2^{31}$@><=a<@t$2^{31}$@>|."""
    z4 = 170

    """|fnt_num_0| 171. Set |f:=0|. Font 0 must previously have been
    defined by a fnt_def instruction, as explained below."""
    fnt_num_0 = 171
    """fnt_num_1 through fnt_num_63} (opcodes 172 to 234). Set
    |f:=1|, \\dots, \\hbox{|f:=63|}, respectively."""
    fnt_num_1 = 172
    fnt_num_2 = 173
    fnt_num_3 = 174
    fnt_num_4 = 175
    fnt_num_5 = 176
    fnt_num_6 = 177
    fnt_num_7 = 178
    fnt_num_8 = 179
    fnt_num_9 = 180
    fnt_num_10 = 181
    fnt_num_11 = 182
    fnt_num_12 = 183
    fnt_num_13 = 184
    fnt_num_14 = 185
    fnt_num_15 = 186
    fnt_num_16 = 187
    fnt_num_17 = 188
    fnt_num_18 = 189
    fnt_num_19 = 190
    fnt_num_20 = 191
    fnt_num_21 = 192
    fnt_num_22 = 193
    fnt_num_23 = 194
    fnt_num_24 = 195
    fnt_num_25 = 196
    fnt_num_26 = 197
    fnt_num_27 = 198
    fnt_num_28 = 199
    fnt_num_29 = 200
    fnt_num_30 = 201
    fnt_num_31 = 202
    fnt_num_32 = 203
    fnt_num_33 = 204
    fnt_num_34 = 205
    fnt_num_35 = 206
    fnt_num_36 = 207
    fnt_num_37 = 208
    fnt_num_38 = 209
    fnt_num_39 = 210
    fnt_num_40 = 211
    fnt_num_41 = 212
    fnt_num_42 = 213
    fnt_num_43 = 214
    fnt_num_44 = 215
    fnt_num_45 = 216
    fnt_num_46 = 217
    fnt_num_47 = 218
    fnt_num_48 = 219
    fnt_num_49 = 220
    fnt_num_50 = 221
    fnt_num_51 = 222
    fnt_num_52 = 223
    fnt_num_53 = 224
    fnt_num_54 = 225
    fnt_num_55 = 226
    fnt_num_56 = 227
    fnt_num_57 = 228
    fnt_num_58 = 229
    fnt_num_59 = 230
    fnt_num_60 = 231
    fnt_num_61 = 232
    fnt_num_62 = 233
    fnt_num_63 = 234

    """|fnt1| 235 |k[1]|. Set |f:=k|. TeX82 uses this command for font
    numbers in the range |64<=k<256|."""
    fnt1 = 235
    """g|@!fnt2| 236 |k[2]|. Same as |fnt1|, except that |k|~is two
    bytes long, so it is in the range |0<=k<65536|. TeX82 never generates this
    command, but large font numbers may prove useful for specifications of
    color or texture, or they may be used for special fonts that have fixed
    numbers in some external coding scheme."""
    fnt2 = 236
    """|@!fnt3| 237 |k[3]|. Same as |fnt1|, except that |k|~is three
    bytes long, so it can be as large as $2^{24}-1$."""
    fnt3 = 237
    """|@!fnt4| 238 |k[4]|. Same as |fnt1|, except that |k|~is four
    bytes long; this is for the really big font numbers (and for the negative ones)."""
    fnt4 = 238

    """|xxx1| 239 |k[1]| |x[k]|. This command is undefined in
    general; it functions as a $(k+2)$-byte |nop| unless special .DVI-reading
    programs are being used. TeX82 generates |xxx1| when a short enough
    \\special appears, setting |k| to the number of bytes being sent. It
    is recommended that |x| be a string having the form of a keyword followed
    by possible parameters relevant to that keyword."""
    xxx1 = 239
    """|@!xxx2| 240 |k[2]| |x[k]|. Like |xxx1|, but |0<=k<65536|."""
    xxx2 = 240
    """|@!xxx3| 241 |k[3]| |x[k]|. Like |xxx1|, but |0<=k<2^{24}|."""
    xxx3 = 241
    """|xxx4| 242 |k[4]| |x[k]|. Like |xxx1|, but |k| can be ridiculously
    large. TeX82 uses |xxx4| when sending a string of length 256 or more."""
    xxx4 = 242

    long_char = 242

    """|fnt_def1| 243 |k[1]| |c[4]| |s[4]| |d[4]| |a[1]| |l[1]| |n[a+l]|.
    Define font |k|, where |0<=k<256|; font definitions will be explained shortly."""
    fnt_def1 = 243

    """|@!fnt_def2| 244 |k[2]| |c[4]| |s[4]| |d[4]| |a[1]| |l[1]| |n[a+l]|.
    Define font |k|, where |0<=k<65536|."""
    fnt_def2 = 244

    """|@!fnt_def3| 245 |k[3]| |c[4]| |s[4]| |d[4]| |a[1]| |l[1]| |n[a+l]|.
    Define font |k|, where |0<=k<@t$2^{24}$@>|."""
    fnt_def3 = 245

    """|@!fnt_def4| 246 |k[4]| |c[4]| |s[4]| |d[4]| |a[1]| |l[1]| |n[a+l]|.
    Define font |k|, where |@t$-2^{31}$@><=k<@t$2^{31}$@>|."""
    fnt_def4 = 246

    """|pre| 247 |i[1]| |num[4]| |den[4]| |mag[4]| |k[1]| |x[k]|.
    Beginning of the preamble; this must come at the very beginning of the
    file. Parameters |i|, |num|, |den|, |mag|, |k|, and |x| are explained below."""
    pre = 247

    """|post| 248. Beginning of the postamble, see below."""
    post = 248

    """|post_post| 249. Ending of the postamble, see below."""
    post_post = 249

    """Commands 250--255 are undefined at the present time."""


class VFError(Exception):
    pass


class VFFile:
    real_font: str | None
    fnt_defs: dict[int, str]
    chars: dict[int, tuple[int, list[int | DVICmd]]]

    def __init__(self, filename: str | Path):
        self.file = open(filename, "rb")
        self.cmd = DVICmd.nop

    def next_cmd(self):
        self.cmd = DVICmd(self.read_byte())

    def read_byte(self) -> int:
        byte = self.file.read(1)
        if not byte:
            raise VFError("unexpected EOF or error")
        return byte[0]

    def read_bytes(self, cnt: int) -> bytes:
        byte = self.file.read(cnt)
        if len(byte) != cnt:
            raise VFError("unexpected EOF or error")
        return byte

    def read_signed(self, k: int) -> int:
        """Read k bytes as singed int"""
        assert 0 < k <= 4
        i = self.read_byte()
        if i >= 128:
            i -= 256
        for _ in range(k - 1):
            i = i * 256 + self.read_byte()
        return i

    def read_unsigned(self, k: int) -> int:
        """Read k bytes as unsinged int"""
        assert 0 < k <= 4
        i = self.read_byte()
        if k == 4 and i >= 128:
            raise VFError("number too big")
        for _ in range(k - 1):
            i = (i << 8) + self.read_byte()
        return i

    def process_preamble(self) -> None:
        self.next_cmd()
        if self.cmd != DVICmd.pre:
            raise VFError("PRE command expected")
        if self.read_byte() != vf_id:
            raise VFError("wrong id byte")
        cmd_length = self.read_byte()
        for _ in range(cmd_length):
            self.read_byte()  # skip the comment
        _chksum = self.read_bytes(4)
        self.font_dsize = self.read_signed(4) // 32

    def process_font_definitions(self) -> None:
        self.real_font = None
        self.fnt_defs = {}
        self.next_cmd()
        while DVICmd.fnt_def1 <= self.cmd <= DVICmd.fnt_def4:
            key = self.read_unsigned(self.cmd - DVICmd.fnt_def1.value + 1)
            font_name = self.read_font()
            self.fnt_defs[key] = font_name
            if self.real_font is None:
                self.real_font = font_name
            # print(f"Font definition: {key} -> {font_name}")
            self.next_cmd()
        # print(f"Font definitions: {self.fnt_defs}, next command: {self.cmd}")

    def read_font(self) -> str:
        _cs = self.read_bytes(4)
        _fs = self.read_signed(4)
        _ds = self.read_signed(4) // 32
        path_len = self.read_byte()
        fontname_len = self.read_byte()
        _font_path = self.read_bytes(path_len)
        font_name = self.read_bytes(fontname_len)
        # print(f"Font: {font_name} at {font_path} with {fs}pt with design size {ds}pt and checksum {cs}")
        return font_name.decode()

    def process_character_packet(self) -> tuple[int, int, list[int | DVICmd]]:
        if self.cmd == DVICmd.long_char:
            packet_length = self.read_unsigned(4)
            cc = self.read_unsigned(4)
            tfm_width = self.read_signed(4)
            # print(f"Long character: {cc} with width {tfm_width}")
        else:
            packet_length = self.cmd
            cc = self.read_byte()
            tfm_width = self.read_unsigned(3)
            # print(f"Short character: {cc} with width {tfm_width}")
        if packet_length < 0 or packet_length > vf_max_packet_length:
            raise VFError("invalid packet length")
        c = []
        stack_level = 0
        while packet_length > 0:
            self.next_cmd()
            packet_length -= 1
            # DVI commands
            if DVICmd.set_char_0 <= self.cmd <= DVICmd.set_char_127:
                cmd_length = 0
            elif (
                DVICmd.fnt_num_0 <= self.cmd <= DVICmd.fnt_num_63
                or DVICmd.fnt1 <= self.cmd <= DVICmd.fnt4
            ):
                if self.cmd >= DVICmd.fnt1:
                    k = self.read_unsigned(self.cmd.value - DVICmd.fnt1.value + 1)
                    packet_length -= self.cmd.value - DVICmd.fnt1.value + 1
                else:
                    k = self.cmd.value - DVICmd.fnt_num_0.value
                if k >= 256:
                    raise VFError("invalid font number")
                assert k in self.fnt_defs, f"Font {k} not defined"
                if k <= 63:
                    c.append(DVICmd(DVICmd.fnt_num_0.value + k))
                else:
                    c.append(DVICmd.fnt1)
                    c.append(k)
                continue
            elif self.cmd in (DVICmd.set_rule, DVICmd.put_rule):
                cmd_length = 8
            elif DVICmd.set1 <= self.cmd <= DVICmd.set4:
                k = self.read_unsigned(self.cmd.value - DVICmd.set1.value + 1)
                cmd_length = self.cmd.value - DVICmd.set1.value + 1
                packet_length -= cmd_length
                c.append(DVICmd.set1)
                c.append(k)
                continue
            elif DVICmd.put1 <= self.cmd <= DVICmd.put4:
                cmd_length = self.cmd.value - DVICmd.put1.value + 1
            elif DVICmd.right1 <= self.cmd <= DVICmd.right4:
                cmd_length = self.cmd.value - DVICmd.right1.value + 1
            elif DVICmd.w1 <= self.cmd <= DVICmd.w4:
                cmd_length = self.cmd.value - DVICmd.w1.value + 1
            elif DVICmd.x1 <= self.cmd <= DVICmd.x4:
                cmd_length = self.cmd.value - DVICmd.x1.value + 1
            elif DVICmd.down1 <= self.cmd <= DVICmd.down4:
                cmd_length = self.cmd.value - DVICmd.down1.value + 1
            elif DVICmd.y1 <= self.cmd <= DVICmd.y4:
                cmd_length = self.cmd.value - DVICmd.y1.value + 1
            elif DVICmd.z1 <= self.cmd <= DVICmd.z4:
                cmd_length = self.cmd.value - DVICmd.z1.value + 1
            elif DVICmd.xxx1 <= self.cmd <= DVICmd.xxx4:
                cmd_length = self.cmd.value - DVICmd.xxx1.value + 1
                packet_length -= cmd_length
                cmd_length = self.read_unsigned(cmd_length)
                if cmd_length > vf_max_packet_length:
                    raise VFError("invalid packet length")
                if cmd_length < 0:
                    raise VFError("invalid packet length")
                c.append(DVICmd.xxx1)
                c.append(cmd_length)
                self.cmd = DVICmd.nop
            elif self.cmd in (DVICmd.w0, DVICmd.x0, DVICmd.y0, DVICmd.z0, DVICmd.nop):
                cmd_length = 0
            elif self.cmd in (DVICmd.push, DVICmd.pop):
                cmd_length = 0
                if self.cmd == DVICmd.push:
                    if stack_level == vf_stack_size:
                        raise VFError("stack overflow")
                    stack_level += 1
                else:
                    if stack_level == 0:
                        raise VFError("stack underflow")
                    stack_level -= 1
            else:
                raise VFError(f"unknown command {self.cmd}")
            # End of DVI cmds
            if self.cmd != DVICmd.nop:
                c.append(self.cmd)
            packet_length -= cmd_length
            while cmd_length > 0:
                cmd_length -= 1
                c.append(self.read_byte())
        assert stack_level == 0, f"stack_level: {stack_level}"
        assert packet_length == 0, f"packet_length: {packet_length}"
        # print(f"Character: {cc} with width {tfm_width} and commands {c}")
        return cc, tfm_width, c

    def process_vf_file(self) -> None:
        # print(f"Processing VF file {self.file.name}")
        self.process_preamble()
        self.process_font_definitions()
        self.chars = {}
        while self.cmd <= DVICmd.long_char:
            cc, tfm_width, cmds = self.process_character_packet()
            self.chars[cc] = (tfm_width, cmds)
            self.next_cmd()
        if self.cmd != DVICmd.post:
            raise VFError("POST command expected")

    def close(self) -> None:
        self.file.close()


def compute_vf_mapping(vf: VFFile) -> dict[int, list[tuple[bytes, int]]]:
    # The actual remapping. [char_idx] = [(real_font_name, idx), ...]
    charmap: dict[int, list[tuple[bytes, int]]] = {}
    for dst_idx, (tfm_width, cmds) in vf.chars.items():
        # print(f"[0x{idx:02X}->0x{dst_idx:02X}]: width: {tfm_width}, cmds: {cmds}")
        fnt = vf.real_font
        i = 0
        cur_chars = []
        while i < len(cmds):
            cmd = cmds[i]
            assert isinstance(cmd, DVICmd), f"{cmds}"
            if cmd >= DVICmd.fnt_num_0 and cmd <= DVICmd.fnt_num_63:
                fnt = vf.fnt_defs[int(cmd.value - DVICmd.fnt_num_0.value)]
                i += 1
            elif cmd == DVICmd.fnt1:
                fnt = vf.fnt_defs[int(cmds[i + 1])]
                i += 2
            elif cmd >= DVICmd.set_char_0 and cmd <= DVICmd.set_char_127:
                cur_chars.append((fnt, int(cmd.value - DVICmd.set_char_0.value)))
                i += 1
            elif cmd == DVICmd.set1:
                cur_chars.append((fnt, int(cmds[i + 1])))
                i += 2
            else:
                i += 1
                while i < len(cmds) and not isinstance(cmds[i], DVICmd):
                    i += 1
        if cur_chars:
            charmap[dst_idx] = cur_chars
        # print(f" map 0x{dst_idx:02X} -> {', '.join(f'{fnt.decode()}[0x{idx:02X}]' for fnt, idx in cur_chars)}")
    return charmap


def get_vf_remapping(vf_file: str | Path) -> tuple[str | None, dict[int, list[tuple[str, int]]]]:
    vf = VFFile(vf_file)
    try:
        vf.process_vf_file()
        return vf.real_font, compute_vf_mapping(vf)
    finally:
        vf.close()


if __name__ == "__main__":
    # Example usage
    real_font, mapping = get_vf_remapping(
        "/home/lvoegtle/.local/share/enroot/pdftex/usr/share/texmf-dist/fonts/vf/adobe/times/ptmr8c.vf"
    )
    # vf = VFFile("/home/lvoegtle/.local/share/enroot/pdftex/usr/share/texmf-dist/fonts/vf/adobe/times/ptmr8t.vf")
    # vf = VFFile("/home/lvoegtle/.local/share/enroot/pdftex/usr/share/texmf-dist/fonts/vf/adobe/times/ptmr8c.vf")
    # vf = VFFile(
    # "/home/lvoegtle/.local/share/enroot/pdftex/usr/share/texmf-dist/fonts/vf/public/libertinus-type1/LibertinusSerifInitials-Regular-tlf-ts1.vf"
    # )
    for idx, chars in mapping.items():
        print(f"0x{idx:02X} -> {', '.join(f'{fnt}[0x{idx:02X}]' for fnt, idx in chars)}")
