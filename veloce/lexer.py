"""Lexer for the Veloce DSL.

Produces a flat list of Token objects.  Newlines are significant at the
statement level (Veloce uses newline / indentation-free statement
separation), so the lexer emits explicit NEWLINE tokens but ignores blank
lines and `#` comments.
"""

from .errors import LexError

# Multi-char operators, longest first.
OPERATORS = [
    "->",
    "==",
    "!=",
    "<=",
    ">=",
    "+=",
    "-=",
    "*=",
    "/=",
    "<-",
    "**",
    "&&",
    "||",
    "//",  # integer division
    "+",
    "-",
    "*",
    "/",
    "%",
    "<",
    ">",
    "=",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    ",",
    ":",
    ".",
    "@",
    ";",
]

KEYWORDS = {
    "fn",
    "let",
    "mut",
    "rw",
    "const",
    "return",
    "if",
    "else",
    "while",
    "for",
    "in",
    "break",
    "continue",
    "true",
    "false",
    "module",
    "extern",
}

DTYPE_NAMES = {"trit", "fp16", "fp32", "f32", "fp8", "i2", "i4", "i8", "i32",
               "i64", "u8", "bool"}


class Token:
    __slots__ = ("kind", "value", "line", "col")

    def __init__(self, kind, value, line, col):
        self.kind = kind
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self):
        return f"Token({self.kind!r}, {self.value!r}, {self.line}:{self.col})"


def tokenize(src):
    tokens = []
    i = 0
    line = 1
    col = 1
    n = len(src)

    def advance(k=1):
        nonlocal i, line, col
        for _ in range(k):
            if i < n and src[i] == "\n":
                line += 1
                col = 1
            else:
                col += 1
            i += 1

    while i < n:
        ch = src[i]

        # Whitespace (but not newline)
        if ch in " \t\r":
            advance()
            continue

        # Comment
        if ch == "#":
            while i < n and src[i] != "\n":
                advance()
            continue

        # Newline
        if ch == "\n":
            tokens.append(Token("NEWLINE", "\n", line, col))
            advance()
            continue

        # String literal
        if ch == '"':
            start_line, start_col = line, col
            advance()
            buf = []
            while i < n and src[i] != '"':
                if src[i] == "\\":
                    advance()
                    esc = src[i] if i < n else ""
                    buf.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(esc, esc))
                    advance()
                else:
                    buf.append(src[i])
                    advance()
            if i >= n:
                raise LexError("unterminated string literal", start_line, start_col)
            advance()  # closing quote
            tokens.append(Token("STRING", "".join(buf), start_line, start_col))
            continue

        # Number literal
        if ch.isdigit():
            start_line, start_col = line, col
            start = i
            is_float = False
            while i < n and (src[i].isdigit() or src[i] == "_"):
                advance()
            if i < n and src[i] == "." and i + 1 < n and src[i + 1].isdigit():
                is_float = True
                advance()
                while i < n and (src[i].isdigit() or src[i] == "_"):
                    advance()
            if i < n and src[i] in "eE":
                is_float = True
                advance()
                if i < n and src[i] in "+-":
                    advance()
                while i < n and src[i].isdigit():
                    advance()
            text = src[start:i].replace("_", "")
            val = float(text) if is_float else int(text)
            tokens.append(Token("FLOAT" if is_float else "INT", val, start_line, start_col))
            continue

        # Identifier / keyword
        if ch.isalpha() or ch == "_":
            start_line, start_col = line, col
            start = i
            while i < n and (src[i].isalnum() or src[i] == "_"):
                advance()
            word = src[start:i]
            if word in KEYWORDS:
                tokens.append(Token("KW", word, start_line, start_col))
            elif word in DTYPE_NAMES:
                tokens.append(Token("DTYPE", word, start_line, start_col))
            else:
                tokens.append(Token("IDENT", word, start_line, start_col))
            continue

        # Operators
        matched = None
        for op in OPERATORS:
            if src.startswith(op, i):
                matched = op
                break
        if matched:
            tokens.append(Token("OP", matched, line, col))
            advance(len(matched))
            continue

        raise LexError(f"unexpected character {ch!r}", line, col)

    tokens.append(Token("EOF", None, line, col))
    return tokens
