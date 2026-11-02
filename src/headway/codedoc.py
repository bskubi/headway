"""What the code says about itself: a step's code reference -> the first paragraph of its docstring or comments,
shown as the step's description until the spec gives one. Read from the files; nothing is run.

The first of:
  1. a Python function's or class's docstring;
  2. the comment block directly above the referenced code (R's #' roxygen, Rust's ///, a /** */ block, # or //
     lines, ...);
  3. the comment lines the referenced code opens with (the "# step 2: ..." an anchor points at), or that open the
     body after its first line.
Comment lines are recognized from tree-sitter's syntax tree where there is a parser (files._comment), so no
language needs code of its own; Python's are lines starting with a comment character.
"""
from __future__ import annotations

import ast
import re
import textwrap
from collections.abc import Callable
from pathlib import Path

from headway import files
from headway.resolve import Location

MARKER = re.compile(r"^\s*(?:#'|#+!?|///?!?|--+|%+|;+|/\*+!?|\*+/?(?!\S))\s?")
TAG = re.compile(r"^\s*[@\\](?:param|return|returns|export|examples?|seealso|rdname|name|title|details|brief)\b")


def describe(loc: Location) -> str | None:
    """The first paragraph of what the code says about the referenced code, or None."""
    try:
        lines = Path(loc.file).read_text().splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    if loc.lang == "python" and (doc := _docstring(loc.source)):
        return doc
    # Python's comment lines by their first characters; other languages' from their syntax tree
    parsed = None if loc.lang == "python" else files._parsed(loc.file, Path(loc.file).stat().st_mtime, loc.lang)
    is_comment = files._comment(parsed, lines)
    first, last = loc.start - 1, loc.end - 1                # 0-based
    above = _run(lines, first - 1, -1, is_comment)
    if above:
        return _paragraph([lines[n] for n in reversed(above)])
    for start in (first, first + 1):                        # the block's own heading, or its body's
        if start <= last:
            inside = _run(lines, start, 1, is_comment, stop=last)
            if inside:
                return _paragraph([lines[n] for n in inside])
    return None


def _docstring(source: str) -> str | None:
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None
    if len(tree.body) == 1 and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        doc = ast.get_docstring(tree.body[0])
        return _paragraph(doc.splitlines(), markers=False) if doc else None
    return None


def _run(lines: list[str], n: int, step: int, is_comment: Callable[[int], bool], stop: int | None = None) -> list[int]:
    """The comment lines next to each other from line n on, going `step` (1 down, -1 up)."""
    out = []
    while 0 <= n < len(lines) and (stop is None or n <= stop) and lines[n].strip() and is_comment(n):
        if n == 0 and lines[n].startswith("#!"):           # a script's interpreter line
            break
        out.append(n)
        n += step
    return out


def _paragraph(lines: list[str], markers: bool = True) -> str | None:
    """The first paragraph: comment markers stripped, rules (# -----) and editor lines (-*- coding -*-) dropped,
    ending at a blank line or a documentation tag (@param ...)."""
    words: list[str] = []
    for line in lines:
        text = MARKER.sub("", line) if markers else line
        text = text.rstrip().removesuffix("*/").strip()
        if TAG.match(text):
            break
        if not text:
            if words:
                break
            continue
        if not any(c.isalnum() for c in text) or "-*-" in text:
            continue
        words.append(text)
    return " ".join(words) or None
