"""Code references -> source locations.

  "pkg.mod:func"                  the whole function (or class, method, property: "pkg.mod:Class.method")
  "pkg.mod:Class.field"           a line of a class body (a dataclass field), or "pkg.mod:CONSTANT" of the module
                                  (the whole statement; "pkg.mod:CONSTANT#text" as below, within it)
  "pkg.mod:func#text"             from the line holding `text` to the line before the next comment line at the same
                                  or a smaller indentation (e.g. "#step 5" for the block under "# step 5: ...")
  "pkg.mod:func#from text..to text"   from the line holding `from text` to the line holding `to text`, inclusive
"""
from __future__ import annotations

import ast
import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path


class RefError(Exception):
    pass


@dataclass
class Location:
    ref: str
    file: str           # absolute
    relpath: str        # from the top-level package's parent folder
    start: int          # 1-based, inclusive
    end: int
    source: str


def resolve(ref: str) -> Location:
    module_name, _, rest = ref.partition(":")
    qualname, _, anchor = rest.partition("#")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise RefError(f"{ref}: cannot import {module_name} ({e})") from e
    file = inspect.getsourcefile(module)
    assert file
    relpath = shortpath(module_name, file)

    obj: object = module
    parts = qualname.split(".") if qualname else []
    for n, part in enumerate(parts):
        if inspect.isclass(obj) and isinstance(inspect.getattr_static(obj, part, None), property):
            obj = inspect.getattr_static(obj, part).fget                # a property: its getter
            continue
        if hasattr(obj, part) and (inspect.isclass(getattr(obj, part)) or inspect.isfunction(getattr(obj, part))
                                   or inspect.ismethod(getattr(obj, part))):
            obj = getattr(obj, part)
            continue
        if n == len(parts) - 1 and (inspect.ismodule(obj) or inspect.isclass(obj)):
            loc = _member_line(ref, obj, part, file, relpath)           # a field or a constant
            if anchor:
                lines, start = _anchored(ref, loc.source.splitlines(keepends=True), loc.start, anchor)
                loc = Location(ref, file, relpath, start, start + len(lines) - 1, "".join(lines))
            return loc
        raise RefError(f"{ref}: no function or class {part!r} in {qualname or module_name}")
    try:
        if obj is module:
            lines, start = inspect.getsourcelines(module)
            start = 1
        else:
            lines, start = inspect.getsourcelines(obj)      # type: ignore[arg-type]
            # the function's own file: it may be imported into the module
            file = inspect.getsourcefile(obj) or file       # type: ignore[arg-type]
            relpath = shortpath(module_name, file)
    except (OSError, TypeError) as e:
        raise RefError(f"{ref}: no source ({e})") from e
    if anchor:
        lines, start = _anchored(ref, lines, start, anchor)
    return Location(ref, file, relpath, start, start + len(lines) - 1, "".join(lines))


def shortpath(module_name: str, file: str) -> str:
    """A file's path from the parent folder of the module's top-level package (e.g. "methembed/core/model.py")."""
    try:
        top = importlib.import_module(module_name.split(".")[0])
        root = Path(inspect.getsourcefile(top) or file).parent.parent
    except (ImportError, TypeError):
        return file
    return str(Path(file).relative_to(root)) if Path(file).is_relative_to(root) else file


def _member_line(ref: str, owner: object, name: str, file: str, relpath: str) -> Location:
    lines, start = inspect.getsourcelines(owner)            # type: ignore[arg-type]
    if inspect.ismodule(owner):
        start = 1
    for n, line in enumerate(lines):
        s = line.strip()
        if s.startswith(name) and s[len(name):].lstrip()[:1] in (":", "="):
            end = n
            if inspect.ismodule(owner):                                 # the whole statement: a dict over lines
                tree = ast.parse("".join(lines))
                end = next((st.end_lineno - 1 for st in tree.body if st.lineno - 1 == n and st.end_lineno), n)
            return Location(ref, file, relpath, start + n, start + end, "".join(lines[n:end + 1]))
    raise RefError(f"{ref}: no {name!r} defined in {getattr(owner, '__name__', owner)}")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _anchored(ref: str, lines: list[str], start: int, anchor: str) -> tuple[list[str], int]:
    first_text, sep, last_text = anchor.partition("..")
    hits = [n for n, line in enumerate(lines) if first_text in line]
    if not hits:
        raise RefError(f"{ref}: no line holds {first_text!r}")
    if len(hits) > 1:
        raise RefError(f"{ref}: {len(hits)} lines hold {first_text!r}; make the anchor unique")
    a = hits[0]
    if sep:
        ends = [n for n in range(a, len(lines)) if last_text in lines[n]]
        if not ends:
            raise RefError(f"{ref}: no line after {first_text!r} holds {last_text!r}")
        b = ends[0]
    else:
        depth = _indent(lines[a])
        b = a
        for n in range(a + 1, len(lines)):
            s = lines[n].strip()
            if s and _indent(lines[n]) < depth:
                break
            if s.startswith("#") and _indent(lines[n]) <= depth:
                break
            b = n
        while b > a and not lines[b].strip():
            b -= 1
    return lines[a:b + 1], start + a
