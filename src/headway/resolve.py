"""Code references -> source locations. Python, by module (below); a file in any language, by path (files.py).

  "pkg.mod:func"                  the whole function (or class, method, property: "pkg.mod:Class.method")
  "pkg.mod:Class.field"           a line of a class body (a dataclass field), or "pkg.mod:CONSTANT" of the module
                                  (the whole statement; "pkg.mod:CONSTANT#text" as below, within it)
  "pkg.mod:func#text"             from the line holding `text` to the line before the next comment line at the same
                                  or a smaller indentation, or the next line indented less (e.g. "#step 5" for the
                                  block under "# step 5: ...")
  "pkg.mod:func#from text..to text"   from the line holding `from text` to the line holding `to text`, inclusive
"""
from __future__ import annotations

import ast
import functools
import importlib
import inspect
import linecache
from collections.abc import Callable, Iterator
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
    lang: str = "python"    # tree-sitter's name for the language


def resolve(ref: str, where: list[Path] | None = None) -> Location:
    """A reference -> its location: a Python module's ("pkg.mod:func"), or a file's in any language
    ("R/score.R:func", files.py), looked for under `where` (files.bases)."""
    from headway import files
    found = files.find(ref, where or [Path.cwd()])
    if found:
        return files.resolve(ref, *found)
    module_name, _, rest = ref.partition(":")
    qualname, _, anchor = rest.partition("#")
    try:
        module = importlib.import_module(module_name)
    except (ImportError, TypeError, ValueError) as e:   # TypeError: "../x.R" read as a relative import
        if files.looks_like_file(ref):
            raise RefError(f"{ref}: no file {files._split(ref)[0]} under {', '.join(map(str, where or [Path.cwd()]))}") from e
        raise RefError(f"{ref}: cannot import {module_name} ({e})") from e
    try:
        file = inspect.getsourcefile(module)
    except TypeError:                                   # a built-in module
        file = None
    if not file:
        raise RefError(f"{ref}: {module_name} has no Python source (a built-in or compiled module)")
    relpath = shortpath(module_name, file)

    obj: object = module
    parts = qualname.split(".") if qualname else []
    for n, part in enumerate(parts):
        static = inspect.getattr_static(obj, part, None) if inspect.isclass(obj) else None
        if isinstance(static, property):
            obj = static.fget                                           # a property: its getter
            continue
        if isinstance(static, functools.cached_property):
            obj = static.func                                           # a cached property: its function
            continue
        value = getattr(obj, part) if hasattr(obj, part) else None
        if callable(value):
            # a function behind a decorator object (functools.cache, jax.jit) by its __wrapped__
            try:
                value = inspect.unwrap(value)
            except ValueError:                          # an object that makes up every attribute, __wrapped__ too
                pass
        if inspect.isclass(value) or inspect.isfunction(value) or inspect.ismethod(value):
            obj = value
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


def package_root(file: str | Path) -> Path | None:
    """The folder of the top-level package holding a file: the topmost of the folders of __init__.py files above it;
    None if the file's own folder is not a package."""
    folder = Path(file).parent
    if not (folder / "__init__.py").exists():
        return None
    while (folder.parent / "__init__.py").exists():
        folder = folder.parent
    return folder


def _member_line(ref: str, owner: object, name: str, file: str, relpath: str) -> Location:
    """The statement assigning `name` in a module or a class body (a constant, a field), also under an if or a try
    there, and as one of several names (A, B = ...): the first line of a field, the whole statement of a module's
    constant (a dict over lines, say). The whole file is parsed, so a nested class need not dedent cleanly."""
    try:
        if inspect.ismodule(owner):
            lines, _ = inspect.getsourcelines(owner)
        else:
            file = inspect.getsourcefile(owner) or file                # type: ignore[arg-type]
            relpath = shortpath(owner.__module__, file)                 # type: ignore[attr-defined]
            lines = linecache.getlines(file)
            _, first = inspect.getsourcelines(owner)                   # type: ignore[arg-type]
    except (OSError, TypeError) as e:           # a class made by a call (Enum("C", ...), namedtuple) or built in
        raise RefError(f"{ref}: no source ({e})") from e
    try:
        tree = ast.parse("".join(lines))
    except SyntaxError as e:
        raise RefError(f"{ref}: cannot parse {file} ({e})") from e
    body: list[ast.stmt] = tree.body
    if not inspect.ismodule(owner):
        cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == owner.__name__  # type: ignore[attr-defined]
                    and min([n.lineno, *(d.lineno for d in n.decorator_list)]) == first), None)
        body = cls.body if cls else []
    for st in _statements(body):
        targets = st.targets if isinstance(st, ast.Assign) else [st.target] if isinstance(st, ast.AnnAssign) else []
        if name in {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}:
            end = (st.end_lineno or st.lineno) if inspect.ismodule(owner) else st.lineno
            return Location(ref, file, relpath, st.lineno, end, "".join(lines[st.lineno - 1:end]))
    raise RefError(f"{ref}: no {name!r} defined in {getattr(owner, '__name__', owner)}")


def _statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """The statements of a body, and those of the ifs and trys in it (not of functions or classes), in order."""
    for st in body:
        yield st
        if isinstance(st, ast.If):
            yield from _statements(st.body + st.orelse)
        elif isinstance(st, (ast.Try, ast.TryStar)):
            yield from _statements(st.body + [s for h in st.handlers for s in h.body] + st.orelse + st.finalbody)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _anchored(ref: str, lines: list[str], start: int, anchor: str,
              comment: Callable[[int], bool] | None = None) -> tuple[list[str], int]:
    """The lines an anchor picks out of `lines` (which start at line `start`). comment(n): is lines[n] a comment."""
    if comment is None:
        comment = lambda n: lines[n].strip().startswith("#")     # noqa: E731
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
            if comment(n) and _indent(lines[n]) <= depth:
                break
            b = n
        while b > a and not lines[b].strip():
            b -= 1
    return lines[a:b + 1], start + a
