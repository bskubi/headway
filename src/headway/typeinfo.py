"""What a type checker infers about the code: the kind of value each name and attribute holds, read from mypy (its
types for every expression, from the code's annotations and the libraries' stubs, followed through calls,
attributes and loops). Used only when type hints are taken as assumptions (headway init --hints, Headway(...,
hints=True)): Python does not check them when the code runs.

Kinds: "array" (numpy.ndarray), "scalar" (a Python or numpy number), "list", "dict", "set", "slice", or
"class:pkg.mod.Name" for a class of the code's own. Keyed by (file, line, column, name), as Python's ast gives them.
The kinds of a package's files are cached (~/.cache/headway/types), and found again while no file of it changes.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path

Key = tuple[str, int, int, str]                     # (file, line, column, name)


class Types:
    def __init__(self, kinds: dict[Key, str], problem: str = ""):
        self.kinds = kinds
        self.problem = problem                      # why there are none (no mypy, it failed), if so

    def kind(self, file: str | Path, line: int, column: int, name: str) -> str | None:
        return self.kinds.get((str(file), line, column, name))

    @classmethod
    def of_package(cls, folder: Path) -> Types:
        """The kinds of the names in a package's files (its folder holds __init__.py); empty if mypy fails."""
        files = sorted(f for f in folder.rglob("*.py") if "__pycache__" not in f.parts)
        stamp = hashlib.sha256(json.dumps([_version()] + [(str(f), f.stat().st_mtime_ns, f.stat().st_size)
                                                          for f in files]).encode()).hexdigest()[:20]
        cache = _cache_dir() / f"{stamp}.json"
        try:
            rows = json.loads(cache.read_text())
            return cls({(r[0], r[1], r[2], r[3]): r[4] for r in rows})
        except (OSError, ValueError):
            pass
        kinds, problem = _infer(folder, files)
        if not problem:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps([[*k, v] for k, v in kinds.items()]))
            except OSError:
                pass
        return cls(kinds, problem)


def _infer(folder: Path, files: list[Path]) -> tuple[dict[Key, str], str]:
    try:
        from mypy import build
        from mypy.modulefinder import BuildSource
        from mypy.options import Options
        from mypy.traverser import NameAndMemberCollector
    except ImportError:
        return {}, "mypy is not installed"
    root = folder.parent
    modules = {".".join(f.relative_to(root).with_suffix("").parts).removesuffix(".__init__"): f for f in files}
    o = Options()
    o.export_types = True
    o.ignore_missing_imports = True             # a library without types (polars, say): its values are not known
    o.incremental = False
    o.check_untyped_defs = True                 # infer inside functions without annotations too
    o.follow_imports = "silent"
    o.preserve_asts = True
    o.mypy_path = [str(root)]
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = build.build(sources=[BuildSource(str(f), m, None) for m, f in modules.items()], options=o)
    except Exception as crash:                  # noqa: BLE001 (a crash of the checker: no types, as without it)
        return {}, f"mypy failed: {crash}"
    kinds: dict[Key, str] = {}
    for m, f in modules.items():
        state = result.graph.get(m)
        if state is None or state.tree is None:
            continue
        v = NameAndMemberCollector()
        state.tree.accept(v)
        for e in [*v.name_exprs, *v.member_exprs]:
            t = result.types.get(e)
            k = _kind(t) if t is not None else None
            if k:
                kinds[(str(f), e.line, e.column, e.name)] = k                 # type: ignore[attr-defined]
    return kinds, ""


def _kind(t: object) -> str | None:
    from mypy.types import Instance, LiteralType, TupleType, TypeVarType, UnionType, get_proper_type
    p = get_proper_type(t)                                          # type: ignore[call-overload]
    if isinstance(p, TypeVarType):
        p = get_proper_type(p.upper_bound)
    if isinstance(p, LiteralType):
        p = p.fallback
    if isinstance(p, UnionType):                                    # every alternative the same kind, or unknown
        kinds = {_kind(i) for i in p.items}
        return kinds.pop() if len(kinds) == 1 else None
    if isinstance(p, TupleType):
        return "tuple"
    if not isinstance(p, Instance):
        return None
    name = p.type.fullname
    if name == "numpy.ndarray":
        return "array"
    numbers = ("builtins.float", "builtins.int", "builtins.bool", "builtins.complex")
    if name in numbers or p.type.has_base("numpy.generic"):
        return "scalar"
    if name in ("builtins.list", "builtins.dict", "builtins.set", "builtins.slice", "builtins.tuple"):
        return name.removeprefix("builtins.")
    if name.split(".")[0] in ("builtins", "typing", "numpy", "collections", "_collections_abc", "types"):
        return None
    return "class:" + name


def _version() -> str:
    try:
        from mypy.version import __version__
        return __version__
    except ImportError:
        return "none"


def _cache_dir() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "headway" / "types"
