"""The notation layer: indices, symbols, operations and groups, declared in a spec module next to the code.

A spec describes a method as a bipartite graph. Symbols are the values (scalars, vectors, matrices, settings, data);
operations turn input symbols into output symbols. Operations point into the code with references
("package.module:qualname", optionally "#from text..to text" for lines inside a function), which the checks
resolve. Groups ("round 1/model") nest operations so the viewer can collapse them.
"""
from __future__ import annotations

import itertools
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

ROLES = ("input", "value", "output", "setting", "constant")


@dataclass
class Index:
    key: str
    tex: str
    name: str
    desc: str = ""
    varies: bool = False        # its size may differ between calls (e.g. a subset of cells); not checked across calls


@dataclass
class Symbol:
    key: str
    tex: str
    name: str
    desc: str = ""
    dims: tuple[str, ...] = ()  # index keys
    per: str | None = None      # an index the symbol has one copy of per value (e.g. per fold)
    role: str = "value"
    value: Any = None           # a setting's default, a constant's value
    code: list[str] = field(default_factory=list)
    was: list[str] = field(default_factory=list)            # earlier keys, for comparing versions (diff.py)

    @property
    def kind(self) -> str:
        return {0: "scalar", 1: "vector", 2: "matrix"}.get(len(self.dims), "array")


@dataclass
class Op:
    key: str
    label: str
    tex: str = ""
    desc: str = ""
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    code: list[str] = field(default_factory=list)
    group: str = ""
    per: str | None = None
    bind: dict[str, str] = field(default_factory=dict)     # code name -> symbol key, for the shape trace
    where: list[tuple[str, str]] = field(default_factory=list)  # (TeX, definition): symbols used only in this step
    was: list[str] = field(default_factory=list)            # earlier keys, for comparing versions (diff.py)


@dataclass
class Decl:
    """Where a declaring call (G.op(...) etc.) is in the spec."""
    file: str
    module: str
    function: str               # "<module>", or the spec's helper function the call is in
    line: int                   # the call's span; 1-based lines, 0-based columns (as in ast)
    col: int
    end_line: int
    end_col: int


@dataclass
class Group:
    path: str
    label: str
    desc: str = ""
    collapsed: bool = False
    was: list[str] = field(default_factory=list)            # earlier paths, for comparing versions (diff.py)


type SymbolLike = Symbol | str
type IndexLike = Index | str


def _key(x: Symbol | Index | str) -> str:
    return x if isinstance(x, str) else x.key


class Headway:
    """A method's notation and computation graph. Declare with index(), symbol(), setting(), op(), group() and
    @verify; build with `headway build <module>` and check with `headway check <module>`.

    `was` (on symbols, settings, constants, steps and groups) names the keys an object had in earlier versions of
    the spec, so that comparing versions (`headway diff`, the viewer's lanes) matches it across a rename. Several
    keys: renamed more than once, or several earlier objects merged into this one; the same earlier key on several
    objects: one object split into these."""

    def __init__(self, title: str, desc: str = ""):
        self.title, self.desc = title, desc
        self.indices: dict[str, Index] = {}
        self.symbols: dict[str, Symbol] = {}
        self.ops: dict[str, Op] = {}
        self.groups: dict[str, Group] = {}
        self.verifiers: list[tuple[tuple[str, ...], Callable[[], object]]] = []
        self.declared: dict[str, Decl] = {}                     # "op:key" etc. -> where the declaring call is

    def index(self, key: str, tex: str, name: str, desc: str = "", *, varies: bool = False) -> Index:
        assert key not in self.indices, f"index {key!r} declared twice"
        self.indices[key] = ix = Index(key, tex, name, desc, varies)
        self.declared["index:" + key] = _caller()
        return ix

    def symbol(self, key: str, tex: str, name: str, desc: str = "", *, dims: Sequence[IndexLike] = (),
               per: IndexLike | None = None, role: str = "value", value: Any = None,
               code: str | Iterable[str] = (), was: str | Iterable[str] = ()) -> Symbol:
        assert key not in self.symbols, f"symbol {key!r} declared twice"
        assert role in ROLES, f"symbol {key!r}: role {role!r} not one of {ROLES}"
        self.symbols[key] = s = Symbol(key, tex, name, desc, tuple(map(_key, dims)),
                                       None if per is None else _key(per), role, value, _refs(code), _refs(was))
        self.declared["symbol:" + key] = _caller()
        return s

    def setting(self, key: str, tex: str, name: str, desc: str = "", *, value: Any = None,
                code: str | Iterable[str] = (), was: str | Iterable[str] = ()) -> Symbol:
        return self.symbol(key, tex, name, desc, role="setting", value=value, code=code, was=was)

    def constant(self, key: str, tex: str, name: str, desc: str = "", *, value: Any,
                 code: str | Iterable[str] = (), was: str | Iterable[str] = ()) -> Symbol:
        """A number fixed in the code (a clip bound, a tolerance, an iteration limit), where a setting is one the
        user can change. Point `code` at the line that sets it."""
        return self.symbol(key, tex, name, desc, role="constant", value=value, code=code, was=was)

    def op(self, key: str, label: str, tex: str = "", desc: str = "", *, inputs: Sequence[SymbolLike] = (),
           outputs: Sequence[SymbolLike] = (), code: str | Iterable[str] = (), group: str = "",
           per: IndexLike | None = None, bind: dict[str, SymbolLike] | None = None,
           where: dict[str, str] | None = None, was: str | Iterable[str] = ()) -> Op:
        r"""`where` defines symbols the equation uses that are not values of the graph, as a paper would after the
        equation: {TeX: definition}, e.g. {r"\mu_f": "mean of $x_{if}$ over the learning cells"}. A value another
        step also uses is better declared as a symbol and listed in both steps' inputs or outputs."""
        assert key not in self.ops, f"op {key!r} declared twice"
        self.ops[key] = o = Op(key, label, tex, desc, list(map(_key, inputs)), list(map(_key, outputs)),
                               _refs(code), group, None if per is None else _key(per),
                               {k: _key(v) for k, v in (bind or {}).items()}, list((where or {}).items()),
                               _refs(was))
        self.declared["op:" + key] = _caller()
        return o

    def group(self, path: str, label: str | None = None, desc: str = "", *, collapsed: bool = False,
              was: str | Iterable[str] = ()) -> Group:
        self.groups[path] = g = Group(path, label or path.rsplit("/", 1)[-1], desc, collapsed, _refs(was))
        self.declared["group:" + path] = _caller()
        return g

    def verify(self, *ops: str) -> Callable[[Callable[[], object]], Callable[[], object]]:
        """Register a check that the named ops' equations match the code (raise AssertionError if not). Run by
        `headway check` and `headway build`; the viewer marks the ops verified or failed."""
        def register(fn: Callable[[], object]) -> Callable[[], object]:
            self.verifiers.append((ops, fn))
            return fn
        return register

    def all_groups(self) -> dict[str, Group]:
        """The declared groups and every ancestor path of an op's group, parents before children."""
        paths = set(self.groups)
        for o in self.ops.values():
            parts = o.group.split("/") if o.group else []
            paths.update("/".join(parts[:n]) for n in range(1, len(parts) + 1))
        return {p: self.groups.get(p) or Group(p, p.rsplit("/", 1)[-1])
                for p in sorted(paths, key=lambda p: (p.count("/"), list(self.groups).index(p)
                                                      if p in self.groups else len(self.groups), p))}


def _caller() -> Decl:
    """Where a declaration is: the call in the first frame outside headway (in a helper of the spec, the helper's)."""
    f = sys._getframe(1)
    while f.f_back and f.f_globals.get("__name__", "").split(".")[0] == "headway":
        f = f.f_back
    # the exact span of the call being made in that frame
    line, end_line, col, end_col = next(itertools.islice(f.f_code.co_positions(), f.f_lasti // 2, None))
    return Decl(f.f_code.co_filename, f.f_globals.get("__name__", ""), f.f_code.co_name,
                line or f.f_lineno, col or 0, end_line or f.f_lineno, end_col or 0)


def _refs(code: str | Iterable[str]) -> list[str]:
    return [code] if isinstance(code, str) else list(code)
