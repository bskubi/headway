"""headway init: a first draft of a spec, read from the code (Python, with the ast module; nothing is imported).

From an entry point ("pkg.mod:func"), every call to a function of the same package is a step, with a code reference
to the function and one to the line that calls it. Calls into other packages (numpy, polars, ...) are not steps.
Steps are wired by binding only (Drafter): what the code states, traced with beniget's def-use chains, and what a
called function changes in place (Drafter.changes). A value made by other code is a gap, an input marked with the
line that made it, and the places where values may change in place unseen are listed. A called function that itself
calls the package's functions is opened up into a group of those calls, `depth` levels down. A loop around a step
makes it a step per value of the loop's variable. With hints, type hints are taken as assumptions: settings
dataclasses, methods of package classes as steps, arrays and lists changed in place. Each symbol gets a short one
chosen from its name (short_symbols). The equations and descriptions are left as TODOs.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import keyword
import re
import tokenize
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import beniget  # type: ignore[import-untyped]

from headway import shapes
from headway.codetex import MODULES, SAME, _notation, _output_only, neighbours
from headway.resolve import RefError, _anchored
from headway.typeinfo import Types


class DraftError(Exception):
    pass


class DraftEdited(DraftError):
    """The spec has been changed by hand since it was drafted: drafting it again would replace those changes."""


@dataclass
class Module:
    name: str
    file: Path
    lines: list[str]
    defs: dict[str, ast.FunctionDef | ast.ClassDef]
    imports: dict[str, str]     # local name -> the dotted name it stands for ("pkg.mod" or "pkg.mod.func")
    aliases: dict[str, ast.expr] = field(default_factory=dict)     # type aliases: `type X = ...`, X: TypeAlias = ...
    tree: ast.Module = field(repr=False, default_factory=lambda: ast.Module([], []))
    _chains: dict[ast.AST, list[Any]] | None = field(default=None, repr=False)
    _parents: dict[ast.AST, ast.AST] | None = field(default=None, repr=False)

    def chains(self) -> dict[ast.AST, list[Any]]:
        """Each variable read -> the definitions that can reach it (beniget's use-def chains, Python's scoping)."""
        if self._chains is None:
            du = beniget.DefUseChains(str(self.file))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                du.visit(self.tree)                     # (its warnings: names it cannot bind, as builtins)
            self._chains = beniget.UseDefChains(du).chains
        return self._chains

    def parents(self) -> dict[ast.AST, ast.AST]:
        if self._parents is None:
            self._parents = {c: n for n in ast.walk(self.tree) for c in ast.iter_child_nodes(n)}
        return self._parents


@dataclass
class Function:
    module: Module
    node: ast.FunctionDef
    cls: str = ""               # the class it is a method of

    @property
    def ref(self) -> str:
        return f"{self.module.name}:{self.cls + '.' if self.cls else ''}{self.node.name}"

    @property
    def params(self) -> list[str]:
        """Its parameters, as a call gives them (a method's self is its receiver, not an argument)."""
        args = [a.arg for a in (*self.node.args.posonlyargs, *self.node.args.args, *self.node.args.kwonlyargs)]
        return args[1:] if self.cls and args and not _is_static(self.node) else args


@dataclass
class Settings:
    """A settings object in scope: a dataclass whose fields all have defaults."""
    module: Module
    node: ast.ClassDef


@dataclass
class Sym:
    key: str
    name: str
    role: str = "value"
    per: str | None = None
    value: object = None
    code: list[str] = field(default_factory=list)
    desc: str = "TODO"
    dims: list[str] = field(default_factory=list)      # index keys, from a shape in a docstring (with hints)


@dataclass
class Step:
    key: str
    label: str
    inputs: list[str]
    outputs: list[str]
    code: list[str]
    group: str
    per: str | None
    call: ast.expr | None = None    # the call it was made for
    names: dict[str, str] = field(default_factory=dict)    # a name in its code -> the symbol it holds (names=)
    condition: str = ""             # when it is done: the tests of the ifs it is in, in the code's words
    relations: list[dict[str, Any]] = field(default_factory=list)   # what its function declares it computes


class Package:
    """The modules of one top-level package, found on disk and parsed on demand."""

    def __init__(self, top: str, search: list[Path]):
        self.top = top
        self.folder = _locate(top, search)
        self.modules: dict[str, Module | None] = {}

    def module(self, name: str) -> Module | None:
        if name.split(".")[0] != self.top:
            return None
        if name not in self.modules:
            rel = Path(*name.split("."))
            base = self.folder.parent
            file = next((f for f in (base / rel.with_suffix(".py"), base / rel / "__init__.py") if f.is_file()), None)
            self.modules[name] = _parse(name, file) if file else None
        return self.modules[name]

    def lookup(self, dotted: str, seen: frozenset[str] = frozenset()) -> Module | ast.FunctionDef | ast.ClassDef | None:
        """A dotted name -> the package's module, function or class it names (following re-exports)."""
        if dotted in seen:
            return None
        m = self.module(dotted)
        if m:
            return m
        owner, _, last = dotted.rpartition(".")
        m = self.module(owner) if owner else None
        if m is None:
            return None
        if last in m.defs:
            return m.defs[last]
        if last in m.imports:
            return self.lookup(m.imports[last], seen | {dotted})
        return None

    def owner(self, node: ast.AST) -> Module:
        return next(m for m in self.modules.values() if m and node in m.defs.values())


def _locate(top: str, search: list[Path]) -> Path:
    for base in search:
        for folder in (base / top, base / "src" / top):
            if (folder / "__init__.py").is_file():
                return folder
    try:
        spec = importlib.util.find_spec(top)            # a top-level package: found without being imported
    except (ImportError, ValueError):
        spec = None
    if spec and spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations)))
    raise DraftError(f"no package {top} under {', '.join(map(str, search))} or on the Python path")


def axes_reader(file: str, name: str = "") -> Callable[[ast.expr], list[str] | None]:
    """The named axes a type hint in the file gives an array (shapes.py): read with the file's imports and type
    aliases (codetex's indices, with hints). name: the file's module, for its relative imports."""
    try:
        m = _parse(name, Path(file))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return lambda ann: None

    def full(dotted: str) -> str:
        head, _, rest = dotted.partition(".")
        return m.imports.get(head, head) + ("." + rest if rest else "")
    return lambda ann: shapes.dims(ann, full, m.aliases.get)


def _parse(name: str, file: Path) -> Module:
    text = file.read_text()
    tree = ast.parse(text, str(file))
    package = name if file.name == "__init__.py" else name.rpartition(".")[0]
    defs: dict[str, ast.FunctionDef | ast.ClassDef] = {}
    imports: dict[str, str] = {}
    for st in tree.body:
        if isinstance(st, (ast.FunctionDef, ast.ClassDef)):
            defs[st.name] = st
        elif isinstance(st, ast.Import):
            for a in st.names:
                if a.asname:
                    imports[a.asname] = a.name
                else:
                    imports[a.name.split(".")[0]] = a.name.split(".")[0]
        elif isinstance(st, ast.ImportFrom):
            source = st.module or ""
            if st.level:
                parts = package.split(".")
                parts = parts[:len(parts) - st.level + 1]
                source = ".".join(parts + ([source] if source else []))
            for a in st.names:
                imports[a.asname or a.name] = f"{source}.{a.name}"
    aliases: dict[str, ast.expr] = {}
    for st in tree.body:
        if isinstance(st, getattr(ast, "TypeAlias", ())) and isinstance(st.name, ast.Name):   # `type X = ...` (3.12+)
            aliases[st.name.id] = st.value
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None \
                and (st.annotation.attr if isinstance(st.annotation, ast.Attribute) else
                     getattr(st.annotation, "id", "")) == "TypeAlias":
            aliases[st.target.id] = st.value
    return Module(name, file, text.splitlines(keepends=True), defs, imports, aliases, tree)


def _is_data(node: ast.ClassDef) -> bool:
    """A class that only holds values (a dataclass, a NamedTuple, a Protocol, or no methods): building one is not
    a step."""
    def name(e: ast.expr) -> str:
        e = e.func if isinstance(e, ast.Call) else e
        return e.attr if isinstance(e, ast.Attribute) else e.id if isinstance(e, ast.Name) else ""
    if any(name(d) == "dataclass" for d in node.decorator_list):
        return True
    if any(name(b) in ("NamedTuple", "TypedDict", "Protocol", "Enum") for b in node.bases):
        return True
    return not any(isinstance(s, ast.FunctionDef) and not s.name.startswith("__") for s in node.body)


def _fields(node: ast.ClassDef) -> dict[str, ast.AnnAssign]:
    return {s.target.id: s for s in node.body if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)}


def _is_settings(node: ast.ClassDef) -> bool:
    f = _fields(node)
    return _is_data(node) and bool(f) and all(a.value is not None for a in f.values())


class Plain:
    """A value made by code that is not a step (numpy, arithmetic, a list built up): what it was made from is not
    followed, so a step that reads it shows it as an input of its own, with the line that made it."""

    def __init__(self, fn: Function, st: ast.stmt, name: str, group: str = ""):
        self.fn, self.st, self.name = fn, st, name
        self.group = group      # the group it was made in: one value per call of a function opened twice
        self.dims: list[str] = []   # its indices, if the code states its shape (a return or an assignment annotated)


class Derived(frozenset):  # type: ignore[type-arg]
    """A value computed from these symbols (an expression of them, a dataclass of them), not one of them: an edge
    from each, but not written as any of them."""


class Record:
    """A dataclass of the package's, built from values (Fold(a, rl, beta)), in clean code: each field holds what it
    was given, so reading a field (fold.a) is an edge from that value only."""

    def __init__(self, fields: dict[str, list["Held"]]):
        self.fields = fields


Held = frozenset | Plain | Settings | Record    # what a variable holds: one of these symbols (or Derived from them),
                                        # plain code, settings


class Drafter:
    """The draft's steps and symbols, wired by binding only: a step's inputs are the variables its arguments read,
    each traced (with beniget's def-use chains, Python's own scoping) to the assignment that made it. A variable
    assigned a step's call holds that step's output; one assigned another variable holds what that one holds; one
    made by any other code is a gap: an input of its own, marked with the line that made it. Nothing is guessed
    about what the code does at run time, so an edge is never wrong, though values that flow by other means (changed
    in place, through a callback) have none; where the code shows such a means, a note says so."""

    def __init__(self, package: Package, hints: bool = False, infer: bool = True, lines: bool = False):
        self.package = package
        self.hints = hints                      # type hints taken as assumptions (headway init --hints)
        self.lines = lines                      # an assignment by other code: a step of its own (--clean)
        self.opening: list[str] = []                # the functions being opened up, outermost first
        self.skipped: set[int] = set()              # the nodes of branches that are not taken (decided by literals)
        # an assignment in a loop that iterates -> (the loop's last line, what assigned the variable before it)
        self.exits: dict[int, tuple[int, set[int]]] = {}
        self.per_element: dict[ast.expr, str | None] = {}   # a comprehension -> the index it is done per
        self.types = Types.of_package(package.folder) if hints and infer else None   # what mypy infers from them
        self.changing: dict[str, dict[str, int]] = {}   # a function -> the parameters it changes, and where
        self.change_lines: dict[str, set[int]] = {}     # a function -> every line that changes a parameter
        self.overrides: list[tuple[Function, str, ast.stmt, list[Held]]] = []   # a call changed a variable
        self.drawn: set[tuple[str, int]] = set()    # (function, line): changes in place drawn as edges
        self.symbols: dict[str, Sym] = {}
        self.steps: list[Step] = []
        self.groups: dict[str, str] = {}        # path -> its description (the function it opens up)
        self.indices: dict[str, str] = {}       # key -> the loop it comes from
        self.notes: list[str] = []              # where values may flow with no edge drawn
        self.held: dict[ast.AST, list[Held]] = {}   # an assignment's target (a def) -> what it holds
        self.gaps: dict[tuple[int, str, str], str] = {}  # (statement, variable, group) -> its gap symbol
        self.results: dict[ast.expr, list[list[Held]]] = {}     # a call (a comprehension) -> what it gives

    # names

    @staticmethod
    def _key(name: str, taken: Collection[str]) -> str:
        base = name if name.isidentifier() and not keyword.iskeyword(name) else "v"
        key, n = base, 2
        while key in taken:
            key, n = f"{base}_{n}", n + 1
        return key

    def symbol(self, name: str, role: str = "value", per: str | None = None) -> str:
        key = self._key(name, self.symbols)
        self.symbols[key] = Sym(key, name, role, per)
        return key

    def setting(self, s: Settings, name: str) -> str:
        ref = f"{s.module.name}:{s.node.name}.{name}"
        for sym in self.symbols.values():
            if sym.role == "setting" and sym.code == [ref]:
                return sym.key
        key = self.symbol(name, "setting")
        sym = self.symbols[key]
        sym.code = [ref]
        try:
            value = _fields(s.node)[name].value
            sym.value = ast.literal_eval(value) if value is not None else None
        except ValueError:
            pass
        field_line = _fields(s.node)[name].lineno if name in _fields(s.node) else 0
        said = _comment(s.module.lines, field_line) or _documented(_docs(s.node), name)
        if said:
            sym.desc = said
        return key

    # shapes stated in type hints (jaxtyping's Float[np.ndarray, "cells genes"]), with hints

    def annotation(self, m: Module, ann: ast.expr | None, depth: int = 0) -> tuple[Module, ast.expr | None]:
        """An annotation with its type aliases expanded, in the module whose imports it is read with."""
        if isinstance(ann, ast.Name) and depth < 10:
            if ann.id in m.aliases:
                return self.annotation(m, m.aliases[ann.id], depth + 1)
            if ann.id in m.imports:
                owner, _, last = m.imports[ann.id].rpartition(".")
                where = self.package.module(owner)
                if where is not None and last in where.aliases:
                    return self.annotation(where, where.aliases[last], depth + 1)
        return m, ann

    def shape(self, m: Module, ann: ast.expr | None) -> list[str] | None:
        """The named axes an annotation gives an array (shapes.py), or None."""
        if not self.hints:
            return None
        where, a = self.annotation(m, ann)

        def full(dotted: str) -> str:
            head, _, rest = dotted.partition(".")
            return where.imports.get(head, head) + ("." + rest if rest else "")
        return shapes.dims(a, full, lambda name: None)

    def shape_items(self, m: Module, ann: ast.expr | None) -> list[list[str] | None] | None:
        """The shapes of a tuple's items (tuple[Float[...], Float[...]])."""
        if not self.hints:
            return None
        where, a = self.annotation(m, ann)
        parts = shapes.items(a, lambda name: None)
        return None if parts is None else [self.shape(where, p) for p in parts]

    def unify(self, scope: Scope, f: Function, call: ast.Call) -> dict[str, str]:
        """f's axis names, bound to the caller's indices through the arguments given as they are (a parameter
        annotated "cells columns", given a symbol of indices (i, j): cells is i and columns j, in this call). An
        argument with no indices yet takes them from the annotation."""
        if not self.hints:
            return {}
        annotations = {a.arg: a.annotation for a in (*f.node.args.posonlyargs, *f.node.args.args,
                                                     *f.node.args.kwonlyargs)}
        bound: dict[str, str] = {}
        pending = []
        for p, a in _given(f, call):
            axes = self.shape(f.module, annotations.get(p))
            same = self.identity(scope, a)
            if not axes or not same:
                continue
            has = self.symbols[same].dims
            if not has:
                pending.append((same, axes))
            elif len(has) == len(axes):
                for name, index in zip(axes, has):
                    bound.setdefault(name, index)
        for same, axes in pending:
            self.give(same, axes, bound, f.ref, self.param_axes(f))
        return bound

    def param_axes(self, f: Function) -> set[str]:
        """The axis names f's parameters' annotations use."""
        return {n for a in (*f.node.args.posonlyargs, *f.node.args.args, *f.node.args.kwonlyargs)
                for n in self.shape(f.module, a.annotation) or ()}

    def axes(self, axes: list[str] | None, bound: dict[str, str], params: set[str], ref: str) -> list[str]:
        """A shape's indices, all or none: an axis name bound in the call is the caller's index; one only the result
        has (no parameter of the function names it) is a new axis, the index of that name; one a parameter names
        but this call does not bind is not known, and then neither is the shape."""
        if not axes or any(a not in bound and a in params for a in axes):
            return []
        out = [bound.get(a, a) for a in axes]
        for d in out:
            self.indices.setdefault(d, f"an axis named in the type hints of {ref}")
        return out

    def give(self, key: str, axes: list[str] | None, bound: dict[str, str], ref: str, params: set[str]) -> None:
        """A symbol's indices from a shape (axes()), if it has none yet."""
        sym = self.symbols[key]
        if not sym.dims:
            sym.dims = self.axes(axes, bound, params, ref)

    def described(self, key: str, text: str | None, type_name: str | None, ref: str) -> None:
        """What a docstring says of a value: its description, and with hints, the indices of a shape its type
        names (`ndarray of shape (n_cells, n_genes)`)."""
        sym = self.symbols[key]
        if text and text.strip() and sym.desc == "TODO":
            sym.desc = " ".join(text.split())
        dims = _shape(type_name) if self.hints else None
        if dims:
            sym.dims = dims
            for d in dims:
                self.indices.setdefault(d, f"a shape in the docstring of {ref}")

    def gap(self, p: Plain) -> str:
        """The symbol of a value made by plain code: an input, with the line that made it."""
        where = (id(p.st), p.name, p.group)
        if where not in self.gaps:
            key = self.symbol(p.name, "input")
            sym = self.symbols[key]
            anchor = _anchor(p.fn, p.st)
            sym.code = [f"{p.fn.ref}#{anchor}" if anchor else p.fn.ref]
            sym.desc = (f"TODO. Made by the code at {p.fn.module.file.name}:{p.st.lineno}, not by a step: headway "
                        "init does not follow values through code between steps.")
            sym.dims = list(p.dims)
            self.gaps[where] = key
        return self.gaps[where]

    # what a name in the code is

    def target(self, fn: Function, e: ast.expr) -> Function | ast.ClassDef | None:
        """The package's function or class an expression names, if it does (by the module's imports and
        definitions; a local variable of the same name is not looked through)."""
        m = fn.module
        if isinstance(e, ast.Name):
            if e.id in _locals(fn.node):
                return None
            found: object = m.defs.get(e.id) or (self.package.lookup(m.imports[e.id]) if e.id in m.imports else None)
        elif isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) and e.value.id in _locals(fn.node):
            # a method of a value: with type hints taken as assumptions, of a value hinted as a package class
            kind = self.hint(fn, e.value) if self.hints else None
            method = next((s for s in kind.body if isinstance(s, ast.FunctionDef) and s.name == e.attr), None) \
                if isinstance(kind, ast.ClassDef) else None
            if method is None or not isinstance(kind, ast.ClassDef):
                return None
            return Function(self.package.owner(kind), method, kind.name)
        elif isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name):
            mod = self.package.lookup(m.imports[e.value.id]) if e.value.id in m.imports else None
            if not isinstance(mod, Module) or e.attr not in mod.defs and e.attr not in mod.imports:
                return None
            found = mod.defs.get(e.attr) or self.package.lookup(mod.imports[e.attr])
        else:
            return None
        if isinstance(found, ast.FunctionDef):
            return Function(self.package.owner(found), found)
        if isinstance(found, ast.ClassDef):
            return found
        return None

    # type hints (with --hints): what kind of value a variable is, as its annotation says

    def hint(self, fn: Function, use: ast.Name) -> str | ast.ClassDef | None:
        """What a variable holds, by its type hints: "array", "list", "dict", "set", "scalar" or a package class;
        None if a definition that can reach it has no hint, or they differ. The hints are the parameter's, the
        annotated assignment's, or the return annotation of the package function whose call it was assigned. What
        the type checker (mypy) infers from them, through calls, attributes and loops, comes first."""
        m = fn.module
        inferred = self.types.kind(m.file, use.lineno, use.col_offset, use.id) if self.types else None
        if inferred is not None:
            if inferred.startswith("class:"):
                found = self.package.lookup(inferred.removeprefix("class:"))
                return found if isinstance(found, ast.ClassDef) else None
            return inferred
        kinds = []
        for d in m.chains().get(use, []):
            node, ann, where = d.node, None, m
            if isinstance(node, ast.arg):
                ann = node.annotation
            else:
                st = _owner_stmt(m, node)
                if isinstance(st, ast.AnnAssign) and st.target is node:
                    ann = st.annotation
                elif isinstance(st, ast.Assign) and st.targets[0] is node and isinstance(st.value, ast.Call):
                    t = self.target(fn, st.value.func)
                    if isinstance(t, Function):
                        ann, where = t.node.returns, t.module
            if ann is None:
                return None
            kinds.append(self.kind(where, ann))
        return kinds[0] if kinds and all(k is kinds[0] or k == kinds[0] for k in kinds) else None

    def kind(self, m: Module, ann: ast.expr) -> str | ast.ClassDef | None:
        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):     # "NDArray[np.float64]"
            try:
                ann = ast.parse(ann.value, mode="eval").body
            except SyntaxError:
                return None
        if isinstance(ann, ast.Subscript):
            ann = ann.value                             # NDArray[np.float64]: NDArray
        dotted = _dotted_name(ann)
        if dotted is None:
            return None
        head, _, rest = dotted.partition(".")
        full = (m.imports.get(head, head) + ("." + rest if rest else ""))
        if full in ("numpy.ndarray", "numpy.typing.NDArray"):
            return "array"
        if full in ("float", "int", "bool", "complex") or full.startswith("numpy.") and full.rpartition(".")[2] in (
                "float64", "float32", "int64", "int32", "floating", "integer", "bool_", "number"):
            return "scalar"
        if full in ("list", "typing.List"):
            return "list"
        if full in ("dict", "typing.Dict"):
            return "dict"
        if full in ("set", "typing.Set"):
            return "set"
        found = m.defs.get(dotted) if dotted in m.defs else self.package.lookup(full) if "." in full else None
        return found if isinstance(found, ast.ClassDef) else None

    # what a function changes in place

    def changes(self, f: Function, seen: frozenset[str] = frozenset()) -> dict[str, int]:
        """The parameters a function changes in place, each with the first line that does: a store into it
        (p[...] = e, p[...] += e, p.a = e), numpy's out=p, or passing it to a package function that changes it.
        With type hints: also p += e on an array, and p.append(...), p.sort(), ... on a hinted list, dict, set or
        array. What it changes through another name (a view) is not seen."""
        if f.ref in self.changing:
            return self.changing[f.ref]
        if f.ref in seen:
            return {}
        out: dict[str, int] = {}
        m = f.module
        params = set(f.params)

        def param(e: ast.expr, depth: int = 0) -> str | None:
            """The parameter a name refers to, on every path (not a variable rebound from it); with type hints,
            also a view of one (v = p[:, c], v = p.T) and an array changed by an operator in place (v -= m keeps
            v the same array): changing those changes the parameter."""
            if not isinstance(e, ast.Name) or depth > 20:
                return None
            defs = [d.node for d in m.chains().get(e, []) if isinstance(d.node, (ast.Name, ast.arg))]
            if len(defs) != 1:
                return None
            d = defs[0]
            if isinstance(d, ast.arg):
                return d.arg if d.arg in params else None
            st = _owner_stmt(m, d)
            if not self.hints:
                return None
            if isinstance(st, ast.Assign) and st.targets == [d]:
                base = self.view_of(f, st.value)
                return param(base, depth + 1) if base is not None else None
            if isinstance(st, ast.AugAssign) and st.target is d and self.hint(f, d) == "array":
                return param(d, depth + 1)              # the same array as before the operator
            return None

        for st in (n for n in _own_nodes(f.node) if isinstance(n, ast.stmt)):
            found: list[str | None] = []
            targets = st.targets if isinstance(st, ast.Assign) else [st.target] if isinstance(
                st, (ast.AugAssign, ast.AnnAssign)) else []
            for t in targets:
                for sub in ast.walk(t):
                    if isinstance(sub, (ast.Subscript, ast.Attribute)) and isinstance(sub.ctx, ast.Store):
                        found.append(param(_root(sub)))
            if isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name) and self.hints \
                    and self.hint(f, _load(st.target)) == "array":
                found.append(param(_load(st.target)))
            for c in (n for e in ast.iter_child_nodes(st) if isinstance(e, ast.expr) for n in ast.walk(e)
                      if isinstance(n, ast.Call)):
                if _module_call(m, c, "numpy"):
                    found += [param(k.value) for k in c.keywords if k.arg == "out"]
                if self.hints and isinstance(c.func, ast.Attribute) and isinstance(c.func.value, ast.Name):
                    kind = self.hint(f, c.func.value)
                    if isinstance(kind, str) and c.func.attr in MUTATING.get(kind, ()):
                        found.append(param(c.func.value))
                callee = self.target(f, c.func)
                if isinstance(callee, Function):
                    changed = self.changes(callee, seen | {f.ref})
                    for q, a in _given(callee, c):
                        if q in changed:
                            found.append(param(a))
            for hit in found:
                if hit is not None:
                    out.setdefault(hit, st.lineno)
                    self.change_lines.setdefault(f.ref, set()).add(st.lineno)
        self.changing[f.ref] = out
        return out

    def view_of(self, f: Function, e: ast.expr) -> ast.Name | None:
        """With type hints: the array a value is a view of, by numpy's rules: basic indexing (slices, integers,
        None, ...), .T, .view() of a variable that is an array. (Indexing by an array or a mask copies.)"""
        base: ast.expr | None = None
        if isinstance(e, ast.Subscript):
            items = e.slice.elts if isinstance(e.slice, ast.Tuple) else [e.slice]
            if all(self.basic(f, i) for i in items):
                base = e.value
        elif isinstance(e, ast.Attribute) and e.attr == "T":
            base = e.value
        elif isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute) and e.func.attr == "view" and not e.args:
            base = e.func.value
        if isinstance(base, ast.Name) and self.hint(f, base) == "array":
            return base
        return None

    def basic(self, f: Function, i: ast.expr) -> bool:
        """An index numpy reads as basic (a view): a slice, an integer, None or ..., as written or as typed."""
        if isinstance(i, ast.Slice):
            return all(x is None or self.basic(f, x) for x in (i.lower, i.upper, i.step))
        if isinstance(i, ast.Constant):
            return i.value is None or i.value is Ellipsis or isinstance(i.value, int) and not isinstance(i.value, bool)
        if isinstance(i, ast.UnaryOp) and isinstance(i.op, ast.USub):
            return self.basic(f, i.operand)
        if isinstance(i, ast.Name) and self.types is not None:
            return self.types.kind(f.module.file, i.lineno, i.col_offset, i.id) in ("slice", "scalar")
        return False

    def changed_by(self, scope: Scope, f: Function, call: ast.Call, st: ast.stmt, made: Callable[[str], Held]) -> None:
        """After a call to f, each variable it was given for a parameter it changes holds what made(variable)
        gives: the step's changed value, or, for a group, a gap (changed by plain code within it)."""
        changed = self.changes(f)
        rebound = {n.id for t in getattr(st, "targets", [getattr(st, "target", None)]) if t is not None
                   for n in ast.walk(t) if isinstance(n, ast.Name)}
        for p, a in _given(f, call):
            if p in changed and isinstance(a, ast.Name):
                self.drawn.add((scope.fn.ref, st.lineno))
                self.drawn |= {(f.ref, line) for line in self.change_lines.get(f.ref, ())}
                if a.id not in rebound:                 # x = f(x): x is rebound to the result anyway
                    self.overrides.append((scope.fn, a.id, st, [made(a.id)]))

    def opens(self, f: Function, depth: int) -> bool:
        """Whether a call to f is opened up into a group: it calls the package's functions itself (and is not
        being opened already: a recursive call is a step)."""
        if self.lines and any(isinstance(n, (ast.For, ast.While)) and _carried(n) for n in _own_nodes(f.node)):
            return False                                # clean code: a function that iterates is one action
        return depth > 0 and f.ref not in self.opening and any(isinstance(n, ast.Call) and self.is_step(f, n) for n in _own_nodes(f.node))

    def is_step(self, fn: Function, call: ast.Call) -> bool:
        t = self.target(fn, call.func)
        return isinstance(t, Function) or isinstance(t, ast.ClassDef) and not _passes_through(t)

    # what a variable holds

    def resolve(self, scope: Scope, use: ast.Name, outside: set[int] | None = None) -> list[Held]:
        """What a variable read holds, from the assignments that can reach it (outside: only those not among these
        nodes, a loop's own, as what comes into the loop)."""
        out: list[Held] = []
        m = scope.fn.module
        mine = {id(n) for n in ast.walk(scope.fn.node)}
        defs = [d.node for d in m.chains().get(use, []) if (outside is None or id(d.node) not in outside)
                and id(d.node) not in self.skipped]     # what a branch that is not taken assigns: never
        for node in list(defs):
            if id(node) in self.exits and use.lineno > self.exits[id(node)][0]:
                # after an iteration: what it left (which comes from what came into it), not what came in
                defs = [n for n in defs if id(n) not in self.exits[id(node)][1]]
        for node in defs:
            if node in self.held:
                out += self.held[node]
            elif isinstance(node, ast.arg) and node.arg in scope.params:
                line = self.changes(scope.fn).get(node.arg)
                if line is not None and line < use.lineno:
                    # changed in place before here (on some path): what it holds then is made by this code
                    at = next(n for n in _own_nodes(scope.fn.node) if isinstance(n, ast.stmt) and n.lineno == line)
                    out.append(Plain(scope.fn, at, node.arg, scope.group))  # type: ignore[arg-type]
                    self.drawn |= {(scope.fn.ref, n) for n in self.change_lines.get(scope.fn.ref, ())}
                else:
                    out += scope.params[node.arg]
            elif isinstance(node, ast.Name) and isinstance(_unpacked(m, node), ast.comprehension):
                # a comprehension's variable: an element of what it runs over, not followed
                out.append(Plain(scope.fn, _owner_stmt(m, node), node.id, scope.group))  # type: ignore[arg-type]
            elif isinstance(node, ast.Name) and id(node) in mine:
                # assigned in this function but not reached yet: by a later line, in an earlier turn of a loop
                self.note(scope, f"`{use.id}` at line {use.lineno} may also hold what line {node.lineno} "
                                      "assigns, in an earlier turn of the loop: not drawn")
            elif isinstance(node, ast.Name) and isinstance(st := _owner_stmt(m, node), ast.Assign) \
                    and st in m.tree.body:                     # a module-level constant
                out.append(frozenset([self.constant(m, node.id, st)]))
        return self.changed(scope, use, out)

    def changed(self, scope: Scope, use: ast.Name, out: list[Held]) -> list[Held]:
        """A variable read after a call that changed it in place holds what the call made of it: instead of what
        it held, if the call is on every path from its assignment to here; as well, if not (in a branch, a loop)."""
        m = scope.fn.module
        defs = [d.node for d in m.chains().get(use, [])]
        for fn, name, st, held in self.overrides:
            if fn.node is not scope.fn.node or name != use.id or st.lineno >= use.lineno \
                    or any(getattr(d, "lineno", 0) > st.lineno for d in defs):
                continue
            if _blocks(m, st) <= _blocks(m, use):
                out = list(held)
            else:
                out = out + held
        return out

    def constant(self, m: Module, name: str, st: ast.Assign) -> str:
        ref = f"{m.name}:{name}"
        for sym in self.symbols.values():
            if sym.code == [ref]:
                return sym.key
        try:
            value = ast.literal_eval(st.value)
        except ValueError:
            value = None
        key = self.symbol(name, "constant" if value is not None else "input")
        self.symbols[key].code, self.symbols[key].value = [ref], value
        return key

    def keys(self, held: list[Held]) -> frozenset[str]:
        out: set[str] = set()
        for h in held:
            if isinstance(h, frozenset):
                out |= h
            elif isinstance(h, Plain):
                out.add(self.gap(h))
            elif isinstance(h, Record):
                out |= self.keys([x for held in h.fields.values() for x in held])
        return frozenset(out)

    def reads(self, scope: Scope, e: ast.AST, outside: set[int] | None = None) -> frozenset[str]:
        """The symbols an expression reads: its variables (traced to what they hold), the settings fields it reads,
        and the results of the steps called inside it. A lambda's parameters are its own. outside: only what
        assignments not among these nodes made (what comes into a loop)."""
        own = {a.arg for lam in ast.walk(e) if isinstance(lam, ast.Lambda) for a in lam.args.args}
        out: set[str] = set()

        def visit(n: ast.AST) -> None:
            if isinstance(n, ast.Call) and n in self.results:            # a step (or a group): what it gives
                out.update(self.keys([h for item in self.results[n] for h in item]))
                return
            if isinstance(n, (ast.ListComp, ast.GeneratorExp, ast.SetComp)) and all(
                    t in self.held for g in n.generators for t in ast.walk(g.target)
                    if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)):
                # its variables bound (clean code): what it runs over is read through them, as far as it is used
                for part in (n.elt, *(c for g in n.generators for c in g.ifs)):
                    visit(part)
                return
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
                held = self.resolve(scope, n.value, outside) if n.value.id not in own else []
                field = _field_of(held, n.attr)
                if field is not None:                   # a record's field: what it was given
                    out.update(self.keys(field))
                    return
                s = next((h for h in held if isinstance(h, Settings)), None)
                if s is not None:
                    if n.attr in _fields(s.node):
                        out.add(self.setting(s, n.attr))
                    return
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in own:
                out.update(self.keys(self.resolve(scope, n, outside)))
                return
            for c in ast.iter_child_nodes(n):
                visit(c)
        visit(e)
        return frozenset(out)

    def identity(self, scope: Scope, e: ast.expr) -> str | None:
        """The one symbol an expression is, as it is (not computed from it): a variable holding exactly it, a
        step's result, or a settings field. For names=, which writes a parameter as that symbol."""
        if isinstance(e, ast.Name):
            held = self.resolve(scope, e)
        elif isinstance(e, ast.Call) and e in self.results:
            held = [h for item in self.results[e] for h in item]
        elif isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name):
            outer = self.resolve(scope, e.value)
            field = _field_of(outer, e.attr)
            if field is None:
                s = next((h for h in outer if isinstance(h, Settings)), None)
                return self.setting(s, e.attr) if s is not None and e.attr in _fields(s.node) else None
            held = field
        else:
            return None
        if any(isinstance(h, (Derived, Settings, Record)) for h in held):
            return None
        keys = self.keys(held)
        return next(iter(keys)) if len(keys) == 1 else None

    def held_by(self, scope: Scope, e: ast.expr, name: str, st: ast.stmt) -> list[list[Held]]:
        """What an expression gives, item by item (a tuple's items; a call's results), for binding to `name`."""
        if e in self.results:
            return self.results[e]
        if isinstance(e, ast.Name):
            return [self.resolve(scope, e)]
        if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) \
                and (field := _field_of(self.resolve(scope, e.value), e.attr)) is not None:
            return [field]                              # a record's field: what it was given
        if isinstance(e, ast.Call) and isinstance(t := self.target(scope.fn, e.func), ast.ClassDef) \
                and _passes_through(t):                  # a dataclass: holds what it is given
            if self.lines:                              # clean code: field by field
                names = list(_fields(t))
                given = [*zip(names, e.args), *((k.arg, k.value) for k in e.keywords if k.arg)]
                if len(e.args) <= len(names) and all(n in names for n, _ in given):
                    return [[Record({n: [h for item in self.held_by(scope, v, n, st) for h in item]
                                     for n, v in given})]]
            return [[Derived(self.reads(scope, e))]]
        if isinstance(e, ast.Tuple):
            return [x[0] if len(x) == 1 else [h for i in x for h in i]
                    for x in (self.held_by(scope, el, name, st) for el in e.elts)]
        return [[Plain(scope.fn, st, name, scope.group)]]

    def note(self, scope: Scope, text: str) -> None:
        line = f"{scope.fn.module.name}:{scope.fn.node.name}: {text}"
        if line not in self.notes:
            self.notes.append(line)

    # the walk

    def function(self, f: Function, params: dict[str, list[Held]], group: str, per: str | None,
                 depth: int, axes: dict[str, str] | None = None,
                 constants: dict[str, object] | None = None, when: list[str] | None = None) -> list[list[Held]]:
        """Draft the steps in f's body; returns what it returns (one item per element of a returned tuple). axes:
        its axis names bound to the caller's indices (unify()); constants: its parameters this call gives a
        literal, or leaves to a literal default."""
        scope = Scope(f, group, per, depth)
        scope.params = params
        scope.constants = dict(constants or {})
        scope.when = list(when or [])                   # opened under an if: its steps are done under it too
        scope.axes = dict(axes or {})
        for n in ast.walk(f.node):
            if isinstance(n, ast.FunctionDef) and n is not f.node and any(
                    isinstance(c, ast.Call) and self.is_step(f, c) for c in ast.walk(n)):
                self.note(scope, f"the function {n.name} defined inside it (line {n.lineno}) calls the "
                                    "package's functions: those calls are not drafted")
        self.body(scope, f.node.body)
        changes = [(n, t) for n, t in _in_place(f.node) if (f.ref, n) not in self.drawn]
        if changes:
            shown = ", ".join(f"{n} (`{t}`)" for n, t in changes[:4]) + (", ..." if len(changes) > 4 else "")
            self.note(scope, f"may change values in place at line{'s' * (len(changes) > 1)} {shown}: what "
                                     "changes there has no edge")
        return scope.returned

    def body(self, scope: Scope, statements: list[ast.stmt]) -> None:
        for st in statements:
            self.statement(scope, st)

    def statement(self, scope: Scope, st: ast.stmt) -> None:
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom,
                           ast.Global, ast.Nonlocal, ast.Pass, ast.Break, ast.Continue)):
            return
        if self.lines and isinstance(st, (ast.For, ast.While)) and not st.orelse and self.loop(scope, st):
            return
        if isinstance(st, ast.If):
            self.calls(scope, st.test, st)
            known = _evaluated(st.test, scope.constants)
            if known is not None:
                # decided by literals alone (this call's, or the defaults it leaves): only the branch taken
                skipped = st.orelse if known else st.body
                self.skipped |= {id(n) for b in skipped for n in ast.walk(b)}
                if skipped:
                    given = ", ".join(f"{k} = {scope.constants[k]!r}" for k in sorted(
                        {n.id for n in ast.walk(st.test) if isinstance(n, ast.Name)} & set(scope.constants)))
                    self.note(scope, f"the branch at line {skipped[0].lineno} is not drafted: `{ast.unparse(st.test)}` "
                                     f"is {known} here ({given})")
                self.body(scope, st.body if known else st.orelse)
                return
            if self.lines and self.decides(scope, st):
                return
            test = self.condition(scope, st.test)
            yes, no = copy.copy(scope), copy.copy(scope)
            yes.when, no.when = [*scope.when, ast.unparse(test)], [*scope.when, _negated(test)]
            self.body(yes, st.body)
            self.body(no, st.orelse)
            return
        if isinstance(st, (ast.For, ast.AsyncFor)):
            self.calls(scope, st.iter, st)
            index = next(iter(_names(st.target)), None)
            for n in (x for x in ast.walk(st.target) if isinstance(x, ast.Name)):
                # the loop's first variable is the index its steps are done per value of; the others are
                # elements of what it runs over: not followed
                self.held[n] = [] if n.id == index else [Plain(scope.fn, st, n.id, scope.group)]
            inner = copy.copy(scope)
            inner.per = scope.per or index
            self.body(inner, st.body)
            self.body(scope, st.orelse)
            return
        if isinstance(st, (ast.If, ast.While)):
            self.calls(scope, st.test, st)
            self.body(scope, st.body)
            self.body(scope, st.orelse)
            return
        if isinstance(st, (ast.With, ast.AsyncWith)):
            for w in st.items:
                self.calls(scope, w.context_expr, st)
                if w.optional_vars is not None:
                    made = [[Plain(scope.fn, st, n, scope.group)] for n in _names(w.optional_vars)]
                    self.bind(scope, w.optional_vars, made, st)  # type: ignore[arg-type]
            self.body(scope, st.body)
            return
        if isinstance(st, ast.Try):
            for part in (st.body, *(h.body for h in st.handlers), st.orelse, st.finalbody):
                self.body(scope, part)
            return
        value = getattr(st, "value", None)
        if not isinstance(value, ast.expr):
            for e in (n for n in ast.iter_child_nodes(st) if isinstance(n, ast.expr)):
                self.calls(scope, e, st)
            return
        self.calls(scope, value, st)
        if isinstance(st, ast.Return) and self.lines and not isinstance(value, ast.Tuple):
            # one value, made by the return's own code (not a tuple: the line would stand for every item)
            by_line = self.returned_line(scope, st, value)
            if by_line is not None:
                scope.returned[:] = _join(scope.returned, [by_line])
                return
        if isinstance(st, ast.Return):
            items = value.elts if isinstance(value, ast.Tuple) else [value]
            got = [h for e in items for h in (self.held_by(scope, e, scope.fn.node.name, st)
                                             if not isinstance(e, ast.Tuple) else [[]])]
            if not isinstance(value, ast.Tuple):
                got = self.held_by(scope, value, scope.fn.node.name, st)
            scope.returned[:] = _join(scope.returned, got)
            return
        if isinstance(st, ast.Expr):
            return
        if isinstance(st, ast.AugAssign):
            self.bind(scope, st.target, [[Plain(scope.fn, st, n, scope.group)] for n in _names(st.target)], st)
            return
        targets = st.targets if isinstance(st, ast.Assign) else [st.target]  # type: ignore[attr-defined]
        if self.lines and self.line(scope, st, targets, value):
            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and self.hints:
                self.annotated(scope, st)
            return
        for t in targets:
            if isinstance(t, (ast.Name, ast.Tuple, ast.List)):     # (x[...] = ...: a change in place, noted)
                self.bind(scope, t, self.held_by(scope, value, t.id if isinstance(t, ast.Name) else "", st), st)
        if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and self.hints:
            self.annotated(scope, st)

    def line(self, scope: Scope, st: ast.stmt, targets: list[ast.expr], value: ast.expr) -> bool:
        """In clean code (--clean): an assignment made by code that is not a step (numpy, arithmetic, another package's call,
        unpacking) as a step of its own, its code the line, its inputs what the line reads (as a call's are), its
        outputs the variables assigned. Not an assignment into part of a value (x[i] = e: a change in place), nor
        one that reads no symbol (a constant: still a gap, made by the code)."""
        if len(targets) != 1 or not _variables(targets[0]):
            return False
        assigned = [n for n in ast.walk(targets[0]) if isinstance(n, ast.Name) and n.id != "_"]
        kept = _converted(value)
        if kept is not None and isinstance(targets[0], ast.Name):
            # a conversion that keeps the values (np.asarray(x, float), x.astype(...)): x, as codetex writes it
            got = self.held_by(scope, kept, targets[0].id, st)
            if not any(isinstance(h, Plain) for item in got for h in item):
                self.held[targets[0]] = [h for item in got for h in item]
                return True
        held = self.held_by(scope, value, "", st)
        unpacked = isinstance(targets[0], (ast.Tuple, ast.List)) and len(held) == 1 and len(assigned) > 1
        if not unpacked and not any(isinstance(h, Plain) for item in held for h in item):
            return False                                # a step's result or a variable's value: bound as it is
        inputs = self.reads(scope, value)
        if not inputs:
            return False
        per = self.per_element.get(value, scope.per)    # a comprehension's line: per element, as it is done
        outs = [self.symbol(n.id, per=per) for n in assigned]
        names = {n.id: same for n in ast.walk(value) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                 and (same := self.identity(scope, n)) is not None}
        names |= {n.id: o for n, o in zip(assigned, outs)}
        anchor = _anchor(scope.fn, st)
        label = ", ".join(n.id for n in assigned)
        key = self._key("_".join(n.id for n in assigned), {x.key for x in self.steps})
        self.record(scope, Step(key, label.replace("_", " "), sorted(inputs), outs,
                               [f"{scope.fn.ref}#{anchor}" if anchor else scope.fn.ref], scope.group, per,
                               None, names))
        for n, o in zip(assigned, outs):
            self.held[n] = [frozenset([o])]
        if self.hints and isinstance(value, ast.Name) and isinstance(targets[0], (ast.Tuple, ast.List)):
            # unpacking a parameter: each part shaped as the parameter's annotation says its items are
            for n, part in self.parts(scope.fn.module, targets[0], _param_annotation(scope.fn, value.id)):
                axes = self.axes(self.shape(scope.fn.module, part), scope.axes, self.param_axes(scope.fn) - set(
                    scope.axes) if scope.group else set(), scope.fn.ref)
                o = outs[assigned.index(n)]
                self.symbols[o].dims = self.symbols[o].dims or axes
        return True

    def parts(self, m: Module, t: ast.expr, ann: ast.expr | None) -> list[tuple[ast.Name, ast.expr]]:
        """The annotation of each variable an unpacking assigns, from the annotation of what it unpacks: a
        tuple's items in turn (tuple[A, B]), or every element alike (Sequence[A], list[A], tuple[A, ...])."""
        if isinstance(t, ast.Name):
            return [(t, ann)] if ann is not None else []
        if not isinstance(t, (ast.Tuple, ast.List)) or ann is None:
            return []
        m, ann = self.annotation(m, ann)
        items = shapes.items(ann, lambda name: None)
        if items is not None and len(items) == len(t.elts):
            return [p for e, a in zip(t.elts, items) for p in self.parts(m, e, a)]
        element = _element(ann)
        return [p for e in t.elts for p in self.parts(m, e, element)] if element is not None else []

    def returned_line(self, scope: Scope, st: ast.Return, e: ast.expr) -> list[Held] | None:
        """In clean code (--clean): the value a function returns, made there by other code (return Model(a, b)): a step of its
        own, as an assignment's is; its symbol is named for what it calls (Model) or the function."""
        if not any(isinstance(h, Plain) for item in self.held_by(scope, e, "", st) for h in item):
            return None
        inputs = self.reads(scope, e)
        if not inputs:
            return None
        called = _dotted_name(e.func) if isinstance(e, ast.Call) else None
        name = called.rpartition(".")[2] if called else scope.fn.node.name
        out = self.symbol(name, per=scope.per)
        names = {n.id: same for n in ast.walk(e) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                 and (same := self.identity(scope, n)) is not None} | {"return": out}
        anchor = _anchor(scope.fn, st)
        key = self._key(name, {x.key for x in self.steps})
        self.record(scope, Step(key, name.replace("_", " "), sorted(inputs), [out],
                               [f"{scope.fn.ref}#{anchor}" if anchor else scope.fn.ref], scope.group, scope.per,
                               None, names))
        return [frozenset([out])]

    def annotated(self, scope: Scope, st: ast.AnnAssign) -> None:
        """x: Float[np.ndarray, "cells genes"] = ...: the value assigned has that shape (bound as the function's
        own axis names are; in the entry function, the names are its own indices)."""
        assert isinstance(st.target, ast.Name) and st.value is not None
        axes = self.axes(self.shape(scope.fn.module, st.annotation), scope.axes, self.param_axes(scope.fn) - set(
            scope.axes) if scope.group else set(), scope.fn.ref)
        if not axes:
            return
        for h in self.held.get(st.target, []):          # what the assignment bound
            if isinstance(h, Plain):
                h.dims = h.dims or axes
            elif type(h) is frozenset and len(h) == 1:
                sym = self.symbols[next(iter(h))]
                sym.dims = sym.dims or axes

    def bind(self, scope: Scope, t: ast.expr, items: list[list[Held]], st: ast.stmt) -> None:
        """What each variable of an assignment's target holds: a tuple target takes the items in turn."""
        if isinstance(t, ast.Name):
            got = [h for item in items for h in item] if len(items) != 1 else items[0]
            plain = [h for h in got if isinstance(h, Plain)]
            renamed = Plain(plain[0].fn, plain[0].st, t.id, plain[0].group) if plain else None
            if renamed is not None:
                renamed.dims = plain[0].dims                # (its shape, if stated, stays)
            self.held[t] = [h for h in got if not isinstance(h, Plain)] + ([renamed] if renamed else [])
            return
        if isinstance(t, (ast.Tuple, ast.List)):
            if len(items) == len(t.elts):
                for el, item in zip(t.elts, items):
                    self.bind(scope, el, [item], st)
            else:
                for el in t.elts:
                    for n in _names(el):
                        node = next(x for x in ast.walk(el) if isinstance(x, ast.Name) and x.id == n)
                        self.held[node] = [Plain(scope.fn, st, n, scope.group)]

    def calls(self, scope: Scope, e: ast.expr, st: ast.stmt) -> None:
        """The steps (and groups) of the package calls in an expression, innermost first."""
        if isinstance(e, ast.Lambda):
            return
        if self.lines and isinstance(e, (ast.ListComp, ast.GeneratorExp)) and len(e.generators) == 1 \
                and not e.generators[0].ifs and not e.generators[0].is_async:
            self.comprehension(scope, e, st)
            return
        for c in ast.iter_child_nodes(e):
            if isinstance(c, ast.expr):
                self.calls(scope, c, st)
            elif isinstance(c, ast.comprehension):
                for x in (c.iter, *c.ifs):
                    self.calls(scope, x, st)
        if not isinstance(e, ast.Call):
            return
        t = self.target(scope.fn, e.func)
        if t is None or isinstance(t, ast.ClassDef) and _passes_through(t):
            return
        if isinstance(t, Function) and self.opens(t, scope.depth):
            self.results[e] = self.open(scope, t, e, st)
            return
        self.step(scope, t, e, st)

    def loop(self, scope: Scope, st: ast.For | ast.While) -> bool:
        """In clean code, a for loop whose body carries values from turn to turn (assigns a variable and reads it
        back: an iteration): one step, as an iteration is one action. Its inputs are what comes into the loop (the
        start values, and what every turn reads), its outputs the variables it assigns that are read after it; its
        equation is the loop written as an algorithm (codetex). The calls inside it are part of it, not steps. False if it carries
        nothing: its steps are then done per its index, as any loop's."""
        if not _carried(st):
            return False
        inside = {id(n) for n in ast.walk(st)}
        inputs = self.reads(scope, st, outside=inside)
        later = {n.id for n in _own_nodes(scope.fn.node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                 and n.lineno > (st.end_lineno or st.lineno)}
        made = [n for n in ast.walk(st) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]
        own = _names(st.target) if isinstance(st, ast.For) else []
        outputs = [v for v in dict.fromkeys(n.id for n in made) if v in later and v not in own]
        outs = [self.symbol(v, per=scope.per) for v in outputs]
        for n in made:
            if n.id in outputs:
                self.held[n] = [frozenset([outs[outputs.index(n.id)]])]
                before = {id(a) for a in ast.walk(scope.fn.node.args) if isinstance(a, ast.arg) and a.arg == n.id} | {
                    id(x) for x in _own_nodes(scope.fn.node) if isinstance(x, ast.Name) and x.id == n.id
                    and isinstance(x.ctx, ast.Store) and x.lineno < st.lineno}
                self.exits[id(n)] = (st.end_lineno or st.lineno, before)
        names: dict[str, str] = {}                      # what comes in as one symbol: written as it
        for x in ast.walk(st):
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load) and x.id not in outputs:
                held = self.resolve(scope, x, outside=inside)
                keys = self.keys(held)
                if held and len(keys) == 1 and not any(isinstance(h, (Derived, Settings, Record)) for h in held):
                    names[x.id] = next(iter(keys))
        anchor = _anchor(scope.fn, st)
        key = self._key("_".join(outputs) or "loop", {x.key for x in self.steps})
        self.record(scope, Step(key, ", ".join(outputs).replace("_", " ") or "loop", sorted(inputs), outs,
                                [f"{scope.fn.ref}#{anchor}" if anchor else scope.fn.ref], scope.group, scope.per,
                                None, names))
        return True

    def decides(self, scope: Scope, st: ast.If) -> bool:
        """In clean code, an if that decides values: its branches only assign, and each value it assigns that is
        read after it either had one before (a branch that does not assign it keeps it) or is assigned by both
        branches. One step, its equation the cases (codetex): its inputs what the test and the branches read and
        the values kept, its outputs those values. The calls inside it are part of it. False if the if is not one:
        its steps are then each done under its condition (a guard)."""
        if any(not isinstance(b, (ast.Assign, ast.AnnAssign, ast.Pass)) for b in (*st.body, *st.orelse)):
            return False
        inside = {id(n) for n in ast.walk(st)}
        later = {n.id for n in _own_nodes(scope.fn.node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                 and n.lineno > (st.end_lineno or st.lineno)}
        made = sorted((n for n in ast.walk(st) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)),
                      key=lambda n: (n.lineno, n.col_offset))
        outputs = [v for v in dict.fromkeys(n.id for n in made) if v in later]
        both = {n.id for n in made if any(n is x for b in st.body for x in ast.walk(b))} & \
            {n.id for n in made if any(n is x for b in st.orelse for x in ast.walk(b))}
        uses = {v: next((n for n in _own_nodes(scope.fn.node) if isinstance(n, ast.Name) and n.id == v
                         and isinstance(n.ctx, ast.Load) and n.lineno > (st.end_lineno or st.lineno)), None)
                for v in outputs}
        kept = {v: self.resolve(scope, u, outside=inside) if u is not None else [] for v, u in uses.items()}
        if not outputs or any(v not in both and not kept[v] for v in outputs):
            return False                                # a value only one branch makes: a guard on its steps
        inputs = set(self.reads(scope, st, outside=inside))
        for v in outputs:
            if v not in both:
                inputs |= self.keys(kept[v])            # a branch that does not assign it keeps it
        outs = [self.symbol(v, per=scope.per) for v in outputs]
        for v, o in zip(outputs, outs):                 # a value it may keep as it was: shaped as it was
            was = [self.symbols[k].dims for k in self.keys(kept[v])]
            if was and all(d == was[0] for d in was):
                self.symbols[o].dims = list(was[0])
        for n in made:
            if n.id in outputs:
                self.held[n] = [frozenset([outs[outputs.index(n.id)]])]
                before = {id(a) for a in ast.walk(scope.fn.node.args) if isinstance(a, ast.arg) and a.arg == n.id} | {
                    id(x) for x in _own_nodes(scope.fn.node) if isinstance(x, ast.Name) and x.id == n.id
                    and isinstance(x.ctx, ast.Store) and x.lineno < st.lineno}
                self.exits[id(n)] = (st.end_lineno or st.lineno, before)    # after it: what it decided
        names: dict[str, str] = {}
        for x in ast.walk(st):
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load):
                held = self.resolve(scope, x, outside=inside)
                keys = self.keys(held)
                if held and len(keys) == 1 and not any(isinstance(h, (Derived, Settings, Record)) for h in held):
                    names[x.id] = next(iter(keys))
            elif isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and (k := self.identity(scope, x)):
                names[ast.unparse(x)] = k
        names |= {v + "=": o for v, o in zip(outputs, outs)}   # how the step writes what it assigns
        anchor = _anchor(scope.fn, st)
        key = self._key("_".join(outputs), {x.key for x in self.steps})
        self.record(scope, Step(key, ", ".join(outputs).replace("_", " "), sorted(inputs), outs,
                                [f"{scope.fn.ref}#{anchor}" if anchor else scope.fn.ref], scope.group, scope.per,
                                None, names))
        return True

    def comprehension(self, scope: Scope, e: ast.ListComp | ast.GeneratorExp, st: ast.stmt) -> None:
        """In clean code, [f(x) for x in xs]: done per element, its steps repeated per= an index. The index is the
        enumerate's (for i, x in enumerate(xs)), or that of the values xs holds if they are repeated per one
        already (iterating [f.a for f in folds] is per fold again), or else the first variable's. An element of a
        value repeated per that index is that value; unpacking an element of any other value is a step of its own,
        repeated per the index (its variables shaped as the annotation of what is iterated says its items are)."""
        g = e.generators[0]
        self.calls(scope, g.iter, st)
        target, over = g.target, g.iter
        index = None
        if isinstance(over, ast.Call) and _dotted_name(over.func) == "enumerate" and len(over.args) == 1 \
                and isinstance(target, ast.Tuple) and len(target.elts) == 2 and isinstance(target.elts[0], ast.Name):
            index = target.elts[0].id
            self.held[target.elts[0]] = []
            target, over = target.elts[1], over.args[0]
        # zip(xs, ys): each variable an element of its own collection
        zipped = isinstance(over, ast.Call) and _dotted_name(over.func) == "zip" and not over.keywords \
            and isinstance(target, ast.Tuple) and len(target.elts) == len(over.args)
        sources = list(over.args) if zipped and isinstance(over, ast.Call) else [over]
        targets = list(target.elts) if zipped and isinstance(target, ast.Tuple) else [target]
        helds = [[h for item in self.held_by(scope, x, "", st) for h in item] for x in sources]
        pers = {self.per_of(h) for held in helds for h in held}
        family = len(pers) == 1 and None not in pers
        if index is None:
            index = next(iter(pers)) if family else self.range_axis(scope, over) or \
                next(iter(_names(target)), None)
        inner = copy.copy(scope)
        inner.per = scope.per or index
        if inner.per and inner.per not in self.indices:
            self.indices[inner.per] = f"the comprehension at {scope.fn.module.file.name}:{e.lineno}"
        names = [n for n in ast.walk(target) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]
        held = [h for x in helds for h in x]
        ranged = isinstance(over, ast.Call) and _dotted_name(over.func) == "range" and isinstance(target, ast.Name)
        if ranged:
            self.held[target] = []                      # for g in range(n): g is the index itself, not a value
        elif family and all(isinstance(t, ast.Name) for t in targets):
            for t, x in zip(targets, helds):
                self.held[t] = x                        # an element of a family: the value, at that index
        elif names:
            inputs = self.keys(held)
            outs = [self.symbol(n.id, per=inner.per) for n in names]
            for n, o in zip(names, outs):
                self.held[n] = [frozenset([o])]
            if self.hints and isinstance(over, ast.Name):
                ann = _param_annotation(scope.fn, over.id)
                for n, part in self.parts(scope.fn.module, target, _element(ann)):
                    axes = self.axes(self.shape(scope.fn.module, part), scope.axes, self.param_axes(scope.fn) - set(
                        scope.axes) if scope.group else set(), scope.fn.ref)
                    o = outs[names.index(n)]
                    self.symbols[o].dims = self.symbols[o].dims or axes
            if inputs:
                anchor = _anchor(scope.fn, st)
                key = self._key("_".join(n.id for n in names), {x.key for x in self.steps})
                self.record(scope, Step(key, ", ".join(n.id for n in names).replace("_", " "), sorted(inputs), outs,
                                       [f"{scope.fn.ref}#{anchor}" if anchor else scope.fn.ref], scope.group,
                                       inner.per, None, {n.id: o for n, o in zip(names, outs)}))
        self.calls(inner, e.elt, st)
        self.per_element[e] = inner.per
        self.results[e] = self.held_by(inner, e.elt, "", st)

    def range_axis(self, scope: Scope, over: ast.expr) -> str | None:
        """for g in range(x.shape[1]) (or range(len(x))), x annotated with its axes: the axis it runs over (genes),
        so that what is done per g is done per gene."""
        if not (isinstance(over, ast.Call) and _dotted_name(over.func) == "range" and len(over.args) == 1):
            return None
        n = over.args[0]
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute) and n.value.attr == "shape" \
                and isinstance(n.value.value, ast.Name) and isinstance(n.slice, ast.Constant) \
                and isinstance(n.slice.value, int):
            name, k = n.value.value.id, n.slice.value
        elif isinstance(n, ast.Call) and _dotted_name(n.func) == "len" and len(n.args) == 1 \
                and isinstance(n.args[0], ast.Name):
            name, k = n.args[0].id, 0
        else:
            return None
        axes = self.shape(scope.fn.module, _param_annotation(scope.fn, name))
        if not axes or not -len(axes) <= k < len(axes):
            return None
        return scope.axes.get(axes[k], axes[k])

    def condition(self, scope: Scope, test: ast.expr) -> ast.expr:
        """An if's test in the graph's terms: each variable (and settings field) that holds one symbol written as
        that symbol's key (embed's `correct`, given the setting correct: correct). Others stay as the code has them."""
        drafter = self

        class Keys(ast.NodeTransformer):
            def visit_Attribute(self, n: ast.Attribute) -> ast.AST:
                k = drafter.identity(scope, n) if isinstance(n.value, ast.Name) else None
                return ast.Name(k, ast.Load()) if k else self.generic_visit(n)

            def visit_Name(self, n: ast.Name) -> ast.AST:
                k = drafter.identity(scope, n)
                return ast.Name(k, ast.Load()) if k else n
        return Keys().visit(copy.deepcopy(test))  # type: ignore[no-any-return]

    def record(self, scope: Scope, step: Step) -> None:
        """A step, with the conditions it is done under."""
        step.condition = " and ".join(scope.when)
        self.steps.append(step)

    def per_of(self, h: Held) -> str | None:
        """The index a value is repeated per, if all of it is repeated per one (a symbol's, or every symbol a
        record's fields hold); None if not, or if not all of it is a symbol."""
        if isinstance(h, Record):
            pers = {self.per_of(x) for held in h.fields.values() for x in held}
            return next(iter(pers)) if len(pers) == 1 else None
        if type(h) is not frozenset or not h:
            return None
        pers = {self.symbols[k].per for k in h}
        return next(iter(pers)) if len(pers) == 1 else None

    def open(self, scope: Scope, f: Function, call: ast.Call, st: ast.stmt) -> list[list[Held]]:
        axes = self.unify(scope, f, call)
        bound: dict[str, list[Held]] = {}
        for p, a in _given(f, call):
            same = self.identity(scope, a) if not isinstance(a, ast.Name) else None
            bound[p] = self.resolve(scope, a) if isinstance(a, ast.Name) else [frozenset([same])] if same \
                else [Derived(self.reads(scope, a))]
        label = f.node.name.replace("_", " ")
        path = f"{scope.group}/{label}" if scope.group else label
        base, n = path, 2
        while path in self.groups:
            path, n = f"{base} {n}", n + 1
        doc = _docs(f.node)
        text = ast.get_docstring(f.node) or ""
        summary = " ".join(text.split("\n\n")[0].split())        # its first paragraph
        self.groups[path] = f"{summary} ({f.ref})" if summary else f"TODO ({f.ref})"
        self.opening.append(f.ref)
        returned = self.function(f, bound, path, scope.per, scope.depth - 1, axes, _literals(f, call), scope.when)
        self.opening.pop()
        parts = [self.shape(f.module, f.node.returns)] if len(returned) == 1 else \
            self.shape_items(f.module, f.node.returns) or []
        params = self.param_axes(f)
        for item, shape in zip(returned, parts) if len(parts) == len(returned) else ():
            if len(item) != 1:
                continue
            if isinstance(item[0], Plain):              # made by its code, shaped as it says
                item[0].dims = item[0].dims or self.axes(shape, axes, params, f.ref)
            elif type(item[0]) is frozenset and len(item[0]) == 1:
                self.give(next(iter(item[0])), shape, axes, f.ref, params)
        documented = doc.many_returns if doc is not None else []
        for item, r in zip(returned, documented) if len(documented) == len(returned) else ():
            keys = self.keys([h for h in item if type(h) is frozenset])
            if len(keys) == 1 and len(item) == 1:       # it returns a step's result as it is: that is described
                self.described(next(iter(keys)), r.description, r.type_name, f.ref)
        changed = self.changes(f)
        first = {p: line for p, line in changed.items()}

        annotated = {a.arg: a.annotation for a in (*f.node.args.posonlyargs, *f.node.args.args,
                                                   *f.node.args.kwonlyargs)}
        param_of = {v.id: p for p, v in _given(f, call) if isinstance(v, ast.Name)}

        def made(variable: str) -> Held:                # changed by the code inside the group: a gap
            line = min(first.values())
            at = next((n for n in _own_nodes(f.node) if isinstance(n, ast.stmt) and n.lineno == line), f.node)
            gap = Plain(f, at, variable, path)  # type: ignore[arg-type]
            gap.dims = self.axes(self.shape(f.module, annotated.get(param_of.get(variable, ""))), axes,
                                 self.param_axes(f), f.ref)
            return gap
        self.changed_by(scope, f, call, st, made)
        return returned

    def step(self, scope: Scope, t: Function | ast.ClassDef, call: ast.Call, st: ast.stmt) -> None:
        if isinstance(t, Function):
            ref, name = t.ref, (t.cls + "." if t.cls else "") + t.node.name
            params = t.params
        else:
            ref, name, params = f"{self.package.owner(t).name}:{t.name}", t.name, []
        given = [*((params[k] if k < len(params) else None, a) for k, a in enumerate(call.args)),
                 *((k.arg, k.value) for k in call.keywords)]
        inputs: set[str] = set()
        names: dict[str, str] = {}
        if isinstance(t, Function) and t.cls and isinstance(call.func, ast.Attribute):
            inputs |= self.reads(scope, call.func.value)        # a method: what it is called on
        for p, a in given:
            inputs |= self.reads(scope, a)
            same = self.identity(scope, a)
            if p and same:                              # the parameter holds that symbol, as it is
                names[p] = same
        per = scope.per
        if per and per not in self.indices:
            self.indices[per] = f"the loop around {ref}"
        outs = [self.symbol(n, per=per) for n in _assigned(st, call, self.lines)] or \
            ([] if isinstance(st, ast.Expr) and st.value is call else [self.symbol(name, per=per)])
        anchor = _anchor(scope.fn, st)
        code = [ref] + ([f"{scope.fn.ref}#{anchor}"] if anchor else [])
        key = self._key(name.replace(".", "_"), {s.key for s in self.steps})
        result: list[list[Held]] = [[frozenset([o])] for o in outs] if len(outs) > 1 else [[frozenset(outs)]]
        bound = self.unify(scope, t, call) if isinstance(t, Function) else {}
        named_axes = self.param_axes(t) if isinstance(t, Function) else set()
        if isinstance(t, Function) and outs:            # its results' shapes, as its return annotation states
            if len(outs) == 1:
                self.give(outs[0], self.shape(t.module, t.node.returns), bound, ref, named_axes)
            elif (parts := self.shape_items(t.module, t.node.returns)) is not None and len(parts) == len(outs):
                for o, axes in zip(outs, parts):
                    self.give(o, axes, bound, ref, named_axes)
        doc = _docs(t.node) if isinstance(t, Function) else None
        returned = doc.many_returns if doc is not None else []
        if returned and len(returned) == len(outs) or len(outs) == 1 and doc is not None and doc.returns:
            documented = returned if len(returned) == len(outs) else [doc.returns]  # type: ignore[union-attr]
            for o, r in zip(outs, documented):
                self.described(o, r.description, r.type_name, ref)
        if isinstance(t, Function):                 # the variables it changes in place: outputs of it too
            annotated = {a.arg: a.annotation for a in (*t.node.args.posonlyargs, *t.node.args.args,
                                                       *t.node.args.kwonlyargs)}
            param_of = {v.id: p for p, v in _given(t, call) if isinstance(v, ast.Name)}

            def made(variable: str) -> Held:
                changed = self.symbol(variable, per=per)
                outs.append(changed)                   # shaped as the parameter it was given for
                self.give(changed, self.shape(t.module, annotated.get(param_of.get(variable, ""))), bound, ref,
                          named_axes)
                return frozenset([changed])
            self.changed_by(scope, t, call, st, made)
        self.record(scope, Step(key, name.replace("_", " "), sorted(inputs), outs, code, scope.group, per, call,
                               names))
        if isinstance(t, Function):
            self.steps[-1].relations = _relations(t)
        self.results[call] = result


class Scope:
    """Where the walk is: the function, its group, the loop's index, how many levels may still be opened up, what
    its parameters hold, and what it returns."""

    def __init__(self, fn: Function, group: str = "", per: str | None = None, depth: int = 0):
        self.fn, self.group, self.per, self.depth = fn, group, per, depth
        self.params: dict[str, list[Held]] = {}
        self.returned: list[list[Held]] = []
        self.axes: dict[str, str] = {}              # its axis names, bound to the caller's indices
        self.when: list[str] = []                   # the tests of the ifs it is in (else: negated), of symbols
        self.constants: dict[str, object] = {}      # parameters given a literal (or left to a literal default)


def _join(a: list[list[Held]], b: list[list[Held]]) -> list[list[Held]]:
    """What two return statements give, together: item by item when they give as many."""
    if not a:
        return b
    if len(a) == len(b):
        return [x + y for x, y in zip(a, b)]
    return [[h for item in (*a, *b) for h in item]]


def _names(t: ast.expr) -> list[str]:
    """The variables an assignment target assigns to (or, for x[...] = and x.a =, changes)."""
    if isinstance(t, ast.Name):
        return [t.id]
    if isinstance(t, (ast.Tuple, ast.List)):
        return [n for e in t.elts for n in _names(e)]
    if isinstance(t, ast.Starred):
        return _names(t.value)
    if isinstance(t, (ast.Subscript, ast.Attribute)):
        return _names(t.value)
    return []


def _assigned(st: ast.stmt, call: ast.Call, clean: bool = False) -> list[str]:
    """The variables a statement assigns a call's result to: x = f(...), x, y = f(...), and x, y = f(a), g(b); in
    clean code also xs = [f(x) for x in ...], whose variable holds f's result per element."""
    if isinstance(st, ast.Assign) and len(st.targets) == 1:
        t, v = st.targets[0], st.value
    elif isinstance(st, ast.AnnAssign) and st.value is not None:
        t, v = st.target, st.value
    else:
        return []
    if (v is call or clean and isinstance(v, (ast.ListComp, ast.GeneratorExp)) and v.elt is call) \
            and isinstance(t, ast.Name):
        return [t.id]
    if v is call and isinstance(t, (ast.Tuple, ast.List)) and all(isinstance(e, ast.Name) for e in t.elts):
        return [e.id for e in t.elts if isinstance(e, ast.Name) and e.id != "_"]
    if isinstance(v, ast.Tuple) and isinstance(t, (ast.Tuple, ast.List)) and len(v.elts) == len(t.elts):
        for e, el in zip(t.elts, v.elts):
            if el is call and isinstance(e, ast.Name) and e.id != "_":
                return [e.id]
    return []


MUTATING = {   # methods that change the value they are called on, by its type (with --hints)
    "list": {"append", "extend", "insert", "remove", "pop", "clear", "sort", "reverse"},
    "dict": {"update", "pop", "popitem", "clear", "setdefault"},
    "set": {"add", "update", "discard", "remove", "clear", "pop"},
    "array": {"sort", "fill", "resize", "put", "partition", "itemset"},
}


def _docs(node: ast.FunctionDef | ast.ClassDef) -> Any:
    """A function's or class's docstring, read into its sections (numpydoc, Google or Sphinx style), or None."""
    text = ast.get_docstring(node)
    if not text:
        return None
    try:
        import docstring_parser
        return docstring_parser.parse(text)
    except Exception:                                   # noqa: BLE001 (not a docstring it can read: none)
        return None


def _documented(doc: Any, name: str, typed: bool = False) -> Any:
    """What a docstring's Parameters (or Attributes) section says of a name: its description, or with typed,
    (description, type)."""
    if doc is None:
        return None
    for p in doc.params:
        if p.arg_name == name:
            return (p.description, p.type_name) if typed else p.description
    return None


def _comment(lines: list[str], line: int) -> str | None:
    """The comment at the end of a line of code (`window_pseudo: float = 5  # pseudocounts ...`)."""
    if not 0 < line <= len(lines):
        return None
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(lines[line - 1]).readline))
    except (tokenize.TokenError, SyntaxError):
        return None
    text = next((t.string for t in tokens if t.type == tokenize.COMMENT), None)
    return text.lstrip("#").strip() or None if text else None


def _shape(type_name: str | None) -> list[str] | None:
    """The indices of a shape a docstring's type names: `ndarray of shape (n_cells, n_genes)`, `array, shape
    (n, p)`: names only (a number, or anything else, says nothing of the index)."""
    found = re.search(r"\bshape\s*[=:]?\s*\(([^()]*)\)", type_name or "")
    if not found:
        return None
    names = [x.strip() for x in found.group(1).split(",") if x.strip()]
    return names if names and all(n.isidentifier() and not keyword.iskeyword(n) for n in names) else None


def _given(f: Function, call: ast.Call) -> list[tuple[str, ast.expr]]:
    """Each parameter of f a call gives an argument for, with the argument."""
    params = f.params
    return [*zip(params, call.args), *((k.arg, k.value) for k in call.keywords if k.arg and k.arg in params)]


def _root(e: ast.expr) -> ast.expr:
    """x in x[i].a[j]: what a store into part of a value stores into."""
    while isinstance(e, (ast.Subscript, ast.Attribute)):
        e = e.value
    return e


def _load(t: ast.Name) -> ast.Name:
    return t                    # (beniget chains a target name of x += e to its earlier definitions too)


def _module_call(m: Module, c: ast.Call, module: str) -> bool:
    """A call of a function of `module` (np.clip, with np imported as numpy)."""
    root = c.func
    while isinstance(root, ast.Attribute):
        root = root.value
    return isinstance(root, ast.Name) and m.imports.get(root.id, "").split(".")[0] == module


def _dotted_name(e: ast.expr) -> str | None:
    if isinstance(e, ast.Name):
        return e.id
    if isinstance(e, ast.Attribute):
        head = _dotted_name(e.value)
        return None if head is None else f"{head}.{e.attr}"
    return None


def _is_static(fn: ast.FunctionDef) -> bool:
    return any(_dotted_name(d) == "staticmethod" for d in fn.decorator_list)


def _blocks(m: Module, node: ast.AST) -> set[int]:
    """The compound statements (if, for, while, try, with) a node is inside, up to its function."""
    out = set()
    parents = m.parents()
    node = parents.get(node)  # type: ignore[assignment]
    while node is not None and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)):
            out.add(id(node))
        node = parents.get(node)  # type: ignore[assignment]
    return out


def _in_place(fn: ast.FunctionDef) -> list[tuple[int, str]]:
    """The lines of a function's own code that may change a value in place, unseen by its bindings: updates
    (x += e), stores into part of a value (x[i] = e, x.a = e), out= arguments, and calls whose result is unused
    (a.sort(), xs.append(e), f(a)); printing and logging excepted."""
    out = []
    for st in (n for n in _own_nodes(fn) if isinstance(n, ast.stmt)):
        targets = st.targets if isinstance(st, ast.Assign) else [st.target] if isinstance(st, ast.AnnAssign) else []
        stores = any(isinstance(t, (ast.Subscript, ast.Attribute)) for tt in targets for t in ast.walk(tt)
                     if not isinstance(t, ast.Name))
        unused = isinstance(st, ast.Expr) and isinstance(st.value, ast.Call) and not _output_only(st.value)
        with_out = any(isinstance(c, ast.Call) and any(k.arg == "out" for k in c.keywords)
                       for e in ast.iter_child_nodes(st) if isinstance(e, ast.expr) for c in ast.walk(e))
        if isinstance(st, ast.AugAssign) or stores or unused or with_out:
            text = ast.unparse(st).splitlines()[0]
            out.append((st.lineno, text if len(text) <= 40 else text[:37] + "..."))
    return out


def _locals(fn: ast.FunctionDef) -> set[str]:
    """The names a function binds itself (parameters, assignments): not the module's functions of the same name."""
    out = {a.arg for a in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)}
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
    return out


def _own_nodes(fn: ast.FunctionDef) -> list[ast.AST]:
    """The nodes of a function's own body: not those of the functions and lambdas defined inside it, nor its
    decorators, defaults and annotations (run when it is defined, not when it is called)."""
    out: list[ast.AST] = []

    def visit(n: ast.AST) -> None:
        for c in ast.iter_child_nodes(n):
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            out.append(c)
            visit(c)
    for st in fn.body:
        out.append(st)
        visit(st)
    return out


def _carried(st: ast.For | ast.While) -> list[str]:
    """The variables a loop's body assigns (by its own statements) and reads: what may carry from turn to turn."""
    assigned = [n.id for b in st.body if isinstance(b, (ast.Assign, ast.AnnAssign)) for t in (
        b.targets if isinstance(b, ast.Assign) else [b.target]) for n in ast.walk(t)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]
    read = {n.id for b in st.body for n in ast.walk(b) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return [v for v in dict.fromkeys(assigned) if v in read]


def _relations(f: Function) -> list[dict[str, Any]]:
    """What a function declares it computes, with headway.clean's decorators (@maximizes(objective, over=("b", "d"),
    subject_to=(c,)), @equals("pkg.mod.func", rtol=...), ...): read from the code, each target a code reference."""
    from headway.clean import KINDS
    m = f.module
    out: list[dict[str, Any]] = []

    def ref(e: ast.expr) -> str | None:
        if isinstance(e, ast.Constant) and isinstance(e.value, str):
            module, _, name = e.value.rpartition(".")
            return f"{module}:{name}"
        if isinstance(e, ast.Name):
            if e.id in m.defs:
                return f"{m.name}:{e.id}"
            if e.id in m.imports:
                module, _, name = m.imports[e.id].rpartition(".")
                return f"{module}:{name}"
        return None
    for d in reversed(f.node.decorator_list):          # in the order they apply
        if not isinstance(d, ast.Call) or not (name := _dotted_name(d.func)):
            continue
        head, _, last = name.rpartition(".")
        full = m.imports.get(name, (m.imports.get(head, head) + "." + last) if head else "")
        if full.rpartition(".")[0] != "headway.clean" or last not in KINDS or not d.args:
            continue
        rel: dict[str, Any] = {"kind": last, "target": ref(d.args[0])}
        for k in d.keywords:
            if k.arg == "over" and isinstance(k.value, (ast.Tuple, ast.List)):
                rel["over"] = [e.value for e in k.value.elts if isinstance(e, ast.Constant)]
            elif k.arg == "subject_to" and isinstance(k.value, (ast.Tuple, ast.List)):
                rel["subject_to"] = [r for e in k.value.elts if (r := ref(e))]
            elif k.arg in ("tol", "rtol", "atol"):
                with contextlib.suppress(ValueError, TypeError, SyntaxError):
                    rel[k.arg] = ast.literal_eval(k.value)
        out.append(rel)
    return out


def _literals(f: Function, call: ast.Call) -> dict[str, object]:
    """f's parameters a call gives a literal (f(x, 2)), or leaves to a literal default (def f(x, k=None))."""
    a = f.node.args
    positional = [*a.posonlyargs, *a.args]
    out: dict[str, object] = {}
    for p, default in [*zip(positional[len(positional) - len(a.defaults):], a.defaults),
                       *zip(a.kwonlyargs, a.kw_defaults)]:
        if default is not None:
            with contextlib.suppress(ValueError, TypeError, SyntaxError):
                out[p.arg] = ast.literal_eval(default)
    for name, given in _given(f, call):
        out.pop(name, None)
        with contextlib.suppress(ValueError, TypeError, SyntaxError):
            out[name] = ast.literal_eval(given)
    return out


class _Unknown:
    pass


def _evaluated(test: ast.expr, constants: dict[str, object]) -> bool | None:
    """An if's test, if literals alone decide it (constants: variables known to hold one); None if not."""
    def value(e: ast.expr) -> object:
        if isinstance(e, ast.Constant):
            return e.value
        if isinstance(e, ast.Name):
            return constants.get(e.id, _Unknown)
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.Not):
            v = value(e.operand)
            return _Unknown if v is _Unknown else not v
        if isinstance(e, ast.BoolOp):
            vs = [value(x) for x in e.values]
            if _Unknown in vs:
                return _Unknown
            return all(vs) if isinstance(e.op, ast.And) else any(vs)
        if isinstance(e, ast.Compare) and len(e.ops) == 1:
            a, b = value(e.left), value(e.comparators[0])
            if a is _Unknown or b is _Unknown:
                return _Unknown
            op = e.ops[0]
            with contextlib.suppress(TypeError):
                return {ast.Is: lambda: a is b, ast.IsNot: lambda: a is not b, ast.Eq: lambda: a == b,
                        ast.NotEq: lambda: a != b, ast.Lt: lambda: a < b, ast.LtE: lambda: a <= b,  # type: ignore[operator]
                        ast.Gt: lambda: a > b, ast.GtE: lambda: a >= b}.get(type(op), lambda: _Unknown)()  # type: ignore[operator]
        return _Unknown
    v = value(test)
    return None if v is _Unknown else bool(v)


NEGATED = {ast.Is: "is not", ast.IsNot: "is", ast.Eq: "!=", ast.NotEq: "==", ast.Lt: ">=", ast.GtE: "<",
           ast.Gt: "<=", ast.LtE: ">", ast.In: "not in", ast.NotIn: "in"}


def _negated(test: ast.expr) -> str:
    """An if's test negated, as an else is done: `a is not None` -> `a is None`, `not c` -> `c`, else not (...)."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return ast.unparse(test.operand)
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and type(test.ops[0]) in NEGATED:
        return f"{ast.unparse(test.left)} {NEGATED[type(test.ops[0])]} {ast.unparse(test.comparators[0])}"
    return f"not {ast.unparse(test)}" if isinstance(test, (ast.Name, ast.Attribute, ast.Call)) \
        else f"not ({ast.unparse(test)})"


def _field_of(held: list[Held], attr: str) -> list[Held] | None:
    """What a field of the record a variable holds was given, if it holds one record with that field."""
    records = [h for h in held if isinstance(h, Record)]
    if len(records) == 1 and len(held) == 1 and attr in records[0].fields:
        return records[0].fields[attr]
    return None


def _param_annotation(f: Function, name: str) -> ast.expr | None:
    a = f.node.args
    return next((x.annotation for x in (*a.posonlyargs, *a.args, *a.kwonlyargs) if x.arg == name), None)


def _element(ann: ast.expr | None) -> ast.expr | None:
    """The annotation of an element of a collection: A, of Sequence[A], list[A] or tuple[A, ...]; None if unknown."""
    if not isinstance(ann, ast.Subscript) or (_dotted_name(ann.value) or "").rpartition(".")[2] not in (
            "Sequence", "list", "List", "tuple", "Tuple", "Iterable"):
        return None
    sl = ann.slice
    if isinstance(sl, ast.Tuple):
        return sl.elts[0] if len(sl.elts) == 2 and isinstance(sl.elts[1], ast.Constant) \
            and sl.elts[1].value is Ellipsis else None
    return sl


def _converted(e: ast.expr) -> ast.expr | None:
    """What a conversion that keeps the values converts, as codetex leaves it out: x.astype(np.float32), x.copy()
    (methods), np.asarray(x, np.float64), float(x) (functions); None if e is not one."""
    if not isinstance(e, ast.Call) or not (name := _dotted_name(e.func)):
        return None
    head, _, last = name.rpartition(".")
    if last not in SAME:
        return None
    if head and head.split(".")[0] not in MODULES:          # a method of the value: x.astype(...)
        return e.func.value if isinstance(e.func, ast.Attribute) else None
    return e.args[0] if e.args else None                     # a function of it: np.asarray(x, ...)


def _variables(t: ast.expr) -> bool:
    """An assignment target of variables only (x; x, y; (a, b), c), not into part of a value."""
    if isinstance(t, ast.Name):
        return True
    return isinstance(t, (ast.Tuple, ast.List)) and all(_variables(e) for e in t.elts)


def _unpacked(m: Module, node: ast.AST) -> ast.AST | None:
    """What a variable is assigned by, through the tuples it is unpacked from (for m, n in ...: the comprehension)."""
    parent = m.parents().get(node)
    while isinstance(parent, (ast.Tuple, ast.List, ast.Starred)):
        parent = m.parents().get(parent)
    return parent


def _owner_stmt(m: Module, node: ast.AST) -> ast.stmt | None:
    parents = m.parents()
    while node is not None and not isinstance(node, ast.stmt):
        node = parents.get(node)  # type: ignore[assignment]
    return node


def _passes_through(node: ast.ClassDef) -> bool:
    """A class that only holds what it is given (a dataclass, a NamedTuple, with no __init__ or __post_init__ of
    its own): building one is not a step, and it holds what it was built from."""
    return _is_data(node) and not any(isinstance(s, ast.FunctionDef) and s.name in ("__init__", "__post_init__")
                                      for s in node.body)


def _anchor(f: Function, st: ast.stmt) -> str:
    """An anchor ("#first..last" or "#first") that picks out exactly the statement's lines in f, or ""."""
    first_line = min([f.node.lineno, *(d.lineno for d in f.node.decorator_list)])
    lines = f.module.lines[first_line - 1:f.node.end_lineno]
    a, b = st.lineno - first_line, (st.end_lineno or st.lineno) - first_line

    def shortest(text: str, ok: Callable[[str], bool], least: int = 8) -> str | None:
        words = text.split(" ")
        for n in range(1, len(words) + 1):
            t = " ".join(words[:n])
            if len(t) >= least and ".." not in t and ok(t):
                return t
        return None

    # always "first..last": a bare "#first" would run on to the end of the block
    first = shortest(lines[a].strip(), lambda t: sum(t in line for line in lines) == 1)
    last = shortest(lines[b].strip(), lambda t: not any(t in line for line in lines[a:b]), least=1)
    if first is None or last is None:
        return ""
    anchor = f"{first}..{last}"
    try:
        got, start = _anchored("", lines, first_line, anchor)
    except RefError:
        return ""
    return anchor if (start, start + len(got) - 1) == (st.lineno, st.end_lineno or st.lineno) else ""


CLEAN_DEPTH = 1000         # clean code: every function that calls the package's functions is opened up


def draft(entry: str, depth: int = 2, search: list[Path] | None = None, hints: bool = False,
          clean: bool = False) -> str:
    """The text of a first draft of a spec, from the entry point "pkg.mod:func". hints: take the code's type
    hints as assumptions (what a variable holds is what its annotation says: a settings dataclass, a package class
    whose methods are then steps, an array or list that `+=` or .append changes in place). clean: the code is
    clean code (written to be read): with hints; an assignment by other code than the package's calls is a step of
    its own, not a gap; and every function that calls the package's functions is opened up, whatever depth says."""
    if clean:
        hints, depth = True, CLEAN_DEPTH
    module_name, _, qual = entry.partition(":")
    if not qual or "." in qual:
        raise DraftError(f"{entry}: give a module-level function, as pkg.mod:func")
    package = Package(module_name.split(".")[0], search or [Path.cwd()])
    m = package.module(module_name)
    if m is None:
        raise DraftError(f"{entry}: no module {module_name} in {package.folder}")
    node = m.defs.get(qual)
    if not isinstance(node, ast.FunctionDef):
        raise DraftError(f"{entry}: no function {qual} in {m.file}")
    d = Drafter(package, hints, lines=clean)
    f = Function(m, node)
    params: dict[str, list[Held]] = {}
    for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        t = d.kind(m, a.annotation) if a.annotation is not None and hints else None
        if isinstance(t, ast.ClassDef) and _is_settings(t):
            params[a.arg] = [Settings(package.owner(t), t)]
        else:
            key = d.symbol(a.arg, "input")
            params[a.arg] = [frozenset([key])]
            d.give(key, d.shape(m, a.annotation), {}, entry, set())
            said = _documented(_docs(node), a.arg, typed=True)
            if said:
                d.described(key, said[0], said[1], entry)
    if d.types is not None and d.types.problem:     # said, not passed over: the draft differs without it
        d.notes.append(f"type hints were read from the annotations only, not inferred: {d.types.problem}")
    returned = d.function(f, params, "", None, depth)
    if not d.steps:
        raise DraftError(f"{entry} calls none of {package.top}'s functions; start from one that does")
    paths = {s.group for s in d.steps}
    d.groups = {g: ref for g, ref in d.groups.items() if any(p == g or p.startswith(g + "/") for p in paths)}
    for key in d.keys([h for item in returned for h in item if not isinstance(h, Plain)]):
        if d.symbols[key].role == "value":
            d.symbols[key].role = "output"
    return _write(d, entry, depth, package.top, hints, clean)


def short_symbols(d: Drafter, used: set[str]) -> dict[str, str]:
    """A short, unique symbol for each index and each used symbol (key -> TeX), taken in turn:
      1. indices (a symbol never takes an index's letter);
      2. names that are notation already (x, x0, mu) take themselves;
      3. inputs, outputs, settings, then values, in the order they were met;
      4. a repeated call's output (scale_2, after scale; the same kind of value) takes its first's symbol, primed."""
    taken: set[str] = set()
    out: dict[str, str] = {}

    def take(key: str, name: str, greek_first: bool = False) -> None:
        tex = next(t for t in neighbours(name, greek_first) if t not in taken)
        taken.add(tex)
        out[key] = tex
    for key in d.indices:
        take("index:" + key, key)
    rows = [s for s in d.symbols.values() if s.key in used]

    def kind(x: Sym) -> str:
        return "setting" if x.role in ("setting", "constant") else "value"

    def repeat(s: Sym) -> tuple[Sym, int] | None:
        """The first of the same kind among scale, scale_2, ... before s (scale_3), and how many primes s gets."""
        stem, _, n = s.key.rpartition("_")
        if not n.isdigit() or not stem:
            return None
        for m in range(1, int(n)):
            first = d.symbols.get(stem if m == 1 else f"{stem}_{m}")
            if first is not None and first.key in used and kind(first) == kind(s):
                return first, int(n) - m
        return None
    firsts = [s for s in rows if repeat(s) is None]
    for s in firsts:                                    # notation already: x, x0, mu
        if _notation(s.name) and (own := neighbours(s.name, False)[0]) not in taken:
            taken.add(own)
            out[s.key] = own
    order = {"input": 0, "output": 1, "setting": 2, "constant": 2, "value": 3}
    for s in sorted((s for s in firsts if s.key not in out), key=lambda s: order.get(s.role, 3)):
        take(s.key, s.name, greek_first=s.role in ("setting", "constant"))
    for s in rows:
        if (found := repeat(s)) is None:
            continue
        first, primes = found
        primed = out[first.key] + "'" * primes
        if primed in taken:
            take(s.key, s.name)
        else:
            taken.add(primed)
            out[s.key] = primed
    return out


def _write(d: Drafter, entry: str, depth: int, top: str, hints: bool = False, clean: bool = False) -> str:
    used = {k for step in d.steps for k in (*step.inputs, *step.outputs)} | {
        n.id for step in d.steps if step.condition for n in ast.walk(ast.parse(step.condition, mode="eval"))
        if isinstance(n, ast.Name) and n.id in d.symbols}     # and what decides whether a step is done
    short = short_symbols(d, used)                  # the TeX of each index and symbol
    command = f"headway init {entry} --clean" if clean else \
        f"headway init {entry} --depth {depth}" + (" --hints" if hints else "")
    given = f"\"entry\": {entry!r}, \"clean\": True" if clean else \
        f"\"entry\": {entry!r}, \"depth\": {depth}" + (", \"hints\": True" if hints else "")
    out = [f'"""A draft of {top}\'s headway, from `{command}`.',
           "",
           ("Every step is a call to one of the package's functions or, in clean code, a line of other code between them;"
            if clean else "Every step is a call to one of the package's functions;")
           + " the symbols are the variables that pass between",
           "them. Still to do (by hand or with an AI assistant): the TeX of each symbol and the equation of each step,",
           "the descriptions (every TODO), the indices each symbol has (dims=), and what to leave out: steps that",
           "are bookkeeping rather than the method. `headway check` this module as you go.",
           '"""',
           "from headway import Headway",
           "",
           f"G = Headway({top!r}, \"TODO: one sentence on what the method computes, from what.\",",
           f"             draft={{{given}}}" + (", hints=True)" if hints else ")") + "   # headway redraft: draft it again",
           ""]
    if d.indices:
        for key, ref in d.indices.items():
            out.append(f"G.index({key!r}, {short['index:' + key]!r}, {key.replace('_', ' ')!r}, "
                       f"\"TODO ({ref})\")")
        out.append("")
    var = {k: k + "_" if k in ("G", "Headway") else k for k in used}     # each symbol a variable of the module
    for role, heading in (("input", "inputs"), ("setting", "settings"), ("constant", "constants"),
                          ("value", "values"), ("output", "outputs")):
        rows = [s for s in d.symbols.values() if s.role == role and s.key in used]
        if not rows:
            continue
        out.append(f"# {heading}")
        for s in rows:
            label = s.name.replace("_", " ")
            if role in ("setting", "constant"):
                out.append(f"{var[s.key]} = G.{role}({s.key!r}, {short[s.key]!r}, {label!r}, {s.desc!r}, "
                           f"value={s.value!r}, code={s.code[0]!r})")
                continue
            extra = (f", role={role!r}" if role != "value" else "") + (f", per={s.per!r}" if s.per else "") + \
                (f", dims={tuple(s.dims)!r}" if s.dims else "") + \
                (f", code={s.code!r}" if s.code else "")
            out.append(f"{var[s.key]} = G.symbol({s.key!r}, {short[s.key]!r}, {label!r}, {s.desc!r}{extra})")
        out.append("")
    if d.groups:
        for path, ref in d.groups.items():
            out.append(f"G.group({path!r}, desc={ref!r})")
        out.append("")
    def listed(keys: list[str]) -> str:
        return "[" + ", ".join(var[k] for k in keys) + "]"
    for step in d.steps:
        code = step.code[0] if len(step.code) == 1 else step.code
        args = [f"{step.key!r}", f"{step.label!r}", 'tex=""', 'desc="TODO"', f"inputs={listed(step.inputs)}",
                f"outputs={listed(step.outputs)}", f"code={code!r}"]
        if step.group:
            args.append(f"group={step.group!r}")
        if step.per:
            args.append(f"per={step.per!r}")
        if step.condition:
            args.append(f"condition={step.condition!r}")
        if step.relations:
            args.append(f"relations={step.relations!r}")
        # how the code's names are written as math ("From the code"): as the symbols they hold
        names = {n: k for n, k in step.names.items() if k in var}
        if len(step.outputs) == 1:                  # what the call returns
            names["return"] = step.outputs[0]
        if names:
            args.append("names={" + ", ".join(f"{n!r}: {var[k]}" for n, k in names.items()) + "}")
        out.append("G.op(" + ",\n     ".join(_wrap(args)) + ")")
    if d.notes:
        out += ["", "# Where values may flow with no edge drawn (headway init follows a value only from the step",
                "# whose result a variable is assigned; check these places in the code):"]
        out += [f"#   {n}" for n in d.notes]
    return stamp("\n".join(out) + "\n")


STAMP = "# drafted by headway init; "              # the last line of a draft: a checksum of the lines above it


def stamp(text: str) -> str:
    return text + f"{STAMP}checksum {hashlib.sha256(text.encode()).hexdigest()[:16]}\n"


def edited(text: str) -> bool:
    """Whether a drafted spec has been changed since headway init wrote it (or has no stamp to tell)."""
    body, _, last = text.rstrip("\n").rpartition("\n")
    return not last.startswith(STAMP) or stamp(body + "\n") != text.rstrip("\n") + "\n"


def arguments(spec_file: Path) -> dict[str, Any]:
    """The draft= a spec's Headway(...) was given (its headway init arguments), read without importing it."""
    tree = ast.parse(spec_file.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name | ast.Attribute) and \
                (node.func.id if isinstance(node.func, ast.Name) else node.func.attr) == "Headway":
            for k in node.keywords:
                if k.arg == "draft":
                    try:
                        got = ast.literal_eval(k.value)
                    except ValueError:
                        break
                    if isinstance(got, dict) and isinstance(got.get("entry"), str):
                        return got
    raise DraftError(f"{spec_file} was not drafted by headway init (its Headway(...) has no draft=)")


def redraft(spec_file: Path, force: bool = False) -> Path | None:
    """Draft the spec again from the code, with its headway init arguments, over the file. A file changed since
    it was drafted is kept beside it (.bak, returned), and only with force; without, DraftEdited."""
    args = arguments(spec_file)
    text = spec_file.read_text()
    backup = None
    if edited(text):
        if not force:
            raise DraftEdited(f"{spec_file} has been edited since it was drafted")
        backup = spec_file.with_name(spec_file.name + ".bak")
        backup.write_text(text)
    from headway import files
    new = draft(args["entry"], int(args.get("depth", 2)), search=files.bases(str(spec_file)),
                hints=bool(args.get("hints", False)), clean=bool(args.get("clean", False)))
    spec_file.write_text(new)
    return backup


def _wrap(args: list[str], width: int = 110) -> list[str]:
    """The arguments of a call, several to a line."""
    lines: list[str] = []
    for a in args:
        if lines and len(lines[-1]) + 2 + len(a) <= width:
            lines[-1] += ", " + a
        else:
            lines.append(a)
    return lines
