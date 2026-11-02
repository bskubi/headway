"""Libraries, as tables of translations into the calculus (docs/core.md §6).

A table maps an operation, by what it is (its qualified name, "numpy.sum", or a method of a type, ("numpy.ndarray",
"sum")), to an entry: a function of the call's arguments (already terms) that gives the call's term, or None when
the call is one the entry does not translate (an argument it does not know, a case it does not cover). A call no
entry translates is an operator: shown by its name, never interpreted.

    @NUMPY.function("numpy.log", "numpy.exp")
    def _elementwise(c: Call) -> Expr | None: ...

Each table's entries are tested (tests/test_core.py) by evaluating their terms (evaluate.py) against the library on
random inputs, chiefly at the edges where a library's meaning is subtler than its syntax.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from headway.core.terms import Const, Expr, Sig, Text, Tup

if TYPE_CHECKING:
    from headway.core.lower import Lowerer

NOT = object()                          # an argument that is not a literal


@dataclass
class Call:
    """A call, its arguments terms: what an entry is given."""
    name: str                           # the qualified name ("numpy.sum"), or the method's ("sum")
    args: list[Expr]                    # a method's receiver first
    kwargs: dict[str, Expr]
    lower: Lowerer
    receiver: bool = False              # a method: args[0] is what it is called on
    node: Any = None                    # the call's syntax (ast.Call)

    def literal(self, e: Expr | None, default: Any = NOT) -> Any:
        """The Python value of an argument written as a literal (3, (0, 1), None, "x"), else NOT (default when
        the argument is not given)."""
        if e is None:
            return default
        if isinstance(e, Const):
            return e.value
        if isinstance(e, Text):
            return e.value if e.value is not None else NOT
        if isinstance(e, Tup):
            items = [self.literal(x) for x in e.items]
            return NOT if any(x is NOT for x in items) else tuple(items)
        return NOT

    def arg(self, k: int, name: str | None = None) -> Expr | None:
        """The k-th positional argument, or the one given by name."""
        if k < len(self.args):
            return self.args[k]
        return self.kwargs.get(name) if name else None

    def only(self, *allowed: str) -> bool:
        """No keyword but these (an entry that does not know an argument translates nothing)."""
        return set(self.kwargs) <= set(allowed)


Entry = Callable[[Call], Expr | None]


@dataclass
class Table:
    """One library's translations."""
    name: str
    functions: dict[str, Entry] = field(default_factory=dict)
    methods: dict[tuple[str, str], Entry] = field(default_factory=dict)
    attributes: dict[tuple[str, str], Entry] = field(default_factory=dict)     # x.T, x.shape: a receiver only
    constants: dict[str, Expr] = field(default_factory=dict)                    # numpy.pi
    types: dict[str, str] = field(default_factory=dict)                         # what an annotation's type is

    def function(self, *names: str) -> Callable[[Entry], Entry]:
        def add(entry: Entry) -> Entry:
            for n in names:
                self.functions[n] = entry
            return entry
        return add

    def method(self, types: tuple[str, ...], *names: str) -> Callable[[Entry], Entry]:
        def add(entry: Entry) -> Entry:
            for t in types:
                for n in names:
                    self.methods[(t, n)] = entry
            return entry
        return add

    def attribute(self, types: tuple[str, ...], *names: str) -> Callable[[Entry], Entry]:
        def add(entry: Entry) -> Entry:
            for t in types:
                for n in names:
                    self.attributes[(t, n)] = entry
            return entry
        return add


TABLES: list[Table] = []


def register(t: Table) -> Table:
    TABLES.append(t)
    return t


def function(name: str) -> Entry | None:
    _load()
    return next((t.functions[name] for t in TABLES if name in t.functions), None)


def method(type_: str, name: str) -> Entry | None:
    _load()
    return next((t.methods[(type_, name)] for t in TABLES if (type_, name) in t.methods), None)


def attribute(type_: str, name: str) -> Entry | None:
    _load()
    return next((t.attributes[(type_, name)] for t in TABLES if (type_, name) in t.attributes), None)


def constant(name: str) -> Expr | None:
    _load()
    return next((t.constants[name] for t in TABLES if name in t.constants), None)


def known_type(name: str) -> str | None:
    _load()
    return next((t.types[name] for t in TABLES if name in t.types), None)


def supported(name: str) -> bool:
    """Whether some table has an entry for a qualified name (the linter: an operation without one stays opaque)."""
    _load()
    return any(name in t.functions or name in t.constants for t in TABLES)


_loaded = False


def _load() -> None:
    global _loaded
    if not _loaded:
        _loaded = True
        from headway.core.libs import numpy_, polars_, python, scipy_, torch_  # noqa: F401


def same_sig(a: Sig, b: Sig) -> bool:
    return a is not None and b is not None and a == b
