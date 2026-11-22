"""The shape trace: run the code (e.g. a test on small data) with the spec's bound functions wrapped, and check every
call's arrays against the symbols' declared dimensions.

An op's `bind` maps the function's parameter names (and "return", "return.0", "return.attr") to symbols. Each call
of an op's first code reference is checked: every bound array has as many axes as its symbol has indices, and an
index has one size within the call (both i x f tables have the same i and f). Across calls, an index has one size
unless declared `varies=True`. A function several ops share passes a call if any of their binds fits it.

    with headway.trace(G) as t:
        method.run(...)
    t.assert_ok()
"""
from __future__ import annotations

import functools
import importlib
import inspect
import sys
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from headway.spec import Headway, Op


class Trace:
    def __init__(self, g: Headway):
        self.g = g
        self.errors: list[str] = []
        self.calls: dict[str, int] = {}                               # op -> calls assigned to it (one op per call)
        self.tried: dict[str, int] = defaultdict(int)                 # op -> calls of its function
        self.sizes: dict[str, set[int]] = defaultdict(set)           # index -> sizes seen

    def _values(self, fn: Callable[..., Any], args: tuple, kwargs: dict, result: Any) -> dict[str, Any]:
        try:
            bound = inspect.signature(fn).bind(*args, **kwargs)
            bound.apply_defaults()
            values = dict(bound.arguments)
        except TypeError:
            values = {}
        values["return"] = result
        return values

    def _lookup(self, values: dict[str, Any], name: str) -> Any:
        head, *rest = name.split(".")
        v = values[head]
        for part in rest:
            v = v[int(part)] if part.isdigit() else getattr(v, part)
        return v

    def _fit(self, op: Op, values: dict[str, Any]) -> tuple[dict[str, int], str | None]:
        sizes: dict[str, int] = {}
        for code_name, key in op.bind.items():
            try:
                v = self._lookup(values, code_name)
            except (KeyError, AttributeError, IndexError, TypeError):
                return sizes, f"{code_name} not found"
            dims = self.g.symbols[key].dims
            shape = np.shape(v)
            if len(shape) != len(dims):
                return sizes, f"{code_name} ({key}) has shape {shape}, declared {' x '.join(dims) or 'scalar'}"
            for d, n in zip(dims, shape):
                if sizes.setdefault(d, n) != n:
                    return sizes, f"{code_name} ({key}): index {d} is {n} here, {sizes[d]} elsewhere in the call"
        return sizes, None

    def record(self, ops: list[Op], fn: Callable[..., Any], args: tuple, kwargs: dict, result: Any) -> None:
        values = self._values(fn, args, kwargs, result)
        problems, fits = [], []
        for op in ops:
            self.tried[op.key] += 1
            sizes, problem = self._fit(op, values)
            if problem is None:
                fits.append((op, sizes))
            else:
                problems.append(f"op {op.key}: {problem}")
        if not fits:
            self.errors.append(f"{fn.__module__}.{fn.__qualname__}: " + "; ".join(problems))
            return

        def agrees(sizes: dict[str, int]) -> bool:          # with the sizes seen so far
            return all(self.g.indices[d].varies or not self.sizes.get(d) or n in self.sizes[d]
                       for d, n in sizes.items())
        # several ops share the function (e.g. one step of both rounds): of those whose sizes agree (else all), the
        # one called least so far, so that calls in order go to the ops in order
        candidates = [fit for fit in fits if agrees(fit[1])] or fits
        op, sizes = min(candidates, key=lambda fit: self.calls.get(fit[0].key, 0))
        self.calls[op.key] = self.calls.get(op.key, 0) + 1
        for d, n in sizes.items():
            self.sizes[d].add(n)

    def finish(self) -> None:
        for d, seen in self.sizes.items():
            if len(seen) > 1 and not self.g.indices[d].varies:
                self.errors.append(f"index {d} took sizes {sorted(seen)} (declare varies=True if it should)")
        for op in self.g.ops.values():
            if op.bind and op.key not in self.calls:
                tried = self.tried.get(op.key, 0)
                self.errors.append(f"op {op.key}: bound, but " + (
                    f"none of the {tried} calls of its function fit its bind (above)"
                    if tried else "its function was never called"))

    def text(self) -> str:
        lines = [f"error: {e}" for e in self.errors]
        lines.append("sizes: " + ", ".join(f"{d}={'/'.join(map(str, sorted(s)))}" for d, s in self.sizes.items()))
        lines.append(f"{sum(self.calls.values())} calls checked over {len(self.calls)} ops")
        return "\n".join(lines)

    def assert_ok(self) -> None:
        assert not self.errors, self.text()


def _function(ref: str) -> tuple[Callable[..., Any], Any]:
    """A reference's function, and what holds it: its module, or its class for a method ("pkg.mod:Class.fit"). For
    a static or class method, the function itself (as the class's __dict__ holds it, inside the descriptor); for a
    property, its getter; for a cached property, its function."""
    module_name, _, qualname = ref.partition("#")[0].partition(":")
    owner: Any = None
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        owner, obj = obj, getattr(obj, part)
    if isinstance(obj, property):
        return obj.fget, owner                          # type: ignore[return-value]
    if isinstance(obj, functools.cached_property):
        return obj.func, owner
    return (getattr(obj, "__func__", obj) if inspect.isclass(owner) else obj), owner


def _holders(fn: Any, owner: Any) -> list[tuple[Any, str, Any]]:
    """Where to put the wrapper of fn: (holder, name, what it holds now). A method: the class along the owner's MRO
    whose __dict__ defines it (as a plain function, inside a staticmethod or classmethod, as a property's getter or
    as a cached property's function); a function: every module that imported it by name."""
    if inspect.isclass(owner):
        return [(c, name, v) for c in owner.__mro__ for name, v in list(vars(c).items())
                if v is fn or getattr(v, "__func__", None) is fn
                or (isinstance(v, property) and v.fget is fn)
                or (isinstance(v, functools.cached_property) and v.func is fn)][:1]
    return [(m, name, v) for m in list(sys.modules.values()) for name, v in list(getattr(m, "__dict__", {}).items())
            if v is fn]


def _rewrapped(value: Any, wrapper: Callable[..., Any], holder: Any, name: str) -> Any:
    """What a holder should hold instead of value: the wrapper, in the same kind of descriptor as value."""
    if isinstance(value, property):
        return value.getter(wrapper)
    if isinstance(value, functools.cached_property):
        cached = functools.cached_property(wrapper)
        cached.__set_name__(holder, name)
        return cached
    if isinstance(value, (staticmethod, classmethod)):
        return type(value)(wrapper)
    return wrapper


@contextmanager
def trace(g: Headway) -> Iterator[Trace]:
    """Wrap every bound op's function (its first code reference) wherever a module holds it (a method: in its class),
    for the duration."""
    t = Trace(g)
    by_function: dict[Any, list[Op]] = defaultdict(list)
    owners: dict[Any, Any] = {}                         # a function -> its module, or its class for a method
    for op in g.ops.values():
        if not op.bind:
            continue
        try:
            fn, owner = _function(op.code[0])
        # no code, a file's ("R/score.R:f"; "../R/score.R:f" is a TypeError, a relative import), or not there
        except (IndexError, ImportError, AttributeError, TypeError, ValueError):
            t.errors.append(f"op {op.key}: bound, but its first code reference names no Python function")
            continue
        owners[fn] = owner
        by_function[fn].append(op)
    patched: list[tuple[Any, str, Any]] = []
    for fn, ops in by_function.items():
        @functools.wraps(fn)
        def wrapper(*args: Any, __fn: Any = fn, __ops: list[Op] = ops, **kwargs: Any) -> Any:
            result = __fn(*args, **kwargs)
            t.record(__ops, __fn, args, kwargs, result)
            return result
        for holder, name, value in _holders(fn, owners[fn]):
            patched.append((holder, name, value))
            # a static or class method stays one, and a property or cached property one with the wrapper as its
            # function, so that it is called as before
            setattr(holder, name, _rewrapped(value, wrapper, holder, name))
    try:
        yield t
    finally:
        for module, name, fn in patched:
            setattr(module, name, fn)
        t.finish()
