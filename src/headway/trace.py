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
        self.calls: dict[str, int] = defaultdict(int)                 # op -> calls that fit it
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
            return all(self.g.indices[d].varies or not self.sizes[d] or n in self.sizes[d] for d, n in sizes.items())
        # several ops share the function (e.g. one step of both rounds): of those whose sizes agree (else all), the
        # one called least so far, so that calls in order go to the ops in order
        candidates = [fit for fit in fits if agrees(fit[1])] or fits
        op, sizes = min(candidates, key=lambda fit: self.calls[fit[0].key])
        self.calls[op.key] += 1
        for d, n in sizes.items():
            self.sizes[d].add(n)

    def finish(self) -> None:
        for d, seen in self.sizes.items():
            if len(seen) > 1 and not self.g.indices[d].varies:
                self.errors.append(f"index {d} took sizes {sorted(seen)} (declare varies=True if it should)")
        for op in self.g.ops.values():
            if op.bind and not self.calls[op.key]:
                self.errors.append(f"op {op.key}: bound, but " + (
                    f"none of the {self.tried[op.key]} calls of its function fit its bind (above)"
                    if self.tried[op.key] else "its function was never called"))

    def text(self) -> str:
        lines = [f"error: {e}" for e in self.errors]
        lines.append("sizes: " + ", ".join(f"{d}={'/'.join(map(str, sorted(s)))}" for d, s in self.sizes.items()))
        lines.append(f"{sum(self.calls.values())} calls checked over {len(self.calls)} ops")
        return "\n".join(lines)

    def assert_ok(self) -> None:
        assert not self.errors, self.text()


def _function(ref: str) -> Callable[..., Any]:
    module_name, _, qualname = ref.partition("#")[0].partition(":")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


@contextmanager
def trace(g: Headway) -> Iterator[Trace]:
    """Wrap every bound op's function (its first code reference) wherever a module holds it, for the duration."""
    t = Trace(g)
    by_function: dict[Any, list[Op]] = defaultdict(list)
    for op in g.ops.values():
        if op.bind:
            by_function[_function(op.code[0])].append(op)
    patched: list[tuple[Any, str, Any]] = []
    for fn, ops in by_function.items():
        @functools.wraps(fn)
        def wrapper(*args: Any, __fn: Any = fn, __ops: list[Op] = ops, **kwargs: Any) -> Any:
            result = __fn(*args, **kwargs)
            t.record(__ops, __fn, args, kwargs, result)
            return result
        for module in list(sys.modules.values()):           # every module that imported the function by name
            for name, value in list(getattr(module, "__dict__", {}).items()):
                if value is fn:
                    patched.append((module, name, fn))
                    setattr(module, name, wrapper)
    try:
        yield t
    finally:
        for module, name, fn in patched:
            setattr(module, name, fn)
        t.finish()
