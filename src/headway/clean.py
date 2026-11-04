"""Declarations for clean code: how a function's algorithm relates to what it computes, and the checks of each.

A clean function (code written to be read; see the guide) may declare, with a decorator, its relation to a
specification: a function, written as plainly, of what it computes. The decorators change nothing about the function;
they attach the declaration, which `headway init --clean` reads from the code (statically) and `check` tests on calls
the code makes (traced, from a real run). The kinds, a closed set:

  maximizes(objective, over=("b", "d"), subject_to=(constraint, ...))   its results are a local optimum of objective
  minimizes(...)                                                         (likewise)
  solves(equation, over=(...))                                           its results make equation zero
  equals(reference, rtol=..., atol=...)                                  it gives what reference gives (exactly, up
                                                                         to rounding); reference may be a dotted
                                                                         name, "pkg.mod.func" (the code it stands for)
  approximates(reference, rtol=..., atol=...)                            it gives what reference gives within a
                                                                         stated error

An objective, equation or constraint names its parameters as the function's own (its arguments, by name) and as the
`over` names, which are the function's results in order (a tuple's items). A constraint is a function that is zero
where it holds (`lambda d: d.sum()`, or a named function, which the viewer can write as math).
"""
from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

KINDS = ("maximizes", "minimizes", "solves", "equals", "approximates")


@dataclass
class Relation:
    kind: str
    target: Callable[..., Any] | str               # the objective, equation or reference
    over: tuple[str, ...] = ()                      # the function's results, by name, in order
    subject_to: tuple[Callable[..., Any], ...] = ()
    tol: float = 1e-6                               # maximizes/minimizes/solves: how near zero, relatively
    rtol: float = 1e-7                              # equals/approximates
    atol: float = 0.0
    adapt: Callable[[Any], Any] | None = None       # equals: the function's result in the reference's form


def _declare(rel: Relation) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def mark(fn: Callable[..., Any]) -> Callable[..., Any]:
        fn.__dict__.setdefault("__headway__", []).append(rel)
        return fn
    return mark


def maximizes(objective: Callable[..., Any], *, over: Sequence[str], subject_to: Sequence[Callable[..., Any]] = (),
              tol: float = 1e-6) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _declare(Relation("maximizes", objective, tuple(over), tuple(subject_to), tol))


def minimizes(objective: Callable[..., Any], *, over: Sequence[str], subject_to: Sequence[Callable[..., Any]] = (),
              tol: float = 1e-6) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _declare(Relation("minimizes", objective, tuple(over), tuple(subject_to), tol))


def solves(equation: Callable[..., Any], *, over: Sequence[str],
           tol: float = 1e-6) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _declare(Relation("solves", equation, tuple(over), (), tol))


def equals(reference: Callable[..., Any] | str, *, rtol: float = 1e-7, atol: float = 0.0,
           adapt: Callable[[Any], Any] | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _declare(Relation("equals", reference, rtol=rtol, atol=atol, adapt=adapt))


def approximates(reference: Callable[..., Any] | str, *, rtol: float, atol: float = 0.0
                 ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _declare(Relation("approximates", reference, rtol=rtol, atol=atol))


def relations(fn: Callable[..., Any]) -> list[Relation]:
    return list(getattr(fn, "__headway__", []))


def resolve(target: Callable[..., Any] | str) -> Callable[..., Any]:
    """A reference given by its dotted name ("pkg.mod.func"), imported."""
    if callable(target):
        return target
    module, _, name = target.rpartition(".")
    return getattr(importlib.import_module(module), name)  # type: ignore[no-any-return]


@dataclass
class Result:
    """One check of one relation on one call."""
    passed: bool
    measure: float                                  # how far from holding (0 is exactly)
    detail: str = ""
    notes: list[str] = field(default_factory=list)


def check(fn: Callable[..., Any], rel: Relation, args: dict[str, Any], result: Any, seed: int = 0) -> Result:
    """Whether a call of fn (its arguments by name, and what it returned) bears out a relation it declares. The
    original reruns from the NumPy random state kept with the arguments, if one was."""
    from headway.core.verify import restore_random, without_random
    args, state = without_random(args)
    if rel.kind in ("equals", "approximates"):
        restore_random(state)
        return _compare(rel, args, result)
    values = dict(args)
    items = result if len(rel.over) > 1 else (result,)
    values |= dict(zip(rel.over, items))
    if rel.kind == "solves":
        r = np.asarray(_call(rel.target, values), np.float64)  # type: ignore[arg-type]
        scale = max(np.max(np.abs([np.max(np.abs(np.asarray(values[v], np.float64))) for v in rel.over])), 1.0)
        m = float(np.max(np.abs(r)) / scale)
        return Result(m <= rel.tol, m, f"largest residual {m:.2g}, relative (at most {rel.tol:g})")
    return _optimum(rel, values, seed)


def _call(f: Callable[..., Any] | str, values: dict[str, Any]) -> Any:
    f = resolve(f)
    names = list(inspect.signature(f).parameters)
    return f(**{n: values[n] for n in names})


def _compare(rel: Relation, args: dict[str, Any], result: Any) -> Result:
    want = _call(rel.target, args)
    got = rel.adapt(result) if rel.adapt else result
    worst = 0.0
    ok = True
    for g, w in zip(_leaves(got), _leaves(want), strict=False):
        g, w = np.asarray(g), np.asarray(w)
        if g.shape != w.shape:
            return Result(False, float("inf"), f"shapes {g.shape} and {w.shape}")
        if g.dtype.kind in "bUSO" or w.dtype.kind in "bUSO":
            ok &= bool((g == w).all())
            continue
        diff = float(np.max(np.abs(g.astype(np.float64) - w.astype(np.float64)), initial=0.0))
        worst = max(worst, diff)
        ok &= bool(np.allclose(g, w, rtol=rel.rtol, atol=rel.atol))
    return Result(ok, worst, f"largest difference {worst:.2g} (allowed: {rel.rtol:g} relative, {rel.atol:g} absolute)")


def _leaves(x: Any) -> list[Any]:
    if callable(x):                                 # a function of a chunk: all of it
        try:
            return [x(slice(None))]
        except Exception:                           # noqa: BLE001 (a function of something else: which one)
            return [getattr(x, "__qualname__", type(x).__name__)]
    if hasattr(x, "__dataclass_fields__"):
        return [leaf for f in x.__dataclass_fields__ for leaf in _leaves(getattr(x, f))]
    if isinstance(x, dict):
        return [leaf for k in sorted(x) for leaf in _leaves(x[k])]
    if isinstance(x, (tuple, list)):
        return [leaf for item in x for leaf in _leaves(item)]
    return [x]


def _optimum(rel: Relation, values: dict[str, Any], seed: int) -> Result:
    """A local optimum, tested along random directions that keep the constraints (their Jacobian's null space):
    the derivative of the objective there is near zero (relative to the objective's size and the variables'),
    and no small step along them improves it."""
    sign = 1.0 if rel.kind == "maximizes" else -1.0
    over = list(rel.over)
    x0 = [np.asarray(values[v], np.float64) for v in over]
    sizes = [x.size for x in x0]
    flat = np.concatenate([x.ravel() for x in x0])

    def unflat(z: np.ndarray) -> dict[str, Any]:
        parts, k = {}, 0
        for v, x, n in zip(over, x0, sizes):
            parts[v] = z[k:k + n].reshape(x.shape) if x.ndim else float(z[k])
            k += n
        return values | parts

    def f(z: np.ndarray) -> float:
        return sign * float(_call(rel.target, unflat(z)))
    # the constraints' Jacobian, by differences (they are cheap; each column one variable)
    rows = []
    for c in rel.subject_to:
        base = np.atleast_1d(np.asarray(_call(c, unflat(flat)), np.float64)).ravel()
        cols = []
        for i in range(flat.size):
            z = flat.copy()
            z[i] += 1e-6
            cols.append((np.atleast_1d(np.asarray(_call(c, unflat(z)), np.float64)).ravel() - base) / 1e-6)
        rows.append(np.array(cols).T)
    jac = np.vstack(rows) if rows else np.zeros((0, flat.size))
    rng = np.random.default_rng(seed)
    f0 = f(flat)
    scale = max(abs(f0), 1.0)
    norm = max(float(np.linalg.norm(flat)), 1.0)
    worst, improved = 0.0, 0
    for _ in range(8):
        v = rng.normal(size=flat.size)
        if jac.size:
            v -= jac.T @ np.linalg.lstsq(jac @ jac.T, jac @ v, rcond=None)[0]
        v /= max(float(np.linalg.norm(v)), 1e-300)
        eps = 1e-5 * norm
        slope = (f(flat + eps * v) - f(flat - eps * v)) / (2 * eps)
        worst = max(worst, abs(slope) * norm / scale)    # relative change of f per relative change of x
        step = 1e-3 * norm
        if max(f(flat + step * v), f(flat - step * v)) > f0 + 1e-12 * scale:
            improved += 1                               # (f is the objective, negated when minimizing)
    ok = worst <= rel.tol and improved == 0
    better = "larger" if rel.kind == "maximizes" else "smaller"
    return Result(ok, worst, f"slope {worst:.2g} along directions that keep the constraints, relative (at most "
                  f"{rel.tol:g}); {improved} of 8 nearby points give a {better} value")
