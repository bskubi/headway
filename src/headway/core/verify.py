"""Checking what is shown against what the code does, on real calls (docs/core.md §7).

For a function and the arguments of a call of it (from a traced run, or given), its Program is run by the reference
evaluator, as lowered (each statement's translation) and as folded (what the viewer shows), and the results compared
with what the function itself returns. A difference means a translation, a signature or the fold says something the
code does not do; agreement, on real inputs, is the evidence that the math shown is the code's.

    report = verify(steps.fit_intercepts, kwargs=dict(methylated=m, total=n, ...))
    report.ok, report.text()
"""
from __future__ import annotations

import ast
import copy
import inspect
import sys
import textwrap
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from headway.core.evaluate import Evaluator, Mismatch, Uncheckable, safe_copy
from headway.core.lower import Lowerer, Module
from headway.core.terms import Program, Shape


@dataclass
class Check:
    """One run of one form of a function's Program on one call."""
    form: str                               # "lowered", "folded"
    ok: bool
    difference: float = 0.0                 # the largest, relative to the values' size
    detail: str = ""


@dataclass
class Report:
    function: str
    checks: list[Check] = field(default_factory=list)
    failed_preconditions: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def text(self) -> str:
        lines = [f"{self.function}:"]
        for c in self.checks:
            lines.append(f"  {c.form}: {'agrees' if c.ok else 'DIFFERS'}"
                         + (f" (largest difference {c.difference:.2g})" if c.ok else "")
                         + (f" — {c.detail}" if c.detail else ""))
        if self.failed_preconditions:
            lines.append(f"  preconditions that did not hold: lines {self.failed_preconditions}")
        return "\n".join(lines)


def program_of(fn: Callable[..., Any], hints: bool = True, given: dict[str, Shape] | None = None,
               constants: bool = True) -> tuple[Program, Module]:
    """A function's Program, read from its source in its module's file."""
    fn = inspect.unwrap(fn)
    file = inspect.getsourcefile(fn)
    module = Module.read(file, fn.__module__) if file else Module(fn.__module__)
    lines, first = inspect.getsourcelines(fn)
    source = textwrap.dedent("".join(lines))
    tree = ast.parse(source)
    node = tree.body[0]
    assert isinstance(node, ast.FunctionDef)
    node.decorator_list = []
    indent = len(lines[0]) - len(lines[0].lstrip())
    low = Lowerer(module, given, hints=hints, offset=first - 1, indent=indent)
    low.inline_constants = constants
    return low.program(node), module


# a recorded call's NumPy global random state, as the call started (kept with its arguments, under this name): a
# rerun of the code, or of the math, draws the numbers the call drew. NumPy's global generator only; a Generator
# passed as an argument is copied with the arguments; Python's random and torch's are not kept
RANDOM = "__headway_numpy_random__"


def random_state() -> dict[str, Any]:
    """The state to keep with a call's arguments, as it starts."""
    return {RANDOM: np.random.get_state()}


def without_random(args: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """A recorded call's arguments, and the random state kept with them (None if none was)."""
    args = dict(args)
    return args, args.pop(RANDOM, None)


def restore_random(state: Any) -> None:
    """NumPy's global random state as a recorded call started (nothing, if none was kept)."""
    if state is not None:
        np.random.set_state(state)


def verify(fn: Callable[..., Any], args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None,
           result: Any = None, rtol: float = 1e-9, atol: float = 1e-12, forms: tuple[str, ...] = ("lowered", "folded"),
           names: dict[str, Any] | None = None, scaled: bool = False,
           given: dict[str, Shape] | None = None, constants: bool = True) -> Report:
    """Run a function's Program (each form) on a call's arguments and compare with the function's own result
    (computed here, on a copy, if not given)."""
    from headway.core.fold import fold
    owner = getattr(fn, "__self__", None)                # a bound method: its object is its self
    target = inspect.unwrap(fn.__func__ if inspect.ismethod(fn) else fn)
    kwargs, state = without_random(kwargs or {})
    bound = inspect.signature(target).bind(*((owner,) if owner is not None else ()), *args, **(kwargs or {}))
    bound.apply_defaults()
    call = dict(bound.arguments)
    if result is None:
        restore_random(state)
        result = target(*(safe_copy(a) for a in bound.args), **{k: safe_copy(v) for k, v in bound.kwargs.items()})
        if owner is not None:
            call["self"] = owner
    prog, _ = program_of(target, given=given, constants=constants)
    scope = dict(target.__globals__) | dict(names or {})
    owner_path = target.__qualname__.split(".")[:-1]   # a method: its class, for super()
    if owner_path and "<locals>" not in owner_path:
        cls: Any = target.__globals__.get(owner_path[0])
        for part in owner_path[1:]:
            cls = getattr(cls, part, None)
        if isinstance(cls, type):
            scope["__class__"] = cls
    report = Report(f"{target.__module__}.{target.__qualname__}")
    for form in forms:
        p = prog if form == "lowered" else fold(prog)
        ev = Evaluator(names=scope)
        restore_random(state)                           # each form draws what the call drew
        try:
            got = ev.run(p, call)
        except (Uncheckable, Mismatch) as e:
            report.checks.append(Check(form, False, float("inf"), f"{type(e).__name__}: {e}"))
            continue
        except Exception as e:                              # a term that fails where the code does not
            report.checks.append(Check(form, False, float("inf"), f"{type(e).__name__}: {e}"))
            continue
        ok, diff, detail = compare(got, result, rtol, atol, scaled)
        report.checks.append(Check(form, ok, diff, detail))
        report.failed_preconditions = sorted({line for line, _ in ev.failed})
    return report


def compare(got: Any, want: Any, rtol: float, atol: float, scaled: bool = False) -> tuple[bool, float, str]:
    """Whether two results agree, item by item, each element within atol + rtol × its size (scaled: atol taken
    relative to the largest element's size, for rounding that grows with the values a result is computed from):
    (agrees, the largest difference, what differs). A comparison that fails is a check that fails, not the run."""
    try:
        return _compare(got, want, rtol, atol, scaled)
    except Exception as e:                              # noqa: BLE001
        return False, float("inf"), f"not compared: {type(e).__name__}: {e}"


def _compare(got: Any, want: Any, rtol: float, atol: float, scaled: bool = False) -> tuple[bool, float, str]:
    import types
    if isinstance(want, types.FunctionType) and callable(got):     # a function of a chunk: all of it
        try:
            got, want = got(slice(None)), want(slice(None))
        except Exception:                               # noqa: BLE001 (a function of something else: which one)
            name = lambda f: getattr(f, "__qualname__", None) or type(f).__name__   # noqa: E731
            same = name(got) == name(want)
            return same, 0.0 if same else float("inf"), "" if same else \
                f"the function {name(got)}, the code's is {name(want)}"
    if type(want).__name__ == "DataFrame" and type(got).__name__ == "DataFrame" and hasattr(want, "columns"):
        # tables: the same columns, and the same rows in any order (a group_by's order is not polars' to keep)
        if sorted(got.columns) != sorted(want.columns):
            return False, float("inf"), f"columns {sorted(got.columns)}, the code's are {sorted(want.columns)}"
        cols = sorted(want.columns)
        try:
            got, want = got.select(cols).sort(cols), want.select(cols).sort(cols)
        except Exception:                               # noqa: BLE001 (columns that cannot be sorted)
            got, want = got.select(cols), want.select(cols)
        return compare([got.get_column(c).to_numpy() for c in cols], [want.get_column(c).to_numpy() for c in cols],
                       rtol, atol, scaled)
    if hasattr(want, "__dataclass_fields__") and hasattr(got, "__dataclass_fields__"):
        got = tuple(getattr(got, f) for f in got.__dataclass_fields__)
        want = tuple(getattr(want, f) for f in want.__dataclass_fields__)
    if type(want) is type(got) and getattr(want, "__dict__", None) is not None \
            and not isinstance(want, (types.FunctionType, types.MethodType, type)) \
            and type(want).__module__.split(".")[0] not in ("numpy", "pandas", "polars", "builtins"):
        return compare(vars(got), vars(want), rtol, atol, scaled)    # objects: their state
    if isinstance(want, dict) and isinstance(got, dict):
        if set(got) != set(want):
            return False, float("inf"), f"keys {sorted(map(str, got))} and {sorted(map(str, want))}"
        got, want = [got[k] for k in want], list(want.values())
    if isinstance(want, (tuple, list)) and isinstance(got, (tuple, list)) and not _numeric_list(want):
        if len(got) != len(want):
            return False, float("inf"), f"{len(got)} items, the code gives {len(want)}"
        worst = 0.0
        for k, (g, w) in enumerate(zip(got, want)):
            ok, d, detail = compare(g, w, rtol, atol, scaled)
            if not ok:
                return False, d, f"item {k}: {detail}"
            worst = max(worst, d)
        return True, worst, ""
    got, want = _numpy(got), _numpy(want)
    if isinstance(want, (slice, str, bytes, type)) or want is None or got is None:
        same = got == want                              # not numbers: equal or not
        return bool(same), 0.0 if same else float("inf"), "" if same else "not equal"
    try:
        g = np.asarray(got)
        w = np.asarray(want)
    except Exception:
        return got == want, 0.0 if got == want else float("inf"), "" if got == want else "not equal"
    if g.dtype == object or w.dtype == object:
        same = bool(got == want) if not isinstance(got == want, np.ndarray) else bool(np.all(got == want))
        return same, 0.0 if same else float("inf"), "" if same else "not equal"
    if g.shape != w.shape:
        return False, float("inf"), f"shape {g.shape}, the code's is {w.shape}"
    if g.dtype.kind in "bUS" or w.dtype.kind in "bUS":
        same = bool(np.array_equal(g, w))
        return same, 0.0 if same else float("inf"), "" if same else "not equal"
    g64, w64 = g.astype(np.float64), w.astype(np.float64)
    both_nan = np.isnan(g64) & np.isnan(w64)
    same_inf = np.isinf(g64) & (g64 == w64)
    diff = np.where(both_nan | same_inf, 0.0, np.abs(g64 - w64))
    finite = np.abs(w64[np.isfinite(w64)])
    allowed = atol * (max(float(finite.max(initial=0.0)), 1.0) if scaled else 1.0) + rtol * np.abs(w64)
    ok = bool(np.all(both_nan | same_inf | (diff <= allowed)))
    worst = float(np.max(diff, initial=0.0))
    return ok, worst, "" if ok else f"largest difference {worst:.3g}"


def _numpy(x: Any) -> Any:
    """A torch tensor's values (detached), as numpy's."""
    if type(x).__module__.split(".")[0] == "torch" and hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return x


def _numeric_list(x: Any) -> bool:
    return isinstance(x, list) and bool(x) and all(isinstance(v, (int, float, np.number)) for v in x)


@contextmanager
def traced(*functions: Callable[..., Any], limit: int = 3) -> Iterator[dict[str, list[tuple[dict[str, Any], Any]]]]:
    """Record calls of functions while code runs (their arguments copied as each starts, and what it returns), up to
    `limit` calls each: for verify(), on the inputs a real run gives. Nothing in the code is changed: a profiler
    watches the calls."""
    codes = {inspect.unwrap(f).__code__: f"{inspect.unwrap(f).__module__}.{inspect.unwrap(f).__qualname__}"
             for f in functions}
    calls: dict[str, list[tuple[dict[str, Any], Any]]] = {name: [] for name in codes.values()}
    started: dict[int, dict[str, Any]] = {}

    def profile(frame: Any, event: str, arg: Any) -> None:
        name = codes.get(frame.f_code)
        if name is None or len(calls[name]) >= limit:
            return
        if event == "call":
            code = frame.f_code
            n = code.co_argcount + code.co_kwonlyargcount
            names = code.co_varnames[:n]
            started[id(frame)] = {k: safe_copy(frame.f_locals[k]) for k in names if k in frame.f_locals} | \
                random_state()
        elif event == "return" and id(frame) in started:
            calls[name].append((started.pop(id(frame)), safe_copy(arg)))

    old = sys.getprofile()
    sys.setprofile(profile)
    try:
        yield calls
    finally:
        sys.setprofile(old)
