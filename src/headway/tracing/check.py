"""The math headway shows for a function, checked on the calls a trace recorded: each recorded call's arguments
given to the math (the function read with the index sets the trace found for its parameters), its result compared
with what the code returned. For code headway did not write: any function a trace recorded.

    headway verify TRACE pydeseq2.glm:irls_solver
"""
from __future__ import annotations

import importlib
import inspect
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from headway.core.terms import Axis, Shape
from headway.core.verify import verify
from headway.tracing.analyze import index_sets, renamed, signatures
from headway.tracing.record import Trace, named
from headway.tracing.tracer import MAX_ITEMS


@dataclass
class Verdict:
    function: str
    checked: int = 0
    agreed: int = 0
    disagreed: int = 0                  # the math ran and gave other values than the code
    reasons: Counter[str] = field(default_factory=Counter)  # why calls disagreed or could not be checked
    worst: float = 0.0

    def text(self) -> str:
        head = f"{self.function}: the math agreed with the code on {self.agreed} of {self.checked} recorded calls"
        if self.disagreed:
            head += f", DISAGREED on {self.disagreed}"
        if self.agreed:                                 # (worst: over the calls that agreed)
            head += f" (largest difference {self.worst:.2g})"
        return "\n".join([head, *(f"  {n} x {r}" for r, n in self.reasons.most_common(5))])


def _random_state(trace: Trace, c: Any) -> dict[str, Any]:
    """The module constants a call read, as recorded; numpy's global random state set to the call's own (so that
    the math's draws are the code's)."""
    consts = {k: trace.value(x) for k, x in c.globals.items() if k != "__numpy_random__"}
    state = c.globals.get("__numpy_random__")
    if state is not None:
        st = trace.value(state)
        if st is not None and len(st) == 5:
            import numpy as np
            np.random.set_state((str(st[0]), np.asarray(st[1], dtype=np.uint32), int(st[2]), int(st[3]),
                                 float(st[4])))
    return consts


_NOT_RUN = ("Uncheckable", "Mismatch", "TypeError", "AttributeError", "ValueError", "KeyError", "IndexError",
            "RuntimeError", "NameError", "ImportError", "IndexingError", "ColumnNotFoundError", "ZeroDivisionError")


def _prepared(trace: Trace, c: Any, v: Verdict, given: dict[str, Shape], rtol: float,
              atol: float) -> tuple[dict[str, Any], dict[str, Shape], tuple[float, float]] | None:
    """A recorded call, counted as checked, made ready to check: its arguments rebuilt, the index sets of those
    whose values have as many axes as the sets (a function called with a vector and with a matrix is given
    neither), and the tolerance (rounding at single precision if the call has any). None if its arguments or its
    result were not recorded whole (the reason counted)."""
    v.checked += 1
    args = {k: trace.value(a) for k, a in c.args.items()}
    missing = [k for k, a in c.args.items() if a.kind != "none" and (args[k] is None or not _rebuilt(a))]
    if missing:
        v.reasons[f"not recorded: {', '.join(missing)}"] += 1
        return None
    if not _rebuilt(c.result):
        v.reasons["its result is not recorded (a function, an object)"] += 1
        return None
    here = {k: s for k, s in given.items()
            if k in c.args and s.sig is not None and len(c.args[k].shape or ()) == len(s.sig)}
    single = any("float32" in (a.dtype or "") for a in (*c.args.values(), c.result) if a is not None)
    return args, here, ((max(rtol, 1e-5), max(atol, 1e-5)) if single else (rtol, atol))


def _check_from_source(trace: Trace, function: str, v: Verdict, given: dict[str, Shape], limit: int,
                       rtol: float, atol: float) -> Verdict:
    """A function whose module does not import here (it needs its own environment): read from its file, run with
    numpy and math and the module constants each call read, as recorded."""
    import ast
    import math
    from pathlib import Path

    import numpy as np

    from headway.core.evaluate import Evaluator
    from headway.core.lower import Lowerer, Module
    from headway.core.verify import compare
    info = trace.functions[function]
    module_name, _, qual = function.partition(":")
    module = Module.read(info.file, module_name)
    tree = ast.parse(Path(info.file).read_text())
    node: Any = tree
    for part in qual.split("."):
        node = next((n for n in getattr(node, "body", []) if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                     and n.name == part), None)
        if node is None:
            v.reasons[f"{qual} not found in {info.file}"] += 1
            return v
    node.decorator_list = []                            # @njit and the like: the function as written
    for c in trace.calls_of(function):
        if not c.kept or c.raised or v.checked >= limit:
            continue
        ready = _prepared(trace, c, v, given, rtol, atol)
        if ready is None:
            continue
        args, here, tol = ready
        consts = _random_state(trace, c)
        # the module constants the call read, as recorded, by their names; others (read where the trace did not
        # look: in a lambda, a generator) written as their values in the file
        low = Lowerer(module, here, hints=True, named=set(consts))
        prog = low.program(node)
        scope = {"np": np, "numpy": np, "math": math} | consts
        try:
            got = Evaluator(names=scope).run(prog, dict(args))
        except Exception as e:                          # noqa: BLE001 - reported, as verify does
            v.reasons[f"{type(e).__name__}: {e}"[:160]] += 1
            continue
        ok, diff, detail = compare(got, trace.value(c.result), tol[0], tol[1], True)
        if ok:
            v.agreed += 1
            v.worst = max(v.worst, diff)
        else:
            v.disagreed += 1
            v.reasons[detail[:160]] += 1
    return v


def _rebuilt(v: Any) -> bool:
    """Whether a recorded value can be rebuilt whole (not an object only described, nor a long sequence of which
    only the first items were kept)."""
    if v is None or v.kind in ("none", "scalar", "str", "array", "slice"):
        return v is None or v.kind != "array" or v.key is not None
    if _cut(v):
        return False
    if v.kind == "seq":
        return v.items is not None and all(_rebuilt(i) for i in v.items)
    if v.kind == "record":
        return all(_rebuilt(i) for i in (v.names or {}).values())
    if v.kind in ("map", "table"):
        return v.names is not None and all(_rebuilt(i) for i in v.names.values())
    return False


def _cut(v: Any) -> bool:
    """Whether a recorded value, or a part of it, was kept only in part: a sequence of which only the first
    MAX_ITEMS items were, a table of which only the first MAX_ITEMS columns were (a trace that does not say how many
    it had: any table of MAX_ITEMS columns)."""
    if v.kind == "seq" and v.items is not None and v.shape and len(v.items) < v.shape[0]:
        return True
    if v.kind == "table" and v.names is not None and (len(v.names) < v.value if isinstance(v.value, int)
                                                      else len(v.names) >= MAX_ITEMS):
        return True
    parts = [*(v.items or ()), *(v.names or {}).values()]
    return any(_cut(p) for p in parts)


def _positional(params: Any, args: dict[str, Any]) -> tuple[Any, ...]:
    """A recorded call's arguments that are given by position, taken out of args: with *args (recorded as one
    value), the parameters before it and its items; else the parameters that can only be given by position.
    **kwargs (recorded as one value) is spread into args."""
    for k, p in params.items():
        if k in args and p.kind == p.VAR_KEYWORD and isinstance(args[k], dict):
            args.update(args.pop(k))
    star = next((k for k, p in params.items() if p.kind == p.VAR_POSITIONAL), None)
    if star is not None and star in args:
        before = [k for k, p in params.items() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        return (*(args.pop(k) for k in before if k in args), *args.pop(star))
    return tuple(args.pop(k) for k, p in params.items() if p.kind == p.POSITIONAL_ONLY and k in args)


def resolve(function: str) -> Callable[..., Any]:
    module, _, qual = function.partition(":")
    obj: Any = importlib.import_module(module)
    for part in qual.split("."):
        owner = obj
        obj = owner.__dict__[part] if isinstance(owner, type) and part in owner.__dict__ else getattr(owner, part)
    # a method behind a descriptor (property, a cached property): its function
    for attr in ("fget", "func", "__func__"):
        if not callable(obj) or isinstance(obj, (property, staticmethod, classmethod)):
            obj = getattr(obj, attr, obj)
    if type(obj).__name__ in ("cache_readonly", "cached_property", "CachedAttribute"):
        obj = getattr(obj, "func", getattr(obj, "fget", obj))
    return obj  # type: ignore[no-any-return]


def given_shapes(trace: Trace, function: str, renames: dict[str, str] | None = None) -> dict[str, Shape]:
    sets = renamed(index_sets(trace, [function]), renames or {})
    out = {}
    for param, dims in signatures(sets, function).items():
        if param != "return":
            out[param] = Shape("numpy.ndarray", tuple(Axis(d) if d != "1" else Axis(None) for d in dims))
    return out


def check_function(trace: Trace, function: str, limit: int = 50, renames: dict[str, str] | None = None,
                   rtol: float = 1e-7, atol: float = 1e-10) -> Verdict:
    full = next((f for f in trace.functions if named(f, function)), function)
    v = Verdict(full)
    given = given_shapes(trace, full, renames)
    try:
        fn = resolve(full)
    except ImportError:                                 # the code's own environment only (an older Python):
        return _check_from_source(trace, full, v, given, limit, rtol, atol)   # read from its file instead
    params = inspect.signature(fn).parameters
    for c in trace.calls_of(full):
        if not c.kept or c.raised or v.checked >= limit:
            continue
        ready = _prepared(trace, c, v, given, rtol, atol)
        if ready is None:
            continue
        args, here, tol = ready
        positional = _positional(params, args)
        want = trace.value(c.result)
        consts = _random_state(trace, c)
        report = verify(fn, args=positional, kwargs=args, result=want, rtol=tol[0], atol=tol[1], scaled=True, given=here,
                        forms=("lowered",), names=consts, constants=not consts)
        bad = [k for k in report.checks if not k.ok]
        if bad:
            v.reasons[bad[0].detail[:160]] += 1
            if not bad[0].detail.startswith(_NOT_RUN):  # it ran and differed (not: it could not be run)
                v.disagreed += 1
        else:
            v.agreed += 1
            v.worst = max(v.worst, max((k.difference for k in report.checks), default=0.0))
    return v
