"""Checking clean code on a real run: the code's own tests run (nothing in the code changed), every call of the
functions clean code stands for watched, and afterwards, on the arguments of those calls:

  - the clean version against the original (its `equals`/`approximates` declaration, within its tolerance);
  - its other declarations (maximizes, minimizes, solves: clean.py);
  - the math shown against the clean code: each clean function (and each function of the clean module it calls)
    run by the reference evaluator, as written in the viewer (core/verify.py).

What a run found is written beside the clean code (.headway-checks.json), for the viewer, with the command and the
time. `swap` instead puts the clean versions in place of the originals and runs the tests, so that the clean code is
seen to work where the original did.

    headway trace clean_methembed.steps -- -q tests/
    headway trace clean_methembed.steps --swap -- -q tests/
"""
from __future__ import annotations

import copy
import datetime
import importlib
import inspect
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from headway.clean import Relation, Result, check as check_relation, relations, resolve
from headway.core.evaluate import safe_copy
from headway.core.verify import compare, traced, verify

MATH_RTOL = 1e-6                                     # the math shown, against the code: up to rounding


@dataclass
class Clean:
    """A clean function and the original it stands for (its equals or approximates declaration)."""
    fn: Callable[..., Any]
    original: str | None
    rel: Relation | None

    @property
    def name(self) -> str:
        return f"{self.fn.__module__}:{self.fn.__qualname__}"


@dataclass
class Findings:
    evidence: dict[str, dict[tuple[str, str], list[Result]]] = field(default_factory=lambda: defaultdict(
        lambda: defaultdict(list)))
    counts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    notes: list[str] = field(default_factory=list)
    unchecked: list[str] = field(default_factory=list)      # what no call of the run tested


def clean_functions(module: ModuleType) -> list[Clean]:
    """The module's functions, and its classes' methods (an nn.Module's forward), with what each stands for."""
    out = []
    found = [fn for fn in vars(module).values() if inspect.isfunction(fn)]
    for cls in vars(module).values():
        if inspect.isclass(cls) and cls.__module__ == module.__name__:
            found += [f for k, f in vars(cls).items() if inspect.isfunction(f) and not k.startswith("__")]
    for fn in found:
        if fn.__module__ != module.__name__:
            continue
        rels = relations(fn)
        rel = next((r for r in rels if r.kind in ("equals", "approximates") and isinstance(r.target, str)), None)
        out.append(Clean(fn, rel.target if rel else None, rel))  # type: ignore[arg-type]
    return out


def _settings(assignments: list[str]) -> None:
    """--set module.NAME=value: a module constant changed before the run (e.g. a chunk size, so the code's chunked
    paths run)."""
    for a in assignments:
        target, _, value = a.partition("=")
        module, _, name = target.rpartition(".")
        m = importlib.import_module(module)
        setattr(m, name, type(getattr(m, name))(value))


def check(module_name: str, pytest_args: list[str], settings: list[str] = (), limit: int = 50,  # type: ignore
          log: Callable[[str], None] = print) -> Findings:
    """Run the tests, recording the calls of every original a clean function stands for; then check each."""
    module = importlib.import_module(module_name)
    _settings(list(settings))
    cleans = clean_functions(module)
    stands = [c for c in cleans if c.original]
    originals = {}
    for c in stands:
        try:
            originals[inspect.unwrap(resolve(c.original)).__code__] = c  # type: ignore[arg-type]
        except (ImportError, AttributeError) as e:
            log(f"{c.name}: what it stands for, {c.original}, is not found ({e})")
    calls: list[tuple[Clean, dict[str, Any], Any]] = []
    started: dict[int, dict[str, Any]] = {}
    per: Counter[str] = Counter()
    # the clean module's own functions, called by the run itself (clean code with no original: written clean)
    own = {inspect.unwrap(c.fn).__code__: c for c in cleans}
    own_calls: list[tuple[Clean, dict[str, Any], Any]] = []

    def hook(frame: Any, event: str, arg: Any) -> None:
        c = originals.get(frame.f_code)
        mine = own.get(frame.f_code) if c is None else None
        if c is None and mine is None:
            return
        which = c or mine
        assert which is not None
        key = ("original:" if c else "own:") + which.name
        if event == "call":
            if per[key] >= limit:
                return
            names = inspect.signature(which.fn).parameters
            started[id(frame)] = {k: safe_copy(v) for k, v in frame.f_locals.items() if k in names}
        elif event == "return" and id(frame) in started:
            per[key] += 1
            (calls if c else own_calls).append((which, started.pop(id(frame)), safe_copy(arg)))

    import pytest
    sys.setprofile(hook)
    try:
        code = pytest.main(list(pytest_args))
    finally:
        sys.setprofile(None)
    log(f"tests: exit {int(code)}; {len(calls)} calls of originals, {len(own_calls)} of the clean code recorded")
    f = Findings()
    helpers = [fn for fn in vars(module).values() if inspect.isfunction(fn) and fn.__module__ == module.__name__]
    math_calls: dict[str, list[tuple[dict[str, Any], Any]]] = defaultdict(list)
    for c, args, result in own_calls:                   # the clean code's own calls: its declarations, its math
        name = f"{c.fn.__module__}.{c.fn.__qualname__}"
        math_calls[name].append((args, result))
        for rel in relations(c.fn):
            if rel is c.rel or result is None:
                continue
            _objective_call(rel, args, result, math_calls)
            target = rel.target if isinstance(rel.target, str) else \
                f"{rel.target.__module__}:{rel.target.__qualname__}"
            f.evidence[c.name][(rel.kind, target)].append(check_relation(c.fn, rel, args, result))
    for c, args, want in calls:
        counts = f.counts[c.name]
        if want is None:
            counts["raised"] += 1
            continue
        try:                                            # the clean version, its helpers' calls recorded too
            with traced(*helpers, limit=limit) as inner:
                got = c.fn(**{k: safe_copy(v) for k, v in args.items()})
        except AssertionError:
            counts["outside its conditions"] += 1
            continue
        for name, recorded in inner.items():
            if len(math_calls[name]) < limit:
                math_calls[name] += recorded[:limit - len(math_calls[name])]
        for k, rel in enumerate(relations(c.fn)):
            if rel is c.rel:
                adapted = rel.adapt(got) if rel.adapt else got
                ok, diff, detail = compare(adapted, want, rel.rtol, rel.atol)
                f.evidence[c.name][(rel.kind, str(rel.target))].append(Result(ok, diff, (detail or (
                    f"largest difference {diff:.2g}")) + f" (allowed: {rel.atol:g} + {rel.rtol:g} × the value)"))
                counts["agreed" if ok else "DISAGREED"] += 1
            else:
                target = rel.target if isinstance(rel.target, str) else \
                    f"{rel.target.__module__}:{rel.target.__qualname__}"
                f.evidence[c.name][(rel.kind, target)].append(check_relation(c.fn, rel, args, got))
                _objective_call(rel, args, got, math_calls)
    # the math shown, against the clean code, on the calls the run made
    for name, recorded in math_calls.items():
        fn = _function(module, name)
        if fn is None:
            continue
        key = f"{fn.__module__}:{fn.__qualname__}"
        for args, result in recorded[:limit]:
            rtol, atol = _math_tolerance(fn, result)
            report = verify(fn, kwargs=args, result=result, rtol=rtol, atol=atol, scaled=True)
            worst = max((c.difference for c in report.checks), default=0.0)
            bad = [c for c in report.checks if not c.ok]
            detail = (f"largest difference {worst:.2g} (allowed: {atol:g} × the largest value + {rtol:g} × the "
                      f"value)" if not bad else f"as {bad[0].form}: {bad[0].detail}")
            f.evidence[key][("math", "")].append(Result(not bad, worst, detail))
    # what the run did not check: functions the math of which no call ran, declarations no call tested
    for c in cleans:
        key = c.name
        rels = f.evidence.get(key, {})
        if ("math", "") not in rels and not c.fn.__name__.startswith("_"):
            f.unchecked.append(f"{key}: no call ran it, so its math is not checked")
        for r in relations(c.fn):
            target = r.target if isinstance(r.target, str) else f"{r.target.__module__}:{r.target.__qualname__}"
            if (r.kind, target if r is not c.rel else str(r.target)) not in rels:
                f.unchecked.append(f"{key}: its {r.kind} {target.split(':')[-1]} is not checked (no call)")
    write(module, f, "pytest " + " ".join(pytest_args), log)
    for u in f.unchecked:
        log(f"  not checked: {u}")
    return f


def _objective_call(rel: Relation, args: dict[str, Any], result: Any,
                    math_calls: dict[str, list[tuple[dict[str, Any], Any]]]) -> None:
    """A declaration's objective (or equation, or constraint), called on what the call gave and took: its math is
    shown in the Specification, so it is checked too."""
    values = dict(args)
    items = result if len(rel.over) > 1 else (result,)
    values |= dict(zip(rel.over, items))
    for target in (rel.target, *rel.subject_to):
        if not callable(target) or isinstance(target, str):
            continue
        try:
            names = list(inspect.signature(target).parameters)
            kwargs = {n: safe_copy(values[n]) for n in names if n in values}
            out = target(**{k: safe_copy(v) for k, v in kwargs.items()})
        except Exception:                               # noqa: BLE001 (not callable on these: not checked)
            continue
        math_calls[f"{target.__module__}.{target.__qualname__}"].append((kwargs, out))


def _math_tolerance(fn: Callable[..., Any], result: Any = None) -> tuple[float, float]:
    """How near the math's values must be to the code's: up to rounding (single precision's, if the code's result
    is in it; the math is run in double where the code may not be), or within the tolerance the function declares
    for itself against its original (an eigendecomposition's sensitivity is the same in both)."""
    rtol = atol = MATH_RTOL
    if "float32" in str(getattr(result, "dtype", "")) or "float16" in str(getattr(result, "dtype", "")):
        rtol = atol = 1e-5 if "32" in str(result.dtype) else 1e-2
    for r in relations(fn):
        if r.kind in ("equals", "approximates"):
            rtol, atol = max(rtol, r.rtol), max(atol, r.atol)
    return rtol, atol


def _function(module: ModuleType, qualname: str) -> Callable[..., Any] | None:
    """A function of the module by its qualified name (module.f, or module.Class.method)."""
    rest = qualname.removeprefix(module.__name__ + ".")
    obj: Any = module
    for part in rest.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj if inspect.isfunction(obj) else None


def write(module: ModuleType, f: Findings, run: str, log: Callable[[str], None] = print) -> Path:
    """What the run showed, beside the clean code: per function and declaration (and "math": the math shown), the
    calls checked, how many bore it out, and the worst of them."""
    out: dict[str, list[dict[str, Any]]] = {}
    for name, rels in f.evidence.items():
        entries = []
        for (kind, target), results in rels.items():
            worst = max(results, key=lambda r: r.measure)
            entries.append({"kind": kind, "target": target, "calls": len(results),
                            "passed": sum(r.passed for r in results), "worst": worst.measure,
                            "detail": worst.detail})
        out[name] = entries
    path = Path(inspect.getfile(module)).parent / ".headway-checks.json"
    path.write_text(json.dumps({"run": run, "when": datetime.datetime.now().isoformat(timespec="minutes"),
                                "checks": out}, indent=1))
    for name, entries in sorted(out.items()):
        for e in entries:
            what = "the math shown" if e["kind"] == "math" else f"{e['kind']} {str(e['target']).split(':')[-1]}"
            log(f"  {name.split(':')[1]:22} {what:45} {e['passed']}/{e['calls']}  {e['detail']}")
    for name, counts in sorted(f.counts.items()):
        others = {k: v for k, v in counts.items() if k not in ("agreed",)}
        if others:
            log(f"  {name.split(':')[1]:22} {dict(others)}")
    log(f"wrote {path}")
    return path


def swap(module_name: str, pytest_args: list[str], settings: list[str] = (),  # type: ignore[assignment]
         log: Callable[[str], None] = print) -> int:
    """The clean versions in place of the originals (every module's name bound to one), the tests run, the calls of
    each counted: a swap that did not take is seen. A clean version whose conditions (its asserts) a call does not
    meet hands the call to the original."""
    module = importlib.import_module(module_name)
    _settings(list(settings))
    counts: Counter[str] = Counter()
    watched = {}
    for c in clean_functions(module):
        if not c.original:
            continue
        original = resolve(c.original)
        watched[inspect.unwrap(original).__code__] = f"{c.fn.__name__} (original)"
        watched[c.fn.__code__] = f"{c.fn.__name__} (clean)"
        adapt = c.rel.adapt if c.rel else None

        def replacement(*a: Any, _fn: Any = c.fn, _original: Any = original, _adapt: Any = adapt, **k: Any) -> Any:
            try:
                out = _fn(*a, **k)
            except AssertionError:
                counts[f"{_fn.__name__}: outside its conditions, the original used"] += 1
                return _original(*a, **k)
            return _adapt(out) if _adapt else out
        package = c.original.split(".")[0]
        bound = 0
        for m in list(sys.modules.values()):
            if getattr(m, "__name__", "").startswith(package):
                for attr, v in list(vars(m).items()):
                    if v is original:
                        setattr(m, attr, replacement)
                        bound += 1
        log(f"swapped {c.original}: {bound} binding(s)")

    def hook(frame: Any, event: str, arg: Any) -> None:
        if event == "call" and frame.f_code in watched:
            counts[watched[frame.f_code]] += 1
    import pytest
    sys.setprofile(hook)
    try:
        code = int(pytest.main(list(pytest_args)))
    finally:
        sys.setprofile(None)
    log(f"tests: exit {code}")
    for k in sorted(counts):
        log(f"  {k}: {counts[k]}")
    return code
