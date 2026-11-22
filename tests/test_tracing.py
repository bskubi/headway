"""The tracer (headway record), its readings of a trace (paths, index sets) and the math checked on a trace's
calls (headway verify), on a small package."""
import importlib
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("scipy")
sys.path.insert(0, str(Path(__file__).parent / "tracing_toy"))

import toypkg  # noqa: E402

from headway.tracing import Trace, Tracer, index_sets, paths, signatures  # noqa: E402
from headway.tracing.analyze import describe_paths  # noqa: E402
from headway.tracing.check import Verdict, _check_from_source, _cut, _positional, _prepared, check_function  # noqa: E402
from headway.tracing.record import Value  # noqa: E402


def run() -> None:
    for n, g, p in [(6, 4, 2), (10, 7, 3), (14, 5, 4)]:
        rng = np.random.default_rng(n)
        X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
        counts = rng.poisson(20, size=(n, g)) + 1.0
        toypkg.fit_all(counts, np.ones(n), X, np.full(g, 0.1))
    toypkg.poisson_rate(np.array([3.0, 5.0, 4.0]), np.array([1.0, 2.0, 1.5]))
    np.random.seed(3)
    toypkg.jitter(np.zeros(4), 0.5)


@pytest.fixture(scope="module")
def trace(tmp_path_factory: pytest.TempPathFactory) -> Trace:
    t = Tracer(["toypkg"], first=5, every=3, limit=20)
    with t:
        run()
    d = t.write(tmp_path_factory.mktemp("trace"))
    return Trace.read(d)


def test_every_call_counted_and_a_sample_recorded(trace: Trace) -> None:
    fits = trace.calls_of("fit")
    assert len(fits) == 4 + 7 + 5 == trace.functions["toypkg:fit"].calls
    kept = [c for c in fits if c.kept]
    assert 5 <= len(kept) < len(fits)
    assert all(c.args["counts"].shape is not None for c in fits)        # sizes of every call
    assert trace.value(kept[0].args["counts"]) is not None
    assert trace.environment["python"] and "numpy" in trace.environment["distributions"]


def test_paths_and_loops(trace: Trace) -> None:
    ps = paths(trace, "fit")
    assert sum(p.calls for p in ps) == 16
    assert any(n > p.calls for p in ps for n in p.loops.values())      # the IRLS loop ran more than once


def test_index_sets_from_sizes(trace: Trace) -> None:
    sigs = signatures(index_sets(trace, ["toypkg:fit"]), "toypkg:fit")
    samples, coefs = sigs["X"]
    assert sigs["counts"] == (samples,) and sigs["size_factors"] == (samples,)
    assert coefs != samples and sigs["return"] == (coefs,)


def test_math_checked_on_traced_calls(trace: Trace) -> None:
    for name in ("toypkg:fit", "toypkg:fit_all", "toypkg:poisson_rate", "toypkg:jitter"):
        v = check_function(trace, name)
        assert v.checked and v.agreed == v.checked, v.text()


def test_settrace_tracer_for_older_pythons(tmp_path: Path) -> None:
    """The tracer older Pythons use (sys.settrace): the same calls, sizes and paths."""
    t = Tracer(["toypkg"], first=5, every=3, limit=20, legacy=True)
    with t:
        run()
    old = Trace.read(t.write(tmp_path))
    assert len(old.calls_of("fit")) == 16
    sigs = signatures(index_sets(old, ["toypkg:fit"]), "toypkg:fit")
    assert sigs["counts"] == sigs["size_factors"] != sigs["return"]
    assert any(n > p.calls for p in paths(old, "fit") for n in p.loops.values() if p.calls)
    assert check_function(old, "toypkg:fit").agreed > 0


def test_calls_past_the_cap_are_counted_by_path(tmp_path: Path) -> None:
    t = Tracer(["toypkg"], first=2, every=1000, limit=5, stored=3)
    with t:
        run()
    capped = Trace.read(t.write(tmp_path))
    assert len(capped.calls_of("fit")) == 3
    assert sum(p.calls for p in paths(capped, "fit")) == 16 == capped.functions["toypkg:fit"].calls


# ---------- packages to trace ----------
def _package(tmp: Path, name: str, source: str) -> Any:
    (tmp / name).mkdir()
    (tmp / name / "__init__.py").write_text(textwrap.dedent(source))
    sys.path.insert(0, str(tmp))
    importlib.invalidate_caches()
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(tmp))


def _package_on_path(tmp: Path, name: str, source: str) -> Any:
    (tmp / name).mkdir()
    (tmp / name / "__init__.py").write_text(textwrap.dedent(source))
    sys.path.insert(0, str(tmp))
    importlib.invalidate_caches()
    return importlib.import_module(name)


def _importable_package(tmp_path: Path, name: str, source: str) -> Any:
    """A package of the given source, importable (the tracer skips test modules, so traced code lives here)."""
    (tmp_path / name).mkdir()
    (tmp_path / name / "__init__.py").write_text(textwrap.dedent(source))
    sys.path.insert(0, str(tmp_path))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(tmp_path))


def _written(t: Tracer, tmp: Path) -> Trace:
    t.write(tmp / "trace")
    return Trace.read(tmp / "trace")


TRACED = """
class It:
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.n <= 0:
            raise StopIteration
        self.n -= 1
        return self.n


def fin():
    try:
        raise ValueError("x")
    finally:
        y = 1


def caught(s):
    try:
        return int(s)
    except ValueError:
        return 0


def loop(n):
    t = 0
    for x in It(n):
        t += x
    return t


def doc(x):
    '''A docstring
    over lines.'''
    y = x + 1
    return y


def gen(n):
    for i in range(n):
        yield i


def f(v):
    return v + 1


def consume(n):
    out = 0
    for v in gen(n):
        out += f(v)
    return out


def total(xs, w):
    s = 0.0
    for x in xs:
        s += x
    return s * w


def spread(*xs):
    s = 0.0
    for x in xs:
        s += x
    return s
"""


@pytest.fixture
def traced_package(tmp_path: Path) -> Any:
    name = f"hw_traced_{abs(hash(str(tmp_path)))}"
    return name, _importable_package(tmp_path, name, TRACED)


# ---------- the tracer: raises, branches, generators ----------
def test_settrace_records_a_raise_and_does_not_count_unfollowed_paths(tmp_path: Path) -> None:
    pkg = _package_on_path(tmp_path, "rd_raises", """
        def boom(x):
            if x > 0:
                raise ValueError("no")
            return x

        def branch(x):
            if x > 0:
                y = 1
            else:
                y = 2
            return y
        """)
    t = Tracer(["rd_raises"], first=2, legacy=True)
    with t:
        try:
            pkg.boom(1)
        except ValueError:
            pass
        for i in range(10):
            pkg.branch(i)
    boom = next(c for c in t.calls if c.function.endswith("boom"))
    assert boom.raised == "ValueError" and boom.result is None
    trace = _written(t, tmp_path)
    assert check_function(trace, "boom").checked == 0      # a call that raised: not checked (nor re-run)
    text = describe_paths(trace, "branch")
    assert "not followed" in text and "8 (80.0%)  the most common path" not in text


def test_settrace_tells_raising_from_returning(traced_package: Any) -> None:
    name, m = traced_package
    t = Tracer([name], first=1, every=1000, limit=1, legacy=True)
    with t:
        with pytest.raises(ValueError):
            m.fin()
        for _ in range(3):
            m.caught("q")
            m.loop(3)
    by = {(c.function.split(":")[1], c.seq): c for c in t.calls}
    assert by[("fin", 0)].raised == "ValueError"
    assert all(by[("caught", k)].raised is None for k in range(3))     # handled inside: not raised
    assert all(by[("loop", k)].raised is None for k in range(3))       # its iterator's StopIteration
    assert by[("It.__next__", 3)].raised == "StopIteration"


def test_settrace_sees_no_branch_from_the_def_line(traced_package: Any) -> None:
    name, m = traced_package
    t = Tracer([name], legacy=True)
    with t:
        m.doc(1)
    (c,) = t.calls
    assert c.branches == {}


@pytest.mark.parametrize("legacy", [False, True])
def test_generators_are_not_traced(traced_package: Any, legacy: bool) -> None:
    name, m = traced_package
    t = Tracer([name], legacy=legacy)
    with t:
        m.consume(3)
    functions = [c.function.split(":")[1] for c in t.calls]
    assert "gen" not in functions
    consume = next(c for c in t.calls if c.function.endswith(":consume"))
    assert all(c.parent == consume.id for c in t.calls if c.function.endswith(":f"))


# ---------- paths of calls that raised ----------
def test_calls_that_raised_are_not_the_most_common_path(tmp_path: Path) -> None:
    pkg = _package(tmp_path, "rd3_raises", """
        def f(x):
            if x > 0:
                raise ValueError("no")
            if x < -5:
                return 1
            return 0
        """)
    t = Tracer(["rd3_raises"], first=3, every=1, limit=3, stored=4)
    with t:
        for i in range(10):
            try:
                pkg.f(1)
            except ValueError:
                pass
        pkg.f(0)
    t.write(tmp_path / "trace")
    trace = Trace.read(tmp_path / "trace")
    ps = paths(trace, "f")
    # stored and past the cap, the calls that raised are one path
    assert sum(p.calls for p in ps if p.branches == frozenset({"raised ValueError"})) == 10
    text = describe_paths(trace, "f")
    assert "the most common path" in text and "10 raised" in text
    common = next(line for line in text.splitlines() if "the most common path" in line)
    assert common.strip().startswith("1 ")


# ---------- recorded values read back and checked ----------
def test_a_long_list_kept_in_part_is_not_checked(tmp_path: Path) -> None:
    pkg = _package_on_path(tmp_path, "rd_long", """
        def total(xs):
            return sum(xs)
        """)
    t = Tracer(["rd_long"])
    with t:
        pkg.total([1.0] * 100)
    v = check_function(_written(t, tmp_path), "total")
    assert v.disagreed == 0 and "not recorded: xs" in v.reasons
    assert "largest difference" not in v.text()             # nothing agreed: no largest difference to report


def test_a_check_from_source_writes_constants_not_recorded(tmp_path: Path) -> None:
    pkg = _package_on_path(tmp_path, "rd_const", """
        SCALE = 3.0

        def scaled(xs):
            return sum(x * SCALE for x in xs)
        """)
    t = Tracer(["rd_const"])
    with t:
        pkg.scaled([1.0, 2.0])
    v = _check_from_source(_written(t, tmp_path), "rd_const:scaled", Verdict("rd_const:scaled"), {}, 50, 1e-7, 1e-10)
    assert v.agreed == 1, v.text()


def test_an_empty_list_is_read_back_as_recorded(traced_package: Any, tmp_path: Path) -> None:
    name, m = traced_package
    t = Tracer([name])
    with t:
        m.total([], 2.0)
        m.total([1.0, 2.0], 2.0)
        m.spread(1.0, 2.0)
    t.write(tmp_path / "trace")
    trace = Trace.read(tmp_path / "trace")
    assert trace.calls[0].args["xs"].items == []
    v = check_function(trace, "total")
    assert (v.agreed, v.checked) == (2, 2), v.text()
    v = check_function(trace, "spread")                 # *args given by position
    assert (v.agreed, v.disagreed) == (1, 0), v.text()


def test_a_table_is_cut_only_if_columns_were_left_out() -> None:
    columns = {f"c{k}": Value("array", dtype="float64", shape=[3], key=f"k{k}") for k in range(32)}
    assert not _cut(Value("table", "pandas.DataFrame", shape=[3], value=32, names=columns))
    assert _cut(Value("table", "pandas.DataFrame", shape=[3], value=40, names=columns))
    assert _cut(Value("table", "pandas.DataFrame", shape=[3], names=columns))     # an older trace


def test_an_argument_rebuilt_only_in_part_is_not_checked() -> None:
    class FakeTrace:
        def value(self, v: Any) -> Any:
            return {"a": None} if v.kind == "map" else 1.0
    from headway.tracing.record import Call
    arg = Value("map", "builtins.dict", names={"a": Value("array", dtype="float64", shape=[10 ** 8])})
    c = Call(0, "m:f", 0, args={"d": arg}, result=Value("scalar", value=1.0))
    v = Verdict("m:f")
    assert _prepared(FakeTrace(), c, v, {}, 1e-7, 1e-10) is None    # type: ignore[arg-type]
    assert v.reasons == {"not recorded: d": 1}


def test_positional_arguments_of_a_recorded_call() -> None:
    import inspect

    def f(a, *xs, k=1):
        return a

    args = {"a": 1, "xs": (2, 3), "k": 4}
    assert _positional(inspect.signature(f).parameters, args) == (1, 2, 3)
    assert args == {"k": 4}
