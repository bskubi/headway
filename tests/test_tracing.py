"""The tracer (headway record), its readings of a trace (paths, index sets) and the math checked on a trace's
calls (headway verify), on a small package."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
sys.path.insert(0, str(Path(__file__).parent / "tracing_toy"))

import toypkg  # noqa: E402

from headway.tracing import Trace, Tracer, index_sets, paths, signatures  # noqa: E402
from headway.tracing.check import check_function  # noqa: E402


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
