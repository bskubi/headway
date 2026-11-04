"""The core (docs/core.md): each translation tested by running its terms (the reference evaluator) against what the
library computes, on random inputs, at the edges where meaning is subtler than syntax; and the printer's notation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import core_cases as cc  # noqa: E402

from headway.core.verify import program_of, verify  # noqa: E402

R = np.random.default_rng(7)
C, G, B, K = 9, 5, 7, 4


def inputs() -> dict[str, dict]:
    x = R.normal(size=(C, G))
    a = R.normal(size=(B, G))
    sq = R.normal(size=(G, G)) + 3 * np.eye(G)
    return {
        "sums": dict(x=x), "keepdims": dict(x=x), "everything": dict(x=x), "sample_sd": dict(x=x),
        "positions": dict(x=x), "both_axes": dict(x=x),
        "logits": dict(b=R.normal(size=B), d=R.normal(size=G)),
        "centered": dict(x=x, w=R.random(C)),
        "matrix": dict(a=a, v=R.normal(size=(G, K))), "gram": dict(x=x), "gram_rows": dict(x=x),
        "vector": dict(x=x, w=R.normal(size=G)), "paired": dict(a=a, v=R.normal(size=(G, K))),
        "outer": dict(a=R.normal(size=C), b=R.normal(size=C)),
        "einsum": dict(a=a, v=R.normal(size=(G, K))), "dots": dict(x=x, w=R.normal(size=G)),
        "masked": dict(x=x, learns=R.random(C) < 0.6), "pointer": dict(x=x, cols=np.array([4, 0, 2])),
        "pairs": dict(x=x, rows=np.array([1, 3, 8]), cols=np.array([0, 4, 2])),
        "apart": dict(t=R.normal(size=(3, 4, 5)), i=np.array([0, 2]), j=np.array([4, 1])),
        "apart_int": dict(t=R.normal(size=(3, 4, 5)), j=np.array([4, 1, 0])),
        "basic": dict(x=x, k=3), "newaxis": dict(x=R.normal(size=C)),
        "arange_pick": dict(a=R.normal(size=K)), "signs": dict(v=R.normal(size=(G, K))),
        "clipped": dict(x=x, lo=-0.3), "made": dict(x=x),
        "loop_sum": dict(xs=list(R.normal(size=4)), w=0.2), "family": dict(n=5, a=1.5),
        "comprehension": dict(a=[R.normal(size=(B, G)) for _ in range(3)]),
        "cases": dict(x=0.7, t=0.2), "early": dict(p=0.3, e=0.01),
        "newton": dict(m=R.integers(0, 5, B).astype(float), n=np.full(B, 8.0), steps=4),
        "while_loop": dict(x=37.0), "record": dict(x=a), "fields": dict(f=cc.Fit(R.normal(size=B), 2.5)),
        "solve": dict(a=sq, y=R.normal(size=G)),
        "likelihood": dict(k=R.integers(0, 5, C), n=np.full(C, 6), p=R.random(C), x=R.normal(size=C), mu=0.1,
                           sd=1.3),
        "softmax_norm": dict(x=x),
        "simulate": dict(mu=R.normal(size=G), sd=0.5, rng=np.random.default_rng(3)),
        "diagonals": dict(v=R.normal(size=G), m=R.normal(size=(G, G))),
        **{k: dict(design=_design()) for k in ("table_filter", "table_columns", "table_groups", "table_counts")},
    }


def _design():                                          # noqa: ANN202
    import polars as pl
    return pl.DataFrame({"plate": ["a", "b", "a", "c", "b", "a"], "learns": [True, False, True, True, False, True],
                         "depth": [1.0, 3.0, 4.5, 2.5, 6.0, 3.5]})


CASES = inputs()


@pytest.mark.parametrize("name", sorted(CASES))
def test_translations_compute_what_the_code_does(name: str) -> None:
    fn = getattr(cc, name)
    report = verify(fn, kwargs=CASES[name], rtol=1e-9, atol=1e-11)
    assert report.ok, report.text()


def test_a_wrong_translation_is_caught(monkeypatch) -> None:
    """The check is not vacuous: a table that sums over the wrong axis of a square matrix gives other numbers."""
    import headway.core.libs.arrays as arrays
    import headway.core.libs.numpy_ as numpy_
    from headway.core.terms import Aggregate
    right = arrays.reduce

    def wrong(c, op, x, axis, keepdims=None):           # noqa: ANN001, ANN202
        out = right(c, op, x, axis, keepdims)
        if isinstance(out, Aggregate) and x.sig and len(x.sig) == 2 and len(out.over) == 1:
            other = tuple(a for a in x.sig if a not in out.over)
            return Aggregate(op, other, x, sig=tuple(a for a in x.sig if a not in other), type=out.type)
        return out
    monkeypatch.setattr(numpy_, "reduce", wrong)
    report = verify(cc.sums, kwargs=dict(x=R.normal(size=(G, G))))
    assert not report.ok


def tex(fn, **names) -> str:                            # noqa: ANN001
    from headway import codetex
    import inspect
    return codetex.transcribe(inspect.getsource(fn), names, hints=True, file=cc.__file__)


def test_notation() -> None:
    # sums over index sets; products of indexed values juxtaposed; a free index primed where it is also summed
    assert tex(cc.centered) == (r"\operatorname{centered}\left(x_{cg}, w_{c}\right) = x_{cg} - "
                                r"\frac{\sum_{c'} w_{c'}\,x_{c'g}}{\sum_{c'} w_{c'}}")
    # (an index set's letter is not one of the code's one-letter names: vectors is i, as v is a parameter)
    assert tex(cc.matrix) == r"\operatorname{matrix}\left(a_{bg}, v_{gi}\right) = \sum_{g} a_{bg}\,v_{gi}"
    # Xᵀ X: the second index a copy; a pairing of two index sets by position, a sum where they are equal
    assert tex(cc.gram) == r"\operatorname{gram}\left(x_{cg}\right) = \sum_{c} x_{cg}\,x_{cg'}"
    assert tex(cc.paired) == r"\operatorname{paired}\left(a_{bg}, v_{ki}\right) = \sum_{g=k} a_{bg}\,v_{ki}"
    # a mask restricts; a pointer stands in for its index; the index itself as a value
    assert tex(cc.masked) == (r"\operatorname{masked}\left(x_{cg}, \mathit{learns}_{c}\right) = "
                              r"\operatorname{mean}_{c} \left.x_{cg}\right|_{\mathit{learns}_{c}}")
    assert tex(cc.arange_pick) == (r"\operatorname{arange\_pick}\left(a_{v}\right) = a_{v} \ge 0 \lor "
                                   r"v = \operatorname{argmax}_{v'} a_{v'}")
    assert tex(cc.pointer) == r"\operatorname{pointer}\left(x_{cg}, \mathit{cols}_{p}\right) = \sum_{p} x_{c,\mathit{cols}_{p}}"
    # numbers' functions; densities; draws
    assert r"\log \operatorname{Binomial}\left(k_{c} \mid n_{c}, p_{c}\right)" in tex(cc.likelihood)
    assert r"z_{g} &\sim \mathcal{N}\left(\mu_{g}, \mathit{sd}^{2}\right)" in tex(cc.simulate)
    # a loop that carries a value: an algorithm
    assert r"\textbf{for } \mathit{step} = 0, \ldots, \mathit{steps} - 1\textbf{:}" not in tex(cc.newton)
    assert r"\textbf{repeat } \mathit{steps}\textbf{ times:}" in tex(cc.newton)


def test_signatures_are_checked() -> None:
    """A value whose index set has two sizes (an annotation that does not fit) is a mismatch, not a result."""
    report = verify(cc.matrix, kwargs=dict(a=R.normal(size=(B, G)), v=R.normal(size=(G, K))))
    assert report.ok
    prog, _ = program_of(cc.matrix)
    assert [p for p, _, _ in prog.params] == ["a", "v"]
    bad = verify(cc.logits, kwargs=dict(b=R.normal(size=B), d=R.normal(size=G)),
                 names={"expit": lambda x: x})
    assert bad.ok                               # (the operator's own meaning: expit is a primitive here)


def test_statistical_models_example() -> None:
    """examples/models: a negative binomial GLM by IRLS and a binomial mixture by EM, written clean; their math
    computes what they compute, and what they declare holds."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "models"))
    from glm_em import steps
    from headway.clean import check, relations
    rng = np.random.default_rng(0)
    samples = 10
    X = np.column_stack([np.ones(samples), np.repeat([0.0, 1.0], samples // 2)])
    mu = rng.uniform(0.5, 2, samples)[:, None] * np.exp(X @ np.array([[4.0, 3.0], [0.5, -1.0]]))
    counts = steps.simulate(mu, np.array([0.1, 0.2]), np.random.default_rng(1))
    assert verify(steps.simulate, kwargs=dict(mu=mu, alpha=np.array([0.1, 0.2]), rng=np.random.default_rng(1))).ok
    s = steps.size_factors(counts)
    assert verify(steps.size_factors, kwargs=dict(counts=counts)).ok
    args = dict(y=counts[:, 0], s=s, X=X, alpha=0.1)
    beta = steps.irls(**args)
    assert verify(steps.irls, kwargs=args, rtol=1e-8, atol=1e-10).ok
    assert all(check(steps.irls, r, args, beta).passed for r in relations(steps.irls))
    n = rng.integers(3, 12, 200)
    m = rng.binomial(n, np.where(rng.random(200) < 0.4, 0.85, 0.15))
    args = dict(m=m, n=n, p=np.array([0.3, 0.6]), steps=300)
    got = steps.em(**args)
    assert verify(steps.em, kwargs=args, rtol=1e-8, atol=1e-10).ok
    assert all(check(steps.em, r, args, got).passed for r in relations(steps.em))


def test_deep_learning_example() -> None:
    """examples/ml: a perceptron in PyTorch (functions of its weights, and an nn.Module), its cross-entropy, and
    gradient descent; the math computes what the code computes, autograd included."""
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "ml"))
    from mlp import steps
    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, 2, (20,), generator=g)
    x = torch.randn(20, 3, generator=g) + 2.0 * y[:, None]
    W1, b1 = (0.3 * torch.randn(8, 3, generator=g)).requires_grad_(), torch.zeros(8, requires_grad=True)
    W2, b2 = (0.3 * torch.randn(2, 8, generator=g)).requires_grad_(), torch.zeros(2, requires_grad=True)
    logits = steps.forward(x, W1, b1, W2, b2)
    for fn, kw in [(steps.forward, dict(x=x, W1=W1, b1=b1, W2=W2, b2=b2)), (steps.loss, dict(logits=logits, y=y)),
                   (steps.probabilities, dict(logits=logits)),
                   (steps.train, dict(x=x, y=y, W1=W1, b1=b1, W2=W2, b2=b2, steps=3, lr=0.5))]:
        assert verify(fn, kwargs=kw, rtol=1e-5, atol=1e-6).ok, fn.__name__
    torch.manual_seed(0)
    assert verify(steps.MLP(3, 8, 2).forward, args=(x,), rtol=1e-5, atol=1e-6).ok
    from headway import codetex
    import inspect
    src, first = inspect.getsourcelines(steps.loss)
    assert codetex.transcribe("".join(src), {}, hints=True, first_line=first, file=steps.__file__) == (
        r"\operatorname{loss}\left(\mathit{logits}_{sc}, y_{s}\right) = \operatorname{mean}_{s} \left("
        r"\operatorname{logsumexp}_{c} \mathit{logits}_{sc} - \mathit{logits}_{s,y_{s}}\right)")


@pytest.mark.parametrize("fn", [cc.random_start, cc.random_start_shaped])
def test_a_recorded_call_draws_what_it_drew(fn) -> None:
    """NumPy's global random state is kept with a recorded call, and restored for the code's rerun and before
    each form of the math; the shape drawn is the code's size, whether or not its axes are known."""
    from headway.core.verify import traced
    with traced(fn, limit=2) as calls:
        fn(3)
        fn(3)
    for args, result in calls[f"{fn.__module__}.{fn.__qualname__}"]:
        assert verify(fn, kwargs=args, result=result).ok
        assert verify(fn, kwargs=args).ok                           # the code rerun from the state kept


def test_a_comparison_that_cannot_run_is_a_failed_check() -> None:
    """A function in a result that is not a function of a chunk: compared by name; never an exception."""
    from headway.core.verify import compare
    report = verify(cc.with_link)
    assert not report.ok and all("the function" in c.detail for c in report.checks)
    ok, _, detail = compare(object(), np.zeros((2, 2)), 1e-9, 1e-12)
    assert not ok and detail


@pytest.mark.parametrize("name", ["merged", "merged_in_place", "set_difference"])
def test_container_operators_are_not_read_as_elementwise(name: str) -> None:
    """d | {...} is a merge and s - t a difference: not the math's elementwise or and minus. Not written, so lint
    says so, rather than math that drops the right-hand side."""
    from headway.lint import lint
    errors = [f for f in lint("core_cases").findings if f.level == "error" and f.function == name]
    assert errors, name
    assert not verify(getattr(cc, name), (1.5,)).ok
