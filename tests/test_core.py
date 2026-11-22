"""The core (docs/core.md): each translation tested by running its terms (the reference evaluator) against what the
library computes, on random inputs, at the edges where meaning is subtler than syntax; and the printer's notation."""
from __future__ import annotations

import ast
import inspect
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import core_cases as cc  # noqa: E402

from headway.core.evaluate import Evaluator  # noqa: E402
from headway.core.fold import fold  # noqa: E402
from headway.core.lower import Lowerer, Module, lower_source  # noqa: E402
from headway.core.printer import indexed, scripted  # noqa: E402
from headway.core.terms import Apply, For, Op, Unwritten, free_vars, walk  # noqa: E402
from headway.core.verify import program_of, traced, verify  # noqa: E402

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


# ---------- single functions checked as written ----------
def _ok(fn: Any, *args: Any) -> bool:
    report = verify(fn, args)
    return report.ok


def _ok_in_two_checks(fn: Any, *args: Any) -> bool:
    report = verify(fn, args)
    return report.ok and len(report.checks) == 2


# ---------- lowering: operators ----------
def bits(n: int) -> int:
    return n & 6


def array_bits(a: np.ndarray) -> np.ndarray:
    return (a & 1) | (a ^ 2)


def masks(x: np.ndarray) -> np.ndarray:
    return (x > 0) & (x < 3)


def test_bitwise_operators_of_integers_are_bitwise() -> None:
    assert _ok_in_two_checks(bits, 5)
    assert _ok_in_two_checks(array_bits, np.array([1, 2, 3, 4]))
    assert _ok_in_two_checks(masks, np.array([-1.0, 1.0, 2.0, 4.0]))
    # booleans combined are still a logical and
    ret = lower_source("def f(x):\n    return (x > 0) & (x < 3)").body[-1]
    assert isinstance(ret.value, Apply) and ret.value.fn == "and"     # type: ignore[attr-defined]


def test_bitwise_of_an_integer_and_a_boolean_is_an_integer() -> None:
    ret = lower_source("def f(n: int):\n    return (n > 0) & 1\n").body[-1]
    assert isinstance(ret.value, Apply) and ret.value.fn == "bitand"    # type: ignore[attr-defined]
    assert ret.value.type == "int"                                      # type: ignore[attr-defined]


def mask64(x):
    return x & 0xFFFFFFFFFFFFFFFF


def shift70(x):
    return x << 70


def test_bitwise_operators_keep_pythons_unbounded_integers() -> None:
    assert _ok(mask64, 5)
    assert _ok(shift70, 1)


def mask_and(n, x):
    return (n > 0) & (x > 0)


def mask_or(n, x):
    return (n > 0) | (x > 0)


def python_and(x):
    return x is not None and x > 0


def test_booleans_combined_by_and_or_keep_the_arrays_shape() -> None:
    x = np.array([1.0, -1.0, 2.0])
    assert _ok(mask_and, 0, x)
    assert _ok(mask_or, 1, x)
    assert _ok(python_and, None)                        # Python's and: the second side not evaluated


def between(xs, k):
    return 0 <= k < len(xs) < 10


def test_a_chained_comparison_is_pythons_and() -> None:
    assert _ok(between, [1, 2], 1)
    prog, _ = program_of(between)
    ret = prog.body[-1]
    assert any(isinstance(e, Apply) and e.fn == "andthen" for e in walk(ret.value))  # type: ignore[attr-defined]


def default(k, x):
    n = k or 10.0
    return x * n


def second(k, x):
    n = k and 2.5
    return x * n


def test_and_or_give_an_operand_not_a_bool() -> None:
    assert _ok(default, 0, 2.0)
    assert _ok(default, 3, 2.0)
    assert _ok(second, 3, 2.0)
    assert _ok(second, 0, 2.0)
    n = program_of(default)[0].body[0].value          # type: ignore[attr-defined]
    assert n.type != "bool" and n.dtype != "bool"


# ---------- lowering: comprehensions ----------
def parities(xs: list) -> set:
    return {x % 2 for x in xs}


def test_a_set_comprehension_is_a_set() -> None:
    assert _ok_in_two_checks(parities, [1, 2, 3, 4])
    ret = lower_source("def f(xs):\n    return {x for x in xs}").body[-1]
    assert any(isinstance(e, Op) and e.qualname == "builtins.set" for e in walk(ret.value))  # type: ignore[attr-defined]


def flat(n, i):
    y = i + 1
    return [j for i in range(n) for j in range(y)]


def test_a_value_is_not_captured_by_an_earlier_binder() -> None:
    assert _ok(flat, 3, 1)


def dict_flat(n, i):
    y = i + 1
    return {(i, j): 1 for i in range(n) for j in range(y)}


def dict_same_name(k):
    xs = [k, k + 1]
    return {xs: 0 for xs in xs}


def test_a_dict_comprehensions_domains_are_outside_its_binders() -> None:
    assert _ok(dict_flat, 3, 1)
    assert _ok(dict_same_name, 1)


# ---------- lowering: try, yield and statements not written ----------
def finally_(x: float) -> float:
    try:
        y = x + 1
    finally:
        y = x * 10
    return y


def try_else(x: float) -> float:
    y = 0.0
    try:
        y = x + 1
    except ValueError:
        y = -1.0
    else:
        y = y * 2
    return y


def test_a_try_is_not_cut_short() -> None:
    assert _ok_in_two_checks(finally_, 2.0)
    assert not verify(try_else, (2.0,)).ok                  # its handler is not written: not checked as written
    assert any(isinstance(st, Unwritten) for st in lower_source(inspect.getsource(try_else)).body)


def finally_returns(a, b):
    try:
        return a
    finally:
        return b


def test_a_try_whose_finally_returns_is_not_written() -> None:
    prog, _ = program_of(finally_returns)
    assert any(isinstance(s, Unwritten) for s in prog.body)


def try_continue(xs):
    count = 0
    for v in xs:
        try:
            if v > 1:
                continue
        finally:
            count += 1
    return count


def try_return(xs):
    out = []
    try:
        return out
    finally:
        out.append(1)


def try_inner_loop(xs):
    total = 0.0
    try:
        for v in xs:
            if v > 1:
                break
            total += v
    finally:
        total = total * 2
    return total


def test_a_try_left_by_return_or_continue_is_not_written_as_if_the_finally_ran_after() -> None:
    for fn, args in [(try_continue, ([0, 2, 3, 0],)), (try_return, ([],))]:
        for check in verify(fn, args).checks:
            assert check.ok or check.detail.startswith("Uncheckable"), check.detail
    assert _ok(try_inner_loop, [0.5, 1.0, 2.0, 0.1])    # a break of a loop inside the try: written


def test_a_yield_is_not_written() -> None:
    body = lower_source("def g(xs):\n    for x in xs:\n        yield 2 * x\n").body
    assert isinstance(body[0], For) and any(isinstance(s, Unwritten) for s in body[0].body)


def side_comprehension(xs):
    out = []
    [out.append(v * 2) for v in xs]
    return out


def walrus(x):
    (y := x * 2)
    return y + 1


def bare_name(x):
    x
    return x + 1


def test_an_expression_statement_that_is_not_a_call_is_not_dropped() -> None:
    for fn, args in [(side_comprehension, ([1.0, 2.0],)), (walrus, (1.0,))]:
        prog = program_of(fn)[0]
        assert any(isinstance(st, Unwritten) for st in prog.body)    # shown as not written, not left out
        assert all(not c.ok and c.detail.startswith("Uncheckable") for c in verify(fn, args).checks)
    assert _ok(bare_name, 1.0)


def history(xs: list) -> list:
    log = []
    for x in xs:
        log.append(x * 2)
    return log


def test_a_variable_named_like_a_logger_is_not_output() -> None:
    assert _ok_in_two_checks(history, [1.0, 2.0])


# ---------- lowering: local and module functions written in place ----------
def inlined_clobbers(x):
    def h(a):
        t = a + 1
        return t * 2
    t = 100.0
    y = h(x)
    return y + t


def inlined_captures(x, n):
    def h(a):
        n = 2.0
        return a * n
    y = h(n)
    return y + x


def test_a_local_function_is_not_written_in_place_over_the_callers_names() -> None:
    assert _ok(inlined_clobbers, 1.0)
    assert _ok(inlined_captures, 1.0, 5.0)


def keyword_default(v, s):
    def scale(x, *, s=2.0):
        return x * s
    return scale(v)


def test_a_function_with_a_keyword_default_is_not_inlined_as_an_expression() -> None:
    # called by name (an operator not run here), not written as v times the caller's s
    for check in verify(keyword_default, (3.0, 10.0)).checks:
        assert check.ok or check.detail.startswith("Uncheckable"), check.detail


def test_a_module_function_written_in_place_does_not_read_the_callers_variables(tmp_path: Path) -> None:
    source = textwrap.dedent("""
        N = 3.0

        def helper(a):
            return a * N

        def caller(x):
            N = 10.0
            y = helper(x)
            return y + N
        """)
    file = tmp_path / "rd3_inline.py"
    file.write_text(source)
    tree = ast.parse(source)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    low = Lowerer(Module.read(file, "rd3_inline"), inline={"helper": fns["helper"]})
    prog = low.program(fns["caller"])
    got = Evaluator(names={"N": 3.0, "helper": lambda a: a * 3.0}).run(prog, {"x": 1.0})
    assert got == 13.0


# ---------- folding loops into sums and families ----------
def accumulate(total: float, xs: list) -> float:
    for x in xs:
        total += x
    return total


def test_a_sum_into_a_parameter_keeps_its_start() -> None:
    assert _ok_in_two_checks(accumulate, 10.0, [1.0, 2.0])


def captured(i: int) -> list:
    y = i * 2
    return [y + i for i in range(3)]


def captured_by_loop(i: int) -> int:
    y = i * 2
    total = 0
    for i in range(3):
        total += y
    return total


def test_a_value_is_not_put_under_a_binder_of_a_name_it_reads() -> None:
    assert _ok_in_two_checks(captured, 5)
    assert _ok_in_two_checks(captured_by_loop, 5)


def domain_reads_outer(x, i):
    m = i + 2
    return sum(x[i] for i in range(m))


def test_a_value_put_into_a_domain_is_not_captured() -> None:
    prog, _ = program_of(domain_reads_outer)
    assert fold(prog).body != prog.body
    assert _ok(domain_reads_outer, np.arange(5.0), 1)
    ret = lower_source("def h(v):\n    return [w * 2 for w in v]\n").body[-1]
    assert free_vars(ret.value) == ["v"]                                # type: ignore[attr-defined]
    same = lower_source("def h(v):\n    return [v * 2 for v in v]\n").body[-1]
    assert free_vars(same.value) == ["v"]                               # type: ignore[attr-defined]


def turn_read_after(x, n):
    t = 0.0
    s = 0.0
    for i in range(n):
        t = x[i]
        s += t
    return s + t


def index_read_after(x, n):
    s = 0.0
    for i in range(n):
        s += x[i]
    return s + i


def test_a_loop_whose_variables_are_read_after_it_is_not_folded() -> None:
    x = np.array([1.0, 2.0, 3.0])
    assert _ok(turn_read_after, x, 3)
    assert _ok(index_read_after, x, 3)


def normalised(x, n):
    s = 0.0
    for i in range(n):
        s += x[i]
    out = []
    for i in range(n):
        out.append(x[i] / s)
    return out


def turn_reassigned(xs):
    s = 0.0
    for x in xs:
        d = x * 2
        s += d
    d = 3.0
    return s + d


def loop_var_read_after(xs):
    s = 0.0
    for x in xs:
        s += x
    return s + x


def test_a_loop_whose_names_are_reassigned_after_it_folds() -> None:
    x = np.arange(1.0, 4.0)
    assert _ok(normalised, x, 3)
    assert len(fold(program_of(normalised)[0]).body) == 1
    assert _ok(turn_reassigned, [1.0, 2.0])
    assert len(fold(program_of(turn_reassigned)[0]).body) == 1
    assert _ok(loop_var_read_after, [1.0, 2.0])
    prog = program_of(loop_var_read_after)[0]
    assert fold(prog).body is prog.body             # x read after: its last value, not a sum's


def cumulative(x, n):
    out = np.zeros(n)
    s = 0.0
    for k in range(n):
        s += x[k]
        out[k] = s
    return out


def lagged(x, n):
    s = 0.0
    t = 0.0
    for k in range(n):
        s += t
        t = x[k]
    return s


def recurrence(n):
    out = np.zeros(n)
    for k in range(n):
        out[k] = out[k - 1] + 1.0
    return out


def plain_sum(x, n):
    s = 0.0
    for k in range(n):
        t = x[k] * 2
        s += t
    return s


def test_a_loop_whose_turns_read_what_it_changes_is_not_a_sum() -> None:
    x = np.arange(5.0)
    assert _ok(cumulative, x, 5)
    assert _ok(lagged, x, 5)
    assert _ok(recurrence, 5)
    assert _ok(plain_sum, x, 5)
    folded = fold(program_of(plain_sum)[0])
    assert len(folded.body) == 1                        # still one equation


def counted(n):
    out = []
    for k in range(n):
        out.append(len(out) * 2)
    return out


def test_a_family_that_reads_itself_is_not_folded() -> None:
    assert _ok(counted, 4)


def alias_list(xs):
    buf = []
    view = buf
    for v in xs:
        buf.append(v * 2)
    return view


def alias_array(x, n):
    out = np.zeros(n)
    view = out
    for k in range(n):
        out[k] = x[k] * 2
    return view


def test_a_family_held_by_another_variable_is_not_folded() -> None:
    assert _ok(alias_list, [1.0, 2.0])
    assert _ok(alias_array, np.arange(3.0), 3)


# ---------- the evaluator ----------
def test_a_program_that_returns_nothing_gives_none() -> None:
    from headway.core.lower import lower_source
    assert Evaluator().run(lower_source("y = 1 + 2"), {}) is None


# ---------- the printer ----------
def test_a_braced_delimiter_does_not_hide_a_subscript() -> None:
    assert scripted(r"\left\{a\right\}_i")
    assert not scripted(r"\left\{a\right\}")
    assert indexed(r"\left\{a\right\}_{0}", "g") == r"\left\{a\right\}_{0,g}"
    assert not scripted(r"a \leftarrow b")


# ---------- verify.traced ----------
def raising_helper(x):
    if x < 0:
        raise ValueError
    return float(np.sqrt(x))


def catching(x):
    try:
        return raising_helper(x)
    except ValueError:
        return 0.0


def test_traced_does_not_record_a_call_that_raised() -> None:
    with traced(raising_helper) as calls:
        catching(-1.0)
        catching(4.0)
    (recorded,) = calls.values()
    assert [r for _, r in recorded] == [2.0]
