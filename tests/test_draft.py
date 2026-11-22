"""headway init (draft.py), headway lint (lint.py), and headway trace and swap of clean versions (cleanrun.py)."""
from __future__ import annotations

import ast
import importlib
import sys
import textwrap
from pathlib import Path

import pytest

from headway.draft import draft


# ---------- packages to draft from ----------
def _package(tmp_path: Path, name: str, code: str) -> list[Path]:
    (tmp_path / name).mkdir()
    (tmp_path / name / "__init__.py").write_text("")
    (tmp_path / name / "m.py").write_text(textwrap.dedent(code))
    return [tmp_path]


STEPS_ABC = """
    def a(x):
        return x

    def b(x):
        return x

    def c(x):
        return x
    """


def _package_abc(tmp_path: Path, name: str, code: str) -> list[Path]:
    (tmp_path / name).mkdir()
    (tmp_path / name / "__init__.py").write_text("")
    (tmp_path / name / "m.py").write_text(textwrap.dedent(STEPS_ABC) + textwrap.dedent(code))
    return [tmp_path]


STEPS_ABG = """
    def a(x):
        return x

    def b(x):
        return x

    def g(u):
        u[0] = 1
    """


def _package_abg(tmp_path: Path, name: str, code: str) -> list[Path]:
    (tmp_path / name).mkdir()
    (tmp_path / name / "__init__.py").write_text("")
    (tmp_path / name / "m.py").write_text(textwrap.dedent(STEPS_ABG) + textwrap.dedent(code))
    return [tmp_path]


STEPS_UPDATE_FIT_H = """
    def update(a):
        a[0] = 1

    def fit(a):
        return a + 1

    def h(a, b):
        return a + b
    """


def _package_update_fit_h(tmp_path: Path, name: str, code: str) -> list[Path]:
    (tmp_path / name).mkdir()
    (tmp_path / name / "__init__.py").write_text("")
    (tmp_path / name / "m.py").write_text(textwrap.dedent(STEPS_UPDATE_FIT_H) + textwrap.dedent(code))
    return [tmp_path]


def _ops(text: str) -> str:
    """The G.op calls of a draft, one per line (a call wrapped over lines joined)."""
    return text.replace(",\n     ", ", ")


def _op(text: str, key: str) -> str:
    """A G.op call of a draft, on one line (a call wrapped over lines joined)."""
    return next(o for o in text.replace(",\n     ", ", ").splitlines() if o.startswith(f"G.op('{key}'"))


# ---------- drafting: branches decided by literals ----------
def test_branch_skipped_by_one_call_is_drafted_in_another(tmp_path: Path) -> None:
    # a function opened twice, with other literals: the branch the first call skips, the second takes
    search = _package(tmp_path, "rc_twice", """
        def a(x):
            return x

        def b(x):
            return x

        def c(x):
            return x

        def inner(x, k=None):
            if k is None:
                y = a(x)
            else:
                y = b(x)
            return c(y)

        def twice(x):
            p = inner(x)
            q = inner(x, 1)
            return p, q
        """)
    ops = _ops(draft("rc_twice.m:twice", search=search))
    line = next(o for o in ops.splitlines() if o.startswith("G.op('c_2'"))
    assert "inputs=[y_2]" in line


def test_branch_skipped_by_one_call_is_seen_by_a_call_that_does_not_decide_it(tmp_path: Path) -> None:
    search = _package_abc(tmp_path, "r2_undecided", """
    def inner(x, k=None):
        if k is None:
            y = a(x)
        else:
            y = b(x)
        return c(y)

    def twice(x, m):
        p = inner(x)
        q = inner(x, m)
        return p, q
    """)
    text = draft("r2_undecided.m:twice", search=search)
    assert "inputs=[y_2, y_3]" in _op(text, "c_2")


def test_literal_parameter_assigned_again_no_longer_decides_an_if(tmp_path: Path) -> None:
    search = _package(tmp_path, "rc_again", """
        def a(x):
            return x

        def b(x):
            return x

        def c(x):
            return x

        def reassign(x, k=None):
            if k is None:
                k = a(x)
            if k is not None:
                y = b(k)
            else:
                y = c(x)
            return y

        def w(x):
            return reassign(x)
        """)
    text = draft("rc_again.m:w", search=search)
    ops = _ops(text)
    assert "G.op('b'" in ops                            # k is a's result here, not None: b(k) is drafted
    assert "condition='k is not None'" in ops
    assert "`k is not None` is False" not in text


def test_parameter_assigned_in_an_if_that_decides_values(tmp_path: Path) -> None:
    search = _package_abc(tmp_path, "r2_decides", """
    def inner(x, mode="a"):
        if x > 0:
            mode = "b"
        else:
            mode = "c"
        if mode == "a":
            y = a(x)
        else:
            y = b(x)
        return y

    def top(x):
        return inner(x)
    """)
    text = draft("r2_decides.m:top", search=search, clean=True)
    assert "is not drafted" not in text
    assert "y = b(x)" in _op(text, "y")                 # the second if, one step with both branches


def test_parameter_assigned_later_in_a_loop(tmp_path: Path) -> None:
    search = _package_abc(tmp_path, "r2_loop", """
    def inner(x, mode="a"):
        for i in range(3):
            if mode == "a":
                y = a(x)
            else:
                y = b(x)
            mode = "b"
        return y

    def top(x):
        return inner(x)
    """)
    text = draft("r2_loop.m:top", search=search)
    assert "is not drafted" not in text
    assert "condition=\"mode != 'a'\"" in _op(text, "b")


def test_parameter_assigned_in_an_if_test(tmp_path: Path) -> None:
    search = _package_abc(tmp_path, "r2_walrus", """
    def inner(x, mode="a"):
        if (mode := x) == "a":
            y = a(x)
        else:
            y = b(x)
        if mode == "a":
            z = a(y)
        else:
            z = b(y)
        return z

    def top(x):
        return inner(x)
    """)
    text = draft("r2_walrus.m:top", search=search)
    assert "is not drafted" not in text
    assert "G.op('b_2'" in text


def test_a_call_that_unpacks_arguments_decides_no_if_by_a_default(tmp_path: Path) -> None:
    search = _package_abg(tmp_path, "r3_unpack", """
    def inner(x, k=None):
        if k is None:
            y = a(x)
        else:
            y = b(x)
        return y

    def by_name(x, opts):
        return inner(x, **opts)

    def by_position(x, rest):
        return inner(x, *rest)
    """)
    for entry in ("by_name", "by_position"):
        text = draft(f"r3_unpack.m:{entry}", search=search)
        assert "is not drafted" not in text
        assert "G.op('a'" in text and "G.op('b'" in text


# ---------- drafting: values changed in place ----------
def test_value_changed_in_a_group_is_placed_at_its_own_line(tmp_path: Path) -> None:
    search = _package(tmp_path, "rc_lines", """
        def a(x):
            return x

        def b(x):
            return x

        def c(x):
            return x

        def both(u, v):
            u[0] = a(u)
            v[0] = b(v)

        def outer(x, y):
            both(x, y)
            return c(x), c(y)
        """)
    text = draft("rc_lines.m:outer", search=search)
    line = next(s for s in text.splitlines() if s.startswith("y_2 = G.symbol("))
    assert "m.py:13" in line and "v[0] = b(v)" in line


def test_one_variable_given_for_two_parameters_one_changed(tmp_path: Path) -> None:
    # g(x, x), g changing only its first parameter: no KeyError looking up the line that changes it
    search = _package_abc(tmp_path, "r2_twice_given", """
    def g(u, v):
        u[0] = 1
        return a(v)

    def top(x):
        y = g(x, x)
        return a(x)
    """)
    text = draft("r2_twice_given.m:top", search=search)
    assert "inputs=[x_2]" in _op(text, "a_2")          # x as g changed it


def test_a_change_in_a_branch_keeps_the_value_given_and_adds_no_gap(tmp_path: Path) -> None:
    search = _package_abg(tmp_path, "r3_branch", """
    def top(x, k):
        if k:
            g(x)
        return a(x)
    """)
    text = draft("r3_branch.m:top", search=search)
    assert "inputs=[x, x_2]" in _op(text, "a")          # what it was given, or what g made of it
    assert "Made by the code" not in text


def test_a_change_by_one_call_of_a_function_is_not_seen_by_another(tmp_path: Path) -> None:
    search = _package_abg(tmp_path, "r3_calls", """
    def inner(x, k=None):
        if k is None:
            g(x)
        return a(x)

    def top(x, y):
        p = inner(x)
        q = inner(y, 1)
        return p, q
    """)
    text = draft("r3_calls.m:top", search=search)
    assert "inputs=[y]" in _op(text, "a_2")             # the second call skips g: y as it was given


def test_a_step_change_is_kept_past_an_assignment_in_a_branch_not_taken(tmp_path: Path) -> None:
    search = _package_update_fit_h(tmp_path, "r4_skip", """
    def inner(p, mode=None):
        update(p)
        if mode is not None:
            p = p * 2
        return fit(p)

    def entry(x):
        return inner(x)
    """)
    op = _op(draft("r4_skip.m:entry", search=search), "fit")
    assert "inputs=[p]" in op and "'a': p" in op


def test_a_change_on_the_path_taken_is_seen_after_one_in_a_branch_not_taken(tmp_path: Path) -> None:
    search = _package_update_fit_h(tmp_path, "r4_later", """
    def inner(p, mode=None):
        if mode is not None:
            p[0] = 0
        p[1] = 5
        return fit(p)

    def entry(x):
        return inner(x)
    """)
    text = draft("r4_later.m:entry", search=search)
    op = _op(text, "fit")
    assert "inputs=[x]" not in op and "'a': x" not in op
    assert "r4_later.m:inner#p[1] = 5" in text            # the gap: made by the line on the path taken


def test_a_variable_changed_in_place_is_not_the_call_return(tmp_path: Path) -> None:
    search = _package_update_fit_h(tmp_path, "r4_return", """
    def entry(x):
        update(x)
        y = fit(x)
        return y
    """)
    text = draft("r4_return.m:entry", search=search)
    assert "'return'" not in _op(text, "update")
    assert "'return': y" in _op(text, "fit")


# ---------- drafting: results, arguments and literals ----------
def test_returned_tuple_with_a_group_of_two_results(tmp_path: Path) -> None:
    # return g2(x), c(x), g2 opened up and returning two values: still two items, so p, q = ret(x) is followed
    search = _package(tmp_path, "rc_tuple", """
        def a(x):
            return x

        def b(x):
            return x

        def c(x):
            return x

        def g2(x):
            return a(x), b(x)

        def ret(x):
            return g2(x), c(x)

        def top(x):
            p, q = ret(x)
            return b(p), c(q)
        """)
    text = draft("rc_tuple.m:top", search=search, depth=3)
    ops = _ops(text).splitlines()
    assert "inputs=[a, b]" in next(o for o in ops if o.startswith("G.op('b_2'"))
    assert "inputs=[c]" in next(o for o in ops if o.startswith("G.op('c_2'"))
    assert "Made by" not in text


def test_names_pair_arguments_with_parameters_as_python_does(tmp_path: Path) -> None:
    search = _package_update_fit_h(tmp_path, "r4_star", """
    def g(a, *rest, scale=1.0):
        return a * scale

    def entry(x, y, xs):
        z = g(x, y)
        w = h(*xs, y)
        return h(z, w)
    """)
    text = draft("r4_star.m:entry", search=search)
    op = _op(text, "g")
    assert "'a': x" in op and "'scale'" not in op and "inputs=[x, y]" in op
    op = _op(text, "h")
    assert "'a'" not in op and "'b'" not in op and "inputs=[xs, y]" in op


def test_unhashable_literal_default_and_constant(tmp_path: Path) -> None:
    # literal_eval raises TypeError on {[1]: 2}: the value is then not known, not a crash
    search = _package(tmp_path, "rc_literal", """
        from dataclasses import dataclass

        TABLE = {[1]: 2}

        @dataclass
        class Settings:
            odd: dict = {[1]: 2}

        def a(x, t):
            return x

        def run(x, s: Settings):
            return a(x, TABLE), a(x, s.odd)
        """)
    text = draft("rc_literal.m:run", search=search, hints=True)
    assert "TABLE = G.symbol('TABLE'" in text
    assert "odd = G.setting('odd'" in text and "value=None" in text


# ---------- drafting: loops ----------
def test_loop_that_carries_a_value_by_augmented_assignment(tmp_path: Path) -> None:
    # total += a(v) carries total as total = total + a(v) does: one loop step, total's start value an input
    search = _package_abc(tmp_path, "r2_aug", """
    def top(xs):
        total = 0.0
        for v in xs:
            total += a(v)
        return a(total)
    """)
    text = draft("r2_aug.m:top", search=search, clean=True)
    assert "inputs=[total, xs], outputs=[total_2]" in _op(text, "total")
    assert "inputs=[total_2]" in _op(text, "a")


def test_a_value_assigned_and_read_in_one_turn_is_not_carried(tmp_path: Path) -> None:
    search = _package_abg(tmp_path, "r3_turn", """
    def top(xs):
        for v in xs:
            y = a(v)
            b(y)
        return xs
    """)
    text = draft("r3_turn.m:top", search=search, clean=True)
    assert "G.op('loop'" not in text
    assert "per=" in _op(text, "a") or "'v'" in _op(text, "a")


def test_carried_values() -> None:
    from headway.draft import _carried
    fn = ast.parse(textwrap.dedent("""
    def f(xs, c):
        for v in xs:
            total += v
        for v in xs:
            y = v
            z = y
        for v in xs:
            if v:
                w = 1
            u = w
        for v in xs:
            if c:
                q = 1
            else:
                r = q
        for v in xs:
            s = [t * 2 for t in v]
            print(s)
        for v in xs:
            print(m)
            m = v
    """)).body[0]
    assert isinstance(fn, ast.FunctionDef)
    loops = [st for st in fn.body if isinstance(st, ast.For)]
    assert [_carried(st) for st in loops] == [["total"], [], ["w"], ["q"], [], ["m"]]


def test_page_of_a_clean_draft_with_a_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import headway
    from headway import page
    search = _package_abg(tmp_path, "r3_page", """
    def iterate(x, n):
        for i in range(n):
            x = x / 2 + 1
        return x

    def top(x, n):
        y = a(x)
        z = iterate(y, n) + 1                       # a line that calls a function that iterates
        return b(z)
    """)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for name in [n for n in sys.modules if n.startswith("r3_page")]:
        monkeypatch.delitem(sys.modules, name)
    (tmp_path / "r3_page_spec.py").write_text(draft("r3_page.m:top", search=search, clean=True))
    G = importlib.import_module("r3_page_spec").G
    assert isinstance(G, headway.Headway) and (G.draft or {}).get("clean")
    d = page.data(G, headway.check(G))
    assert d["ops"]


# ---------- lint ----------
def test_lint_a_file_that_does_not_parse(tmp_path: Path) -> None:
    from headway.lint import lint
    bad = tmp_path / "bad.py"
    bad.write_text("def f(:\n")
    with pytest.raises(ValueError, match="does not parse"):
        lint(str(bad))


def test_lint_counts_indexing_a_jax_array_as_opaque(tmp_path: Path) -> None:
    from headway.lint import lint
    f = tmp_path / "jx.py"
    f.write_text(textwrap.dedent("""
        import jax

        def first(x: jax.Array):
            return x[0]
        """))
    assert lint(str(f)).opaque >= 1


def test_lint_allows_adding_into_a_number_in_place(tmp_path: Path) -> None:
    from headway.lint import lint
    f = tmp_path / "acc.py"
    f.write_text(textwrap.dedent("""
        import numpy as np

        def total(x: np.ndarray, k: int) -> float:
            s = 0.0
            for i in range(k):
                s += x[i]
            return s

        def scale(x: np.ndarray) -> np.ndarray:
            x *= 2
            return x
        """))
    errors = [e for e in lint(str(f)).findings if e.level == "error"]
    assert [e.function for e in errors] == ["scale"]


# ---------- trace and swap of clean versions (cleanrun.py) ----------
def _traced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, impl: str, clean: str, test: str) -> object:
    from headway import cleanrun
    (tmp_path / "rc_orig").mkdir()
    (tmp_path / "rc_orig" / "__init__.py").write_text("")
    (tmp_path / "rc_orig" / "impl.py").write_text(textwrap.dedent(impl))
    (tmp_path / "rc_orig" / "tidy.py").write_text(textwrap.dedent(clean))
    (tmp_path / "test_rc_orig.py").write_text(textwrap.dedent(test))
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [n for n in sys.modules if n.startswith("rc_orig")]:
        monkeypatch.delitem(sys.modules, name)
    return cleanrun.check("rc_orig.tidy", [str(tmp_path / "test_rc_orig.py"), "-q", "-p", "no:cacheprovider",
                                           "--rootdir", str(tmp_path)], log=lambda s: None)


def _modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, files: dict[str, str]) -> None:
    (tmp_path / name).mkdir()
    (tmp_path / name / "__init__.py").write_text("")
    for file, code in files.items():
        (tmp_path / name / file).write_text(textwrap.dedent(code))
    monkeypatch.syspath_prepend(str(tmp_path))
    for n in [n for n in sys.modules if n.startswith(name)]:
        monkeypatch.delitem(sys.modules, n)


def test_trace_tells_a_none_result_from_a_raise_and_survives_a_failing_clean_version(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    found = _traced(tmp_path, monkeypatch, impl="""
        def fill(x):
            x[0] = 1

        def boom(x):
            return x + 1

        def fails(x):
            raise ValueError(x)
        """, clean="""
        from headway.clean import equals

        @equals("rc_orig.impl.fill")
        def fill(x):
            x[0] = 1

        @equals("rc_orig.impl.boom")
        def boom(x):
            raise ValueError(x)

        @equals("rc_orig.impl.fails")
        def fails(x):
            return x
        """, test="""
        import pytest
        from rc_orig import impl

        def test_it():
            impl.fill([0])
            impl.boom(1)
            with pytest.raises(ValueError):
                impl.fails(1)
        """)
    counts = found.counts                               # type: ignore[attr-defined]
    assert counts["rc_orig.tidy:fill"] == {"agreed": 1}     # returned None: compared, not "raised"
    assert counts["rc_orig.tidy:boom"] == {"the clean version raised ValueError": 1}
    assert counts["rc_orig.tidy:fails"] == {"raised": 1}


def test_trace_does_not_check_a_generator_original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from headway import cleanrun
    (tmp_path / "r2_gen").mkdir()
    (tmp_path / "r2_gen" / "__init__.py").write_text("")
    (tmp_path / "r2_gen" / "impl.py").write_text(textwrap.dedent("""
        def items(n):
            yield None
            yield n
        """))
    (tmp_path / "r2_gen" / "tidy.py").write_text(textwrap.dedent("""
        from headway.clean import equals

        @equals("r2_gen.impl.items")
        def items(n):
            return [None, n]
        """))
    (tmp_path / "test_r2_gen.py").write_text(textwrap.dedent("""
        from r2_gen import impl

        def test_it():
            assert list(impl.items(1)) == [None, 1]
        """))
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [n for n in sys.modules if n.startswith("r2_gen")]:
        monkeypatch.delitem(sys.modules, name)
    found = cleanrun.check("r2_gen.tidy", [str(tmp_path / "test_r2_gen.py"), "-q", "-p", "no:cacheprovider",
                                           "--rootdir", str(tmp_path)], log=lambda s: None)
    assert "raised" not in found.counts["r2_gen.tidy:items"]     # its first yield is not taken for a raise
    assert any("generator" in u for u in found.unchecked)


def test_trace_and_swap_with_an_original_that_is_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from headway import cleanrun
    _modules(tmp_path, monkeypatch, "r3_lost", {"tidy.py": """
        from headway.clean import equals

        @equals("r3_lost.impl.gone")
        def gone(n):
            return n
        """})
    (tmp_path / "test_r3_lost.py").write_text("def test_it():\n    assert True\n")
    args = [str(tmp_path / "test_r3_lost.py"), "-q", "-p", "no:cacheprovider", "--rootdir", str(tmp_path)]
    found = cleanrun.check("r3_lost.tidy", args, log=lambda s: None)
    said = [u for u in found.unchecked if "r3_lost.tidy:gone" in u and "math" not in u]
    assert len(said) == 1 and "is not found" in said[0]
    logged: list[str] = []
    assert cleanrun.swap("r3_lost.tidy", args, log=logged.append) == 0
    assert any("not swapped" in line for line in logged)


def test_trace_and_swap_with_an_original_that_is_not_a_python_function(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    from headway import cleanrun
    _modules(tmp_path, monkeypatch, "r4_builtin", {"tidy.py": """
        from headway.clean import equals

        @equals("math.sqrt")
        def root(x):
            return x ** 0.5
        """})
    (tmp_path / "test_r4_builtin.py").write_text("def test_it():\n    assert True\n")
    args = [str(tmp_path / "test_r4_builtin.py"), "-q", "-p", "no:cacheprovider", "--rootdir", str(tmp_path)]
    found = cleanrun.check("r4_builtin.tidy", args, log=lambda s: None)
    said = [u for u in found.unchecked if u.startswith("r4_builtin.tidy:root")]
    assert any("is not a Python function" in u for u in said)
    assert not any("is not found" in u for u in said)
    logged: list[str] = []
    assert cleanrun.swap("r4_builtin.tidy", args, log=logged.append) == 0
    assert any("not a Python function: not swapped" in line for line in logged)
