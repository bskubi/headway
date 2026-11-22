"""Resolving code references (resolve.py), tracing a spec's calls (headway.trace), the page's data, equals of clean
versions, what check and the spec read, and the cache of the type checker's kinds."""
from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from headway import Headway, check, diff, page, typeinfo
from headway.check import Report
from headway.clean import Relation, _compare
from headway.resolve import RefError, package_root, resolve
from headway.spec import Decl, declared_file
from headway.trace import trace


# ---------- modules to resolve and trace ----------
SETTINGS_MOD = '''\
"""A module.

N_ITER: how many iterations
"""
from dataclasses import dataclass

N_ITER = {"a": 1,
          "b": 2}


@dataclass
class Settings:
    """Settings.

    pcs: the number of components
    """
    pcs: int = 50


class Model:
    def fit(self, x):
        return x * 2


def run(x):
    return Model().fit(x)


def unused(x):
    return x
'''


@pytest.fixture
def settings_pkg(tmp_path, monkeypatch):
    (tmp_path / "refpkg1").mkdir()
    (tmp_path / "refpkg1" / "__init__.py").write_text("")
    (tmp_path / "refpkg1" / "mod.py").write_text(SETTINGS_MOD)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    importlib.invalidate_caches()
    return importlib.import_module("refpkg1.mod")


METHODS_MOD = '''\
import numpy as np


class Base:
    def fit(self, x):
        return x * 2

    @staticmethod
    def scale(x):
        return x * 3

    @classmethod
    def make(cls, x):
        return x + 1


class Sub(Base):
    pass


class Outer:
    class Settings:
        """Settings.

The rest of this docstring is flush left."""
        pcs: int = 50


try:
    import numpy as _np
    LIMIT = 3
except ImportError:
    LIMIT = 4
if True:
    DEPTH = 2
A, B = 1, 2
'''


@pytest.fixture
def methods_pkg(tmp_path, monkeypatch):
    (tmp_path / "refpkg2").mkdir()
    (tmp_path / "refpkg2" / "__init__.py").write_text("")
    (tmp_path / "refpkg2" / "mod.py").write_text(METHODS_MOD)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for name in [n for n in sys.modules if n.split(".")[0] == "refpkg2"]:   # a fresh import from this tmp_path
        monkeypatch.delitem(sys.modules, name)
    importlib.invalidate_caches()
    return importlib.import_module("refpkg2.mod")


DECORATED_MOD = '''\
import collections
import enum
import functools


@functools.cache
def cached(n):
    return n + 1


Color = enum.Enum("Color", "RED GREEN")
Point = collections.namedtuple("Point", "x y")


class Base:
    @property
    def size(self):
        return 3
'''


def _fresh(tmp_path, monkeypatch, top: str) -> None:
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for name in [n for n in sys.modules if n.split(".")[0] == top]:      # a fresh import from this tmp_path
        monkeypatch.delitem(sys.modules, name)
    importlib.invalidate_caches()


@pytest.fixture
def decorated_pkg(tmp_path, monkeypatch):
    (tmp_path / "refpkg3").mkdir()
    (tmp_path / "refpkg3" / "__init__.py").write_text("")
    (tmp_path / "refpkg3" / "mod.py").write_text(DECORATED_MOD)
    _fresh(tmp_path, monkeypatch, "refpkg3")
    return importlib.import_module("refpkg3.mod")


CONFIG_MOD = '''\
import functools

import numpy as np

from refpkg4.other import Fitter


class Box(dict):
    def __getattr__(self, k):              # a config object: a missing key is a new, empty Box
        return self.setdefault(k, Box())


CONFIG = Box(alpha=0.5)


class Model:
    @functools.cached_property
    def table(self):
        return np.zeros((2, 3))


def fit(x: "Float[np.ndarray, 'a b']") -> None:
    return None
'''


FITTER_MOD = '''\
class Fitter:
    def fit(self, x):
        return x
'''


@pytest.fixture
def config_pkg(tmp_path, monkeypatch):
    (tmp_path / "refpkg4").mkdir()
    (tmp_path / "refpkg4" / "__init__.py").write_text("")
    (tmp_path / "refpkg4" / "mod.py").write_text("from jaxtyping import Float  # noqa: F401\n" + CONFIG_MOD)
    (tmp_path / "refpkg4" / "other.py").write_text(FITTER_MOD)
    _fresh(tmp_path, monkeypatch, "refpkg4")
    return importlib.import_module("refpkg4.mod")


# ---------- resolving references ----------
def test_fields_and_constants_not_docstring_lines(settings_pkg) -> None:
    field = resolve("refpkg1.mod:Settings.pcs")
    assert field.source == "    pcs: int = 50\n"
    const = resolve("refpkg1.mod:N_ITER")
    assert const.source.startswith("N_ITER = {") and const.end == const.start + 1
    with pytest.raises(RefError):
        resolve("refpkg1.mod:Settings.nope")


def test_fields_and_constants_anywhere_in_a_module(methods_pkg) -> None:
    assert resolve("refpkg2.mod:Outer.Settings.pcs").source == "        pcs: int = 50\n"
    assert resolve("refpkg2.mod:LIMIT").source == "    LIMIT = 3\n"
    assert resolve("refpkg2.mod:DEPTH").source == "    DEPTH = 2\n"
    assert resolve("refpkg2.mod:B").source == "A, B = 1, 2\n"


def test_builtin_module_is_a_ref_error() -> None:
    with pytest.raises(RefError, match="no Python source"):
        resolve("math:sqrt")


def test_package_root(methods_pkg, tmp_path) -> None:
    assert package_root(methods_pkg.__file__) == tmp_path / "refpkg2"
    assert package_root(tmp_path / "loose.py") is None


def test_resolve_a_function_behind_a_decorator_object(decorated_pkg) -> None:
    loc = resolve("refpkg3.mod:cached")
    assert loc.source.startswith("@functools.cache\ndef cached(n):")


@pytest.mark.parametrize("ref", ["refpkg3.mod:Color.RED", "refpkg3.mod:Point.x", "numpy:ndarray.T"])
def test_resolve_a_member_of_a_class_without_source_is_a_ref_error(decorated_pkg, ref) -> None:
    with pytest.raises(RefError, match="no source"):
        resolve(ref)


def test_resolve_an_object_that_makes_up_every_attribute(config_pkg) -> None:
    loc = resolve("refpkg4.mod:CONFIG")
    assert loc.source == "CONFIG = Box(alpha=0.5)\n"


def test_resolve_a_cached_property(config_pkg) -> None:
    loc = resolve("refpkg4.mod:Model.table")
    assert "def table(self):" in loc.source


def test_check_reports_a_missing_file_by_a_dotdot_path(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RefError, match="no file"):
        resolve("../R/gone.R:g", [tmp_path])
    with pytest.raises(RefError):
        resolve(":f", [tmp_path])                       # an empty module name


# ---------- tracing a spec's calls (headway.trace) ----------
def _traced() -> Headway:
    G = Headway("t")
    G.index("i", "i", "i")
    G.symbol("x", "x", "x", dims=["i"], role="input")
    G.symbol("y", "y", "y", dims=["i"], role="output")
    G.op("fit", "fit", inputs=["x"], outputs=["y"], code="refpkg1.mod:Model.fit", bind={"x": "x", "return": "y"})
    return G


def test_trace_wraps_methods(settings_pkg) -> None:
    G = _traced()
    with trace(G) as t:
        settings_pkg.run(np.ones(3))
    assert not t.errors, t.text()
    assert t.calls["fit"] == 1
    assert not hasattr(settings_pkg.Model.fit, "__wrapped__")            # restored afterwards


def test_trace_summary_counts_only_called_ops(settings_pkg) -> None:
    G = _traced()
    G.symbol("q", "q", "q", dims=["i"], role="input")
    G.symbol("z", "z", "z", dims=["i"], role="output")
    G.op("run", "run", inputs=["q"], outputs=["z"], code="refpkg1.mod:run", bind={"x": "q", "return": "z"})
    G.op("idle", "idle", inputs=["q"], code="refpkg1.mod:unused", bind={"x": "q"})
    with trace(G) as t:
        settings_pkg.run(np.ones(3))
    assert t.errors == ["op idle: bound, but its function was never called"]
    assert t.text().splitlines()[-1] == "2 calls checked over 2 ops"


def _three() -> Headway:
    G = Headway("t")
    G.index("i", "i", "i")
    G.symbol("x", "x", "x", dims=["i"], role="input")
    for k in ("y", "z", "w"):
        G.symbol(k, k, k, dims=["i"], role="output")
    G.op("fit", "fit", inputs=["x"], outputs=["y"], code="refpkg2.mod:Sub.fit", bind={"x": "x", "return": "y"})
    G.op("scale", "scale", inputs=["x"], outputs=["z"], code="refpkg2.mod:Base.scale",
         bind={"x": "x", "return": "z"})
    G.op("make", "make", inputs=["x"], outputs=["w"], code="refpkg2.mod:Sub.make", bind={"x": "x", "return": "w"})
    return G


def test_trace_wraps_inherited_static_and_class_methods(methods_pkg) -> None:
    with trace(_three()) as t:
        assert methods_pkg.Sub().fit(np.ones(3)).tolist() == [2, 2, 2]
        assert methods_pkg.Base.scale(np.ones(3)).tolist() == [3, 3, 3]
        assert methods_pkg.Sub.make(np.ones(3)).tolist() == [2, 2, 2]
    assert not t.errors, t.text()
    assert t.calls == {"fit": 1, "scale": 1, "make": 1}
    assert isinstance(vars(methods_pkg.Base)["scale"], staticmethod) and isinstance(vars(methods_pkg.Base)["make"], classmethod)
    assert not hasattr(methods_pkg.Base.fit, "__wrapped__") and not hasattr(methods_pkg.Base.scale, "__wrapped__")


def test_trace_a_property(decorated_pkg) -> None:
    G = Headway("t")
    G.index("i", "i", "item")
    G.symbol("out", "o", "out", role="output")
    G.op("size", "size", outputs=["out"], code="refpkg3.mod:Base.size", bind={"return": "out"})
    with trace(G) as t:
        assert decorated_pkg.Base().size == 3                      # still a property while traced
    assert t.errors == [] and t.calls == {"size": 1}
    assert isinstance(vars(decorated_pkg.Base)["size"], property) and decorated_pkg.Base().size == 3     # restored


def test_trace_keeps_a_cached_property_one(config_pkg) -> None:
    G = Headway("t")
    G.index("a", "a", "a")
    G.index("b", "b", "b")
    G.symbol("w", "w", "w", dims=["a", "b"], role="output")
    G.op("t", "t", outputs=["w"], code="refpkg4.mod:Model.table", bind={"return": "w"})
    original = vars(config_pkg.Model)["table"]
    with trace(G) as t:
        m = config_pkg.Model()
        assert m.table.shape == (2, 3)                  # a value, not a bound method
        assert m.table is m.table                       # cached
    assert vars(config_pkg.Model)["table"] is original
    assert not t.errors and t.calls == {"t": 1}, t.text()


def test_trace_of_a_bound_step_without_python_code_is_an_error() -> None:
    G = Headway("t")
    G.symbol("x", "x", "x", role="input")
    G.op("a", "a", inputs=["x"], bind={"x": "x"})
    G.op("b", "b", inputs=["x"], code="R/score.R:score", bind={"x": "x"})
    with trace(G) as t:
        pass
    assert t.errors[:2] == ["op a: bound, but its first code reference names no Python function",
                            "op b: bound, but its first code reference names no Python function"]


def test_trace_reports_a_bound_file_reference_by_a_dotdot_path() -> None:
    G = Headway("t")
    G.symbol("x", "x", "x", role="input")
    G.symbol("y", "y", "y", role="output")
    G.op("f", "f", inputs=["x"], outputs=["y"], code="../R/score.R:f", bind={"x": "x"})
    with trace(G) as t:
        pass
    assert any("names no Python function" in e for e in t.errors)


# ---------- equals (headway.clean) ----------
def test_equals_with_a_missing_or_extra_output_fails() -> None:
    assert not _compare(Relation("equals", lambda x: (x, x + 1)), {"x": 1.0}, (1.0,)).passed
    assert not _compare(Relation("equals", lambda x: (x,)), {"x": 1.0}, (1.0, 99.0)).passed
    assert _compare(Relation("equals", lambda x: (x, x + 1)), {"x": 1.0}, (1.0, 2.0)).passed


def test_equals_with_a_nan_on_one_side_reports_it() -> None:
    r = _compare(Relation("equals", lambda x: np.array([1.0, x])), {"x": 2.0}, np.array([1.0, np.nan]))
    assert not r.passed and r.measure == float("inf")
    assert _compare(Relation("equals", lambda x: np.array([1.0, x])), {"x": 2.0}, np.array([1.0, 2.0])).passed


def test_equals_nan_in_the_same_places_and_opposite_infinities() -> None:
    rel = Relation("equals", lambda x: x)
    assert _compare(rel, {"x": np.array([np.nan, 1.0])}, np.array([np.nan, 1.0])).passed
    r = _compare(rel, {"x": np.array([np.inf, 1.0])}, np.array([-np.inf, 1.0]))
    assert not r.passed and r.measure == float("inf")
    assert not _compare(rel, {"x": np.array([np.nan, 1.0])}, np.array([0.0, 1.0])).passed


def test_equals_measures_the_imaginary_part() -> None:
    rel = Relation("equals", lambda x: x * np.array([1 + 2j]))
    result = _compare(rel, {"x": 1.0}, np.array([1 + 1j]))
    assert not result.passed and result.measure == pytest.approx(1.0)
    assert _compare(rel, {"x": 1.0}, np.array([1 + 2j])).passed


# ---------- the page's data ----------
def test_page_data_cannot_end_or_unbalance_its_script(tmp_path) -> None:
    d = {"title": "My <T>", "ops": [], "symbols": [], "groups": [],
         "x": 'code: .replace("<!--TITLE-->", t) <!-- <script> </script>'}
    h = page.html(d)
    script = h[h.index("let D = "):].split("\n", 1)[0]
    assert "<" not in script                            # no "<!--", "<script" or "</script>" inside it
    assert "<title>My &lt;T&gt;</title>" in h
    f = tmp_path / "p.html"
    f.write_text(h)
    assert diff.load(f)["x"] == d["x"]                  # read back as written, the title marker untouched


def test_page_data_holding_the_library_marker() -> None:
    d = {"title": "T", "ops": [], "symbols": [], "groups": [], "x": "/*HEADWAY_LIBRARY*/null"}
    h = page.html(d)
    script = h[h.index("let D = "):].split("\n", 1)[0]
    assert "/*HEADWAY_LIBRARY*/null" in script
    assert "const LIB = null;" in h


def test_bind_of_a_result_outranks_the_order_of_outputs() -> None:
    G = Headway("t")
    G.symbol("x", "x", "x", role="input")
    G.symbol("b", "b", "b", role="output")
    G.symbol("d", "d", "d", role="output")
    o = G.op("s", "s", inputs=["x"], outputs=["b", "d"], bind={"return.0": "d", "return.1": "b"})
    names = page._symbol_names(G, o)
    assert names["return.0"].tex == "d" and names["return.1"].tex == "b"
    assert page.relation_views(G, o, Report()) == []


# ---------- check, spec and type information ----------
def test_shapes_stated_in_a_namespace_package(tmp_path, monkeypatch) -> None:
    (tmp_path / "rev3bns").mkdir()                       # no __init__.py
    (tmp_path / "rev3bns" / "steps.py").write_text(
        "import numpy as np\nfrom jaxtyping import Float\n\n\n"
        "def means(x: Float[np.ndarray, 'rows cols']) -> Float[np.ndarray, 'cols']:\n    return x.mean(0)\n")
    _fresh(tmp_path, monkeypatch, "rev3bns")
    G = Headway("t", hints=True)
    G.index("c", "c", "cell")
    G.symbol("x", "x", "x", dims=("c",), role="input")
    G.symbol("m", "m", "m", dims=("c",), role="output")
    G.op("means", "means", inputs=["x"], outputs=["m"], code="rev3bns.steps:means", bind={"x": "x", "return": "m"})
    assert any("x is annotated with 2 axes (rows cols)" in w for w in check(G).warnings)


def test_shapes_stated_does_not_take_a_module_function_for_a_method(config_pkg) -> None:
    pytest.importorskip("jaxtyping")
    G = Headway("t", hints=True)
    G.index("i", "i", "i")
    G.symbol("x", "x", "x", dims=["i"], role="input")
    G.symbol("y", "y", "y", role="output")
    G.op("f", "f", inputs=["x"], outputs=["y"], code="refpkg4.mod:Fitter.fit", bind={"x": "x"})
    report = check(G)
    assert not any("annotated" in w for w in report.warnings), report.warnings


def test_declared_file_tie_goes_to_the_first_declared() -> None:
    G = Headway("t")
    for n, f in enumerate(["/b.py", "/a.py", "/a.py", "/b.py"]):
        G.declared[f"symbol:{n}"] = Decl(f, "m", "<module>", 1, 0, 1, 1)
    assert declared_file(G) == "/b.py"
    assert declared_file(Headway("empty")) is None


def test_kinds_are_made_again_when_their_version_changes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    (tmp_path / "kpkg").mkdir()
    (tmp_path / "kpkg" / "__init__.py").write_text("x = 1\n")
    made = []

    def infer(folder, files):
        made.append(folder)
        return {}, ""
    monkeypatch.setattr(typeinfo, "_infer", infer)
    typeinfo.Types.of_package(tmp_path / "kpkg")
    typeinfo.Types.of_package(tmp_path / "kpkg")
    assert len(made) == 1                               # the second from the cache
    monkeypatch.setattr(typeinfo, "KINDS_VERSION", typeinfo.KINDS_VERSION + 1)
    typeinfo.Types.of_package(tmp_path / "kpkg")
    assert len(made) == 2
