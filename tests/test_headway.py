"""headway's checks catch what they should: a renamed function, an equation that no longer matches the code, an
array with the wrong dimensions."""
import ast
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import headway
from headway.resolve import RefError, resolve

import toy


def spec(code: str = "toy:center", bind_return: str = "xc") -> headway.Headway:
    G = headway.Headway("toy")
    n, p = G.index("n", "n", "row"), G.index("p", "p", "column")
    x = G.symbol("x", "x_{np}", "data", dims=(n, p), role="input")
    xc = G.symbol("xc", r"\tilde x_{np}", "centered", dims=(n, p))
    gm = G.symbol("gm", "C_{pq}", "gram", dims=(p, p), role="output")
    G.op("center", "center", r"\tilde x_{np} = x_{np} - \bar x_p", inputs=[x], outputs=[xc], code=code,
         group="a", bind={"x": x, "return": bind_return})
    G.op("gram", "gram", r"C = \tilde x^\top \tilde x", inputs=[xc], outputs=[gm], code="toy:gram", group="a/b",
         bind={"x": xc, "return": gm})
    return G


def test_good_spec_passes() -> None:
    assert headway.check(spec()).ok


def test_renamed_function_fails() -> None:
    report = headway.check(spec(code="toy:centre"))
    assert not report.ok and "centre" in report.errors[0]


def test_anchors() -> None:
    loc = resolve("toy:center#step 1")                     # up to the next comment at the same indentation
    assert loc.source.splitlines() == ["    # step 1: the means", "    mu = x.mean(0)"] and loc.end == loc.start + 1
    loc = resolve("toy:center#step 1..return")
    assert loc.source.splitlines()[-1].strip() == "return x - mu"
    assert resolve("toy:SCALE").source.strip() == "SCALE = 2.0"
    assert resolve("toy:SIZES").source.count("\n") == 4                # a constant over lines: the whole statement
    assert resolve('toy:SIZES#"b"').source.strip() == '"b": 20,'      # an anchor within it
    assert "return self.x.sum()" in resolve("toy:Box.total").source    # a property: its getter
    with pytest.raises(RefError):
        resolve("toy:center#no such text")


def test_wrong_equation_fails() -> None:
    G = spec()

    @G.verify("center")
    def centers() -> None:
        x = np.arange(6.0).reshape(3, 2)
        assert np.allclose(toy.center(x), x - x.mean(1, keepdims=True))     # rows, not columns: wrong

    report = headway.check(G)
    assert not report.ok and report.verified["center"][0] is False


def test_shapes() -> None:
    with headway.trace(spec()) as t:
        toy.gram(toy.center(np.ones((5, 3))))
    t.assert_ok()
    assert t.sizes == {"n": {5}, "p": {3}}
    # declared n x p, but gram returns p x p
    with headway.trace(spec(bind_return="gm")) as t:
        toy.center(np.ones((5, 3)))
    assert any("center" in e for e in t.errors)


def test_build(tmp_path) -> None:
    from headway import page
    G = spec()
    d = page.data(G, headway.check(G), collapse=["a/b"])
    assert [g["path"] for g in d["groups"]] == ["a", "a/b"] and d["groups"][1]["collapsed"]
    assert "toy" in page.html(d)


LIVE = '''from headway import Headway
G = Headway("live")
n = G.index("n", "n", "row")
x = G.symbol("x", "x_n", "data", dims=(n,), role="input")
y = G.symbol("y", "y_n", "doubled", dims=(n,), role="output")
G.op("double", "double", r"y_n = TEX", inputs=[x], outputs=[y])
'''


def test_serve(tmp_path) -> None:
    import json
    import threading
    import time
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    from headway.serve import Server
    spec = tmp_path / "livespec.py"
    spec.write_text(LIVE.replace("TEX", "2 x_n"))
    server = Server("livespec", cwd=str(tmp_path))
    assert server.watched() == {spec}
    server.rebuild()
    op = server.data["ops"][0]
    assert server.version == 1 and op["decl"]["line"] == 6 and op["decl"]["file"] == str(spec)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.handler())
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    threading.Thread(target=server.watch, daemon=True).start()
    get = lambda path: urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_port}{path}").read().decode()

    def settle(version: int) -> dict:
        for _ in range(100):
            state = json.loads(get("/state.json"))
            if state["version"] >= version:
                return state
            time.sleep(0.1)
        raise AssertionError("no rebuild")
    try:
        assert '"live": true' in get("/")
        # this server process's id, in the page and in the stream: a page that finds another one reloads
        from headway.serve import BOOT
        assert f'"boot": "{BOOT}"' in get("/") and json.loads(get("/state.json"))["boot"] == BOOT
        # a POST changes files: only from headway's own pages (its header; no other site's origin)
        def post(headers: dict) -> int:
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_port}/redraft", data=b"{}", method="POST",
                                         headers=headers)
            try:
                return urllib.request.urlopen(req).status
            except urllib.error.HTTPError as e:
                return e.code
        assert post({}) == 403 and post({"X-Headway": "1", "Origin": "https://example.com"}) == 403
        assert post({"X-Headway": "1"}) == 400                       # allowed, but this spec was not drafted
        time.sleep(0.1)
        spec.write_text(LIVE.replace("TEX", "x_n + x_n"))               # an edit is picked up
        assert settle(2)["ok"] and json.loads(get("/data.json"))["ops"][0]["tex"] == "y_n = x_n + x_n"
        spec.write_text(LIVE.replace("TEX", "x_n") + "G.op(\n")          # a broken spec keeps the last good graph
        state = settle(3)
        assert not state["ok"] and "SyntaxError" in state["error"]
        assert json.loads(get("/data.json"))["ops"][0]["tex"] == "y_n = x_n + x_n"
        # an edit of the same length within the same second (stale bytecode would hide it)
        spec.write_text(LIVE.replace("TEX", "x_n + x_m"))
        assert settle(4)["ok"] and json.loads(get("/data.json"))["ops"][0]["tex"] == "y_n = x_n + x_m"
    finally:
        server.close()
        httpd.shutdown()


def test_declarations_and_editor(monkeypatch) -> None:
    from headway import page
    G = spec()                                  # declared in this file, by the helper `spec`
    d = page.data(G, headway.check(G))
    op = next(o for o in d["ops"] if o["key"] == "center")
    lines = open(__file__).read().splitlines()
    line, col = op["decl"]["fields"]["tex"]
    assert op["decl"]["function"] == "spec" and lines[line - 1][col - 1:].startswith('r"\\tilde x_{np}')
    assert set(op["decl"]["fields"]) >= {"key", "label", "tex", "inputs", "outputs", "code", "group", "bind"}
    assert d["editor"]["url"].startswith("vscode://")
    monkeypatch.setenv("HEADWAY_EDITOR", "cursor")
    assert page.editor()["name"] == "Cursor"
    assert page.editor("idea://open?file={file}&line={line}")["url"].startswith("idea://")
    with pytest.raises(AssertionError):
        page.editor("emacs")


def test_where() -> None:
    from headway import page
    G = spec()
    G.op("scale", "scale", r"y = \tilde x_{np} / s_p", inputs=["xc"], outputs=[],
         where={"s_p": "column s.d.", r"\tilde x_{np}": "listed already"})
    d = page.data(G, report := headway.check(G))
    op = next(o for o in d["ops"] if o["key"] == "scale")
    assert op["where"] == [("s_p", "column s.d."), (r"\tilde x_{np}", "listed already")] and "where" in op["decl"]["fields"]
    assert any("where-entry" in w and "xc" in w for w in report.warnings) and report.ok
    G.op("bad", "bad", "y = C", inputs=["xc"], where={"C_{pq}": "the gram"})     # a symbol the step does not list
    assert any("where-entry" in e and "gm" in e for e in headway.check(G).errors)


def test_constants() -> None:
    G = spec()
    tiny = G.constant("tiny", r"\epsilon", "floor", value=1e-8, code="toy:SCALE")
    G.op("floor", "floor", r"y = \max(C, \epsilon)", inputs=["gm", tiny], outputs=[], code="toy:gram")
    assert G.symbols["tiny"].role == "constant" and headway.check(G).ok
    G.constant("loose", "c", "uncited", value=2)
    G.op("make", "make", "c = 2", inputs=["x"], outputs=["loose"], code="toy:gram")
    report = headway.check(G)
    assert any("produces loose, declared as constant" in e for e in report.errors)
    assert any("constant loose: no code reference" in w for w in report.warnings)


def test_conditions() -> None:
    """A step done only under a condition: an expression of symbols, written as math with its truth values
    explicit, its symbols used (they decide whether the step is done)."""
    from headway import Headway
    from headway.page import condition_of
    G = Headway("cond", "")
    G.symbol("x", "x", "data", role="input")
    G.setting("correct", r"\kappa", "correct", "remove plate effects", value=True, code="toy:SCALE")
    G.symbol("y", "y", "result", role="output")
    G.op("fix", "fix", inputs=["x"], outputs=["y"], code="toy:gram", condition="correct")
    report = headway.check(G)
    assert report.ok and not [w for w in report.warnings if "correct" in w], report.text()
    assert condition_of(G, G.ops["fix"]) == {"condition_tex": r"\kappa = \text{True}", "condition_on": ["correct"]}
    G.ops["fix"].condition = "not correct and mode == 'exact'"
    assert condition_of(G, G.ops["fix"])["condition_tex"] == r"\kappa = \text{False} \land \mathit{mode} = \text{“exact”}"
    assert any("names mode, not symbols" in w for w in headway.check(G).warnings)
    G.ops["fix"].condition = "correct and"
    assert any("is not an expression" in e for e in headway.check(G).errors)


def test_relations(tmp_path, monkeypatch) -> None:
    """headway.clean: a clean function declares what it computes (an objective it maximizes, an equation it
    solves, a reference it equals); the declaration changes nothing about the function, the checks test it on a
    call, the draft reads it from the code, and the page writes it as math, with what a run's checks found."""
    import json

    from headway import clean
    from headway.draft import draft

    def objective(x, y):
        return -((x - 3.0) ** 2).sum() - ((y - x.sum()) ** 2)

    def total(x):
        return x.sum() - 1.0

    @clean.maximizes(objective, over=("y",))
    def fit(x):
        return x.sum()

    @clean.maximizes(objective, over=("x", "y"), subject_to=(total,))
    def fit_both(n):
        x = np.full(n, 1.0 / n)                          # the constrained optimum: x sums to 1, y to x's sum
        return x, 1.0

    @clean.equals(lambda x: x.sum() + 1e-12, rtol=1e-9)
    def plain(x):
        return x.sum()
    x = np.array([1.0, 2.0])
    assert fit(x) == 3.0 and [r.kind for r in clean.relations(fit)] == ["maximizes"]
    assert clean.check(fit, clean.relations(fit)[0], {"x": x}, fit(x)).passed           # y = sum(x): its optimum
    assert not clean.check(fit, clean.relations(fit)[0], {"x": x}, 2.0).passed           # not there: it says so
    assert clean.check(fit_both, clean.relations(fit_both)[0], {"n": 3}, fit_both(3)).passed  # along the constraint
    assert not clean.check(fit_both, clean.relations(fit_both)[0], {"n": 3}, (np.array([1.0, 0, 0]), 1.0)).passed
    assert clean.check(plain, clean.relations(plain)[0], {"x": x}, plain(x)).passed
    assert not clean.check(plain, clean.relations(plain)[0], {"x": x}, 4.0).passed
    # drafted from the code, and written as math with the evidence beside the code
    pkg = tmp_path / "cleanpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "steps.py").write_text(
        "import numpy as np\nfrom headway.clean import maximizes\n\n\n"
        "def objective(x, y):\n    \"\"\"How near y is to the sum of x.\"\"\"\n    return -((y - x.sum()) ** 2)\n\n\n"
        "@maximizes(objective, over=(\"y\",))\ndef fit(x):\n    for _ in range(3):\n        x = x * 1.0\n    return x.sum()\n\n\n"
        "def run(x):\n    y = fit(x)\n    return fit(y)\n")
    (pkg / ".headway-checks.json").write_text(json.dumps({"run": "pytest", "when": "today", "checks": {
        "cleanpkg.steps:fit": [{"kind": "maximizes", "target": "cleanpkg.steps:objective", "calls": 2, "passed": 2,
                                "worst": 0.0, "detail": "fine"}]}}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    ns: dict = {}
    exec(draft("cleanpkg.steps:run", clean=True), ns)
    G = ns["G"]
    assert G.ops["fit"].relations == [{"kind": "maximizes", "target": "cleanpkg.steps:objective", "over": ["y"]}]
    report = headway.check(G)
    assert report.ok, report.text()
    view = headway.page.relation_views(G, G.ops["fit"], report)[0]
    assert view["tex"].startswith(r"y = \operatorname*{arg\,max}_{y} \operatorname{objective}") and \
        view["about"] == "How near y is to the sum of x." and view["evidence"]["passed"] == 2
    G.ops["fit"].relations[0]["kind"] = "optimizes"
    assert any("relation kind 'optimizes' is not one of" in e for e in headway.check(G).errors)


def test_offline(monkeypatch) -> None:
    from headway import page
    fetched = []

    def fake(url: str) -> bytes:                # no network: a stand-in for each CDN file
        fetched.append(url)
        if url.endswith(".css"):
            return b'@font-face{font-family:K;src:url(fonts/K.woff2) format("woff2"),url(fonts/K.woff) format("woff")}'
        return b"var lib = /<!--/;"
    monkeypatch.setattr(page, "_fetch", fake)
    G = spec()
    page = page.html(page.data(G, headway.check(G)), offline=True)
    assert "https://" not in page.split("<style>\n:root")[0] and "url(fonts/" not in page
    assert '<script src="data:text/javascript;base64,' in page and 'media="(prefers-color-scheme: dark)"' in page
    assert any(u.endswith("/fonts/K.woff2") for u in fetched) and not any(u.endswith(".woff") for u in fetched)


# ---------- versions: `was`, lineages, diffs (examples/cellscore: one method developed over four versions) ----------
EXAMPLE = Path(__file__).parents[1] / "examples" / "cellscore"


def test_was_checked() -> None:
    G = spec()
    G.op("again", "again", "y = x", inputs=["xc"], outputs=[], code="toy:gram", was="center")    # still declared
    assert any("was='center'" in e for e in headway.check(G).errors)


def built(version: str) -> dict:
    """An example version, built as `headway build` would (each vN/ is the package as it was at that point)."""
    from headway import page
    for name in [m for m in sys.modules if m == "cellscore" or m.startswith("cellscore.")]:
        del sys.modules[name]
    sys.path.insert(0, str(EXAMPLE / version))
    try:
        G = importlib.import_module("cellscore.graphspec").G
        report = headway.check(G)
        assert report.ok and not report.warnings, report.text()
        return page.data(G, report) | {"name": version}
    finally:
        sys.path.remove(str(EXAMPLE / version))
        for name in [m for m in sys.modules if m == "cellscore" or m.startswith("cellscore.")]:
            del sys.modules[name]


@pytest.fixture(scope="module")
def versions() -> list[dict]:
    return [built(v) for v in ("v1", "v2", "v3", "v4")]


def test_lineages(versions) -> None:
    from headway.diff import lineages
    lin = lineages(versions)
    # named by the latest key; a rename (score -> weighted_score) and a split (normalize -> two steps) are matched
    assert lin[(0, "op", "score")] == lin[(3, "op", "weighted_score")] == "weighted_score"
    assert lin[(0, "symbol", "score")] == lin[(3, "symbol", "S")] == "S"
    assert lin[(1, "op", "normalize")] == lin[(2, "op", "sizefactor")] == lin[(3, "op", "scale_counts")]
    assert lin[(0, "group", "prep")] == lin[(3, "group", "norm")] == "norm"


def test_diff(versions) -> None:
    from headway.diff import compare
    def summary(a: int, b: int) -> dict[tuple[str, str], tuple[str, list[str]]]:
        return {(c.kind, c.key): (c.status, c.what) for c in compare(versions, a, b)}
    assert summary(0, 1) == {("op", "center"): ("added", []), ("symbol", "Zc"): ("added", []),
                             ("op", "score"): ("changed", ["structure", "equation"])}
    assert summary(1, 2) == {("op", "sizefactor"): ("split", ["structure"]),
                             ("op", "scale_counts"): ("split", ["structure"]),
                             ("op", "log"): ("changed", ["code"]),       # the code changed, the equation did not
                             ("symbol", "f"): ("added", []), ("symbol", "scale"): ("removed", []),
                             ("group", "norm"): ("changed", ["renamed", "wording"])}
    v3_v4 = summary(2, 3)
    assert v3_v4[("op", "weighted_score")] == ("changed", ["renamed", "structure", "equation", "wording", "code"])
    assert v3_v4[("symbol", "S")] == ("changed", ["renamed"])
    assert ("op", "sizefactor") not in v3_v4                                     # the halves of a split pair by key
    assert {k for k, (st, _) in v3_v4.items() if st == "added"} == {("op", "weights"), ("symbol", "w"),
                                                                     ("symbol", "sd_floor")}
    # straight from v1 to v4, across the rename and the split
    v1_v4 = summary(0, 3)
    assert v1_v4[("op", "weighted_score")][0] == "changed" and v1_v4[("op", "sizefactor")][0] == "split"


def test_load_and_compare(versions, tmp_path) -> None:
    import json
    from headway import cli, diff, page
    (tmp_path / "v1.json").write_text(json.dumps(versions[0]))
    (tmp_path / "v2.html").write_text(page.html(versions[1]))
    for path in ("v1.json", "v2.html"):
        keys = [o["key"] for o in versions[int(path[1]) - 1]["ops"]]
        assert [o["key"] for o in diff.load(tmp_path / path)["ops"]] == keys
    (tmp_path / "other.html").write_text("<html></html>")
    with pytest.raises(ValueError):
        diff.load(tmp_path / "other.html")
    lanes = headway.api.lanes([str(tmp_path / "v1.json"), f"second={tmp_path / 'v2.html'}"])
    assert [l["name"] for l in lanes] == ["v1", "second"]
    page = page.html(versions[3] | {"lanes": lanes})
    assert '"lanes": [{"name": "v1"' in page
    assert "lanes" not in diff.load(tmp_path / "v2.html")      # a page's own lanes are not taken along


def test_diff_command(versions, tmp_path, capsys) -> None:
    import json
    from headway import cli
    for n, d in enumerate(versions[:2]):
        (tmp_path / f"v{n + 1}.json").write_text(json.dumps(d))
    assert cli.main(["diff", str(tmp_path / "v1.json"), str(tmp_path / "v2.json")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("v1 -> v2: 3 changes") and "+ op center" in out and "inputs: -Z, +Zc" in out


def test_source(tmp_path, monkeypatch) -> None:
    import shutil
    import subprocess
    from headway import page
    if not shutil.which("git"):
        pytest.skip("no git")
    (tmp_path / "gitspec.py").write_text('from headway import Headway\nG = Headway("g")\nG.index("i", "i", "row")\n')
    git = lambda *a: subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                                    check=True, capture_output=True)
    git("init", "-q")
    git("add", ".")
    git("commit", "-qm", "first")
    monkeypatch.syspath_prepend(str(tmp_path))
    G = importlib.import_module("gitspec").G
    src = page.source(G)
    assert src["file"].endswith("gitspec.py") and len(src["git"]["commit"]) == 40 and not src["git"]["dirty"]
    (tmp_path / "gitspec.py").write_text((tmp_path / "gitspec.py").read_text() + "# edited\n")
    assert page.source(G)["git"]["dirty"]


# ---------- benchmark suites and results ----------
SUITE = '''import numpy as np
from headway import Suite
B = Suite("toy bench", version="2")
s = B.symbol("s", "s_i", "score", dims=(), role="input")
B.dataset("a", "A")
B.dataset("b", "B")
B.dataset("h", "held out", heldout=True)
B.metric("acc", "a", "accuracy", equation="a = s", inputs=[s], better="higher", range=(0, 1), code="toy:gram",
         interval="95%, bootstrap over items")
B.metric("err", "e", "error", equation="e = 1 - s", inputs=[s], better="lower", range=(0, 2), code="toy:gram")
B.figure("samples", "samples", "what the method makes", code="toy:gram")
B.objective("overall", "overall", metrics={"acc": 3, "err": 1})
B.objective("rank", "mean rank", metrics=["acc", "err"], how="rank")
'''


def test_suite_checks() -> None:
    B = headway.Suite("bad")
    s = B.symbol("s", "s", "score", role="input")
    B.metric("m", "m", "metric", equation="m = s", inputs=[s], better="higher", code="toy:gram")    # no range
    B.objective("o", "o", metrics=["m", "nope"])
    errors = headway.check(B).errors
    assert any("undeclared metric 'nope'" in e for e in errors) and any("m has no range" in e for e in errors)
    with pytest.raises(AssertionError):
        B.metric("x", "x", "x", better="best")


def test_results(tmp_path, monkeypatch, capsys) -> None:
    import json
    from headway import bench, cli, page
    (tmp_path / "toysuite.py").write_text(SUITE)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    G = spec()
    for name, values in [("one", {"a": {"acc": [0.8, 0.9], "err": [0.4]}, "b": {"acc": [0.5], "err": [1.0]}}),
                         ("two", {"a": {"acc": [0.9], "err": [0.2]}, "b": {"acc": [0.7], "err": [0.6]},
                                  "zz": {"acc": [1]}}),                       # a dataset the suite does not have
                         ("three", {"a": {"acc": [1.0], "err": [0.0]}})]:     # b not run
        (tmp_path / f"{name}.json").write_text(json.dumps(page.data(G, headway.check(G)) | {"name": name}))
        (tmp_path / f"{name}.res").write_text(json.dumps({"suite": "toysuite", "values": values}))
        assert cli.main(["attach", str(tmp_path / f"{name}.json"), str(tmp_path / f"{name}.res")]) == 0
    capsys.readouterr()
    headways = [json.loads((tmp_path / f"{n}.json").read_text()) for n in ("one", "two", "three")]
    r = [g["results"][0] for g in headways]
    assert r[0]["id"] == "toy bench@2" and r[0]["suite"]["suite"]["version"] == "2"
    assert r[1]["warnings"] == ["dataset 'zz' is not in the suite; left out"] and "zz" not in r[1]["values"]
    suite = r[0]["suite"]
    overall, rank = suite["suite"]["objectives"]
    # overall: (3 * acc + (1 - err / 2)) / 4, averaged over a and b; three is incomplete (no b)
    one = (3 * 0.85 + (1 - 0.2) + 3 * 0.5 + (1 - 0.5)) / 8
    assert bench.scores(r, overall, suite)[0] == pytest.approx(one) and bench.scores(r, overall, suite)[2] is None
    assert bench.scores(r, rank, suite)[:2] == [2.0, 1.0]                  # two is better on everything
    assert cli.main(["results", *(str(tmp_path / f"{n}.json") for n in ("one", "two", "three"))]) == 0
    out = capsys.readouterr().out
    assert "suite toy bench@2" in out and "incomplete" in out and "0.850±0.0707" in out


def test_example_suite(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(EXAMPLE / "bench"))
    B = importlib.import_module("cellscore_bench.suite").B
    report = headway.check(B)
    assert report.ok and not report.warnings and all(ok for ok, _ in report.verified.values())
    assert [m for m in B.metrics] == ["auroc", "depth_dep", "split"] and B.datasets["heldout"].heldout
    assert B.datasets["depth"].values["depth_sd"] == 1.1           # the dataset's values are the simulation's


def test_datasets() -> None:
    from headway import page
    B = headway.Suite("data bench")
    s = B.symbol("s", "s", "score", role="input")
    n = B.setting("n", "n", "size")
    sd = B.setting("sd", r"\sigma", "spread", value=1.0)
    c = B.constant("c", "c", "a constant", value=2, code="toy:SCALE")
    x = B.symbol("x", "x", "data", role="output")
    B.op("make", "make", r"x \sim N(0, \sigma)", inputs=[n, sd, c], outputs=[x], code="toy:gram", group="data")
    B.metric("m", "m", "metric", equation="m = s", inputs=[s], better="higher", range=(0, 1), code="toy:gram")
    B.dataset("a", "A", values={"n": 10, "sd": 0.5}, facts={"cells": 1000}, code='toy:SIZES#"a"..,')
    B.dataset("b", "B", values={"sd": 2.0})                     # n: no value here, and no default
    report = headway.check(B)
    assert report.ok and any("setting n: no value for dataset b" in w for w in report.warnings)
    d = page.data(B, report)["suite"]["datasets"][0]
    assert d["values"] == {"n": "10", "sd": "0.5"} and d["facts"] == {"cells": "1000"}
    assert d["code"][0]["source"].strip() == '"a": 10,' and d["decl"]["fields"]["values"]
    B.dataset("c", "C", values={"c": 3, "zz": 1})
    errors = headway.check(B).errors
    assert any("a value for c, a constant" in e for e in errors)
    assert any("'zz', which is not declared" in e for e in errors)


def test_api(tmp_path, monkeypatch, capsys) -> None:
    """From a script in another codebase: build with results given in Python, then attach more to the page."""
    import json
    from headway import cli, diff
    (tmp_path / "toysuite.py").write_text(SUITE)
    monkeypatch.syspath_prepend(str(tmp_path))
    suite = importlib.import_module("toysuite").B
    d = headway.build(spec(), tmp_path / "v1.html", json=tmp_path / "v1.json", name="v1",
                      results=[{"suite": suite, "values": {"a": {"acc": [0.8]}}, "run": {"seed": 1}}])
    assert d["name"] == "v1" and d["results"][0]["id"] == "toy bench@2" and d["results"][0]["run"] == {"seed": 1}
    assert diff.load(tmp_path / "v1.json")["results"][0]["values"] == {"a": {"acc": [0.8]}}
    headway.build(spec(), tmp_path / "v2.html", name="v2", compare=[tmp_path / "v1.json"])
    # attach to a page: its other versions stay, the results go in
    (tmp_path / "more.json").write_text(json.dumps({"suite": "toysuite", "values": {"b": {"err": [0.3]}}}))
    headway.attach(tmp_path / "v2.html", tmp_path / "more.json")
    page = (tmp_path / "v2.html").read_text()
    data = json.loads(next(ln for ln in page.splitlines() if ln.startswith("let D = "))[8:-1])
    assert data["lanes"][0]["name"] == "v1" and data["results"][0]["values"] == {"b": {"err": [0.3]}}
    with pytest.raises(ValueError):
        headway.build(spec("toy:no_such_function"), None, strict=True)
    assert cli.main(["guide"]) == 0 and "headway" in capsys.readouterr().out


def test_replicates_figures_keys(tmp_path, monkeypatch, capsys) -> None:
    """Replicate ids pair values across versions; figures' images go in with their paths; keys label a version."""
    import json
    from headway import cli
    (tmp_path / "toysuite.py").write_text(SUITE)
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "figs").mkdir()
    (tmp_path / "figs" / "a.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    (tmp_path / "notes.txt").write_text("not an image")
    results = {"suite": "toysuite", "replicates": ["s0", "s1", "s2"],
               "values": {"a": {"acc": [0.8, float("nan"), 0.9], "err": [0.1, 0.2]}},
               "figures": {"a": {"samples": [{"image": "figs/a.svg", "replicate": "s0", "keys": {"judged_by": "agent"}},
                                             {"url": "https://example.org/p.png"},
                                             {"image": "notes.txt"},                       # not an image
                                             {"image": "figs/a.svg", "url": "x"},          # one or the other
                                             {"image": "figs/a.svg", "replicate": "s9"}],
                                 "nope": {"url": "u"}},                                     # not declared
                           "b": {"samples": {"url": "https://example.org/b.png"}},          # one image, not a list
                           "zz": {"samples": {"url": "u"}}}}
    (tmp_path / "res.json").write_text(json.dumps(results))
    d = headway.build(spec(), None, results=[tmp_path / "res.json"], keys={"lr": 0.01, "variant": "small"})
    r = d["results"][0]
    # the NaN is left out with its id; err's two values cannot be paired with three ids
    assert r["values"]["a"] == {"acc": [0.8, 0.9], "err": [0.1, 0.2]} and r["replicates"] == {"a": {"acc": ["s0", "s2"]}}
    ev = r["figures"]["a"]["samples"]
    assert len(ev) == 3 and set(r["figures"]["a"]) == {"samples"} and len(r["figures"]["b"]["samples"]) == 1
    assert ev[0]["src"].startswith("data:image/svg+xml;base64,") and ev[0]["path"] == str(tmp_path / "figs" / "a.svg")
    assert ev[0]["keys"] == {"judged_by": "agent"} and ev[0]["replicate"] == "s0"
    assert ev[1]["src"] == ev[1]["url"] == "https://example.org/p.png" and "path" not in ev[1]
    w = "\n".join(r["warnings"])
    assert "a/err: 2 values but 3 replicate ids" in w and "notes.txt is not an image" in w and "not both" in w
    assert "replicate 's9' is not among" in w and "figures: dataset 'zz'" in w and "figure 'nope' is not in" in w
    figure = r["suite"]["suite"]["figures"][0]
    assert figure["name"] == "samples" and figure["code"][0]["ok"] and figure["decl"]["line"]
    assert d["keys"] == {"lr": 0.01, "variant": "small"}
    # a key named like a constant of the spec must agree with it
    G = spec()
    G.constant("floor", r"\epsilon", "floor", value=0.001, code="toy:SCALE")
    assert not any("key floor" in w for w in headway.build(G, None, keys={"floor": 1e-3})["check"]["warnings"])
    d = headway.build(G, None, keys={"floor": 0.01})
    assert "key floor=0.01, but the spec's floor is 0.001" in d["check"]["warnings"]
    with pytest.raises(AssertionError):
        headway.build(spec(), None, keys={"lr": [1, 2]})
    # the command line: --key, numbers and booleans read as such
    (tmp_path / "kspec.py").write_text("import headway\nG = headway.Headway('k')\n")
    assert cli.main(["build", "kspec", "-o", str(tmp_path / "p.html"), "--json", str(tmp_path / "p.json"),
                     "--key", "lr=0.5", "--key", "warm=true", "--key", "arch=resnet"]) == 0
    assert json.loads((tmp_path / "p.json").read_text())["keys"] == {"lr": 0.5, "warm": True, "arch": "resnet"}


def test_home(tmp_path, monkeypatch) -> None:
    """`headway serve` with no spec: a home page listing specs and built headways, opening and comparing them."""
    import json
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    import urllib.parse

    from headway.home import Home, module_of
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))                # recents, kept apart
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / ".cache"))
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "graphspec.py").write_text(LIVE.replace("TEX", "2 x_n"))
    (tmp_path / "test_x.py").write_text(LIVE.replace("TEX", "2 x_n"))                 # tests are not specs
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "v.json").write_text("{}")
    assert module_of(tmp_path / "pkg" / "graphspec.py") == ("pkg.graphspec", tmp_path)
    for name in ("v1", "v2"):
        headway.build(spec(), tmp_path / f"{name}.html", json=tmp_path / f"{name}.json", name=name,
                      keys={"warm": name == "v2"})
    (tmp_path / "other.json").write_text('{"not": "a headway"}')
    home = Home(tmp_path)
    found = [x for x in map(home.what, home.files()[0]) if x]
    assert sorted(Path(x["abs"]).relative_to(tmp_path).as_posix() for x in found) == \
        ["pkg/graphspec.py", "v1.html", "v1.json", "v2.html", "v2.json"]

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), home.handler())
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def get(path: str, host: str | None = None) -> str:
        req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_port}{path}",
                                     headers={"Host": host} if host else {})
        return urllib.request.urlopen(req).read().decode()
    try:
        assert "let D = null;" in get("/") and str(tmp_path) in get("/")      # nothing open: the library
        listed = json.loads(get("/api/list"))
        assert sorted(x["path"] for x in listed["items"]) == ["pkg/graphspec.py", "v1.html", "v1.json", "v2.html",
                                                            "v2.json"]
        assert listed["folders"] == ["pkg"] and listed["dir"] == str(tmp_path)
        inner = json.loads(get("/api/list?dir=" + urllib.parse.quote(str(tmp_path / "pkg"))))
        assert [x["path"] for x in inner["items"]] == ["graphspec.py"] and inner["parent"] == str(tmp_path)
        assert next(x for x in listed["items"] if x["path"] == "v2.json")["keys"] == {"warm": True}
        assert json.loads(get("/api/recent"))["recent"] == []
        cached = json.loads((tmp_path / ".cache" / "headway" / "library.json").read_text())
        assert cached[str(tmp_path / "v1.json")][1]["name"] == "v1" and cached[str(tmp_path / "other.json")][1] is None
        assert Home(tmp_path).seen[tmp_path / "v1.json"][1]["name"] == "v1"           # a new server starts with it
        both = get("/?open=v2.json&open=" + urllib.parse.quote(str(tmp_path / "v1.json")))
        assert '"lanes": [{"name": "v1"' in both and '"name": "v2"' in both and f'"opened": "{tmp_path / "v2.json"}"' in both
        assert [r["path"].rsplit("/", 1)[-1] for r in json.loads(get("/api/recent"))["recent"]] == ["v2.json", "v1.json"]
        assert get("/view?file=v2.json&compare=v1.json") == both                  # the old form, redirected
        assert json.loads(get("/api/file?path=v1.json"))["name"] == "v1"
        live = get("/live/pkg/graphspec.py/")
        assert '"live": true' in live and "y_n = 2 x_n" in live and str(tmp_path / "pkg" / "graphspec.py") in home.live
        assert json.loads(get("/live/pkg/graphspec.py/state.json"))["ok"]
        assert '"lanes": [{"name": "v1"' in get("/live/pkg/graphspec.py/?open=v1.json")
        assert '"root": "' in live and '"root": "' in both                          # the page links to the library
        for bad in ("/?open=other.json", "/live/test_x.py/", "/api/file?path=other.json",
                    "/api/file?path=pkg/graphspec.py", "/api/list?dir=nowhere"):
            with pytest.raises(urllib.error.HTTPError):
                get(bad)
        with pytest.raises(urllib.error.HTTPError, match="403"):                   # another site's name for it
            get("/api/list", host="evil.example:80")
    finally:
        home.close()
        httpd.shutdown()


def test_intervals_and_differences(tmp_path, monkeypatch, capsys) -> None:
    """Values and differences are points or points with intervals; methods are compared by pairing replicates."""
    from headway import bench, cli
    (tmp_path / "toysuite.py").write_text(SUITE)
    monkeypatch.syspath_prepend(str(tmp_path))
    pt = lambda v, lo, hi: {"value": v, "lo": lo, "hi": hi}                      # noqa: E731
    one = headway.read_results({"suite": "toysuite", "replicates": ["s0", "s1", "s2"],
                                "values": {"a": {"acc": [pt(0.80, 0.70, 0.90), 0.82, {"value": 0.81}],
                                                 "err": [pt(0.2, 0.1, 0.3), 0.3, 0.25]}}})
    assert one["values"]["a"] == {"acc": [0.80, 0.82, 0.81], "err": [0.2, 0.3, 0.25]}
    assert one["intervals"] == {"a": {"acc": [[0.70, 0.90], None, None]}}         # err declares no interval
    assert any("err: intervals given, but the metric does not say" in w for w in one["warnings"])
    two = headway.read_results({
        "suite": "toysuite", "replicates": ["s1", "s2", "s3"],                     # s0 not run, s3 new
        "values": {"a": {"acc": [0.90, 0.85, pt(0.5, 0.6, 0.7)], "err": [0.1, 0.2, 0.3]}},   # an interval not holding
        "differences": {"one": {"a": {"acc": [pt(0.08, 0.02, 0.12), pt(0.04, -0.01, 0.09), 0.0],
                                      "err": [0.1]},                                # 1 for 3 ids
                                "b": {"acc": pt(0.05, 0.01, 0.09)}}}})              # pooled
    assert any("does not hold its value" in w for w in two["warnings"])
    assert any("err: 1 differences but 3 replicate ids" in w for w in two["warnings"])
    assert two["differences"]["one"]["a"]["acc"]["ids"] == ["s1", "s2", "s3"]
    assert two["differences"]["one"]["b"]["acc"]["pooled"]
    # paired by id: s1 and s2 in common; the benchmark's intervals: one of two exclude 0
    p = bench.paired(two, one, "a", "acc", "two", "one")
    assert p["paired"] and p["n"] == 2 and p["d"] == pytest.approx((0.90 - 0.82 + 0.85 - 0.81) / 2)
    assert p["agree"] == 2 and p["marker"] == "◐" and not p["supported"]
    # turned around: one's difference from two, from two's results
    q = bench.paired(one, two, "a", "acc", "one", "two")
    assert q["d"] == pytest.approx(-p["d"]) and q["intervals"][0] == pytest.approx([-0.12, -0.02])
    # without intervals, every paired replicate agreeing in sign supports it; unpaired never does
    e = bench.paired(two, one, "a", "err", "two", "one")
    assert e["marker"] == "" and e["agree"] == 2 and e["supported"]
    unpaired = {"values": {"a": {"acc": [0.9]}}}
    u = bench.paired(unpaired, one, "a", "acc", "u", "one")
    assert not u["paired"] and not u["supported"] and u["d"] == pytest.approx(0.9 - 0.81)
    # the text table: a block of differences from the method before
    for name, r in [("one", one), ("two", two)]:
        (tmp_path / f"{name}.json").write_text(json.dumps(page_data() | {"name": name, "results": [r]}))
    assert cli.main(["results", str(tmp_path / "one.json"), str(tmp_path / "two.json")]) == 0
    out = capsys.readouterr().out
    assert "two - one" in out and "2/2 ◐" in out


def page_data() -> dict:
    from headway import page
    G = spec()
    return page.data(G, headway.check(G))


def test_file_references(tmp_path, monkeypatch) -> None:
    from headway import files
    monkeypatch.chdir(tmp_path)
    (tmp_path / "R").mkdir()
    (tmp_path / "R" / "score.R").write_text("# scoring\nSCALE <- 2\n\nnormalize <- function(x, f) {\n  x / f\n}\n\n"
                                            "score <- function(x) {\n  y <- normalize(x, SCALE)\n  # the mean\n"
                                            "  mean(y)\n}\n")
    (tmp_path / "lib.rs").write_text("pub struct Model { w: f64 }\nimpl Model {\n    pub fn score(&self) -> f64 {\n"
                                     "        self.w\n    }\n}\n")
    (tmp_path / "main.nf").write_text("process ALIGN {\n    script:\n    \"\"\"\n    bwa mem\n    \"\"\"\n}\n\n"
                                      "workflow {\n    ALIGN()\n}\n")
    (tmp_path / "Snakefile").write_text("rule align:\n    input: \"a.fq\"\n    shell: \"bwa\"\n\nrule sort:\n"
                                        "    shell: \"samtools\"\n")
    span = lambda ref: (lambda loc: (loc.start, loc.end))(resolve(ref, [tmp_path]))      # noqa: E731
    # lines and text anchors: any file, no parser needed; on a line of code, the construct that starts there
    assert span("main.nf:2-4") == (2, 4) and span("main.nf:9") == (9, 9)
    assert span("main.nf#process ALIGN") == (1, 6) and span("main.nf#workflow") == (8, 10)
    assert span("main.nf#script..bwa") == (2, 4)
    loc = resolve("main.nf#workflow", [tmp_path])
    assert loc.lang == "nextflow" and loc.relpath == "main.nf" and loc.source.startswith("workflow {")
    with pytest.raises(RefError, match="no parser"):
        resolve("main.nf:ALIGN", [tmp_path])
    with pytest.raises(RefError, match="has 10"):
        resolve("main.nf:30", [tmp_path])
    with pytest.raises(RefError, match="no file"):
        resolve("R/gone.R:score", [tmp_path])
    if files._parser("r") is None or files._parser("rust") is None or files._parser("snakemake") is None:
        pytest.skip("tree-sitter parsers not available (they download on first use)")
    # names, found with tree-sitter
    assert span("R/score.R:normalize") == (4, 6) and span("R/score.R:score") == (8, 12)
    assert span("R/score.R:score#the mean") == (10, 11) and span("R/score.R#SCALE <-") == (2, 2)
    assert resolve("R/score.R:score", [tmp_path]).lang == "r"
    assert span("lib.rs:Model.score") == (3, 5) and span("lib.rs:Model") == (1, 1)
    assert span("Snakefile:sort") == (5, 6) and span("Snakefile#rule align") == (1, 3)
    with pytest.raises(RefError, match="defines normalize, score"):
        resolve("R/score.R:scores", [tmp_path])
    # in a spec: checked, and in the built page with the language
    G = headway.Headway("polyglot")
    x = G.symbol("x", "x", "x", role="input")
    y = G.symbol("y", "y", "y", role="output")
    G.op("score", "score", "y = x", inputs=[x], outputs=[y], code=["R/score.R:score", "main.nf#process ALIGN"])
    report = headway.check(G)
    assert report.ok, report.text()
    code = headway.build(G, None)["ops"][0]["code"]
    assert [c["lang"] for c in code] == ["r", "nextflow"] and code[0]["relpath"] == "R/score.R"
    G.op("gone", "gone", "y = x", inputs=[x], outputs=[y], code="R/score.R:gone")
    assert any("no definition 'gone'" in e for e in headway.check(G).errors)


def test_init(tmp_path, monkeypatch, capsys) -> None:
    from headway import cli
    from headway.draft import DraftError, draft
    pkg = tmp_path / "drafty"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "core.py").write_text(
        "from dataclasses import dataclass\nimport numpy as np\n\n\n"
        "def standardize(x):\n    mean, sd = stats(x)\n    for block in blocks(len(x)):\n"
        "        part = x[:, block]\n        part -= mean[block]\n        part /= sd[block]\n    return x\n\n\n"
        "def stats(x):\n    return x.mean(0), x.std(0)\n\n\n"
        "def blocks(n):\n    return [slice(0, n)]\n\n\n"
        "def fit(x, alpha):\n    return x.T @ x + alpha\n\n\n"
        "@dataclass\nclass Model:\n    weights: np.ndarray\n    spread: float\n")
    (pkg / "method.py").write_text(
        "from dataclasses import dataclass\nimport numpy as np\nfrom drafty import core\n"
        "from drafty.core import Model, fit\n\n\n"
        "@dataclass(frozen=True)\nclass Settings:\n    alpha: float = 1.0\n    folds: int = 2\n\n\n"
        "def run(x, labels, settings: Settings = Settings()):\n    s = settings\n    z = core.standardize(x)\n"
        "    models = []\n    for f in range(s.folds):\n        models.append(fit(z[labels == f], s.alpha))\n"
        "    spread = np.std(z)\n    model = Model(np.stack(models), spread)\n    return model.weights\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    ns: dict = {}
    plain = {}
    exec(draft("drafty.method:run"), plain)         # without type hints, settings is an input like any other
    assert plain["G"].ops["fit"].inputs == ["labels", "settings", "x"] and "alpha" not in plain["G"].symbols
    text = draft("drafty.method:run", hints=True)   # with them: a dataclass with defaults is a settings object
    exec(text, ns)
    G = ns["G"]
    assert G.hints and not plain["G"].hints         # and the panel takes the hints too
    ops = G.ops
    # steps: the package's calls; a called function that calls the package's own is a group of its calls
    assert set(ops) == {"stats", "blocks", "fit"} and set(G.groups) == {"standardize"}
    assert ops["stats"].group == "standardize" and ops["stats"].inputs == ["x"]
    assert ops["stats"].outputs == ["mean", "sd"]
    # wiring by binding only: standardize returns its parameter, so z holds what x holds; fit reads z, labels and
    # a settings field (a step in a loop is done per value of the loop's variable)
    assert ops["fit"].per == "f" and "f" in G.indices and ops["fit"].inputs == ["alpha", "labels", "x"]
    assert ops["fit"].outputs == ["fit"] and ops["fit"].code == [
        "drafty.core:fit", "drafty.method:run#models.append(fit(z[labels..models.append(fit(z[labels"]
    s = G.symbols
    assert s["alpha"].role == "setting" and s["alpha"].value == 1.0
    assert s["alpha"].code == ["drafty.method:Settings.alpha"]
    assert s["x"].role == "input" and s["labels"].role == "input"
    report = headway.check(G)
    assert report.ok, report.text()
    # what the bindings cannot show is listed: values changed in place (part -= ... changes x through a view;
    # models.append(...) collects fit's results)
    assert "drafty.core:standardize: may change values in place at lines 9 (`part -= mean[block]`), 10" in text
    assert "drafty.method:run: may change values in place at line 18 (`models.append(" in text
    # short symbols: a name that is notation already keeps it (x); others take the first free of their near
    # neighbours (initials, capitals, greek; settings greek first)
    assert [(k, s[k].tex) for k in ("x", "labels", "alpha", "mean", "sd", "fit")] == \
        [("x", "x"), ("labels", "l"), ("alpha", r"\alpha"), ("mean", "m"), ("sd", "s"), ("fit", "F")]
    from headway.codetex import neighbours as _neighbours
    assert _neighbours("size_factors", False)[:6] == ["s", "S", "f", "F", r"\sigma", r"\Sigma"]
    assert _neighbours("window_pseudo", True)[:2] == [r"\omega", r"\Omega"] and _neighbours("x0", False)[0] == "x_{0}"
    assert _neighbours("mu", False)[:2] == [r"\mu", "m"] and "e" not in _neighbours("effect", False)
    # each step's code names, as the symbols they hold ("From the code"): a parameter given a symbol as it is,
    # the call's return; once a symbol has indices, its step's math uses them
    assert ops["fit"].names == {"alpha": {"symbol": "alpha"}, "return": {"symbol": "fit"}}
    assert ops["stats"].names == {"x": {"symbol": "x"}}
    i, g = G.index("i", "i", "row"), G.index("g", "g", "column")
    s["x"].tex, s["x"].dims, s["mean"].tex = "x_{ig}", ("i", "g"), r"\mu_g"
    assert headway.page.code_tex(G, ops["stats"], headway.check(G)) == \
        r"\left(\mu_g, s\right) = \left(\operatorname{mean}_{i} x_{ig}, \operatorname{sd}_{i} x_{ig}\right)"   # into its outputs
    # m, n = f(a), f(b): each value named after its own variable. A value made by other code (ratio) is a gap:
    # an input of its own, with the line that made it; so is a loop's element (row). A result unused: noted
    (pkg / "pairs.py").write_text("from drafty.core import fit\n\n\ndef run(x, y):\n    m, n = fit(x, 1), fit(y, 2)\n"
                                  "    ratio = m.sum() / n.sum()\n    out = fit(ratio, 3)\n"
                                  "    for k, row in enumerate(x):\n        fit(row, k)\n    return out\n")
    ns = {}
    text = draft("drafty.pairs:run")
    exec(text, ns)
    ops, s = ns["G"].ops, ns["G"].symbols
    assert ops["fit"].outputs == ["m"] and ops["fit_2"].outputs == ["n"]
    assert ops["fit_3"].inputs == ["ratio"] and ops["fit_3"].outputs == ["out"] and s["out"].role == "output"
    assert s["ratio"].role == "input" and s["ratio"].code == ["drafty.pairs:run#ratio = m.sum()..ratio"]
    assert "Made by the code at pairs.py:6" in s["ratio"].desc
    assert ops["fit_4"].inputs == ["row"] and ops["fit_4"].outputs == [] and ops["fit_4"].per == "k"
    assert "Made by the code at pairs.py:8" in s["row"].desc
    assert "drafty.pairs:run: may change values in place at line 9 (`fit(row, k)`)" in text
    assert headway.check(ns["G"]).ok
    # changed in place: a parameter stored into (out[:] = x) is changed by the call, so the variable given for it
    # then holds the step's changed value; with type hints, so is a list a hinted parameter .append()s to
    (pkg / "inplace.py").write_text(
        "from drafty.core import fit\n\n\ndef fill(out, x):\n    out[:] = x\n\n\ndef grow(xs: list, v):\n"
        "    xs.append(v)\n\n\ndef run(x, y):\n    out = fit(x, 1)\n    fill(out, y)\n    total = fit(out, 2)\n"
        "    return total\n\n\ndef run2(xs: list, v):\n    grow(xs, v)\n    return fit(xs, 3)\n")
    ns = {}
    exec(draft("drafty.inplace:run"), ns)
    ops = ns["G"].ops
    assert ops["fill"].outputs == ["out_2"] and ns["G"].symbols["out_2"].name == "out"
    assert ops["fit_2"].inputs == ["out_2"]
    # names= only for a parameter that is a symbol as it is: inner(x * 2) gives v a value computed from x, an edge
    # from x but not x; a field of an object that is not known to be a settings object is not that object
    (pkg / "derived.py").write_text("from drafty.core import fit\n\n\ndef inner(v):\n    return fit(v, 1)\n\n\n"
                                    "def run(x, s):\n    a = inner(x * 2)\n    b = inner(x)\n    return fit(s.k, 2)\n")
    ns = {}
    exec(draft("drafty.derived:run"), ns)
    ops = ns["G"].ops
    assert ops["fit"].inputs == ["x"] and "x" not in ops["fit"].names           # in inner(x * 2)
    assert ops["fit_2"].inputs == ["x"] and ops["fit_2"].names["x"] == {"symbol": "x"}   # in inner(x)
    assert ops["fit_3"].inputs == ["s"] and "x" not in ops["fit_3"].names        # s.k is not s
    assert plain["G"].ops["fit"].names == {"return": {"symbol": "fit"}}           # nor is z[labels == f] x
    # docstrings (numpydoc, Google, Sphinx): what they say of the parameters and returns describes the symbols, a
    # settings field's comment its setting, the summary its group; with type hints, a shape names the indices
    (pkg / "documented.py").write_text(
        "from dataclasses import dataclass\nfrom drafty.core import fit\n\n\n@dataclass\nclass Opts:\n"
        "    k: float = 2.0      # the pseudocount\n\n\n"
        "def model(x, k):\n    \"\"\"Fit the model.\n\n    Returns\n    -------\n    w : ndarray of shape (n_genes,)\n"
        "        The weights.\n    \"\"\"\n    return fit(x, k)\n\n\n"
        "def run(counts, o: Opts):\n    \"\"\"Score.\n\n    Parameters\n    ----------\n"
        "    counts : ndarray of shape (n_cells, n_genes)\n        UMI counts.\n    \"\"\"\n"
        "    w = model(counts, o.k)\n    return fit(w, 1)\n")
    for hints in (False, True):
        ns = {}
        exec(draft("drafty.documented:run", depth=0 if not hints else 2, hints=hints), ns)
        sy = ns["G"].symbols
        w = ns["G"].ops["model" if not hints else "fit"].outputs[0]               # opened at depth 2: fit's result
        assert sy["counts"].desc == "UMI counts." and sy[w].desc == "The weights."
        assert sy[w].dims == (("n_genes",) if hints else ())
        assert sy["counts"].dims == (("n_cells", "n_genes") if hints else ())
        if hints:
            assert sy["k"].desc == "the pseudocount" and ns["G"].groups["model"].desc == \
                "Fit the model. (drafty.documented:model)"
        assert headway.check(ns["G"]).ok, headway.check(ns["G"]).text()
    # with type hints, shapes stated as jaxtyping does (Float[np.ndarray, "cells genes"]): the entry's axis names are
    # the spec's indices; a call binds the function's names to the caller's through the arguments given as they are
    # (column_means calls its axis "cols": here, genes); an axis only a result has is a new index (components); a
    # name a parameter has but the call does not bind leaves the shape unknown, not guessed
    (pkg / "shaped.py").write_text(
        "import numpy as np\nfrom jaxtyping import Float\nfrom drafty.core import fit\n\n\n"
        "def column_means(x: Float[np.ndarray, 'rows cols']) -> Float[np.ndarray, 'cols']:\n    return x.mean(0)\n\n\n"
        "def project(x: Float[np.ndarray, 'rows cols']) -> Float[np.ndarray, 'rows components']:\n"
        "    return x[:, :2]\n\n\n"
        "def run(counts: Float[np.ndarray, 'cells genes']):\n    mu = column_means(counts)\n    z = project(counts)\n"
        "    sub = counts[:3]\n    nu = column_means(sub)\n    return fit(mu, z), nu\n")
    for hints, dims in ((False, {}), (True, {"counts": ("cells", "genes"), "mu": ("genes",),
                                            "z": ("cells", "components"), "nu": ()})):
        ns = {}
        exec(draft("drafty.shaped:run", hints=hints), ns)
        sy = ns["G"].symbols
        assert {k: sy[k].dims for k in dims} == dims
        assert "components" in ns["G"].indices if hints else not ns["G"].indices
        assert headway.check(ns["G"]).ok
    # ... and with hints=True, headway check holds each step's bound symbols to those shapes
    G2 = ns["G"]
    G2.ops["column_means"].bind = {"x": "counts", "return": "mu"}
    assert [w for w in headway.check(G2).warnings if "annotated" in w or "the axis" in w] == [
        "op column_means_2: x is annotated with axes (rows cols) in drafty.shaped:column_means, but sub has no dims= "
        "(a scalar, or not given yet)",
        "op column_means_2: return is annotated with axes (cols) in drafty.shaped:column_means, but nu has no dims= "
        "(a scalar, or not given yet)"]                  # what the draft could not tell: to give
    G2.symbols["mu"].dims = ("cells",)
    G2.ops["column_means"].bind = {"x": "counts", "return": "mu"}
    assert any("the axis cols of drafty.shaped:column_means is genes in x but cells in return" in w
               for w in headway.check(G2).warnings)
    G2.symbols["counts"].dims = ("cells",)
    assert any("x is annotated with 2 axes (rows cols)" in w for w in headway.check(G2).warnings)
    # with type hints: a method of a value hinted as a package class is a step (self: what it is called on)
    (pkg / "methods.py").write_text(
        "from drafty.core import fit\n\n\nclass Source:\n    def load(self, n):\n        return fit(n, 1)\n\n\n"
        "def run(src: Source, n):\n    data = src.load(n)\n    return fit(data, 2)\n")
    for hints, steps in ((False, ["fit"]), (True, ["Source_load", "fit"])):
        ns = {}
        exec(draft("drafty.methods:run", depth=0, hints=hints), ns)
        assert list(ns["G"].ops) == steps
    assert ns["G"].ops["Source_load"].inputs == ["n", "src"] and ns["G"].ops["Source_load"].outputs == ["data"]
    assert ns["G"].ops["Source_load"].code[0] == "drafty.methods:Source.load"
    assert ns["G"].ops["fit"].inputs == ["data"]
    # with type hints, and what mypy infers from them: a view of an array (part = x[:, c], c a slice) changed in
    # place changes the array, so scale changes its parameter x
    (pkg / "views.py").write_text(
        "import numpy as np\nfrom drafty.core import fit\n\n\ndef scale(x: np.ndarray) -> np.ndarray:\n"
        "    for c in [slice(0, 1), slice(1, 2)]:\n        part = x[:, c]\n        part -= 1\n    return x\n\n\n"
        "def run(x: np.ndarray):\n    scale(x)\n    return fit(x, 1)\n")
    for hints, read in ((False, ["x"]), (True, ["x_2"])):
        ns = {}
        exec(draft("drafty.views:run", hints=hints), ns)
        assert ns["G"].ops["fit"].inputs == read
    assert ns["G"].ops["scale"].outputs == ["x_2"]
    for hints, read in ((False, "xs"), (True, "xs_2")):
        ns = {}
        exec(draft("drafty.inplace:run2", hints=hints), ns)
        assert ns["G"].ops["fit"].inputs == [read]
    # depth 0: no groups; the call to standardize is one step
    ns = {}
    exec(draft("drafty.method:run", depth=0, hints=True), ns)
    assert list(ns["G"].ops) == ["standardize", "fit"] and not ns["G"].groups
    assert ns["G"].ops["fit"].inputs == ["alpha", "labels", "z"]                # z holds standardize's result
    # clean code (--clean, written to be read): an assignment by other code is a step of its own, its code that line,
    # its inputs what it reads; a conversion that keeps the values (np.asarray) is the value; unpacking a
    # parameter shapes the parts as its annotation says; a single value a return makes is a step too; a
    # decorator is not a call the function makes; a constant stays a gap
    (pkg / "lines.py").write_text(
        "from collections.abc import Sequence\nimport numpy as np\nfrom jaxtyping import Float\n"
        "from drafty.core import fit\n\n\ndef mark(f):\n    return f\n\n\n@mark\n"
        "def weighted(x: Float[np.ndarray, 'rows cols'], w: Float[np.ndarray, 'rows']"
        ") -> Float[np.ndarray, 'rows cols']:\n    return np.sqrt(w)[:, None] * x\n\n\n"
        "def run(pair: Sequence[Float[np.ndarray, 'cells genes']], w: Float[np.ndarray, 'cells']):\n"
        "    a, b = pair\n    w = np.asarray(w, np.float64)\n    z = weighted(a, w)\n"
        "    top: Float[np.ndarray, 'genes'] = z.max(0) - b.mean(0)\n    k = 3\n    return fit(top, k)\n")
    ns = {}
    exec(draft("drafty.lines:run", clean=True), ns)
    G = ns["G"]
    assert list(G.ops) == ["a_b", "weighted", "top", "fit"]
    assert G.ops["a_b"].inputs == ["pair"] and G.ops["a_b"].outputs == ["a", "b"]
    assert G.ops["weighted"].inputs == ["a", "w"]                   # w: the parameter, converted
    assert G.ops["top"].inputs == ["b", "z"] and G.ops["top"].code == ["drafty.lines:run#top: Float[np.ndarray,..top:"]
    assert G.ops["fit"].inputs == ["k", "top"] and "Made by the code" in G.symbols["k"].desc
    assert {k: G.symbols[k].dims for k in ("a", "b", "z", "top")} == {
        "a": ("cells", "genes"), "b": ("cells", "genes"), "z": ("cells", "genes"), "top": ("genes",)}
    assert headway.check(G).ok
    # clean code: a record of the package's (a dataclass) holds each field's value, so reading a field is an edge
    # from it only; a comprehension is done per element (per= its enumerate's index, or that of the family it runs
    # over); zip runs over families side by side; every function is opened, all the way down, but not recursively
    (pkg / "folds.py").write_text(
        "from collections.abc import Sequence\nfrom dataclasses import dataclass\nimport numpy as np\n"
        "from jaxtyping import Float\nfrom drafty.core import fit\n\n\n@dataclass\nclass Fold:\n"
        "    a: Float[np.ndarray, 'rows cols']\n    b: Float[np.ndarray, 'rows']\n\n\n"
        "def inner(x: Float[np.ndarray, 'rows cols']) -> Float[np.ndarray, 'rows cols']:\n    return fit(x, 1)\n\n\n"
        "def middle(x: Float[np.ndarray, 'rows cols']) -> Fold:\n    a = inner(x)\n"
        "    b: Float[np.ndarray, 'rows'] = x.sum(1)\n    return Fold(a, b)\n\n\n"
        "def again(n):\n    return again(n - 1) if n else fit(n, 0)\n\n\n"
        "def run(xs: Sequence[Float[np.ndarray, 'cells genes']]):\n"
        "    folds = [middle(x) for k, x in enumerate(xs)]\n    tops = [f.a.max(0) for f in folds]\n"
        "    sums = [f.b + t.sum() for f, t in zip(folds, tops)]\n    return again(3), sums\n")
    ns = {}
    exec(draft("drafty.folds:run", clean=True), ns)
    G = ns["G"]
    assert {k: (o.inputs, o.outputs, o.per) for k, o in G.ops.items() if not k.startswith(("again", "fit_"))} == {
        "x": (["xs"], ["x"], "k"),                      # an element of xs, unpacked: per k
        "fit": (["x"], ["fit"], "k"),                   # inside middle, inside inner: opened all the way down
        "b": (["x"], ["b"], "k"),
        "tops": (["fit"], ["tops"], "k"),               # f.a: only the field read; per k, as folds is
        "sums": (["b", "tops"], ["sums"], "k")}         # zip: folds and tops side by side, per k
    assert G.symbols["x"].dims == ("cells", "genes") and G.symbols["fit"].per == "k"
    assert G.ops["again"].group == "again" and G.ops["again"].code[0] == "drafty.folds:again"   # its own call: a
                                                                                                # step, not opened
    # clean code: an if that decides values (its branches only assign; a value one does not assign keeps its own)
    # is one step, its math the cases, what its test reads an ordinary input; an if that makes a value only under
    # its test is a guard: its steps done under the test, in symbols; an if that literals decide (a default left as
    # it is) drafts only the branch taken, and says so; a loop that carries values (an iteration) is one step, from
    # what comes into it to what it leaves, and so is a function of the package's that iterates
    (pkg / "loops.py").write_text(
        "from drafty.core import fit\n\n\n"
        "def adjust(x, scaled=None):\n    if scaled is not None:\n        x = fit(x, scaled)\n    return fit(x, 3)\n\n\n"
        "def iterate(x, n):\n    for _ in range(n):\n        x = x - fit(x, 1)\n    return x\n\n\n"
        "def run(x, rounds, correct, extra):\n    if correct:\n        x = fit(x, 2)\n    else:\n        x = adjust(x)\n"
        "    if extra:\n        y = fit(x, 4)\n        print(y)\n"
        "    for r in range(rounds):\n        step = fit(x, 1)\n        x = x - step\n    return iterate(adjust(x), 2)\n")
    ns = {}
    exec(text := draft("drafty.loops:run", clean=True), ns)
    G = ns["G"]
    assert [(k, o.condition, o.inputs, o.outputs) for k, o in G.ops.items()] == [
        ("x", "", ["correct", "x"], ["x_2"]),            # the if that decides x: one step, correct an input
        ("fit", "extra", ["x_2"], ["y"]),               # y only when extra: a guard
        ("x_2", "", ["rounds", "x_2"], ["x_3"]),        # the loop: one step, what comes in to what it leaves
        ("fit_2", "", ["x_3"], ["fit"]),                # in adjust: its if is decided by its default
        ("iterate", "", ["fit"], ["iterate"])]          # a function that iterates: one step
    assert G.ops["fit_2"].group == "adjust" and G.ops["x"].code == ["drafty.loops:run#if correct:..x = adjust(x)"]
    assert {c: n["symbol"] for c, n in G.ops["x"].names.items()} == {"correct": "correct", "x": "x", "x=": "x_2",
                                                                    "return": "x_2"}
    assert "adjust: the branch at line 6 is not drafted: `scaled is not None` is False here (scaled = None)" in text
    assert headway.check(G).ok, headway.check(G).text()
    ns = {}
    exec(draft("drafty.lines:run", hints=True), ns)                   # without: the lines' values are gaps
    assert list(ns["G"].ops) == ["weighted", "fit"]
    # the command
    assert cli.main(["init", "drafty.method:run", "-o", "spec.py"]) == 0
    assert (tmp_path / "spec.py").read_text().startswith('"""A draft of drafty')
    with pytest.raises(SystemExit):
        cli.main(["init", "drafty.method:run", "-o", "spec.py"])         # will not overwrite
    assert "exists" in capsys.readouterr().err
    # drafting again: the spec keeps its init arguments, and a checksum tells whether it was edited since
    from headway.draft import DraftEdited, arguments, edited, redraft
    spec = tmp_path / "spec.py"
    assert arguments(spec) == {"entry": "drafty.method:run", "depth": 2} and not edited(spec.read_text())
    assert cli.main(["init", "drafty.method:run", "--hints", "-o", "hinted.py"]) == 0
    assert arguments(tmp_path / "hinted.py") == {"entry": "drafty.method:run", "depth": 2, "hints": True}
    assert cli.main(["init", "drafty.lines:run", "--clean", "-o", "lined.py"]) == 0
    assert arguments(tmp_path / "lined.py") == {"entry": "drafty.lines:run", "clean": True}
    method = pkg / "method.py"
    method.write_text(method.read_text().replace("spread = np.std(z)", "spread = core.stats(z)"))
    assert redraft(spec) is None and "'stats_2'" in spec.read_text()          # the code's new call, drafted
    spec.write_text(spec.read_text().replace('"TODO: one sentence', '"Fits a model per fold', 1))
    assert edited(spec.read_text())
    with pytest.raises(DraftEdited):
        redraft(spec)
    assert cli.main(["redraft", str(spec), "--force"]) == 0
    assert "Fits a model per fold" in (tmp_path / "spec.py.bak").read_text() and not edited(spec.read_text())
    with pytest.raises(DraftError, match="calls none"):
        draft("drafty.core:stats")
    with pytest.raises(DraftError, match="no package"):
        draft("nothere.mod:f")


def test_codetex(tmp_path, monkeypatch, capsys) -> None:
    from headway import cli
    from headway.codetex import Name, transcribe
    t = lambda src, **names: transcribe(src, names)                     # noqa: E731
    # numpy and base Python, written as math; unknown functions as operators, never guessed at
    assert t("def f(y):\n    return np.log1p(np.maximum(y, 0))") == \
        r"f\left(y\right) = \log\left(1 + \max\left(y, 0\right)\right)"
    assert t("def f(p):\n    return np.clip(p, 1e-4, None) ** 2") == \
        r"f\left(p\right) = \max\left(p, 10^{-4}\right)^{2}"
    assert t("def f(a, b):\n    return a @ b.T - a * 2") == r"f\left(a, b\right) = a\,b^\top - a \cdot 2"
    assert t("def f(x):\n    return frob(x, k=3)") == \
        r"f\left(x\right) = \operatorname{frob}\left(x, \text{k} = 3\right)"
    # strict: nothing that could depend on what the code does at run time is guessed at
    assert t("def f(a, b):\n    return a * b") == r"f\left(a, b\right) = a \odot b"      # elementwise, not A B
    assert t("def f(x):\n    return x.std(0, ddof=1)") == \
        r"f\left(x\right) = \operatorname{std}\left(x, 0, \text{ddof} = 1\right)"   # an unknown argument: as is
    assert t("def f(x):\n    return int(x) + np.nan_to_num(x)") == \
        r"f\left(x\right) = \operatorname{int}\left(x\right) + \operatorname{nan\_to\_num}\left(x\right)"
    assert t("def f(x):\n    return x[-1]") == r"f\left(x\right) = x_{[-1]}"
    # a call whose result is unused may change its arguments: it is written, and the code is not one equation
    assert t("def f(a):\n    y = a\n    a.sort()\n    return y") == \
        r"\begin{aligned}y &= a \\ &\operatorname{sort}\left(a\right) \\ f\left(a\right) &= y\end{aligned}"
    assert t("def f(a):\n    print(a)\n    return a") == r"f\left(a\right) = a"                   # printing is not
    # float() and int() give numbers, whatever they are given (the code says so): a product with one is ordinary
    assert t("def f(x, y):\n    s = float(x.sum())\n    return s * y") == r"f\left(x, y\right) = \sum x\,y"
    # with type hints taken as assumptions, a parameter annotated as a number is a scalar; without, unknown (⊙)
    src = "def f(k: float, p: np.float64, x: NDArray[np.float64]):\n    return k * p * x"
    assert t(src) == r"f\left(k, p, x\right) = k \odot p \odot x"
    assert transcribe(src, hints=True) == r"f\left(k, p, x\right) = k\,p\,x"
    assert transcribe("def f(a: NDArray, b: NDArray):\n    return a * b", hints=True) == \
        r"f\left(a, b\right) = a \odot b"
    # ... and a name a type checker infers is a number (typeinfo.py, by its place in the file), though unannotated
    kinds = lambda line, col, name: "scalar" if (line, col, name) == (12, 15, "c") else None   # noqa: E731
    method = "    def f(c, b):\n        return c * b"           # (a method's source, indented: its columns in the file)
    assert transcribe(method, hints=True, kinds=kinds, first_line=11) == r"f\left(c, b\right) = c\,b"
    assert transcribe(method, kinds=kinds, first_line=11) == r"f\left(c, b\right) = c \odot b"   # without hints
    # jaxtyping's named axes, with hints: indices (an axis's first letter no other index or one-letter name has, or
    # the spec's index for it), so sums are over indices and products are index notation
    def axes(ann: ast.expr) -> list[str] | None:
        return ann.slice.elts[1].value.split() if isinstance(ann, ast.Subscript) else None    # type: ignore
    src = ('def f(n: Float[np.ndarray, "bins groups"], q: Float[np.ndarray, "bins groups"], '
           'b: Float[np.ndarray, "bins"]):\n    return (n * q).sum(0) + b.sum()')
    assert transcribe(src, hints=True, axes=axes) == \
        r"f\left(n_{ig}, q_{ig}, b_{i}\right) = \sum_{i} n_{ig}\,q_{ig} + \sum_{i} b_{i}"
    assert transcribe(src, hints=True, axes=axes, indices={"bins": "j"}) == \
        r"f\left(n_{jg}, q_{jg}, b_{j}\right) = \sum_{j} n_{jg}\,q_{jg} + \sum_{j} b_{j}"
    assert transcribe(src, axes=axes) == r"f\left(n, q, b\right) = \sum_{\text{axis } 0} n \odot q + \sum b"
    # a mask with the axis's index restricts it: x where learns_c
    src = 'def f(x: Float[np.ndarray, "cells genes"], learns: Bool[np.ndarray, "cells"]):\n    return x[learns].mean(0)'
    assert transcribe(src, hints=True, axes=axes) == \
        r"f\left(x_{cg}, \mathit{learns}_{c}\right) = \operatorname{mean}_{c} \left.x_{cg}\right|_{\mathit{learns}_{c}}"
    # an assert is a condition the code holds under, under the equation: x.all() as the indexed inequality it
    # states, a condition of several parts (and, & in an all()) one part per line
    src = "def f(x, y):\n    assert (y > 0).all() and np.all((x < 1) & (x > 0))\n    return x / y"
    assert t(src) == (r"\begin{array}{l}\displaystyle f\left(x, y\right) = \frac{x}{y} \\[0.6em] \displaystyle "
                      r"\text{provided that } y > 0 \\ \displaystyle \text{and } x < 1 \\ \displaystyle \text{and } x > 0"
                      r"\end{array}")
    # an array made in a shape taken from indexed values has their indices (and an axis whose first letter is
    # taken, the first free usual index letter); a loop over range(n) is written as an algorithm is, "repeat n
    # times:" (or "for r = 0, ..., n - 1:" if it reads its turn), a variable assigned again written x ← ...
    src = ('def f(n: Float[np.ndarray, "bins groups"], b: Float[np.ndarray, "bins"]):\n'
           '    d = np.zeros(n.shape[1])\n    for _ in range(3):\n        d = d + (n * b[:, None]).sum(0)\n    return d')
    assert transcribe(src, hints=True, axes=axes) == (
        r"\begin{array}{l}\displaystyle d_{g} = 0 \\ \displaystyle \textbf{repeat } 3\textbf{ times:} \\ \displaystyle "
        r"\quad d_{g} \leftarrow d_{g} + \sum_{i} n_{ig}\,b_{i} \\ \displaystyle f\left(n_{ig}, b_{i}\right) = d_{g}"
        r"\end{array}")
    assert t("def f(x, n):\n    for r in range(n):\n        x = g(x, r)\n    return x") == (
        r"\begin{array}{l}\displaystyle \textbf{for } r = 0, \ldots, n - 1\textbf{:} \\ \displaystyle \quad x \leftarrow "
        r"\operatorname{g}\left(x, r\right) \\ \displaystyle f\left(x, n\right) = x\end{array}")
    # ... a function given to write in place (a helper's small loop) is written where it is called, its parameters
    # replaced by what the call gives; its return assigned (and nothing, for return x into x)
    helper = {"upd": "def upd(m, d, k):\n    for s in range(k):\n        d = d - g(m, d)\n    return d - d.mean()"}
    assert transcribe("def fit(m, d, rounds, newton):\n    for r in range(rounds):\n        d = upd(m, d, newton)\n"
                      "    return d", inline=helper) == (
        r"\begin{array}{l}\displaystyle \textbf{repeat } \mathit{rounds}\textbf{ times:} \\ \displaystyle \quad "
        r"\textbf{repeat } \mathit{newton}\textbf{ times:} \\ \displaystyle \quad \quad d \leftarrow d - "
        r"\operatorname{g}\left(m, d\right) \\ \displaystyle \quad d \leftarrow d - \operatorname{mean} d \\ "
        r"\displaystyle \operatorname{fit}\left(m, d, \mathit{rounds}, \mathit{newton}\right) = d\end{array}")
    # an if that decides values (each branch one assignment; a value a branch does not assign keeps its own) is
    # cases, a value the code had before written primed, or as the name given for what the step assigns (z=)
    src = "def f(z, d, correct):\n    if correct:\n        z, e = plate(z, d)\n    else:\n        e = {}\n    return z, e"
    assert t(src) == (r"\begin{aligned}\left(z', e\right) &= \begin{cases}\operatorname{plate}\left(z, d\right) & "
                      r"\text{if } \mathit{correct} \\ \left(z, \varnothing\right) & \text{otherwise}\end{cases} \\ "
                      r"f\left(z, d, \mathit{correct}\right) &= \left(z', e\right)\end{aligned}")
    assert transcribe("if correct:\n    z, e = plate(z, d)\nelse:\n    e = {}",
                      {"z": Name("z_{cs}"), "z=": Name(r"\hat z_{cs}")}).startswith(r"\left(\hat z_{cs}, e\right) = ")
    # a conversion that keeps the values, into a name assigned once and changed nowhere, is that value
    assert t("def f(a):\n    m = a.astype(np.float64)\n    for _ in range(2):\n        a2 = g(m)\n    return a2") \
        .startswith(r"\begin{array}{l}\displaystyle \textbf{repeat } 2\textbf{ times:} \\ \displaystyle \quad "
                    r"a_{2} = \operatorname{g}\left(a\right)")
    # a matrix product of values written with their indices shows its sum
    a, v = Name("a_{bg}", ("b", "g")), Name("v_{gk}", ("g", "k"))
    assert t("def f(a, v):\n    return a @ v", a=a, v=v) == r"f\left(a_{bg}, v_{gk}\right) = \sum_{g} a_{bg}\,v_{gk}"
    assert t("def f(a, v):\n    return a @ v", a=a, v=Name("v_{dk}", ("d", "k"))) == \
        r"f\left(a_{bg}, v_{dk}\right) = \sum_{g=d} a_{bg}\,v_{dk}"
    # a stepped slice keeps its step (x[::-1] is reversed, not x)
    assert t("def f(x):\n    return x[::-1]") == r"f\left(x\right) = x_{[::-1]}"
    # indices after a subscript of the symbol's own: p_{0} with indices b, v is p_{0,bv}
    from headway.codetex import _indexed_tex
    assert [_indexed_tex("p_{0}", ("b", "v")), _indexed_tex("x", ("i",)), _indexed_tex(r"\hat{x}^{2}", ("i",))] == \
        ["p_{0,bv}", "x_{i}", r"{\hat{x}^{2}}_{i}"]
    # a local name that is notation (r) does not take a symbol a given name has with its indices (r_{ig})
    got = transcribe("def f(m):\n    r = m * 2\n    return r - r.mean(0)", {"return": Name("r_{ig}", ("i", "g")),
                                                                         "m": Name("m_{ig}", ("i", "g"))}, short=True)
    assert got == r"\begin{aligned}R_{ig} &= m_{ig} \cdot 2 \\ r_{ig} &= R_{ig} - \operatorname{mean}_{i'} R_{i'g}\end{aligned}"
    # short symbols change no line: a name that is notation already (p) is written in as any other
    src = "def f(m, n):\n    p = m / n\n    return np.log(p)"
    assert transcribe(src, short=True) == transcribe(src) == r"f\left(m, n\right) = \log\left(\frac{m}{n}\right)"
    # for _ in range(n), a loop that carries nothing: repeated
    assert t("def f(xs, n):\n    for _ in range(n):\n        print(xs)\n    return xs") == (
        r"\begin{array}{l}\displaystyle \textbf{repeat } n\textbf{ times:} \\ \displaystyle f\left(\mathit{xs}, n\right) = "
        r"\mathit{xs}\end{array}")
    # an in-place update of a value another variable holds (y = x; x += 1): not substituted
    assert t("def f(x):\n    y = x\n    x += 1\n    return y") == \
        r"\begin{aligned}y &= x \\ x &\leftarrow x + 1 \\ f\left(x\right) &= y\end{aligned}"
    assert t("def f(x):\n    y = x * 2\n    x = x + 1\n    return y") == r"f\left(x\right) = x \cdot 2"
    # a comprehension's own variable is not the outer one
    assert t("def f(x, xs):\n    y = x + 1\n    return sum(y for y in xs)") == \
        r"f\left(x, \mathit{xs}\right) = \sum_{y \in \mathit{xs}} y"
    # a family only when the loop fills what was made for it whole
    assert "array" in t("def f(n, a):\n    out = np.zeros(n)\n    for k in range(1, n):\n        out[k] = a\n"
                        "    return out")
    assert t('def f(d):\n    return d["sample_type"]') == r"f\left(d\right) = d_{[\text{“sample\_type”}]}"
    # names: a symbol's TeX and indices; reductions over a known axis are written over its index (primed where the
    # same index is free outside it), and a variable assigned from a value with known indices is written with them
    z = Name("z_{ig}", ("i", "g"))
    assert t("def f(z):\n    return z - z.mean(0)", z=z, **{"return": Name(r"\tilde z_{ig}")}) == \
        r"\tilde z_{ig} = z_{ig} - \operatorname{mean}_{i'} z_{i'g}"
    assert t("def f(z):\n    return z - z.mean(0)") == r"f\left(z\right) = z - \operatorname{mean}_{\text{axis } 0} z"
    got = t("def f(z, module, weights):\n    v = weights[module]\n    return (z[:, module] * v).sum(1)",
            z=z, weights=Name("w_g", ("g",)))
    # straight-line code is one equation: each variable replaced by what was assigned to it; values written with
    # all their indices multiply as index notation does (z_{ig} w_g), not with ⊙
    assert got == (r"f\left(z_{ig}, \mathit{module}, w_g\right) = \sum_{g} \left.z_{ig}\right|_{g \in "
                   r"\mathit{module}}\,\left.w_g\right|_{g \in \mathit{module}}")
    # ... except a variable read more than once, which keeps its line (else it would be written out again each time)
    got = t("def f(z, module, weights):\n    v = weights[module]\n    return (z[:, module] * v).sum(1) / v.sum()",
            z=z, weights=Name("w_g", ("g",)))
    assert got == (r"\begin{aligned}v_{g} &= \left.w_g\right|_{g \in \mathit{module}} \\ f\left(z_{ig}, \mathit{module}, "
                   r"w_g\right) &= \frac{\sum_{g} \left.z_{ig}\right|_{g \in \mathit{module}}\,v_{g}}{\sum_{g} v_{g}}"
                   r"\end{aligned}")
    # ... except a variable given a name, which keeps its line; updates (+=, out=) are substituted too
    src = "def f(m, n):\n    p = m / n\n    p += 1\n    np.clip(p, 0, 1, out=p)\n    return np.log(p / (1 - p))"
    assert t(src) == (r"f\left(m, n\right) = \log\left(\frac{\min\left(\max\left(\frac{m}{n} + 1, 0\right), 1\right)}"
                      r"{1 - \min\left(\max\left(\frac{m}{n} + 1, 0\right), 1\right)}\right)")
    assert t(src, p=Name(r"\pi")) == (r"\begin{aligned}\pi &= \frac{m}{n} \\ \pi &\leftarrow \pi + 1 \\ "
                                      r"\pi &\leftarrow \min\left(\max\left(\pi, 0\right), 1\right) \\ "
                                      r"f\left(m, n\right) &= \log\left(\frac{\pi}{1 - \pi}\right)\end{aligned}")
    # an if/else and early returns: cases
    assert t("def f(x, t):\n    if x > t:\n        y = x - t\n    else:\n        y = 0\n    return 2 * y") == \
        r"f\left(x, t\right) = 2\,\begin{cases}x - t & \text{if } x > t \\ 0 & \text{otherwise}\end{cases}"
    assert t("def f(p, e):\n    if p < e:\n        return 0\n    if p > 1 - e:\n        return 1\n"
             "    return np.log(p)") == \
        (r"f\left(p, e\right) = \begin{cases}0 & \text{if } p < e \\ 1 & \text{if } p > 1 - e \\ \log\left(p\right) "
         r"& \text{otherwise}\end{cases}")
    # loops that add up or fill in: sums, products, indexed families
    assert t("def f(xs, w):\n    total = 0\n    for i in range(len(xs)):\n        d = xs[i] - w\n"
             "        total += d ** 2\n"
             "    return np.sqrt(total)") == \
        (r"f\left(\mathit{xs}, w\right) = \sqrt{\sum_{i=0}^{\operatorname{len}\left(\mathit{xs}\right) - 1} "
         r"\left(\mathit{xs}_{[i]} - w\right)^{2}}")
    assert t("def f(ps, a):\n    s = a\n    for p in ps:\n        s -= np.log(p)\n    return s") == \
        r"f\left(\mathit{ps}, a\right) = a - \sum_{p \in \mathit{ps}} \log\left(p\right)"
    assert t("def f(ps):\n    s = 1.0\n    for p in ps:\n        s *= p\n    return s") == \
        r"f\left(\mathit{ps}\right) = \prod_{p \in \mathit{ps}} p"
    assert t("def f(n, a):\n    out = np.zeros(n)\n    for k in range(n):\n        out[k] = a ** k\n"
             "    return out") == \
        r"f\left(n, a\right) = \left(a^{k}\right)_{k=0}^{n - 1}"
    assert t("def f(xs, w):\n    out = []\n    for i, x in enumerate(xs):\n        out.append(w[i] * x)\n"
             "    return np.array(out)") == \
        (r"f\left(\mathit{xs}, w\right) = \left(w_{[i]} \odot \mathit{xs}_{[i]}\right)_{i=0}^"
         r"{\operatorname{len}\left(\mathit{xs}\right) - 1}")
    assert "array" in t("def f(ps):\n    s = 0\n    for p in ps:\n        s += s * p\n    return s")   # not a sum
    # short: the other names as short symbols too, the first free of their neighbours; notation keeps itself (p),
    # and none looks like a symbol the step's names show (m_{if} takes m: mean becomes M)
    src = ("def f(m, pseudo, p):\n    shrunk = (m + pseudo * p) / (1 + pseudo)\n    mean = shrunk.mean()\n"
           "    return mean")
    assert transcribe(src, {"m": Name("m_{if}", ("i", "f"))}, short=True) == \
        r"f\left(m_{if}, P, p\right) = \operatorname{mean} \frac{m_{if} + P \odot p}{1 + P}"
    loop = "def f(xs):\n    for block in xs:\n        part_sum = block.sum()\n        print(part_sum)\n    return xs"
    assert r"\text{for } b \in" in transcribe(loop, short=True) and "mathit" not in transcribe(loop, short=True)
    # what cannot be written is named by its line in the file (the code starts at first_line), not in the snippet
    assert r"\text{(not written: line 51)}" in transcribe("def f(x):\n    g = lambda *t: t\n    for k in x:\n"
                                                           "        continue\n    return g", first_line=50)
    assert r"\text{continue}" in transcribe("def f(x):\n    for k in x:\n        continue\n    return x")
    assert transcribe("def f(names):\n    return {p: i for i, p in enumerate(names)}") == \
        (r"f\left(\mathit{names}\right) = \left\{p \mapsto i\right\}_{\left(i, p\right) \in "
         r"\operatorname{enumerate}\left(\mathit{names}\right)}")
    # a module imported by the file is not a value: embed.Embedding(a) is Embedding(a)
    assert transcribe("def f(a):\n    return embed.Embedding(a)", imported={"embed"}) == \
        r"f\left(a\right) = \operatorname{Embedding}\left(a\right)"
    loop = "def f(x, s):\n    for c in chunks(x):\n        x[c] += s.alpha\n    return x"   # not straight-line
    # written line by line, flush left, each line indented by its depth in the loops
    assert t(loop, **{"s.alpha": Name(r"\alpha")}) \
        == (r"\begin{array}{l}\displaystyle \text{for } c \in \operatorname{chunks}\left(x\right)\text{:} \\ "
            r"\displaystyle \quad x_{[c]} \leftarrow x_{[c]} + \alpha \\ \displaystyle f\left(x, s\right) = x"
            r"\end{array}")
    assert t("x = (") is None and t("def f():\n    pass") is None
    # in a spec: bind and names give the code's names; the page carries the step's code written as math
    G = headway.Headway("toy")
    n, p = G.index("n", "n", "row"), G.index("p", "p", "column")
    x = G.symbol("x", "x_{np}", "data", dims=(n, p), role="input")
    xc = G.symbol("xc", r"\tilde x_{np}", "centered", dims=(n, p), role="output")
    G.op("center", "center", inputs=[x], outputs=[xc], code="toy:center", names={"x": x, "return": xc, "mu": r"\mu_p"})
    report = headway.check(G)
    assert report.ok, report.text()
    d = headway.page.data(G, report)
    assert d["ops"][0]["code_tex"] == (r"\begin{aligned}\mu_p &= \operatorname{mean}_{n} x_{np} \\ \tilde x_{np} &= "
                                       r"x_{np} - \mu_p\end{aligned}")
    # the code's own names instead (the panel's switch), still with the indices for the sums
    assert headway.page.code_tex(G, G.ops["center"], report, code_names=True) == \
        r"\begin{aligned}\mu &= \operatorname{mean}_{n} x \\ \operatorname{center}\left(x\right) &= x - \mu\end{aligned}"
    # what the code calls each symbol (the viewer's Code names): the name its steps' code has for it, a setting's
    # field, else its key; a drafted spec's keys are its variables' names already
    G.setting("k", r"\kappa", "pseudocounts", value=5, code="toy:Box.total")
    G.constant("s", "s", "scale", value=2.0, code="toy:SCALE")
    G.ops["center"].names = {"data": {"symbol": "x"}, "return": {"symbol": "xc"}, "mu": {"tex": r"\mu_p"}}
    assert headway.page.code_names(G) == {"x": "data", "xc": "xc", "k": "total", "s": "SCALE"}
    G.draft = {"entry": "toy:center"}
    G.symbols["xc_2"] = G.symbols["xc"]
    assert headway.page.code_names(G)["x"] == "x" and headway.page.code_names(G)["xc_2"] == "xc"
    G.draft = None
    del G.symbols["xc_2"], G.symbols["k"], G.symbols["s"]
    G.ops["center"].names = {"x": {"symbol": "x"}, "return": {"symbol": "xc"}, "mu": {"tex": r"\mu_p"}}
    G.ops["center"].names["y"] = {"symbol": "gone"}
    assert headway.Headway("t", hints=True).hints and not headway.Headway("t").hints
    assert any("undeclared symbol 'gone'" in e for e in headway.check(G).errors)
    # the command
    assert cli.main(["tex", "toy:gram", "--name", "x=X"]) == 0
    assert capsys.readouterr().out.strip() == r"\operatorname{gram}\left(X\right) = X^\top\,X"


def test_codedoc(tmp_path, monkeypatch) -> None:
    from headway import files
    from headway.codedoc import describe
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "docmod.py").write_text(
        "#!/usr/bin/env python\n"
        "def a(x):\n    \"\"\"Centre the columns.\n\n    More detail, not shown.\"\"\"\n    return x\n\n\n"
        "# -----------------\n# Scale each row by its\n# total.\n# -----------------\ndef b(x):\n    return x\n\n\n"
        "def c(x):\n    # step 1: the means\n    mu = x.mean(0)\n    # step 2: subtract\n    return x - mu\n\n\n"
        "def d(x):\n    return x\n")
    (tmp_path / "score.R").write_text("#' Score each cell against a module.\n#'\n#' @param x counts\n"
                                      "score <- function(x) {\n  sum(x)\n}\n")
    (tmp_path / "lib.rs").write_text("pub struct S;\nimpl S {\n    /// The weighted total.\n"
                                     "    pub fn total(&self) -> f64 {\n        1.0\n    }\n}\n")
    (tmp_path / "run.sh").write_text("#!/bin/bash\nrun() {\n  # keep the samples with enough reads\n"
                                     "  awk '$2 > 5'\n}\n")
    doc = lambda ref: describe(resolve(ref, [tmp_path]))                      # noqa: E731
    assert doc("docmod:a") == "Centre the columns."                          # a docstring's first paragraph
    assert doc("docmod:b") == "Scale each row by its total."                 # the comments above, rules dropped
    assert doc("docmod:c#step 2") == "step 2: subtract"                      # the comment a block opens with
    assert doc("docmod:d") is None
    if files._parser("r") is None or files._parser("rust") is None or files._parser("bash") is None:
        pytest.skip("tree-sitter parsers not available (they download on first use)")
    assert doc("score.R:score") == "Score each cell against a module."       # roxygen, up to its tags
    assert doc("lib.rs:S.total") == "The weighted total."                    # a /// doc comment
    assert doc("run.sh:run") == "keep the samples with enough reads"         # the comment opening the body
    # in the page: a step with no description (or a TODO) has the code's
    G = headway.Headway("t")
    x = G.symbol("x", "x", "x", role="input")
    G.op("a", "a", desc="TODO", inputs=[x], outputs=[G.symbol("y", "y", "y", role="output")], code="docmod:a")
    d = headway.page.data(G, headway.check(G))
    assert d["ops"][0]["code_doc"] == "Centre the columns."
