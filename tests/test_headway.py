"""headway's checks catch what they should: a renamed function, an equation that no longer matches the code, an
array with the wrong dimensions."""
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


def test_home(tmp_path) -> None:
    """`headway serve` with no spec: a home page listing specs and built headways, opening and comparing them."""
    import json
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    from headway.home import Home, module_of
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
    found = [x for x in map(home.what, home.files()) if x]
    assert sorted(x["path"] for x in found) == ["pkg/graphspec.py", "v1.html", "v1.json", "v2.html", "v2.json"]

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), home.handler())
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def get(path: str) -> str:
        return urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_port}{path}").read().decode()
    try:
        page = get("/")
        assert "pkg.graphspec" in page and "warm=true" in page and "/view?file=v1.html" in page
        both = get("/view?file=v2.json&compare=v1.json")
        assert '"lanes": [{"name": "v1"' in both and '"name": "v2"' in both
        live = get("/live/pkg/graphspec.py/")
        assert '"live": true' in live and "y_n = 2 x_n" in live and "pkg/graphspec.py" in home.live
        assert json.loads(get("/live/pkg/graphspec.py/state.json"))["ok"]
        for bad in ("/view?file=../x.json", "/view?file=other.json", "/live/test_x.py/"):
            with pytest.raises(urllib.error.HTTPError):
                get(bad)
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
