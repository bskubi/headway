"""The atlas (atlas.py): versions of a method, their results, and the relations among them, in one folder."""
import copy
import json
import shutil
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

import headway
from headway import atlas as atlas_mod
from headway import bench, cli, diff
from headway.atlas import Atlas, AtlasError, page_id, slug, suite_hash

from test_headway import SUITE, built, spec


@pytest.fixture(scope="module")
def cellscore() -> dict:
    return {v: built(v) for v in ("v1", "v2", "v3", "v4")}


@pytest.fixture(scope="module")
def v1() -> dict:
    return built("v1")


def results(n: int, other: str | None = None) -> dict:
    """Toy results (toysuite), better as n grows; with differences from `other` if given."""
    r = {"suite": "toysuite", "replicates": ["s0", "s1", "s2"],
         "values": {"a": {"acc": [0.5 + n / 10 + e for e in (0, 0.01, -0.01)], "err": [0.5 - n / 10] * 3},
                    "b": {"acc": [0.4 + n / 10] * 3, "err": [0.6 - n / 10] * 3}}}
    if other:
        r["differences"] = {other: {"a": {"acc": [{"value": 0.1, "lo": 0.05, "hi": 0.15}] * 3}}}
    return r


@pytest.fixture
def toysuite(tmp_path, monkeypatch):
    (tmp_path / "toysuite.py").write_text(SUITE)
    monkeypatch.syspath_prepend(str(tmp_path))


def ladder(root: Path, cellscore: dict) -> Atlas:
    """v1 -> v2 -> v3 -> v4, each with results and its difference from the one before."""
    a = Atlas.init(root, "cellscore")
    for n, v in enumerate(("v1", "v2", "v3", "v4")):
        prev = f"v{n}" if n else None
        a.add_version(cellscore[v] | {"built_at": f"2026-01-0{n + 1}T00:00:00"}, v, parents=[prev] if prev else [],
                      why=f"why {v}" if prev else "", prediction="better" if prev else "")
        a.attach(v, headway.read_results(results(n, prev)))
    return a


def test_versions_results_relations(tmp_path, toysuite, cellscore) -> None:
    a = ladder(tmp_path / "at", cellscore)
    assert a.names == ["v1", "v2", "v3", "v4"]
    assert sorted(p.name for p in (tmp_path / "at").iterdir()) == \
        ["atlas.json", "relations.jsonl", "relresults", "results", "suites", "versions"]
    stored = json.loads((tmp_path / "at" / "versions" / "v2.json").read_text())
    assert "results" not in stored and stored["provenance"]["parents"] == \
        [{"name": "v1", "model": a.meta["versions"]["v1"]["model"], "why": "why v2", "prediction": "better"}]
    # the suite is stored once; each result refers to it
    assert list(a.meta["suites"]) == ["toy bench@2"] and len(list((tmp_path / "at" / "suites").iterdir())) == 1
    assert a.results("v3")[0]["suite"] == "toy bench@2" and "suite" not in a.results("v3")[0]["values"]
    # the parents made steps, and the differences went onto them (not into a version)
    steps = [r for r in a.relations if r["kind"] == "step"]
    assert [r["id"] for r in steps] == ["v1->v2", "v2->v3", "v3->v4"] and steps[0]["why"] == "why v2"
    rr = a.relresults("v2->v3")["toy bench@2"]
    assert rr["differences"][0]["of"] == ["v3", "v2"] and set(rr["members"]) == {"v2", "v3"}
    # entry() gives a version as the viewer and headway diff/results read it: results with suite and differences
    e = a.entry("v3")
    assert e["results"][0]["suite"]["suite"]["metrics"][0]["key"] == "acc"
    assert e["results"][0]["differences"]["v2"]["a"]["acc"]["intervals"][0] == [0.05, 0.15]
    p = bench.paired(e["results"][0], a.entry("v2")["results"][0], "a", "acc", "v3", "v2")
    assert p["marker"] == "●" and p["supported"]
    assert a.check() == [] and a.validate() == []
    # a prose-only rebuild keeps the model; a change to the code is a new model, its results now on an earlier build
    prose = json.loads(json.dumps(cellscore["v4"]))
    prose["ops"][0]["desc"] = "reworded"
    assert "the same model" in " ".join(a.add_version(prose, "v4"))
    assert a.check() == [] and a.meta["versions"]["v4"]["history"]
    code = json.loads(json.dumps(cellscore["v4"]))
    code["ops"][0]["code"] = [{"ref": "x:y", "source": "def y(): pass"}]
    a.add_version(code, "v4")
    assert a.check() == ["v4: its results on toy bench@2 were computed on an earlier build",
                         "relation v3->v4: its results on toy bench@2 were computed on an earlier build of v4"]
    assert a.entry("v4")["results"][0]["earlier_build"]
    assert a.summary()["results"]["toy bench@2"]["v4"]["earlier"]
    # keep the old build under another name: a step from it to the new one
    a.add_version(cellscore["v4"], "v4", keep_old="v4-old")
    assert a.has("v4-old") and a.results("v4-old") and not a.results("v4")
    assert any(r["id"] == "v4-old->v4" for r in a.relations)
    a.add_version(cellscore["v3"], "v3", drop_results=True)
    assert a.results("v3") == []


def test_relations_summary_select(tmp_path, toysuite, cellscore) -> None:
    a = ladder(tmp_path / "at", cellscore)
    alt = a.relate("alternatives", ["v2", "v3", "v4"], base="v1", label="ways on", tags=["t"])
    assert alt["id"] == "alternatives-1" and alt["members"][0] == {"v": "v1", "role": "base"}
    with pytest.raises(AtlasError, match="has no key"):
        a.relate("factorial", ["v1", "v2"], factors=["centered"])
    for n, v in enumerate(("v1", "v2")):                                # the levels are read from the keys
        a.add_version(cellscore[v] | {"keys": {"centered": bool(n)}, "built_at": f"2026-01-0{n + 1}T00:00:00"}, v)
    f = a.relate("factorial", ["v1", "v2"], factors=["centered"])
    assert [m["levels"] for m in f["members"]] == [{"centered": False}, {"centered": True}]
    a.update_relation("v1->v2", outcome="as predicted", tags=["x"])
    assert a.rel("v1->v2")["outcome"] == "as predicted"
    # a pair made for differences becomes the step drawn later between the same two
    a.attach("v4", headway.read_results(results(3, "v2")))
    assert a.between("v4", "v2")["kind"] == "pair"
    a.relate("step", ["v2", "v4"])
    assert a.between("v2", "v4")["id"] == "v2->v4" and not any(r["kind"] == "pair" for r in a.relations)
    assert a.relresults("v2->v4")["toy bench@2"]["differences"][0]["of"] == ["v4", "v2"]
    s = a.summary()
    assert s["order"] == ["v1", "v2", "v3", "v4"] and set(s["versions"]["v2"]) >= {"model", "ops", "keys", "parents"}
    assert s["scores"]["toy bench@2"]["overall"]["v4"] > s["scores"]["toy bench@2"]["overall"]["v1"]
    assert s["diffs"]["v1->v2"]["changed"] >= 1 and s["suites"]["toy bench@2"]["metrics"][0]["name"] == "accuracy"
    assert {x["key"] for x in s["lineages"]} >= set(s["versions"]["v4"]["ops"])
    assert a.select(["near:v2~1"]) == ["v1", "v2", "v3", "v4"]       # the alternatives join v1 to the others
    assert a.select(["path:v1..v3"]) == ["v1", "v3"] and a.select(["key:centered=true"]) == ["v2"]
    assert a.select(["relation:alternatives-1", "-v1"]) == ["v2", "v3", "v4"] and a.select(["tag:x"]) == ["v1", "v2"]
    with pytest.raises(AtlasError, match="not a version or a term"):
        a.select(["nope"])


def test_reorganize(tmp_path, toysuite, cellscore) -> None:
    a = ladder(tmp_path / "at", cellscore)
    a.set_view("line", {"kind": "stack", "path": ["v1", "v2", "v3"], "baseline": "v2"})
    a.rename("v2", "two")
    assert a.names == ["v1", "two", "v3", "v4"] and a.version("two")["name"] == "two"
    assert a.rel("v1->two")["members"][1]["v"] == "two" and a.results("two")[0]["version"] == "two"   # ids follow
    assert a.version("v3")["provenance"]["parents"][0]["name"] == "two"
    assert a.meta["views"]["line"] == {"kind": "stack", "path": ["v1", "two", "v3"], "baseline": "two"}
    assert a.relresults("two->v3")["toy bench@2"]["differences"][0]["of"] == ["v3", "two"]
    assert a.check() == [] and a.validate() == []
    # extract: what the versions need comes along; relations keep their members inside
    b = a.extract(tmp_path / "small", ["two", "v3"])
    assert b.names == ["two", "v3"] and [r["id"] for r in b.relations] == ["two->v3"]
    assert b.meta["views"]["line"]["path"] == ["two", "v3"] and b.validate() == [] and b.check() == []
    assert b.entry("v3")["results"][0]["differences"]["two"]
    # merge: the same name and model is one version; the same name, another model, needs a prefix
    c = Atlas.init(tmp_path / "other", "other")
    c.add_version(cellscore["v1"], "two")
    with pytest.raises(AtlasError, match="the same name, different models: two"):
        Atlas.merge(b, c, tmp_path / "m1")
    m, notes = Atlas.merge(b, c, tmp_path / "m2", prefix="c-")
    assert m.names == ["c-two", "two", "v3"] or sorted(m.names) == ["c-two", "two", "v3"]
    assert notes == ["renamed two to c-two"] and m.validate() == []
    # remove
    a.remove("v3")
    assert not a.has("v3") and not any("v3" in [x["v"] for x in r["members"]] for r in a.relations)
    # validate finds hand edits that break the format
    (tmp_path / "at" / "relations.jsonl").write_text(json.dumps({"id": "x", "kind": "nope", "members": [{"v": "v1"}]}) + "\n")
    errs = Atlas.open(tmp_path / "at").validate()
    assert any("kind 'nope'" in e for e in errs) and any("two or more" in e for e in errs)
    # a rebuild with a different suite of one id keeps both, apart
    with pytest.raises(AtlasError, match="not an atlas"):
        Atlas.open(tmp_path / "nowhere")


def test_import_and_command_line(tmp_path, toysuite, cellscore, capsys, monkeypatch) -> None:
    files = []
    for n, v in enumerate(("v1", "v2", "v3")):
        r = headway.read_results(results(n, f"v{n}" if n else None))
        p = tmp_path / f"{v}.json"
        p.write_text(json.dumps(cellscore[v] | {"built_at": f"2026-01-0{n + 1}T00:00:00", "results": [r]}))
        files.append(str(p))
    # a page with another version in a lane: both come in, and the page becomes a stack view
    page = headway.build(spec(), None, name="toy")
    (tmp_path / "cmp.html").write_text(headway.page.html(page | {"lanes": [{"name": "v1", "data": json.loads(Path(files[0]).read_text())}]}))
    at = str(tmp_path / "at")
    assert cli.main(["atlas", "init", at, "--name", "cs"]) == 0
    assert cli.main(["atlas", "import", at, *files, str(tmp_path / "cmp.html")]) == 0
    a = Atlas.open(at)
    assert a.names == ["v1", "v2", "v3", "toy"] and a.meta["views"]["cmp"]["path"] == ["v1", "toy"]
    assert a.between("v2", "v1")["kind"] == "pair"
    assert cli.main(["relate", at, "step", "v1", "v2", "--label", "first", "--why", "w"]) == 0
    assert Atlas.open(at).validate() == [] and Atlas.open(at).between("v1", "v2")["kind"] == "step"
    assert cli.main(["relate", at, "update", "v1->v2", "--outcome", "fine"]) == 0
    assert Atlas.open(at).rel("v1->v2")["outcome"] == "fine"
    capsys.readouterr()
    # results and diff read from the atlas are what they are from the files
    assert cli.main(["results", at, "v1", "v2", "v3"]) == 0
    from_atlas = capsys.readouterr().out
    assert cli.main(["results", *files]) == 0
    assert capsys.readouterr().out == from_atlas
    assert cli.main(["diff", at, "v1", "v3"]) == 0
    assert capsys.readouterr().out == diff.text([cellscore["v1"], cellscore["v3"]], ["v1", "v3"]) + "\n"
    assert cli.main(["atlas", "view", at, "two", "v1", "v3"]) == 0
    assert cli.main(["atlas", "view", at, "near", "--graph", "near:v2~1"]) == 0
    assert Atlas.open(at).meta["views"]["near"] == {"kind": "graph", "query": "near:v2~1"}
    assert cli.main(["atlas", "list", at]) == 0 and "v1->v2: step of v1, v2 (first)" in capsys.readouterr().out
    assert cli.main(["atlas", "check", at]) == 0 and "nothing to report" in capsys.readouterr().out
    assert cli.main(["atlas", "validate", at]) == 0
    assert cli.main(["atlas", "extract", at, str(tmp_path / "ex"), "path:v1..v2"]) == 0
    assert Atlas.open(tmp_path / "ex").names == ["v1", "v2"]
    assert cli.main(["atlas", "rename", at, "toy", "the-toy"]) == 0 and Atlas.open(at).has("the-toy")
    # a page of the atlas: the summary, and the versions in full
    assert cli.main(["page", at, "-o", str(tmp_path / "at.html"), "--view", "two"]) == 0
    text = (tmp_path / "at.html").read_text()
    assert 'id="atlas-main"' in text and '"atlas_view": "two"' in text and '"atlas_versions": {"v1"' in text
    d = Atlas.open(at).page_data("two", full=False)
    assert list(d["atlas_versions"]) == ["v1", "v3"] and len(d["atlas"]["versions"]) == 4
    with pytest.raises(AtlasError, match="no view"):
        Atlas.open(at).page_data("nope")
    # build --atlas: a version from a spec, with its results and parent
    monkeypatch.chdir(tmp_path)
    (tmp_path / "kspec.py").write_text("import headway\nG = headway.Headway('k')\n")
    (tmp_path / "res.json").write_text(json.dumps(results(5)))
    assert cli.main(["build", "kspec", "--atlas", at, "--name", "k1", "--from", "v3", "--why", "try",
                     "--results", str(tmp_path / "res.json")]) == 0
    a = Atlas.open(at)
    assert a.has("k1") and a.rel("v3->k1")["why"] == "try" and a.results("k1")
    assert not (tmp_path / "headway.html").exists()                # no page unless -o
    with pytest.raises(SystemExit):
        cli.main(["build", "kspec", "--atlas", at])                 # needs --name
    with pytest.raises(SystemExit):
        cli.main(["build", "kspec", "--from", "v1"])               # --from needs --atlas


def test_served(tmp_path, toysuite, cellscore, monkeypatch) -> None:
    from headway.home import Home
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / ".cache"))
    ladder(tmp_path / "at", cellscore)
    home = Home(tmp_path)
    found = [x for x in map(home.what, home.files()[0]) if x]
    assert [(x["kind"], x["name"], x.get("versions")) for x in found if x["kind"] != "spec"] == \
        [("atlas", "cellscore", 4)]                                         # its files are its own
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), home.handler())
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def get(path: str) -> str:
        return urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_port}{path}").read().decode()
    try:
        page = get("/atlas/at/")
        assert '"atlas_versions": {"v4"' in page and '"order": ["v1", "v2", "v3", "v4"]' in page
        assert json.loads(get("/atlas/at/atlas-version?name=v2"))["results"][0]["differences"]["v1"]
        assert get("/atlas/at") == page                                      # redirected to the folder
        with pytest.raises(urllib.error.HTTPError):
            get("/atlas/nowhere/")
    finally:
        httpd.shutdown()
    assert atlas_mod.is_atlas(tmp_path / "at") and not atlas_mod.is_atlas(tmp_path)


# ---------- versions: names, rebuilds, parents ----------
def test_names_never_share_a_file(tmp_path, v1) -> None:
    assert slug("v1") == "v1" and slug("x/y") != slug("x_y") and slug("x/y").startswith("x_y-")
    a = Atlas.init(tmp_path / "at")
    a.add_version(v1, "x/y")
    a.add_version(v1 | {"keys": {"other": True}}, "x_y")
    assert a.version("x/y").get("keys") != a.version("x_y")["keys"]


def test_rebuild_keeps_the_recorded_parents(tmp_path, toysuite, cellscore) -> None:
    a = ladder(tmp_path / "a", cellscore)
    before = a.version("v2")["provenance"]["parents"]
    a.add_version(cellscore["v2"], "v2")                    # a rebuild naming no parents
    assert a.version("v2")["provenance"]["parents"] == before
    assert a.summary()["versions"]["v2"]["parents"] == ["v1"]
    a.add_version(cellscore["v2"], "v2", parents=["v3"])    # naming others replaces them
    assert [p["name"] for p in a.version("v2")["provenance"]["parents"]] == ["v3"]


def test_a_later_result_does_not_vouch_for_an_earlier_one(tmp_path, toysuite, cellscore) -> None:
    a = Atlas.init(tmp_path / "a")
    a.add_version(cellscore["v1"], "x")
    a.add_version(cellscore["v2"], "y")
    a.attach("y", headway.read_results(results(2, "x")))   # y - x, on this x
    a.add_version(cellscore["v3"], "x")                     # x rebuilt: another model
    stale = ["relation pair:y|x: its results on toy bench@2 were computed on an earlier build of x"]
    assert a.check() == stale
    a.attach("x", headway.read_results(results(3, "y")))   # x - y, on the new x
    assert a.check() == stale
    # a file written before the models were kept per result: its "members" are read in their place
    f = next((tmp_path / "a" / "relresults").glob("*/*.json"))
    rr = json.loads(f.read_text())
    for x in rr["differences"]:
        del x["models"]
    f.write_text(json.dumps(rr))
    assert a.check() == []
    rr["members"]["x"] = "sha256:old"
    f.write_text(json.dumps(rr))
    assert a.check() == stale
    a.rename("x", "z")                                      # renamed in the models too
    assert a.check() == [s.replace("of x", "of z").replace("pair:y|x", "pair:y|z") for s in stale]


def test_arguments_not_dropped(tmp_path, toysuite, cellscore, capsys) -> None:
    with pytest.raises(SystemExit):
        cli.main(["build", "nowhere.spec", "--atlas", str(tmp_path / "a"), "--name", "v1", "--why", "because"])
    assert "--why and --prediction go with --from" in capsys.readouterr().err
    a = Atlas.init(tmp_path / "a")
    notes = a.add_version(cellscore["v1"], "v1", keep_old="v0")
    assert any("no earlier build to keep" in n for n in notes) and a.names == ["v1"]


def test_parent_added_after_child(tmp_path, toysuite, cellscore) -> None:
    """A parent not in the atlas gets no step until it is added; then the step is made, whichever came first."""
    a = Atlas.init(tmp_path / "at")
    notes = a.add_version(cellscore["v2"] | {"built_at": "2026-01-02"}, "v2", parents=["v1"], why="w")
    assert a.relations == [] and any("when it is added" in n for n in notes) and a.check() == []
    a.add_version(cellscore["v1"] | {"built_at": "2026-01-01"}, "v1")
    assert [(r["id"], r.get("why")) for r in a.relations] == [("v1->v2", "w")]
    assert a.scan() == [] and a.check() == []


def test_remove_drops_parent_and_scan_skips_absent(tmp_path, toysuite, cellscore) -> None:
    a = ladder(tmp_path / "at", cellscore)
    a.remove("v2")
    assert [p["name"] for p in a.version("v3").get("provenance", {}).get("parents", [])] == []
    assert a.scan() == [] and a.check() == []
    # a parent not in the atlas, written by hand: no step to it
    f = tmp_path / "at" / a.meta["versions"]["v4"]["file"]
    d = json.loads(f.read_text())
    d["provenance"]["parents"].append({"name": "ghost"})
    f.write_text(json.dumps(d))
    a.scan()
    assert not any(m["v"] == "ghost" for r in a.relations for m in r["members"])


def test_scan_reports_two_files_of_one_name(tmp_path, toysuite, cellscore) -> None:
    a = ladder(tmp_path / "at", cellscore)
    shutil.copy(tmp_path / "at/versions/v2.json", tmp_path / "at/versions/v2b.json")
    shutil.copy(tmp_path / "at/versions/v3.json", tmp_path / "at/versions/a3.json")    # sorted before v3.json
    notes = a.scan()
    assert a.meta["versions"]["v2"]["file"] == "versions/v2.json"
    assert a.meta["versions"]["v3"]["file"] == "versions/v3.json"
    assert any("v2b.json: also named v2" in n for n in notes) and any("a3.json: also named v3" in n for n in notes)


def test_page_id_leaves_out_editor_and_release(cellscore) -> None:
    d = cellscore["v1"]
    assert page_id(d) == page_id(d | {"editor": "pycharm://{file}", "headway": "99.0"})
    assert page_id(d) != page_id(d | {"desc": "reworded"})


# ---------- relations ----------
def test_step_absorbs_pair_and_rename_keeps_datasets(tmp_path, toysuite, v1) -> None:
    a = Atlas.init(tmp_path / "at")
    a.add_version(v1, "v1")
    a.attach("v1", headway.read_results(results(1, "v2")))                  # differences from v2, not there yet
    assert a.between("v1", "v2")["kind"] == "pair"
    a.add_version(v1 | {"desc": "x"}, "v2", parents=["v1"], why="w")
    assert [r["kind"] for r in a.relations] == ["step"] and a.relresults("v1->v2")
    assert a.version("v2")["provenance"]["parents"][0]["why"] == "w"
    a.add_version(a.version("v2"), "v2", parents=["v1"], why="explicit")    # what is typed wins
    assert a.version("v2")["provenance"]["parents"][0]["why"] == "explicit"
    a.rename("v1", "a")                                                     # dataset "a" keeps its name
    vals = a.relresults("a->v2")["toy bench@2"]["differences"][0]["values"]
    assert "a" in vals and vals["a"]["acc"]["values"]
    with pytest.raises(AtlasError, match="no version 'nope'"):
        a.relate("factorial", ["a", "nope"], factors=["k"])
    with pytest.raises(AtlasError, match="near:NAME~N"):
        a.select(["near:a~x"])


def test_absorbed_pair_rewritten_in_queries(tmp_path, toysuite, cellscore) -> None:
    a = Atlas.init(tmp_path / "at")
    a.add_version(cellscore["v1"] | {"built_at": "2026-01-01"}, "v1")
    a.add_version(cellscore["v2"] | {"built_at": "2026-01-02"}, "v2")
    a.attach("v2", headway.read_results(results(1, "v1")))
    a.set_view("g", {"kind": "graph", "query": "relation:pair:v2|v1"})
    a.relate("step", ["v1", "v2"])
    assert a.meta["views"]["g"]["query"] == "relation:v1->v2"


def test_attach_relation_checks_differences(tmp_path, toysuite, cellscore, capsys) -> None:
    """A relation's differences attached by hand are checked and stored as a version's are, so that `headway
    results` and the viewer read their intervals."""
    a = ladder(tmp_path / "at", cellscore)
    a.relate("pair", ["v1", "v4"])
    rr = tmp_path / "rr.json"
    rr.write_text(json.dumps({"replicates": ["s0", "s1", "s2"], "differences": [
        {"of": ["v4", "v1"], "values": {"a": {"acc": [{"value": 0.3, "lo": 0.2, "hi": 0.4}] * 3, "nope": [1]}}},
        {"of": ["v4"], "values": {}}]}))
    assert cli.main(["attach", str(tmp_path / "at"), "--relation", "pair:v1|v4", str(rr), "--suite",
                     "toy bench@2"]) == 0
    err = capsys.readouterr().err
    assert "'nope' is not in the suite" in err and "needs \"of\"" in err
    stored = Atlas.open(tmp_path / "at").relresults("pair:v1|v4")["toy bench@2"]["differences"]
    assert len(stored) == 1 and stored[0]["values"]["a"]["acc"] == {
        "values": [0.3] * 3, "intervals": [[0.2, 0.4]] * 3, "ids": ["s0", "s1", "s2"], "pooled": False}
    assert cli.main(["results", str(tmp_path / "at"), "v1", "v4"]) == 0
    assert "●" in capsys.readouterr().out


# ---------- renaming ----------
def test_rename_reids_default_ids(tmp_path, toysuite, cellscore) -> None:
    a = ladder(tmp_path / "at", cellscore)
    a.relate("pair", ["v1", "v3"])
    a.update_relation("v2->v3", outcome="kept")
    a.set_view("g", {"kind": "graph", "query": "relation:v2->v3 -relation:v3->v4"})
    a.rename("v3", "v3b")
    ids = [r["id"] for r in a.relations]
    assert ids == ["v1->v2", "v2->v3b", "v3b->v4", "pair:v1|v3b"]
    assert a.rel("v2->v3b")["outcome"] == "kept"
    rr = a.relresults("v2->v3b")["toy bench@2"]
    assert rr["relation"] == "v2->v3b" and rr["differences"][0]["of"] == ["v3b", "v2"]
    assert not (tmp_path / "at" / "relresults" / atlas_mod.slug("v2->v3")).exists()
    assert a.meta["views"]["g"]["query"] == "relation:v2->v3b -relation:v3b->v4"
    assert a.validate() == [] and a.check() == []
    # an id given by hand is kept
    a.relate("group", ["v1", "v3b"], rid="mine")
    a.rename("v3b", "v3c")
    assert "mine" in [r["id"] for r in a.relations]


def test_rename_rewrites_names_in_queries(tmp_path, toysuite, cellscore) -> None:
    a = ladder(tmp_path / "at", cellscore)
    a.set_view("g", {"kind": "graph", "query": "near:v2~1 -v4 path:v2..v3 in:v1,v2 v2 key:v2=1 relation:v1->v2"})
    a.rename("v2", "two")
    assert a.meta["views"]["g"]["query"] == "near:two~1 -v4 path:two..v3 in:v1,two two key:v2=1 relation:v1->two"
    assert a.select(["near:two~1", "-v4"]) == ["v1", "two", "v3"]


# ---------- merging atlases ----------
def changed_suite(entry: dict) -> dict:
    """Results whose suite has the id of toysuite's but other content."""
    suite = entry["suite"] | {"suite": entry["suite"]["suite"] | {"datasets": entry["suite"]["suite"]["datasets"][:1]}}
    return entry | {"suite": suite}                     # (a description alone does not make another suite)


def test_merge_keeps_relations_media_views(tmp_path, toysuite, v1) -> None:
    a = Atlas.init(tmp_path / "a")
    a.add_version(v1, "v1")
    b = Atlas.init(tmp_path / "b")
    b.add_version(v1, "v1")
    b.add_version(v1 | {"desc": "other"}, "v2", parents=["v1"])
    (tmp_path / "f.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    r = headway.read_results({"suite": "toysuite", "values": {"a": {"acc": [0.5]}},
                              "figures": {"a": {"samples": {"image": str(tmp_path / "f.svg")}}}})
    b.attach("v1", r)
    b.set_view("w", {"kind": "stack", "path": ["v1", "v2"], "baseline": "v2", "hidden": ["v2"]})
    m, _ = Atlas.merge(a, b, tmp_path / "m")
    assert [x["id"] for x in m.relations] == ["v1->v2"]                     # a step touching a shared version
    assert m.entry("v1")["results"][0]["figures"]["a"]["samples"][0]["src"].startswith("data:image/svg")
    c = Atlas.init(tmp_path / "c")
    c.add_version(v1 | {"ops": []}, "v2")                                   # v2 in both, another model
    m2, _ = Atlas.merge(c, b, tmp_path / "m2", prefix="b-")
    assert m2.meta["views"]["w"] == {"kind": "stack", "path": ["v1", "b-v2"], "baseline": "b-v2", "hidden": ["b-v2"]}


def test_merge_keeps_relation_results_on_their_suite(tmp_path, toysuite, cellscore) -> None:
    a = Atlas.init(tmp_path / "a")
    a.add_version(cellscore["v1"], "v1")
    a.attach("v1", headway.read_results(results(1)))
    b = Atlas.init(tmp_path / "b")
    b.add_version(cellscore["v3"], "x")
    b.add_version(cellscore["v4"], "y")
    b.attach("x", changed_suite(headway.read_results(results(2))))
    b.attach("y", changed_suite(headway.read_results(results(3, "x"))))
    out, notes = Atlas.merge(a, b, tmp_path / "out")
    forked = [k for k in out.meta["suites"] if "#" in k]
    assert len(forked) == 1 and any("differs from the one stored" in n for n in notes)
    assert [r["suite"] for r in out.results("y")] == forked
    assert {rr["suite"] for rr in out.relresults("pair:y|x").values()} == set(forked)
    assert out.entry("y")["results"][0]["differences"]["x"]
    assert out.validate() == []


def test_merge_prefix_that_takes_a_name(tmp_path, toysuite, cellscore) -> None:
    a = Atlas.init(tmp_path / "a")
    a.add_version(cellscore["v1"], "v1")
    a.add_version(cellscore["v2"], "b_v1")
    b = Atlas.init(tmp_path / "b")
    b.add_version(cellscore["v3"], "v1")
    with pytest.raises(AtlasError, match="b_v1"):
        Atlas.merge(a, b, tmp_path / "out", prefix="b_")
    assert not (tmp_path / "out").exists()
    out, notes = Atlas.merge(a, b, tmp_path / "out", prefix="c_")
    assert out.names == ["v1", "b_v1", "c_v1"] and notes == ["renamed v1 to c_v1"]


def test_merge_reids_relations_of_prefixed_versions(tmp_path, toysuite, cellscore) -> None:
    a = ladder(tmp_path / "a", cellscore)
    b = ladder(tmp_path / "b", cellscore)
    code = copy.deepcopy(cellscore["v4"])
    code["ops"][0]["code"] = [{"ref": "x:y", "source": "def y(): pass"}]
    b.add_version(code, "v4")
    b.set_view("g", {"kind": "graph", "query": "relation:v3->v4 near:v4~1"})
    out, _ = Atlas.merge(a, b, tmp_path / "out", prefix="b_")
    ids = {r["id"]: [m["v"] for m in r["members"]] for r in out.relations}
    assert ids["v3->v4"] == ["v3", "v4"] and ids["v3->b_v4"] == ["v3", "b_v4"]
    assert out.relresults("v3->b_v4")["toy bench@2"]["differences"][0]["of"] == ["b_v4", "v3"]
    assert out.meta["views"]["g"]["query"] == "relation:v3->b_v4 near:b_v4~1"
    assert out.validate() == []


# ---------- suites and their hashes ----------
def test_suite_hash_ignores_wording_and_lines(tmp_path, toysuite, cellscore) -> None:
    """A line added to the suite's module, or a reworded description, keeps the suite (results stay comparable);
    a change to what is measured does not."""
    a = Atlas.init(tmp_path / "at")
    for v in ("v1", "v2", "v3"):
        a.add_version(cellscore[v], v)
    a.attach("v1", headway.read_results(results(1)))
    reworded = "# a note\n" + SUITE.replace('B.dataset("a", "A")', 'B.dataset("a", "A, reworded", "described now")') \
        .replace('"what the method makes"', '"what the method draws"')
    (tmp_path / "toysuite.py").write_text(reworded)
    sys.modules.pop("toysuite", None)
    notes = a.attach("v2", headway.read_results(results(2)))
    assert list(a.meta["suites"]) == ["toy bench@2"] and not any("warning" in n for n in notes)
    assert "no suite" not in bench.table([a.entry("v1"), a.entry("v2")], ["v1", "v2"])
    (tmp_path / "toysuite.py").write_text(reworded.replace("range=(0, 2)", "range=(0, 2.5)"))  # another size: no stale .pyc
    sys.modules.pop("toysuite", None)
    notes = a.attach("v3", headway.read_results(results(3)))
    sys.modules.pop("toysuite", None)                  # later tests import the suite as written in test_headway
    assert len(a.meta["suites"]) == 2 and any("differs" in n for n in notes)


def test_suite_entry_without_hash(tmp_path, toysuite, cellscore) -> None:
    """A suites entry written by hand without its hash: validate says so, and attaching results does not fail."""
    a = ladder(tmp_path / "at", cellscore)
    m = json.loads((tmp_path / "at/atlas.json").read_text())
    m["suites"]["toy bench@2"] = {"file": m["suites"]["toy bench@2"]["file"]}
    (tmp_path / "at/atlas.json").write_text(json.dumps(m))
    a = Atlas.open(tmp_path / "at")
    assert any("no hash" in e for e in a.validate())
    notes = a.attach("v1", headway.read_results(results(0)))
    assert list(a.meta["suites"]) == ["toy bench@2"] and not any("warning" in n for n in notes)
    assert a.validate() == []


def test_suite_hash_covers_dataset_code(tmp_path, toysuite, cellscore) -> None:
    """How a dataset is made or loaded makes another suite; how a figure is drawn does not."""
    a = ladder(tmp_path / "at", cellscore)
    s = a.suite("toy bench@2")
    data = copy.deepcopy(s)
    data["suite"]["datasets"][0]["code"] = [{"ref": "toy:load", "source": "def load(): return 1"}]
    other = copy.deepcopy(data)
    other["suite"]["datasets"][0]["code"][0]["source"] = "def load(): return 2"
    figure = copy.deepcopy(s)
    figure["suite"]["figures"][0]["code"] = [{"ref": "toy:draw", "source": "def draw(): pass"}]
    assert len({suite_hash(s), suite_hash(data), suite_hash(other)}) == 3
    assert suite_hash(figure) == suite_hash(s)


# ---------- the summary: fingerprints and lineages ----------
def test_split_step_fingerprint(tmp_path, toysuite, cellscore) -> None:
    """Two ops of one version in one lineage (a split step): a change to either changes the lineage's fingerprint."""
    a = ladder(tmp_path / "at", cellscore)
    d = a.version("v4")
    for o in d["ops"]:
        if o["key"] == "sizefactor":
            o["code"] = [{"ref": "x:y", "source": "def changed(): pass"}]
    d.pop("provenance", None)
    a.add_version(d, "v5", parents=["v4"])
    s = a.summary()
    assert s["versions"]["v4"]["ops"]["sizefactor"] != s["versions"]["v5"]["ops"]["sizefactor"]


def test_lineages_in_pipeline_order(tmp_path, toysuite, cellscore) -> None:
    a = ladder(tmp_path / "at", cellscore)
    lin = [x["label"] for x in a.summary()["lineages"]]
    newest = [o.get("label") or o["key"] for o in a.version("v4")["ops"]]
    shown = [x for x in newest if x in lin]
    assert [x for x in lin if x in shown] == shown        # the newest version's order, not its reverse


# ---------- views and the command line ----------
def test_graph_view_keeps_viewer_terms(tmp_path, toysuite, cellscore) -> None:
    ladder(tmp_path / "a", cellscore)
    assert cli.main(["atlas", "view", str(tmp_path / "a"), "g", "--graph", "contains:op:x", "score>0.5"]) == 0
    assert Atlas.open(tmp_path / "a").meta["views"]["g"] == {"kind": "graph", "query": "contains:op:x score>0.5"}
    with pytest.raises(SystemExit):                         # a stack's terms are still read here
        cli.main(["atlas", "view", str(tmp_path / "a"), "s", "contains:op:x"])


def test_cli_atlas_terms_matching_nothing(tmp_path, toysuite, cellscore, capsys) -> None:
    ladder(tmp_path / "at", cellscore)
    with pytest.raises(SystemExit):
        cli.main(["results", str(tmp_path / "at"), "kind:alternatives"])
    assert "no version of the atlas matches kind:alternatives" in capsys.readouterr().err


def test_cli_serve_atlas_with_compare(tmp_path, capsys) -> None:
    Atlas.init(tmp_path / "at")
    (tmp_path / "v1.json").write_text(json.dumps(built("v1")))
    with pytest.raises(SystemExit):
        cli.main(["serve", str(tmp_path / "at"), "--compare", str(tmp_path / "v1.json")])
    assert "--compare needs a spec" in capsys.readouterr().err


# ---------- atlases served and listed by the home ----------
def test_atlas_assets_and_redirect(tmp_path, toysuite, cellscore, monkeypatch) -> None:
    from headway.home import Home
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / ".cache"))
    doc = tmp_path / "paper.pdf"
    doc.write_bytes(b"%PDF-1.4 not really")
    other = tmp_path / "secret.txt"
    other.write_text("no")
    a = ladder(tmp_path / "at", cellscore)
    a.add_version(cellscore["v4"] | {"documents": [{"key": "p", "path": str(doc)}]}, "v4")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Home(tmp_path).handler())
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def get(path: str) -> bytes:
        return urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_port}{path}").read()
    try:
        assert get(f"/atlas/at/asset?name=v4&path={doc}") == doc.read_bytes()
        for bad in (f"/atlas/at/asset?name=v4&path={other}", f"/atlas/at/asset?name=v3&path={doc}",
                    f"/atlas/at/asset?name=nope&path={doc}"):
            with pytest.raises(urllib.error.HTTPError):
                get(bad)
        a.set_view("late", {"kind": "stack", "path": ["v1"]})
        assert b'"atlas_view": "late"' in get("/atlas/at?view=late")     # the redirect keeps the query
    finally:
        httpd.shutdown()


def test_home_lists_folder_with_foreign_atlas_json(tmp_path, monkeypatch) -> None:
    from headway.home import Home
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / ".cache"))
    game = tmp_path / "game"
    (game / "deeper").mkdir(parents=True)
    (game / "atlas.json").write_text(json.dumps({"frames": {}, "meta": {}}))     # a texture atlas, not headway's
    (game / "deeper" / "v1.json").write_text(json.dumps(built("v1")))
    found, _ = Home(tmp_path).files()
    assert game / "deeper" / "v1.json" in found
    real = tmp_path / "real"
    Atlas.init(real)
    (real / "inside.json").write_text("{}")
    found, _ = Home(tmp_path).files()
    assert real / "atlas.json" in found and real / "inside.json" not in found
