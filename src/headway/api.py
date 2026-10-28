"""headway from Python: build a method's headway, with its benchmark results, from a script in any codebase.

    import headway
    values = run_my_benchmark()                         # {dataset: {metric: [one value per replicate]}}
    headway.build("mypkg.graphspec", "headways/v5.html", json="headways/v5.json", name="v5",
                  results=[{"suite": "mybench.suite", "values": values, "run": {"seeds": [0, 1, 2]}}],
                  compare=["headways/v4.json"])

A spec or a suite can be given as a module name ("pkg.mod", or "pkg.mod:attribute" when the module holds more
than one) or as the object itself. The command line (cli.py) does its work through these functions.
"""
from __future__ import annotations

import importlib
import json as jsonlib
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from headway import bench, diff, page
from headway.check import check
from headway.spec import Headway

Log = Callable[[str], None] | None


def load(target: str | Headway) -> Headway:
    """The Headway (or Suite) a module holds: "pkg.mod", or "pkg.mod:attribute"."""
    if isinstance(target, Headway):
        return target
    module_name, _, attr = target.partition(":")
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    module = importlib.import_module(module_name)
    if attr:
        return getattr(module, attr)
    found = [v for v in vars(module).values() if isinstance(v, Headway)]
    assert len(found) == 1, f"{module_name}: {len(found)} Headway objects; name one as {module_name}:<attribute>"
    return found[0]


def _name(target: str | Headway, g: Headway) -> str:
    """The spec's module name, as the viewer shows it in commands (`headway check <it>`)."""
    if isinstance(target, str):
        return target
    return next((d.module for d in g.declared.values()), "")


def read_results(r: Mapping[str, Any] | str | Path, log: Log = None) -> dict[str, Any]:
    """A results file (or the same as a dict) -> the entry kept in a headway: the values, checked against the
    suite, and the suite, checked and built. {"suite": module name or Suite, "values": {dataset: {metric:
    [values]}}, "replicates": [ids], "figures": {dataset: {figure: image}}, "run": {...}}: see bench.py. Images
    are relative to the results file, or to the working directory for a dict."""
    where = str(r) if isinstance(r, (str, Path)) else "results"
    base = Path(r).parent if isinstance(r, (str, Path)) else Path.cwd()        # what images are relative to
    if isinstance(r, (str, Path)):
        r = jsonlib.loads(Path(r).read_text())
    assert isinstance(r, Mapping) and "suite" in r and "values" in r, f"{where}: not results (needs suite, values)"
    suite = load(r["suite"])
    assert isinstance(suite, bench.Suite), f"{r['suite']}: not a headway.Suite"
    report = check(suite)
    if not report.ok and log:
        log(f"suite {suite.title}:\n{report.text()}")
    e = bench.entry({**r, "suite": _name(r["suite"], suite)},
                    page.data(suite, report, spec=_name(r["suite"], suite)), base)
    for w in e["warnings"]:
        if log:
            log(f"warning: {where}: {w}")
    return e


def lanes(versions: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Other versions to show beside the spec, each "[NAME=]FILE" (a --json file or a built page)."""
    out = []
    for item in map(str, versions):
        name, _, path = item.rpartition("=") if "=" in item and not Path(item).exists() else ("", "", item)
        d = diff.load(path)
        out.append({"name": name or d.get("name") or Path(path).stem, "from": str(Path(path).resolve()), "data": d})
    return out


def build(spec: str | Headway, out: str | Path | None = "headway.html", *, json: str | Path | None = None,
          name: str | None = None, results: Iterable[Mapping[str, Any] | str | Path] = (),
          compare: Iterable[str | Path] = (), collapse: Iterable[str] = (), hide: Iterable[str] = (),
          keys: Mapping[str, str | int | float | bool] | None = None, verify: bool = True, offline: bool = False,
          editor: str | None = None, strict: bool = False, log: Log = None) -> dict[str, Any]:
    """Check the spec, build it with any benchmark results, and write the page (`out`) and the JSON (`json`);
    either can be None. Returns the built data. keys: free-form labels of this version (a hyperparameter, a
    variant) to group or pair versions by; a key named like a setting or constant of the spec is checked against
    its value. strict: raise if the check finds errors (else they are in the data, under "check", and shown in the
    page)."""
    g = load(spec)
    report = check(g, run_verifiers=verify)
    if log:
        log(report.text())
    if strict and not report.ok:
        raise ValueError(f"{_name(spec, g)}: the check found errors:\n{report.text()}")
    d = page.data(g, report, list(collapse), list(hide), spec=_name(spec, g), editor_=page.editor(editor))
    if name:
        d["name"] = name
    if keys:
        d["keys"] = bench.keys(keys, "keys")
        for w in _key_conflicts(g, d["keys"]):
            d["check"]["warnings"].append(w)
            if log:
                log(f"warning: {w}")
    for r in results:
        bench.attach(d, read_results(r, log))
    compared = lanes(compare)
    if out is not None:
        Path(out).write_text(page.html(d | ({"lanes": compared} if compared else {}), offline=offline))
        if log:
            log(f"wrote {out}")
    if json is not None:
        Path(json).write_text(jsonlib.dumps(d, indent=1))
        if log:
            log(f"wrote {json}")
    return d


def read(path: str | Path) -> tuple[dict[str, Any], bool, bool]:
    """A built headway, with the other versions a page shows (its lanes): (data, is it a page, does the page work
    offline). diff.load reads the same but drops the lanes."""
    text = Path(path).read_text()
    is_page = not text.lstrip().startswith("{")
    if not is_page:
        return diff.load(path), False, False
    m = re.search(r"^let D = (\{.*\});$", text, re.M)
    if not m:
        raise ValueError(f"{path}: neither a headway JSON file nor a page built by headway")
    return jsonlib.loads(m[1]), True, "data:text/javascript;base64" in text


def _key_conflicts(g: Headway, keys: Mapping[str, Any]) -> list[str]:
    """Keys named like a symbol of the spec with a value, whose value differs from it."""
    return [f"key {k}={v!r}, but the spec's {k} is {g.symbols[k].value!r}" for k, v in keys.items()
            if k in g.symbols and g.symbols[k].value is not None and str(v) != str(g.symbols[k].value)]


def attach(path: str | Path, results: Mapping[str, Any] | str | Path, out: str | Path | None = None,
           log: Log = None) -> dict[str, Any]:
    """Put benchmark results into a built headway: the JSON from --json, or a built page (rewritten with the
    current viewer, keeping its other versions and whether it works offline). Results on the same suite are
    replaced. Writes `out` (default: the file itself) and returns the data."""
    path = Path(path)
    d, is_page, offline = read(path)
    bench.attach(d, read_results(results, log))
    target = Path(out or path)
    if is_page:
        target.write_text(page.html(d, offline=offline))
    else:
        target.write_text(jsonlib.dumps(d, indent=1))
    if log:
        log(f"wrote {target}")
    return d
