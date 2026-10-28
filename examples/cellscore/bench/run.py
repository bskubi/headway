"""Runs each version of cellscore on the benchmark and builds that version's headway with the results, through
headway's Python API: what a benchmark script in a method's own repository does after each change. Writes
../results/vN.json (the same results, for `headway attach` or `build --results`) and ../headways/vN.{html,json}.
Five replicates (simulation seeds) per dataset, named by seed so that versions pair up; for each dataset, one
figure (seed 0's scores against depth, results/figs/), the suite's figure "score against depth"; and keys saying which of the
method's options each version has.

    python bench/run.py            every version
    python bench/run.py v4         one

v1 was never run on the noisy dataset, and only v1 and v4 on the held-out one: results can be missing."""
import importlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import headway  # noqa: E402
from cellscore_bench import data, figures, metrics, suite  # noqa: E402

VERSIONS = ("v1", "v2", "v3", "v4")
# what each version does, as labels to group or pair versions by
KEYS = {"v1": {"centered": False, "weighted": False}, "v2": {"centered": True, "weighted": False},
        "v3": {"centered": True, "weighted": False}, "v4": {"centered": True, "weighted": True}}
FIGS = HERE.parent / "results" / "figs"

SEEDS = range(5)
SKIP = {("v1", "noisy"), ("v2", "heldout"), ("v3", "heldout")}


def version(v: str, module: str = "core"):
    """That version's cellscore.core, or its spec, cellscore.graphspec (each vN/ holds the package as it was)."""
    loaded = sys.modules.get("cellscore")
    if loaded is None or not str(loaded.__file__).startswith(str(HERE.parent / v)):     # another version's
        for m in [m for m in sys.modules if m == "cellscore" or m.startswith("cellscore.")]:
            del sys.modules[m]
    sys.path.insert(0, str(HERE.parent / v))
    try:
        return importlib.import_module(f"cellscore.{module}")
    finally:
        sys.path.pop(0)


def run(v: str) -> dict:
    core = version(v)
    values: dict = {}
    figs: dict = {}
    FIGS.mkdir(parents=True, exist_ok=True)
    for name in data.DATASETS:
        if (v, name) in SKIP:
            continue
        out = values.setdefault(name, {"auroc": [], "depth_dep": [], "split": []})
        for seed in SEEDS:
            d = data.simulate(name, seed)
            score = core.score_cells(d.counts, d.module)
            genes = np.random.default_rng(seed).permutation(np.flatnonzero(d.module))
            half = np.zeros_like(d.module)
            half[genes[: len(genes) // 2]] = True
            out["auroc"].append(metrics.auroc(score, d.active))
            out["depth_dep"].append(metrics.depth_dependence(score, d.depth))
            out["split"].append(metrics.split_agreement(core.score_cells(d.counts, half),
                                                        core.score_cells(d.counts, d.module & ~half)))
            if seed == 0:
                fig = FIGS / f"{v}_{name}.svg"
                fig.write_text(figures.score_vs_depth(score, d.depth, d.active, f"{v} on {name}, seed 0"))
                figs[name] = {"score_depth": {"image": str(fig), "replicate": "seed0"}}
    return {"suite": "cellscore_bench.suite", "values": values, "replicates": [f"seed{s}" for s in SEEDS],
            "figures": figs, "run": {"seeds": list(SEEDS), "note": "simulated data; see cellscore_bench/data.py"}}


if __name__ == "__main__":
    out = HERE.parent
    (out / "results").mkdir(exist_ok=True)
    (out / "headways").mkdir(exist_ok=True)
    for v in sys.argv[1:] or VERSIONS:
        results = run(v)
        (out / "results" / f"{v}.json").write_text(json.dumps(results, indent=1))
        # the version's headway, with its results; the suite given as the object, already imported
        headway.build(version(v, "graphspec").G, out / "headways" / f"{v}.html", json=out / "headways" / f"{v}.json",
                      name=v, keys=KEYS[v], results=[results | {"suite": suite.B}], strict=True)
        print(f"{v}: wrote results/{v}.json, headways/{v}.html and .json", file=sys.stderr)
