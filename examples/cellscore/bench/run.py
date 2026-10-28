"""Runs each version of cellscore on the benchmark and builds that version's headway with the results, through
headway's Python API: what a benchmark script in a method's own repository does after each change. Writes
../results/vN.json (the same results, for `headway attach` or `build --results`) and ../headways/vN.{html,json}.
Five replicates (simulation seeds) per dataset, named by seed so that versions pair up, each value with a 95%
interval from resampling the cells (200 draws), and each version's difference from the version before it and from
v1, with its own interval from the same draws; for each dataset, one
figure (seed 0's scores against depth, results/figs/), the suite's figure "score against depth"; and keys saying which of the
method's options each version has.

    python bench/run.py            every version
    python bench/run.py v4         one

v1 was never run on the noisy dataset, and only v1 and v4 on the held-out one: results can be missing."""
import functools
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


# the uncertainty within one seed: each metric recomputed on cells drawn with replacement, the same draws for every
# version, so that two versions' difference has an interval of its own (narrower than either version's)
RESAMPLES = 200
METRICS = ("auroc", "depth_dep", "split")


@functools.cache
def simulated(name: str, seed: int) -> data.Data:
    return data.simulate(name, seed)


@functools.cache
def draws(name: str, seed: int) -> np.ndarray:
    """RESAMPLES draws of the cells, with replacement: one row of cell indices per draw."""
    n = len(simulated(name, seed).active)
    return np.random.default_rng([seed, 7, len(name)]).integers(0, n, (RESAMPLES, n))


def scores(v: str) -> dict:
    """The version's scores on every dataset and seed it is run on: the module's, and its two random halves'."""
    core = version(v)
    out = {}
    for name in data.DATASETS:
        if (v, name) in SKIP:
            continue
        for seed in SEEDS:
            d = simulated(name, seed)
            genes = np.random.default_rng(seed).permutation(np.flatnonzero(d.module))
            half = np.zeros_like(d.module)
            half[genes[: len(genes) // 2]] = True
            out[name, seed] = (core.score_cells(d.counts, d.module), core.score_cells(d.counts, half),
                               core.score_cells(d.counts, d.module & ~half))
    return out


def measure(s: tuple, d: data.Data, cells: slice | np.ndarray = slice(None)) -> dict[str, float]:
    score, a, b = (x[cells] for x in s)
    return {"auroc": metrics.auroc(score, d.active[cells]),
            "depth_dep": metrics.depth_dependence(score, d.depth[cells]),
            "split": metrics.split_agreement(a, b)}


def with_interval(value: float, resampled: list[float]) -> dict[str, float]:
    """The value and the 95% percentile interval of its resampled values (widened to hold the value, which a
    percentile interval need not)."""
    lo, hi = np.percentile(resampled, [2.5, 97.5])
    return {"value": value, "lo": min(float(lo), value), "hi": max(float(hi), value)}


def run(v: str, scored: dict, against: list[str]) -> dict:
    """The results of version v, with its differences from the versions `against` (on the same cells and draws)."""
    values: dict = {}
    differences: dict = {}
    figs: dict = {}
    FIGS.mkdir(parents=True, exist_ok=True)
    for name in data.DATASETS:
        if (v, name) in SKIP:
            continue
        others = [o for o in against if (o, name) not in SKIP]
        for seed in SEEDS:
            d = simulated(name, seed)
            full = measure(scored[v][name, seed], d)
            boot = [measure(scored[v][name, seed], d, cells) for cells in draws(name, seed)]
            for m in METRICS:
                values.setdefault(name, {}).setdefault(m, []).append(with_interval(full[m], [b[m] for b in boot]))
            for o in others:
                full_o = measure(scored[o][name, seed], d)
                boot_o = [measure(scored[o][name, seed], d, cells) for cells in draws(name, seed)]
                for m in METRICS:
                    differences.setdefault(o, {}).setdefault(name, {}).setdefault(m, []).append(
                        with_interval(full[m] - full_o[m], [x[m] - y[m] for x, y in zip(boot, boot_o)]))
            if seed == 0:
                fig = FIGS / f"{v}_{name}.svg"
                fig.write_text(figures.score_vs_depth(scored[v][name, seed][0], d.depth, d.active,
                                                      f"{v} on {name}, seed 0"))
                figs[name] = {"score_depth": {"image": str(fig), "replicate": "seed0"}}
    return {"suite": "cellscore_bench.suite", "values": values, "replicates": [f"seed{s}" for s in SEEDS],
            "differences": differences, "figures": figs,
            "run": {"seeds": list(SEEDS), "resamples": RESAMPLES,
                    "note": "simulated data; see cellscore_bench/data.py"}}


if __name__ == "__main__":
    out = HERE.parent
    (out / "results").mkdir(exist_ok=True)
    (out / "headways").mkdir(exist_ok=True)
    scored = {v: scores(v) for v in VERSIONS}            # every version: each is compared with the one before
    for v in sys.argv[1:] or VERSIONS:
        # compared with the version before it and with the first (the baseline), where those differ
        n = VERSIONS.index(v)
        results = run(v, scored, list(dict.fromkeys(VERSIONS[k] for k in (n - 1, 0) if 0 <= k < n)))
        (out / "results" / f"{v}.json").write_text(json.dumps(results, indent=1))
        # the version's headway, with its results; the suite given as the object, already imported
        headway.build(version(v, "graphspec").G, out / "headways" / f"{v}.html", json=out / "headways" / f"{v}.json",
                      name=v, keys=KEYS[v], results=[results | {"suite": suite.B}], strict=True)
        print(f"{v}: wrote results/{v}.json, headways/{v}.html and .json", file=sys.stderr)
