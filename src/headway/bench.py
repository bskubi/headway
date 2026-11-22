"""Benchmark suites, and the results of methods on them.

A suite is a Headway: its steps are the metrics' computations (an equation, references into the metric code, @verify
checks), its outputs the metrics, so a metric is annotated and checked like any step of a method. On top of that
a suite declares the datasets, each metric's direction (is higher better?) and range, and objectives: named ways
to combine metrics into one number to rank by, kept apart from the metrics.

A dataset is annotated the same way. If the suite makes its data (a simulation, a preprocessing), those steps are
steps of the suite, with their equations, code and checks, and each dataset gives its values of their settings:
the steps are written once and the datasets differ in their values. A dataset can also say where it comes from
(`source`), anything else worth knowing (`facts`: size, platform, accession) and where it is made or loaded (`code`).

headway computes no metric. Your benchmark computes them and writes a results file:

    {"suite": "my_bench.suite",                     # the suite module (or "module:attribute")
     "values": {"dataset": {"metric": [0.81, 0.79, ...], ...}, ...},     # one value per replicate (seed, fold)
     "replicates": ["seed0", "seed1", ...],          # what each value is, in the same order (optional)
     "differences": {"v3": {"dataset": {"metric": [...]}}},     # this method minus another, per replicate (optional)
     "figures": {"dataset": {"figure": {"image": "figs/x.png"}, ...}, ...},     # images to look at (optional)
     "run": {...}}                                  # anything worth keeping: date, settings, seeds (optional)

The replicate ids name the replicates, so that two methods' values can be paired (the same seed, the same fold);
without them the values are unpaired. They are a list, the same for every dataset, or {dataset: list}.

Every number, in "values" and in "differences", is one of two forms: a point, 0.81, or a point with an interval,
{"value": 0.81, "lo": 0.78, "hi": 0.84}. What the interval is (95%, a bootstrap over embryos, ...) is declared once,
on the metric (`B.metric(..., interval=...)`), so every method's intervals on it are of one kind. "differences"
gives this method's difference from another (named as that method's headway is named), per replicate, in the
replicates' order, or one pooled over them: the benchmark computes it, on the same samples for both, which is the
only way to its interval (two methods' own intervals overlap even when their difference is sure).

headway compares two methods on a metric by pairing their values by replicate id (`paired`): the mean of the
per-replicate differences, their spread, how many agree in sign, and, where the benchmark gave the difference's
intervals (in either method's results), how many of them exclude 0. A difference is "supported" when every
interval excludes 0 or, without intervals, when there are at least two paired replicates, all agreeing in sign.

A figure is what is looked at rather than measured: an image the benchmark makes (samples, reconstructions, a
diagnostic plot). The suite declares it like a metric (`B.figure`), and it is a column of the results beside the
metrics, each method's image in it, compared side by side; it is never ranked. If a judgment of it should count,
that is a metric of its own, with its values in "values"; the judgment's image can say so in its keys. An image:

    {"image": "figs/v5_even.png",      # written into the headway (relative to the results file), its path kept,
                                       #   for agents to open; or "url": a link, not written in
     "caption": "...", "replicate": "seed0", "keys": {"judged_by": "agent"}}     # all optional

or a list of them (one per replicate, say), the first shown in the table.

The results file is put into the method's built headway, together with the built suite, by `headway build
--results` or `headway attach`. Methods whose results are on the same suite (title and version) are compared: in the viewer's
Results view, or with `headway results`. The objectives are the one computation done here, from the stored
values; the viewer computes them the same way (viewer.html, "results"), so keep the two in step.

  how="mean"   the weighted mean, over the objective's metrics and datasets, of each metric's mean scaled to 0..1
               by its range and turned so that 1 is best; absolute (needs every metric's range)
  how="rank"   the weighted mean rank among the methods compared (1 is best); relative: it depends on which
               methods are compared
  and other ways of combining the metrics' rankings without comparing their values across metrics (HOWS:
  median, worst, wins, kemeny, pareto), each giving up a different one of the properties Arrow's theorem says no
  rule has all of. An objective combines metrics (each metric on each dataset ranks the methods: one "voter"), or
  categories of metrics (each category one voter, so a category of many metrics counts no more than a category
  of one: it ranks the methods by its metrics' mean rank, or with how="mean" scores them by its metrics' mean).
A method missing any value an objective needs gets no score for it ("incomplete").
"""
from __future__ import annotations

import math
import mimetypes
from base64 import b64encode
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from headway.spec import Headway, Symbol, SymbolLike, _caller, _refs


@dataclass
class Dataset:
    key: str
    name: str
    desc: str = ""
    source: str = ""                # where it comes from: a paper, an accession, a URL
    heldout: bool = False           # kept out of development: hidden in the viewer unless asked for
    values: dict[str, Any] = field(default_factory=dict)       # setting -> its value for this dataset
    facts: dict[str, str] = field(default_factory=dict)        # anything else worth knowing: size, platform, ...
    code: list[str] = field(default_factory=list)              # where it is made or loaded


@dataclass
class Metric:
    key: str
    better: str                     # "higher" or "lower"
    range: tuple[float, float] | None = None
    unit: str = ""
    source: str = ""                # where the metric comes from: a paper, a package
    interval: str = ""              # what its intervals are, if results give them: "95%, bootstrap over embryos"


@dataclass
class Figure:
    key: str
    name: str
    desc: str = ""
    source: str = ""
    code: list[str] = field(default_factory=list)              # where it is drawn


@dataclass
class Category:
    key: str
    name: str
    desc: str = ""
    metrics: dict[str, float] = field(default_factory=dict)     # metric -> weight within the category
    datasets: list[str] | None = None                           # only these datasets count for it (default: all)


@dataclass
class Objective:
    key: str
    name: str
    desc: str = ""
    weights: dict[str, float] = field(default_factory=dict)     # metric -> weight; empty with categories
    how: str = "mean"
    datasets: list[str] | None = None                           # default: every dataset not held out
    tex: str = ""
    categories: dict[str, float] | None = None                  # category -> weight: each category one voter


# how an objective combines its voters (metrics on datasets, or categories), and whether fewer is better. Only "mean"
# compares values across metrics (scaled by their ranges); the rest use only each voter's order of the methods.
HOWS = {
    "mean": ("higher", "the weighted mean of the metrics, each scaled by its range so that 1 is best (absolute: "
             "each method scored from its own values; needs every metric's range)"),
    "rank": ("lower", "the weighted mean rank (Borda), 1 is best; relative: adding or hiding a method can reorder "
             "two others (it gives up independence of irrelevant alternatives)"),
    "median": ("lower", "the weighted median rank, 1 is best: one metric far off does not move it (relative, like "
               "the mean rank)"),
    "worst": ("lower", "the worst rank on any voter, 1 is best: a method is as good as its weakest metric "
              "(egalitarian; relative)"),
    "wins": ("higher", "Copeland: how many other methods it beats on more voters (by weight) than it loses, a tie "
             "counting a half; a method that beats every other has the most (Condorcet-consistent; relative)"),
    "kemeny": ("lower", "the place in the Kemeny ranking: the order that disagrees least, summed over voters, with "
               "each voter's order (Condorcet-consistent; exact for up to 12 methods, else a local search)"),
    "pareto": ("lower", "the Pareto layer: 1 if no other method is at least as good on every voter and better on "
               "one, 2 if only layer 1 is, and so on. Methods in one layer are not ranked against each other (it "
               "gives up completeness, the price of avoiding Arrow's other compromises)"),
}


def lower_better(how: str) -> bool:
    return HOWS.get(how, ("higher", ""))[0] == "lower"


RELATION_KINDS = ("step", "pair", "alternatives", "factorial", "group")      # an atlas's relations (atlas.py)


@dataclass
class Contrast:
    key: str
    tex: str
    name: str
    desc: str = ""
    over: list[str] = field(default_factory=list)              # the relation kinds it applies to (atlas.py)
    equation: str = ""
    metrics: list[str] = field(default_factory=list)
    interval: str = ""


class Suite(Headway):
    """A benchmark: datasets, metrics (each a step of the headway, with its equation and code), figures (images to
    look at, not ranked) and objectives."""

    def __init__(self, title: str, desc: str = "", *, version: str = "1"):
        super().__init__(title, desc)
        self.version = version
        self.datasets: dict[str, Dataset] = {}
        self.metrics: dict[str, Metric] = {}
        self.figures: dict[str, Figure] = {}
        self.objectives: dict[str, Objective] = {}
        self.categories: dict[str, Category] = {}
        self.contrasts: dict[str, Contrast] = {}

    def dataset(self, key: str, name: str, desc: str = "", *, source: str = "", heldout: bool = False,
                values: Mapping[str, Any] | None = None, facts: Mapping[str, Any] | None = None,
                code: str | Iterable[str] = ()) -> Dataset:
        """A dataset the metrics are computed on. If the suite's steps make the data (a simulation, a
        preprocessing), `values` gives this dataset's value of each of their settings, so the datasets are
        annotated by the steps' equations once and differ in their values; `facts` is for the rest
        ({"cells": "12,000", "platform": "10x v3"}); `code` points to where the dataset is made or loaded."""
        assert key not in self.datasets, f"dataset {key!r} declared twice"
        self.datasets[key] = d = Dataset(key, name, desc, source, heldout, dict(values or {}),
                                         {k: str(v) for k, v in (facts or {}).items()}, _refs(code))
        self.declared["dataset:" + key] = _caller()
        return d

    def metric(self, key: str, tex: str, name: str, desc: str = "", *, equation: str = "",
               inputs: Sequence[SymbolLike] = (), better: str, range: tuple[float, float] | None = None,
               unit: str = "", source: str = "", interval: str = "", code: str | Iterable[str] = (),
               where: dict[str, str] | None = None, group: str = "metrics", was: str | Iterable[str] = (),
               category: str | Iterable[str] = ()) -> Symbol:
        """A metric: the output symbol `key` and the step `key` that computes it, by `equation`, from `inputs`.
        `interval` says what the intervals in its results are (results may give them only if it is set).
        `category`: the categories it belongs to (made if not declared with B.category, named by their key)."""
        assert better in ("higher", "lower"), f"metric {key!r}: better={better!r}, not 'higher' or 'lower'"
        s = self.symbol(key, tex, name, desc, role="output", was=was)
        self.op(key, name, equation, desc, inputs=inputs, outputs=[s], code=code, group=group, where=where, was=was)
        self.metrics[key] = Metric(key, better, None if range is None else (float(range[0]), float(range[1])),
                                   unit, source, interval)
        for c in [category] if isinstance(category, str) else category:
            self.categories.setdefault(c, Category(c, c)).metrics[key] = 1.0
        return s

    def category(self, key: str, name: str, desc: str = "", *, metrics: Mapping[str, float] | Sequence[str] = (),
                 datasets: Sequence[str] | None = None) -> Category:
        """A category of metrics (biological, technical, scalability, ...): metrics can be in several, a category
        can count only some datasets, and its metrics can be weighted within it. An objective over categories
        (`categories=`) takes each category as one voter: it ranks the methods by its metrics' mean rank (with
        how="mean", it scores them by its metrics' weighted mean)."""
        c = self.categories.setdefault(key, Category(key, name))
        c.name, c.desc = name, desc
        c.metrics.update(dict(metrics) if isinstance(metrics, Mapping) else {m: 1.0 for m in metrics})
        c.datasets = None if datasets is None else list(datasets)
        return c

    def figure(self, key: str, name: str, desc: str = "", *, source: str = "",
               code: str | Iterable[str] = ()) -> Figure:
        """A figure: an image the benchmark makes for each dataset, shown in a column beside the metrics, every
        method's side by side; looked at, never ranked. `code` points to where it is drawn."""
        assert key not in self.figures, f"figure {key!r} declared twice"
        self.figures[key] = f = Figure(key, name, desc, source, _refs(code))
        self.declared["figure:" + key] = _caller()
        return f

    def objective(self, key: str, name: str, desc: str = "", *, metrics: Mapping[str, float] | Sequence[str] = (),
                  categories: Mapping[str, float] | Sequence[str] | None = None, how: str = "mean",
                  datasets: Sequence[str] | None = None, tex: str = "") -> Objective:
        """One number per method from its metrics: `metrics` ({metric: weight} or a list), each metric on each
        dataset a voter; or `categories` ({category: weight} or a list), each category a voter. `how` (HOWS):
        mean, rank, median, worst, wins, kemeny, pareto."""
        weights = dict(metrics) if isinstance(metrics, Mapping) else {m: 1.0 for m in metrics}
        cats = None if categories is None else dict(categories) if isinstance(categories, Mapping) else \
            {c: 1.0 for c in categories}
        self.objectives[key] = o = Objective(key, name, desc, weights, how,
                                             None if datasets is None else list(datasets), tex, cats)
        return o


    def contrast(self, key: str, tex: str, name: str, desc: str = "", *, over: str | Sequence[str],
                 equation: str = "", metrics: Sequence[str] = (), interval: str = "") -> Contrast:
        """A number over a set of versions that only the benchmark can compute, because it needs the samples: an
        interaction in a factorial with its own interval, say. `over` names the relation kinds of an atlas it
        applies to (pair, step, alternatives, factorial, group); its values, given for a relation, have the shape
        of a metric's. What can be worked out from the versions' own values (a spread, a best-minus-base) is left
        to the viewer and needs no declaration."""
        assert key not in self.contrasts, f"contrast {key!r} declared twice"
        self.contrasts[key] = c = Contrast(key, tex, name, desc, [over] if isinstance(over, str) else list(over),
                                           equation, list(metrics), interval)
        return c


def check_suite(s: Suite, errors: list[str], warnings: list[str]) -> None:
    for m in s.metrics.values():
        if m.range and not m.range[0] < m.range[1]:
            errors.append(f"metric {m.key}: range {m.range} is empty")
    for k in s.figures:
        if k in s.metrics:
            errors.append(f"figure {k}: a metric has the same key")
    if not s.datasets:
        warnings.append("suite: no datasets declared")
    for d in s.datasets.values():
        for k in d.values:
            if k not in s.symbols:
                errors.append(f"dataset {d.key}: a value for {k!r}, which is not declared")
            elif s.symbols[k].role != "setting":
                errors.append(f"dataset {d.key}: a value for {k}, a {s.symbols[k].role}, not a setting")
    for k in {k for d in s.datasets.values() for k in d.values if k in s.symbols}:
        lacking = [d.key for d in s.datasets.values() if k not in d.values]
        if lacking and s.symbols[k].value is None:
            warnings.append(f"setting {k}: no value for dataset{'s' if len(lacking) > 1 else ''} "
                            f"{', '.join(lacking)}, and no default")
    for cat in s.categories.values():
        for k in cat.metrics:
            if k not in s.metrics:
                errors.append(f"category {cat.key}: undeclared metric {k!r}")
        for dk in cat.datasets or []:
            if dk not in s.datasets:
                errors.append(f"category {cat.key}: undeclared dataset {dk!r}")
        if not cat.metrics:
            errors.append(f"category {cat.key}: no metrics")
        for k, w in cat.metrics.items():
            if w <= 0:
                errors.append(f"category {cat.key}: weight of {k} is {w}; weights are positive")
    for o in s.objectives.values():
        if o.how not in HOWS:
            errors.append(f"objective {o.key}: how={o.how!r}, not one of {', '.join(HOWS)}")
        if not o.datasets and s.datasets and all(d.heldout for d in s.datasets.values()):
            errors.append(f"objective {o.key}: counts no dataset (every dataset is held out); give its datasets=")
        if o.categories is not None:
            if o.weights:
                errors.append(f"objective {o.key}: metrics or categories, not both")
            uses = o.datasets or [d.key for d in s.datasets.values() if not d.heldout]
            for k, w in o.categories.items():
                if k not in s.categories:
                    errors.append(f"objective {o.key}: undeclared category {k!r}")
                elif w <= 0:
                    errors.append(f"objective {o.key}: weight of category {k} is {w}; weights are positive")
                elif (only := s.categories[k].datasets) is not None and not set(only) & set(uses):
                    errors.append(f"objective {o.key}: category {k} counts none of its datasets "
                                  f"({', '.join(uses) or 'none'})")
            if not o.categories:
                errors.append(f"objective {o.key}: no categories")
            elif o.how == "mean":
                for k in {m for c in o.categories if c in s.categories for m in s.categories[c].metrics}:
                    if k in s.metrics and s.metrics[k].range is None:
                        errors.append(f"objective {o.key}: metric {k} has no range, which how='mean' needs to scale it")
        elif not o.weights:
            errors.append(f"objective {o.key}: no metrics (give metrics= or categories=)")
        for k, w in o.weights.items():
            if k not in s.metrics:
                errors.append(f"objective {o.key}: undeclared metric {k!r}")
            elif o.how == "mean" and s.metrics[k].range is None:
                errors.append(f"objective {o.key}: metric {k} has no range, which how='mean' needs to scale it")
            if w <= 0:
                errors.append(f"objective {o.key}: weight of {k} is {w}; weights are positive")
        for dk in o.datasets or []:
            if dk not in s.datasets:
                errors.append(f"objective {o.key}: undeclared dataset {dk!r}")
    for c in s.contrasts.values():
        for k in c.over:
            if k not in RELATION_KINDS:
                errors.append(f"contrast {c.key}: over={k!r}, not a kind of relation ({', '.join(RELATION_KINDS)})")
        for k in c.metrics:
            if k not in s.metrics:
                errors.append(f"contrast {c.key}: undeclared metric {k!r}")


def suite_data(s: Suite, code: Callable[[list[str]], list[dict[str, Any]]],
               decl: Callable[[str], dict[str, Any] | None]) -> dict[str, Any]:
    """What a built suite adds to a built headway (`code` and `decl` as page.data resolves them)."""
    return {"version": s.version,
            "datasets": [vars(d) | {"values": {k: str(v) for k, v in d.values.items()}, "code": code(d.code),
                                    "decl": decl("dataset:" + d.key)} for d in s.datasets.values()],
            "metrics": [vars(m) | {"range": None if m.range is None else list(m.range)} for m in s.metrics.values()],
            "figures": [vars(f) | {"code": code(f.code), "decl": decl("figure:" + f.key)} for f in s.figures.values()],
            "objectives": [_objective_data(s, o) for o in s.objectives.values()],
            "categories": [vars(c) for c in s.categories.values()],
            "contrasts": [vars(c) for c in s.contrasts.values()]}


def _objective_data(s: Suite, o: Objective) -> dict[str, Any]:
    """An objective as built: one over categories also lists its metrics (weight 1), for what shows them."""
    d = vars(o).copy()
    if o.categories:
        d["weights"] = {m: 1.0 for c in o.categories if c in s.categories for m in s.categories[c].metrics}
    return d


def suite_id(suite: Mapping[str, Any]) -> str:
    """Results are comparable when their suites have the same title and version."""
    return f"{suite['title']}@{suite['suite']['version']}"


IMAGES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
BIG_IMAGE = 5_000_000           # bytes; larger images are written in, with a warning


def keys(k: Any, where: str) -> dict[str, str | int | float | bool]:
    """Free-form labels to group or pair by: names -> strings, numbers or booleans."""
    assert isinstance(k, Mapping), f"{where}: keys are a mapping of names to values, not {type(k).__name__}"
    for name, v in k.items():
        assert isinstance(name, str) and isinstance(v, (str, int, float, bool)), \
            f"{where}: key {name!r}: {v!r} is not a string, number or boolean"
    return dict(k)


def entry(results: Mapping[str, Any], suite: dict[str, Any], base: Path | None = None) -> dict[str, Any]:
    """A results file, with its built suite, as kept in a method's headway; values that do not fit are reported.
    base: the directory that figures' images are relative to (the results file's)."""
    meta = suite["suite"]
    datasets = {d["key"]: d for d in meta["datasets"]}
    metrics = {m["key"]: m for m in meta["metrics"]}
    warnings: list[str] = []
    reps = _replicates(results.get("replicates"), datasets, warnings)
    values: dict[str, dict[str, list[float]]] = {}
    ids: dict[str, dict[str, list[str]]] = {}
    intervals: dict[str, dict[str, list[list[float] | None]]] = {}
    for dk, by_metric in results.get("values", {}).items():
        if dk not in datasets:
            warnings.append(f"dataset {dk!r} is not in the suite; left out")
            continue
        for mk, vals in by_metric.items():
            if mk not in metrics:
                warnings.append(f"metric {mk!r} is not in the suite; left out")
                continue
            vals = list(vals) if isinstance(vals, (list, tuple)) else [vals]
            named = reps.get(dk)
            if named is not None and len(named) != len(vals):
                warnings.append(f"{dk}/{mk}: {len(vals)} values but {len(named)} replicate ids; left unpaired")
                named = None
            points = [_point(v, f"{dk}/{mk}", metrics[mk], warnings) for v in vals]
            keep = [n for n, pt in enumerate(points) if pt is not None]
            clean = [points[n][0] for n in keep]          # type: ignore[index]
            if len(clean) < len(vals):
                warnings.append(f"{dk}/{mk}: {len(vals) - len(clean)} values not finite numbers; left out")
            rng = metrics[mk]["range"]
            if rng and any(not rng[0] <= v <= rng[1] for v in clean):
                warnings.append(f"{dk}/{mk}: values outside the metric's range {rng}")
            if clean:
                values.setdefault(dk, {})[mk] = clean
                if named is not None:
                    ids.setdefault(dk, {})[mk] = [named[n] for n in keep]
                if any(points[n][1] for n in keep):     # type: ignore[index]
                    intervals.setdefault(dk, {})[mk] = [points[n][1] for n in keep]     # type: ignore[index]
    figures = _figures(results.get("figures", {}), datasets, {f["key"] for f in meta.get("figures", [])}, reps,
                       base or Path.cwd(), warnings)
    differences = _differences(results.get("differences", {}), datasets, metrics, reps, warnings)
    return {"id": suite_id(suite), "suite": suite, "values": values, "run": results.get("run", {}),
            "warnings": warnings} | ({"replicates": ids} if ids else {}) | ({"intervals": intervals} if intervals else {}) | \
        ({"differences": differences} if differences else {}) | ({"figures": figures} if figures else {})


def _finite(y: Any) -> bool:
    return isinstance(y, (int, float)) and not isinstance(y, bool) and math.isfinite(y)


def _point(x: Any, where: str, metric: Mapping[str, Any], warnings: list[str]) -> tuple[float, list[float] | None] | None:
    """A number as given -> (value, [lo, hi] or None); None if it is not a finite number. An interval is kept only
    if the metric says what its intervals are and it holds the value."""
    if not isinstance(x, Mapping):
        return (float(x), None) if _finite(x) else None
    if not _finite(x.get("value")):
        return None
    v = float(x["value"])
    if "lo" not in x and "hi" not in x:
        return v, None
    if not (_finite(x.get("lo")) and _finite(x.get("hi"))):
        warnings.append(f"{where}: an interval needs finite value, lo and hi; interval left out")
        return v, None
    lo, hi = float(x["lo"]), float(x["hi"])
    if not metric.get("interval"):
        warnings.append(f"{where}: intervals given, but the metric does not say what they are "
                        f"(B.metric(..., interval=...)); intervals left out")
        return v, None
    if not lo <= v <= hi:
        warnings.append(f"{where}: interval [{lo}, {hi}] does not hold its value {v}; interval left out")
        return v, None
    return v, [lo, hi]


def _differences(given: Mapping[str, Any], datasets: Mapping[str, Any], metrics: Mapping[str, Any],
                 reps: Mapping[str, list[str]], warnings: list[str]) -> dict[str, Any]:
    """The differences, as given -> other method -> dataset -> metric -> {values, intervals, ids, pooled}."""
    out: dict[str, Any] = {}
    for other, by_dataset in given.items():
        for dk, by_metric in by_dataset.items():
            if dk not in datasets:
                warnings.append(f"differences from {other}: dataset {dk!r} is not in the suite; left out")
                continue
            for mk, vals in by_metric.items():
                where = f"differences from {other}, {dk}/{mk}"
                if mk not in metrics:
                    warnings.append(f"{where}: metric {mk!r} is not in the suite; left out")
                    continue
                pooled = not isinstance(vals, (list, tuple))
                vals = [vals] if pooled else list(vals)
                ids = None if pooled else reps.get(dk)
                if ids is not None and len(ids) != len(vals):
                    warnings.append(f"{where}: {len(vals)} differences but {len(ids)} replicate ids; left out")
                    continue
                points = [_point(v, where, metrics[mk], warnings) for v in vals]
                if any(pt is None for pt in points):
                    warnings.append(f"{where}: not all finite numbers; left out")
                    continue
                out.setdefault(str(other), {}).setdefault(dk, {})[mk] = {
                    "values": [pt[0] for pt in points],                 # type: ignore[index]
                    "intervals": [pt[1] for pt in points],              # type: ignore[index]
                    "ids": ids, "pooled": pooled}
    return out


def _replicates(r: Any, datasets: Mapping[str, Any], warnings: list[str]) -> dict[str, list[str]]:
    """The replicate ids, as given (a list, or {dataset: list}) -> dataset -> ids."""
    if r is None:
        return {}
    by = {dk: r for dk in datasets} if isinstance(r, (list, tuple)) else dict(r)
    out = {}
    for dk, names in by.items():
        if dk not in datasets:
            warnings.append(f"replicates: dataset {dk!r} is not in the suite; left out")
            continue
        names = [str(n) for n in names]
        if len(set(names)) < len(names):
            warnings.append(f"replicates of {dk}: ids repeat; left unpaired")
            continue
        out[dk] = names
    return out


def _figures(given: Mapping[str, Any], datasets: Mapping[str, Any], declared: set[str],
             reps: Mapping[str, list[str]], base: Path, warnings: list[str]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """The figures, as given -> dataset -> figure -> images, each written in as a data URL (its path kept)."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for dk, by_figure in given.items():
        if dk not in datasets:
            warnings.append(f"figures: dataset {dk!r} is not in the suite; left out")
            continue
        for fk, items in by_figure.items():
            if fk not in declared:
                warnings.append(f"figure {fk!r} is not in the suite; left out")
                continue
            one = isinstance(items, (Mapping, str))        # one image (a string: left out with a warning), or a list
            for n, item in enumerate([items] if one else items):
                image = _image(item, f"figure {fk} on {dk}" + (f", image {n + 1}" if not one
                                                                 else ""), reps.get(dk), base, warnings)
                if image:
                    out.setdefault(dk, {}).setdefault(fk, []).append(image)
    return out


def _image(item: Any, where: str, reps: list[str] | None, base: Path, warnings: list[str]) -> dict[str, Any] | None:
    if not isinstance(item, Mapping) or ("image" in item) == ("url" in item):
        warnings.append(f"{where}: needs an image or a url (not both); left out")
        return None
    e: dict[str, Any] = {"caption": str(item.get("caption", ""))}
    if item.get("replicate") is not None:
        e["replicate"] = str(item["replicate"])
        if reps is not None and e["replicate"] not in reps:
            warnings.append(f"{where}: replicate {e['replicate']!r} is not among the replicate ids")
    try:
        e["keys"] = keys(item.get("keys", {}), where)
    except AssertionError as err:
        warnings.append(f"{err}; keys left out")
        e["keys"] = {}
    if "url" in item:
        e["src"] = e["url"] = str(item["url"])
        return e
    path = (base / str(item["image"])).resolve()
    if path.suffix.lower() not in IMAGES or not path.is_file():
        warnings.append(f"{where}: {path} is not an image file ({', '.join(sorted(IMAGES))}); left out")
        return None
    data = path.read_bytes()
    if len(data) > BIG_IMAGE:
        warnings.append(f"{where}: {path.name} is {len(data) / 1e6:.1f} MB; written in all the same "
                        "(give a url to link it instead)")
    kind = "image/svg+xml" if path.suffix.lower() == ".svg" else mimetypes.guess_type(path.name)[0]
    e["src"] = f"data:{kind};base64,{b64encode(data).decode()}"
    e["path"] = str(path)
    return e


def attach(d: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    """The headway with these results in it, replacing any earlier results on the same suite."""
    d["results"] = [r for r in d.get("results", []) if r["id"] != results["id"]] + [results]
    return d


# ---------- objectives, for `headway results` (the viewer computes them the same way) ----------
def _mean(v: Sequence[float]) -> float:
    return sum(v) / len(v)


def voters(objective: Mapping[str, Any], meta: Mapping[str, Any]) -> list[tuple[float, list[tuple[str, str, float]]]]:
    """An objective's voters: (weight, its cells (dataset, metric, weight)). Without categories, each metric on each
    dataset; with them, each category, over its metrics and the datasets it and the objective count. A category the
    suite does not declare, or one that counts none of the objective's datasets (both check errors), is no voter."""
    datasets = objective.get("datasets") or [d["key"] for d in meta["datasets"] if not d["heldout"]]
    if objective.get("categories"):
        defs = {c["key"]: c for c in meta.get("categories", [])}
        out = []
        for ck, w in objective["categories"].items():
            c = defs.get(ck)
            if c is None:
                continue
            ds = [d for d in datasets if not c.get("datasets") or d in c["datasets"]]
            cells = [(dk, mk, mw) for mk, mw in c["metrics"].items() for dk in ds]
            if cells:
                out.append((w, cells))
        return out
    return [(w, [(dk, mk, 1.0)]) for mk, w in objective["weights"].items() for dk in datasets]


def _ranks(vals: Sequence[float]) -> list[float]:
    """1 + how many are larger, ties sharing (larger is better)."""
    return [1 + sum(o > v for o in vals) + (sum(o == v for o in vals) - 1) / 2 for v in vals]


def scores(entries: Sequence[Mapping[str, Any] | None], objective: Mapping[str, Any],
           suite: Mapping[str, Any]) -> list[float | None]:
    """One objective for each method (None: incomplete); entries are the methods' results on one suite. The
    viewer computes the same (viewer.html, objScores): keep the two in step."""
    meta = suite["suite"]
    metrics = {m["key"]: m for m in meta["metrics"]}
    vs = voters(objective, meta)
    cells = list(dict.fromkeys((dk, mk) for _, cs in vs for dk, mk, _ in cs))
    means = [{(dk, mk): _mean(e["values"][dk][mk]) for dk, mk in cells
              if e and e["values"].get(dk, {}).get(mk)} for e in entries]
    done = [n for n, m in enumerate(means) if len(m) == len(cells)]
    out: list[float | None] = [None] * len(entries)
    if not done or not cells:                               # no cells: the objective counts no dataset
        return out
    how = objective["how"]
    good = lambda n, dk, mk: means[n][(dk, mk)] * (1 if metrics[mk]["better"] == "higher" else -1)  # noqa: E731
    if how == "mean":
        for n in done:
            total = 0.0
            for w, cs in vs:
                part = 0.0
                for dk, mk, mw in cs:
                    lo, hi = metrics[mk]["range"]
                    x = (means[n][(dk, mk)] - lo) / (hi - lo)
                    part += mw * (x if metrics[mk]["better"] == "higher" else 1 - x)
                total += w * part / sum(mw for _, _, mw in cs)
            out[n] = total / sum(w for w, _ in vs)
        return out
    # each voter's value of each complete method (larger is better): its metric, or minus its cells' mean rank
    value: list[list[float]] = []
    for _, cs in vs:
        if len(cs) == 1:
            dk, mk, _ = cs[0]
            value.append([good(n, dk, mk) for n in done])
        else:
            rk = [_ranks([good(n, dk, mk) for n in done]) for dk, mk, _ in cs]
            tot = sum(mw for _, _, mw in cs)
            value.append([-sum(mw * rk[c][i] for c, (_, _, mw) in enumerate(cs)) / tot for i in range(len(done))])
    rank = [_ranks(v) for v in value]                       # rank[voter][i]
    ws = [w for w, _ in vs]
    k = len(done)
    if how == "rank":
        res = [sum(w * rank[j][i] for j, w in enumerate(ws)) / sum(ws) for i in range(k)]
    elif how == "median":
        res = [_wmedian([rank[j][i] for j in range(len(ws))], ws) for i in range(k)]
    elif how == "worst":
        res = [max(rank[j][i] for j in range(len(ws))) for i in range(k)]
    else:
        pref = [[sum(w for j, w in enumerate(ws) if value[j][a] > value[j][b]) for b in range(k)] for a in range(k)]
        if how == "wins":
            res = [sum(1.0 if pref[a][b] > pref[b][a] else 0.5 if pref[a][b] == pref[b][a] else 0.0
                       for b in range(k) if b != a) for a in range(k)]
        elif how == "kemeny":
            order = _kemeny(pref, [sum(w * rank[j][i] for j, w in enumerate(ws)) for i in range(k)])
            res = [float(order.index(i) + 1) for i in range(k)]
        else:                                               # pareto
            res = [float(x) for x in _layers(value)]
    for i, n in enumerate(done):
        out[n] = res[i]
    return out


def _wmedian(xs: Sequence[float], ws: Sequence[float]) -> float:
    pairs = sorted(zip(xs, ws))
    half, acc = sum(ws) / 2, 0.0
    for i, (x, w) in enumerate(pairs):
        acc += w
        if abs(acc - half) < 1e-12 and i + 1 < len(pairs):
            return (x + pairs[i + 1][0]) / 2
        if acc > half:
            return x
    return pairs[-1][0]


def _kemeny(pref: Sequence[Sequence[float]], borda: Sequence[float]) -> list[int]:
    """The order (best first) maximizing the weight of the voters' pairwise preferences it agrees with: exact
    (over subsets) for up to 12, else the mean-rank order improved by swapping neighbours while that helps."""
    k = len(pref)
    if k <= 12:
        best = [-1.0] * (1 << k)
        pick = [-1] * (1 << k)
        best[0] = 0.0
        for mask in range(1 << k):
            if best[mask] < 0:
                continue
            for x in range(k):
                if mask >> x & 1:
                    continue
                gain = sum(pref[x][y] for y in range(k) if y != x and not mask >> y & 1)
                nm = mask | 1 << x
                if best[mask] + gain > best[nm] + 1e-12:
                    best[nm], pick[nm] = best[mask] + gain, x
        order, mask = [], (1 << k) - 1
        while mask:
            order.append(pick[mask])
            mask &= ~(1 << pick[mask])
        return order[::-1]
    order = sorted(range(k), key=lambda i: (borda[i], i))
    changed = True
    while changed:
        changed = False
        for i in range(k - 1):
            a, b = order[i], order[i + 1]
            if pref[b][a] > pref[a][b] + 1e-12:
                order[i], order[i + 1], changed = b, a, True
    return order


def _layers(value: Sequence[Sequence[float]]) -> list[int]:
    """Each method's Pareto layer over the voters' values (larger is better)."""
    k = len(value[0])
    dom = lambda a, b: all(v[a] >= v[b] for v in value) and any(v[a] > v[b] for v in value)  # noqa: E731
    layer, left, n = [0] * k, set(range(k)), 1
    while left:
        front = {a for a in left if not any(dom(b, a) for b in left if b != a)}
        for a in front:
            layer[a] = n
        left -= front
        n += 1
    return layer


def paired(b: Mapping[str, Any], a: Mapping[str, Any], dk: str, mk: str, b_name: str, a_name: str) -> dict[str, Any] | None:
    """Method b's difference from method a on one metric and dataset (b and a: their results on one suite), as the
    viewer shows it under b's value (viewer.html, pairedDiff; keep the two in step). Paired by replicate id where
    both have ids in common, else the difference of the means (unpaired). The intervals are the benchmark's: b's
    differences from a, or a's from b turned around. None if either has no values."""
    vb, va = b["values"].get(dk, {}).get(mk), a["values"].get(dk, {}).get(mk)
    if not vb or not va:
        return None
    ib, ia = b.get("replicates", {}).get(dk, {}).get(mk), a.get("replicates", {}).get(dk, {}).get(mk)
    pos = {k: n for n, k in enumerate(ia or [])}
    diffs = [vb[n] - va[pos[k]] for n, k in enumerate(ib or []) if k in pos]
    d = _mean(diffs) if diffs else _mean(vb) - _mean(va)
    side = _sign(d)
    out: dict[str, Any] = {"d": d, "paired": bool(diffs), "n": len(diffs),
                           "sd": _sd(diffs) if len(diffs) > 1 else None,
                           "agree": sum(_sign(x) == side != 0 for x in diffs)}
    given = b.get("differences", {}).get(a_name, {}).get(dk, {}).get(mk)
    flip = a.get("differences", {}).get(b_name, {}).get(dk, {}).get(mk)
    ivs = [iv for iv in (given or flip or {}).get("intervals", []) if iv]
    if not given and flip:
        ivs = [[-hi, -lo] for lo, hi in ivs]
    out["intervals"] = ivs
    out["pooled"] = bool((given or flip or {}).get("pooled"))
    excl = sum(lo > 0 if side > 0 else hi < 0 if side < 0 else False for lo, hi in ivs)
    out["excluding"] = excl
    out["marker"] = "" if not ivs else "●" if excl == len(ivs) else "○" if excl == 0 else "◐"
    out["supported"] = out["marker"] == "●" or (not ivs and out["n"] > 1 and out["agree"] == out["n"])
    return out


def _sign(x: float) -> int:
    """-1, 0 or 1; a difference within rounding error of 0 is 0."""
    return 0 if abs(x) < 1e-12 else 1 if x > 0 else -1


SIG = 3


def num(x: float) -> str:
    """A number to SIG significant figures, as the viewer shows it: scientific notation below 0.001 and from
    10000 up; 1000-9999 as the whole number."""
    if x != x:
        return "–"
    if x in (float("inf"), float("-inf")):
        return "∞" if x > 0 else "−∞"
    a = abs(x)
    if a == 0:
        return "0"
    if 1e-3 <= a < 1e4:
        s = f"{x:#.{SIG}g}"
        return str(math.floor(x + 0.5)) if "e" in s else s.rstrip(".")      # rounds half up, as JS does
    m, e = f"{x:.{SIG - 1}e}".split("e")
    return f"{m}e{int(e)}"


def _paired_text(p: dict[str, Any] | None) -> str:
    if p is None:
        return "-"
    if abs(p["d"]) < 1e-12 and (p["sd"] or 0) < 1e-12:          # the same, up to rounding error
        return "=" + (" " + p["marker"] if p["marker"] else "")
    t = ("+" if p["d"] >= 0 else "") + num(p["d"])
    if p["paired"]:
        t += (f"±{num(p['sd'])}" if p["sd"] is not None else "") + f" {p['agree']}/{p['n']}"
    else:
        t += " unpaired"
    return t + (" " + p["marker"] if p["marker"] else "")


def table(headways: Sequence[Mapping[str, Any]], names: Sequence[str], heldout: bool = False) -> str:
    """The methods' results on their (first) common suite, as text: one row per method."""
    ids = [{r["id"] for r in g.get("results", [])} for g in headways]
    common = set.intersection(*ids) if ids else set()
    if not common:
        return "no suite that every method has results on"
    sid = next(r["id"] for r in headways[0]["results"] if r["id"] in common)
    entries = [next(r for r in g["results"] if r["id"] == sid) for g in headways]
    suite = entries[-1]["suite"]
    meta = suite["suite"]
    datasets = [d for d in meta["datasets"] if heldout or not d["heldout"]]
    cols = [(m, d) for m in meta["metrics"] for d in datasets]
    head = ["method"] + [o["key"] for o in meta["objectives"]] + \
        [f"{m['key']}{'↑' if m['better'] == 'higher' else '↓'}/{d['key']}" for m, d in cols]
    rows = [head]
    objs = [scores(entries, o, suite) for o in meta["objectives"]]
    for n, (e, name) in enumerate(zip(entries, names)):
        row = [name] + ["incomplete" if (x := s[n]) is None else num(x) for s in objs]
        for m, d in cols:
            v = e["values"].get(d["key"], {}).get(m["key"])
            row.append("-" if not v else num(_mean(v)) + (f"±{num(_sd(v))}" if len(v) > 1 else ""))
        rows.append(row)
    lines = [f"suite {sid}; objectives: " + "; ".join(f"{o['key']} ({o['how']}): {o['name']}" for o in meta["objectives"])]
    lines += _columns(rows)
    if len(entries) > 1:
        diff = [["difference"] + [f"{m['key']}/{d['key']}" for m, d in cols]]
        for n in range(1, len(entries)):
            diff.append([f"{names[n]} - {names[n - 1]}"] + [_paired_text(paired(
                entries[n], entries[n - 1], d["key"], m["key"], names[n], names[n - 1])) for m, d in cols])
        lines += ["", "differences from the method before: the mean of the per-replicate differences ± their s.d. and "
                  "how many replicates agree in sign (paired by replicate id); ● every interval of the difference "
                  "excludes 0, ◐ some do, ○ none (where the benchmark gave them)"] + _columns(diff)
    return "\n".join(lines)


def _columns(rows: list[list[str]]) -> list[str]:
    width = [max(len(r[k]) for r in rows) for k in range(len(rows[0]))]
    return ["  ".join(c.ljust(w) for c, w in zip(r, width)).rstrip() for r in rows]


def _sd(v: Sequence[float]) -> float:
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
