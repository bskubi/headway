# The headway guide

This guide is for people and for coding agents. It says what headway is, how to write a headway for a codebase,
how to keep it true as the code changes, and how to put benchmark results into it. It assumes nothing beyond this
file. `headway guide` prints it.

## What headway is

A **headway** describes a method (a computational or statistical procedure in some codebase) as a computation
graph: the values it works with, the steps that turn some values into others, each step's equation, and a
reference into the code that computes it. You write it as a Python module, the **spec**, kept next to the code.
headway then:

- **checks the spec against the code**: every code reference resolves; the graph is well formed; each equation you
  attach a check to reproduces what the code computes; on a real run, arrays have the dimensions the spec says.
  So the description cannot quietly go stale.
- **draws it** as one interactive HTML page: the graph (groups collapse and expand), a legend of indices, symbols
  and equations, and for each step its equation, description and the code itself, with links that open your
  editor at the line. It also writes prompts for an AI assistant about whatever is selected.
- **compares versions**: built versions of a headway (one per commit, experiment or competing method) open side
  by side in lanes, registered on one layout, with markers for what was added, changed or removed.
- **shows benchmark results**: a benchmark suite is declared the same way (its metrics are steps with equations
  and checks; its datasets are annotated), and each version's results are stored in its headway. The viewer shows
  a heatmap of versions × metrics × datasets, so you can see what each change gained or lost.

What headway does **not** do: it does not run your method, compute metrics, manage git or experiments, or read
equations. Equations are TeX strings shown as written; headway never parses them. Whether an equation matches the
code is established only by a check function you write (`@G.verify`).

## Installing it in another codebase

headway is a Python package (Python 3.12+, numpy). Install it into the environment that can import the method:

```
pip install headway                     # or: uv pip install headway
pip install -e /path/to/headway         # a local checkout, to work on headway itself
headway guide                           # this file
```

The spec imports headway; the method's own code never imports the spec or headway. A suggested layout:

```
mypkg/                     the method
mypkg/graphspec.py         the spec: one Headway object, conventionally named G
bench/mybench/suite.py     a benchmark suite (optional): one headway.Suite, conventionally named B
bench/run.py               runs the method on the benchmark and builds a headway with the results
                           (run with bench/ on PYTHONPATH, so that `mybench.suite` imports)
headways/                  built versions: v1.json, v1.html, ...
tests/test_headway.py      the check and the shape trace, in the test suite
```

## The pieces of a spec

```python
from headway import Headway
G = Headway("my method", "One sentence on what it computes, from what.")
```

**Indices** name the axes of arrays: `G.index(key, tex, name, desc="", *, varies=False)`.
`i = G.index("i", "i", "cell")`. `varies=True` if the axis may have different sizes in different calls (a subset).

**Symbols** are the values: data, intermediate results, outputs, settings and constants.
`G.symbol(key, tex, name, desc="", *, dims=(), per=None, role="value", value=None, code=(), was=())`

- `dims`: the indices, in order (`dims=(i, g)` is an i × g array; `()` a scalar).
- `role`: `"input"` (data given to the method), `"value"` (computed inside), `"output"` (a result of the method),
  `"setting"` (configurable), `"constant"` (a number fixed in the code).
- `per`: an index the symbol has one copy of per value of. The copies themselves have `dims` without it: one
  model per fold is `dims=(g,), per=fold`; one error per candidate penalty, computed by one call each, is
  `dims=(), per=a`, not `dims=(a,)` (the shape trace checks each call's value against `dims`).
- `code`: optional. For an input, where it is read, loaded or defined, if there is such a place; for a setting,
  its config field; for a constant, the line that sets it (a constant without one is a warning).
- Shortcuts: `G.setting(key, tex, name, desc="", *, value=None, code=(), was=())` (point `code` at the config
  field, e.g. `"mypkg.config:Settings.alpha"`) and `G.constant(key, tex, name, desc="", *, value, code=(), was=())`.

**Steps** (ops) turn input symbols into output symbols:

```python
G.op(key, label, tex="", desc="", *, inputs=(), outputs=(), code=(), group="", per=None, bind=None, where=None,
     was=())
```

- `tex`: the equation, as the code computes it (not as a paper states it, if they differ). Display TeX, KaTeX
  dialect (`\begin{aligned}` works).
- `desc`: one or two sentences: what the step does and why it is there. `$...$` in any description is TeX.
- `inputs`, `outputs`: symbols (objects or keys). Each symbol is produced by at most one step.
- `code`: one reference or a list (see "Code references").
- `group`: a path, `"round 1/model"`; groups nest and collapse in the viewer. `G.group(path, label=None, desc="",
  *, collapsed=False, was=())` gives one a label, a description or a collapsed start; it is optional.
- `per`: the step is done once per value of this index (drawn as stacked cards); its per-value outputs are
  symbols with the same `per`.
- `where`: `{TeX: definition}` for symbols the equation uses that are not values of the graph (a step's own
  intermediates, functions like σ). Shown under the equation, as a paper would. A value two steps share should be
  a symbol listed in both instead.
- `bind`: for the shape trace, `{code name: symbol}`, naming the parameters of the function in the step's first
  code reference, and `"return"`, `"return.0"` (a tuple's first item) or `"return.attr"` (an attribute of the
  result). Optional; see "Shape trace" for which steps can be bound.

**Equation checks**: a function that computes the equation as written, plainly (numpy on small made-up data),
and asserts the code gives the same. Registered for one or more steps:

```python
@G.verify("center", "score")
def as_written() -> None:
    X = np.random.default_rng(0).normal(size=(6, 4))
    expected = (X - X.mean(0)).sum(1)                  # the equations, written out
    assert np.allclose(mypkg.core.score(X), expected)  # the code
```

A step whose check passes is marked ✓ in the viewer; one that fails, ✗ (and `headway check` fails).

**Shape trace**, in your tests: run the method on small data with the bound functions wrapped; every call's
bound values must have as many axes as their symbols have `dims`, and each index one size (within a call, and
across calls unless the index `varies`). How it works, and so what to bind:

- The trace wraps the function named by a bound step's **first** code reference (an anchor is ignored: the whole
  function is wrapped). So bind a step only when its first reference is a function that computes that step. A
  step whose code is a block inside a larger function cannot be traced; leave it unbound. Unbound steps are not
  traced and nothing warns.
- A function several steps bind (the same helper used in two places): each call is given to a step whose bind
  fits it, in order. A function called on subsets of the rows (inside cross-validation folds, say) sees an index
  at several sizes: declare that index `varies=True`, or leave that step unbound if the looser check is not worth it.
- Every bound step must be called at least once by the traced run.

The test file, in full:

```python
import headway
import mypkg
from mypkg import graphspec

def test_spec_matches_code():
    report = headway.check(graphspec.G)
    assert report.ok, report.text()

def test_shapes():
    with headway.trace(graphspec.G) as t:
        mypkg.run(small_data())                        # the method's entry point, on a few rows
    t.assert_ok()
```

## Code references

```
"pkg.mod:func"                      the whole function (or class, method or property: "pkg.mod:Class.method")
"pkg.mod:Class.field"               one line of a class body (a dataclass field)
"pkg.mod:CONSTANT"                  a module-level assignment, the whole statement
"pkg.mod:func#text"                 from the line holding `text` to the line before the next comment line at the
                                    same or a smaller indentation, or to the end of the enclosing block if there
                                    is none (so "#step 2" takes the block under "# step 2: ...")
"pkg.mod:func#from text..to text"   from the line holding `from text` to the next line holding `to text`, inclusive
"pkg.mod:CONSTANT#text..text"       the same, within a module-level statement (one entry of a dict)
```

The anchor text must occur on exactly one line. Anchor on something that would change if the step changed: for a
constant, the literal value (`"pkg.mod:fit#np.clip(p, 1e-3..np.clip(p, 1e-3"`), so changing it in the code
fails the check until the spec is updated.

## What `headway check` reports

Errors (exit 1): a reference that does not resolve; an undeclared symbol or index; a symbol produced by two steps;
a step producing an input, setting or constant; a value or output no step produces; a cycle; a `where` entry
whose TeX is exactly a declared symbol's TeX (list the symbol instead; the comparison is of whole strings, so an
entry defining several symbols at once, `"n_1, n_0"`, is fine); a failed `@verify`; a `was` naming a key the spec
still declares; and, in a suite, the errors listed under "Benchmarks". Warnings: a symbol no step uses (outputs
excepted), a step with no code reference, a constant with no code reference, a group with no steps.

Not checked: that the graph lists all the data flow there is, and that a step without a `@verify` matches its
code. Say in a step's description when its equation is a simplification of the code.

## Writing a headway for a codebase: the procedure

1. **Read the method end to end.** Find the entry point and follow the data from the inputs to the outputs. Read
   the docs and the code; where they disagree, the code is the ground truth, and the spec describes the code.
2. **Choose the steps.** One step per meaningful mathematical operation: usually a function, or a commented block
   inside a long one. Glue (I/O, logging, reshaping) is not a step unless it changes the meaning of the data. A
   method of moderate size has 8–40 steps; group them by stage (`"preprocessing"`, `"model"`, `"round 1/fit"`).
3. **Declare the indices and the symbols**, with notation that follows the project's paper or docs if there is
   one. One meaning per symbol across the spec; the same operation written the same way everywhere. Every setting
   points to its config field; every number fixed in the code that matters is a constant pointing to its line.
4. **Declare each step**: its equation as the code computes it, a description of what it does and why, its inputs
   and outputs, its code references, `where` for its local symbols, and `bind` if you will trace shapes.
5. **Write `@verify` checks** for the steps that carry the math (at least the ones whose equations someone will
   rely on). Small random data; the equation written out plainly; compared with the code's function.
6. **Run `headway check pkg.graphspec`** until there are no errors, and fix the warnings or know why they stay.
7. **Add the test** (`headway.check(G).ok` and a shape trace) so the spec breaks loudly when the code changes.
8. **Build and look**: `headway build pkg.graphspec -o headway.html`, or `headway serve pkg.graphspec --open` to
   rebuild live while editing. Read the graph as a newcomer would: does every step say what it is for?

When the code changes later, rerun `headway check`; it names what went stale. Update the spec in the same change
as the code.

## Versions: seeing what changed

A **version** is a built headway: the JSON from `headway build --json`, or the HTML page itself. Keep one wherever
it is useful (each commit you care about, each experiment, each competing method), with `--name` to label it.
headway records the spec's file, the build time, and, in a git repository, the commit, branch and whether the work
tree had changes.

```
headway build pkg.graphspec --name v5 -o headways/v5.html --json headways/v5.json \
    --compare headways/v3.json --compare headways/v4.json        # v3 and v4 in lanes beside v5
headway diff headways/v4.json headways/v5.json                   # the same comparison, as text
```

In the viewer, **Compare…** opens more versions. A version built without `--name` is called by its file name; lanes are ordered by build time and can be dragged, closed or
pinned as the baseline. Markers: **+** added, **Δ** changed, **−** gone, **⋔** split or merged. **One at a time**
shows one version, keeping the others open.

Objects correspond across versions by key. When you rename a key, say so in the new version with `was=`:

```python
G.op("weighted_score", "module score", ..., was="score")              # renamed
G.op("sizefactor", ..., was="normalize")                              # split: each part names
G.op("scale_counts", ..., was="normalize")                            # the step it came from
G.symbol("S", ..., was=["score", "cell_score"])                       # renamed twice, or two merged into one
```

What counts as a change: inputs, outputs, group, dimensions or role (structure); the equation text; the code the
references resolve to (so a code change under an unchanged equation shows); a setting's value; the check result;
and labels and descriptions (shown only with "wording" / `--wording`).

## Benchmarks

A benchmark suite is a `headway.Suite`, which is a Headway: you declare it the same way, and it adds datasets,
metrics and objectives.

```python
from headway import Suite
B = Suite("my benchmark", "What it measures.", version="1")
i = B.index("i", "i", "cell")
S = B.symbol("S", "S_i", "score", "The method's output.", dims=(i,), role="input")
y = B.symbol("y", "y_i", "truth", dims=(i,), role="input")

B.metric("auroc", r"\mathrm{AUROC}", "separation", "How often a positive scores above a negative.",
         equation=r"\mathrm{AUROC} = \frac{1}{n_1 n_0} \sum_{i: y_i=1} \sum_{j: y_j=0} [S_i > S_j]",
         inputs=[S, y], better="higher", range=(0, 1), code="mybench.metrics:auroc",
         where={"n_1, n_0": "the numbers of positives and negatives"}, source="Mann-Whitney U, scaled")

B.dataset("pbmc", "PBMC", "Blood cells with sorted labels.", source="GEO GSE123456",
          facts={"cells": "12,000", "platform": "10x v3"}, code="mybench.data:load_pbmc")
B.dataset("check", "held-out mix", "Not looked at during development.", heldout=True,
          code="mybench.data:load_check")

B.objective("overall", "overall", "Separation counts twice.", metrics={"auroc": 2, "stability": 1})
B.objective("rank", "mean rank", metrics=["auroc", "stability"], how="rank")

@B.verify("auroc")
def auroc_as_written(): ...
```

**How a suite meets the method.** A suite describes the benchmark, not the method. Its `role="input"` symbols are
what the method hands to the metrics (its outputs: scores, predictions, coefficients); write them with the same
notation as the method's spec, since a reader goes between the two (nothing links them automatically). If the
suite also makes the data the method runs on, declare a step for the method itself in the suite: inputs the data,
outputs the method's results, `code` pointing to the method's entry function, with a description saying it is
the method being benchmarked. The graph then reads data → method → metrics, and no data symbol is left unused.

- **`B.metric(key, tex, name, desc="", *, equation="", inputs=(), better, range=None, unit="", source="",
  code=(), where=None, group="metrics", was=())`** declares the output symbol `key` and the step `key` that
  computes it, so a metric has a notation, an equation, code and a check like any step. `better` is `"higher"` or
  `"lower"`; `range` is its possible extent (needed by `how="mean"` objectives). For a metric with no natural
  bound (a mean squared error), either use a bounded version of it (R², or the error relative to a baseline's) or
  leave out `range` and combine it only in `how="rank"` objectives; a guessed range warns when values fall outside.
- **`B.dataset(key, name, desc="", *, source="", heldout=False, values=None, facts=None, code=())`**. If the suite
  makes its data (a simulation, a preprocessing), declare those steps in the suite like any other (with their
  settings, equations and checks) and give each dataset its `values` of those settings: the steps are written
  once and the datasets differ in their values. `values` may only name declared settings. Keep one copy of the
  values: either the benchmark code holds them and the suite reads them (`values=data.DATASETS["pbmc"]`), or the
  suite holds them and the benchmark code reads them back (`B.datasets["pbmc"].values`, a dict). Replicates (seeds,
  folds) are not dataset values; record them in the results' `run`. `facts` is anything
  else worth knowing; `source`, where the data come from; `code`, where they are made or loaded. A held-out
  dataset is hidden in the viewer unless asked for.
- **`B.objective(key, name, desc="", *, metrics, how="mean", datasets=None, tex="")`** combines metrics into one
  number, kept apart from the metrics. `metrics` is `{metric: weight}` or a list (weight 1). `how="mean"`: each
  metric's mean scaled to 0..1 by its range, turned so 1 is best, weighted and averaged over the datasets
  (absolute). `how="rank"`: the weighted mean rank among the versions shown, 1 best (relative). `datasets`:
  default every dataset not held out. A version missing any value an objective needs gets no score ("incomplete").

Suite errors: an empty range, an objective naming an undeclared metric or dataset, a `how="mean"` objective over a
metric without a range, a weight that is not positive, a dataset value for anything but a declared setting.

### Results

headway computes no metrics. Your benchmark runs the method, computes the metrics, and hands headway the values:
one per replicate (seed, fold), for each dataset and metric it ran. Missing combinations are fine (shown hatched).

```json
{"suite": "mybench.suite",
 "values": {"pbmc": {"auroc": [0.91, 0.90, 0.93], "stability": [0.8, 0.82, 0.79]}},
 "run": {"seeds": [0, 1, 2], "date": "2026-10-27"}}
```

`suite` names the module holding the Suite (or `"module:attribute"`); in Python it can be the Suite itself. `run`
is anything worth keeping. Results are stored in the version's headway together with the built suite; versions
are compared on results with the same suite title and version, so bump `version` when a change to the suite makes
old results incomparable. Values for a dataset or metric the suite does not declare are left out with a warning.

From the command line:

```
headway build pkg.graphspec --name v5 --results results/v5.json -o headways/v5.html --json headways/v5.json
headway attach headways/v4.json results/v4.json                # add results to a built version (JSON or page)
headway results headways/v*.json                               # the table, as text
```

From Python, in the codebase's own benchmark script (the usual way: run, then build, in one go):

```python
import headway
from mybench import suite, metrics, data

values = {}
for name in ("pbmc", "check"):
    for seed in range(3):
        d = data.load(name, seed)
        s = mypkg.score(d.counts)
        values.setdefault(name, {}).setdefault("auroc", []).append(metrics.auroc(s, d.truth))

headway.build("mypkg.graphspec", "headways/v5.html", json="headways/v5.json", name="v5",
              results=[{"suite": suite.B, "values": values, "run": {"seeds": [0, 1, 2]}}],
              compare=["headways/v4.json"])
```

In the viewer, **Graph | Results** switches to the heatmap of the open versions: objectives first, then metric ×
dataset columns (group by metric or by dataset), each cell the mean ± s.d. of the replicates and its difference
from the version it is compared with. Click a column for the metric's definition and every version's replicates;
click a dataset for what it is; click a version for what changed in it.

## Command line

```
headway check SPEC [--no-verify]                  exit 1 if the spec and the code disagree
headway build SPEC [-o headway.html] [--json FILE] [--name NAME] [--results FILE ...] [--compare [NAME=]FILE ...]
              [--collapse GROUP ...] [--hide GROUP ...] [--no-verify] [--offline] [--editor EDITOR]
headway serve SPEC [--port 8765] [--open] [--compare ...] [--collapse ...] [--hide ...] [--editor EDITOR]
headway diff VERSION VERSION ... [--wording]      a VERSION is a built file, or a spec module (built as it is now)
headway attach VERSION RESULTS.json [-o OUT]
headway results VERSION VERSION ... [--heldout]
headway guide
```

SPEC is an importable module holding one Headway (or `module:attribute`); run from a directory where it imports.
`--offline` writes the page's libraries into it (about 3 MB) so it opens without internet. `--editor` (or
`$HEADWAY_EDITOR`): `vscode` (default), `cursor`, `pycharm`, `file`, or a URL template with `{file}`, `{line}`,
`{col}`. `serve` rebuilds when the spec or any file its references point to changes, and the open page keeps its
view.

## Python API

```python
import headway
headway.Headway, headway.Suite                    # declaring
headway.check(G) -> Report                        # .ok, .errors, .warnings, .text()
headway.trace(G)                                  # context manager; .assert_ok()
headway.build(spec, out="headway.html", *, json=None, name=None, results=(), compare=(), collapse=(), hide=(),
              verify=True, offline=False, editor=None, strict=False, log=None) -> dict
headway.attach(path, results, out=None, log=None) -> dict
headway.read_results(results, log=None) -> dict   # a results dict or file -> the entry stored in a headway
headway.load("pkg.mod[:attr]") -> Headway
```

`spec` is a module name or a Headway object; `out` or `json` may be None to skip that file; `results` items are
results dicts or paths to results files; `strict=True` raises if the check finds errors (otherwise they are in
the page); `log` receives progress messages (e.g. `log=print`).

## Rules of thumb

- The spec describes the code as it is. If the code and a paper differ, write the code's equation and say so in
  the description.
- Keep the spec and the code in the same change; run `headway check` in the tests.
- Prefer a symbol over a `where` entry when a value is used by more than one step.
- Name things for a reader new to the method: labels are short and plain ("size factors", not "sf"); descriptions
  say what and why, not how the code is organized.
- Do not compute metrics, or anything else, in headway; it shows precomputed results against the methods.
- Do not make headway parse TeX; equations are checked only by `@verify` functions.
