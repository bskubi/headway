# headway

See how a method works, and how much headway each change to it makes. A spec module declares a method's indices,
symbols (with their dimensions), steps (with their equations) and groups, each pointing into the code. headway
checks it against the code, so the picture cannot quietly go stale, and draws it as an interactive computation
graph, with a legend of indices, symbols and equations and the code behind every step. Versions of the method
(commits, experiments, competing methods) open side by side, and their benchmark results show as a heatmap, so
you can see what each change gained.

The code can be in any language: the spec is Python, and a reference like `"R/score.R:score_cells"`,
`"src/lib.rs:Model.score"` or `"Snakefile:align"` finds the definition with tree-sitter (`"main.nf#process TRIM"`
anchors on text, and `"config.yaml:2"` names lines, for files it cannot parse). `examples/pipeline` is a headway
of a pipeline in Nextflow, Snakemake, bash, R and Rust.

**[GUIDE.md](https://github.com/bskubi/headway/blob/main/src/headway/GUIDE.md)** (also `headway guide`) is the full description: how headway works and how
to write a headway for a codebase. It is written to be handed to a coding agent working in another repository.

```
pip install headway                             # into the environment that imports the method
headway check my_pkg.graphspec                  # exit 1 if the spec and the code disagree
headway build my_pkg.graphspec -o headway.html  # one HTML file; open it in a browser
    [--collapse "round 0"] [--hide setup] [--json headway.json] [--no-verify] [--offline]
    [--name v3] [--compare v1.json --compare v2.html] [--results results.json]
headway serve [--open]                          # the library at http://127.0.0.1:8765/: open, compare, serve
headway serve my_pkg.graphspec [--open]         # the same, opened on the spec, rebuilt live
    [--compare old.json]
headway diff v1.json v2.json my_pkg.graphspec   # what changed from each version to the next
headway attach v3.json results.json             # put benchmark results into a built headway (or build --results)
headway results v1.json v2.json v3.json         # the versions' results on their common benchmark suite
headway guide                                   # the guide
```

From Python, e.g. at the end of a benchmark script in the method's own repository:

```python
import headway
headway.build("my_pkg.graphspec", "headways/v5.html", json="headways/v5.json", name="v5",
              results=[{"suite": "my_bench.suite", "values": values, "replicates": ["seed0", "seed1", "seed2"]}],
              compare=["headways/v4.json"])
```

`serve` watches the spec's package and every file its code references point to. On a save it rebuilds in a fresh
Python process (about a second, including the equation checks) and the open page takes the new graph without
losing its view: the selection, the zoom, the open groups. Steps that changed pulse, and a note says how many
changed. A spec that fails to build leaves the last good graph up, with the error over it. Each step, value and
group in the side panel links to its declaration in the spec ("✎ declared at …").

`headway serve` serves the **library**: one place to open headways from. It lists the headways opened recently
(in any directory, kept in `~/.local/state/headway/recent.json`) and what is in a folder: the specs (modules
that make a `Headway` or `Suite` at their top level; test files are skipped) and the built headways (pages and
`--json` files), with when each was built, its results and its keys, sortable by any column. It starts in the
directory the server was started in; browse to any other (the path bar, a folder's subfolders, **launch** (the folder it was
launched in), **home**, or type a path). **Open** a headway, tick several to open them side by side (the most recently built in
the page, the others in lanes beside it), or **Serve live** a spec, as `headway serve SPEC` does. Built files
open in the current viewer. Hidden directories, virtual environments, `node_modules`, `build` and `dist` are not
looked in.

With something open, **Open…** shows the same library over the page, to **Add beside** what is open (a lane of
its own) or **Open** something instead; **←** closes everything and goes back to the library. The address says
what is open (`/?open=v4.json&open=v3.json`), so a reload, a bookmark or the browser's Back button returns to it.
The server answers only requests to 127.0.0.1 or localhost.

## Editing from the page

The page stays a static view; edits happen in the spec, and the page takes you there.

- **✎ links** open the spec in your editor at the exact argument: the equation, the description, the inputs, the
  code references. They are in the side panel, and on the right of each legend row when you hover over it. For an
  object declared inside a helper function of the spec, the panel says so: editing the helper changes everything
  it declares.
- **`--editor`** (on `build` and `serve`, or `$HEADWAY_EDITOR`): `vscode` (the default), `cursor`, `pycharm`,
  `file`, or a URL template with `{file}`, `{line}` and `{col}`.
- **Ask an AI**, in the side panel, writes a prompt to paste into any assistant that can read the repository. It
  carries the object's declaration and location, its inputs and outputs, the code it points to and its equation
  check; with nothing selected, it is about the spec as a whole. It has two modes:
  - **Change the spec**: the request, with instructions to keep the references resolving, keep the `@G.verify`
    function in line with any changed equation, flag needed code changes, and run `headway check`. Preset requests
    fill in the box, worded for what is selected (a step, the steps around a symbol, a group, or the whole spec):
    **define local symbols** (put the symbols an equation uses but the graph does not define in `where=`), **add
    explanations** (a sentence or two in the spec on what each computation and output is, and what it is for),
    **check descriptions** (against the code), **reconcile with the code** (after the code changed), **standardize
    notation** (the same operation written the same way; fixed numbers as constants) and **disambiguate symbols**
    (one meaning per symbol across the spec).
  - **Explain**: a question, answered as a separate product; the assistant is told to change nothing, to take the
    code as the ground truth and to say which reasons are documented and which are its own reading. Presets: what is
    it for, read the equation, why this way, how it fits in, code vs. math, check my understanding. **Answer as**
    sets how elaborate the answer is: a quick answer, a full explanation, a tutoring session, a worked example,
    study notes to keep, or an interactive page (shown the way the assistant's environment shows pages, or as one
    HTML file).

  The mode and the answer format are remembered per browser; what is typed is kept per object and per mode.

The built page is a static file: open it from disk, send it, or put it on any web server. It loads KaTeX, elkjs and
highlight.js from CDNs; with `--offline` they are written into the file instead (about 3 MB), and the page needs no
network at all. `--offline` downloads them once, into `~/.cache/headway`. The ✎ links point to the spec on the
machine that built the page.

The viewer fills the window: the graph, the side panel on the right and
the legend below. Drag a divider to resize; drag it nearly shut, double-click it or use its arrow to collapse a
pane. The sizes are remembered per browser. Click a step or a value for its equation, inputs, outputs and
code (with a link that opens your editor at the line); double-click a group to collapse or expand it; show only what is
upstream or downstream of the selection; hide groups; show the settings as nodes. Steps done once per fold or per
cell are drawn as stacked cards; ✓ marks an equation checked against the code.

## A spec

```python
from headway import Headway
G = Headway("my method")
j = G.index("j", "j", "bin")
g = G.index("g", "g", "group")
m = G.symbol("m", "m_{jg}", "calls", dims=(j, g), role="input")
R = G.symbol("R", "R_{jg}", "residuals", dims=(j, g))
kr = G.setting("kr", r"\kappa_r", "pseudocounts", value=5, code="my_pkg.settings:Settings.pseudo")
G.op("residuals", "residuals", r"R_{jg} = \frac{m_{jg} + \kappa_r q}{n_{jg} + \kappa_r} - q",
     inputs=[m, kr], outputs=[R], group="model",
     code="my_pkg.model:residuals",                 # or "my_pkg.model:fit#step 5" / "#from text..to text"
     bind={"methylated": m, "return": R},           # for the shape trace
     where={"q": "the genome-wide fraction"})       # symbols used only in this step's equation

@G.verify("residuals")                              # the equation, written plainly, against the code
def residuals_as_written(): ...
```

`where` is for symbols an equation uses that are not values of the graph: a step's own intermediates, functions
like the logistic σ. The viewer shows them under the equation, as a paper would ("where q is …"). A value another
step also uses should be a symbol, listed in both steps' inputs or outputs, so it shows as a node and an edge.
headway does not read the TeX, so it cannot tell you which symbols lack a definition; it only checks that a
where-entry is not written exactly as a declared symbol the step should list instead.

## Comparing versions

A version is a built headway: the JSON from `build --json`, or a page from `build`. Keep one wherever it is useful
(at a tag, before an experiment, next to a set of results) and compare any of them, with each other or with the
spec as it is now. headway does not manage the versions or need git; it records the spec's file and, if it is in
a git repository, the commit, branch and whether the work tree had changes, and shows them with the version.

- **In the viewer**, `--compare` (on `build` and `serve`) or **Open…** → **Add beside** opens other versions, each
  in a lane of its own (served by `headway serve`, from the library; a page opened as a file uses the browser's
  file picker). The lanes share one layout, so a step sits in the same place in every lane: stacked one above
  another when the graph flows right, side by side when it flows down. A marker on each node says what changed
  against the lane before it (or against a lane pinned as the baseline): **+** added, **Δ** changed, **−** gone
  (drawn dashed where it was), **⋔** split or merged. The lanes start in the order the versions were built (a
  version opened later takes its place in that order); drag them to reorder, × to close one, any number of them.
  Opening a version that is open already (the same build, or one given the same `--name`) reloads it. **One at a
  time** draws only the version you are reading, with its markers, and keeps the others open: click another to
  switch. Click a lane to read that version in the side panel and the legend. Selecting an object selects it in every
  lane; the panel lists it version by version and shows what changed, with the equations one above the other
  and a diff of the code. Ask-AI prompts say which version they are about and what changed in it; Explain has
  **What changed?**. **Collapse unchanged** (in the **Groups** menu) leaves open only the groups where something
  changed.
- **`headway diff`** prints the same comparison as text, version after version (`--wording` for labels and
  descriptions too). A spec module is built as it is now.

Objects correspond by key. When a key changes, say so with `was=`, on symbols, settings, constants, steps and
groups:

```python
G.op("weighted_score", "module score", ..., was="score")                    # renamed
G.op("sizefactor", "size factors", ..., was="normalize")                    # split: both halves name
G.op("scale_counts", "scale counts", ..., was="normalize")                  # the step they came from
G.symbol("S", "S_i", "module score", ..., was=["score", "cell_score"])      # renamed twice
```

A merged object lists every key it replaces. `was` is matched across all the open versions at once, so a rename two
versions back still lines up; `headway check` rejects a `was` naming a key the spec still declares. What counts
as a change is read from the built versions, never from the TeX: inputs and outputs, group, dimensions (structure),
the equation's text, the code the references resolve to (so a code change under an unchanged equation shows), a
setting's value, the equation check, and the wording (labels, descriptions; hidden unless asked for). Code no
reference points to is not seen.

`examples/cellscore` is a small method developed over four versions (gene centering; size factors, splitting a
step; gene weights, renaming the score), with `build.sh` building each version and the comparison page.

## Benchmarks

A benchmark suite is declared like a method, as a `headway.Suite` (a Headway): each metric is a step, with its
notation, equation, description, code reference and `@verify` check, so it is annotated and checked the same way.
A suite adds its datasets (some can be held out), each metric's direction and range, and objectives, which combine
metrics into one number to rank by and are declared apart from them.

```python
B = Suite("my benchmark", version="1")
S = B.symbol("S", "S_i", "score", dims=(i,), role="input")
B.dataset("deep", "deep sequencing", "...", values={"depth_sd": 0.3}, code="my_bench.data:simulate")
B.dataset("atlas", "cell atlas", "...", source="GSE123456", facts={"cells": "12,000", "platform": "10x v3"},
          heldout=True, code="my_bench.data:load_atlas")
B.metric("auroc", r"\mathrm{AUROC}", "separation", "How often ...", equation=r"...", inputs=[S, y],
         better="higher", range=(0, 1), code="my_bench.metrics:auroc", source="Mann-Whitney U, scaled")
B.objective("overall", "overall", metrics={"auroc": 2, "depth_dep": 1})        # how="mean" (absolute) or "rank"
```

Datasets are annotated too. When the suite makes its data (a simulation, a preprocessing), declare those steps in
the suite like any other, with their equations, code and checks, and give each dataset its `values` of their
settings: the steps are written once, and the datasets differ in their values (`headway check` rejects a value
for anything but a declared setting). A dataset can also have a `source`, `facts` (anything else worth knowing)
and `code` (where it is made or loaded).

headway computes no metric. Your benchmark writes a results file, `{"suite": "my_bench.suite", "values":
{dataset: {metric: [one value per replicate]}}, "replicates": [ids], "figures": {...}, "run": {...}}`, and `build
--results` or `attach` puts it into a version's headway with the suite, built and checked. Versions with results on
the same suite (title and version) are compared, wherever their files are. `replicates` names the values (seeds,
folds) so that versions pair up. `figures` gives the images the benchmark made for each dataset and figure the
suite declares (`B.figure`); they are written into the headway and shown as columns beside the metrics, a
thumbnail per version, compared side by side when clicked: to look at, never ranked (a judgment of them is a
metric of its own). `--key NAME=VALUE` (or `keys=` in Python) labels a version with a hyperparameter, a variant or an option.
Any value can carry an interval, `{"value": 0.81, "lo": 0.78, "hi": 0.84}`, of a kind the metric declares
(`interval=`), and `differences` gives a version's difference from another with its own interval, computed by the
benchmark on the same samples for both. The viewer pairs versions by replicate: under each value, `+0.026 5/5 ●`
is the mean per-replicate difference, how many replicates agree, and whether the benchmark's intervals of the
difference exclude 0 (● all, ◐ some, ○ none); it is colored only when supported.

In the viewer, **Graph | Results** switches between the computation graph and the results of the same open
versions: a heatmap of versions × metrics × datasets (grouped by metric or by dataset), each value the mean over
replicates ± their s.d., shaded from worst to best among the versions (or across the metric's range), with its
difference from the version it is compared with. Missing results are hatched, never filled in; an objective missing
any value it needs is "incomplete". Click a column for what the metric or objective is (its equation, code, check
and source) and the replicates of every version, plotted; click it again to sort by it. Click a dataset (under
the table, or its column when grouped by dataset) for what it is: its settings, how it is made, its facts and code,
and every version's results on it. In the suite's own page, the legend lists the datasets, and a setting shows
its value in each. Click a version for what
changed in it, and to go to it in the graph. The Ask-AI prompts there carry the results, the metrics' definitions
and what changed between the versions: **Why do they differ?** and **What to try next?**. `headway results`
prints the same table as text. The objectives are the only computation headway does on results (bench.py; the
viewer computes them the same way).

## What is checked

| check | catches | where |
|---|---|---|
| references | a renamed, moved or deleted function (in any language tree-sitter parses); an anchor that is gone | `headway check` |
| structure | undeclared symbols or indices, two producers of one value, cycles, unused values | `headway check` |
| equations (`@G.verify`) | code that no longer computes the equation as written | `headway check`, marked ✓/✗ in the viewer |
| shapes (`headway.trace`) | arrays whose axes do not match the declared indices, an index with two sizes | your test suite |

```python
with headway.trace(G) as t:
    run_the_method_on_small_data()
t.assert_ok()
```

Not checked: that the graph's edges are all the data flow there is (a step could use a value the spec does not
list), or that an equation without a `@verify` matches its code.
