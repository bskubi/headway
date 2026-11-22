# Design: versions and the atlas

Status: implemented in headway 0.12.0 (`src/headway/atlas.py`, the explorer in `viewer.html`, tests in
`tests/test_atlas.py`; how to use it: the guide's "Atlases" section). Where the implementation departs from the
design below, or leaves part of it for later, the section says so in a note marked **Implemented:**. Not built
yet: content-addressed objects in pages (a page carries whole versions), `|` (or) in queries and the
`diff<`/`effect:`/`best:` terms, manual and baseline orientations, a hull drawing of a set, and saving a view from
the page (the page shows the command or the JSON for it instead).

## The problem

Today a version (the JSON from `headway build --json`, or the data inside a page) holds three kinds of things:

1. **The model**: its groups, steps, symbols, equations and code, plus the source it was built from (file and git
   commit).
2. **Its results**: one entry per suite, each with a full copy of the built suite and this version's values,
   replicates, intervals and figures.
3. **Things about other versions:**
   - `differences`, keyed by another version's name;
   - in a page, `lanes`, which are copies of other versions;
   - in the viewer, the lane order, the baseline and which versions are hidden. These live only in the browser.

The third kind is the problem. A relation between versions is stored inside one of them, or in a page that copies
them, or nowhere. So a version changes whenever we compare it with something new. A paired difference has an owner
for no good reason. Some things have no place at all: the "why / prediction / outcome" of a step in the ladder, a
set of alternatives, a factorial design.

## The principle

**Headway is for seeing and exploring models. It is not a provenance, verification or debugging tool.** When
there is a choice of how to treat something, headway gives the choice to the user, or more often to the AI agent
working for them. Whoever has a view on the best presentation can make it.

In practice:

- Headway reports what it notices: a result computed on an earlier build, two descriptions of one step that
  differ. It doesn't block, delete or decide which version is right.
- Every file is plain, documented JSON that an agent can read and edit directly. A command earns its place when it
  does part of that work deterministically, by an algorithm.
- **Atlases are self-contained.** None refers into another. Making one atlas from another (a subset, a merge, a
  reorganisation) is a conversion. An agent can do it by editing files. Headway provides commands only for the
  parts that are mechanical.

## The decision

**A version file describes one model and nothing else. Everything involving more than one version lives in an
atlas.** This holds for the graph view and for today's stacked lanes. The stacked view becomes one kind of view of
an atlas.

A version also stops holding its results. Results are separate records, each naming a version and a suite:

- **A version is a snapshot of one model.** Rerunning a benchmark, adding a suite or attaching a judged figure
  creates a new result, not a new version. Rebuilding a version replaces the snapshot (see
  [Rebuilding](#rebuilding)).
- **Results are facts about a version on a suite.** A version can have results on many suites, and on several
  versions of one suite.
- **Relations are facts about sets of versions.** These are paired differences, steps with their reasons, sets of
  alternatives and factorials.

Headway computes none of the numbers. The benchmark computes values, paired differences and set-level contrasts.
Headway checks them against the suite, stores them and displays them.

## On disk

An atlas is a folder:

```
atlas/
  atlas.json              the manifest: name, versions, suites, views
  versions/
    rung13.json           a version: one model (as `headway build --json` writes it, minus results)
    rung14a.json
  suites/
    methembed@3.json      a built suite, stored once and referred to by id
  results/
    rung13/methembed@3.json       one version on one suite
    rung14a/methembed@3.json
  relations.jsonl         one relation per line (merges well in git)
  relresults/
    r14a/methembed@3.json         a relation's results on one suite: paired differences, contrasts
  media/                  images from results, by content hash
  .cache/                 derived data: structural diffs, layouts (not in git)
```

A page (`.html`) contains a whole atlas, or part of one, together with the viewer. This is what `--compare`
produces today: the versions, results and relations in the view, so the page works on its own.

`headway build pkg.graphspec -o v5.html` with no atlas still works. It makes a page holding a one-version atlas.

### Identity

- **Version name**: the handle (`--name`), unique within an atlas. Every file in the atlas refers to a version by
  name. Renaming a version is a deterministic command (`headway atlas rename`) that updates every reference,
  including the ids of steps and pairs made from the members' names (`v3->v4`, `pair:v3|v4`) and the names and
  relation ids in views' queries (`near:`, `path:`, `in:`, `relation:`, bare names); `atlas merge --prefix`
  renames the same way.
- **Build ids**: two hashes, recorded on the version and on every result and relation result computed from it.
  They are information for display, not a lock:
  - the **model id** hashes what determines the numbers: the ops' code, the settings and constants, and the graph's
    structure;
  - the **page id** hashes everything, including descriptions, equations as written and documents.

  So the viewer can tell a result computed on the build now shown from one computed on an earlier build, and can
  tell whether the difference matters to the numbers. Leaving out the name, the build time and the `dirty` flag
  means rebuilding unchanged code gives the same ids.
- **Suite id**: `title@version`, as now, plus a content hash. The hash covers what makes results comparable: the
  suite's model id, the code that makes or loads each dataset, and its declarations (datasets, metrics,
  objectives, ...) without their wording, where each is declared or the headway release (a figure's code is left
  out: it changes no number), so a reworded description or a line added to the suite's module keeps it. If
  two suites have the same id but different hashes, both are kept, told apart by hash, with a warning that
  suggests bumping `version=`.
- **Metric id**: a hash of a metric's definition (equation, code, range, `better`, `interval`). This lets the
  viewer tell that a metric is unchanged across suite versions (see [Heterogeneous suites](#heterogeneous-suites)).

### `atlas.json`

```json
{"atlas": 1,
 "name": "methembed ladder",
 "versions": {
   "rung13":  {"file": "versions/rung13.json",  "model": "sha256:9f…", "page": "sha256:07…",
               "built_at": "2026-11-20T09:12:00"},
   "rung14a": {"file": "versions/rung14a.json", "model": "sha256:41…", "page": "sha256:e3…",
               "built_at": "2026-11-21T10:02:00"}},
 "suites": {"methembed@3": {"file": "suites/methembed@3.json", "hash": "sha256:c2…"}},
 "views": {
   "ladder": {"kind": "stack", "path": ["rung10", "rung11", "rung12", "rung13", "rung14c"],
              "baseline": "rung10", "suites": ["methembed@3"], "hidden": []},
   "rung14 alternatives": {"kind": "graph", "query": "relation:r14", "orient": "objective:overall",
                           "suites": ["methembed@3"]}}}
```

**Implemented:** a version's entry holds its `model` and `page` ids (and `history`, the ids of earlier builds
of its name), a suite's its content `hash` (`atlas.suite_hash`).

The manifest's `versions` is an index, not the source of truth. `headway atlas scan` rebuilds it from
`versions/`. Two files of one name (a copy not yet renamed) are reported, and only the one at the name's own file
name is kept.

### Rebuilding

Rebuilding means running `headway build --atlas atlas/ --name rung14a` again after the spec or code changed. The
change might be a bug fix, a prose edit, or a new commit with the same computation. Headway doesn't decide what the
rebuild means. The person or agent rebuilding says what they want, and the defaults cover the usual case:

- **Default: replace.** The new build takes the name's place. Results and relations stay attached and are shown as
  usual, and so do the parents the earlier build recorded, unless `--from` names others. A result computed on an earlier model id gets a quiet "earlier build" mark, which the viewer can show
  or hide. A change only to the page id gets no mark.
- **`--keep-old NAME`**: the old build stays as its own version under NAME, with its results. The new build gets a
  step `--from` it. The fix becomes visible in the graph.
- **`--drop-results`**: the old results are removed, for a fix that makes them meaningless and not worth showing.

Each of these is a deterministic operation on the files. An agent could do the same by editing them, and that is
equally valid.

### A version file

This is today's built JSON without `results`, `differences` or `lanes`, plus `provenance`:

```json
{"name": "rung14a", "spec": "methembed.graphspec", "groups": […], "ops": […], "symbols": […],
 "keys": {"weighting": "uniform"},
 "source": {"file": "…", "git": {"commit": "…", "branch": "…", "dirty": false}},
 "provenance": {"parents": [{"name": "rung13", "id": "sha256:9f…", "kind": "derived",
                             "why": "Directions weighted equally, as a baseline for 14b–h.",
                             "prediction": "About equal to rung13."}]},
 "built": "2026-11-21T10:02:00"}
```

These are written at build time with `headway build … --from rung13 --why "…" --prediction "…"`. A version can
have several parents, for example a merge of two branches.

### Results

A result file is today's results input, checked against its suite. It refers to the suite instead of copying it:

```json
{"version": "rung14a", "version_id": "sha256:41…", "suite": "methembed@3",
 "values": {"embryos": {"auroc": [0.91, 0.90, 0.93]}},
 "replicates": ["seed0", "seed1", "seed2"],
 "intervals": {…}, "figures": {…}, "run": {"date": "2026-11-20"}, "warnings": []}
```

Producing results stays the same for the benchmark. `headway attach atlas/ rung14a results.json` (or
`headway.build(..., atlas="atlas/", results=[...])`) checks the input, writes the file, and moves any
`differences` it contains onto the matching pair relations (see [Migration](#migration)).

### Relations

A relation is a set of versions, each with a role, plus authored data. `relations.jsonl` holds one per line:

```json
{"id": "r14", "kind": "step", "members": [{"v": "rung13", "role": "from"}, {"v": "rung14a", "role": "to"}],
 "label": "uniform weighting", "why": "…", "prediction": "…", "outcome": "As predicted: −0.002, 3/5.",
 "tags": ["weighting"]}
{"id": "a14", "kind": "alternatives", "members": [{"v": "rung13", "role": "base"}, {"v": "rung14a"}, {"v": "rung14b"},
 {"v": "rung14c"}, {"v": "rung14d"}], "label": "how the directions are weighted"}
{"id": "f1", "kind": "factorial", "factors": ["crossfit", "plate"],
 "members": [{"v": "a", "levels": {"crossfit": false, "plate": false}}, {"v": "b", "levels": {"crossfit": true, "plate": false}},
             {"v": "c", "levels": {"crossfit": false, "plate": true}},  {"v": "d", "levels": {"crossfit": true, "plate": true}}]}
{"id": "p7", "kind": "pair", "members": [{"v": "rung10"}, {"v": "rung14c"}], "label": "against the first rung"}
```

**Kinds.** Each kind fixes what its members' roles mean, which the views and orientations rely on:

| kind | members | meaning |
|---|---|---|
| `pair` | two, unordered | compared; no claim of derivation |
| `step` | `from`, `to` | `to` was made from `from` (provenance) |
| `alternatives` | optional `base`; others | different answers to one question |
| `factorial` | `levels` per member | a design over named factors; the members are its cells |
| `group` | any | a named set (a sweep, a submission round, "everything run on GPU") |

The set of kinds is closed but can grow. An unknown kind is shown as a `group`.

**Provenance is stored in both places.** A version's `provenance.parents` records what was said at build time.
Adding a version makes a `step` relation from each declared parent in the atlas, and to each version in it that
declares the new one a parent, so the order versions are added in does not matter; `headway atlas scan` does the
same for every declared parent in the atlas that lacks one (after files are edited by hand), copying `why` and
`prediction` from the version. The atlas adds what a version can't know at build time, such as the outcome, a
label and tags. It also holds steps between versions built without `--from`.

The atlas's relations are what the views show. The two copies may come to differ: a step's text edited afterwards,
or a step between versions that declare other parents. In that case `atlas check` lists the differences, and the
user or agent decides whether that matters. Neither copy overrides the other.

**Generated relations are not stored.** A view can ask for relations made from rules:

- `keys-differ-by-one`: an edge between versions differing in one key, labelled by that key, so a sweep becomes a
  grid;
- `git`: the commit graph among the versions;
- `nearest:k`: the k nearest versions by size of structural diff.

They are computed when the view opens. "Keep" turns one into a stored relation.

### Relation results

A benchmark computes results that belong to a set of versions on one suite. These are kept per relation and suite:

```json
{"relation": "p7", "suite": "methembed@3",
 "members": {"rung10": "sha256:…", "rung14c": "sha256:…"},
 "differences": [{"of": ["rung14c", "rung10"],
                  "values": {"embryos": {"auroc": {"values": [0.012, 0.010, 0.015], "intervals": [[0.004, 0.020], null, null],
                                                   "ids": ["seed0", "seed1", "seed2"], "pooled": false}}},
                  "models": {"rung14c": "sha256:…", "rung10": "sha256:…"}}],
 "contrasts": {},
 "contrast_models": {}}
```

**Implemented:** `differences` is a list, one entry per direction a benchmark gave (a − b and b − a may both
come), each stored as `bench.entry` checks it (values, intervals, replicate ids, pooled), whether it came with a
version's results or was attached to the relation (`headway attach ATLAS --relation ID`, which also takes
`"replicates"`).

- **`differences`** applies only to a pair, or to a step, where it is `to − from`. The value forms (a point, or
  a point with `lo` and `hi`, one per replicate or pooled) are today's. **The order is written as
  `"of": [a, b]`, meaning a − b.** The order is never implied by which file the values sit in. The viewer flips
  the sign when it shows the pair the other way round.
- **`contrasts`** holds results for sets of versions beyond a pair's differences. **The suite declares only what
  needs the samples.** That means a number with its own interval, which only the benchmark can compute. Anything
  computable from the versions' stored values is display arithmetic in the viewer and is never declared: the
  spread across alternatives, best minus base, a ranking. A declared contrast has a definition, a notation and a
  type of interval, like a metric:

  ```python
  B.contrast("interaction", r"\Delta_{AB}", "interaction", "How much more A helps with B than without it.",
             over=["factorial"], equation=r"(y_{11}-y_{01})-(y_{10}-y_{00})", metrics=["auroc"],
             interval="95%, bootstrap over embryos")
  ```

  A contrast value has exactly the shape of a metric value: `{"contrasts": {"interaction": {"embryos": {"auroc": [...]}}}}`.
  `over=` lists the relation kinds the contrast applies to. It tells the viewer where to draw the value: on a
  factorial's face, on a pair's edge, or in a generic table of the relation's results for anything else.
  **Headway doesn't check the set.** It doesn't verify, for example, that a factorial is complete. The benchmark
  computed the number, and headway shows it beside what the relation says it covers. A factorial missing a cell is
  drawn with that cell hatched, like a missing result.

Each difference's `models` (and each contrast's entry in `contrast_models`) records which builds its numbers were
computed on; `members` holds the members' model ids when the file was last written (the only record in files
written before the others were kept, and read in their place there). If a member is rebuilt, the result is kept,
with the same "earlier build" mark as a version's results.

Headway still pairs versions' values by replicate id itself (a mean difference, with a count of replicates agreeing in sign) when no relation
result exists. This is display arithmetic and is not stored.

### Views

A view is a saved way of looking at part of the atlas. There are two kinds:

- **`stack`**: today's lanes and Results table. It holds an ordered `path`, a `baseline`, `hidden`, `suites`,
  and column order. Each version is compared with the shown one before it, as now. This is exactly the meaning of
  a path through the graph.
- **`graph`**: a `query` picking the versions and relations, `orient` (see below), `generate` (rules for
  generated relations), `suites`, the node glyph, and a pinned layout (optional; it is otherwise computed and
  cached).

What the viewer keeps in the browser today (column order and visibility, one's own objectives) can be saved into a
view, which shares it. Unsaved changes stay in the browser, as now.

## Heterogeneous suites

An atlas holds every suite any of its versions has results on. Each view names the suites it compares on
(`"suites": [...]`). The view loads only results on those suites, and objectives come from those suites alone.
`headway results atlas/ --suite methembed@3` does the same on the command line.

Within one view, the viewer handles mismatches as it does now, and can do more over time:

- A version with no results on a selected suite is shown as having none (a hatched row), not left out.
- Two versions of one suite are kept apart by default. Because metric ids are hashed, the viewer can offer to
  compare metrics that are **unchanged** between suite versions, marking those columns "across suite versions".
  The suite author chooses nothing here: equal definitions are the evidence.
- Datasets match by key and dataset values (the settings that make the data). A dataset whose values changed is a
  different column.

Headway never transforms numbers to make them comparable. That is the benchmark's job: it bumps the suite version
and reruns.

## Orientation

Relations are stored without a direction (except a step's `from` and `to`, which is data). A view chooses its
orientation:

- **`provenance`**: from → to for steps, plus git ancestry where known.
- **`baseline`**: a relation's `base` → its other members; for a pair, the earlier build → the later one.
- **`objective:<key>`**: worse → better on that objective. An edge is directed only when the paired difference is
  supported, by the same test as the colouring in the Results table, and otherwise stays undirected. A rank
  objective is relative to the versions shown, so its orientation changes with the query. The view says so.
- **`manual`**: per-relation directions saved in the view (`"orient": {"manual": {"p7": "rung10"}}`), which
  point away from the named member.

## Queries

A view's subset is a query string. It is saved, readable, and can be typed into a search box:

```
key:weighting=uniform                    versions with that key
contains:op:crossfit                     versions with a step of that key (lineage-aware, via was=)
changed:op:score                         relations whose structural diff touches that step
relation:r14   kind:factorial   tag:weighting
near:rung13~2                            within two relations of rung13
path:rung10..rung14c                     a shortest path (or best:overall for the best one by an objective)
diff<5   effect:auroc>0.01               by structural diff size, by supported paired difference
text:"plate correction"                  names, descriptions, why/prediction/outcome
```

Terms combine with spaces (and), `|` (or) and `-` (not). The structural parts use `headway diff`'s lineages,
so renames followed with `was=` match.

**Implemented:** `key:`, `near:`, `path:`, `relation:`, `kind:`, `tag:`, `contains:STEP` and `changed:STEP` (a
step's key or label), `score>`/`score<` (the objective shown), `in:A,B,C`, `earlier:` (results on an earlier build)
and words; spaces and `-`. On the command line (`atlas extract`, `atlas view`, `results`), the ones that need no
results: `near:`, `path:`, `key:`, `relation:`, `kind:`, `tag:` and names.

## Display (sketch; the format above doesn't depend on it)

- **Nodes, at three levels of zoom:**
  - a dot coloured by the view's objective;
  - a strip of objective and metric cells, with the version's keys;
  - a thumbnail of the version's graph on a layout shared across the view, with its differences from a focused
    version highlighted.
- **Edges**: the diff summary (`+1 Δ3 −0`), the sign of the change on the objective, and thickness for diff size
  or effect size.
- **Relations of more than two versions:**
  - a small square joined to each member, or a hull for a small set;
  - for many overlapping sets, a membership matrix (relations × versions);
  - a factorial drawn as its grid, with the contrasts on its faces.
- **Collapsing**: versions with identical structure (same id apart from settings and keys) fold into one node with
  a settings axis.
- **Into the stack**:
  - a selected path opens as a `stack` view;
  - a relation opens its members with `base` as the baseline;
  - a factorial opens one row or column.

## Commands

```
headway atlas init atlas/
headway build pkg.graphspec --atlas atlas/ --name rung14a --from rung13 --why "…" --prediction "…"
headway attach atlas/ rung14a results.json          # checks it; differences go to pair relations
headway relate atlas/ step rung13 rung14a --outcome "…"
headway relate atlas/ alternatives --base rung13 rung14a rung14b rung14c --label "…"
headway atlas scan atlas/                           # manifest and step relations from versions/
headway atlas check atlas/                          # a report: earlier-build results, differing provenance, unknown suites
headway atlas rename atlas/ rung14a uniform          # every reference updated
headway diff atlas/ rung13 rung14a                  # as now, cached
headway results atlas/ --view ladder                # the table, as text
headway page atlas/ --view ladder -o ladder.html    # a page with what that view needs
headway serve atlas/
```

### Converting between atlases

Atlases are independent. Moving work between them is a conversion that copies files, and an agent can do it
directly. The commands do only the deterministic parts:

```
headway atlas extract SRC/ DST/ near:rung13~2           # a new atlas with the matching versions, their results,
                                                            # the relations among them, and the suites and media they use
headway atlas merge A/ B/ OUT/ [--prefix b_]                # versions with the same name and model id are one;
                                                            # same name, different model id: renamed by the prefix, or listed
                                                            # (a prefixed name that is taken already: refused)
headway atlas validate atlas/                               # the format is followed (for an agent's hand edits)
```

`merge` never resolves a conflict by judgment. It applies its stated rule or lists the cases and stops. An agent
decides what to do with that list.

## Migration

Nobody else uses the format, so this is a clean break with an importer:

- `headway atlas import atlas/ headways/*.json headways/*.html` takes each version, strips its results into
  `results/`, turns each `differences: {other: …}` into a `pair` relation (or the `step`, if one exists) with
  `"of": [this, other]`, and turns a page's lanes into a `stack` view.
- `headway build --compare a.json b.json` keeps working. It builds an atlas in memory and writes a page from it.
- The methembed ladder (`ladder_runs/eighth/headways_0.6.0/`) is the test case: rungs as versions, the "Why /
  Prediction" prose of each rung moved into its step, rung13 → 14a–h as an `alternatives` relation, and the main
  line as a `stack` view.

## Media

Images from results are copied into `media/`, named by content hash, so the atlas is self-contained: copying the
folder copies everything. Where the folder lives is the user's choice: in the repo, in git LFS, or gitignored.
Headway takes no position.

- **External links.** A result can still give `url` instead of `image`, as today, for files too big to copy in.
  `atlas check` notes links that no longer resolve.
- **Thumbnails.** These are made into `.cache/` for the graph view and the Results table. A page embeds
  thumbnails for every version it holds and full images only for the versions in its views.

## Size

Today a version's JSON is about 350–500 KB (the methembed ladder), and roughly 60% of that is `results`, mostly
the built suite copied into every version. Without results, and with the suite stored once, a version is about
200 KB. At hundreds of versions, a static page can't simply contain everything:

1. **Summaries.** The graph view needs only a small record per version: name, keys, ids, objective scores, the diff
   summaries on its edges, and a thumbnail. That's a few KB each. A version's full data is needed only when it
   opens as a lane.
2. **Content-addressed objects in pages.** Versions in one atlas share most of their ops and symbols. A page
   stores each object once, by hash, and each version lists its hashes. The page then grows roughly with the number of
   distinct objects, not with versions × objects. Version files in the atlas stay whole and readable, so an agent
   can open one without reassembling it.
3. **`headway serve`** loads full data on demand, so size matters only for static pages.

So `headway page` writes all summaries, plus the full data of the versions in its views, deduplicated by hash.
`--full` embeds every version in full.

## Open questions

None outstanding. Details to settle during implementation:

- the exact list of objects that go into the model id and the page id;
- the summary record's fields;
- the query grammar's precedence rules.
