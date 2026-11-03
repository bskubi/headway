# The headway guide

This guide is for people and for coding agents. It says what headway is, how to write a headway for a codebase,
how to keep it true as the code changes, and how to put benchmark results into it. It assumes nothing beyond this
file. `headway guide` prints it.

## What headway is

A **headway** describes a method (a computational or statistical procedure in some codebase) as a computation
graph: the values it works with, the steps that turn some values into others, each step's equation, and a
reference into the code that computes it, in Python or another language. You write it as a Python module, the
**spec**, kept next to the code.
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

headway is a Python package (Python 3.11+, numpy). Install it into the environment that can import the method:

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

`Headway(..., hints=True)` takes the code's type hints as assumptions where the code is written as math (see
"The code written as math"); `headway init --hints` sets it.

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
- `condition`: when the step is done, if not always: a Python expression whose names are symbol keys,
  `"correct"` (a setting that is True), `"not correct"`, `"mode == 'exact'"`. The step is drawn dashed, with
  "if"; each symbol the condition names has a dotted edge to it (it decides whether the step is done, rather
  than being an input); its math says when ("... only if κ"), and what each of those symbols is (its description).
- `relations`: what the step's code declares it computes, with `headway.clean` (see "Clean code: what it
  computes, and why"): `[{"kind": "maximizes", "target": "pkg.mod:objective", "over": ["b", "d"], "subject_to":
  ["pkg.mod:constraint"]}]`, or `"solves"`, `"equals"`, `"approximates"`. Drafted from the code's decorators; the
  targets are code references, checked as any are.
- `where`: `{TeX: definition}` for symbols the equation uses that are not values of the graph (a step's own
  intermediates, functions like σ). Shown under the equation, as a paper would. A value two steps share should be
  a symbol listed in both instead.
- `bind`: for the shape trace, `{code name: symbol}`, naming the parameters of the function in the step's first
  code reference, and `"return"`, `"return.0"` (a tuple's first item) or `"return.attr"` (an attribute of the
  result). Optional; see "Shape trace" for which steps can be bound.
- `names`: how to write the code's names when the step's code is written as math (see "The code written as
  math"): `{code name: symbol or TeX}`, e.g. `{"z": z, "pseudo": r"\kappa", "s.alpha": r"\alpha", "return": out}`.
  A symbol object stands for its TeX and its indices; a string is TeX. The names in `bind` count too.

**Equation checks**: a function that computes the equation as written, plainly (numpy on small made-up data),
and asserts the code gives the same. Registered for one or more steps:

```python
@G.verify("center", "score")
def as_written() -> None:
    X = np.random.default_rng(0).normal(size=(6, 4))
    expected = (X - X.mean(0)).sum(1)                  # the equations, written out
    assert np.allclose(mypkg.core.score(X), expected)  # the code
```

A step whose check fails is marked ✗ in the viewer (and `headway check` fails); a passing check shows only in the
step's side panel.

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

### Code in other languages

The spec is always Python, but the code it points to can be in any language: a reference that starts with a file
path points into that file.

```
"R/score.R"                         the whole file
"R/score.R:score_cells"             a definition: a function, class, method, struct, rule, ... ("Model.score" for
                                    one inside another: a method of a class, a function in a Rust impl)
"R/score.R:score_cells#text"        anchors as above, within the definition (or "R/score.R#text", within the file)
"main.nf#process ALIGN"             on a line of code (not a comment), "#text" takes the whole construct that
                                    starts on that line: the process, rule, function or block
"config.yaml:12-14", "...:12"       lines
```

Definitions are found with tree-sitter (the `tree-sitter-language-pack` dependency), which knows R, Rust, C, C++,
bash, Julia, Go, Java, JavaScript, Snakemake, Fortran, MATLAB and a few hundred others; each language's parser
downloads on first use into a cache (`~/.cache/tree-sitter-language-pack`). For a language it lacks (Nextflow, for
one), names do not resolve; text anchors and lines still do. For such a construct, the end is the last line
indented under the anchored one, plus a closing line ("}", "end", ...). Prefer names, then anchors, and lines only
when nothing else works: the check catches a renamed definition or a vanished anchor, while lines go stale
silently as the file changes (`headway diff` still shows the code they now point to).

A path is relative to the working directory, else to the git work tree holding the spec, else to the folder
above the spec's outermost package, else to the spec's own folder; or absolute. The viewer shows each reference's
code with its language's highlighting. The equation checks (`@G.verify`) and the shape trace are for Python code
only.

## What `headway check` reports

With `Headway(..., hints=True)`, each step's values given symbols by `bind=` or `names=` are also held to the shapes
the code's type hints state (jaxtyping): a warning if a symbol has another number of axes than its annotation (or
no `dims=` where the code states axes), or if one axis name of the function is two different indices of the spec
within the step.

Errors (exit 1): a reference that does not resolve; an undeclared symbol or index; a symbol produced by two steps;
a step producing an input, setting or constant; a value or output no step produces; a cycle; a `condition` that
is not an expression; a relation of an unknown kind, or naming no function; a `where` entry
whose TeX is exactly a declared symbol's TeX (list the symbol instead; the comparison is of whole strings, so an
entry defining several symbols at once, `"n_1, n_0"`, is fine); a failed `@verify`; a `was` naming a key the spec
still declares; and, in a suite, the errors listed under "Benchmarks". Warnings: a symbol no step uses (outputs
excepted; one a condition names is used), a step with no code reference, a constant with no code reference, a
group with no steps, a condition naming what is not a symbol (the viewer cannot say what it is).

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

### The code written as math

The side panel shows each step's code, written as math, under "From the code". It is open when the step has no
equation yet, and folded when it has one. headway reads the code of the step's first Python reference (a
function, or an anchored block, read as part of the function it is in) with the `ast` module, without running it.

How it reads it is headway's core (`docs/core.md`): the code is translated into a small calculus (index sets,
tensors as functions of indices, sums with conditions, lookups by pointers, cases, iterations, distributions),
each library's operations by a table of translations (numpy; scipy.special and scipy.stats; Python's builtins and
`math`; the first of PyTorch's), then written by one printer. An operation no table translates is written as a
named operator applied to its arguments, never interpreted. What is shown can be **checked** against what the
code computes, on a real run: `headway trace` (below) runs the translation itself, on the arguments of the
code's calls in its tests, and compares; the panel then says "✓ this math computes what the code computes, on all
N traced calls" under the math. It writes:

It writes only what the code states: where the meaning could depend on what happens at run time, it falls back
to writing the code line by line, or as a function of its arguments, rather than guess.

- **Base Python:** arithmetic (`/` as a fraction, `**` as a power, `@` as a matrix product), comparisons,
  `x if c else y` as cases, `sum(f(a) for a in xs)` as a sum over `xs`. A product `a * b` is written $a \odot b$
  (elementwise, as numpy does it) unless one side is a scalar: a number, `float(...)` or `int(...)`, a symbol
  with no `dims=`, or, with `Headway(..., hints=True)`, a name the code annotates as a number (`k: float`,
  `p: np.float64`) or that a type checker (mypy) infers is one. Type hints are not checked by Python, so they are
  used only when the spec says so.
- **numpy:** elementwise functions (`exp`, `log`, `log1p`, `sqrt`, `abs`, `maximum`, `clip` as max and min,
  `where` as cases, ...), reductions (`sum`, `mean`, `std`, `max`, ...), `.T`, `dot`, `outer`, `linalg.inv`,
  `solve` and `norm`. Conversions that keep the values (`asarray`, `astype`, `copy`, `float`) are left out;
  `reshape` keeps the values but not their indices. A call with an argument the table does not know (`ddof=1`,
  `where=`) is written as it is, keywords named: $\operatorname{std}(x, 0, \text{ddof} = 1)$. So are `int`,
  `len`, `nan_to_num`, positions from the end (`x[-1]`) and stepped slices.
- **One equation from straight-line code.** Code made only of assignments to variables, updates of them (`x +=
  e`, `out=x`) and a return is written as one equation: each variable is replaced by what was assigned to it.
  So the code's structure decides the equation:
  - a sub-computation moved into its own function stays a named call, `\operatorname{helper}(...)`;
  - a variable given a name with `names=` keeps its own line (and its updates are written `x ← ...`);
  - so does a variable read more than once (`f` in `np.log(f / (1 - f))`), when it is assigned once, by a
    statement of the function itself, and never changed in place: written in, it would be written out again
    wherever it is read;
  - so does a variable annotated with named axes (below), with hints;
  - an `assert` at the top level is a precondition: the result is claimed only where it holds (the code stops
    if it fails). It is written under the equations, "provided that ...", one line per part (`a and b`, and
    `a & b` inside an `all()`), worded apart from cases, which choose between alternatives.
    `(x > 0).all()` and `np.all(x > 0)` are written $x > 0$, as an indexed inequality means. An assert inside a
    loop or branch is written in its place.

  Not when a variable is updated in place while another holds the same value (`y = x; x += 1`): `y` would change
  too, which replacing variables cannot show. Nor when a call's result is unused (`a.sort()`, `f(a)`): it may
  change its arguments. That code is written line by line, the call as a line of its own.
- **Branches and loops in one equation.**
  - An `if`/`else`, or an early `return` (`if p < eps: return 0`), becomes cases.
  - A loop that adds up becomes a sum or product over the loop's variable: `acc += e` is $\sum$, `acc -= e`
    subtracts it, `acc *= e` is $\prod$. A sum over `range(n)` has bounds, $\sum_{i=0}^{n-1}$.
  - A loop that fills in element by element becomes an indexed family, $(e)_{i=0}^{n-1}$, when it fills
    something whole: appending to an empty list, or `out[i] = e` over `range(n)` into `out = np.zeros(n)` (the
    same `n`). With `for i, x in enumerate(xs)`, `x` is written as the i-th of `xs`.
- **Other code** (a `while` loop, a loop doing anything else, an assignment into part of an array, a named
  variable assigned inside a branch or loop) is written line by line, as an algorithm is written: each
  assignment a line, a variable assigned again (or updated: `x += e`) as `x ← ...`, calls as they are, flush
  left and indented by depth as the code is. A loop over `range(n)` is "**repeat** n **times**:", or "**for**
  r = 0, …, n − 1:" if its body reads `r`. Asserts are preconditions ("provided that ..."); printing (`print`,
  `logging`, `warnings`) is left out. An iteration is written this way, not as a recurrence: it should never
  read worse than the code.
- **Cases from an if.** An `if` that decides values, each branch one assignment (a value a branch does not assign
  keeps the one it had), is one equation with cases, the values the code had before primed:
  $(z', e) = \{\, f(z, d)$ if $\kappa$; $(z, \varnothing)$ otherwise $\}$ (`if correct: z, e = f(z, d)`, `else:
  e = {}`; `{}` is $\varnothing$).
- **Another name for a value.** `m = a.astype(np.float64)`, or `m = a`, where `m` is assigned only there and
  neither name is changed or assigned again, is `a`: written as `a`, with no line of its own.
- **Imported names**: a name the file imports is a module, not a value: `embed.Embedding(a)` is written
  `Embedding(a)`.
- **Anything else** is written as a function of its arguments, `\operatorname{name}(...)`. It is the code,
  never a guess at what it means.

Names are written as given by `names=` and `bind=`. A code name given a symbol carries the symbol's indices, and
they are used:

- **Reductions** over an axis of an indexed value are written over its index: `z.mean(0)` on `z` named as
  $z_{ig}$ is $\operatorname{mean}_i z_{ig}$. Without known indices it is "axis 0".
- **Selections** keep or drop indices: `z[:, module]` is $z_{ig}|_{g \in \mathit{module}}$, `x[0]` drops the
  axis, and `x[:, None]` broadcasts.
- **Indices carry through** arithmetic as numpy broadcasts them. A local variable whose indices are all known
  is written with them (`mu = x.mean(0)` is $\mu_p$).
- **Products** of values written with all their indices are index notation, $n_{ig}\,q_{ig}$ and $w_i\,r_{ig}$,
  not $\odot$. A matrix product of such values (`a @ v`) is written with its sum: $\sum_g a_{bg}\,v_{gk}$ (if the
  two name the summed axis differently, $\sum_{g=d} a_{bg}\,v_{dk}$).
- **A mask** with an axis's index restricts that axis: `x[learns]` with `learns` indexed by $c$ is
  $x_{cg}|_{\mathit{learns}_c}$ (a boolean array; an array of positions is a pointer, below).
- **A pointer** (an integer array of positions) stands in for the index it indexes: `x[:, cols]` is
  $x_{c,\mathit{cols}_p}$; `np.arange(len(a))` is $a$'s index itself, $v$; numpy's rule that pointers apart put
  their axes first is followed.
- **An index summed over that is also free** where the sum is used is primed:
  $\tilde z_{ig} = z_{ig} - \operatorname{mean}_{i'} z_{i'g}$; $K_v = (A_v \ge 0) \lor v = \operatorname{argmax}_{v'} A_{v'}$.
  So is the second index over the same set in a product such as `x.T @ x`: $\sum_c x_{cg}\,x_{cg'}$.
- **Linear algebra** inside an inverse or a solve is written in matrix notation, where index notation would not
  say which index is contracted: `np.linalg.solve(X.T @ (w[:, None] * X) + r * np.eye(k), X.T @ (w * z))` is
  $(X^\top \operatorname{diag}(w)\,X + r\,I)^{-1}\,(X^\top (w \odot z))$. The identity elsewhere is
  $\delta_{cc'}$.
- **Sizes**: `len(x)` and `x.shape[k]` of an indexed value are the size of the index set, $|\mathit{learners}|$,
  or of the part of it a mask keeps, $|\{c : \ell_c\}|$.
- **Distributions**: a draw from numpy's generator (`rng.normal(mu, sd)`, `rng.binomial`, `rng.gamma`, ...) is a
  line of its own, $z_g \sim \mathcal{N}(\mu_g, \sigma^2)$; scipy.stats' densities are
  $\log \operatorname{Binomial}(k_c \mid n_c, p_c)$, so a likelihood is a sum of them; `logsumexp` is a
  reduction like `sum`.
- **Tables (polars)**: a data frame is a table, named columns over an index set of rows (a parameter `design`'s
  rows are the set design, $d$); a column is $\mathit{learns}_d$. `df.filter(e)` is the rows where $e$ holds,
  `df.with_columns(y=e)` a column defined, and `df.group_by("k").agg(pl.col("x").sum())` a row per value of $k$,
  $x_k = \sum_{d : k_d = k} x_d$ (`.len()`: $|\{d : k_d = k\}|$). A verb the table does not know (sort, join for
  now) is an operator.
- **Deep learning (PyTorch)**: elementwise functions and reductions over `dim=` as numpy's; `torch.softmax` as
  $e^{x_{nk}} / \sum_{k'} e^{x_{nk'}}$; `F.cross_entropy(z, y)` as $\operatorname{mean}_n (\operatorname{logsumexp}_k
  z_{nk} - z_{n,y_n})$; `F.linear` and an `nn.Module`'s `nn.Linear` layers (made in its `__init__`) as
  $\sum_f x_{nf} W^{(\mathit{layer1})}_{hf} + b^{(\mathit{layer1})}_h$; `torch.autograd.grad(L, θ)` as
  $\nabla_\theta L$, so a training loop reads as gradient descent.
- **The code's own helpers**: a call to a function of the code with no shape in its annotation has the shape its
  own code gives it (read with the call's arguments), so `logit(baseline)` keeps $j$. A module's numeric
  constant (`RIDGE = 1e-6`) is written as its value, unless the spec names it.

With `Headway(..., hints=True)` (or `headway tex --hints`), a parameter or variable annotated with jaxtyping's
named axes, `Float[np.ndarray, "cells genes"]`, has indices too, without `names=`: an axis that is a spec index
of the same key takes its TeX; another takes the first letter of its name that no other index and no one-letter
name of the code has, or else the first such of the usual index letters, $i, j, k, \ldots$ (`cells` is $c$;
`bins` beside a parameter `b` is $i$). So a function whose parameters state their shapes is written in index
notation: $\sum_c$, not "axis 0". An array made in a shape taken from indexed values has their indices:
`d = np.zeros(n.shape[1])` with `n` as $n_{ig}$ is $d_g$, as is `np.zeros_like` of a $d_g$ (`len(x)` is axis 0).

The code's other names (its local variables, loop variables, constants) are written as short symbols, chosen as
`headway init` chooses a draft's: a name that is notation already keeps it (`p`, `x0` is $x_0$, `mu` is
$\mu$), and every other takes the first free one of its near neighbours (`penalty`: $p$, $P$, $\pi$, ...). None
looks like a symbol the step's names show: beside $m_{if}$, `mean` is not $m$. The **Code names** switch (in the
graph bar, or the button beside "From the code") writes every name as the code has it instead (a longer name as an
italic word), keeping the same lines and the indices of the symbols for the sums.

The same switch shows every value as the code names it everywhere else too: the graph's nodes, the legend, the
panel's inputs and outputs, in code type. What the code calls a symbol:

- **Drafted spec:** its key, which is the variable's name (`scale_2`, a repeated call's, is `scale` again).
- **Otherwise:** the name the steps' code most often has for it (`names=`, `bind=`). Failing that, a setting's
  field or a module constant (`Settings.pcs` is `pcs`). Failing that, its key.

The equations stay as written, since headway does not read TeX. The choice is remembered in the browser.

`headway tex pkg.mod:func --name z='z_{ig}' [--hints]` prints a function's code written this way, for trying it
out.

### Code that `headway init` can draft well

`init` reads only what the code states, so code written plainly drafts best. None of this is specific to headway;
it is ordinary good style, which a coding agent can follow as it writes the method:

- **Give results names.** `scores = score(cells, model)` is an edge; `use(score(cells, model).mean())` is a gap
  where the mean is taken.
- **Return values rather than changing arguments.** `x = scale(x)` returning a new array is an edge;
  changing `x` through a view inside `scale` is a note.
- **Call the method's functions directly.** A function passed to another (`pool.map(f, cells)`) is not drafted
  as a step; a loop that calls `f` is.
- **Type hints** (with `--hints`): annotate parameters and returns (`NDArray[np.float64]`, `float`, your own
  classes), and put settings in a dataclass with defaults, each field with a comment saying what it is.
- **Docstring sections**: numpydoc (or Google) Parameters and Returns, with shapes in the types
  (`ndarray of shape (n_cells, n_genes)`): descriptions for the values, and with `--hints`, their indices.
- **Shapes in the type hints** (with `--hints`): jaxtyping's `Float[np.ndarray, "cells genes"]` on parameters and
  returns, one name per axis meaning; and annotate where arrays are first made (`counts: Float32[np.ndarray, "cells
  windows"] = ...`), since a shape is only followed from where it is stated. Check the annotations by running your
  tests with jaxtyping's checker (`pytest --jaxtyping-packages=mypkg,beartype.beartype`): it found three wrong ones
  in the first codebase annotated this way.
- **Docstrings** become the steps' first descriptions (see below).

### Descriptions from the code's comments

A step whose description is empty or a placeholder ("TODO") shows what its code says about itself, in muted
italics and marked "from the code's comments". The text comes from the step's first code reference, taking the
first of:

1. **A docstring** of a Python function or class.
2. **The comment block directly above** the referenced code, in any language: R's `#'`, Rust's `///`, a `/** */`
   block, `#` or `//` lines.
3. **The comment lines the referenced code opens with**: the `# step 2: ...` an anchor points at, or the first
   comment in a function's body.

Comment markers, rule lines (`# -----`) and documentation tags (`@param` and after) are dropped, and only the first
paragraph is kept. Comment lines are recognized from tree-sitter's syntax tree, so there is no code of headway's own
per language. A description written in the spec replaces it.

### Starting from a draft: `headway init`

For a Python method, steps 1, 2 and part of 4 can start from a draft read from the code:

```
headway init pkg.method:run -o pkg/graphspec.py [--depth 2] [--hints] [--clean]
```

`headway init` reads the source with Python's `ast` module and traces variables with `beniget` (Python's own
scoping rules). It does not import or run anything, and it gives the same draft for the same code. It draws only
what the code states: an edge is never a guess at what the code does at run time, so the draft can be incomplete,
and says where, but is not wrong. Starting from the entry function:

- **Steps.** Every call to a function or class of the same top-level package is a step. Its code references
  are the function and the line that calls it. Nested calls (`f(g(x))`) are separate steps. Calls into other
  packages (numpy, polars), building a dataclass, and a function passed to another call (`pool.map(f, xs)`: it
  is not called here) are not steps.
- **Wiring, by binding only.** A step's inputs are the variables its arguments read, each traced to the
  assignment that made it. A variable assigned a step's call holds that step's output, named after the variable
  (`x = f(...)`, `x, y = f(...)`, `m, n = f(a), f(b)`); one assigned another variable holds the same value;
  a dataclass holds what it was built from. The entry function's parameters are inputs, and what it returns
  (directly, or in a dataclass) its outputs.
- **Gaps.** A variable made by any other code (`z = np.log(x)`, a list built up, a loop's element) is not
  followed: a step that reads it shows it as an input of its own, marked "Made by the code at file:line", with a
  code reference to that line. So between two steps joined by plain code there is no edge, but a visible gap.
- **Clean code (`--clean`).** Clean code is code written to be read: say, a clean version of a method, kept
  beside it, its math in plain numpy between calls, each value made once by an assignment and never changed after
  (no changes in place, no values shared by two names, no functions passed around). In such code what a variable
  holds is exactly what its assignment computed, so binding alone gives all the data flow there is, and `--clean`
  drafts all of it:
  - type hints are taken (as `--hints`);
  - every function that calls the package's functions is opened up into a group, all the way down (`--depth`
    does not apply; a recursive call is a step): the code's own functions decide the graph's steps and groups;
  - an assignment made by other code than the package's calls (numpy, arithmetic, another package's call,
    unpacking a value into variables) is a step of its own instead of a gap: its code is the line, its inputs
    what the line reads (traced as a call's arguments are), its outputs the variables it assigns, and its
    equation the line written as math. So is the one value a `return` makes there (`return Model(a, b)`; a
    tuple's items are not, as the line would stand for all of them: assign the item to a variable first);
  - a conversion that keeps the values (`np.asarray(w, np.float64)`, `x.astype(...)`), as the math panel leaves it
    out, is the value it converts;
  - unpacking a parameter (`(m0, n0), (m1, n1) = pseudobulks`) shapes each part as the parameter's annotation says
    its items are (`Sequence[tuple[A, B]]`).

  - an `if` that decides values (its branches only assign; each value it assigns that is read after it either had
    one before, which a branch that does not assign it keeps, or is assigned by both branches) is one step, as
    the choice is one action: its math the cases, what its test reads (a setting, an input) an ordinary input,
    its outputs the values it decides. The calls inside it are part of it. What its test reads is explained in
    the same block: "where κ = *correct* (setting, default True): remove plate effects ...";
  - an `if` that makes a value only under its test (no value to keep otherwise) is a guard: the steps under it
    have its test as their `condition`, written in the graph's terms (each variable that holds one symbol written
    as that symbol's key: `embed`'s `correct`, given the setting `correct`, is `correct`; under `else`, the test
    negated, `a is None`), shown in their math ("... if κ"). A function opened under an `if` has its steps under
    it too;
  - an `if` that literals alone decide (the call gives a literal, or leaves a parameter to its literal default,
    `agreement=None`) is neither: only the branch taken is drafted, and a note says which was left out and why;
  - a list made by a comprehension, `[f(m, n) for fold, (m, n) in enumerate(pairs)]`, is the steps of its
    element done per an index: the enumerate's (`fold`); or, over a value made per an index already (`[f.a for f in
    folds]`, `zip(folds, tops)`), that index again; or else its variable. Unpacking an element of another value
    is a step of its own, per that index, its variables shaped as the annotation of what it runs over says. A
    comprehension reads what it runs over only through its variables, as far as they are used;
  - a dataclass of the package's (`Fold(a, rl, beta)`) holds each field's value: reading a field (`fold.beta`)
    is an edge from that value only;
  - a loop that carries values from turn to turn (its body assigns a variable and reads it back: an iteration)
    is one step, as an iteration is one action: its inputs what comes into it (the start values, and what every
    turn reads), its outputs the variables it assigns that are read after it, its equation the loop written as an algorithm
    (above). The calls inside it are part of it, not steps. A function of the package that iterates is likewise
    one step, not opened. Its math is an algorithm (above): the small functions of its own file that iterate are
    written in place where it calls them (inline, their parameters replaced by what the call gives); the others
    it calls are defined under it ("where ..."), in one flush-left block. Put each loop in a function of its own
    (a loop in a loop: the inner one in a function the outer one calls), so each reads alone;
  - a loop that carries nothing has its steps done per its index (its variable; for `_` in `range(n)`, `n`'s
    name).

  Not steps: an assignment into part of a value (`x[i] = e`, a change in place, noted as in any code), one that
  reads no symbol (a constant: still a gap), and the call inside a line (`a = np.sqrt(w) * f(x)` is written with
  the call: put the call on a line of its own to see its result as a symbol). Whether the code is clean is not
  checked yet: where it is not, the notes say so, as for any code.
- **Clean code: what it computes, and why.** A clean function may declare what it computes, beside how, with a
  decorator from `headway.clean` (it attaches the declaration and changes nothing else). The kinds are a closed set:

  ```python
  from headway.clean import maximizes, equals

  def intercept_objective(methylated, total, p_genome, k, b, d):
      """The penalized binomial log-likelihood of the calls: ..."""
      ...

  def groups_centered(d):
      return d.sum()                      # a constraint: zero where it holds

  @equals("methembed.core.model.fit_intercepts", rtol=1e-6, atol=1e-8)       # the code it stands for
  @maximizes(intercept_objective, over=("b", "d"), subject_to=(groups_centered,))
  def fit_intercepts(methylated, total, baseline, p_genome, k, rounds=6, newton=3):
      ...                                  # the algorithm: alternating Newton steps
  ```

  - `maximizes` / `minimizes(objective, over=..., subject_to=..., tol=1e-6)`: the function's results (named, in
    order, by `over`) are a local optimum of the objective under the constraints;
  - `solves(equation, over=..., tol=...)`: they make the equation zero;
  - `equals(reference, rtol=, atol=, adapt=)`: it gives what the reference gives, exactly up to rounding (the
    reference may be named, `"pkg.mod.func"`: the original code a clean version stands for);
  - `approximates(reference, rtol=, atol=)`: within a stated error.

  An objective, equation or constraint names its parameters as the function's own arguments, or as its results
  (the `over` names). `headway.clean.check(fn, relation, args, result)` tests one call: an optimum by the
  derivative of the objective along random directions that keep the constraints (near zero, relative to the
  objective's size and the variables') and by small steps along them (none improves it); an equation by its
  residual; equals and approximates by comparing. `headway trace CLEAN_MODULE -- PYTEST ARGS` runs the code's
  tests (nothing in the code changed: a profiler watches the calls) and checks, on each recorded call, the clean
  version against the original it stands for, its other declarations, and the math the viewer shows (each
  function of the clean module, and each objective, run by headway's reference evaluator and compared with what
  the code returns, within its declared tolerance or else to rounding). It writes what it found beside the clean
  code, `.headway-checks.json`: per function and declaration (and "math"), the calls checked, how many bore it
  out, the worst, the command and when. `--swap` instead runs the tests with the clean versions in place of the
  originals. Clean code with no original (written clean) is checked on its own calls.

  The draft gives the step `relations=`. In its side panel, an objective or equation it declares is its
  **Specification**: the declaration as math, `(b, d) = argmax_{b, d : Σ_g d_g = 0} objective(...)`, the
  objective's definition and docstring, and one line saying whether the algorithm's results bore it out on the
  run's traced calls (✓, ✗ with how far off, or not checked yet; the run and when on hover). The code written as
  math is then its **Algorithm**, its last line naming its results as the specification does. A reference to
  other code it stands for (`equals`, `approximates` the original) is a line in its **Code** section: whether this
  code gives the same. The prose is the author's word; what is checked is the declaration.
- **Changes in place.** A package function that changes a parameter in place (stores into it, `p[i] = e`,
  `p[i] += e`, `p.a = e`; numpy's `out=p`; or passes it to a package function that does) changes the variable it
  is given: after the call, that variable holds the step's changed value, an output of the step named after it.
  If the function is opened into a group, the change is made by its code, so the variable holds a gap marked with
  that line. A parameter read after the line that changes it holds the changed value too. A change through
  another name (`part = x[:, c]; part -= m`, a view) is not seen: whether that is a view depends on run time.
- **Notes.** At the end of the draft, a comment lists where values may change with no edge drawn: each
  function's lines that may change a value in place (`x += e`, `x[i] = e`, `out=`, a call whose result is unused,
  such as `xs.append(e)` or `a.sort()`), a value a loop carries from one turn to the next, and calls inside a
  function defined within another. Check these places in the code.
- **Short symbols.** Each index and symbol gets a short, unique symbol. Names that are notation already keep it
  (`x`, `x0` is $x_0$, `mu` is $\mu$). Every other name takes the first free one of its near neighbours:
  - its words' initials, then their capitals (`size_factors`: $s$, $S$, $f$, $F$);
  - their greek counterparts ($\sigma$, $\phi$, ...), tried first for settings and constants;
  - the first initial accented ($\tilde s$, $\hat s$, $\bar s$), subscripted by the second ($s_f$), or numbered.

  Indices are assigned first, then inputs, outputs, settings and values. A repeated call's output (`scale_2`)
  takes its first's symbol, primed ($s'$). The readable name stays the symbol's `name`. Redrafting after the code
  gains a variable can shift the letters of later ones.
- **Groups.** A called function that itself calls the package's functions is opened up into a group of those
  calls, `--depth` levels down (0: every call in the entry function is one step). Its decorators, defaults and
  annotations are not calls it makes (they run when it is defined).
- **Per.** A step inside a `for` loop is done `per=` the loop's (first) variable.
- **Constants.** A module-level constant read is a constant, with its line as `code`.
- **Docstrings.** What the code's docstrings say (numpydoc, Google or Sphinx sections, read with
  `docstring_parser`) describes the symbols: the entry function's Parameters its inputs, a called function's
  Returns its outputs (by position, for a tuple), a settings field's comment (`k: float = 2  # the pseudocount`)
  or its class's Attributes its setting, and a called function's first paragraph its group.
- **Type hints, as assumptions (`--hints`).** The code's type hints are not checked by Python, so `init` uses
  them only when asked. With `--hints`, what a variable holds is what a type checker (mypy, run on the whole
  package) infers from them, following them through calls, attributes, loops and the libraries' own stubs; failing
  that, what its annotation says (a parameter's, an annotated assignment's, or the return annotation of the package
  function whose call it was assigned). If mypy cannot run, the draft says so in its notes. With the types:
  - a parameter typed as a dataclass whose fields all have defaults is a settings object: the fields read are
    settings, with their default as `value` and the field as `code`;
  - a method called on a value typed as a package class is a step (`source: CallSource`, then
    `source.map(...)` is the step `CallSource.map`), reading what it is called on;
  - `x += e` on an array (`NDArray`, `np.ndarray`), and `.append`, `.sort`, `.update`, ... on a list, dict, set
    or array, change it in place, as above;
  - a view of an array changed in place changes the array: by numpy's rules, `v = p[...]` indexed only by
    slices, integers, `None` or `...` (as written, or names mypy knows are slices or integers), `p.T` and
    `p.view()` are views; indexing by an array or a mask copies;
  - a shape a docstring gives in a type (`counts : ndarray of shape (n_cells, n_genes)`) is the symbol's
    `dims=`, each name an index (declared, with "TODO");
  - a shape the type hints state, as jaxtyping writes it (`x: Float[np.ndarray, "cells genes"]`, read from the
    syntax, with the module's imports and `type` aliases), is the symbol's `dims=`. A function's axis names are its
    own, like type variables: each call binds them to the caller's indices through the arguments given as they are
    (`scale(x: Float[np.ndarray, "cells columns"])` given an x of indices (cells, seen): columns is seen, in that
    call), so its results take those indices. An axis only a result has (`-> Float[np.ndarray, "cells
    components"]`) is a new index, named as the code names it. An axis a parameter names but the call cannot bind
    (its argument's shape is not known) leaves the result's shape unknown: no `dims=`, rather than a guess. The
    entry function's axis names, and those of a variable annotated where it is assigned (`x: Float32[np.ndarray,
    "cells windows"] = ...`), are the spec's indices. Only named axes are read: `...`, `*batch`, `#n`, `_` or a
    number say nothing.

  The choice is kept in `draft=` (`"hints": True`, or `"clean": True`), so Redraft makes the same one, and the draft sets
  `hints=True`, so "From the code" takes the hints too.
- **Names.** Each step gets `names=` for "From the code": a parameter given a variable or a settings field as
  it is (not an expression of it) is written as that symbol, and the call's return as its output. Give a symbol its
  TeX and `dims=` and every step that touches it writes it with its indices (`mean_i x_{ig}`).

The draft records its arguments in the spec, `Headway(..., draft={"entry": ..., "depth": ...})`, and ends with a
checksum of what `init` wrote. When the code changes, `headway redraft SPEC` (or **Redraft** in the viewer, for
a spec served live) drafts it again with the same arguments, over the file. A spec edited since it was drafted is
only redrafted with `--force` (in the viewer, after you confirm it), and the edited version is kept beside it as
`.bak`: redrafting replaces hand edits.

The draft passes `headway check` with no errors. What it leaves to you (or a coding agent):

- **Notation.** Every symbol's TeX is a short symbol chosen from its name, and every step's equation is empty:
  choose the notation you want (the paper's, the field's) and write the equations.
- **Text.** Every description is "TODO".
- **Dimensions.** `dims=` are not drafted.
- **Pruning.** Steps that are bookkeeping rather than the method should go, and repeated calls (the same
  function called twice, numbered `_2`) may be one step `per=` something.
- **The wiring the code does not state.** Each gap (an input "made by the code") and each note marks a place
  where values flow by plain code, in place, or through a callback: draw those edges (or steps) where they belong
  in the method.

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

In the viewer, **Open…** opens more versions beside the page: under `headway serve`, from the library (recent
headways, and any folder's); from a page opened as a file, with the file picker. A version built without `--name` is called by its file name; lanes are ordered by build time and can be dragged, closed or
pinned as the baseline. Markers: **+** added, **Δ** changed, **−** gone, **⋔** split or merged. **One at a time**
shows one version, keeping the others open. A version's eye (in the version bar or on its Results row; drag across
several, or shift-click, to sweep a range) hides it from both views while it stays open: the objectives are scored
over the versions shown, each version is compared with the shown one before it, and a hidden baseline is set aside.
In the Results view, drag a column's header to move it among its kind (objectives, metrics, datasets, figures), and
**Columns** shows, hides and reorders them; this is display only (kept in the browser, per suite), and the objectives
are scored as defined. **Copy image** copies a PNG of the whole graph or of the results table; the side panel's
equations and code have small copy buttons of their own.

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
  interval="", code=(), where=None, group="metrics", was=())`** declares the output symbol `key` and the step `key` that
  computes it, so a metric has a notation, an equation, code and a check like any step. `better` is `"higher"` or
  `"lower"`; `range` is its possible extent (needed by `how="mean"` objectives). For a metric with no natural
  bound (a mean squared error), either use a bounded version of it (R², or the error relative to a baseline's) or
  leave out `range` and combine it only in `how="rank"` objectives; a guessed range warns when values fall outside.
  `interval` says what the intervals in its results are, e.g. `"95%, bootstrap over embryos (200 draws) within one
  seed"`; results may give intervals for a metric only if it is set (see Intervals and differences).
- **`B.dataset(key, name, desc="", *, source="", heldout=False, values=None, facts=None, code=())`**. If the suite
  makes its data (a simulation, a preprocessing), declare those steps in the suite like any other (with their
  settings, equations and checks) and give each dataset its `values` of those settings: the steps are written
  once and the datasets differ in their values. `values` may only name declared settings. Keep one copy of the
  values: either the benchmark code holds them and the suite reads them (`values=data.DATASETS["pbmc"]`), or the
  suite holds them and the benchmark code reads them back (`B.datasets["pbmc"].values`, a dict). Replicates (seeds,
  folds) are not dataset values; name them in the results' `replicates`. `facts` is anything
  else worth knowing; `source`, where the data come from; `code`, where they are made or loaded. A held-out
  dataset is hidden in the viewer unless asked for.
- **`B.figure(key, name, desc="", *, source="", code=())`** declares a figure: an image the benchmark makes for
  each dataset (samples, reconstructions, an embedding, a diagnostic plot), for what cannot be reduced to a
  number or should be seen as well as measured. It is a column of the results beside the metrics, a thumbnail
  per version and dataset; clicking it compares the versions' images side by side. A figure is never shaded,
  sorted, ranked or part of an objective. `desc` says what to look for; `code` points to the function that draws
  it.
- **`B.objective(key, name, desc="", *, metrics, how="mean", datasets=None, tex="")`** combines metrics into one
  number, kept apart from the metrics. `metrics` is `{metric: weight}` or a list (weight 1). `how="mean"`: each
  metric's mean scaled to 0..1 by its range, turned so 1 is best, weighted and averaged over the datasets
  (absolute). `how="rank"`: the weighted mean rank among the versions shown, 1 best (relative). `datasets`:
  default every dataset not held out. A version missing any value an objective needs gets no score ("incomplete").
  A rank objective can reorder two versions when others are added or hidden (a version that loses narrowly on one
  metric loses as many places as there are versions in between); a `how="mean"` objective scores each version from
  its own values, so it cannot. In the viewer, **+ Objective** (in the Results view) makes one's own objective of that kind:
  the sum, over the metrics and the datasets shown, of a transform of each version's mean typed per metric (`-x`,
  `2*x`, `x/0.002`, `log(x)`, with `sd`, `n`, `lo`, `hi` and a few functions), kept in the browser, not the suite.

Suite errors: an empty range, an objective naming an undeclared metric or dataset, a `how="mean"` objective over a
metric without a range, a weight that is not positive, a dataset value for anything but a declared setting, a
figure with the same key as a metric.

### Results

headway computes no metrics. Your benchmark runs the method, computes the metrics, and hands headway the values:
one per replicate (seed, fold), for each dataset and metric it ran. Missing combinations are fine (shown hatched).

```json
{"suite": "mybench.suite",
 "values": {"pbmc": {"auroc": [0.91, 0.90, 0.93], "stability": [0.8, 0.82, 0.79]}},
 "replicates": ["seed0", "seed1", "seed2"],
 "differences": {"v4": {"pbmc": {"auroc": [{"value": 0.012, "lo": 0.004, "hi": 0.020}, 0.010, 0.015]}}},
 "figures": {"pbmc": {"umap": {"image": "figs/v5_pbmc_umap.png", "replicate": "seed0"}}},
 "run": {"date": "2026-10-27"}}
```

`suite` names the module holding the Suite (or `"module:attribute"`); in Python it can be the Suite itself. `run`
is anything worth keeping. Results are stored in the version's headway together with the built suite; versions
are compared on results with the same suite title and version, so bump `version` when a change to the suite makes
old results incomparable. Values for a dataset or metric the suite does not declare are left out with a warning.

**Name the replicates.** `replicates` says what each value is, in the same order as every value list: one list for
all datasets, or `{dataset: [ids]}` when they differ (different folds per dataset). Two versions' values with the
same id are paired (the same seed, the same split of the data), which is what a paired comparison needs; use the
same ids for the same seeds in every version's run. Without `replicates` the values are unpaired. A value list
whose length differs from its ids is kept unpaired, with a warning.

**Intervals and differences.** Every number in `values` and `differences` takes one of two forms: a point,
`0.81`, or a point with an interval, `{"value": 0.81, "lo": 0.78, "hi": 0.84}`. There are no others: convert a
standard error to an interval, pick one interval level, report the interval of a difference instead of a p-value.
What the intervals are is declared once, on the metric (`interval=`), so every version's are of one kind; an
interval for a metric without `interval`, or one that does not hold its value, is left out with a warning.

`differences` gives this version's difference from another version, named as that version's headway is named
(`name=` / `--name`): `{other: {dataset: {metric: [one per replicate, in the replicates' order]}}}`, or a single
number for one difference pooled over the replicates. Give it when the benchmark can compute the difference's own
interval: on the same samples for both versions (the same cells, embryos, folds), resampling them for both at once.
That interval is usually much narrower than either version's own, because what moves one version's number moves
the other's too; two versions' own intervals can overlap while the interval of their difference excludes 0, so
the difference's is what to judge by. Only the benchmark can compute it (headway never sees the samples), so give
it for the comparisons that matter: the version before, the baseline.

headway itself pairs two versions' values by replicate id: the difference of each replicate in common, their mean
and spread, and how many agree in sign. Under each value (the Results view's Differences), the version's
difference from the one it is compared with (the row above it as the table is shown, sorted or dragged into order,
or the baseline) reads `+0.026 5/5 ●`: the mean of the per-replicate differences, how
many replicates agree in sign, and, where the benchmark gave intervals of that difference (in either version's
results), ● if every one excludes 0, ◐ if some do, ○ if none. It is in color only when supported: every interval
excludes 0 or, without intervals, every paired replicate agrees in sign; otherwise grey. Without replicate ids in
common it is the difference of the means, marked "unpaired". Clicking the value plots each version's replicates
with their intervals, and the per-replicate differences around 0 with the benchmark's intervals: each version
against the one it is compared with, then every other pair among the open versions that the benchmark gave a
difference for (so differences from both the version before and the baseline are all seen; making a version the
baseline puts its pairs in the table's cells too). The
two kinds of uncertainty stay apart: the spread over replicates, and the intervals within one replicate; headway
does not combine them (a benchmark that wants one interval over both computes it and gives it pooled). Objectives
get no intervals. With many metrics and datasets, an interval excludes 0 by chance now and then: take a single
cell as a hint, and agreement across replicates and datasets as evidence.

**Figures: images to look at.** `figures` gives, for each dataset and declared figure (`B.figure`), the image the
benchmark made, or a list of them (one per replicate, say; the table shows the first). An image is:

- `image`: a file (PNG, JPEG, GIF, WebP or SVG), relative to the results file (to the working directory when
  results are given as a dict in Python; absolute paths are fine). It is written into the headway, so it is still
  there when the version is opened elsewhere later, and its absolute path is kept: the viewer's Ask-AI prompts
  list it so an agent can open the file. Or `url` instead: a link, not written in (for very large images);
- `caption`, `replicate` (one of the replicate ids), and `keys`: free-form labels, `{name: string, number or
  boolean}`.

Images for an undeclared figure or dataset, missing files and files that are not images are left out with a
warning. The benchmark draws the image (with any plotting library); headway only shows it. Draw every version's
image the same way (same axes, same size), since they are compared side by side.

If a judgment of a figure should count, make it a metric: declare it in the suite (e.g. `visual_quality`,
`range=(1, 5)`, higher is better, its description saying how it is judged), and give its values in `values` like
any other. Whoever judges (a person, or an agent asked to look at the images in the page) records the score there.
On the image, say what the judgment was made from and by whom with keys, e.g. `"keys": {"judged_for":
"visual_quality", "judged_by": "agent"}`.

**Keys: labels of a version.** `headway.build(..., keys={"lr": 3e-4, "variant": "no-qc"})` (or `--key lr=3e-4`
on the command line) labels the version with free-form keys: a hyperparameter, an ablation, an option. They are
stored in the headway, shown with the version in the Results view and passed to the Ask-AI prompts, and will be
what versions are grouped and paired by. A key named like a setting or constant of the spec with a value is
checked against it (a warning if they differ). Use keys for what describes the version as a whole; what describes
one run goes in `run`.

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
              results=[{"suite": suite.B, "values": values, "replicates": ["seed0", "seed1", "seed2"]}],
              keys={"variant": "weighted"}, compare=["headways/v4.json"])
```

In the viewer, **Graph | Results** switches to the heatmap of the open versions: objectives first, then the
figures' columns (a thumbnail in each cell), then metric × dataset columns (group by metric or by dataset), each
cell the mean ± s.d. of the replicates and its difference from the version it is compared with, paired by
replicate (see Intervals and differences). Click a column for the
metric's definition and every version's replicates, or for the figure's images side by side (click one to enlarge
it); click a dataset for what it is; click a version for what changed in it, its keys and its figures.

## Command line

```
headway check SPEC [--no-verify]                  exit 1 if the spec and the code disagree
headway build SPEC [-o headway.html] [--json FILE] [--name NAME] [--results FILE ...] [--compare [NAME=]FILE ...]
              [--key NAME=VALUE ...] [--collapse GROUP ...] [--hide GROUP ...] [--no-verify] [--offline] [--editor EDITOR]
headway serve [SPEC] [--port 8765] [--open] [--compare ...] [--collapse ...] [--hide ...] [--editor EDITOR]
                                                  the library (recent headways; specs and built headways in any
                                                  folder), opened on SPEC served live if given
headway diff VERSION VERSION ... [--wording]      a VERSION is a built file, or a spec module (built as it is now)
headway attach VERSION RESULTS.json [-o OUT]
headway results VERSION VERSION ... [--heldout]
headway init pkg.mod:func [-o SPEC.py] [--depth 2] [--hints] [--clean]
                                                  a first draft of a spec, read from the code (see above)
headway redraft SPEC [--force]                    a spec drafted by init, drafted again from the code
headway tex REF [--name CODE=TEX ...] [--hints] [--trace DIR [--set FOUND=NAME ...]]
                                                  a Python function's (or block's) code written as math; with a
                                                  trace, its parameters indexed by the index sets the trace found
headway record -p PACKAGE [-p ...] -o DIR [--first 20] [--every 50] [--limit 200] [--no-jit] [--python EXE]
                                                  -- PYTEST ARGS
                                                  run the tests with the package's functions traced: each call's
                                                  arguments and result (a sample of calls in full, every call's
                                                  sizes and path), and the environment, written to DIR
headway paths DIR [FUNCTION ...]                  the paths a trace's calls took (branches, loop counts), and how
                                                  many calls took each
headway sets DIR [FUNCTION]                       the index sets the trace's array sizes show, and the axes in each
headway verify DIR [MODULE:FUNCTION ...]          the math shown for each function (every function traced, if none
                                                  is named), checked on the recorded calls
headway options DIR FUNCTION ...                  the settings each function was called with, counted
headway env DIR                                   the environment the trace ran in
headway lint MODULE_OR_FILE                       where clean code leaves the clean subset (changes in place,
                                                  statements the math cannot say) or its math stays opaque
                                                  (operations no table translates, arrays without named axes),
                                                  each by its line; and the share of it written as math
headway trace CLEAN_MODULE [--swap] [--set MODULE.NAME=VALUE ...] [--limit 50] -- PYTEST ARGS
                                                  run the tests; check the clean code on their calls: against what it
                                                  stands for, its declarations, and the math shown
headway guide
```

SPEC is an importable module holding one Headway (or `module:attribute`); run from a directory where it imports.
`--offline` writes the page's libraries into it (about 3 MB) so it opens without internet. `--editor` (or
`$HEADWAY_EDITOR`): `vscode` (default), `cursor`, `pycharm`, `file`, or a URL template with `{file}`, `{line}`,
`{col}`. `serve` rebuilds when the spec or any file its references point to changes, and the open page keeps its
view. When headway's own code changes (working on headway itself, with an editable install), `serve` restarts
itself on the same port once the new code imports, and the open pages reload. Without a SPEC, `serve` lists the specs (modules making a Headway or Suite at their top level) and the built
headways under the current directory: click a spec to serve it live, a headway to open it, or tick several to
compare them. Each spec is imported from the directory above its outermost package.

### Tracing code headway did not write

For code with no named axes and no spec (a library, someone else's method), a trace of a real run fills in what
the code does not say. `headway record` runs the tests with Python's monitoring on, recording the calls of the
package's functions: the first calls of each in full and then every n-th, and for every call its arrays' sizes and
the branches it took. joblib's work runs in the same process while recording (its workers would not be seen). The
trace is a directory: `manifest.json` (the environment: Python, every installed package's version, the thread
settings, numpy's error handling and random state before and after; the run; the calls) and `arrays.npz`.

From a trace:

- `headway sets` groups the arrays' axes into index sets by their sizes: axes whose sizes agree on every call that
  has both are one set. Sizes that never vary cannot tell two sets apart, so runs on data of several sizes help.
  Each set is named after a member (rename with `--set FOUND=NAME`).
- `headway tex REF --trace DIR` writes a function with its parameters indexed by those sets, so that its products
  and sums are written with indices instead of positions.
- `headway paths` lists the paths the calls took: which branches, how often, and how many times the loops ran
  (an IRLS that fell back to an optimizer, a fit that iterated 3 times on average).
- `headway verify` gives each recorded call's arguments to the math shown and compares its result with the code's,
  with the module constants each call read as they were then (a test may change one).
- `headway options` counts the settings each function was called with (the design formulas a test suite tried, how
  often a fit used its prior), and `headway env` shows the environment the run had.

Code for an older Python than headway's runs in its own interpreter: `--python .venv/bin/python` runs the tracer
there (it needs only numpy and pytest). numba's compiled functions are not seen by a tracer; `--no-jit` runs them
as Python (slowly) so that they are. An object's slots named by a string (`self.var["dispersions"]`) are written
as symbols with their container above them, dispersions^(var).

Functions passed to `scipy.optimize.minimize` (a lambda, a function defined inside the one being read) are read as
math: the result is written as argmin over the function's parameter, with the method, start, bounds and gradient
under it, and checked by running the same optimizer on the function as written.

## Python API

```python
import headway
headway.Headway, headway.Suite                    # declaring
headway.check(G) -> Report                        # .ok, .errors, .warnings, .text()
headway.trace(G)                                  # context manager; .assert_ok()
headway.build(spec, out="headway.html", *, json=None, name=None, results=(), compare=(), collapse=(), hide=(),
              keys=None, verify=True, offline=False, editor=None, strict=False, log=None) -> dict
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
