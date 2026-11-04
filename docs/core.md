# The core: what clean code means, as mathematics

Status: built (2026-10-29), in `src/headway/core/`: the calculus (`terms.py`), Python read into it (`lower.py`),
the tables (`libs/`: numpy, scipy, Python, the start of PyTorch), the fold (`fold.py`), the printer (`printer.py`)
and the reference evaluator (`evaluate.py`), with `verify.py` and `headway trace` checking what is shown against
what the code computes on real calls. `codetex.py` is the front door the viewer uses.

## 1. What it is for

headway turns part of a codebase into clean code (written to be read, by an AI or a person), checks that the clean
code agrees with the original, and shows the clean code as mathematics. This document defines the mathematics: a
small calculus that every construct of clean code translates into, one printer from the calculus to KaTeX, and the
contract for supporting a library (numpy first).

Three guarantees, for any clean code:

1. **Sound.** What is shown means what the code computes: the calculus term a construct translates into has the
   same value as the code, on every input satisfying the code's stated preconditions ("provided that").
2. **Total.** Everything is shown. A construct the calculus cannot express is shown as a named operator applied
   to its arguments, or its statements as an algorithm block; never dropped, never guessed.
3. **Tested.** Every translation rule (each library table entry, §6) has tests, aimed at where a library's meaning
   is subtler than its syntax (axes that move under advanced indexing, broadcasting and `@` on batches, defaults
   such as `std`'s divisor, `argmax` without an axis, stepped slices). Most entries are one for one and obvious;
   these edges are where a wrong translation would still look like reasonable mathematics.

Coverage (how much code reads as compact mathematics rather than operators or algorithm blocks) is not guaranteed:
it is measured on real projects, and grown one construct at a time by the constructs that rescue the most code.

## 2. Objects

- **Index sets.** A named finite set: cells, windows, bins, groups, folds. It has an **extent** (its size, a symbol
  such as $W$) and a **letter** for its indices ($w$). Two different index sets may have the same extent.
- **Indices.** Variables ranging over an index set: $w \in$ windows. An expression may need two independent
  indices over one set (a covariance, windows × windows): the second is a **copy**, written primed ($w'$).
- **Values.** Real, integer, boolean, and category (a label from a finite set, such as a cell's group).
- **Tensors.** A tensor is a function from a product of index sets to values: $x$ : cells × windows → ℝ, written
  $x_{cw}$. A scalar is a tensor over no index sets. A mask is a boolean tensor; a **pointer** is a tensor whose
  values are indices of some set (a cell's group, $\mathrm{group}_c \in$ groups).
- **Records.** A named tuple of values (a dataclass): its fields are values.
- **Families.** A value per element of an index set, when the values are not scalars (a fold's model, per fold):
  the graph's `per=`.
- **Random variables and distributions** (§5).

## 3. Expressions

Every expression has a **signature**: its free indices (each with its index set) and its value type. Signatures are
computed by the rules below from the signatures of the inputs; an input's comes from the spec's `dims=` or the code's
shape annotations. Where a signature cannot be determined it is **unknown**, and unknown is never resolved by a
guess: an expression of unknown signature is written without indices, and an operation that needs them (an
aggregation over a named index) becomes a named operator (abstract interpretation: unknown is the top of the lattice).

| Construct | Meaning | Signature | Written |
|---|---|---|---|
| **Variable** $x$ | the value bound to x | as bound | $x_{cw}$ |
| **Constant** | a number | scalar | $0.5$ |
| **Apply** $f(e_1, \ldots)$ | a scalar function, elementwise | union of the arguments' indices | per f: $\log e$, $\sigma(e)$, $a + b$, $\frac{a}{b}$ |
| **Aggregate** $\bigoplus_{i \,:\, \phi} e$ | combine $e$ over $i$ where $\phi$ holds; ⊕ ∈ {Σ, Π, max, min, mean, logsumexp, any, all, argmax, argmin} | $e$'s indices without $i$ | $\sum_{i : \phi} e$ |
| **Lookup** $x_{\pi(i)}$ | $x$ at the index a pointer gives | the pointer's indices | $x_{\mathrm{group}_c}$ |
| **Build** $[e]_{i}$ | the tensor whose value at $i$ is $e$ | $e$'s indices, with $i$ | $(e)_{i}$, or as $x_i = e$ |
| **Restrict** $e\,|_{\phi}$ | $e$ only where $\phi$ holds (a mask) | $e$'s indices, over the subset | $e_{cw}\,|_{\mathit{learns}_c}$ |
| **Rename** | an index taken as a copy, or set as another | renamed | $x_{w'}$ |
| **Cases** | a value by conditions | union | $\begin{cases}a & \text{if } \phi \\ b & \text{otherwise}\end{cases}$ |
| **Operator** $F(e_1, \ldots)$ | a named function the calculus does not open (eig, kmeans, a fitted model) | declared, or unknown | $\operatorname{F}(\ldots)$, defined under "where" if its definition is known |

What these cover, by composition:

- matrix products, `einsum`, contractions: Aggregate-Σ of an Apply-product over the shared index;
- group-by and segment sums: Aggregate with a condition on a pointer, $\sum_{c : \mathrm{group}_c = g} x_{cw}$;
- joins: Apply over tensors sharing an index, with Lookup through pointers;
- selection by a mask: Restrict; by integer positions: Lookup; scatter: Aggregate over the preimage;
- comprehensions and per-element construction: Build.

## 4. Statements: a program

| Construct | Meaning | Written |
|---|---|---|
| **Let** $x := e$ | name a value | $x = e$ (substituted into its use when used once; see the printer) |
| **Update** $x \leftarrow e$ | a variable's next value (in an iteration) | $x \leftarrow e$ |
| **Iterate** | repeat $n$ times, or for each index of a set, a block of statements | **repeat** $n$ **times:**, **for** $r = 1, \ldots, R$**:** |
| **Choose** | an if that decides values | Cases, one equation |
| **Guard** | statements done only under a condition | their equation, "only if φ" |
| **Function** | a named function of the clean code | applied as an Operator; defined under "where", or written in place (a small iteration) |
| **Precondition** (assert) | the result is claimed only where φ holds | "provided that φ" |
| **Declaration** | what a function's results are: maximizes, minimizes, solves, equals, approximates (headway.clean) | the Specification: $(b, d) = \arg\max_{b, d : \ldots} L$ |
| **Sample** | §5 | $x_i \sim \mathcal{N}(\mu_i, \sigma^2)$ |
| **Return** | the step's results | the step's output symbols: $(\hat b, D) = (b, d)$ |

The **algorithm block** is the fallback for statements: the statements written one per line, Iterate and Guard as
indented blocks, the expressions in them written as far as §3 goes.

## 5. Randomness

Statistical models and deep learning need sampling and densities as first-class constructs, not operators.

- **Distributions.** A named family with parameters: $\mathcal{N}(\mu, \sigma^2)$, Binomial($n$, $p$),
  Categorical($\pi$), Dirichlet($\alpha$). The parameters are expressions; their indices broadcast like Apply's.
- **Sample** $x \sim D(\theta)$: draws, independently across the free indices of $\theta$ unless stated otherwise:
  $x_{cw} \sim \mathrm{Binomial}(n_{cw}, p_w)$. The random number generator, its seed and its keys are not part of the
  mathematics: they are recorded on the step (for checking), not written.
- **Density** $\log p_D(x \mid \theta)$: an expression, so likelihoods and losses are ordinary terms, summable by
  Aggregate: $\sum_{c,w} \log \mathrm{Binomial}(m_{cw} \mid n_{cw}, p_w)$.
- **Gradient** $\nabla_\theta e$: an expression of an expression, for training loops (deep learning) and Newton steps;
  written $\nabla_\theta$, or opened into its formula when known.
- **Checking randomness** is distributional, not exact: a sampled step is checked by running the original and the
  clean code from the same seed where their draws coincide, or else by comparing summary statistics within a
  stated tolerance (a new check kind, to define with the test linter).

## 6. Libraries: tables of translations

A library is supported by a table: for each operation (function, method, operator), how it translates into §3. An
entry states:

- **the pattern** it matches: `numpy.sum(x, axis=a)`, `x.sum(a)`, `x @ y`, `scipy.special.expit(x)`,
  `torch.distributions.Normal(mu, sd).sample()`;
- **the translation**: a calculus term in its arguments: `sum(x, axis=a)` → Aggregate-Σ over the index of x's
  axis a;
- **how axes are identified**: by position (numpy, PyTorch: axis 0 is the first index of x's signature; broadcasting
  aligns from the last axis) or by name (xarray);
- **its tests**: examples with expected results, chiefly of its edge cases (above). Later, when the tables grow
  (PyTorch) or an AI writes entries, generated tests (§7).

An operation with no entry is an Operator: shown by name, never interpreted. The linter uses the same tables: clean
code may use any operation, but those without entries are reported as places where the mathematics stays opaque.

Library policy: numpy (with scipy.special and scipy.stats) first; PyTorch next, for deep learning; another library
only when a project needs it and neither covers it.

Built so far (2026-10-29): numpy (elementwise, reductions, broadcasting, basic and advanced indexing, matmul and
einsum, linalg, made arrays, diag, random draws); scipy.special and scipy.stats (densities and draws); Python's
builtins and math; polars (tables: columns, filter, with_columns, group_by/agg/len); PyTorch (elementwise, `dim=`
reductions, softmax and log-softmax, cross-entropy, MSE, `linear` and `nn.Linear` layers, autograd's gradients).
Tables extend the calculus with a value of named columns over rows (`Table`) and a group's key (`Key`): a group-by
is an aggregate whose condition ties a row's key to the group's, key_d = k, the same form as a segment sum.

`headway lint` measures coverage (the share of a module's expressions written as math) and lists what is opaque.
On 2026-10-29: clean code 94-100%; unclean original code (methembed's core, PyDESeq2 read statically) 46-100% per
file, the opaque part mostly changes in place, pandas, and arrays made at sizes no index set gives.

## 7. The reference evaluator, and checking what is shown

`evaluate.py` runs terms on real values by the calculus's own meaning: an aggregate combines over the axis its
index set names, wherever that axis is; an elementwise operation aligns its operands by index set, not by
position; only where index sets are not known does it follow numpy's positional rules (as the term then says).
Operators are called (what one means is what its function does); a draw is taken from the generator the code
passes, so a run from the same seed draws the same values when the draws come in the same order, in the shape
the code's `size` gives. A recorded call keeps NumPy's global random state as it started, and each rerun (the code,
and each form of the math) starts from it; Python's `random` and torch's generators are not kept. It uses no
table, so a wrong translation, a wrong signature or a wrong fold gives other numbers than the code.

It is used in two ways:

- **Tests of the tables** (`tests/test_core.py`): small clean functions at the edges (axes moved by advanced
  indexing, broadcasting, copies in `x.T @ x`, keepdims, stepped slices, einsum, densities, draws), each lowered,
  folded and run against what it computes itself.
- **Checks of what is shown** (`verify.py`, `headway trace`): every function of a clean module, on the arguments
  of the calls a real run made (the code's tests, watched by a profiler), its Program run as lowered and as
  folded (what the viewer shows) and compared with the code's own result. The checks have already found bugs the
  printer had before the core (a loop that updates a value from one computed from it, written as a sum).

## 8. The printer

One printer, from calculus terms to KaTeX, with all notation choices in one place:

- index letters: an index set's letter, the first letter of its name unless taken, else i, j, k, ...; a copy primed;
- symbols: the spec's for values the graph shows; short symbols for the code's other names (a switch shows the
  code's names instead);
- substitution: a Let used once is substituted; one used more than once, or named in the spec, keeps its line;
- products of fully indexed values are juxtaposed ($n_{jg}\,q_{jg}$); an Aggregate-Σ shows its index; an extent
  shows as its symbol;
- layout: one flat block per step, flush left: the specification, then the equations or the algorithm, then
  "provided that", then "where" (the settings it is decided by, the functions it uses).

## 9. Outside the core, deliberately

- Mutable state, aliasing and in-place changes: the clean subset forbids them (the linter refuses, naming the line).
- Strings, parsing, I/O, plotting: Operators at most; usually outside the steps shown.
- General recursion and data structures (trees, graphs): the algorithm block.

## 10. From codetex today to the core

Each current codetex rule becomes a table entry (numpy), a printer rule, or a calculus construct; anything that fits
none is reviewed rather than carried over. The mapping, as a checklist for the restructuring:

| codetex today | becomes |
|---|---|
| FUNCS, REDUCE, SAME, RESHAPE tables; `call()` special cases (`clip`, `where`, `outer`, `dot`, ...) | numpy table entries, each tested |
| index tracking (`dims` on `T`, `_broadcast`, `_matmul`, `_contracted`, `selection`) | §3 signatures (core) |
| jaxtyping axes, spec `dims=` | inputs' signatures |
| `_single` / `_Flow` substitution, `_reused`, aliases | printer substitution rules over Let |
| cases from `if`/`IfExp`, `cases()` | Choose / Cases |
| loops: sums and families from loops, algorithm style, inlining | Iterate, Build, Aggregate; printer |
| asserts | Precondition |
| `definitions`, `where` | Function, printer |
| conditions (`condition()`) | Guard |
| short symbols, index letters, `_indexed_tex`, `_primed` | printer |
