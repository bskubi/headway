# What the survey implies for tracing

Drawn from the 22 records (2026-10-29). Each point names the codebases that force it.

1. **One trace format for every language.** A trace is a set of boundary records: a call's function, its
   arguments and results (arrays with shapes, dtypes and labels), the options and classes in effect, and the
   branch it took. They are written as files: arrays in npz or Arrow, the rest in JSON. Tracers are thin and
   per runtime (Python first; R at function boundaries; workflows from the manager's own trace). Checking
   happens once, in Python, against the calculus.
   Forced by: deseq2-r, sctransform-r (R originals as references for the Python ports), hich (SQL), methylseq.

2. **Use the library's own instrumentation where it exists:** numpyro's trace handler (sites, shapes, plates),
   torch module hooks, Nextflow's trace and versions.yml, a SQL engine's plan.
   Forced by: pertpy-sccoda, scvi-tools, methylseq.

3. **Control the volume.**
   - Framework loops (Lightning, NUTS, IRLS over thousands of genes) record their first and last iterations and
     every k-th.
   - Families over items (genes, cells) record a sample of items in full, and counts of the path every item
     took.
   Forced by: scvi-tools, pydeseq2, batchglm, inmoose.

4. **Join traces across processes through the files they pass,** identified by content hash, so that a
   multi-stage method (CLI stages, workflow tasks, SQL then Python) is one trace.
   Forced by: methscan, allcools, hich, methylseq.

5. **Record the environment with every trace:**
   - interpreter and package versions;
   - which optional packages were importable;
   - thread and BLAS settings;
   - the global random state before and after;
   - global settings changed at import.
   Forced by: pysctransform (rpy2), sctransform-r (glmGamPoi version), methscan (seed at import), diffxpy and
   batchglm (version pairing), and every codebase that needed pins to run.

6. **Record configuration and paths per call:** the classes chosen (family, link, node types), option values, and
   the branch taken (IRLS, then optimizer, then grid); summed over items.
   Forced by: glum, statsmodels (mode flags), pydeseq2, scanpy (regress_out), mofapy2.

7. **Index sets from traces:** match arrays' sizes to named sets (from parameter names, xarray dims, plates,
   shape asserts), separating equal sizes with runs on generated data of several sizes.
   Forced by: every codebase but allcools and pertpy-sccoda.

8. **Stated meanings for anything not read:** compiled kernels, C++, SQL engines, tools. A meaning written in the
   calculus is checked on the traced calls at the boundary. It is the same mechanism for all of them.
   Forced by: harmonypy, hmmlearn, methscan (CSR walking), deseq2-r, alnbase.

## Readers worth adding, by what they unlock

- **A PPL table (numpyro, pyro):** cheap; models are declared with named index sets (pertpy-sccoda, and
  scvi-tools' distributions).
- **A SQL reader:** filters, group-bys and counts as tables and sums. It is the user's own processing (hich),
  upstream of methembed's data.
- **An xarray table:** named dims (allcools).
- **R:** a tracer before a reader. DESeq2's math is in C++, sctransform's in R.

## Status (2026-10-29)

Built in src/headway/tracing:
- the trace format (record.py);
- environment capture (env.py);
- the Python tracer on sys.monitoring, with joblib run in-process (tracer.py);
- paths and index sets from traces (analyze.py);
- math checked on recorded calls (check.py).

CLI: `record`, `paths`, `sets`, `verify`, `tex --trace`.

Also new in the core: the Lambda and Optimum terms, so scipy.optimize.minimize is read as argmin (and evaluated by
the same optimizer); `n = x.shape[k]` carries the index set's size; QR, solve(assume_a), typing.cast, an integer
math.floor; `and`/`or` short-circuit in the evaluator.

Math agreeing with the code on recorded calls:

| codebase | functions |
|---|---|
| pydeseq2 | irls_solver 30/30, fit_alpha_mle 37/37 (L-BFGS-B rerun from the math), trimmed_mean 22/22, nb_nll 50/50, wald_test 25/26, mean_absolute_deviation, fit_rough_dispersions, fit_lin_mu, trimmed_variance |
| glmpca-py | est_nb_theta 16/16, ortho 20/20 |
| methembed (original code) | sigmoid, logit, fit_intercept, binomial_fit, shrunk_logits, baselines, pca, ridge_penalty |

Since then:
- records, slices and polars frames are rebuilt from traces;
- module constants are recorded per call (methembed's tests change CHUNK);
- the sys.settrace tracer runs standalone in a project's own older Python (`--python`), and `--no-jit` traces
  numba code;
- `headway options` and `headway env` show the configuration and environment;
- object slots are written as symbols (dispersions^(var)).
- Views (part = x[:, b]) changed in place now change x in the math. Before, the math showed x unchanged: a silent
  error that verify caught in methembed's embed.scale. If it is not known whether a name is a view, the line is not
  written.

methembed's original code: 29 of 50 traced functions agree on every recorded call. The rest take objects not
recorded (a CallSource, an h5 file), return functions, or use polars beyond its table (filter with a Series, an
unsupported group_by).

Not yet:
- showing configuration and environment on pages (the commands print them);
- reading slots across methods as one step graph;
- `min(..., key=abs)`, reshape;
- SQL;
- an R tracer.

Later still (2026-10-29):
- `verify` reports agreed / disagreed / not checkable separately. Of all traced functions, none disagree
  (statsmodels 79 agreed of 119, methembed 29 of 50, pydeseq2 25 of 64); the rest could not be checked.
- The tracer stores at most 2000 calls per function (`stored`). Calls past that are counted by path only, after
  MethSCAn's no-JIT run (millions of kernel calls) was killed for memory.
- joblib's Parallel(delayed(f)(...) for ...) is read as a family of f.
- Objects are compared by their state.

To add (agreed with the user): views decided from traces. For each line where the reader cannot tell whether a
name is a view, the tracer records np.shares_memory with its base. The reader then writes the line from that; lint
suggests the explicit rewrite for clean code, and `headway trace` checks the rewrite against the original.

## Avenues to explore (agreed 2026-10-29)

- **SymPy, attached to the calculus, not replacing it.** Export the scalar parts of the calculus to SymPy to check
  hand-written gradients symbolically, to prove two implementations equivalent where they are algebraically so,
  and to offer simplified forms shown next to the faithful one. SymPy is not used as the representation: it
  canonicalizes as it builds (log1p becomes log(x+1), terms cancel and reorder), has no algorithms (loops,
  optimizers, state), and lambdify evaluates indexed sums as scalar Python loops.
- **mpmath for a high-precision reference mode of the evaluator.** Scalar primitives are computed at e.g. 40
  digits on a sample of recorded calls. This checks numpy and scipy's primitives independently, exposes
  numerically unstable formulas (log(1+x) for tiny x), and reports the code's error against the exact value of the
  math, so tolerances come from evidence.
- **SymEngine**, if symbolic differentiation in SymPy is too slow.
- **E-graphs (egglog and its kind)** to prove expressions equal by rewriting, for equivalence between
  implementations (per-gene against batched; Python against Cython paths).
