# pipeline: a headway of code in other languages

A made-up RNA-seq pipeline with no Python in it: reads are trimmed (Nextflow, `main.nf`), aligned and counted
(Snakemake, `Snakefile`), filtered (bash, `scripts/qc.sh`), normalized (R, `R/normalize.R`) and scored (Rust,
`rust/src/lib.rs`). The spec, `spec.py`, is Python as always; its code references are file paths:

| reference | resolves by |
|---|---|
| `"R/normalize.R:size_factors"`, `"scripts/qc.sh:filter_samples"`, `"Snakefile:align"` | a definition's name, found with tree-sitter |
| `"rust/src/lib.rs:Scorer.score"` | a method inside an `impl Scorer` |
| `"R/normalize.R#PSEUDOCOUNT <-"` | a text anchor: the line, or the construct starting on it |
| `"main.nf#process TRIM"` | a text anchor in a language tree-sitter lacks (Nextflow): the block under the line |
| `"config.yaml:2"` | a line |

```
./build.sh          # PYTHON=... to choose the interpreter; writes pipeline.html
```

Nothing here runs; the code is only pointed to. Rename `size_factors` in `R/normalize.R` and `headway check spec`
fails.
