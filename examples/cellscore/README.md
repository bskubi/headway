# cellscore: a method developed in four versions

A small example for comparing versions of a headway. cellscore scores how strongly each cell expresses a module (a
set of genes), from a UMI count matrix. Each `vN/` directory is the package as it was at one point of its
development, standing in for a git checkout.

| version | what changed | what the comparison shows |
|---|---|---|
| v1 | library size, normalize to `s` counts, log, mean over the module | |
| v2 | genes centered before scoring | a step and a value added; the score rewired, its equation changed |
| v3 | size factors (median-relative) instead of a fixed scale; the log clips at 0 | `normalize` split in two (`was="normalize"` on both); the scale setting gone; the group renamed (`was="prep"`); the log's code changed under an unchanged equation |
| v4 | genes weighted by 1 / spread, with a floor | a step, a value and a constant added; the score step and its output renamed (`was="score"`) |

```
./build.sh          # PYTHON=... to choose the interpreter
```

It runs the benchmark (`bench/`: simulated cells whose module is known to be on or off; made-up, for the example)
on every version into `results/`, and `bench/run.py` builds each version's headway with its results
(`headways/`) through headway's Python API, as a benchmark script in a method's own repository would
(`python bench/run.py v4` redoes one version). Then it writes:

- `compare.html`: v4 with v1–v3 in lanes above it; **Results** switches to the benchmark results;
- `suite.html`: the benchmark's own headway: the simulation that makes the datasets and the metrics, as steps with
  their equations and checks, and the datasets with their settings;
- `changes.txt` (`headway diff`) and `results.txt` (`headway results`).

The benchmark has three metrics (separation, higher is better; depth dependence, lower; split agreement, higher),
three development datasets and one held out, and two objectives (a weighted mean and a mean rank). v1 was not run on
the noisy dataset, and only v1 and v4 on the held-out one. What the results show: centering (v2) changes nothing,
since it shifts every cell's score alike; size factors (v3) help most, above all with uneven depth; weighting (v4)
helps on the noisy module. The datasets are simulated by the suite's own steps (module on or off, rates, counts);
each dataset is those steps with its values of their settings, taken from `bench/cellscore_bench/data.py`.
The replicates are named by seed, so the versions' values pair up. The suite also declares a figure, "score against
depth" (seed 0: each cell's score against its sequencing depth, drawn into `results/figs/`): in the Results view it
is a column beside the metrics, a thumbnail per version and dataset; click it to see the versions' images side by
side, large. Each version is labelled with keys
saying which options it has (`centered`, `weighted`).
The tests (`tests/test_headway.py`) build the four versions and check the comparison.
