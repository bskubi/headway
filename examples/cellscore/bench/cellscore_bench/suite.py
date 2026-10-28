"""The cellscore benchmark, as a headway suite: its datasets, its metrics (each with its equation, its code and
a check that the code computes the equation) and the objectives the methods are ranked by. The data are simulated
and the metrics are for the example; they are not a real benchmark."""
import numpy as np

from cellscore_bench import data, metrics
from headway import Suite

B = Suite("cellscore benchmark", "Scores of a module's activity per cell, against simulated cells whose module is "
          "known to be on or off.", version="1")
i = B.index("i", "i", "cell")
g = B.index("g", "g", "gene")

# the data: simulated, so the datasets are the simulation's steps (below) with different settings
B.group("data", "simulated data", "How each dataset's cells are made. The datasets differ only in these settings.")
N = B.setting("cells", "N", "cells", "The number of cells.", code="cellscore_bench.data:simulate#cells, genes =")
G_ = B.setting("genes", "G", "genes", "The number of genes.", code="cellscore_bench.data:simulate#cells, genes =")
m = B.setting("module", "m", "module size", "The number of genes in the module.")
sd = B.setting("depth_sd", r"\sigma_d", "depth spread",
               "The s.d. of log depth: how much more some cells are sequenced than others.")
k = B.setting("noisy", "k", "noisy genes", "How many of the module's genes swing at random, module on or off.")
eff = B.setting("effect", r"\varepsilon", "effect", "How much more the module's genes are expressed when it is on.")
pi = B.constant("on", r"\pi", "share on", "The chance that the module is on in a cell.", value=0.3,
                code="cellscore_bench.data:ON")
y = B.symbol("y", "y_i", "truth", "1 if the module is on in cell $i$, else 0: what the scores are judged by.",
             dims=(i,))
lam = B.symbol("lam", r"\lambda_{ig}", "rate", "The expected count of gene $g$ in cell $i$.", dims=(i, g))
X = B.symbol("X", "X_{ig}", "counts", "The simulated counts: what the method is given.", dims=(i, g), role="output")
n = B.symbol("n", "n_i", "depth", "The total counts of cell $i$.", dims=(i,))
B.op("truth", "module on or off", r"y_i \sim \mathrm{Bernoulli}(\pi), \quad i = 1, \dots, N",
     inputs=[N, pi], outputs=[y], group="data", code="cellscore_bench.data:simulate#the truth")
B.op("rates", "rates", r"\lambda_{ig} = 4\, d_i\, b_g \left(1 + \varepsilon\, y_i\, [g \in M]\right) \xi_{ig}",
     "The module raises its genes' expression in the cells where it is on; cells differ in depth, and $k$ of the "
     "module's genes swing for reasons of their own, which a score should not be fooled by.",
     inputs=[y, G_, m, sd, k, eff], outputs=[lam], group="data",
     code=["cellscore_bench.data:simulate#the rates", "cellscore_bench.data:rates"],
     where={r"d_i \sim \mathrm{LogNormal}(0, \sigma_d^2)": "the cell's depth",
            r"b_g \sim \mathrm{LogNormal}(-1, 1.2^2)": "the gene's usual level",
            "M": "the module: $m$ of the $G$ genes, at random",
            r"\xi_{ig}": r"$\sim \mathrm{LogNormal}(0, 1.5^2)$ for $k$ of the module's genes, else 1"})
B.op("counts", "counts", r"X_{ig} \sim \mathrm{Poisson}(\lambda_{ig})", inputs=[lam], outputs=[X], group="data",
     code="cellscore_bench.data:simulate#the counts")
B.op("depth", "depth", r"n_i = \sum_g X_{ig}", inputs=[X], outputs=[n], group="data",
     code="cellscore_bench.data:Data.depth")

# the method's scores: S from all of the module's genes, S^A and S^B from two random halves of them
S = B.symbol("S", "S_i", "score", "The method's score of cell $i$.", dims=(i,), role="input")
SA = B.symbol("SA", "S^A_i", "score on half A", "The score from a random half of the module's genes.", dims=(i,),
              role="input")
SB = B.symbol("SB", "S^B_i", "score on half B", "The score from the other half.", dims=(i,), role="input")

B.dataset("even", "even depth", "Cells of similar depth and a clean module: the easy case.",
          values=data.DATASETS["even"], code='cellscore_bench.data:DATASETS#"even"..),')
B.dataset("depth", "uneven depth", "As even depth, but cells differ about tenfold in depth, which a score that "
          "follows depth gets wrong.", values=data.DATASETS["depth"], code='cellscore_bench.data:DATASETS#"depth"..),')
B.dataset("noisy", "noisy module", "As even depth, but 8 of the module's 20 genes swing at random in every cell.",
          values=data.DATASETS["noisy"], code='cellscore_bench.data:DATASETS#"noisy"..),')
B.dataset("heldout", "mixed", "Different sizes, depths, noise and a weaker effect; not looked at during "
          "development.", heldout=True, values=data.DATASETS["heldout"],
          code='cellscore_bench.data:DATASETS#"heldout"..),',
          facts={"chosen": "before v1, by someone other than the method's developer",
                 "looked at": "only to check v1 and v4"})

B.metric("auroc", r"\mathrm{AUROC}", "separation",
         "How often a cell with the module on scores above one with it off: 1 is perfect, 0.5 is chance.",
         equation=r"\mathrm{AUROC} = \frac{1}{n_1 n_0} \sum_{i: y_i = 1} \sum_{j: y_j = 0} "
                  r"\left( [S_i > S_j] + \tfrac{1}{2} [S_i = S_j] \right)",
         inputs=[S, y], better="higher", range=(0, 1), code="cellscore_bench.metrics:auroc",
         where={"n_1, n_0": "the numbers of cells with the module on and off"},
         source="the Mann-Whitney U statistic, scaled")
B.metric("depth_dep", r"\rho_n", "depth dependence",
         "How much the score follows sequencing depth, which it should not: 0 is none.",
         equation=r"\rho_n = \left| \operatorname{corr}\left(\operatorname{rank} S, "
                  r"\operatorname{rank} n\right) \right|",
         inputs=[S, n], better="lower", range=(0, 1), code="cellscore_bench.metrics:depth_dependence",
         source="Spearman's rank correlation")
B.metric("split", r"r_{AB}", "split agreement",
         "Whether the score depends on which of the module's genes are used: 1 is not at all.",
         equation=r"r_{AB} = \operatorname{corr}(S^A, S^B)", inputs=[SA, SB], better="higher", range=(-1, 1),
         code="cellscore_bench.metrics:split_agreement")

B.objective("overall", "overall", "Separation counts twice; each metric scaled by its range so that 1 is best, "
            "averaged over the development datasets.", metrics={"auroc": 2, "depth_dep": 1, "split": 1},
            tex=r"\tfrac{1}{4}\left(2\,\mathrm{AUROC} + (1 - \rho_n) + \tfrac{1 + r_{AB}}{2}\right)")
B.objective("rank", "mean rank", "The mean rank over every metric and development dataset (1 is best); it depends "
            "on which methods are compared.", metrics=["auroc", "depth_dep", "split"], how="rank")


@B.verify("rates")
def rates_as_written() -> None:
    rng = np.random.default_rng(3)
    d, b, xi = rng.lognormal(size=6), rng.lognormal(size=5), rng.lognormal(size=(6, 5))
    on, module = np.arange(6) % 2 == 0, np.arange(5) < 2
    lam = 4 * d[:, None] * b[None, :] * (1 + 0.7 * on[:, None] * module[None, :]) * xi
    assert np.allclose(data.rates(d, b, on, module, 0.7, xi), lam)


@B.verify("auroc")
def auroc_as_written() -> None:
    rng = np.random.default_rng(0)
    s, on = np.round(rng.normal(size=40), 1), rng.random(40) < 0.4      # rounded: some ties
    pairs = [(a > b) + 0.5 * (a == b) for a in s[on] for b in s[~on]]
    assert np.isclose(metrics.auroc(s, on), np.mean(pairs))


@B.verify("depth_dep")
def depth_as_written() -> None:
    rng = np.random.default_rng(1)
    s, d = rng.normal(size=30), rng.lognormal(size=30)
    rs, rd = s.argsort().argsort(), d.argsort().argsort()
    assert np.isclose(metrics.depth_dependence(s, d), abs(np.corrcoef(rs, rd)[0, 1]))


@B.verify("split")
def split_as_written() -> None:
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=30), rng.normal(size=30)
    assert np.isclose(metrics.split_agreement(a, b), np.corrcoef(a, b)[0, 1])
