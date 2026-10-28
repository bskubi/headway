"""The headway spec of cellscore."""
import numpy as np

from cellscore import core
from headway import Headway

G = Headway("cellscore", "A score per cell for how strongly it expresses a module (a set of genes), from UMI counts.")
i = G.index("i", "i", "cell")
g = G.index("g", "g", "gene")
X = G.symbol("X", "X_{ig}", "counts", "UMI counts of gene $g$ in cell $i$.", dims=(i, g), role="input")
a = G.symbol("a", "a_g", "module", "1 for the module's genes, 0 for the others.", dims=(g,), role="input")
n = G.symbol("n", "n_i", "library size", "All the counts of cell $i$.", dims=(i,))
f = G.symbol("f", "f_i", "size factor", "Cell $i$'s depth relative to the median cell's.", dims=(i,))
Y = G.symbol("Y", "Y_{ig}", "normalized counts", dims=(i, g))
Z = G.symbol("Z", "Z_{ig}", "log expression", dims=(i, g))
Zc = G.symbol("Zc", r"\tilde Z_{ig}", "centered expression", "Above or below the gene's usual level.",
              dims=(i, g))
w = G.symbol("w", "w_g", "gene weight", "Down-weights genes that vary a lot between cells.", dims=(g,))
eps = G.constant("sd_floor", r"\epsilon", "spread floor", value=1e-3, code="cellscore.core:SD_FLOOR")
S = G.symbol("S", "S_i", "module score", "How strongly cell $i$ expresses the module.", dims=(i,), role="output",
             was="score")

G.group("norm", "normalization", "From counts to log expression that is comparable between cells.", was="prep")
G.group("scoring", "scoring")
G.op("libsize", "library size", r"n_i = \sum_g X_{ig}", inputs=[X], outputs=[n], code="cellscore.core:library_size",
     group="norm")
G.op("sizefactor", "size factors", r"f_i = \frac{n_i}{\operatorname{median}_{i'} n_{i'}}",
     "Depth relative to the typical cell, so normalized counts stay on the scale of the raw ones.",
     inputs=[n], outputs=[f], code="cellscore.core:size_factors", group="norm", was="normalize")
G.op("scale_counts", "scale counts", r"Y_{ig} = \frac{X_{ig}}{f_i}", "Removes the effect of sequencing depth.",
     inputs=[X, f], outputs=[Y], code="cellscore.core:normalize", group="norm", was="normalize")
G.op("log", "log transform", r"Z_{ig} = \log(1 + Y_{ig})", inputs=[Y], outputs=[Z],
     code="cellscore.core:log_transform", group="norm")
G.op("center", "center genes", r"\tilde Z_{ig} = Z_{ig} - \frac{1}{N} \sum_{i'} Z_{i'g}",
     "Highly expressed genes would otherwise dominate the score.", inputs=[Z], outputs=[Zc],
     code="cellscore.core:center_genes", group="scoring", where={"N": "the number of cells"})
G.op("weights", "gene weights", r"w_g = \frac{1}{\max(\sigma_g, \epsilon)}",
     "So that a few noisy genes cannot carry the score.", inputs=[Zc, eps], outputs=[w],
     code="cellscore.core:gene_weights", group="scoring",
     where={r"\sigma_g": "the s.d. of $\tilde Z_{ig}$ over the cells"})
G.op("weighted_score", "module score", r"S_i = \frac{\sum_g a_g w_g \tilde Z_{ig}}{\sum_g a_g w_g}",
     "The weighted mean over the module's genes.", inputs=[Zc, a, w], outputs=[S],
     code="cellscore.core:weighted_score", group="scoring", was="score")


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return rng.poisson(3.0, (6, 8)).astype(float) + 1, np.array([1, 1, 0, 0, 1, 0, 0, 1], bool)


@G.verify("libsize", "sizefactor", "scale_counts", "log", "center", "weights", "weighted_score")
def as_written() -> None:
    X, a = _data()
    n = X.sum(1)
    Z = np.log(1 + X / (n / np.median(n))[:, None])
    Zc = Z - Z.mean(0)
    w = 1 / np.maximum(Zc.std(0), 1e-3)
    assert np.allclose(core.score_cells(X, a), (a * w * Zc).sum(1) / (a * w).sum())
