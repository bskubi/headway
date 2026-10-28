"""The headway spec of cellscore."""
import numpy as np

from cellscore import core
from headway import Headway

G = Headway("cellscore", "A score per cell for how strongly it expresses a module (a set of genes), from UMI counts.")
i = G.index("i", "i", "cell")
g = G.index("g", "g", "gene")
X = G.symbol("X", "X_{ig}", "counts", "UMI counts of gene $g$ in cell $i$.", dims=(i, g), role="input")
a = G.symbol("a", "a_g", "module", "1 for the module's genes, 0 for the others.", dims=(g,), role="input")
s = G.setting("scale", "s", "scale", "The counts each cell is scaled to.", value=1e4, code="cellscore.core:normalize")
n = G.symbol("n", "n_i", "library size", "All the counts of cell $i$.", dims=(i,))
Y = G.symbol("Y", "Y_{ig}", "normalized counts", dims=(i, g))
Z = G.symbol("Z", "Z_{ig}", "log expression", dims=(i, g))
Zc = G.symbol("Zc", r"\tilde Z_{ig}", "centered expression", "Above or below the gene's usual level.",
              dims=(i, g))
S = G.symbol("score", "S_i", "module score", "How strongly cell $i$ expresses the module.", dims=(i,), role="output")

G.group("prep", "preprocess", "From counts to log expression that is comparable between cells.")
G.group("scoring", "scoring")
G.op("libsize", "library size", r"n_i = \sum_g X_{ig}", inputs=[X], outputs=[n], code="cellscore.core:library_size",
     group="prep")
G.op("normalize", "normalize", r"Y_{ig} = s\,\frac{X_{ig}}{n_i}", "Removes the effect of sequencing depth.",
     inputs=[X, n, s], outputs=[Y], code="cellscore.core:normalize", group="prep")
G.op("log", "log transform", r"Z_{ig} = \log(1 + Y_{ig})", inputs=[Y], outputs=[Z],
     code="cellscore.core:log_transform", group="prep")
G.op("center", "center genes", r"\tilde Z_{ig} = Z_{ig} - \frac{1}{N} \sum_{i'} Z_{i'g}",
     "Highly expressed genes would otherwise dominate the score.", inputs=[Z], outputs=[Zc],
     code="cellscore.core:center_genes", group="scoring", where={"N": "the number of cells"})
G.op("score", "module score", r"S_i = \frac{\sum_g a_g \tilde Z_{ig}}{\sum_g a_g}", "The mean over the module's genes.",
     inputs=[Zc, a], outputs=[S], code="cellscore.core:module_score", group="scoring")


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return rng.poisson(3.0, (6, 8)).astype(float) + 1, np.array([1, 1, 0, 0, 1, 0, 0, 1], bool)


@G.verify("libsize", "normalize", "log", "center", "score")
def as_written() -> None:
    X, a = _data()
    n = X.sum(1)
    Z = np.log(1 + 1e4 * X / n[:, None])
    Zc = Z - Z.mean(0)
    assert np.allclose(core.score_cells(X, a), (a * Zc).sum(1) / a.sum())
