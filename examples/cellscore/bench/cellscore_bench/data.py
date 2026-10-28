"""Simulated count matrices with a known module activity per cell (made up; for the example only)."""
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Data:
    counts: NDArray[np.floating]      # cells x genes
    module: NDArray[np.bool_]         # the module's genes
    active: NDArray[np.bool_]         # the cells in which the module is on: the truth the scores are judged by

    @property
    def depth(self) -> NDArray[np.floating]:
        return self.counts.sum(1)


# the datasets, by the simulation's settings (the suite gives each dataset these values)
DATASETS = {
    "even": dict(cells=300, genes=200, module=20, depth_sd=0.3, noisy=0, effect=1.0),
    "depth": dict(cells=300, genes=200, module=20, depth_sd=1.1, noisy=0, effect=1.0),
    "noisy": dict(cells=300, genes=200, module=20, depth_sd=0.3, noisy=8, effect=1.0),
    "heldout": dict(cells=400, genes=250, module=24, depth_sd=0.7, noisy=5, effect=0.8),
}
ON = 0.3                              # the chance that the module is on in a cell
SCALE = 4.0                           # counts per unit of depth and level


def rates(depth: NDArray[np.floating], base: NDArray[np.floating], active: NDArray[np.bool_],
          module: NDArray[np.bool_], effect: float, swing: NDArray[np.floating]) -> NDArray[np.floating]:
    """Each cell's expected count of each gene."""
    rate = SCALE * np.outer(depth, base)
    rate[np.ix_(active, module)] *= 1 + effect
    return rate * swing


def simulate(name: str, seed: int) -> Data:
    p = DATASETS[name]
    cells, genes = p["cells"], p["genes"]
    rng = np.random.default_rng([seed, sum(map(ord, name))])
    # the genes: each one's usual level, and the module
    base = rng.lognormal(-1.0, 1.2, genes)
    module = np.zeros(genes, bool)
    module[rng.choice(genes, p["module"], replace=False)] = True
    # the truth: the cells with the module on
    active = rng.random(cells) < ON
    # the rates: depth, and module genes that swing for reasons of their own
    depth = rng.lognormal(0, p["depth_sd"], cells)
    swing = np.ones((cells, genes))
    swing[:, np.flatnonzero(module)[: p["noisy"]]] = rng.lognormal(0, 1.5, (cells, p["noisy"]))
    rate = rates(depth, base, active, module, p["effect"], swing)
    # the counts
    return Data(rng.poisson(rate).astype(float), module, active)
