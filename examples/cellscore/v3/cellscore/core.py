"""cellscore: how strongly each cell expresses a set of genes (a module), from a count matrix."""
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]


def library_size(counts: Array) -> Array:
    return counts.sum(1)


def size_factors(total: Array) -> Array:
    """Each cell's depth relative to the typical cell's."""
    return total / np.median(total)


def normalize(counts: Array, factors: Array) -> Array:
    return counts / factors[:, None]


def log_transform(y: Array) -> Array:
    return np.log1p(np.maximum(y, 0))


def center_genes(z: Array) -> Array:
    """Each gene relative to its mean over the cells."""
    return z - z.mean(0)


def module_score(z: Array, module: NDArray[np.bool_]) -> Array:
    """The mean over the module's genes."""
    return z[:, module].mean(1)


def score_cells(counts: Array, module: NDArray[np.bool_]) -> Array:
    z = log_transform(normalize(counts, size_factors(library_size(counts))))
    return module_score(center_genes(z), module)
