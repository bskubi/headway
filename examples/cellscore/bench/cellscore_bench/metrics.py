"""The benchmark's metrics: how well a score separates the cells with the module on, how much it follows
sequencing depth, and how much it depends on which of the module's genes are used."""
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]


def _ranks(x: Array) -> Array:
    """Ranks from 1, ties sharing their mean rank."""
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x[np.diff(np.sort(x), prepend=np.nan) == 0]):
        r[x == v] = r[x == v].mean()
    return r


def auroc(score: Array, active: NDArray[np.bool_]) -> float:
    """The Mann-Whitney statistic, from the ranks."""
    r = _ranks(score)
    n1, n0 = active.sum(), (~active).sum()
    return float((r[active].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def depth_dependence(score: Array, depth: Array) -> float:
    """|Spearman correlation| of the score with the cells' total counts."""
    return float(abs(np.corrcoef(_ranks(score), _ranks(depth))[0, 1]))


def split_agreement(score_a: Array, score_b: Array) -> float:
    """Pearson correlation of the scores from two random halves of the module."""
    return float(np.corrcoef(score_a, score_b)[0, 1])
