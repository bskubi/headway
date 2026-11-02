"""The models on simulated data: the fits find what they were simulated with, roughly (the checks of what they
declare, and of the math, are headway trace's)."""
import numpy as np

from glm_em import steps as models


def test_glm() -> None:
    rng = np.random.default_rng(0)
    samples, genes = 12, 30
    X = np.column_stack([np.ones(samples), np.repeat([0.0, 1.0], samples // 2)])
    beta = np.column_stack([rng.uniform(2, 6, genes), rng.normal(0, 1, genes)])
    depth = rng.uniform(0.5, 2, samples)
    alpha = rng.uniform(0.05, 0.3, genes)
    counts = models.simulate(depth[:, None] * np.exp(X @ beta.T), alpha, rng)
    s = models.size_factors(counts)
    assert np.corrcoef(s, depth)[0, 1] > 0.8
    fits = np.array(models.de_fit(counts[:, :4], X, alpha[:4]))
    assert np.abs(fits[:, 1] - beta[:4, 1]).max() < 1.5


def test_mixture() -> None:
    rng = np.random.default_rng(1)
    reads = 400
    which = rng.random(reads) < 0.3
    n = rng.integers(3, 15, reads)
    m = rng.binomial(n, np.where(which, 0.9, 0.1))
    weights, p = models.em(m, n, np.array([0.3, 0.6]))
    assert abs(sorted(p)[0] - 0.1) < 0.05 and abs(sorted(p)[1] - 0.9) < 0.05
