"""Statistical models, written as clean code: counts simulated from a negative binomial, their size factors (median
of ratios), a negative binomial GLM fit by iteratively reweighted least squares (DESeq2's), and a binomial mixture of
reads' methylation fit by expectation-maximization.

Each fit declares what its result is (headway.clean): the GLM's coefficients minimize the penalized negative
log-likelihood, the mixture's weights and fractions maximize its likelihood. `headway trace glm_em.steps -- tests/` checks
those on the tests' calls, and that the math the viewer shows computes what this code computes.
"""
from __future__ import annotations

import numpy as np
from jaxtyping import Bool, Float, Integer
from scipy.special import logsumexp
from scipy.stats import binom, nbinom

from headway.clean import maximizes, minimizes


# ---- counts, simulated -------------------------------------------------------------------------------------------

def simulate(mu: Float[np.ndarray, "samples genes"], alpha: Float[np.ndarray, "genes"],
             rng: np.random.Generator) -> Integer[np.ndarray, "samples genes"]:
    """Counts with means mu and dispersions alpha (variance mu + alpha mu^2): Poisson counts at gamma-distributed
    rates."""
    rate = rng.gamma(1 / alpha, alpha * mu)
    return rng.poisson(rate)


# ---- size factors: the median of ratios ---------------------------------------------------------------------------

def size_factors(counts: Integer[np.ndarray, "samples genes"]) -> Float[np.ndarray, "samples"]:
    """Each sample's depth relative to the others: the median, over the genes counted in every sample, of its log
    count less the gene's mean log count, exponentiated."""
    with np.errstate(divide="ignore"):
        logs = np.log(counts)
    everywhere: Bool[np.ndarray, "genes"] = np.isfinite(logs).all(0)
    ratios = logs[:, everywhere] - logs[:, everywhere].mean(0)
    return np.exp(np.median(ratios, axis=1))


# ---- a negative binomial GLM, by iteratively reweighted least squares ---------------------------------------------

def nb_log_likelihood(y: Integer[np.ndarray, "samples"], mu: Float[np.ndarray, "samples"], alpha: float) -> float:
    """Counts y at means mu and dispersion alpha: a negative binomial of n = 1/alpha successes and success
    probability 1/(1 + alpha mu)."""
    return nbinom.logpmf(y, 1 / alpha, 1 / (1 + alpha * mu)).sum()


def penalized_nll(y: Integer[np.ndarray, "samples"], s: Float[np.ndarray, "samples"],
                  X: Float[np.ndarray, "samples coefs"], alpha: float, beta: Float[np.ndarray, "coefs"]) -> float:
    """The GLM's negative log-likelihood, its means s exp(X beta), with a small ridge penalty on beta."""
    mu = s * np.exp(X @ beta)
    return -nb_log_likelihood(y, mu, alpha) + RIDGE / 2 * (beta ** 2).sum()


RIDGE = 1e-6


@minimizes(penalized_nll, over=("beta",), tol=1e-6)
def irls(y: Integer[np.ndarray, "samples"], s: Float[np.ndarray, "samples"], X: Float[np.ndarray, "samples coefs"],
         alpha: float, tol: float = 1e-10) -> Float[np.ndarray, "coefs"]:
    """A gene's coefficients: least squares of log(y/s) to start, then weighted least squares on the working response
    until the deviance changes by less than tol, relatively."""
    beta = np.linalg.solve(X.T @ X, X.T @ np.log(y / s + 0.1))
    deviance = 1000.0
    change = 1.0
    while change > tol:
        mu = s * np.exp(X @ beta)
        w = mu / (1 + alpha * mu)
        z = np.log(mu / s) + (y - mu) / mu
        beta = np.linalg.solve(X.T @ (w[:, None] * X) + RIDGE * np.eye(X.shape[1]), X.T @ (w * z))
        old = deviance
        deviance = -2 * nb_log_likelihood(y, s * np.exp(X @ beta), alpha)
        change = abs(deviance - old) / (abs(deviance) + 0.1)
    return beta


# ---- a binomial mixture of reads, by expectation-maximization ------------------------------------------------------

def mixture_log_likelihood(m: Integer[np.ndarray, "reads"], n: Integer[np.ndarray, "reads"],
                           weights: Float[np.ndarray, "components"], p: Float[np.ndarray, "components"]) -> float:
    """Each read's m methylated calls of n, from one of the components (in proportions weights) with its fraction
    p."""
    return logsumexp(np.log(weights) + binom.logpmf(m[:, None], n[:, None], p), axis=1).sum()


def weights_sum_to_one(weights: Float[np.ndarray, "components"]) -> float:
    return weights.sum() - 1


@maximizes(mixture_log_likelihood, over=("weights", "p"), subject_to=(weights_sum_to_one,), tol=1e-5)
def em(m: Integer[np.ndarray, "reads"], n: Integer[np.ndarray, "reads"], p: Float[np.ndarray, "components"],
       steps: int = 500) -> tuple[Float[np.ndarray, "components"], Float[np.ndarray, "components"]]:
    """The components' proportions and fractions, from equal proportions and the fractions p: each step, each read's
    responsibilities (its posterior over the components), then the proportions and fractions they weigh."""
    weights = np.full(len(p), 1 / len(p))
    for step in range(steps):
        log_r = np.log(weights) + binom.logpmf(m[:, None], n[:, None], p)
        r = np.exp(log_r - logsumexp(log_r, axis=1, keepdims=True))
        weights = r.mean(0)
        p = (r * m[:, None]).sum(0) / (r * n[:, None]).sum(0)
    return weights, p


# ---- the steps together --------------------------------------------------------------------------------------------

def de_fit(counts: Integer[np.ndarray, "samples genes"], X: Float[np.ndarray, "samples coefs"],
           alpha: Float[np.ndarray, "genes"]) -> list[Float[np.ndarray, "coefs"]]:
    """Each gene's coefficients, its counts normalized by the samples' size factors."""
    s = size_factors(counts)
    return [irls(counts[:, g], s, X, alpha[g]) for g in range(counts.shape[1])]
