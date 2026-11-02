"""A small package for the tracer's tests: a per-item IRLS fit with a fallback, and an optimizer fit."""
import numpy as np
from scipy.optimize import minimize


def fit(counts, size_factors, X, disp, maxiter=50):
    beta = np.zeros(X.shape[1])
    mu = size_factors * np.exp(X @ beta)
    for it in range(maxiter):
        W = mu / (1 + mu * disp)
        z = np.log(mu / size_factors) + (counts - mu) / mu
        H = (X.T * W) @ X
        new = np.linalg.solve(H, X.T @ (W * z))
        if np.abs(new).max() > 30:
            def f(b):
                m = size_factors * np.exp(X @ b)
                return np.sum(m - counts * np.log(m))
            res = minimize(f, np.zeros(X.shape[1]), method="L-BFGS-B")
            return res.x
        if np.abs(new - beta).max() < 1e-8:
            break
        beta = new
        mu = size_factors * np.exp(X @ beta)
    return beta


def fit_all(counts, size_factors, X, disp):
    return np.array([fit(counts[:, g], size_factors, X, disp[g]) for g in range(counts.shape[1])])


def poisson_rate(counts, exposure):
    res = minimize(lambda r: np.sum(exposure * r[0] - counts * np.log(exposure * r[0])), np.array([1.0]),
                   method="L-BFGS-B", bounds=[(1e-9, None)])
    return res.x


def jitter(x, scale):
    """Draws from numpy's global generator: checkable only with the generator as the call found it."""
    return x + scale * np.random.normal(size=x.shape)
