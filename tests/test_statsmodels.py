"""statsmodels' table (core/libs/statsmodels_.py): each translation run by the reference evaluator (its own fits, its
own formulas) against what statsmodels computes, on random inputs and at the edges (ties and boundaries in the
corrections, offsets, weights, a design changed after its fit); and the notation."""
from __future__ import annotations

import inspect
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("statsmodels")
sys.path.insert(0, str(Path(__file__).parent))
import sm_cases as cc  # noqa: E402

from headway.core.verify import verify  # noqa: E402

R = np.random.default_rng(11)
N, K = 40, 3
X = np.column_stack([np.ones(N), R.normal(size=(N, K - 1))])
MU = np.exp(X @ np.array([0.5, 0.3, -0.4]))
DEPTH = R.uniform(0.5, 2, N)
TIES = np.r_[R.random(16) ** 3, 0.01, 0.01, 0.01, 0.5, 0.5, 1.0]

# fits: the evaluator's own IRLS against statsmodels', which stops at its tolerance: as close as trace asks
FITS = {
    "ols": dict(y=R.normal(size=N), x=X), "ols_resid": dict(y=R.normal(size=N), x=X),
    "wls": dict(y=R.normal(size=N), x=X, w=R.uniform(0.2, 2, N)),
    "poisson_glm": dict(y=R.poisson(MU * DEPTH).astype(float), x=X, depth=DEPTH),
    "poisson_mu": dict(y=R.poisson(MU * DEPTH).astype(float), x=X, depth=DEPTH),
    "gaussian_resid": dict(y=R.normal(size=N), x=X),
    "binomial_glm": dict(y=(R.random(N) < 0.4).astype(float), x=X, x_new=R.normal(size=(5, K))),
    "nb_glm": dict(y=R.negative_binomial(2, 2 / (2 + MU)).astype(float), x=X, theta=2.0),
    "gamma_log": dict(y=R.gamma(2.0, MU / 2), x=X),
    "weighted_poisson": dict(y=R.poisson(MU).astype(float), x=X, w=R.uniform(0.2, 2, N)),
    "logit": dict(y=(R.random(N) < 0.4).astype(float), x=X),
    "discrete_poisson": dict(y=R.poisson(MU).astype(float), x=X),
    "bse": dict(y=R.poisson(MU).astype(float), x=X),
    "refit": dict(y=R.normal(size=N), x=X),
}
# formulas: the same numbers
EXACT = {
    "family_parts": dict(eta=R.normal(size=N), theta=3.0), "binomial_parts": dict(eta=R.normal(size=N) * 40),
    "poisson_parts": dict(eta=np.r_[R.normal(size=N), -800.0]), "gamma_parts": dict(eta=R.uniform(0.5, 2, N)),
    "bh": dict(p=TIES), "bh_reject": dict(p=TIES, alpha=0.1), "bonferroni": dict(p=R.random(20) ** 4),
    "fdr": dict(p=TIES), "holm": dict(p=R.random(20)), "spread": dict(x=R.normal(size=N)),
    "design": dict(x=R.normal(size=N)),
}


@pytest.mark.parametrize("name", sorted(FITS))
def test_fits_agree_with_statsmodels(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = verify(getattr(cc, name), kwargs=FITS[name], rtol=1e-6, atol=1e-6, scaled=True)
    assert report.ok, report.text()


@pytest.mark.parametrize("name", sorted(EXACT))
def test_formulas_agree_with_statsmodels(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = verify(getattr(cc, name), kwargs=EXACT[name], rtol=1e-12, atol=1e-15)
    assert report.ok, report.text()


def test_the_step_up_at_its_boundary() -> None:
    """Benjamini-Hochberg's rejections where p_(r) is exactly r α / m, and where every p ties."""
    m, alpha = 8, 0.08
    for p in (np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.5]) * 1.0,
              np.full(m, 0.04), np.full(m, 0.08), np.r_[np.full(4, 0.04), np.full(4, 0.9)]):
        report = verify(cc.bh_reject, kwargs=dict(p=p, alpha=alpha), rtol=1e-12, atol=1e-15)
        assert report.ok, report.text()


def test_a_wrong_translation_is_caught(monkeypatch) -> None:
    """Not vacuous: the logit's inverse read as exp, or a rank that counts only smaller p-values, gives other
    numbers (the second only where p-values tie)."""
    import headway.core.libs.statsmodels_ as sm_
    right = sm_.inverse_link
    monkeypatch.setattr(sm_, "inverse_link", lambda link, eta: right("log" if link == "logit" else link, eta))
    assert not verify(cc.binomial_glm, kwargs=FITS["binomial_glm"], rtol=1e-6, atol=1e-6, scaled=True).ok
    monkeypatch.undo()
    right_across = sm_._across
    monkeypatch.setattr(sm_, "_across", lambda fn, a, b: right_across("lt" if fn == "le" else fn, a, b))
    assert not verify(cc.bh, kwargs=dict(p=TIES)).ok


def tex(fn) -> str:                                     # noqa: ANN001
    from headway import codetex
    return codetex.transcribe(inspect.getsource(fn), {}, hints=True, file=cc.__file__)


def test_notation() -> None:
    # a fit: the coefficients where the likelihood is greatest, with the model's mean written out
    assert tex(cc.ols) == (r"\operatorname{ols}\left(y_{i}, x_{ic}\right) = \left(\operatorname*{arg\,min}_{\beta} "
                           r"\sum_{i} \left(y_{i} - \sum_{c'} x_{ic'}\,\beta_{c'}\right)^{2}\right)_{c}")
    assert (r"\operatorname*{arg\,max}_{\beta} \sum_{c} \log \operatorname{Poisson}\left(y_{c} \mid \exp\left(\sum_{i'} "
            r"x_{ci'}\,\beta_{i'} + \log\left(\mathit{depth}_{c}\right)\right)\right)") in tex(cc.poisson_glm)
    assert r"\operatorname{NegBin}\left(y_{c} \mid \theta, " in tex(cc.nb_glm)
    # what is read from a fit: its means, its residuals, a prediction for new rows
    assert tex(cc.binomial_glm).startswith(r"\operatorname{binomial\_glm}\left(y_{c}, x_{ci}, \mathit{x\_new}_{pi}"
                                           r"\right) = \sigma\left(\sum_{i} \mathit{x\_new}_{pi}\,")
    # what is not a formula here: the library's, by name
    assert r"\frac{\mathit{res}_{i}}{\mathit{res.bse}}" in tex(cc.bse)
    # a design changed after the fit: its fitted values are not written with the new one
    assert r"\right).\mathit{fittedvalues}" in tex(cc.refit)
    # Benjamini-Hochberg: each p-value's rank, the smallest adjusted value from it on
    assert (r"\min\left(\min_{g' : p_{g'} \ge p_{g}} \frac{\left|\mathit{genes}\right|\,p_{g'}}"
            r"{\left|\left\{g'' : p_{g''} \le p_{g'}\right\}\right|}, 1\right)") in tex(cc.bh)


def test_lint_knows_what_is_translated() -> None:
    from headway.lint import lint
    warned = {f.function for f in lint("sm_cases").findings if f.level in ("warning", "error")}
    assert warned == {"holm"}                           # the default and Holm's methods: operators, and said so
