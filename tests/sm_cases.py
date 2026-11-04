"""Clean functions that use statsmodels, for test_statsmodels.py: each lowered, run by the reference evaluator (its
own fits) and compared with what statsmodels computes."""
from __future__ import annotations

import numpy as np
import statsmodels.api as sm
import statsmodels.genmod.families as smf
from jaxtyping import Float
from statsmodels.robust import mad
from statsmodels.stats.multitest import fdrcorrection, multipletests


# fits: the coefficients, and what is read from them

def ols(y: Float[np.ndarray, "cells"], x: Float[np.ndarray, "cells covariates"]) -> Float[np.ndarray, "covariates"]:
    return sm.OLS(y, x).fit().params


def ols_resid(y: Float[np.ndarray, "cells"], x: Float[np.ndarray, "cells covariates"]) -> Float[np.ndarray, "cells"]:
    res = sm.OLS(y, x).fit()
    return res.resid


def wls(y: Float[np.ndarray, "cells"], x: Float[np.ndarray, "cells covariates"],
        w: Float[np.ndarray, "cells"]) -> Float[np.ndarray, "cells"]:
    res = sm.WLS(y, x, weights=w).fit(method="qr")
    return res.fittedvalues


def poisson_glm(y: Float[np.ndarray, "cells"], x: Float[np.ndarray, "cells covariates"],
                depth: Float[np.ndarray, "cells"]) -> Float[np.ndarray, "covariates"]:
    model = sm.GLM(y, x, family=sm.families.Poisson(), offset=np.log(depth))
    res = model.fit()
    return res.params


def poisson_mu(y: Float[np.ndarray, "cells"], x: Float[np.ndarray, "cells covariates"],
               depth: Float[np.ndarray, "cells"]) -> Float[np.ndarray, "cells"]:
    res = sm.GLM(y, x, family=sm.families.Poisson(), exposure=depth).fit()
    return res.mu


def gaussian_resid(y: Float[np.ndarray, "cells"], x: Float[np.ndarray, "cells covariates"]) -> Float[np.ndarray, "cells"]:
    result = sm.GLM(y, x, family=sm.families.Gaussian()).fit()      # scanpy's regress_out
    return result.resid_response


def binomial_glm(y: Float[np.ndarray, "cells"], x: Float[np.ndarray, "cells covariates"],
                 x_new: Float[np.ndarray, "points covariates"]) -> Float[np.ndarray, "points"]:
    res = sm.GLM(y, x, family=sm.families.Binomial()).fit()
    return res.predict(x_new)


def nb_glm(y: Float[np.ndarray, "cells"], x: Float[np.ndarray, "cells covariates"],
           theta: float) -> Float[np.ndarray, "covariates"]:
    res = sm.GLM(y, x, family=sm.families.NegativeBinomial(alpha=1 / theta)).fit()
    return res.params


def gamma_log(y: Float[np.ndarray, "cells"], x: Float[np.ndarray, "cells covariates"]) -> Float[np.ndarray, "covariates"]:
    res = sm.GLM(y, x, family=sm.families.Gamma(link=sm.families.links.Log())).fit()
    return res.params


def weighted_poisson(y: Float[np.ndarray, "cells"], x: Float[np.ndarray, "cells covariates"],
                     w: Float[np.ndarray, "cells"]) -> Float[np.ndarray, "covariates"]:
    return sm.GLM(y, x, family=sm.families.Poisson(), var_weights=w).fit().params


def logit(y: Float[np.ndarray, "cells"], x: Float[np.ndarray, "cells covariates"]) -> Float[np.ndarray, "cells"]:
    res = sm.Logit(y, x).fit(disp=0)
    return res.predict()


def discrete_poisson(y: Float[np.ndarray, "cells"], x: Float[np.ndarray, "cells covariates"]) -> Float[np.ndarray, "covariates"]:
    return sm.Poisson(y, x).fit(disp=0).params


def bse(y: Float[np.ndarray, "cells"], x: Float[np.ndarray, "cells covariates"]) -> Float[np.ndarray, "covariates"]:
    res = sm.GLM(y, x, family=sm.families.Poisson()).fit()
    return res.params / res.bse                         # bse: not a formula here, the library's


# families as glmpca uses them

def family_parts(eta: Float[np.ndarray, "cells"], theta: float) -> Float[np.ndarray, "cells"]:
    family = smf.NegativeBinomial(alpha=1 / theta)
    mu = family.link.inverse(eta)
    w = 1 / family.variance(mu)
    return (mu ** 2) * w + family.link.inverse_deriv(eta) + family.link.deriv(mu) + family.link(mu)


def binomial_parts(eta: Float[np.ndarray, "cells"]) -> Float[np.ndarray, "cells"]:
    family = smf.Binomial()
    p = family.link.inverse(eta)
    return family.variance(p) + family.link.inverse_deriv(eta) + family.link.deriv(p) + family.link(p)


def poisson_parts(eta: Float[np.ndarray, "cells"]) -> Float[np.ndarray, "cells"]:
    family = smf.Poisson()
    m = family.link.inverse(eta)
    return family.variance(m) + family.link.deriv(m)


def gamma_parts(eta: Float[np.ndarray, "cells"]) -> Float[np.ndarray, "cells"]:
    family = smf.Gamma()
    m = family.link.inverse(eta)
    return family.variance(m) + family.link.inverse_deriv(eta) + family.link.deriv(m) + family.link(m)


# multiple testing

def bh(p: Float[np.ndarray, "genes"]) -> Float[np.ndarray, "genes"]:
    _, q, _, _ = multipletests(p, method="fdr_bh")
    return q


def bh_reject(p: Float[np.ndarray, "genes"], alpha: float) -> tuple:
    return multipletests(p, alpha=alpha, method="fdr_bh")


def bonferroni(p: Float[np.ndarray, "genes"]) -> tuple:
    return multipletests(p, 0.1, "bonferroni")


def fdr(p: Float[np.ndarray, "genes"]) -> tuple:
    return fdrcorrection(p, alpha=0.2)


def holm(p: Float[np.ndarray, "genes"]) -> Float[np.ndarray, "genes"]:
    return multipletests(p, method="holm")[1]           # not translated: the library's


def spread(x: Float[np.ndarray, "cells"]) -> float:
    return mad(x)


def design(x: Float[np.ndarray, "cells"]) -> Float[np.ndarray, "cells"]:
    return sm.add_constant(x) @ np.array([1.0, 2.0])


# what is not trusted: a design changed after the fit

def refit(y: Float[np.ndarray, "cells"], x: Float[np.ndarray, "cells covariates"]) -> Float[np.ndarray, "cells"]:
    res = sm.OLS(y, x).fit()
    x = x * 2
    return res.fittedvalues + x[:, 0]
