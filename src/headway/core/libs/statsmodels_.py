"""statsmodels' table: regressions as the coefficients that maximize their likelihood (OLS, WLS, GLM, Logit, Poisson),
what a fit gives as its formulas (coefficients, fitted values, residuals, predictions), GLM families and links as
theirs (variance, link, inverse link and their derivatives, statsmodels' clipping included), and multiple testing
corrections (Benjamini-Hochberg and Bonferroni).

A model (sm.GLM(y, X, family=...)) is an operator the code makes (the library's own object); its fit is a Fit term,
which the evaluator fits by its own IRLS or least squares (not the library's), so a wrong translation gives other
coefficients. What a fit has that no formula here says (its standard errors, its p-values) is the library's, read
from the code's own fit, as an operator's value is.

Only what is exact is translated: a fit by a method that stops short of the optimum (lbfgs, bfgs), a family or link
not listed, a 2-column binomial response, NaN handling (missing=): operators, as written."""
from __future__ import annotations

from headway.core.libs import Call, Table, register
from headway.core.libs.arrays import copy_of, elementwise, matmul
from headway.core.terms import (FIT, Aggregate, Align, Apply, Axis, Binder, Const, Density, Dist, Expr, Extent, Field,
                                Fit, Lambda, Op, Text, Tup, Var, broadcast)

SM = register(Table("statsmodels"))
ND = "numpy.ndarray"
MODEL = "statsmodels.model"
FAMILY = "statsmodels.family"
LINK = "statsmodels.link"
FLOAT_EPS = 2.220446049250313e-16                       # np.finfo(float).eps: where statsmodels clips a mean

FAMILY_MODULES = ("statsmodels.api.families", "statsmodels.genmod.families", "statsmodels.genmod.families.family",
                  "statsmodels.genmod.api.families", "statsmodels.api.genmod.families")
LINK_MODULES = tuple(m + ".links" for m in FAMILY_MODULES if not m.endswith(".family")) + (
    "statsmodels.genmod.families.links",)
FAMILIES = {"Gaussian": "gaussian", "Poisson": "poisson", "Binomial": "binomial", "Gamma": "gamma",
            "NegativeBinomial": "negative_binomial"}
DEFAULT_LINK = {"gaussian": "identity", "poisson": "log", "binomial": "logit", "gamma": "inverse_power",
                "negative_binomial": "log"}
LINKS = {"Log": "log", "log": "log", "Logit": "logit", "logit": "logit", "Identity": "identity",
         "identity": "identity", "InversePower": "inverse_power", "inverse_power": "inverse_power"}
LINK_CLASS = {"log": "Log", "logit": "Logit", "identity": "Identity", "inverse_power": "InversePower"}
MODELS = {"statsmodels.api.GLM": "glm", "statsmodels.genmod.generalized_linear_model.GLM": "glm",
          "statsmodels.genmod.api.GLM": "glm",
          "statsmodels.api.OLS": "ols", "statsmodels.regression.linear_model.OLS": "ols",
          "statsmodels.api.WLS": "wls", "statsmodels.regression.linear_model.WLS": "wls",
          "statsmodels.api.Logit": "logit", "statsmodels.discrete.discrete_model.Logit": "logit",
          "statsmodels.api.Poisson": "poisson", "statsmodels.discrete.discrete_model.Poisson": "poisson"}
# what each model's fit may be told and still reach the optimum (how far, and what it reports, aside)
FIT_KWARGS = {"glm": {"maxiter", "tol", "scale", "cov_type", "cov_kwds", "use_t", "disp", "start_params"},
              "ols": {"method", "cov_type", "cov_kwds", "use_t"}, "wls": {"method", "cov_type", "cov_kwds", "use_t"},
              "logit": {"disp", "maxiter", "start_params", "full_output", "cov_type", "cov_kwds", "use_t"},
              "poisson": {"disp", "maxiter", "start_params", "full_output", "cov_type", "cov_kwds", "use_t"}}


def _literal(e: Expr | None) -> object:
    """A constant's value (None for None, or for an argument not given); a term that is not one, itself."""
    return e.value if isinstance(e, (Const, Text)) else e


def _c(v: float) -> Const:
    return Const(v, sig=(), type="float")


def _f(fn: str, *args: Expr) -> Expr:
    """An elementwise primitive, of the arguments' broadcast signature."""
    out = elementwise(fn, list(args), ND if any(a.sig for a in args) else "float")
    return out


# ---- families and links ------------------------------------------------------------------------------------------

def _family_of(op: Expr | None) -> tuple[str, str, Expr | None] | None:
    """(family, link, alpha) of a family the code makes (sm.families.Poisson(), NegativeBinomial(alpha=0.5),
    Gamma(link=sm.families.links.Log())); None if it is not one of those translated."""
    if not isinstance(op, Op) or op.type != FAMILY:
        return None
    family = FAMILIES.get(op.qualname.rpartition(".")[2])
    if family is None:
        return None
    kw = dict(op.kwargs)
    names = ("link", "alpha") if family == "negative_binomial" else ("link",)
    if len(op.args) > len(names) or set(kw) - set(names) - {"check_link"}:
        return None
    given = dict(zip(names, op.args)) | kw
    link = DEFAULT_LINK[family]
    if "link" in given and not (isinstance(given["link"], Const) and given["link"].value is None):
        name = _link_of(given["link"])
        if name is None:
            return None
        link = name
    alpha = given.get("alpha") if family == "negative_binomial" else None
    if family == "negative_binomial" and (alpha is None or isinstance(alpha, Const) and alpha.value is None):
        alpha = _c(1.0)                                 # statsmodels' default (with a warning)
    return family, link, alpha


def _link_of(op: Expr | None) -> str | None:
    if not isinstance(op, Op) or op.type != LINK or op.args or op.kwargs:
        return None
    return LINKS.get(op.qualname.rpartition(".")[2])


def inverse_link(link: str, eta: Expr) -> Expr:
    """g⁻¹(η): the mean a linear predictor gives."""
    if link == "identity":
        return eta
    if link == "log":
        return _f("exp", eta)
    if link == "logit":
        return _f("sigmoid", eta)
    return _f("div", _c(1.0), eta)                      # inverse_power


def _clip(mu: Expr, upper: bool) -> Expr:
    """statsmodels' clean: the mean kept at least ε (and at most 1 - ε, for a probability)."""
    low = _f("max2", mu, _c(FLOAT_EPS))
    return _f("min2", low, _c(1.0 - FLOAT_EPS)) if upper else low


@SM.function(*(f"{m}.{n}" for m in FAMILY_MODULES for n in FAMILIES))
def _family(c: Call) -> Expr | None:
    return Op(c.name.rpartition(".")[2], tuple(c.args), tuple(c.kwargs.items()), qualname=c.name, known=True,
              type=FAMILY)


@SM.function(*(f"{m}.{n}" for m in LINK_MODULES for n in LINKS))
def _link(c: Call) -> Expr | None:
    if c.args or c.kwargs:
        return None
    return Op(c.name.rpartition(".")[2], (), qualname=c.name, known=True, type=LINK)


@SM.attribute((FAMILY,), "link")
def _family_link(c: Call) -> Expr | None:
    """family.link: the link it was given, or its family's default."""
    made = c.lower.made_by(c.args[0])
    info = _family_of(made)
    if info is None or not isinstance(made, Op):
        return None
    given = dict(made.kwargs).get("link") or (made.args[0] if made.args else None)
    if isinstance(given, Op) and given.type == LINK:
        return given
    module = made.qualname.rpartition(".")[0].removesuffix(".family")
    name = LINK_CLASS[info[1]]
    return Op(name, (), qualname=f"{module}.links.{name}", known=True, type=LINK)


@SM.method((FAMILY,), "variance")
def _variance(c: Call) -> Expr | None:
    """V(μ), as statsmodels computes it: |μ| (Poisson), |μ|² (Gamma), p(1 - p) (binomial) and p + αp² (negative
    binomial) with p the mean clipped. A constant variance (Gaussian) is an array of ones: an operator."""
    info = _family_of(c.lower.made_by(c.args[0]))
    if info is None or len(c.args) != 2 or c.kwargs:
        return None
    family, _, alpha = info
    mu = c.args[1]
    if family == "poisson":
        return _f("abs", mu)
    if family == "gamma":
        return _f("pow", _f("abs", mu), _c(2.0))
    if family == "binomial":
        p = _clip(mu, True)
        return _f("mul", p, _f("sub", _c(1.0), p))
    if family == "negative_binomial" and alpha is not None:
        p = _clip(mu, False)
        return _f("add", p, _f("mul", alpha, _f("pow", p, _c(2.0))))
    return None


def _link_name(c: Call) -> str | None:
    return _link_of(c.lower.made_by(c.args[0]))


@SM.method((LINK,), "inverse")
def _inverse(c: Call) -> Expr | None:
    link = _link_name(c)
    if link is None or len(c.args) != 2 or c.kwargs:
        return None
    return inverse_link(link, c.args[1])


@SM.method((FAMILY,), "link")
def _family_link_call(c: Call) -> Expr | None:
    """family.link(μ): its link, applied."""
    link = _family_link(Call("link", c.args[:1], {}, c.lower, receiver=True))
    if link is None:
        return None
    return _link_call(Call("__call__", [link, *c.args[1:]], c.kwargs, c.lower, receiver=True))


@SM.method((LINK,), "__call__")
def _link_call(c: Call) -> Expr | None:
    """g(μ), the mean clipped as statsmodels clips it (log: at least ε; logit: within [ε, 1 - ε])."""
    link = _link_name(c)
    if link is None or len(c.args) != 2 or c.kwargs:
        return None
    mu = c.args[1]
    if link == "identity":
        return mu
    if link == "log":
        return _f("log", _clip(mu, False))
    if link == "logit":
        return _f("logit", _clip(mu, True))
    return _f("pow", mu, _c(-1.0))


@SM.method((LINK,), "inverse_deriv")
def _inverse_deriv(c: Call) -> Expr | None:
    """dμ/dη. The identity's (an array of ones): an operator."""
    link = _link_name(c)
    if link is None or len(c.args) != 2 or c.kwargs:
        return None
    eta = c.args[1]
    if link == "log":
        return _f("exp", eta)
    if link == "logit":
        t = _f("exp", eta)
        return _f("div", t, _f("pow", _f("add", _c(1.0), t), _c(2.0)))
    if link == "inverse_power":                         # -η⁻² (statsmodels': η⁻² / -1, the same number)
        return _f("neg", _f("pow", eta, _c(-2.0)))
    return None


@SM.method((LINK,), "deriv")
def _deriv(c: Call) -> Expr | None:
    """dη/dμ, the mean clipped as statsmodels clips it. The identity's (an array of ones): an operator."""
    link = _link_name(c)
    if link is None or len(c.args) != 2 or c.kwargs:
        return None
    mu = c.args[1]
    if link == "log":
        return _f("div", _c(1.0), _clip(mu, False))
    if link == "logit":
        p = _clip(mu, True)
        return _f("div", _c(1.0), _f("mul", p, _f("sub", _c(1.0), p)))
    if link == "inverse_power":                         # -μ⁻² (statsmodels': -1 · μ⁻², the same number)
        return _f("neg", _f("pow", mu, _c(-2.0)))
    return None


# ---- models and their fits ---------------------------------------------------------------------------------------

@SM.function(*MODELS)
def _model(c: Call) -> Expr | None:
    return Op(c.name.rpartition(".")[2], tuple(c.args), tuple(c.kwargs.items()), qualname=c.name, known=True,
              type=MODEL)


def _model_of(op: Expr | None) -> dict[str, object] | None:
    """What a model the code makes says: its kind, response, design, family, link, weights and offset; None if it
    says something not translated."""
    if not isinstance(op, Op) or op.type != MODEL:
        return None
    kind = MODELS.get(op.qualname)
    if kind is None:
        return None
    names = {"glm": ("endog", "exog", "family", "offset", "exposure", "freq_weights", "var_weights"),
             "ols": ("endog", "exog"), "wls": ("endog", "exog", "weights"),
             "logit": ("endog", "exog"), "poisson": ("endog", "exog", "offset", "exposure")}[kind]
    kw = dict(op.kwargs)
    if _literal(kw.get("missing")) == "none":           # statsmodels' default: no rows dropped
        del kw["missing"]
    if len(op.args) > len(names) or set(kw) - set(names):
        return None
    given = {k: v for k, v in (dict(zip(names, op.args)) | kw).items() if _literal(v) is not None}
    if "endog" not in given or "exog" not in given:
        return None
    info: tuple[str, str, Expr | None] | None = {
        "ols": ("gaussian", "identity", None), "wls": ("gaussian", "identity", None),
        "logit": ("binomial", "logit", None), "poisson": ("poisson", "log", None)}.get(kind)
    if kind == "glm":
        fam = given.get("family")
        info = _family_of(fam) if fam is not None else ("gaussian", "identity", None)
    if info is None:
        return None
    family, link, alpha = info
    weights: Expr | None = None
    for k in ("weights", "freq_weights", "var_weights"):
        if k in given:
            weights = given[k] if weights is None else _f("mul", weights, given[k])
    offset = given.get("offset")
    if "exposure" in given:
        log_exposure = _f("log", given["exposure"])
        offset = log_exposure if offset is None else _f("add", offset, log_exposure)
    return {"kind": kind, "y": given["endog"], "X": given["exog"], "family": family, "link": link, "alpha": alpha,
            "weights": weights, "offset": offset}


def _rows_and_columns(y: Expr, X: Expr) -> tuple[Axis, Axis] | None:
    """The observations (y's axis, X's first) and the coefficients (X's second): their index sets."""
    if y.sig is None or X.sig is None or len(y.sig) != 1 or len(X.sig) != 2:
        return None
    rows = y.sig[0]
    if not isinstance(rows, Axis) or not rows.known or X.sig[0] != rows or not isinstance(X.sig[1], Axis):
        return None
    return rows, X.sig[1]


def _reciprocal(x: Expr) -> Expr:
    """1/x, written as what x is the reciprocal of if it is one (1/(1/θ): θ)."""
    if isinstance(x, Apply) and x.fn == "div" and isinstance(x.args[0], Const) and x.args[0].value == 1:
        return x.args[1]
    return _f("div", _c(1.0), x)


def _linear(X: Expr, beta: Expr, offset: Expr | None) -> Expr:
    eta = matmul(X, beta, ND)
    return eta if offset is None else _f("add", eta, offset)


def _unit(family: str, y: Expr, mu: Expr, alpha: Expr | None) -> tuple[str, Expr] | None:
    """What one observation adds to what is optimized: its log likelihood (op "max"), or its squared residual
    (Gaussian: "min"); terms that do not depend on the coefficients left out only where no density is written."""
    if family == "gaussian":
        return "min", _f("pow", _f("sub", y, mu), _c(2.0))
    sig = broadcast(y.sig, mu.sig)
    if family == "poisson":
        dist = Dist("poisson", (("mu", mu),), sig=mu.sig, type="dist")
    elif family == "binomial":
        dist = Dist("bernoulli", (("p", mu),), sig=mu.sig, type="dist")
    elif family == "negative_binomial" and alpha is not None:
        n = _reciprocal(alpha)
        p = _f("div", _c(1.0), _f("add", _c(1.0), _f("mul", alpha, mu)))
        dist = Dist("negative_binomial", (("n", n), ("p", p)), sig=broadcast(n.sig, p.sig), type="dist")
    elif family == "gamma":                             # the log likelihood's part in μ (the shape is not fitted)
        log_mu = mu.args[0] if isinstance(mu, Apply) and mu.fn == "exp" else _f("log", mu)
        return "max", _f("neg", _f("add", _f("div", y, mu), log_mu))
    else:
        return None
    return "max", Density(dist, y, True, sig=sig, type=ND)


@SM.method((MODEL,), "fit")
def _fit(c: Call) -> Expr | None:
    """model.fit(): the coefficients where the likelihood is greatest (the squared residuals least):
    argmax_β Σ_i w_i log p(y_i | g⁻¹(Σ_k X_ik β_k + o_i))."""
    made = c.lower.made_by(c.args[0])
    m = _model_of(made)
    if m is None or len(c.args) != 1 or set(c.kwargs) - FIT_KWARGS[str(m["kind"])]:
        return None
    method = c.kwargs.get("method")
    if method is not None and m["kind"] in ("ols", "wls") and c.literal(method) not in ("pinv", "qr"):
        return None
    y, X = m["y"], m["X"]
    assert isinstance(y, Expr) and isinstance(X, Expr)
    axes = _rows_and_columns(y, X)
    if axes is None:
        return None
    rows, cols = axes
    family, link = str(m["family"]), str(m["link"])
    alpha = m["alpha"] if isinstance(m["alpha"], Expr) else None
    weights = m["weights"] if isinstance(m["weights"], Expr) else None
    offset = m["offset"] if isinstance(m["offset"], Expr) else None
    beta = Var("beta", sig=(cols,), type=ND)
    mu = inverse_link(link, _linear(X, beta, offset))
    unit = _unit(family, y, mu, alpha)
    if unit is None:
        return None
    op, term = unit
    if weights is not None:
        term = _f("mul", weights, term)
    objective = Lambda((Binder("beta", X),), Aggregate("sum", (rows,), term, sig=(), type="float"))
    return Fit(op, family, link, y, X, objective, weights, offset, alpha, made, tuple(c.kwargs.items()),
               str(m["kind"]), type=FIT)


def _fit_of(c: Call) -> tuple[Fit, Expr] | None:
    """The fit a result is (as made), and the coefficients as read from it (res.params)."""
    made = c.lower.made_by(c.args[0])
    if not isinstance(made, Fit) or made.X.sig is None:
        return None
    return made, Field(c.args[0], "params", sig=(made.X.sig[1],), type=ND)


@SM.attribute((FIT,), "params")
def _params(c: Call) -> Expr | None:
    got = _fit_of(c)
    return got[1] if got is not None else None


@SM.attribute((FIT,), "fittedvalues", "mu", "resid", "resid_response")
def _fitted(c: Call) -> Expr | None:
    """The fitted means g⁻¹(Xβ̂ + o) (a discrete model's fittedvalues: the linear predictor Xβ̂ + o, as
    statsmodels has them), and the residuals y - μ̂ (resid: OLS's and WLS's only)."""
    got = _fit_of(c)
    if got is None:
        return None
    fit, params = got
    kind, attr = fit.kind, c.name
    eta = _linear(fit.X, params, fit.offset)
    if attr == "mu" and kind != "glm" or attr == "resid" and kind not in ("ols", "wls"):
        return None
    if attr == "fittedvalues" and kind in ("logit", "poisson"):
        return eta
    mu = inverse_link(fit.link, eta)
    return mu if attr in ("fittedvalues", "mu") else _f("sub", fit.y, mu)


@SM.method((FIT,), "predict")
def _predict(c: Call) -> Expr | None:
    """res.predict(): the fitted means; res.predict(X'): g⁻¹(X'β̂), with no offset (as statsmodels predicts for
    new rows not given one)."""
    got = _fit_of(c)
    if got is None or c.kwargs or len(c.args) > 2:
        return None
    fit, params = got
    if len(c.args) == 1:
        return inverse_link(fit.link, _linear(fit.X, params, fit.offset))
    X = c.args[1]
    if X.sig is None or len(X.sig) != 2 or X.sig[1] != fit.X.sig[1]:     # type: ignore[index]
        return None
    if fit.kind in ("logit", "poisson") and fit.offset is not None:
        return None
    return inverse_link(fit.link, _linear(X, params, None))


# ---- what else statsmodels is used for -----------------------------------------------------------------------------

@SM.function("statsmodels.api.add_constant", "statsmodels.tools.add_constant", "statsmodels.tools.tools.add_constant")
def _add_constant(c: Call) -> Expr | None:
    """A column of ones put before X's (unless X has one already: its value is the library's). Its rows are X's;
    its columns a set of their own."""
    if len(c.args) != 1 or c.kwargs:
        return None
    x = c.args[0]
    if x.sig is None or len(x.sig) not in (1, 2):
        return None
    return Op("add_constant", (x,), qualname=c.name, known=True, sig=(x.sig[0], Axis(None)), type=ND)


BH = {"fdr_bh", "fdr_i", "fdr_p", "fdri", "fdrp"}
BONFERRONI = {"b", "bonf", "bonferroni"}


def _across(fn: str, a: Expr, b: Expr) -> Expr:
    """A comparison of values over two index sets (an index set and its copy): over both, each value aligned by
    its index set, not by position."""
    assert a.sig is not None and b.sig is not None
    return Apply(fn, (a, b), sig=tuple(dict.fromkeys([*a.sig, *b.sig])), type=ND, dtype="bool")


def _corrected(c: Call, p: Expr, alpha: Expr, method: str) -> tuple[Expr, Expr] | None:
    """(reject, the corrected p-values), as statsmodels computes them. Benjamini-Hochberg:
    q_i = min(1, min_{j : p_j ≥ p_i} m p_j / r_j), r_j = |{k : p_k ≤ p_j}| (ties all take the rank of the last of
    them, as the step-up does), rejected where some p_j ≥ p_i has p_j ≤ (r_j / m) α. Bonferroni: min(1, m p_i),
    rejected where p_i ≤ α / m."""
    if p.sig is None or len(p.sig) != 1 or not isinstance(p.sig[0], Axis) or not p.sig[0].known:
        return None
    a = p.sig[0]
    m = Extent(a, p, sig=(), type="int")
    one = _c(1.0)
    if method in BONFERRONI:
        return _f("le", p, _f("div", alpha, m)), _f("min2", _f("mul", p, m), one)
    j = copy_of(a, {a})
    k = copy_of(a, {a, j})
    pj = Align(p, "rename", sig=(j,), type=p.type)
    pk = Align(p, "rename", sig=(k,), type=p.type)
    rank = Aggregate("count", (k,), pk, _across("le", pk, pj), sig=(j,), type=ND, dtype="int")
    later = _across("ge", pj, p)                        # p_j ≥ p_i: over both index sets
    adjusted = Aggregate("min", (j,), _f("div", _f("mul", m, pj), rank), later, sig=(a,), type=ND)
    reject = Aggregate("any", (j,), _f("le", pj, _f("mul", _f("div", rank, m), alpha)), later,
                       sig=(a,), type=ND, dtype="bool")
    return reject, _f("min2", adjusted, one)


@SM.function("statsmodels.stats.multitest.multipletests", "statsmodels.stats.api.multipletests",
             "statsmodels.sandbox.stats.multicomp.multipletests")
def _multipletests(c: Call) -> Expr | None:
    """(reject, corrected p-values, Šidák's α, Bonferroni's α) for method fdr_bh or bonferroni; the default
    (Holm-Šidák) and the rest: operators."""
    if not c.args or len(c.args) > 3 or not c.only("alpha", "method", "is_sorted", "returnsorted"):
        return None
    method = c.literal(c.arg(2, "method"), "hs")
    if method not in BH | BONFERRONI:
        return None
    for flag in ("is_sorted", "returnsorted"):
        if c.literal(c.kwargs.get(flag), False) is not False:
            return None
    p = c.args[0]
    alpha = c.arg(1, "alpha") or _c(0.05)
    got = _corrected(c, p, alpha, method)
    if got is None:
        return None
    reject, q = got
    m = Extent(p.sig[0], p, sig=(), type="int")           # type: ignore[index]
    sidak = _f("sub", _c(1.0), _f("pow", _f("sub", _c(1.0), alpha), _f("div", _c(1.0), m)))
    return Tup((reject, q, sidak, _f("div", alpha, m)), "tuple", type="tuple",
               parts=(reject.shape, q.shape, sidak.shape, _f("div", alpha, m).shape))


@SM.function("statsmodels.stats.multitest.fdrcorrection", "statsmodels.stats.api.fdrcorrection")
def _fdrcorrection(c: Call) -> Expr | None:
    """(reject, corrected p-values) by Benjamini-Hochberg (method "indep", the default)."""
    if not c.args or len(c.args) > 3 or not c.only("alpha", "method", "is_sorted"):
        return None
    if c.literal(c.arg(2, "method"), "indep") not in ("indep", "i", "p", "poscorr") or \
            c.literal(c.kwargs.get("is_sorted"), False) is not False:
        return None
    got = _corrected(c, c.args[0], c.arg(1, "alpha") or _c(0.05), "fdr_bh")
    if got is None:
        return None
    return Tup(got, "tuple", type="tuple", parts=(got[0].shape, got[1].shape))


MAD_C = 0.6744897501960817                              # Φ⁻¹(3/4): the MAD of a standard normal


@SM.function("statsmodels.robust.mad", "statsmodels.robust.scale.mad", "statsmodels.api.robust.scale.mad")
def _mad(c: Call) -> Expr | None:
    """The median absolute deviation of a vector, scaled to a normal's standard deviation:
    median_i |x_i - median(x)| / Φ⁻¹(3/4)."""
    if len(c.args) != 1 or c.kwargs:
        return None
    x = c.args[0]
    if x.sig is None or len(x.sig) != 1 or not isinstance(x.sig[0], Axis) or not x.sig[0].known:
        return None
    a = x.sig[0]
    center = Aggregate("median", (a,), x, sig=(), type="float")
    return Aggregate("median", (a,), _f("div", _f("abs", _f("sub", x, center)), _c(MAD_C)), sig=(), type="float")


__all__ = ["SM", "inverse_link"]
