"""scipy's table: scipy.special's functions, and scipy.stats's distributions (their densities and samples)."""
from __future__ import annotations

from headway.core.libs import Call, Table, register
from headway.core.libs.arrays import elementwise, reduce
from headway.core.terms import Const, Density, Dist, Draw, Expr, broadcast

SCIPY = register(Table("scipy"))

SPECIAL = {"expit": "sigmoid", "logit": "logit", "erf": "erf", "gammaln": "gammaln", "digamma": "digamma",
           "psi": "digamma", "gamma": "gamma", "log1p": "log1p", "expm1": "expm1"}


@SCIPY.function("scipy.special.polygamma")
def _polygamma(c: Call) -> Expr | None:
    """polygamma(0, x): ψ(x); polygamma(1, x): ψ₁(x) (the trigamma function)."""
    if len(c.args) != 2 or c.kwargs or c.literal(c.args[0]) not in (0, 1):
        return None
    return elementwise("digamma" if c.literal(c.args[0]) == 0 else "trigamma", [c.args[1]],
                       "numpy.ndarray" if c.args[1].sig else "float")


@SCIPY.function(*(f"scipy.special.{n}" for n in SPECIAL))
def _special(c: Call) -> Expr | None:
    if len(c.args) != 1 or c.kwargs:
        return None
    return elementwise(SPECIAL[c.name.rpartition(".")[2]], c.args,
                       "numpy.ndarray" if c.args[0].sig else "float")


@SCIPY.function("scipy.special.logsumexp")
def _logsumexp(c: Call) -> Expr | None:
    if not c.args or not c.only("axis", "keepdims") or len(c.args) > 2:
        return None
    return reduce(c, "logsumexp", c.args[0], c.arg(1, "axis"), c.kwargs.get("keepdims"))


# scipy.stats: a distribution's family and its parameters, by scipy's names (loc and scale where scipy has them)
STATS = {"norm": ("normal", ("loc", "scale"), {"loc": 0.0, "scale": 1.0}),
         "binom": ("binomial", ("n", "p"), {}), "bernoulli": ("bernoulli", ("p",), {}),
         "poisson": ("poisson", ("mu",), {}), "nbinom": ("negative_binomial", ("n", "p"), {}),
         "beta": ("beta", ("a", "b"), {}), "betabinom": ("betabinomial", ("n", "a", "b"), {}),
         "gamma": ("gamma", ("a", "loc", "scale"), {"loc": 0.0, "scale": 1.0}),
         "expon": ("exponential", ("loc", "scale"), {"loc": 0.0, "scale": 1.0}),
         "t": ("student_t", ("df", "loc", "scale"), {"loc": 0.0, "scale": 1.0}),
         "laplace": ("laplace", ("loc", "scale"), {"loc": 0.0, "scale": 1.0}),
         "cauchy": ("cauchy", ("loc", "scale"), {"loc": 0.0, "scale": 1.0}),
         "uniform": ("uniform", ("loc", "scale"), {"loc": 0.0, "scale": 1.0}),
         "geom": ("geometric", ("p",), {})}
METHODS = {"logpdf": True, "logpmf": True, "pdf": False, "pmf": False}


def _dist(c: Call, name: str, args: list[Expr]) -> tuple[Dist, list[Expr]] | None:
    family, names, defaults = STATS[name]
    params: dict[str, Expr] = dict(zip(names, args))
    rest = args[len(names):]
    for k, v in c.kwargs.items():
        if k in names:
            params[k] = v
        elif k not in ("size", "random_state"):
            return None
    for k, d in defaults.items():
        params.setdefault(k, Const(d, sig=(), type="float"))
    if set(params) != set(names):
        return None
    sig = broadcast(*(p.sig for p in params.values()))
    return Dist(family, tuple((k, params[k]) for k in names), sig=sig, type="dist"), rest


def _density(c: Call) -> Expr | None:
    """norm.logpdf(x, loc, scale), binom.logpmf(k, n, p): the (log) density at x."""
    parts = c.name.split(".")
    dist_name, method = parts[-2], parts[-1]
    if not c.args:
        return None
    x, *params = c.args
    got = _dist(c, dist_name, params)
    if got is None or got[1] or "size" in c.kwargs:
        return None
    dist = got[0]
    sig = broadcast(x.sig, dist.sig)
    return Density(dist, x, METHODS[method], sig=sig, type="numpy.ndarray" if sig else "float")


def _rvs(c: Call) -> Expr | None:
    parts = c.name.split(".")
    got = _dist(c, parts[-2], list(c.args))
    if got is None or got[1] or "size" in c.kwargs:
        return None
    dist = got[0]
    rng = c.kwargs.get("random_state")
    return Draw(dist, rng, sig=dist.sig, type="numpy.ndarray" if dist.sig else "float")


for _d in STATS:
    for _m in METHODS:
        SCIPY.function(f"scipy.stats.{_d}.{_m}")(_density)
    SCIPY.function(f"scipy.stats.{_d}.rvs")(_rvs)
