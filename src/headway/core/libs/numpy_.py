"""numpy's table: its functions and its arrays' methods, as terms (docs/core.md §6).

An entry translates only the arguments it knows: a call with any other (ddof=1, where=, out= inside an expression)
is not translated, so it stays an operator, as written. NaN-aware variants (nansum, fmax) are operators too: what they
do with NaN is not what the calculus's sum and max say.
"""
from __future__ import annotations

from dataclasses import replace

from headway.core.libs import NOT, Call, Table, register
from headway.core.libs.arrays import (elementwise, extent, getitem, matmul, outer, reduce, shape_sig, transpose,
                                      with_type)
from headway.core.terms import (NEW, Aggregate, Align, Apply, Axis, Cases, Const, Dist, Draw, Expr, Extent, Idx,
                                Op, Pos, Sig, Take, Tup, broadcast)

NUMPY = register(Table("numpy"))
ND = "numpy.ndarray"
ARRAYS = (ND, "?")                       # "?": a receiver whose type is not known, read as numpy's (duck)

NUMPY.constants.update({
    "numpy.pi": Const(3.141592653589793, sig=(), type="float"), "math.pi": Const(3.141592653589793, sig=(), type="float"),
    "numpy.e": Const(2.718281828459045, sig=(), type="float"), "math.e": Const(2.718281828459045, sig=(), type="float"),
    "numpy.inf": Const(float("inf"), sig=(), type="float"), "math.inf": Const(float("inf"), sig=(), type="float"),
    "numpy.nan": Const(float("nan"), sig=(), type="float"), "math.nan": Const(float("nan"), sig=(), type="float"),
    "numpy.newaxis": Const(None, type="none"),
})
NUMPY.types.update({"numpy.ndarray": ND, "numpy.typing.NDArray": ND, "numpy.random.Generator": "numpy.random.Generator"})

UNARY = {"log": "log", "exp": "exp", "sqrt": "sqrt", "abs": "abs", "absolute": "abs", "fabs": "abs",
         "log1p": "log1p", "expm1": "expm1", "log2": "log2", "log10": "log10", "tanh": "tanh", "sinh": "sinh",
         "cosh": "cosh", "sin": "sin", "cos": "cos", "tan": "tan", "arcsin": "arcsin", "arccos": "arccos",
         "arctan": "arctan", "floor": "floor", "ceil": "ceil", "sign": "sign", "isnan": "isnan",
         "isfinite": "isfinite", "square": "square", "negative": "neg", "logical_not": "not", "rint": "round",
         "isinf": "isinf"}
BINARY = {"add": "add", "subtract": "sub", "multiply": "mul", "divide": "div", "true_divide": "div",
          "power": "pow", "float_power": "pow", "maximum": "max2", "minimum": "min2", "mod": "mod",
          "remainder": "mod", "floor_divide": "floordiv", "logical_and": "and", "logical_or": "or",
          "equal": "eq", "not_equal": "ne", "less": "lt", "less_equal": "le", "greater": "gt",
          "greater_equal": "ge"}
BOOLEAN = {"isnan", "isfinite", "isinf", "not", "and", "or", "eq", "ne", "lt", "le", "gt", "ge"}


@NUMPY.function("numpy.invert", "numpy.bitwise_not")
def _invert(c: Call) -> Expr | None:
    """~x: of a mask, not (bitwise for integers: an operator)."""
    if len(c.args) != 1 or c.kwargs or c.args[0].dtype != "bool":
        return None
    out = elementwise("not", c.args, _type(c.args))
    return replace(out, dtype="bool")
REDUCE = {"sum": "sum", "prod": "prod", "mean": "mean", "max": "max", "amax": "max", "min": "min", "amin": "min",
          "any": "any", "all": "all", "median": "median", "argmax": "argmax", "argmin": "argmin"}
SAME = {"asarray", "array", "asanyarray", "ascontiguousarray", "copy", "float64", "float32", "float16",
        "int64", "int32", "double", "single"}


@NUMPY.function(*(f"numpy.{n}" for n in UNARY))
def _unary(c: Call) -> Expr | None:
    if len(c.args) != 1 or not c.only("dtype"):
        return None
    fn = UNARY[c.name.rpartition(".")[2]]
    out = elementwise(fn, c.args, _type(c.args))
    return replace(out, dtype="bool") if fn in BOOLEAN else out


@NUMPY.function(*(f"numpy.{n}" for n in BINARY))
def _binary(c: Call) -> Expr | None:
    if len(c.args) != 2 or not c.only("dtype"):
        return None
    fn = BINARY[c.name.rpartition(".")[2]]
    out = elementwise(fn, c.args, _type(c.args))
    return replace(out, dtype="bool") if fn in BOOLEAN else out


def _type(args: list[Expr]) -> str:
    return ND if any(a.sig for a in args) or any(a.type == ND for a in args) else "float"


@NUMPY.function("numpy.where")
def _where(c: Call) -> Expr | None:
    if len(c.args) != 3 or c.kwargs:
        return None
    cond, a, b = c.args
    return Cases(((cond, a),), b, sig=broadcast(cond.sig, a.sig, b.sig), type=ND)


@NUMPY.function("numpy.clip")
@NUMPY.method(ARRAYS, "clip")
def _clip(c: Call) -> Expr | None:
    if not c.only("a_min", "a_max", "min", "max"):
        return None
    x = c.args[0]
    lo = c.arg(1, "a_min") or c.kwargs.get("min")
    hi = c.arg(2, "a_max") or c.kwargs.get("max")
    out = x
    if lo is not None and c.literal(lo) is not None:
        out = elementwise("max2", [out, lo], _type([x]))
    if hi is not None and c.literal(hi) is not None:
        out = elementwise("min2", [out, hi], _type([x]))
    return out


def _reduction(c: Call) -> Expr | None:
    name = c.name.rpartition(".")[2]
    op = REDUCE.get(name) or {"std": "sd", "var": "var"}.get(name)
    if op is None or not c.args:
        return None
    x = c.args[0]
    allowed = ("axis", "keepdims", "dtype") + (("ddof",) if op in ("sd", "var") else ())
    if not c.only(*allowed):
        return None
    if op in ("sd", "var") and c.literal(c.kwargs.get("ddof"), 0) != 0:
        return None                                     # ddof=1: an operator, as written
    axis = c.arg(1, "axis")
    keep = c.kwargs.get("keepdims")
    if len(c.args) > 2:
        return None
    if op in ("argmax", "argmin") and axis is None and (x.sig is None or len(x.sig) != 1):
        return None                                     # the position in the flattened array
    from headway.core.terms import Build
    if isinstance(x, Build) and c.literal(axis) == 0 and not c.kwargs.get("keepdims") and op in (
            "sum", "prod", "mean", "max", "min"):          # np.sum([f(c) for c in cs], 0): over c
        return Aggregate(op, x.over, x.body, x.where, sig=x.body.sig, type=x.body.type or ND)
    out = reduce(c, op, x, axis, keep)
    if out is not None and op in ("any", "all"):
        out = replace(out, dtype="bool")
    if out is not None and op in ("argmax", "argmin"):
        out = replace(out, dtype="int")
    return out


for _n in [*REDUCE, "std", "var"]:
    NUMPY.function(f"numpy.{_n}")(_reduction)
    NUMPY.method(ARRAYS, _n)(_reduction)


@NUMPY.function("numpy.matmul", "numpy.dot")
@NUMPY.method(ARRAYS, "dot")
def _matmul(c: Call) -> Expr | None:
    if len(c.args) != 2 or c.kwargs:
        return None
    a, b = c.args
    if c.name.endswith("dot") and (a.sig is None or b.sig is None or len(a.sig) > 2 or len(b.sig) > 2):
        return None                                     # np.dot beyond matrices is not matmul
    if a.sig == () or b.sig == ():
        return elementwise("mul", [a, b])
    return matmul(a, b, ND)


@NUMPY.function("numpy.outer")
def _outer(c: Call) -> Expr | None:
    if len(c.args) != 2 or c.kwargs:
        return None
    return outer(c.args[0], c.args[1], ND)


@NUMPY.function("numpy.transpose")
@NUMPY.method(ARRAYS, "transpose")
def _transpose(c: Call) -> Expr | None:
    if len(c.args) != 1 or c.kwargs:
        return None
    return transpose(c.args[0])


@NUMPY.attribute(ARRAYS, "T")
def _T(c: Call) -> Expr | None:
    return transpose(c.args[0])


@NUMPY.attribute(ARRAYS, "shape")
def _shape(c: Call) -> Expr | None:
    x = c.args[0]
    if x.sig is None:
        return None
    return Tup(tuple(extent(x, k) for k in range(len(x.sig))), type="tuple")


@NUMPY.attribute(ARRAYS, "real", "values", "data")
def _same_attr(c: Call) -> Expr | None:
    return c.args[0] if c.name != "values" or c.args[0].type in (ND,) else None


@NUMPY.function(*(f"numpy.{n}" for n in SAME))
def _same(c: Call) -> Expr | None:
    """Conversions that keep the values and their axes."""
    if len(c.args) < 1 or not c.only("dtype", "copy") or len(c.args) > 2:
        return None
    x = c.args[0]
    return with_type(x, ND) if x.sig is not None or x.type in ("", ND) else x


@NUMPY.method(ARRAYS, "astype", "copy", "view", "to_numpy")
def _astype(c: Call) -> Expr | None:
    if not c.only("dtype", "copy"):
        return None
    return c.args[0]


@NUMPY.method(ARRAYS, "item")
def _item(c: Call) -> Expr | None:
    x = c.args[0]
    return replace(x, type="float") if len(c.args) == 1 and x.sig in ((), None) else None


@NUMPY.method(ARRAYS, "squeeze", "ravel", "flatten", "reshape")
@NUMPY.function("numpy.squeeze", "numpy.ravel", "numpy.reshape")
def _reshape(c: Call) -> Expr | None:
    """The same values, on axes not known (unless squeeze takes out only axes of length one put in to broadcast)."""
    x = c.args[0]
    if c.name.endswith("squeeze") and x.sig is not None and len(c.args) == 1 and not c.kwargs:
        kept = tuple(a for a in x.sig if a != NEW)
        if any(not a.known for a in kept):
            return Align(x, "squeeze", sig=None, type=x.type)
        return Align(x, "squeeze", sig=kept, type=x.type)
    if c.name.endswith(("ravel", "flatten")) and x.sig is not None and len([a for a in x.sig if a != NEW]) == 1:
        return Align(x, "squeeze", sig=tuple(a for a in x.sig if a != NEW), type=x.type)
    return Align(x, "reshape", sig=None, type=x.type)


@NUMPY.function("numpy.zeros", "numpy.ones", "numpy.empty", "numpy.zeros_like", "numpy.ones_like",
                "numpy.empty_like", "numpy.full", "numpy.full_like")
def _made(c: Call) -> Expr | None:
    name = c.name.rpartition(".")[2]
    if not c.only("dtype") or not c.args:
        return None
    like = name.endswith("_like")
    sig: Sig = c.args[0].sig if like else shape_sig(c, c.args[0])
    if name.startswith("full"):
        if len(c.args) < 2:
            return None
        v = c.args[1]
        if v.sig == sig or sig is None or any(not a.known for a in sig):
            return v if v.sig == sig else None
        return Align(v, "broadcast", sig=broadcast(v.sig, sig), type=ND, dtype=v.dtype)  # v everywhere
    if name.startswith("empty") or sig is None or any(not a.known for a in sig):
        return Op(name, tuple(c.args), sig=sig, type=ND, qualname=c.name, known=True)  # its size not a set's
    return Const(1 if name.startswith("ones") else 0, sig=sig, type=ND)


@NUMPY.function("numpy.eye", "numpy.identity")
def _eye(c: Call) -> Expr | None:
    if len(c.args) != 1 or not c.only("dtype"):
        return None
    sig = shape_sig(c, c.args[0])
    if sig is None or len(sig) != 1:
        return Op("I", tuple(c.args), sig=None, type=ND, qualname=c.name, known=True)
    a = sig[0]
    if not a.known:
        return Op("I", tuple(c.args), sig=(a, Axis(None)), type=ND, qualname=c.name, known=True)
    b = Axis(a.name, a.copy + 1, letter=a.letter)
    return Apply("eq", (Idx(a, sig=(a,), type="int"), Idx(b, sig=(b,), type="int")), sig=(a, b), type=ND,
                 dtype="bool")


@NUMPY.function("numpy.diag", "numpy.diagonal")
def _diag(c: Call) -> Expr | None:
    """diag(v): the matrix with v on its diagonal, v_i δ_{ii'}; diag(M) of a square matrix: its diagonal, M_{ii}."""
    if len(c.args) != 1 or c.kwargs:
        return None
    x = c.args[0]
    if x.sig is None or not all(a.known for a in x.sig):
        return None
    if len(x.sig) == 1 and c.name.endswith("diag"):
        a = x.sig[0]
        b = Axis(a.name, a.copy + 1, a.where, a.part, a.letter)
        delta = Apply("eq", (Idx(a, sig=(a,), type="int"), Idx(b, sig=(b,), type="int")), sig=(a, b), dtype="bool")
        return Apply("mul", (Align(x, "expand", sig=(a, NEW), type=ND), delta), sig=(a, b), type=ND)
    if len(x.sig) == 2 and x.sig[0].name == x.sig[1].name:
        a = x.sig[0]
        i = Idx(a, sig=(a,), type="int")
        return Take(x, (0, 1), (i, i), 0, sig=(a,), type=ND)
    return None


@NUMPY.function("numpy.arange")
def _arange(c: Call) -> Expr | None:
    """arange(len(x)): x's index itself, as a value."""
    if len(c.args) != 1 or not c.only("dtype"):
        return None
    n = c.args[0]
    if isinstance(n, Extent) and isinstance(n.axis, Axis):
        return Idx(n.axis, sig=(n.axis,), type=ND, dtype="int")
    return None


@NUMPY.function("numpy.einsum")
def _einsum(c: Call) -> Expr | None:
    """einsum("ij,jk->ik", a, b): the product of the operands, summed over the indices the result does not have."""
    spec = c.literal(c.args[0]) if c.args else NOT
    if not isinstance(spec, str) or "->" not in spec or "." in spec or c.kwargs:
        return None
    ins, out = spec.replace(" ", "").split("->")
    operands = c.args[1:]
    subs = ins.split(",")
    if len(subs) != len(operands) or any(o.sig is None or len(o.sig) != len(s) for o, s in zip(operands, subs)):
        return None
    letter_axis: dict[str, Axis] = {}
    renamed = []
    for o, s in zip(operands, subs):
        assert o.sig is not None
        new = []
        for ch, a in zip(s, o.sig):
            if not a.known:
                return None
            if ch not in letter_axis:
                taken = set(letter_axis.values())
                letter_axis[ch] = a if a not in taken else Axis(a.name, a.copy + 1 + len(taken), letter=a.letter)
            new.append(letter_axis[ch])
        renamed.append(o if tuple(new) == o.sig else Align(o, "rename", sig=tuple(new), type=o.type))
    if len(set(subs[0])) != len(subs[0]) or any(len(set(s)) != len(s) for s in subs):
        return None                                     # a repeated index in one operand: a diagonal
    product = renamed[0]
    for r in renamed[1:]:                               # by index set: the union of the operands' axes
        assert product.sig is not None and r.sig is not None
        product = Apply("mul", (product, r), sig=tuple(dict.fromkeys([*product.sig, *r.sig])), type=ND)
    over = tuple(letter_axis[ch] for ch in dict.fromkeys("".join(subs)) if ch not in out)
    sig = tuple(letter_axis[ch] for ch in out)
    if not over:
        return Align(product, "transpose", sig=sig, type=ND) if product.sig != sig else product
    return Aggregate("sum", over, product, sig=sig, type=ND if sig else "float")


# linear algebra

@NUMPY.function("numpy.linalg.inv", "scipy.linalg.inv")
def _inv(c: Call) -> Expr | None:
    if len(c.args) != 1 or c.kwargs:
        return None
    x = c.args[0]
    sig = tuple(reversed(x.sig)) if x.sig is not None and len(x.sig) == 2 else None
    return Apply("inv", (x,), sig=sig, type=ND)


@NUMPY.function("numpy.linalg.solve", "scipy.linalg.solve")
def _solve(c: Call) -> Expr | None:
    """The solution of a x = b (what kind of matrix a is, assume_a, only chooses how it is solved)."""
    if len(c.args) != 2 or set(c.kwargs) - {"assume_a", "check_finite", "overwrite_a", "overwrite_b"}:
        return None
    a, b = c.args
    sig = (a.sig[1], *b.sig[1:]) if a.sig is not None and b.sig is not None and len(a.sig) == 2 and b.sig else None
    return Apply("solve", (a, b), sig=sig, type=ND)


@NUMPY.function("numpy.linalg.det", "scipy.linalg.det", "numpy.trace")
@NUMPY.method(ARRAYS, "trace")
def _det(c: Call) -> Expr | None:
    if len(c.args) != 1 or c.kwargs:
        return None
    return Apply("trace" if c.name.endswith("trace") else "det", (c.args[0],), sig=(), type="float")


@NUMPY.function("numpy.linalg.norm")
def _norm(c: Call) -> Expr | None:
    if len(c.args) != 1 or c.kwargs:
        return None
    x = c.args[0]
    return Apply("norm", (x,), sig=(), type="float") if x.sig is None or len(x.sig) <= 1 else None


@NUMPY.function("numpy.linalg.eigh", "scipy.linalg.eigh", "numpy.linalg.svd", "numpy.linalg.eig")
def _eigh(c: Call) -> Expr | None:
    """Eigenvalues and vectors (or a singular value decomposition): an operator, its results' axes those of the
    matrix and a new one (the components), whose index set is not known."""
    if len(c.args) != 1 or c.kwargs:
        return None
    x = c.args[0]
    name = c.name.rpartition(".")[2]
    from headway.core.terms import Shape
    if x.sig is None or len(x.sig) != 2:
        return Op(name, (x,), known=True, qualname=c.name, type="tuple")
    comp = Axis(None)
    parts: tuple[Shape, ...]
    if name == "svd":
        parts = (Shape(ND, (x.sig[0], comp)), Shape(ND, (comp,)), Shape(ND, (comp, x.sig[1])))
    else:
        parts = (Shape(ND, (comp,)), Shape(ND, (x.sig[0], comp)))
    return Op(name, (x,), known=True, qualname=c.name, type="tuple", parts=parts)


@NUMPY.function("numpy.linalg.qr", "scipy.linalg.qr")
def _qr(c: Call) -> Expr | None:
    """A QR decomposition (reduced): Q with the matrix's rows and a new axis, R with that axis and the matrix's
    columns, so that R⁻¹ of something is indexed by the columns."""
    from headway.core.terms import Shape
    if len(c.args) != 1 or set(c.kwargs) - {"mode"} or c.literal(c.kwargs.get("mode", Const("reduced"))) not in (
            "reduced", None):
        return None
    x = c.args[0]
    if x.sig is None or len(x.sig) != 2:
        return Op("qr", (x,), known=True, qualname=c.name, type="tuple")
    comp = Axis(None)
    return Op("qr", (x,), known=True, qualname=c.name, type="tuple",
              parts=(Shape(ND, (x.sig[0], comp)), Shape(ND, (comp, x.sig[1]))))


@NUMPY.function("numpy.argsort", "numpy.sort", "numpy.cumsum", "numpy.unique", "numpy.concatenate",
                "numpy.stack", "numpy.vstack", "numpy.hstack", "numpy.diff", "numpy.bincount", "numpy.isin",
                "numpy.nonzero", "numpy.flatnonzero", "numpy.nansum", "numpy.nanmean", "numpy.nanmax",
                "numpy.nanmin", "numpy.fmax", "numpy.fmin", "numpy.round", "numpy.around", "numpy.histogram",
                "numpy.searchsorted", "numpy.repeat", "numpy.tile", "numpy.cov", "numpy.corrcoef",
                "numpy.percentile", "numpy.quantile", "numpy.nan_to_num", "numpy.linalg.cholesky",
                "numpy.linalg.lstsq", "numpy.linalg.pinv", "numpy.linalg.matrix_rank",
                "numpy.linspace", "numpy.logspace", "numpy.meshgrid", "numpy.expand_dims", "numpy.broadcast_to",
                "numpy.argwhere", "numpy.count_nonzero", "numpy.cumprod", "numpy.lexsort", "numpy.split",
                "numpy.array_split", "numpy.pad", "numpy.roll", "numpy.flip", "numpy.isclose", "numpy.allclose",
                "numpy.full_like", "numpy.digitize", "numpy.interp", "numpy.convolve", "numpy.polyfit")
def _named(c: Call) -> Expr | None:
    """Operators: shown by their name (no translation, so nothing to get wrong). Listed so that the linter knows
    they are numpy's and deliberate."""
    return None


# randomness: numpy's Generator (rng = np.random.default_rng(seed)) and the legacy np.random functions

RANDOM = {"normal": ("normal", ("loc", "scale")), "binomial": ("binomial", ("n", "p")),
          "poisson": ("poisson", ("lam",)), "gamma": ("gamma", ("shape", "scale")), "beta": ("beta", ("a", "b")),
          "uniform": ("uniform", ("low", "high")), "exponential": ("exponential", ("scale",)),
          "dirichlet": ("dirichlet", ("alpha",)), "multinomial": ("multinomial", ("n", "pvals")),
          "negative_binomial": ("negative_binomial", ("n", "p")), "lognormal": ("lognormal", ("mean", "sigma")),
          "standard_normal": ("normal", ()), "integers": ("integers", ("low", "high")),
          "choice": ("choice", ("a", "size", "replace", "p")), "permutation": ("permutation", ("x",)),
          "laplace": ("laplace", ("loc", "scale")), "geometric": ("geometric", ("p",)),
          "multivariate_normal": ("multivariate_normal", ("mean", "cov")), "random": ("uniform", ())}
DEFAULTS = {"normal": {"loc": 0.0, "scale": 1.0}, "uniform": {"low": 0.0, "high": 1.0},
            "exponential": {"scale": 1.0}, "gamma": {"scale": 1.0}, "laplace": {"loc": 0.0, "scale": 1.0},
            "lognormal": {"mean": 0.0, "sigma": 1.0}}


@NUMPY.function("numpy.random.default_rng", "numpy.random.RandomState", "numpy.random.Generator")
def _rng(c: Call) -> Expr | None:
    return Op("rng", tuple(c.args), tuple(c.kwargs.items()), known=True, qualname=c.name, type="numpy.random.Generator")


def _draw(c: Call, rng: Expr | None, args: list[Expr]) -> Expr | None:
    name = c.name.rpartition(".")[2]
    family, names = RANDOM[name]
    if family in ("choice", "permutation", "integers", "multinomial", "dirichlet", "multivariate_normal"):
        return None                                     # their axes are not the parameters': operators for now
    params: dict[str, Expr] = {}
    for n, a in zip(names, args):
        params[n] = a
    for k, v in c.kwargs.items():
        if k in names:
            params[k] = v
        elif k != "size":
            return None
    if len(args) > len(names) + 1:
        return None
    size = c.kwargs.get("size") or (args[len(names)] if len(args) > len(names) else None)
    for k, d in DEFAULTS.get(family, {}).items():
        params.setdefault(k, Const(d, sig=(), type="float"))
    if family == "normal" and name == "standard_normal":
        params = {"loc": Const(0.0, sig=(), type="float"), "scale": Const(1.0, sig=(), type="float")}
    if set(params) != set(names) and name not in ("standard_normal", "random"):
        return None
    sig = broadcast(*(p.sig for p in params.values()))
    if size is not None:
        made = shape_sig(c, size)
        if made is None or sig is None:
            sig = None
        else:
            sig = broadcast(made, sig) if len(made) >= len(sig) else None
    dist = Dist(family, tuple(params.items()), sig=sig, type=ND)
    return Draw(dist, rng, size, sig=sig, type=ND if sig or size is not None else "float",
                dtype="int" if family in ("binomial", "poisson", "negative_binomial", "geometric") else "float")


@NUMPY.method(("numpy.random.Generator",), *RANDOM)
def _generator(c: Call) -> Expr | None:
    return _draw(c, c.args[0], c.args[1:])


@NUMPY.function(*(f"numpy.random.{n}" for n in RANDOM))
def _legacy(c: Call) -> Expr | None:
    return _draw(c, None, c.args)


@NUMPY.method(ARRAYS, "__getitem__")
def _getitem(c: Call) -> Expr | None:
    out = getitem(c, c.args[0], c.args[1:])
    return replace(out, dtype=c.args[0].dtype) if out is not None else None


@NUMPY.method(ARRAYS, "sum_", "argsort", "sort", "cumsum", "nonzero", "tolist", "fill", "max_", "round")
def _method_named(c: Call) -> Expr | None:
    return None


__all__ = ["NUMPY", "Pos", "Tup"]
