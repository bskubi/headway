"""Python's builtins and math module, as terms."""
from __future__ import annotations

from dataclasses import replace

from headway.core.libs import Call, Table, register
from headway.core.libs.arrays import elementwise, extent
from headway.core.terms import Aggregate, Apply, Build, Const, Expr, Op, Range, Shape, Tup

PYTHON = register(Table("python"))

MATH = {"log": "log", "exp": "exp", "sqrt": "sqrt", "fabs": "abs", "log1p": "log1p", "expm1": "expm1",
        "log2": "log2", "log10": "log10", "tanh": "tanh", "sin": "sin", "cos": "cos", "tan": "tan",
        "floor": "floor", "ceil": "ceil", "erf": "erf", "lgamma": "gammaln", "gamma": "gamma", "isnan": "isnan",
        "isfinite": "isfinite", "atan": "arctan", "asin": "arcsin", "acos": "arccos", "sinh": "sinh",
        "cosh": "cosh"}


@PYTHON.function(*(f"math.{n}" for n in MATH))
def _math(c: Call) -> Expr | None:
    name = c.name.rpartition(".")[2]
    if name == "log" and len(c.args) == 2:
        return Apply("div", (Apply("log", (c.args[0],), sig=(), type="float"),
                             Apply("log", (c.args[1],), sig=(), type="float")), sig=(), type="float")
    if len(c.args) != 1 or c.kwargs:
        return None
    return Apply(MATH[name], (c.args[0],), sig=(), type="int" if name in ("floor", "ceil") else "float")


@PYTHON.function("typing.cast")
def _cast(c: Call) -> Expr | None:
    """cast(T, x): x (a type checker's note, not a computation)."""
    return c.args[1] if len(c.args) == 2 and not c.kwargs else None


@PYTHON.function("builtins.sum")
def _sum(c: Call) -> Expr | None:
    """sum(f(x) for x in xs): a sum over x; sum(xs, start): start plus it."""
    if not c.args or c.kwargs or len(c.args) > 2:
        return None
    x = c.args[0]
    if isinstance(x, Build):
        out: Expr = Aggregate("sum", x.over, x.body, x.where, sig=x.body.sig, type=x.body.type or "float")
    elif x.type == "list" and x.elem is not None:
        return None
    elif x.sig is not None and len(x.sig) >= 1:          # sum over an array's first axis, as Python iterates it
        from headway.core.libs.arrays import reduce
        out = reduce(c, "sum", x, Const(0, sig=(), type="int"))  # type: ignore[assignment]
        if out is None:
            return None
    else:
        return None
    if len(c.args) == 2 and not (isinstance(c.args[1], Const) and c.args[1].value == 0):
        out = elementwise("add", [c.args[1], out])
    return out


@PYTHON.function("builtins.min", "builtins.max")
def _minmax(c: Call) -> Expr | None:
    name = c.name.rpartition(".")[2]
    if c.kwargs:
        return None
    if len(c.args) >= 2:                                # max(a, b, ...): of these values (each a number)
        if any(a.sig not in ((),) for a in c.args):
            return None                                 # an array: Python compares it whole, not elementwise
        out = c.args[0]
        for a in c.args[1:]:
            out = Apply(name + "2", (out, a), sig=(), type="float" if "float" in (out.type, a.type) else out.type)
        return out
    x = c.args[0] if c.args else None
    if isinstance(x, Build):
        return Aggregate(name, x.over, x.body, x.where, sig=x.body.sig, type=x.body.type)
    if isinstance(x, Tup) and len(x.items) >= 2 and all(i.sig == () for i in x.items):     # min(x.shape)
        out = x.items[0]
        for a in x.items[1:]:
            out = Apply(name + "2", (out, a), sig=(), type="int" if out.type == a.type == "int" else "float")
        return out
    return None


@PYTHON.function("builtins.abs")
def _abs(c: Call) -> Expr | None:
    return elementwise("abs", c.args) if len(c.args) == 1 and not c.kwargs else None


@PYTHON.function("builtins.round")
def _round(c: Call) -> Expr | None:
    return Apply("round", (c.args[0],), sig=c.args[0].sig, type="int") if len(c.args) == 1 and not c.kwargs \
        else None


@PYTHON.function("builtins.pow")
def _pow(c: Call) -> Expr | None:
    return elementwise("pow", c.args) if len(c.args) == 2 and not c.kwargs else None


@PYTHON.function("builtins.float", "builtins.int", "builtins.bool")
def _number(c: Call) -> Expr | None:
    """float(x): the number x is (written as x); int(x): its integer part."""
    if len(c.args) != 1 or c.kwargs:
        return None
    x = c.args[0]
    name = c.name.rpartition(".")[2]
    if name == "float":
        return replace(x, sig=(), type="float") if x.sig in ((), None) else None
    if name == "int":
        return Apply("int", (x,), sig=(), type="int")
    return None


@PYTHON.function("builtins.len")
def _len(c: Call) -> Expr | None:
    if len(c.args) != 1 or c.kwargs:
        return None
    x = c.args[0]
    if x.sig:
        return extent(x, 0)
    return Apply("len", (x,), sig=(), type="int")


@PYTHON.function("builtins.range")
def _range(c: Call) -> Expr | None:
    if c.kwargs or not 1 <= len(c.args) <= 2:
        return None
    lo, hi = (Const(0, sig=(), type="int"), c.args[0]) if len(c.args) == 1 else (c.args[0], c.args[1])
    return Range(lo, hi, type="range", elem=Shape("int", ()))


@PYTHON.function("builtins.enumerate", "builtins.zip")
def _iterate(c: Call) -> Expr | None:
    if c.kwargs:
        return None
    return Op(c.name.rpartition(".")[2], tuple(c.args), qualname=c.name, type="iterator")


@PYTHON.function("builtins.list", "builtins.tuple")
def _list(c: Call) -> Expr | None:
    if len(c.args) != 1 or c.kwargs:
        return None
    x = c.args[0]
    if isinstance(x, Build):
        return x
    return x if x.type in ("list", "tuple") else None


@PYTHON.attribute(("?", "pandas.Series", "pandas.DataFrame"), "values")
@PYTHON.method(("?", "pandas.Series", "pandas.DataFrame", "polars.Series"), "to_numpy")
def _values(c: Call) -> Expr | None:
    """x.values, x.to_numpy() (a pandas or polars column or table as an array): the same values. (dict.values()
    is a call, so an attribute .values is a data frame's.)"""
    return c.args[0] if len(c.args) == 1 and not c.kwargs else None
