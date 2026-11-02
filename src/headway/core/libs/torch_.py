"""PyTorch's table (docs/core.md §6: next after numpy, for deep learning). Where torch's meaning is numpy's (sum
over dim=, elementwise functions, @), the same translations; where it differs (std is the sample's by default,
max(dim) gives values and positions), none, so that such calls stay operators until an entry says what they mean.

Deep learning's own operations are written as their formulas, so that the math is the model's:

  torch.softmax(x, dim=1)            e^{x_{nk}} / Σ_{k'} e^{x_{nk'}}
  F.log_softmax(x, dim=1)            x_{nk} - logsumexp_{k'} x_{nk'}
  F.cross_entropy(z, y)              mean_n (logsumexp_k z_{nk} - z_{n,y_n})        (the default: mean, classes on 1)
  F.mse_loss(a, b)                   mean (a - b)^2
  F.linear(x, W, b), nn.Linear       Σ_f x_{nf} W_{hf} + b_h
  torch.autograd.grad(L, θ)          ∇_θ L
"""
from __future__ import annotations

from dataclasses import replace

from headway.core.libs import Call, Table, register
from headway.core.libs.arrays import elementwise, getitem, matmul, reduce, transpose
from headway.core.terms import (NEW, Aggregate, Align, Apply, Axis, Const, Expr, Field, Idx, Op, Shape, Take, Tup,
                                broadcast)

TORCH = register(Table("torch"))
T = "torch.Tensor"
LINEAR = "torch.nn.Linear"
TORCH.types.update({"torch.Tensor": T, "torch.nn.Linear": LINEAR, "torch.nn.Module": "torch.nn.Module"})

UNARY = {"log": "log", "exp": "exp", "sqrt": "sqrt", "abs": "abs", "log1p": "log1p", "expm1": "expm1",
         "tanh": "tanh", "sin": "sin", "cos": "cos", "sigmoid": "sigmoid", "logit": "logit", "floor": "floor",
         "ceil": "ceil", "sign": "sign", "square": "square", "relu": "relu", "neg": "neg", "isnan": "isnan",
         "lgamma": "gammaln", "digamma": "digamma", "erf": "erf", "softplus": "softplus"}
BINARY = {"maximum": "max2", "minimum": "min2", "add": "add", "sub": "sub", "mul": "mul", "div": "div",
          "pow": "pow"}
F = "torch.nn.functional"


@TORCH.function(*(f"torch.{n}" for n in UNARY), f"{F}.relu", f"{F}.softplus", f"{F}.sigmoid", f"{F}.tanh")
@TORCH.method((T,), *UNARY)
def _unary(c: Call) -> Expr | None:
    if len(c.args) != 1 or c.kwargs:
        return None
    name = c.name.rpartition(".")[2]
    return elementwise(UNARY.get(name, name), c.args, T)


@TORCH.function(*(f"torch.{n}" for n in BINARY))
@TORCH.method((T,), *BINARY)
def _binary(c: Call) -> Expr | None:
    if len(c.args) != 2 or c.kwargs:
        return None
    return elementwise(BINARY[c.name.rpartition(".")[2]], c.args, T)


@TORCH.function("torch.sum", "torch.mean", "torch.prod", "torch.logsumexp", "torch.any", "torch.all")
@TORCH.method((T,), "sum", "mean", "prod", "logsumexp", "any", "all")
def _reduce(c: Call) -> Expr | None:
    if not c.args or not c.only("dim", "keepdim") or len(c.args) > 2:
        return None
    name = c.name.rpartition(".")[2]
    out = reduce(c, name, c.args[0], c.arg(1, "dim"), c.kwargs.get("keepdim"))
    return replace(out, dtype="bool") if out is not None and name in ("any", "all") else out


@TORCH.function("torch.matmul")
@TORCH.method((T,), "matmul")
def _matmul(c: Call) -> Expr | None:
    if len(c.args) != 2 or c.kwargs:
        return None
    return matmul(c.args[0], c.args[1], T)


@TORCH.attribute((T,), "T")
@TORCH.method((T,), "t")
def _T(c: Call) -> Expr | None:
    return transpose(c.args[0]) if len(c.args) == 1 else None


@TORCH.method((T,), "__getitem__")
def _getitem(c: Call) -> Expr | None:
    return getitem(c, c.args[0], c.args[1:])


@TORCH.method((T,), "float", "double", "detach", "clone", "contiguous", "cpu")
def _same(c: Call) -> Expr | None:
    return c.args[0] if len(c.args) == 1 and not c.kwargs else None


@TORCH.method((T,), "unsqueeze")
def _unsqueeze(c: Call) -> Expr | None:
    x = c.args[0]
    k = c.literal(c.arg(1, "dim"))
    if x.sig is None or not isinstance(k, int) or not -len(x.sig) - 1 <= k <= len(x.sig):
        return None
    sig = list(x.sig)
    sig.insert(k % (len(x.sig) + 1), NEW)
    return Align(x, "expand", sig=tuple(sig), type=T)


def _dim(c: Call, x: Expr, default: int | None = None) -> int | None:
    d = c.literal(c.arg(1, "dim")) if len(c.args) > 1 or "dim" in c.kwargs else default
    if not isinstance(d, int) or x.sig is None or not -len(x.sig) <= d < len(x.sig):
        return None
    return d % len(x.sig)


@TORCH.function("torch.softmax", f"{F}.softmax")
@TORCH.method((T,), "softmax")
def _softmax(c: Call) -> Expr | None:
    """e^{x_k} / Σ_{k'} e^{x_{k'}}, over the dimension it is given."""
    x = c.args[0]
    d = _dim(c, x)
    if d is None or set(c.kwargs) - {"dim"}:
        return None
    assert x.sig is not None
    e = Apply("exp", (x,), sig=x.sig, type=T)
    total = Aggregate("sum", (x.sig[d],), e, sig=tuple(NEW if k == d else a for k, a in enumerate(x.sig)), type=T)
    return Apply("div", (e, total), sig=x.sig, type=T)


@TORCH.function("torch.log_softmax", f"{F}.log_softmax")
@TORCH.method((T,), "log_softmax")
def _log_softmax(c: Call) -> Expr | None:
    """x_k - logsumexp_{k'} x_{k'}."""
    x = c.args[0]
    d = _dim(c, x)
    if d is None or set(c.kwargs) - {"dim"}:
        return None
    assert x.sig is not None
    lse = Aggregate("logsumexp", (x.sig[d],), x, sig=tuple(NEW if k == d else a for k, a in enumerate(x.sig)), type=T)
    return Apply("sub", (x, lse), sig=x.sig, type=T)


@TORCH.function(f"{F}.cross_entropy")
def _cross_entropy(c: Call) -> Expr | None:
    """The mean, over the samples, of the log of the softmax at each sample's label, negated (the default:
    targets as class positions, classes on dim 1, reduction="mean", no weights, no smoothing)."""
    if len(c.args) != 2 or c.kwargs:
        return None
    z, y = c.args
    if z.sig is None or len(z.sig) != 2 or y.sig is None or len(y.sig) != 1 or not z.sig[0].known \
            or y.dtype == "float":
        return None
    n, k = z.sig
    lse = Aggregate("logsumexp", (k,), z, sig=(n,), type=T)
    at_label = Take(z, (0, 1), (Idx(n, sig=(n,), type="int"), y), 0, sig=(n,), type=T)   # z_{n, y_n}
    return Aggregate("mean", (n,), Apply("sub", (lse, at_label), sig=(n,), type=T), sig=(), type=T)


@TORCH.function(f"{F}.mse_loss")
def _mse(c: Call) -> Expr | None:
    if len(c.args) != 2 or c.kwargs:
        return None
    a, b = c.args
    d = elementwise("sub", [a, b], T)
    return Aggregate("mean", (), Apply("square", (d,), sig=d.sig, type=T), sig=(), type=T)


@TORCH.function(f"{F}.linear")
def _linear(c: Call) -> Expr | None:
    """x Wᵀ + b."""
    if not 2 <= len(c.args) <= 3 or set(c.kwargs) - {"bias"}:
        return None
    x, W = c.args[0], c.args[1]
    b = c.args[2] if len(c.args) > 2 else c.kwargs.get("bias")
    out = matmul(x, transpose(W), T)
    return out if b is None else elementwise("add", [out, b], T)


@TORCH.function("torch.nn.Linear")
def _make_linear(c: Call) -> Expr | None:
    """A linear layer, its weight's and bias's index sets the sizes it is made with (nn.Linear(features,
    hidden): hidden × features), named after the variables that give them."""
    if len(c.args) < 2:
        return None
    sets = []
    for a in c.args[:2]:
        name = getattr(a, "name", None)
        sets.append(Axis(name) if isinstance(name, str) else Axis(None))
    fin, fout = sets
    return Op("Linear", tuple(c.args), tuple(c.kwargs.items()), known=True, qualname=c.name, type=LINEAR,
              parts=(Shape(T, (fout, fin)), Shape(T, (fout,))))


@TORCH.method((LINEAR,), "__call__", "forward")
def _apply_linear(c: Call) -> Expr | None:
    """layer(x): x Wᵀ + b, its weight and bias the layer's own (W^{layer1})."""
    layer, *rest = c.args
    if len(rest) != 1 or c.kwargs or layer.parts is None or len(layer.parts) != 2:
        return None
    x = rest[0]
    w_shape, b_shape = layer.parts
    W = Field(layer, "weight", sig=w_shape.sig, type=T)
    b = Field(layer, "bias", sig=b_shape.sig, type=T)
    if x.sig is not None and w_shape.sig is not None and x.sig and x.sig[-1].known and w_shape.sig[1] != x.sig[-1]:
        # the layer's input set is x's last (as it is applied): its weight indexed by it
        W = replace(W, sig=(w_shape.sig[0], x.sig[-1]))
    return elementwise("add", [matmul(x, transpose(W), T), b], T)


@TORCH.function("torch.autograd.grad")
def _grad(c: Call) -> Expr | None:
    """∇_θ L: the gradient of a scalar with respect to each of the values it is given."""
    if len(c.args) != 2 or c.kwargs:
        return None
    L, wrt = c.args
    items = wrt.items if isinstance(wrt, Tup) else (wrt,)
    return Op("grad", (L, wrt), known=True, qualname=c.name, type="tuple", parts=tuple(i.shape for i in items))


@TORCH.function("torch.zeros", "torch.ones", "torch.zeros_like", "torch.ones_like")
def _made(c: Call) -> Expr | None:
    from headway.core.libs.arrays import shape_sig
    if not c.args or not c.only("dtype", "requires_grad"):
        return None
    name = c.name.rpartition(".")[2]
    sig = c.args[0].sig if name.endswith("_like") else shape_sig(c, c.args[0] if len(c.args) == 1 else Tup(
        tuple(c.args)))
    if sig is None or any(not a.known for a in sig):
        return Op(name, tuple(c.args), tuple(c.kwargs.items()), sig=sig, type=T, qualname=c.name)
    return Const(1 if name.startswith("ones") else 0, sig=sig, type=T)


__all__ = ["TORCH", "broadcast"]
