"""What arrays' operations mean, for the libraries whose arrays have numpy's semantics (numpy, and where they agree
PyTorch and JAX): elementwise functions with broadcasting, reductions over axes, matrix products, indexing. Each
library's table calls these with its own conventions (numpy's axis=, torch's dim=).
"""
from __future__ import annotations

from typing import Any

from headway.core.libs import NOT, Call
from headway.core.terms import (NEW, Aggregate, Align, Apply, Axis, Const, Expr, Extent, Idx, Index, Pos, Restrict,
                                Sig, SliceKey, Take, Tup, Var, broadcast, known, with_sig)


def elementwise(fn: str, args: list[Expr], type_: str = "") -> Apply:
    return Apply(fn, tuple(args), sig=broadcast(*(a.sig for a in args)),
                 type=type_ or next((a.type for a in args if a.sig), "") or (args[0].type if args else ""))


def axes_arg(c: Call, x: Expr, value: Expr | None) -> Any:
    """The axes an axis= argument names, as positions of x (negative ones counted from the end); None for all of
    them; NOT if it is not a literal, or names an axis x does not have."""
    if value is None:
        return None
    got = c.literal(value)
    if got is None:
        return None
    if isinstance(got, bool) or got is NOT:
        return NOT
    items = [got] if isinstance(got, int) else list(got) if isinstance(got, tuple) else None
    if items is None or not all(isinstance(k, int) for k in items):
        return NOT
    if x.sig is None:
        return items if all(k >= 0 for k in items) else NOT
    n = len(x.sig)
    if not all(-n <= k < n for k in items):
        return NOT
    return sorted({k % n for k in items})


def reduce(c: Call, op: str, x: Expr, axis: Expr | None, keepdims: Expr | None = None) -> Expr | None:
    """sum, mean, max, ... over some axes of x (or all): over their index sets, if known; else over positions."""
    axes = axes_arg(c, x, axis)
    if axes is NOT:
        return None
    keep = c.literal(keepdims, False)
    if keep is NOT:
        return None
    type_ = x.type if x.type not in ("bool",) else "numpy.ndarray"
    if x.sig is None:
        if keep and axes is not None:
            return Aggregate(op, tuple(Pos(k) for k in axes), x, sig=None, type=type_)   # type: ignore[union-attr]
        return Aggregate(op, () if axes is None else tuple(Pos(k) for k in axes), x,  # type: ignore[union-attr]
                         sig=() if axes is None else None, type=type_)
    positions = list(range(len(x.sig))) if axes is None else axes
    over: list[Index] = [x.sig[k] if x.sig[k].known else Pos(k) for k in positions if x.sig[k] != NEW]  # type: ignore
    if axes is None and any(isinstance(i, Pos) for i in over):
        over = []                                       # all of them, some not known: over all
    sig = tuple(NEW if k in positions else a for k, a in enumerate(x.sig)) if keep else \
        tuple(a for k, a in enumerate(x.sig) if k not in positions)
    return Aggregate(op, tuple(over), x, sig=sig, type=type_ if sig else _scalar_type(x))


def _scalar_type(x: Expr) -> str:
    return "float"


def copy_of(a: Axis, taken: set[Axis]) -> Axis:
    """A copy of an index set's axis that no other axis in `taken` is (k', k'', ...)."""
    k = a.copy + 1
    while Axis(a.name, k) in taken:
        k += 1
    return Axis(a.name, k, a.where, a.part, a.letter)


def matmul(a: Expr, b: Expr, type_: str = "") -> Expr:
    """a @ b: a sum over the axis the two share (a's last, b's second to last, or its only one) of their product.
    With the indices known, written as such: Σ_g a_{bg} v_{gk}; the two named differently (a's g, v's d), a sum over
    both where they are equal. A free axis of b that is also one of a's (Xᵀ X) is a copy of it (k'). Without the
    indices, a matrix product, as written."""
    type_ = type_ or a.type or b.type
    if not (known(a.sig) and known(b.sig)) or not a.sig or not b.sig:
        if a.sig is not None and b.sig is not None and a.sig and b.sig:
            sig: Sig = (*a.sig[:-1], *(b.sig[:-2] + b.sig[-1:] if len(b.sig) > 1 else ()))
            return Apply("matmul", (a, b), sig=sig, type=type_)
        return Apply("matmul", (a, b), sig=None, type=type_)
    assert a.sig is not None and b.sig is not None
    if len(a.sig) > 2 or len(b.sig) > 2:            # batches: not yet
        return Apply("matmul", (a, b), sig=None, type=type_)
    left = a.sig[-1]
    kb = 0 if len(b.sig) == 1 else len(b.sig) - 2
    right = b.sig[kb]
    a_free = list(a.sig[:-1])
    b_free = [x for k, x in enumerate(b.sig) if k != kb]
    # b's free axes that a has too, and its summed axis if a has it free: copies
    taken = set(a.sig)
    renamed = list(b.sig)
    for k, x in enumerate(b.sig):
        if k != kb and x in taken:
            renamed[k] = copy_of(x, taken | set(renamed))
            taken.add(renamed[k])
    if right != left and right in a_free:
        renamed[kb] = copy_of(right, taken | set(renamed))
    b2 = b if tuple(renamed) == b.sig else Align(b, "rename", sig=tuple(renamed), type=b.type)
    right = renamed[kb]
    b_free = [x for k, x in enumerate(renamed) if k != kb]
    product = Apply("mul", (a, b2), sig=tuple(dict.fromkeys([*a.sig, *renamed])), type=type_)
    out_sig = (*a_free, *b_free)
    if left == right:
        return Aggregate("sum", (left,), product, sig=out_sig, type=type_ if out_sig else "float")
    tie = Apply("eq", (Idx(left, sig=(left,), type="int"), Idx(right, sig=(right,), type="int")),
                sig=(left, right), type="bool")
    return Aggregate("sum", (left, right), product, tie, sig=out_sig, type=type_ if out_sig else "float")


def outer(a: Expr, b: Expr, type_: str = "") -> Expr:
    """a's and b's elements, every pair multiplied: a_i b_j (b's axis a copy if it is a's)."""
    if a.sig is not None and b.sig is not None and len(a.sig) == 1 and len(b.sig) == 1:
        rb = b.sig[0]
        if rb == a.sig[0] and rb.known:
            rb = copy_of(rb, {a.sig[0]})
            b = Align(b, "rename", sig=(rb,), type=b.type)
        sig = (a.sig[0], rb)
        return Apply("mul", (Align(a, "expand", sig=(a.sig[0], NEW), type=a.type),
                             Align(b, "expand", sig=(NEW, rb), type=b.type)), sig=sig, type=type_ or a.type)
    return Apply("outer", (a, b), sig=None, type=type_ or a.type)


def transpose(x: Expr) -> Expr:
    return Align(x, "transpose", sig=None if x.sig is None else tuple(reversed(x.sig)), type=x.type)


def shape_sig(c: Call, shape: Expr | None) -> Sig:
    """The axes of an array made in a shape: each length an extent of an index set (len(x), x.shape[1]), else an
    axis whose set is not known; None if the number of axes is not known."""
    if shape is None:
        return None
    items = list(shape.items) if isinstance(shape, Tup) else [shape]
    out: list[Axis] = []
    for it in items:
        if isinstance(it, Extent) and isinstance(it.axis, Axis):
            out.append(it.axis)
        elif it.parts and len(it.parts) == 1 and it.parts[0].type == "extent" and it.parts[0].sig:
            out.append(it.parts[0].sig[0])              # a name holding an index set's size (n = x.shape[1])
        elif it.sig == () or isinstance(it, Const):
            out.append(Axis(None))
        else:
            return None
    return tuple(out)


def extent(x: Expr, k: int) -> Expr:
    """The length of x's axis k (len(x): k = 0): its index set's size (or the size of the part of it x has)."""
    if x.sig is not None and -len(x.sig) <= k < len(x.sig) and x.sig[k].known and x.sig[k] != NEW:
        return Extent(x.sig[k], x, sig=(), type="int")
    return Extent(Pos(k), x, sig=(), type="int")


def getitem(c: Call, base: Expr, keys: list[Expr]) -> Expr | None:
    """x[...], numpy's indexing (basic and advanced): an integer drops its axis, a slice keeps a part of it, None
    puts in an axis of length one, a mask keeps the part where it holds, an array of positions (a pointer) puts its
    own axes in place of the one it indexes (several pointers together: their broadcast axes, at the first one's
    place if they are side by side, else first). None if not known well enough to say."""
    if base.sig is None:
        if all(_full(k) for k in keys):
            return base
        return None
    sig = list(base.sig)
    # expand an ellipsis; a mask over several axes is not read
    consuming = [k for k in keys if not _none(k) and not _ellipsis(k)]
    if sum(_ellipsis(k) for k in keys) > 1 or len(consuming) > len(sig):
        return None
    if any(_ellipsis(k) for k in keys):
        at = next(i for i, k in enumerate(keys) if _ellipsis(k))
        keys = keys[:at] + [SliceKey(None, None, None)] * (len(sig) - len(consuming)) + keys[at + 1:]
    keys = keys + [SliceKey(None, None, None)] * (len(sig) - len(consuming))
    kinds = [_kind(k) for k in keys]
    if "unknown" in kinds and (kinds.count("unknown") > 1 or any(x in ("mask", "pointer") for x in kinds)):
        return None
    advanced = [i for i, k in enumerate(kinds) if k in ("mask", "pointer")]
    if advanced:
        ints = [i for i, k in enumerate(kinds) if k == "int"]
        if any(k in ("slice", "new") for k in kinds):
            return None                                 # advanced with partial slices or new axes: not yet
        positions = sorted(advanced + ints)
        # each key's input axis
        axis_of = {i: i for i in range(len(kinds))}       # (no new axes here: each key is its axis)
        if len(positions) == 1 and kinds[positions[0]] == "mask":
            i = positions[0]
            mask = keys[i]
            pos = axis_of[i]
            a = sig[pos]
            restricted = Axis(a.name, a.copy, mask, a.part, a.letter) if a.known else Axis(None)
            new = tuple(restricted if k == pos else x for k, x in enumerate(sig))
            return Restrict(base, pos, mask=mask, sig=new, type=base.type)
        if any(kinds[i] == "mask" for i in positions):
            return None                                 # a mask with other arrays: not yet
        index_sig = broadcast(*(keys[i].sig for i in positions))
        if index_sig is None:
            return None
        adjacent = positions == list(range(positions[0], positions[-1] + 1))
        rest = [x for k, x in enumerate(sig) if k not in [axis_of[i] for i in positions]]
        at = positions[0] if adjacent else 0
        new_sig = tuple(rest[:at]) + tuple(index_sig) + tuple(rest[at:])
        return Take(base, tuple(axis_of[i] for i in positions), tuple(keys[i] for i in positions), at,
                    sig=new_sig, type=base.type if new_sig else "float")
    # basic indexing, one key at a time, left to right
    out = base
    cur = list(sig)
    p = 0
    for k, kind in zip(keys, kinds, strict=True):
        if kind == "new":
            cur.insert(p, NEW)
            out = Align(out, "expand", sig=tuple(cur), type=out.type)
            p += 1
        elif kind == "full":
            p += 1
        elif kind == "unknown":                         # x[:, module], module not known: the part of g in it
            a = cur[p]
            cur[p] = Axis(a.name, a.copy, k, a.part, a.letter) if a.known else Axis(None)
            out = Restrict(out, p, mask=k, sig=tuple(cur), type=out.type)
            p += 1
        elif kind == "int":
            del cur[p]
            out = Take(out, (p,), (k,), p, sig=tuple(cur), type=out.type if cur else "float")
        elif kind == "slice":
            a = cur[p]
            if not isinstance(k, SliceKey):             # a slice held by a variable: the part it names
                k = SliceKey(k, None, None, type="slice")
                span = _text(k.lo)
            else:
                span = ":".join(_text(x) for x in (k.lo, k.hi)) + (":" + _text(k.step) if k.step is not None
                                                                   else "")
            cur[p] = Axis(a.name, a.copy, a.where, (a.part + "|" if a.part else "") + span, a.letter) if a.known \
                else Axis(None)
            out = Restrict(out, p, part=(k.lo, k.hi, k.step), sig=tuple(cur), type=out.type)
            p += 1
        else:
            return None
    return out


def _text(e: Expr | None) -> str:
    if e is None:
        return ""
    if isinstance(e, Const):
        return repr(e.value)
    if isinstance(e, Var):
        return e.name
    return repr(e)


def _full(k: Expr) -> bool:
    return isinstance(k, SliceKey) and k.lo is None and k.hi is None and k.step is None


def _none(k: Expr) -> bool:
    return isinstance(k, Const) and k.value is None


def _ellipsis(k: Expr) -> bool:
    return isinstance(k, Const) and k.value is Ellipsis


def _kind(k: Expr) -> str:
    if _none(k):
        return "new"
    if _full(k):
        return "full"
    if isinstance(k, SliceKey) or k.type == "slice":   # x[a:b], or x[c] with c a slice
        return "slice"
    if isinstance(k, Const) and type(k.value) is int:
        return "int"
    if isinstance(k, Const) and isinstance(k.value, bool):
        return "bad"
    if k.sig == () and k.type in ("int", ""):
        return "int" if k.type == "int" else "unknown"
    if k.sig is None:
        return "unknown"
    if k.sig == ():
        return "int"
    if k.type == "bool" or k.dtype == "bool":
        return "mask"
    if k.dtype == "int" or k.type == "int":
        return "pointer"
    return "unknown"                                    # positions or a mask: not known which


def with_type(e: Expr, type_: str) -> Expr:
    return with_sig(e, e.sig, type_)
