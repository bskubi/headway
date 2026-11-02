"""The reference evaluator: terms run on real values, by the calculus's own meaning (docs/core.md §7).

It is how what is shown is checked. A function's Program (as lowered, and as folded for display) is run on the
arguments of a traced call, and its results compared with the code's own: if a library's translation, a signature,
or the fold says something the code does not do, the values differ. It does not use the tables: an aggregate sums
over the axis its index set names, wherever that axis is; an elementwise operation aligns its operands by index set,
not by position. Only where the indices are not known does it follow numpy's positional rules (as the term then says).

Operators (Op) are called: what one means is what the function it names does. A random draw is taken from the
generator the code passes, so a run from the same seed draws the same values if the draws come in the same order.
"""
from __future__ import annotations

import copy
import importlib
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from headway.core.terms import (NEW, Aggregate, Align, Apply, Assume, Axis, Binder, Build, Cases, Const, Density,
                                Dist, Draw, Effect, Expr, Extent, Field, For, Idx, If, Item, Jump, Key, Lambda, Let,
                                Mapping, Op, Optimum, Pos, Program, Range, Restrict, Return, SetPart, Sig, SliceKey, Stmt, Table, Take,
                                Text, Tup, Unwritten, Var, While)


class Uncheckable(Exception):
    """The program has a part that cannot be run (a statement not written, an operator not found)."""


class Mismatch(Exception):
    """A value is not what its signature says (an index set of two sizes, an axis not where it should be)."""


class _Return(Exception):
    def __init__(self, value: Any):
        self.value = value


class _Jump(Exception):
    def __init__(self, kind: str):
        self.kind = kind


def _ln(x: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(x)


PRIMITIVES: dict[str, Callable[..., Any]] = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.true_divide, "pow": np.power,
    "neg": np.negative, "mod": np.mod, "floordiv": np.floor_divide, "lshift": np.left_shift, "rshift": np.right_shift, "log": _ln, "exp": np.exp, "sqrt": np.sqrt,
    "abs": np.abs, "log1p": np.log1p, "expm1": np.expm1, "log2": np.log2, "log10": np.log10,
    "sigmoid": lambda x: 1 / (1 + np.exp(-np.asarray(x, np.float64))),
    "logit": lambda p: _ln(np.asarray(p, np.float64) / (1 - np.asarray(p, np.float64))),
    "tanh": np.tanh, "sinh": np.sinh, "cosh": np.cosh, "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan, "floor": np.floor, "ceil": np.ceil,
    "round": np.round, "sign": np.sign, "isnan": np.isnan, "isfinite": np.isfinite, "square": np.square,
    "isinf": np.isinf,
    "max2": np.maximum, "min2": np.minimum, "relu": lambda x: np.maximum(x, 0),
    "softplus": lambda x: np.logaddexp(0, x),
    "eq": lambda a, b: a == b if _scalars(a, b) else np.equal(a, b),
    "ne": lambda a, b: a != b if _scalars(a, b) else np.not_equal(a, b),
    "lt": np.less, "le": np.less_equal, "gt": np.greater, "ge": np.greater_equal,
    "and": lambda a, b: (a and b) if _scalars(a, b) else np.logical_and(a, b),
    "or": lambda a, b: (a or b) if _scalars(a, b) else np.logical_or(a, b),
    "not": lambda a: (not a) if _scalars(a) else np.logical_not(a),
    "in": lambda a, b: a in b, "notin": lambda a, b: a not in b,
    "is": lambda a, b: a is b, "isnot": lambda a, b: a is not b,
    "matmul": lambda a, b: a @ b, "inv": np.linalg.inv, "det": np.linalg.det, "trace": np.trace,
    "solve": np.linalg.solve, "norm": np.linalg.norm, "transpose": np.transpose,
    "float": float, "int": int, "len": len, "size": np.size,
}


def _scalars(*xs: Any) -> bool:
    """Python's values (None, a dict, a bool), not arrays: compared and combined as Python does."""
    return any(x is None or isinstance(x, (dict, list, tuple, str, bool)) for x in xs) \
        and not any(isinstance(x, np.ndarray) for x in xs)


def _special(name: str) -> Callable[..., Any]:
    import scipy.special as sp
    return getattr(sp, name)  # type: ignore[no-any-return]


for _n, _s in (("erf", "erf"), ("gammaln", "gammaln"), ("digamma", "digamma"), ("gamma", "gamma")):
    PRIMITIVES[_n] = (lambda s: lambda x: _special(s)(x))(_s)
PRIMITIVES["trigamma"] = lambda x: _special("polygamma")(1, x)

WHOLE = {"matmul", "inv", "det", "trace", "solve", "norm", "transpose", "len", "size", "in", "notin", "float",
         "int", "is", "isnot"}                                        # primitives of whole values, not elementwise

REDUCERS: dict[str, Callable[..., Any]] = {
    "sum": np.sum, "prod": np.prod, "mean": np.mean, "max": np.max, "min": np.min, "any": np.any, "all": np.all,
    "var": np.var, "sd": np.std, "median": np.median,
}


class Evaluator:
    """Runs Programs. resolve(qualname) -> the function an operator names; names: the module's other values (its
    functions, constants), for operators named by a variable."""

    def __init__(self, resolve: Callable[[str], Any] | None = None, names: dict[str, Any] | None = None):
        self.resolve = resolve or _import
        self.names = dict(names or {})
        self.extents: dict[str, int] = {}               # index set -> its size
        self.labels: dict[Axis, dict[str, Any]] = {}    # a group_by's groups -> their keys' values
        self.failed: list[tuple[int, str]] = []          # preconditions that did not hold: (line, what)
        self.env: dict[str, Any] = {}

    # ---- programs ---------------------------------------------------------------------------------------------

    def run(self, prog: Program | list[Stmt], args: dict[str, Any], sigs: dict[str, Sig] | None = None) -> Any:
        """Run a program on its arguments (by name; a copy is taken), its parameters' signatures checked against
        their values. The returned value, or the environment when the program returns nothing."""
        body = prog.body if isinstance(prog, Program) else prog
        self.env = {k: safe_copy(v) for k, v in args.items()}
        params = dict(sigs or {})
        if isinstance(prog, Program):
            params |= {n: s for n, s, _ in prog.params}
        for n, s in params.items():
            if n in self.env:
                self.observe(self.env[n], s, n)
        try:
            self.block(list(body))
        except _Return as r:
            return r.value
        return None

    def block(self, body: list[Stmt] | tuple[Stmt, ...]) -> None:
        for st in body:
            self.stmt(st)

    def stmt(self, st: Stmt) -> None:
        if isinstance(st, Let):
            value = self.ev(st.value)
            if len(st.targets) == 1:
                self.env[st.targets[0]] = value
            else:
                items = list(value)
                if len(items) != len(st.targets):
                    raise Mismatch(f"line {st.line}: {len(items)} values for {len(st.targets)} names")
                for n, v in zip(st.targets, items):
                    self.env[n] = v
        elif isinstance(st, Return):
            raise _Return(None if st.value is None else self.ev(st.value))
        elif isinstance(st, Assume):
            ok = self.ev(st.test)
            if not bool(np.all(ok)):
                self.failed.append((st.line, "does not hold"))
        elif isinstance(st, For):
            for item in self.domain(st.domain):
                if st.var is not None:
                    self.bind(st.var, item, st.domain)
                try:
                    self.block(st.body)
                except _Jump as j:
                    if j.kind == "break":
                        break
            else:
                self.block(st.orelse)
        elif isinstance(st, While):
            while bool(self.ev(st.test)):
                try:
                    self.block(st.body)
                except _Jump as j:
                    if j.kind == "break":
                        break
            else:
                self.block(st.orelse)
        elif isinstance(st, If):
            self.block(st.body if bool(self.ev(st.test)) else st.orelse)
        elif isinstance(st, Effect):
            self.ev(st.value)
        elif isinstance(st, SetPart):
            value = self.ev(st.value)
            t = st.target
            if isinstance(t, Item):
                self.ev(t.base)[self.key(t.key)] = value
            elif isinstance(t, Field):
                setattr(self.ev(t.base), t.attr, value)
            else:
                raise Uncheckable(f"line {st.line}: a change in place")
        elif isinstance(st, Jump):
            raise _Jump(st.kind)
        elif isinstance(st, Unwritten):
            raise Uncheckable(f"line {st.line}: not written")
        else:
            raise Uncheckable(f"line {st.line}: {type(st).__name__}")

    def domain(self, d: Expr) -> Any:
        if isinstance(d, Range):
            return range(int(self.ev(d.lo)), int(self.ev(d.hi)))
        return self.ev(d)

    def bind(self, var: str | tuple[Any, ...], item: Any, domain: Expr | None = None) -> None:
        if isinstance(var, str):
            self.env[var] = item
            return
        if len(var) == 1:
            self.env[var[0]] = item
            return
        items = list(item)
        if len(items) != len(var):
            raise Mismatch(f"{len(items)} values for {var}")
        for v, x in zip(var, items):
            self.bind(v, x)

    # ---- signatures ------------------------------------------------------------------------------------------

    def observe(self, value: Any, sig: Sig, what: str) -> None:
        """A value said to have a signature: as many axes, and each index set of one size everywhere."""
        if sig is not None and len(sig) == 1 and _is_table(value) and sig[0].known and sig[0].name:
            seen = self.extents.setdefault(sig[0].name, value.height)       # a table: its rows
            if seen != value.height:
                raise Mismatch(f"{what}: {value.height} rows, its index set {sig[0].name} has {seen} elsewhere")
            return
        if sig is None or not (isinstance(value, np.ndarray | np.generic | float | int) or _is_torch(value)):
            return
        a = value if _is_torch(value) else np.asarray(value)
        if a.ndim != len(sig):
            raise Mismatch(f"{what}: {a.ndim} axes, its signature has {len(sig)}")
        for n, ax in zip(a.shape, sig):
            if ax.known and ax != NEW and ax.where is None and ax.part is None:
                assert ax.name is not None
                seen = self.extents.setdefault(ax.name, n)
                if seen != n:
                    raise Mismatch(f"{what}: index set {ax.name} has {n} elements here, {seen} elsewhere")

    # ---- expressions -----------------------------------------------------------------------------------------

    def function(self, e: Lambda) -> Any:
        """A function of the code as a value: called with its parameters' values, its body evaluated with them,
        in the variables it had where it was made."""
        names = [b.var for b in e.over if isinstance(b.var, str)]
        captured = dict(self.env)

        def call(*values: Any) -> Any:
            saved = self.env
            self.env = {**captured, **dict(zip(names, values))}
            try:
                return self.ev(e.body)
            finally:
                self.env = saved
        return call

    def optimum(self, e: Optimum) -> Any:
        """The optimizer the code names, run on the function as written (its result: x, fun, success)."""
        import scipy.optimize
        fun = self.ev(e.fun)
        jac = self.ev(e.jac) if e.jac is not None else None
        method = self.ev(e.method) if e.method is not None else None
        bounds = self.ev(e.bounds) if e.bounds is not None else None
        sign = 1.0 if e.op == "min" else -1.0
        f = fun if sign > 0 else (lambda x: -fun(x))
        return scipy.optimize.minimize(f, self.ev(e.start), jac=jac, method=method, bounds=bounds)

    def ev(self, e: Expr) -> Any:
        v = self._ev(e)
        if e.sig is not None and isinstance(v, np.ndarray | np.generic) and not isinstance(e, Var):
            if np.ndim(v) != len(e.sig):
                raise Mismatch(f"{_what(e)}: {np.ndim(v)} axes, its signature has {len(e.sig)}")
        return v

    def _ev(self, e: Expr) -> Any:
        if isinstance(e, Var):
            if e.name in self.env:
                v = self.env[e.name]
            elif e.name in self.names:
                v = self.names[e.name]
            elif (v := _named_value(e.name)) is not None and (e.type == "module" or isinstance(v, type)):
                pass                                    # int, dict, np.int64: a type the code names
            else:
                raise Uncheckable(f"{e.name} has no value")
            self.observe(v, e.sig, e.name)
            return v
        if isinstance(e, Const):
            if e.sig:
                shape = [self.extent(a) for a in e.sig]
                return np.full(shape, e.value, dtype=np.float64)
            return e.value
        if isinstance(e, Text):
            return e.value
        if isinstance(e, Field):
            base = self.ev(e.base)
            if _is_table(base):                         # a table's column: its values
                return np.asarray(base.get_column(e.attr).to_numpy())
            return base[e.attr] if isinstance(base, dict) else getattr(base, e.attr)
        if isinstance(e, Table):
            return self.table(e)
        if isinstance(e, Key):
            if e.axis not in self.labels:
                raise Uncheckable(f"the groups of {e.axis.name}")
            return self.labels[e.axis][e.name]
        if isinstance(e, Apply):
            v = self.apply(e)
            if e.type == "int" and e.fn in ("floor", "ceil") and np.ndim(v) == 0:
                return int(v)                           # math.floor gives an int (np.floor a float)
            return v
        if isinstance(e, Aggregate):
            return self.aggregate(e)
        if isinstance(e, Build):
            return list(self.family(e.over, e.body, e.where))
        if isinstance(e, Take):
            return self.take(e)
        if isinstance(e, Restrict):
            return self.restrict(e)
        if isinstance(e, Align):
            return self.align_term(e)
        if isinstance(e, Cases):
            return self.cases(e)
        if isinstance(e, Op):
            return self.op(e)
        if isinstance(e, Lambda):
            return self.function(e)
        if isinstance(e, Optimum):
            return self.optimum(e)
        if isinstance(e, Item):
            base, key = self.ev(e.base), self.key(e.key)
            if isinstance(base, list) and isinstance(key, np.ndarray):
                base = np.asarray(base)                 # np.array(a list) read as the list: indexed as an array
            return base[key]
        if isinstance(e, Tup):
            items = [self.ev(x) for x in e.items]
            return tuple(items) if e.kind == "tuple" else set(items) if e.kind == "set" else items
        if isinstance(e, Mapping):
            if not e.over:
                return {self.ev(k): self.ev(v) for k, v in e.pairs}
            (k, v), = e.pairs
            return dict(self.family(e.over, Tup((k, v)), None))
        if isinstance(e, Idx):
            return np.arange(self.extent(e.axis))
        if isinstance(e, Extent):
            of = np.shape(self.ev(e.of)) if not isinstance(self.ev(e.of), (list, tuple)) else (len(self.ev(e.of)),)
            if isinstance(e.axis, Pos):
                return of[e.axis.k]
            if e.of.sig is not None and e.axis in [a.base() for a in e.of.sig]:
                return of[[a.base() for a in e.of.sig].index(e.axis)]
            return self.extent(e.axis)  # type: ignore[arg-type]
        if isinstance(e, Range):
            return range(int(self.ev(e.lo)), int(self.ev(e.hi)))
        if isinstance(e, SliceKey):
            return self.key(e)
        if isinstance(e, Draw):
            return self.draw(e)
        if isinstance(e, Density):
            return self.density(e)
        raise Uncheckable(type(e).__name__)

    def extent(self, a: Axis) -> int:
        if a.where is not None and a.part is None:      # the part of a set where a mask holds: how many
            return int(np.sum(np.asarray(self.ev(a.where), bool)))
        if a.name is None or a.name not in self.extents or a.where is not None or a.part is not None:
            raise Uncheckable(f"the size of index set {a.name}")
        return self.extents[a.name]

    def key(self, k: Expr) -> Any:
        if isinstance(k, SliceKey):
            return slice(*(None if x is None else self.ev(x) for x in (k.lo, k.hi, k.step)))
        if isinstance(k, Tup):
            return tuple(self.key(x) for x in k.items)
        return self.ev(k)

    def aligned(self, x: Expr) -> Any:
        return self.ev(x)

    def apply(self, e: Apply) -> Any:
        f = PRIMITIVES.get(e.fn)
        if f is None:
            raise Uncheckable(e.fn)
        if e.fn in ("and", "or") and len(e.args) == 2:   # Python's: the second not evaluated if the first decides
            first = self.ev(e.args[0])
            if np.ndim(first) == 0 and not _is_torch(first):
                if (e.fn == "and" and not first) or (e.fn == "or" and first):
                    return first
                vals = [first, self.ev(e.args[1])]
            else:
                vals = [first, self.ev(e.args[1])]
        else:
            vals = [self.ev(a) for a in e.args]
        if e.fn not in WHOLE and _named(e.sig) and all(_named(a.sig) for a in e.args):
            assert e.sig is not None
            vals = [align(v, a.sig, e.sig) if a.sig else v for v, a in zip(vals, e.args)]  # type: ignore[arg-type]
        if _torch_mode(*vals):                          # torch's values: torch's functions, for autograd
            like = next(v for v in vals if _is_torch(v))
            tf = _torch_primitives().get(e.fn)
            if tf is not None:
                return tf(*(_as_torch(v, like) for v in vals))
            vals = [v.detach().cpu().numpy() if _is_torch(v) else v for v in vals]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return f(*vals)

    def aggregate(self, e: Aggregate) -> Any:
        binders = [i for i in e.over if isinstance(i, Binder)]
        if binders:
            if len(binders) != len(e.over):
                raise Uncheckable("a sum over loop variables and axes")
            values = list(self.family(e.over, e.body, e.where))
            if e.op == "sum":
                return sum(values, start=0) if values else 0
            if e.op == "prod":
                return math.prod(values)
            if e.op in ("max", "min"):
                return (max if e.op == "max" else min)(values)
            if e.op == "mean":
                return sum(values) / len(values)
            raise Uncheckable(e.op)
        body = self.ev(e.body)
        b = _arr(body)
        sig = e.body.sig
        if not e.over:                                      # all of it
            positions = tuple(range(b.ndim))
        elif sig is None or any(isinstance(i, Pos) for i in e.over):
            if not all(isinstance(i, Pos) for i in e.over):
                raise Mismatch("a sum over an index set of a value whose index sets are not known")
            positions = tuple(i.k for i in e.over)  # type: ignore[union-attr]
        else:
            positions = tuple(_position(sig, i) for i in e.over)  # type: ignore[arg-type]
        mask = None
        if e.where is not None:
            mask = np.asarray(self.ev(e.where))
            if sig is None or not _named(e.where.sig) or not _named(sig):
                raise Uncheckable("a condition on a sum of a value whose index sets are not known")
            # a condition with index sets of its own (the group's k in key_r = k): the body taken along them too
            extra = tuple(a for a in e.where.sig if a not in sig and a != NEW)  # type: ignore[union-attr]
            full = (*sig, *extra)
            shape = [*b.shape, *(mask.shape[e.where.sig.index(a)] for a in extra)]  # type: ignore[union-attr]
            b = align(b, sig, full)
            b = b.expand(*shape) if _is_torch(b) else np.broadcast_to(b, shape)
            mask = np.broadcast_to(align(mask, e.where.sig, full), shape)  # type: ignore[arg-type]
            sig = full
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if _is_torch(b):
                out = _torch_reduce(e.op, b, positions, mask)
            elif e.op in ("argmax", "argmin"):
                if len(positions) != 1 or mask is not None:
                    raise Uncheckable(e.op)
                out = (np.argmax if e.op == "argmax" else np.argmin)(b, axis=positions[0])
            elif e.op == "logsumexp":
                top = np.max(b, axis=positions, keepdims=True, where=mask, initial=-np.inf) if mask is not None \
                    else np.max(b, axis=positions, keepdims=True)
                top = np.where(np.isfinite(top), top, 0)
                s = np.sum(np.exp(b - top), axis=positions, where=mask) if mask is not None else \
                    np.sum(np.exp(b - top), axis=positions)
                out = np.log(s) + np.squeeze(top, axis=positions)
            elif e.op == "count":
                out = np.sum(mask if mask is not None else np.ones_like(b, bool), axis=positions)
            else:
                f = REDUCERS[e.op]
                kw: dict[str, Any] = {"axis": positions}
                if mask is not None:
                    kw["where"] = mask
                    if e.op in ("max", "min"):
                        kw["initial"] = -np.inf if e.op == "max" else np.inf
                out = f(b, **kw)
        if sig is not None and e.sig is not None and _named(sig) and _named(e.sig):
            rest = tuple(a for k, a in enumerate(sig) if k not in positions)
            out = align(out, rest, e.sig)
        elif e.sig is not None and NEW in e.sig and _arr(out).ndim < len(e.sig):
            out = _expand(out, tuple(k for k, a in enumerate(e.sig) if a == NEW))
        return out

    def family(self, over: tuple[Any, ...], body: Expr, where: Expr | None) -> Any:
        """The values of body over binders (nested, the first outermost), where `where` holds."""
        if not over:
            if where is None or bool(self.ev(where)):
                yield self.ev(body)
            return
        first, rest = over[0], over[1:]
        if not isinstance(first, Binder):
            raise Uncheckable("a family over an axis")
        saved = dict(self.env)
        try:
            for item in self.domain(first.domain):
                self.bind(first.var, item)
                yield from self.family(rest, body, where)
        finally:
            self.env = saved

    def take(self, e: Take) -> Any:
        base = _arr(self.ev(e.base))
        idx = [_arr(self.ev(i)) for i in e.index]
        if _is_torch(base):
            idx = [i.long() if _is_torch(i) else __import__("torch").as_tensor(np.asarray(i)).long() for i in idx]
        n = base.ndim
        # the indexed axes moved to the front, in order; the pointers then index them together (numpy puts their
        # broadcast axes first, then the rest)
        order = [*e.pos, *(k for k in range(n) if k not in e.pos)]
        moved = _transpose(base, order)
        if all(_named(i.sig) for i in e.index) and len(e.index) > 1:
            index_sig = _broadcast_named([i.sig for i in e.index])  # type: ignore[misc]
            idx = [np.asarray(align(v, i.sig, index_sig)) if i.sig else v
                   for v, i in zip(idx, e.index)]  # type: ignore[arg-type]
        out = moved[tuple(idx)]
        k = len(np.broadcast_shapes(*(tuple(i.shape) for i in idx))) if idx else 0
        if e.base.sig is not None and e.sig is not None and _named(e.base.sig) and _named(e.sig) \
                and all(_named(i.sig) for i in e.index):
            index_sig = _broadcast_named([i.sig for i in e.index])  # type: ignore[misc]
            rest = tuple(a for j, a in enumerate(e.base.sig) if j not in e.pos)
            return align(out, (*index_sig, *rest), e.sig)
        # positional: the pointers' axes put where the term says (numpy's rule)
        axes = list(range(out.ndim))
        front, rest_axes = axes[:k], axes[k:]
        perm = rest_axes[:e.at] + front + rest_axes[e.at:]
        return _transpose(out, perm)

    def table(self, e: Table) -> Any:
        """A table: its base's columns and its own (a group_by's: one row per combination of its keys' values, in
        their sorted order)."""
        import polars as pl
        if e.groups:
            keys = {n: np.asarray(self.ev(x)) for n, x in e.groups}
            frame = pl.DataFrame(keys).unique().sort(list(keys))
            assert e.sig is not None
            self.labels[e.sig[0]] = {n: frame.get_column(n).to_numpy() for n in keys}
            self.extents[e.sig[0].name or ""] = frame.height
        base = self.ev(e.base) if e.base is not None else None
        cols = {n: np.asarray(self.ev(x)) for n, x in e.cols}
        if base is None:
            return pl.DataFrame(cols)
        return base.with_columns(**{n: pl.Series(n, v) for n, v in cols.items()})

    def restrict(self, e: Restrict) -> Any:
        raw = self.ev(e.base)
        if _is_table(raw) and e.mask is not None and e.pos == 0:  # a table's rows where the mask holds
            import polars as pl
            return raw.filter(pl.Series(np.asarray(self.ev(e.mask), bool)))
        base = _arr(raw)
        if e.part is not None:
            parts = [None if x is None else self.ev(x) for x in e.part]
            sl = parts[0] if isinstance(parts[0], slice) else slice(*parts)    # a slice a variable holds
            index: list[Any] = [slice(None)] * base.ndim
            index[e.pos] = sl
            return base[tuple(index)]
        assert e.mask is not None
        m = np.asarray(self.ev(e.mask)) if not _is_torch(self.ev(e.mask)) else self.ev(e.mask).cpu().numpy()
        if m.ndim != 1:
            raise Mismatch(f"{_what(e)}: a mask or positions of {m.ndim} axes, where its signature keeps one")
        where: list[Any] = [slice(None)] * base.ndim
        where[e.pos] = m if m.dtype == bool else m.astype(np.int64)
        return base[tuple(where)]

    def align_term(self, e: Align) -> Any:
        v = self.ev(e.base)
        if e.how == "rename":
            return v
        a = _arr(v)
        if e.how == "transpose":
            if _named(e.base.sig) and _named(e.sig) and e.base.sig is not None and e.sig is not None \
                    and sorted(map(repr, e.base.sig)) == sorted(map(repr, e.sig)) \
                    and len(set(e.base.sig)) == len(e.base.sig):
                return align(a, e.base.sig, e.sig)
            return _transpose(a)
        if e.how == "expand":
            if e.sig is None:
                return a
            new = [k for k, x in enumerate(e.sig) if x == NEW]
            have = [k for k, x in enumerate(e.base.sig or ()) if x == NEW]
            return _expand(a, tuple(new)) if not have else a.reshape(
                [1 if x == NEW else n for x, n in zip(e.sig, _fill(tuple(a.shape), e.sig))])
        if e.how == "broadcast":
            assert e.sig is not None
            shape = [self.extent(x) if x != NEW else 1 for x in e.sig]
            return a.expand(*shape).clone() if _is_torch(a) else np.broadcast_to(a, shape).copy()
        if e.how == "squeeze":
            return _squeeze(a) if e.base.sig is None else _squeeze(
                a, tuple(k for k, x in enumerate(e.base.sig) if x == NEW))
        raise Uncheckable(e.how)

    def cases(self, e: Cases) -> Any:
        conds = [self.ev(c) for c, _ in e.branches]
        if all(np.ndim(c) == 0 for c in conds):             # decided once: only the branch taken is run
            for c, (_, v) in zip(conds, e.branches):
                if bool(c):
                    return self.ev(v)
            return self.ev(e.otherwise)
        sig = e.sig
        vals = [self.ev(v) for _, v in e.branches] + [self.ev(e.otherwise)]
        exprs = [v for _, v in e.branches] + [e.otherwise]
        cexprs = [c for c, _ in e.branches]
        if _named(sig) and all(_named(x.sig) for x in [*exprs, *cexprs]):
            assert sig is not None
            vals = [align(v, x.sig, sig) if x.sig else v for v, x in zip(vals, exprs)]  # type: ignore[arg-type]
            conds = [align(c, x.sig, sig) if x.sig else c for c, x in zip(conds, cexprs)]  # type: ignore[arg-type]
        out = vals[-1]
        for c, v in reversed(list(zip(conds, vals[:-1]))):
            if _torch_mode(c, v, out):
                import torch
                like = next(x for x in (c, v, out) if _is_torch(x))
                out = torch.where(_as_torch(c, like).bool(), _as_torch(v, like), _as_torch(out, like))
            else:
                out = np.where(c, v, out)
        return out

    def op(self, e: Op) -> Any:
        args: list[Any] = []
        for a in e.args:
            if isinstance(a, Op) and a.qualname == "*":         # f(*xs): its items as arguments
                args += list(self.ev(a.args[0]))
            else:
                args.append(self.ev(a))
        kwargs = {k: self.ev(v) for k, v in e.kwargs}
        if e.qualname == "builtins.super" and not args and "__class__" in self.names and "self" in self.env:
            return super(self.names["__class__"], self.env["self"])     # super() in a method
        if e.qualname == "builtins.str.format":        # an f-string: its text
            return args[0].format(*args[1:])
        if e.qualname in ("builtins.enumerate", "builtins.zip"):
            return list(enumerate(*args) if e.qualname.endswith("enumerate") else zip(*args))
        f: Any = None
        if e.name in self.env and callable(self.env[e.name]) and not e.local:
            f = self.env[e.name]
        elif e.local and e.name in self.names:
            f = self.names[e.name]
        elif e.qualname and not e.qualname.startswith(("?.", "numpy.ndarray.")) and "." in e.qualname:
            try:
                f = self.resolve(e.qualname)
            except (ImportError, AttributeError):
                f = None
        elif e.name in self.names:
            f = self.names[e.name]
        if f is None and args and hasattr(args[0], e.name):   # a method of what it is called on
            return getattr(args[0], e.name)(*args[1:], **kwargs)
        if f is None:
            raise Uncheckable(f"operator {e.qualname or e.name}")
        return f(*args, **kwargs)

    def draw(self, e: Draw) -> Any:
        if e.rng is None:
            gen: Any = np.random
        else:
            gen = self.ev(e.rng)
        params = {k: self.ev(v) for k, v in e.dist.params}
        size: tuple[int, ...] | None = tuple(self.extent(a) for a in e.sig) if e.sig else None
        names = {"normal": ("loc", "scale"), "binomial": ("n", "p"), "poisson": ("lam",), "gamma": ("shape", "scale"),
                 "beta": ("a", "b"), "uniform": ("low", "high"), "exponential": ("scale",),
                 "negative_binomial": ("n", "p"), "lognormal": ("mean", "sigma"), "laplace": ("loc", "scale"),
                 "geometric": ("p",)}
        if e.dist.family not in names:
            raise Uncheckable(e.dist.family)
        if e.sig and _named(e.sig):
            assert size is not None
            params = {k: np.broadcast_to(align(np.asarray(v), x.sig, e.sig), size) if x.sig else v
                      for (k, x), v in zip(e.dist.params, params.values())}  # type: ignore[arg-type]
        args = [params[k] for k in names[e.dist.family]]
        return getattr(gen, e.dist.family)(*args, size=size)

    def density(self, e: Density) -> Any:
        x = self.ev(e.at)
        p = {k: np.asarray(self.ev(v), np.float64) for k, v in e.dist.params}
        if _named(e.sig) and e.sig is not None:
            x = align(np.asarray(x), e.at.sig, e.sig) if e.at.sig else x  # type: ignore[arg-type]
            p = {k: align(v, d.sig, e.sig) if d.sig else v for (k, d), v in zip(e.dist.params, p.values())}
        lg = log_density(e.dist, np.asarray(x, np.float64), p)
        return lg if e.log else np.exp(lg)


def _named_value(name: str) -> Any:
    """A builtin's or numpy's name used as a value (a dtype, a constructor), if it is one."""
    import builtins
    if hasattr(builtins, name):
        return getattr(builtins, name)
    return getattr(np, name, None) if not name.startswith("_") else None


def log_density(d: Dist, x: np.ndarray, p: dict[str, np.ndarray]) -> np.ndarray:
    """The log density (or mass) of a distribution, from its formula (not from scipy's code)."""
    from scipy.special import gammaln
    f = d.family
    with np.errstate(divide="ignore", invalid="ignore"):
        if f == "normal":
            z = (x - p["loc"]) / p["scale"]
            return -0.5 * z ** 2 - np.log(p["scale"]) - 0.5 * np.log(2 * np.pi)
        if f == "binomial":
            n, q = p["n"], p["p"]
            return gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1) + _xlogy(x, q) + _xlogy(n - x, 1 - q)
        if f == "bernoulli":
            return _xlogy(x, p["p"]) + _xlogy(1 - x, 1 - p["p"])
        if f == "poisson":
            return _xlogy(x, p["mu"]) - p["mu"] - gammaln(x + 1)
        if f == "beta":
            a, b = p["a"], p["b"]
            return (a - 1) * np.log(x) + (b - 1) * np.log1p(-x) - (gammaln(a) + gammaln(b) - gammaln(a + b))
        if f == "gamma":
            a, s = p["a"], p["scale"]
            y = (x - p["loc"]) / s
            return (a - 1) * np.log(y) - y - gammaln(a) - np.log(s)
        if f == "exponential":
            y = (x - p["loc"]) / p["scale"]
            return np.where(y >= 0, -y - np.log(p["scale"]), -np.inf)
        if f == "negative_binomial":                     # scipy's nbinom(n, p): failures before the n-th success
            n, q = p["n"], p["p"]
            return gammaln(x + n) - gammaln(n) - gammaln(x + 1) + _xlogy(n, q) + _xlogy(x, 1 - q)
    raise Uncheckable(f)


def safe_copy(x: Any) -> Any:
    """A copy of a value, to run code on without changing it: a deep copy; a torch tensor made by a computation (not
    a leaf, which torch cannot deep-copy) as a new leaf of the same values (requiring a gradient if it did)."""
    if _is_torch(x) and hasattr(x, "grad_fn") and x.grad_fn is not None:
        return x.detach().clone().requires_grad_(x.requires_grad)
    if isinstance(x, (list, tuple)) and any(_is_torch(i) for i in x):
        return type(x)(safe_copy(i) for i in x)
    if isinstance(x, dict):
        return {k: safe_copy(v) for k, v in x.items()}
    try:
        return copy.deepcopy(x)
    except Exception:                                   # noqa: BLE001 (a value that cannot be copied: itself)
        return x


def _xlogy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x == 0, 0.0, x * np.log(y))


# ---- arrays: numpy's, or torch's (a torch value stays one, so that autograd sees what the math computes) ---------

def _is_torch(x: Any) -> bool:
    return type(x).__module__.split(".")[0] == "torch"


def _torch_mode(*xs: Any) -> bool:
    return any(_is_torch(x) for x in xs)


def _as_torch(x: Any, like: Any) -> Any:
    import torch
    if _is_torch(x):
        return x
    if isinstance(x, (bool, int, float)) and not isinstance(x, np.ndarray):
        return torch.as_tensor(x, dtype=like.dtype if x.__class__ is float or like.dtype.is_floating_point else None)
    return torch.as_tensor(np.asarray(x))


def _transpose(a: Any, perm: list[int] | tuple[int, ...] | None = None) -> Any:
    if _is_torch(a):
        return a.permute(*(perm if perm is not None else reversed(range(a.dim()))))
    return np.transpose(a, perm)


def _expand(a: Any, axes: tuple[int, ...]) -> Any:
    if _is_torch(a):
        for k in sorted(axes):
            a = a.unsqueeze(k)
        return a
    return np.expand_dims(a, axes)


def _squeeze(a: Any, axes: tuple[int, ...] | None = None) -> Any:
    if _is_torch(a):
        if axes is None:
            return a.squeeze()
        for k in sorted(axes, reverse=True):
            a = a.squeeze(k)
        return a
    return np.squeeze(a) if axes is None else np.squeeze(a, axes)


def _arr(x: Any) -> Any:
    """x as an array: a torch tensor as it is, anything else numpy's."""
    return x if _is_torch(x) else np.asarray(x)


def _torch_primitives() -> dict[str, Callable[..., Any]]:
    import torch
    import torch.nn.functional as F
    return {
        "add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div, "pow": torch.pow, "neg": torch.neg,
        "log": torch.log, "exp": torch.exp, "sqrt": torch.sqrt, "abs": torch.abs, "log1p": torch.log1p,
        "expm1": torch.expm1, "sigmoid": torch.sigmoid, "logit": torch.logit, "tanh": torch.tanh,
        "relu": torch.relu, "softplus": F.softplus, "max2": torch.maximum, "min2": torch.minimum,
        "eq": torch.eq, "ne": torch.ne, "lt": torch.lt, "le": torch.le, "gt": torch.gt, "ge": torch.ge,
        "and": torch.logical_and, "or": torch.logical_or, "not": torch.logical_not, "floor": torch.floor,
        "ceil": torch.ceil, "sign": torch.sign, "square": torch.square, "sin": torch.sin, "cos": torch.cos,
        "matmul": torch.matmul, "inv": torch.linalg.inv, "solve": torch.linalg.solve, "det": torch.linalg.det,
        "trace": torch.trace, "norm": torch.linalg.norm, "gammaln": torch.lgamma, "digamma": torch.digamma,
        "isnan": torch.isnan, "isfinite": torch.isfinite, "round": torch.round, "mod": torch.remainder,
        "floordiv": torch.floor_divide, "log2": torch.log2, "log10": torch.log10, "erf": torch.erf,
    }


def _torch_reduce(op: str, b: Any, dims: tuple[int, ...], mask: Any) -> Any:
    import torch
    if op in ("argmax", "argmin"):
        return (torch.argmax if op == "argmax" else torch.argmin)(b, dim=dims[0])
    if mask is not None:
        mask = _as_torch(mask, b).bool()
    if op == "count":
        return (mask if mask is not None else torch.ones_like(b, dtype=torch.bool)).sum(dim=dims)
    if op in ("sum", "mean"):
        total = torch.where(mask, b, torch.zeros_like(b)).sum(dim=dims) if mask is not None else b.sum(dim=dims)
        if op == "sum":
            return total
        n = mask.sum(dim=dims) if mask is not None else torch.tensor(float(np.prod([b.shape[d] for d in dims])))
        return total / n
    if op in ("max", "min", "logsumexp"):
        fill = -torch.inf if op != "min" else torch.inf
        x = torch.where(mask, b, torch.full_like(b, fill)) if mask is not None else b
        return {"max": torch.amax, "min": torch.amin, "logsumexp": torch.logsumexp}[op](x, dim=dims)
    if op == "prod":
        return torch.where(mask, b, torch.ones_like(b)).prod() if mask is not None and len(dims) == b.dim() \
            else b.prod(dim=dims[0]) if len(dims) == 1 else b.prod()
    if op in ("var", "sd"):
        v = b.var(dim=dims, unbiased=False)
        return v if op == "var" else v.sqrt()
    raise Uncheckable(op)


def _is_table(x: Any) -> bool:
    return type(x).__name__ in ("DataFrame", "LazyFrame") and hasattr(x, "get_column")


def _named(sig: Sig) -> bool:
    return sig is not None and all(a.known for a in sig) and len({a for a in sig if a != NEW}) == \
        len([a for a in sig if a != NEW])


def _position(sig: tuple[Axis, ...], a: Axis) -> int:
    if a in sig:
        return sig.index(a)
    raise Mismatch(f"no axis {a.name} to combine over, in {[x.name for x in sig]}")


def _broadcast_named(sigs: list[tuple[Axis, ...]]) -> tuple[Axis, ...]:
    out: list[Axis] = []
    for s in sigs:
        for a in s:
            if a != NEW and a not in out:
                out.append(a)
    return tuple(out)


def _fill(shape: tuple[int, ...], sig: tuple[Axis, ...]) -> list[int]:
    it = iter(shape)
    return [1 if x == NEW else next(it) for x in sig]


def align(x: Any, frm: tuple[Axis, ...], to: tuple[Axis, ...]) -> Any:
    """x, whose axes are the index sets `frm` in order, with its axes moved to `to`'s order (an axis `to` has and
    x does not: of length one, to broadcast along)."""
    if frm == to or not frm and not np.ndim(x):
        return x
    a = _arr(x)
    if a.ndim != len(frm):
        raise Mismatch(f"{a.ndim} axes, the signature has {len(frm)} ({[x.name for x in frm]})")
    keep = [k for k, ax in enumerate(frm) if ax != NEW]
    a = a.reshape([a.shape[k] for k in keep])
    have = [frm[k] for k in keep]
    missing = [ax for ax in have if ax not in to]
    if missing:
        raise Mismatch(f"axes {[m.name for m in missing]} are not in the result's {[t.name for t in to]}")
    order = [have.index(ax) for ax in to if ax in have]
    a = _transpose(a, order)
    shape: list[int] = []
    it = iter(a.shape)
    for ax in to:
        shape.append(next(it) if ax in have else 1)
    return a.reshape(shape)


def _what(e: Expr) -> str:
    return type(e).__name__


def _import(qualname: str) -> Any:
    """A function by its qualified name: module.function, or module.Class.method."""
    parts = qualname.split(".")
    for k in range(len(parts) - 1, 0, -1):
        try:
            obj: Any = importlib.import_module(".".join(parts[:k]))
        except ImportError:
            continue
        for p in parts[k:]:
            obj = getattr(obj, p)
        return obj
    raise ImportError(qualname)


__all__ = ["Evaluator", "Uncheckable", "Mismatch", "align", "Dist", "Draw"]
