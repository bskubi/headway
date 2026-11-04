"""The calculus: what clean code means, as terms (docs/core.md §2-§5).

Every expression has a signature (`sig`): the index sets of its axes, in the order the code's arrays have them
(numpy's axis 0 first), or None when not known. An axis is an `Axis`: a named index set (bins, groups), a copy of
one (a covariance's second index), the subset of one where a mask holds, or a positional axis whose index set is not
known (`Axis(None)`). A scalar's signature is (). Unknown is never resolved by a guess (docs/core.md §3).

Terms are immutable and compared by value, so the fold pass and the printer can share and rewrite them freely.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Union


@dataclass(frozen=True)
class Axis:
    """An index set an array's axis ranges over. name None: not known (only its position is). copy: a second
    index over the same set (k'). where: the subset where a mask (over this set) holds. part: a positional part
    (a slice, "lo:hi:step" as code), so that x[:, :k] is over the first k of its columns. letter: the index's
    TeX if fixed already (a spec's), else the printer picks one."""
    name: str | None
    copy: int = 0
    where: Expr | None = None
    part: str | None = None
    letter: str | None = field(default=None, compare=False)

    @property
    def known(self) -> bool:
        return self.name is not None

    def base(self) -> Axis:
        """The whole set this one is part of."""
        return Axis(self.name, self.copy, None, None, self.letter)


NEW = Axis("1")                     # an axis of length one, to broadcast along (x[:, None])
Sig = tuple[Axis, ...] | None


def known(sig: Sig) -> bool:
    """Every axis's index set known (no positional unknowns): the value can be written with its indices."""
    return sig is not None and all(a.known for a in sig)


def broadcast(*sigs: Sig) -> Sig:
    """The signature of an elementwise result: numpy's broadcasting, aligned on the last axis. Two different known
    index sets at one position pair their elements by position, which no named calculus says: that axis is not
    known."""
    if any(s is None for s in sigs):
        return None
    n = max((len(s) for s in sigs if s is not None), default=0)
    out: list[Axis] = []
    for k in range(n):
        here = [s[len(s) - n + k] for s in sigs if s is not None and len(s) - n + k >= 0]
        real = list(dict.fromkeys(a for a in here if a != NEW))
        out.append(real[0] if len(real) == 1 else NEW if not real else Axis(None))
    return tuple(out)


# ---- expressions -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    """What a value is, as far as known: its type ("numpy.ndarray", "float", "list", "record:Fold", "" if not
    known), its signature, and the shapes of its parts (a tuple's items) or of its elements (a list's)."""
    type: str = ""
    sig: Sig = None
    parts: tuple[Shape, ...] | None = None
    elem: Shape | None = None
    dtype: str = ""                     # an array's elements: "bool", "int", "float", "" if not known
    columns: tuple[tuple[str, Shape], ...] | None = None    # a table's columns, if known


UNKNOWN = Shape()
SCALAR = Shape("float", ())


@dataclass(frozen=True)
class Expr:
    sig: Sig = field(default=None, kw_only=True)
    type: str = field(default="", kw_only=True)          # what kind of value: "numpy.ndarray", "float", "list", ...
    parts: tuple[Shape, ...] | None = field(default=None, kw_only=True, compare=False)
    elem: Shape | None = field(default=None, kw_only=True, compare=False)
    dtype: str = field(default="", kw_only=True, compare=False)
    columns: tuple[tuple[str, Shape], ...] | None = field(default=None, kw_only=True, compare=False)

    @property
    def shape(self) -> Shape:
        return Shape(self.type, self.sig, self.parts, self.elem, self.dtype, self.columns)


@dataclass(frozen=True)
class Var(Expr):
    """A variable of the code (or a parameter): its value."""
    name: str


@dataclass(frozen=True)
class Const(Expr):
    value: Any


@dataclass(frozen=True)
class Field(Expr):
    """A record's field (a dataclass's attribute, a settings object's): s.pcs."""
    base: Expr
    attr: str


@dataclass(frozen=True)
class Apply(Expr):
    """A primitive (PRIMITIVES), elementwise unless it says otherwise."""
    fn: str
    args: tuple[Expr, ...]


@dataclass(frozen=True)
class Binder:
    """An index of a sum or a family that is a variable of the code: for x in xs, for i in range(n), for (i, x)
    in enumerate(xs) (var: the names, as the code unpacks them)."""
    var: str | tuple[Any, ...]
    domain: Expr                    # Range, or a collection


@dataclass(frozen=True)
class Pos:
    """An axis of a value whose index sets are not known, by its position (numpy's axis 0)."""
    k: int


@dataclass(frozen=True)
class Range(Expr):
    lo: Expr
    hi: Expr                        # exclusive, as Python's


Index = Union[Axis, Binder, Pos]


@dataclass(frozen=True)
class Aggregate(Expr):
    """Combine body over indices, where `where` holds: sum, prod, mean, max, min, ... (AGGREGATES)."""
    op: str
    over: tuple[Index, ...]
    body: Expr
    where: Expr | None = None


@dataclass(frozen=True)
class Build(Expr):
    """The family whose element at each index is body (where `where` holds, if given: a comprehension's if): a
    comprehension, a loop filling an array."""
    body: Expr
    over: tuple[Index, ...]
    where: Expr | None = None


@dataclass(frozen=True)
class Table(Expr):
    """A table: named columns over one index set of rows (the signature's). base: a table whose columns it keeps
    (with_columns), or None (select, a group_by's agg). groups: the key columns (over base's rows) a group_by
    groups by, if the rows are its groups: each row one combination of their values, Key(rows) its values."""
    base: Expr | None
    cols: tuple[tuple[str, Expr], ...]
    groups: tuple[tuple[str, Expr], ...] = ()


@dataclass(frozen=True)
class Key(Expr):
    """The value of a group's key: what identifies the group k (unit_k), the rows of whose key column equal it."""
    axis: Axis
    name: str


@dataclass(frozen=True)
class Extent(Expr):
    """The size of an index set: len(x), x.shape[1]. axis: the set's Axis, or Pos(k) of `of` if not known."""
    axis: Index
    of: Expr


@dataclass(frozen=True)
class Idx(Expr):
    """An index, as a value: its position along its axis (np.arange(len(x)), the k of a family)."""
    axis: Axis


@dataclass(frozen=True)
class Take(Expr):
    """base at the positions arrays of positions give, along some of its axes (numpy's advanced indexing):
    x[:, module] (a pointer: that axis replaced by the pointer's own axes), x[3] (a number: the axis dropped),
    v[rows, cols] (pointers together: their axes in place of both). The result's axes are the signature's; `at`
    says where the pointers' axes are in it (numpy puts them first when the pointers are apart)."""
    base: Expr
    pos: tuple[int, ...]
    index: tuple[Expr, ...]
    at: int = 0


@dataclass(frozen=True)
class Restrict(Expr):
    """base on the part of one axis where a mask holds (x[learns]), or a positional part of it (x[:, :k])."""
    base: Expr
    pos: int
    mask: Expr | None = None
    part: tuple[Expr | None, Expr | None, Expr | None] | None = None


@dataclass(frozen=True)
class Align(Expr):
    """base with its axes moved (a transpose), or with axes of length one put in to broadcast along (x[:, None]),
    or taken out (squeeze): the same values. how: "transpose", "expand", "squeeze"."""
    base: Expr
    how: str


@dataclass(frozen=True)
class Cases(Expr):
    branches: tuple[tuple[Expr, Expr], ...]      # (condition, value)
    otherwise: Expr


@dataclass(frozen=True)
class Lambda(Expr):
    """One of the code's functions as a value (a lambda, a function defined inside another): its body, in its
    parameters (each a Binder, its domain the value the function is first given)."""
    over: tuple[Binder, ...]
    body: Expr
    name: str = ""                  # the def's name, "" for a lambda


@dataclass(frozen=True)
class Optimum(Expr):
    """Where a function is least (op "min") or greatest ("max"), as an optimizer finds it from a start:
    scipy.optimize.minimize(fun, x0, jac=, method=, bounds=). Its value is the optimizer's result: its x (the
    point), its fun (the value there), its success. fun and jac: Lambdas, or Ops naming functions not read."""
    op: str
    fun: Expr
    start: Expr
    method: Expr | None = None
    bounds: Expr | None = None
    jac: Expr | None = None


OPTIMUM = "scipy.optimize.OptimizeResult"


@dataclass(frozen=True)
class Fit(Expr):
    """A regression fitted by maximum likelihood (or least squares): statsmodels' OLS, WLS, GLM, Logit, Poisson.
    Its value is the fit (its params the coefficients where `objective` is greatest, op "max", or least, "min").
    objective: what is optimized, a function of the coefficients (over X's columns), for the reader; the family,
    link, y, X, weights and offset say the same thing in a form the evaluator fits by (its own IRLS or least squares,
    not the library's). call: the code's model, for what the fit has that the calculus does not say (bse, ...)."""
    op: str                             # "max" or "min"
    family: str                         # "gaussian", "poisson", "binomial", "negative_binomial", "gamma"
    link: str                           # "identity", "log", "logit", "inverse_power"
    y: Expr
    X: Expr
    objective: Expr                     # a Lambda: the coefficients (bound, so no variable of the code is them)
    weights: Expr | None = None
    offset: Expr | None = None
    alpha: Expr | None = None           # the negative binomial's dispersion
    call: Expr | None = None            # the model as the code makes it (an Op), and how it is fitted
    fit_kwargs: tuple[tuple[str, Expr], ...] = ()
    kind: str = "glm"                   # "ols", "wls", "glm", "logit", "poisson": the model the code names


FIT = "statsmodels.results"


@dataclass(frozen=True)
class Op(Expr):
    """A function the calculus does not open: shown by its name, applied to its arguments; its meaning is the
    code's (qualname: what it is, to look it up and to call it, "methembed.core.embed.kmeans"). local: one of the
    code's own functions (it can be defined under "where")."""
    name: str
    args: tuple[Expr, ...]
    kwargs: tuple[tuple[str, Expr], ...] = ()
    qualname: str = ""
    local: bool = False
    known: bool = field(default=False, compare=False)    # a table's on purpose (eigh, a draw's generator, ∇)


@dataclass(frozen=True)
class Item(Expr):
    """base[key], as the code has it, when what it means is not known: a dict's entry, a position from the end."""
    base: Expr
    key: Expr


@dataclass(frozen=True)
class Tup(Expr):
    """A tuple, a list or a set written out: (a, b), [a, b]. kind: "tuple", "list", "set"."""
    items: tuple[Expr, ...]
    kind: str = "tuple"


@dataclass(frozen=True)
class Mapping(Expr):
    """A dict written out ({k: v}), or built ({k: v for x in xs}: over)."""
    pairs: tuple[tuple[Expr, Expr], ...]
    over: tuple[Index, ...] = ()


@dataclass(frozen=True)
class SliceKey(Expr):
    """a:b:step, inside x[...]."""
    lo: Expr | None
    hi: Expr | None
    step: Expr | None


@dataclass(frozen=True)
class Text(Expr):
    """A string (an f-string: its text not shown)."""
    value: str | None


# randomness (docs/core.md §5)

@dataclass(frozen=True)
class Dist(Expr):
    """A distribution of a named family (DISTRIBUTIONS), its parameters by name."""
    family: str
    params: tuple[tuple[str, Expr], ...]


@dataclass(frozen=True)
class Draw(Expr):
    """A sample from a distribution: independent across its parameters' indices (and `size`'s)."""
    dist: Dist
    rng: Expr | None = None             # the generator it comes from (recorded, not written)
    size: Expr | None = None            # the code's size, as written: the shape drawn, whether or not its axes are known


@dataclass(frozen=True)
class Density(Expr):
    """log p(x | the distribution's parameters), or p itself."""
    dist: Dist
    at: Expr
    log: bool = True


# ---- statements --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Stmt:
    line: int = field(default=0, kw_only=True)            # the line of the code it is (in the file)


@dataclass(frozen=True)
class Let(Stmt):
    """Names a value (or several: a tuple's items). again: the variable had a value before (x ←)."""
    targets: tuple[str, ...]
    value: Expr
    again: bool = False
    inplace: bool = False               # x += e, f(x, out=x): the value x holds changed, not replaced


@dataclass(frozen=True)
class SetPart(Stmt):
    """A part of a variable's value changed in place (x[c] = e): code that is not clean, written as it is.
    shown: the part, as the math writes it (x_{ik}|_{k ∈ b}), if it can say it."""
    target: Expr
    value: Expr
    shown: Expr | None = None


@dataclass(frozen=True)
class Effect(Stmt):
    """A call whose result is not used: it may change what it is given (a.sort())."""
    value: Expr


@dataclass(frozen=True)
class For(Stmt):
    """Statements repeated: for each index (var) of a domain, or `count` times (var None)."""
    var: tuple[str, ...] | None         # the loop's variables (for i, x in ...), None for _ (repeat)
    domain: Expr
    body: tuple[Stmt, ...]
    orelse: tuple[Stmt, ...] = ()       # done if the loop ends without a break


@dataclass(frozen=True)
class While(Stmt):
    test: Expr
    body: tuple[Stmt, ...]
    orelse: tuple[Stmt, ...] = ()


@dataclass(frozen=True)
class If(Stmt):
    test: Expr
    body: tuple[Stmt, ...]
    orelse: tuple[Stmt, ...]


@dataclass(frozen=True)
class Assume(Stmt):
    """An assert: the result is claimed only where it holds ("provided that")."""
    test: Expr


@dataclass(frozen=True)
class Return(Stmt):
    value: Expr | None


@dataclass(frozen=True)
class Jump(Stmt):
    kind: str                           # "break", "continue"


@dataclass(frozen=True)
class Unwritten(Stmt):
    """A statement the calculus cannot say: named by its line, never guessed at."""


@dataclass
class Program:
    """A function (or a block) as terms: its parameters (name, signature, type), its statements."""
    name: str | None
    params: list[tuple[str, Sig, str]]
    body: list[Stmt]
    qualname: str = ""


# ---- the primitives: their names, how many arguments ------------------------------------------------------------

PRIMITIVES = {
    # arithmetic
    "add": 2, "sub": 2, "mul": 2, "div": 2, "pow": 2, "neg": 1, "mod": 2, "floordiv": 2,
    # elementwise functions
    "log": 1, "exp": 1, "sqrt": 1, "abs": 1, "log1p": 1, "expm1": 1, "log2": 1, "log10": 1, "sigmoid": 1,
    "logit": 1, "tanh": 1, "sinh": 1, "cosh": 1, "sin": 1, "cos": 1, "tan": 1, "arcsin": 1, "arccos": 1,
    "arctan": 1, "floor": 1, "ceil": 1, "round": 1, "sign": 1, "erf": 1, "gammaln": 1, "digamma": 1,
    "gamma": 1, "max2": 2, "min2": 2, "isnan": 1, "isfinite": 1, "isinf": 1, "trigamma": 1, "relu": 1, "softplus": 1, "square": 1,
    # comparisons and logic
    "eq": 2, "ne": 2, "lt": 2, "le": 2, "gt": 2, "ge": 2, "and": 2, "or": 2, "not": 1, "in": 2, "notin": 2,
    "is": 2, "isnot": 2,
    # linear algebra on matrices (the last two axes); matmul only where the indices are not known
    "matmul": 2, "inv": 1, "det": 1, "trace": 1, "solve": 2, "norm": 1, "transpose": 1,
    # conversions that change a value's kind, not what it is
    "float": 1, "int": 1, "len": 1, "size": 1,
}

AGGREGATES = {"sum", "prod", "mean", "max", "min", "any", "all", "var", "sd", "median", "logsumexp",
              "argmax", "argmin", "count"}

DISTRIBUTIONS = {"normal", "binomial", "bernoulli", "poisson", "gamma", "beta", "dirichlet", "categorical",
                 "multinomial", "uniform", "exponential", "negative_binomial", "multivariate_normal", "lognormal",
                 "student_t", "laplace", "cauchy", "geometric", "betabinomial", "integers", "choice", "permutation"}


def shaped(e: Expr, s: Shape) -> Expr:
    """e, as being of shape s."""
    return replace(e, sig=s.sig, type=s.type, parts=s.parts, elem=s.elem, dtype=s.dtype, columns=s.columns)


def with_sig(e: Expr, sig: Sig, type: str | None = None) -> Expr:
    return replace(e, sig=sig, type=e.type if type is None else type)


# ---- traversal ---------------------------------------------------------------------------------------------------

def resig(e: Expr, axes: dict[Axis, Axis], names: dict[str, Expr] | None = None) -> Expr:
    """e with its index sets moved (each signature's, an aggregate's, an index's) and its variables replaced: an
    expression of a row (a polars expression) put in a table's rows."""
    names = names or {}
    if isinstance(e, Var) and e.name in names:
        return names[e.name]

    def ax(a: Any) -> Any:
        return axes.get(a, a) if isinstance(a, Axis) else a
    changes: dict[str, Any] = {}
    if e.sig is not None:
        changes["sig"] = tuple(ax(a) for a in e.sig)
    for name in e.__dataclass_fields__:
        if name in ("sig", "type", "parts", "elem", "dtype"):
            continue
        v = getattr(e, name)
        if isinstance(v, Expr):
            changes[name] = resig(v, axes, names)
        elif isinstance(v, Axis):
            changes[name] = ax(v)
        elif isinstance(v, tuple):
            changes[name] = tuple(_resig_item(x, axes, names) for x in v)
    return replace(e, **changes)


def _resig_item(x: Any, axes: dict[Axis, Axis], names: dict[str, Expr]) -> Any:
    if isinstance(x, Expr):
        return resig(x, axes, names)
    if isinstance(x, Axis):
        return axes.get(x, x)
    if isinstance(x, Binder):
        return Binder(x.var, resig(x.domain, axes, names))
    if isinstance(x, tuple):
        return tuple(_resig_item(i, axes, names) for i in x)
    return x

def _map(x: Any, f: Any) -> Any:
    if isinstance(x, Expr):
        return f(x)
    if isinstance(x, Binder):
        return Binder(x.var, f(x.domain))
    if isinstance(x, tuple):
        return tuple(_map(i, f) for i in x)
    return x


def map_children(e: Expr, f: Any) -> Expr:
    """e with f applied to each of its subexpressions (not inside signatures)."""
    changes = {}
    for name in e.__dataclass_fields__:
        if name in ("sig", "type", "parts", "elem", "dtype"):
            continue
        v = getattr(e, name)
        nv = _map(v, f)
        if nv is not v:
            changes[name] = nv
    return replace(e, **changes) if changes else e


def children(e: Expr) -> list[Expr]:
    out: list[Expr] = []

    def take(x: Expr) -> Expr:
        out.append(x)
        return x
    map_children(e, take)
    return out


def walk(e: Expr) -> list[Expr]:
    """e and all its subexpressions."""
    out = [e]
    for c in children(e):
        out += walk(c)
    return out


def bound(e: Expr) -> set[str]:
    """The variables an expression binds for its body (a sum's or a family's over a loop variable)."""
    over = getattr(e, "over", ())
    names: set[str] = set()
    for i in over:
        if isinstance(i, Binder):
            names |= set(_flat(i.var))
    return names


def _flat(v: Any) -> list[str]:
    return [v] if isinstance(v, str) else [n for x in v for n in _flat(x)]


def free_vars(e: Expr) -> list[str]:
    """The variables an expression reads (each time it reads them), not those it binds itself."""
    if isinstance(e, Var):
        return [e.name]
    inner = bound(e)
    out: list[str] = []
    for c in children(e):
        out += [n for n in free_vars(c) if n not in inner]
    return out


def stmt_exprs(st: Stmt) -> list[Expr]:
    if isinstance(st, Let):
        return [st.value]
    if isinstance(st, (SetPart,)):
        return [st.target, st.value]
    if isinstance(st, (Effect,)):
        return [st.value]
    if isinstance(st, For):
        return [st.domain]
    if isinstance(st, (While, If)):
        return [st.test]
    if isinstance(st, Assume):
        return [st.test]
    if isinstance(st, Return):
        return [st.value] if st.value is not None else []
    return []


def stmt_body(st: Stmt) -> list[Stmt]:
    if isinstance(st, (For, While)):
        return [*st.body, *st.orelse]
    if isinstance(st, If):
        return [*st.body, *st.orelse]
    return []


def walk_stmts(body: list[Stmt] | tuple[Stmt, ...]) -> list[Stmt]:
    out: list[Stmt] = []
    for st in body:
        out.append(st)
        out += walk_stmts(stmt_body(st))
    return out
