r"""Terms as KaTeX: the one place notation is decided (docs/core.md §8).

  symbols     a name the spec gives keeps its TeX; the code's other names are written as themselves (x, \mathit{xs},
              \mu for mu), or with `short`, as short symbols (the first free of each name's near neighbours)
  indices     a value whose index sets are all known is written with its indices (x_{cg}); an index set's letter
              is the spec's for it, else its name's first letter if free, else the first free of i, j, k, ...; a
              copy primed (g')
  products    of values written with their indices, or with a number, juxtaposed (n_{jg} q_{jg}); of arrays whose
              indices are not known, elementwise (⊙), as numpy's * is
  sums        over their indices (\sum_g), over positions where the indices are not known (\sum_{\text{axis } 0})
  statements  x = e; x ← e (a value again); cases; loops as an algorithm is written ("repeat n times:"); asserts
              as "provided that"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from headway.core.terms import (NEW, Aggregate, Lambda, Optimum, Align, Apply, Assume, Axis, Binder, Build, Cases, Const, Density,
                                Dist, Draw, Effect, Expr, Extent, Field, For, Idx, If, Item, Jump, Key, Let, Mapping,
                                Op, Pos, Range, Restrict, Return, SetPart, Sig, SliceKey, Stmt, Table, Take, Text, Tup,
                                Unwritten, Var, While, free_vars, known, walk_stmts)

ATOM, POWER, UNARY, PRODUCT, SUM, COMPARE, NOT, AND, OR = 13, 12, 11, 10, 9, 4, 3, 2, 1
ASSUMING = r"&\text{provided that } "               # a precondition's line (an assert)
INDEX_LETTERS = "ijklmnpqrstuvxyzabcdefgh"           # an axis's index when its first letter is taken
GREEK = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu",
         "xi", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega", "Gamma", "Delta", "Theta",
         "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega"}
GREEK_OF = {"a": "alpha", "b": "beta", "c": "chi", "d": "delta", "f": "phi", "g": "gamma", "h": "eta", "i": "iota",
            "k": "kappa", "l": "lambda", "m": "mu", "n": "nu", "o": "omega", "p": "pi", "q": "theta", "r": "rho",
            "s": "sigma", "t": "tau", "u": "upsilon", "v": "nu", "w": "omega", "x": "xi", "y": "psi", "z": "zeta"}
CAPITAL_GREEK = {"gamma", "delta", "theta", "lambda", "xi", "pi", "sigma", "upsilon", "phi", "psi", "omega"}
NOT_SYMBOLS = {"e"}                                 # read as Euler's number
GREEK_WORDS = {g for g in GREEK if g.islower()}
UNARY_TEX = {"log": r"\log", "exp": r"\exp", "sin": r"\sin", "cos": r"\cos", "tan": r"\tan", "tanh": r"\tanh",
             "sinh": r"\sinh", "cosh": r"\cosh", "arcsin": r"\arcsin", "arccos": r"\arccos", "arctan": r"\arctan",
             "sign": r"\operatorname{sign}", "det": r"\det", "trace": r"\operatorname{tr}",
             "isnan": r"\operatorname{isnan}", "isfinite": r"\operatorname{isfinite}", "isinf": r"\operatorname{isinf}",
             "trigamma": r"\psi_1", "round": r"\operatorname{round}",
             "sigmoid": r"\sigma", "logit": r"\operatorname{logit}", "erf": r"\operatorname{erf}",
             "gammaln": r"\log\Gamma", "digamma": r"\psi", "gamma": r"\Gamma", "relu": r"\operatorname{relu}",
             "int": r"\operatorname{int}", "len": r"\operatorname{len}", "size": r"\operatorname{size}"}
AGGREGATE_TEX = {"sum": r"\sum", "prod": r"\prod", "mean": r"\operatorname{mean}", "max": r"\max", "min": r"\min",
                 "sd": r"\operatorname{sd}", "var": r"\operatorname{var}", "median": r"\operatorname{median}",
                 "any": r"\operatorname{any}", "all": r"\operatorname{all}", "logsumexp": r"\operatorname{logsumexp}",
                 "argmax": r"\operatorname{argmax}", "argmin": r"\operatorname{argmin}", "count": r"\#"}
COMPARE_TEX = {"eq": "=", "ne": r"\neq", "lt": "<", "le": r"\le", "gt": ">", "ge": r"\ge", "in": r"\in",
               "notin": r"\notin", "is": "=", "isnot": r"\neq"}
DIST_TEX = {"normal": r"\mathcal{N}", "binomial": r"\operatorname{Binomial}", "bernoulli": r"\operatorname{Bernoulli}",
            "poisson": r"\operatorname{Poisson}", "gamma": r"\operatorname{Gamma}", "beta": r"\operatorname{Beta}",
            "dirichlet": r"\operatorname{Dirichlet}", "categorical": r"\operatorname{Categorical}",
            "multinomial": r"\operatorname{Multinomial}", "uniform": r"\operatorname{Uniform}",
            "exponential": r"\operatorname{Exponential}", "negative_binomial": r"\operatorname{NegBin}",
            "lognormal": r"\operatorname{LogNormal}", "student_t": r"t", "laplace": r"\operatorname{Laplace}",
            "cauchy": r"\operatorname{Cauchy}", "geometric": r"\operatorname{Geometric}",
            "betabinomial": r"\operatorname{BetaBinomial}", "multivariate_normal": r"\mathcal{N}",
            "nbinom": r"\operatorname{NB}"}
MADE = {"numpy.zeros": "0", "numpy.ones": "1", "numpy.zeros_like": "0", "numpy.ones_like": "1"}
TABLE_TYPES = ("polars.DataFrame", "pandas.DataFrame")
VALUES_OF_FIRST = {"numpy.reshape", "numpy.ravel", "reshape", "ravel", "flatten"}


@dataclass
class Name:
    """What a code name is written as: TeX, and the indices of its axes if it is a symbol with them. per: the
    symbol is a family, one value per element of an index set (the spec's per=): the code's name is a list."""
    tex: str
    dims: tuple[str | None, ...] | None = None
    per: str | None = None


@dataclass
class T:
    """An expression, written: its TeX and how tightly it binds (for parentheses)."""
    tex: str
    prec: int = ATOM


@dataclass
class Printer:
    names: dict[str, Name] = field(default_factory=dict)
    short: bool = False
    indices: dict[str, str] = field(default_factory=dict)      # index set name -> its letter (TeX)
    taken: set[str] = field(default_factory=set)                # symbols written already
    reserved: set[str] = field(default_factory=set)             # one-letter names of the code (not index letters)
    defined: set[str] = field(default_factory=set)
    given: dict[str, Name] = field(default_factory=dict)        # a parameter renamed after a new value: as given
    letters: dict[tuple[str | None, int], str] = field(default_factory=dict)
    relabel: dict[Axis, Axis] = field(default_factory=dict)
    outer: set[Axis] = field(default_factory=set)              # the indices free in the statement being written
    aliases: dict[str, str] = field(default_factory=dict)       # x = y, x another name for y
    bare: bool = False                                          # matrix notation: symbols without indices

    def __post_init__(self) -> None:
        self.taken |= {n.tex for n in self.names.values()}
        self.fixed = set(self.names)                    # the names given: written as given
        self.fixed_letters: set[str] = set(self.indices.values())     # the spec's index letters

    # ---- symbols ----------------------------------------------------------------------------------------------

    def symbol(self, n: str) -> str:
        """A code name's base symbol (without indices)."""
        n = self.aliases.get(n, n)
        if n in self.names:
            return self.names[n].tex
        if not self.short:
            return ident(n)
        tex = self.condense(n)
        self.names[n] = Name(tex)
        return tex

    def free(self, t: str) -> bool:
        """Whether a symbol is not written already, nor the start of one (x is not free beside x_{ig}), nor an
        index's letter."""
        return t not in self.letters.values() and t not in self.fixed_letters and \
            not any(x == t or x.startswith((t + "_", t + "^", t + "'")) for x in self.taken)

    def prepare(self, axes: list[Axis]) -> None:
        """The letters of the index sets a program has, chosen before any symbol (so that no short symbol takes
        one)."""
        for a in axes:
            if a.known and a != NEW and a.letter is None:
                self.letter(Axis(a.name))
            elif a.letter is not None:
                self.fixed_letters.add(a.letter)

    def condense(self, n: str) -> str:
        """A short symbol for a code name: the first of its neighbours that is free."""
        tex = next(t for t in neighbours(n.replace(".", "_"), False) if self.free(t))
        self.taken.add(tex)
        return tex

    def letter(self, a: Axis) -> str | None:
        """An index set's letter: the spec's, else its name's first letter if no other index or one-letter name
        of the code has it, else the first such of i, j, k, ...; a copy primed."""
        seen = {a}                                      # renames compose (a product's copy of g, renamed again by a
        while a in self.relabel:                        # diagonal): followed to the last
            a = self.relabel[a]
            if a in seen:
                break
            seen.add(a)
        if a == NEW or not a.known:
            return None
        if a.letter is not None:
            base = a.letter
        else:
            key = (a.name, 0)
            if key not in self.letters:
                assert a.name is not None
                if a.name in self.indices:
                    self.letters[key] = self.indices[a.name]
                else:
                    used = set(self.letters.values()) | self.reserved | {t for t in self.taken if len(t) == 1}
                    first = a.name[:1].lower()
                    pick = next((c for c in [first, *INDEX_LETTERS] if c.isascii() and c.isalpha() and c not in used),
                                None)
                    self.letters[key] = pick or r"\text{" + text(a.name) + "}"
            base = self.letters[key]
        return base + "'" * a.copy

    def index_tex(self, sig: Sig) -> str | None:
        """The indices of a value of this signature, if all are known (an axis of length one left out)."""
        if not known(sig) or not sig:
            return None
        letters = [self.letter(a) for a in sig if a != NEW]
        if any(x is None for x in letters):
            return None
        return "".join(x for x in letters if x)

    def var(self, e: Var) -> T:
        n = self.aliases.get(e.name, e.name)
        if self.bare:                                   # matrix notation: the symbol alone
            name = self.names.get(n)
            if name is not None and name.dims is not None:
                return T(_strip_indices(name.tex, name.dims))
            return T(self.symbol(n))
        if n in self.fixed and (self.names[n].dims is None or not self.relabel):
            return T(self.names[n].tex)                 # a symbol (with its indices) as the spec writes it
        base = self.symbol(n)
        dims = self.names[n].dims if n in self.names else None
        if dims is not None:
            base = _strip_indices(self.names[n].tex, dims)
        idx = self.index_tex(e.sig)
        return T(indexed(base, idx) if idx else base)

    # ---- expressions -----------------------------------------------------------------------------------------

    def expr(self, e: Expr) -> T:
        method = getattr(self, "e_" + type(e).__name__, None)
        if method is None:
            return T(r"\text{(not written)}")
        return method(e)  # type: ignore[no-any-return]

    def e_Var(self, e: Var) -> T:
        return self.var(e)

    def e_Const(self, e: Const) -> T:
        return constant(e.value)

    def e_Text(self, e: Text) -> T:
        return T(r"\text{(text)}") if e.value is None else T(r"\text{“" + text(e.value) + "”}")

    def e_Field(self, e: Field) -> T:
        if isinstance(e.base, Restrict) and e.base.base.type in TABLE_TYPES and e.base.mask is not None:
            # a column of a table's rows where a mask holds: the column, restricted (depth_d|_{learns_d})
            whole = self.e_Field(Field(e.base.base, e.attr, sig=e.base.base.sig, type=e.type))
            return T(r"\left." + whole.tex + r"\right|_{" + self.expr(e.base.mask).tex + "}")
        if e.attr in ("weight", "bias") and e.base.type == "torch.nn.Linear":
            # a layer's weight and bias: W and b, the layer named above (W^{layer1}_{hf})
            letter = "W" if e.attr == "weight" else "b"
            layer = _dotted(e.base) or ""
            sup = r"^{(\mathit{" + text(layer.removeprefix("self.")) + "})}"
            idx = self.index_tex(e.sig)
            return T(letter + sup + ("_{" + idx + "}" if idx else ""))
        if e.base.type in TABLE_TYPES:
            # a table's column: the column's name, indexed by the table's rows (learns_d)
            n = f"{_dotted(e.base) or ''}.{e.attr}"
            base = self.names[n].tex if n in self.names else ident(e.attr)
            idx = self.index_tex(e.sig)
            return T(indexed(base, idx) if idx and not scripted(base) or idx else base)
        dotted = _dotted(e)
        if dotted is not None and dotted in self.names:
            return T(self.names[dotted].tex)
        if dotted is not None:
            base = self.symbol(dotted)
            idx = self.index_tex(e.sig)
            return T(indexed(base, idx) if idx else base)
        return T(base_tex(self.expr(e.base)) + r".\mathit{" + text(e.attr) + "}")

    def e_Apply(self, e: Apply) -> T:
        fn, args = e.fn, e.args
        if fn in ("add", "sub", "mul", "div", "pow", "mod", "floordiv", "and", "or", "matmul"):
            return self.binary(fn, args[0], args[1])
        if fn in ("lshift", "rshift"):                  # a << k: a times 2^k (>>: divided, rounded down)
            shifted, by = self.expr(args[0]), self.expr(args[1])
            if fn == "lshift":
                two = T("2^{" + by.tex + "}", POWER)
                if isinstance(args[0], Const) and args[0].value == 1:
                    return two
                return T(wrap(shifted, PRODUCT) + r"\cdot " + two.tex, PRODUCT)
            return T(r"\left\lfloor\frac{" + shifted.tex + "}{2^{" + by.tex + r"}}\right\rfloor")
        if fn == "eq" and all(isinstance(a, Idx) for a in args):            # the identity: δ_{cc'}
            return T(r"\delta_{" + "".join(self.letter(a.axis) or "" for a in args if isinstance(a, Idx)) + "}")
        if fn in ("inv", "solve") and (m := self.matrix(e)) is not None:
            return m
        if fn in COMPARE_TEX:
            left, right = self.expr(args[0]), self.expr(args[1])
            return T(f"{wrap(left, COMPARE + 1)} {COMPARE_TEX[fn]} {wrap(right, COMPARE + 1)}", COMPARE)
        a = [self.expr(x) for x in args]
        if fn == "neg":
            return T("-" + wrap(a[0], UNARY), UNARY)
        if fn == "not":
            return T(r"\lnot " + wrap(a[0], UNARY), NOT)
        if fn in ("max2", "min2"):
            parts = self._flat_minmax(e)
            return T((r"\max" if fn == "max2" else r"\min") + r"\left(" + ", ".join(p.tex for p in parts)
                     + r"\right)")
        if fn == "sqrt":
            return T(r"\sqrt{" + a[0].tex + "}")
        if fn == "abs":
            return T(r"\left|" + a[0].tex + r"\right|")
        if fn == "square":
            return T(base_tex(a[0]) + "^{2}", POWER)
        if fn == "log1p":
            return T(r"\log\left(1 + " + a[0].tex + r"\right)")
        if fn == "expm1":
            return T(r"\left(e^{" + a[0].tex + r"} - 1\right)")
        if fn in ("log2", "log10"):
            return T(r"\log_{" + fn[3:] + r"}\left(" + a[0].tex + r"\right)")
        if fn in ("floor", "ceil"):
            return T(rf"\left\l{fn}" + a[0].tex + rf"\right\r{fn}")
        if fn == "softplus":
            return T(r"\log\left(1 + e^{" + a[0].tex + r"}\right)")
        if fn == "float":
            return a[0]
        if fn == "inv":
            return T(base_tex(a[0]) + "^{-1}", POWER)
        if fn == "solve":
            return T(base_tex(a[0]) + r"^{-1}\," + wrap(a[1], PRODUCT + 1), PRODUCT)
        if fn == "norm":
            return T(r"\left\lVert " + a[0].tex + r"\right\rVert")
        if fn == "transpose":
            return T(base_tex(a[0]) + r"^\top", POWER)
        if fn == "outer":
            return T(wrap(a[0], PRODUCT) + r"\," + base_tex(a[1]) + r"^\top", PRODUCT)
        if fn in UNARY_TEX:
            return T(UNARY_TEX[fn] + r"\left(" + ", ".join(x.tex for x in a) + r"\right)")
        return T(operator(fn) + r"\left(" + ", ".join(x.tex for x in a) + r"\right)")

    # ---- matrix notation ------------------------------------------------------------------------------------------

    def matrix_form(self, e: Expr) -> bool:
        """Whether a statement's value is linear algebra best written in matrix notation: an inverse, a solve."""
        while isinstance(e, Align) and e.how == "rename":
            e = e.base
        return isinstance(e, Apply) and e.fn in ("inv", "solve") and self.matrix(e) is not None

    def matrix(self, e: Expr) -> T | None:
        """e in matrix notation (vectors and matrices by their symbols alone, products juxtaposed, ⊤ for a
        transpose, diag(w) for an elementwise scaling of rows, I for the identity), if it is linear algebra of
        values of at most two axes; None if it is not."""
        got = self._m(e)
        return None if got is None else got[0]

    def _m(self, e: Expr) -> tuple[T, tuple[Axis | None, Axis | None]] | None:
        """(the TeX, (its rows' index set, its columns')): a column vector has no columns, a row vector no rows."""
        if isinstance(e, Align) and e.how in ("expand", "squeeze"):     # w[:, None]: w, to scale rows by
            inner = self._m(e.base)
            return inner
        sig = e.sig if e.sig is None else tuple(a for a in e.sig if a != NEW)
        if sig is None or len(sig) > 2 or not known(sig):
            return None
        if isinstance(e, Const) and sig == ():
            return constant(e.value), (None, None)
        if isinstance(e, Apply) and e.fn == "eq" and all(isinstance(a, Idx) for a in e.args) and len(sig) == 2:
            return T("I"), (sig[0], sig[1])
        if isinstance(e, Align) and e.how == "transpose":
            inner = self._m(e.base)
            if inner is None:
                return None
            t, (r, c) = inner
            return T(base_tex(t) + r"^\top", POWER), (c, r)
        if isinstance(e, Align) and e.how == "rename":
            inner = self._m(e.base)
            if inner is None or e.base.sig is None:
                return None
            t, (r, c) = inner
            moved = dict(zip(e.base.sig, sig))
            return t, (moved.get(r, r) if r else None, moved.get(c, c) if c else None)
        if isinstance(e, Apply) and e.fn in ("inv", "solve"):
            a = self._m(e.args[0])
            if a is None:
                return None
            ta, (ra, ca) = a
            inv = T(base_tex(ta) + "^{-1}", POWER)
            if e.fn == "inv":
                return inv, (ca, ra)
            b = self._m(e.args[1])
            if b is None:
                return None
            tb, (rb, cb) = b
            return T(wrap(inv, PRODUCT) + r"\," + wrap(tb, PRODUCT + 1), PRODUCT), (ca, cb)
        if isinstance(e, Aggregate) and e.op == "sum" and len(e.over) == 1 and isinstance(e.over[0], Axis) \
                and e.where is None and isinstance(e.body, Apply) and e.body.fn == "mul":
            k = e.over[0]
            x, y = e.body.args
            mx, my = self._m(x), self._m(y)
            if mx is None or my is None:
                return None
            (tx, (rx, cx)), (ty, (ry, cy)) = mx, my
            if cx != k:                                 # the left factor's columns are what is summed over
                if rx != k:
                    return None
                tx, rx, cx = T(base_tex(tx) + r"^\top", POWER), cx, rx
            if ry != k:                                 # the right factor's rows
                if cy != k:
                    return None
                ty, ry, cy = T(base_tex(ty) + r"^\top", POWER), cy, ry
            return T(wrap(tx, PRODUCT) + r"\," + wrap(ty, PRODUCT), PRODUCT), (rx, cy)     # (products associate)
        if isinstance(e, Apply) and e.fn == "mul" and len(e.args) == 2:
            x, y = e.args
            if x.sig == () or y.sig == ():              # a number times a matrix
                s, m = (x, y) if x.sig == () else (y, x)
                mm = self._m(m)
                if mm is None:
                    return None
                tm, shape = mm
                return T(wrap(self.expr(s), PRODUCT) + r"\," + wrap(tm, PRODUCT + 1), PRODUCT), shape
            xs, ys = _core(x.sig), _core(y.sig)
            if xs is not None and ys is not None and {len(xs), len(ys)} == {1, 2}:
                v, mat, vs = (x, y, xs) if len(xs) == 1 else (y, x, ys)
                mm = self._m(mat)
                if mm is None:
                    return None
                tm, (r, c) = mm
                saved, self.bare = self.bare, True
                try:
                    tv = self.expr(v.base if isinstance(v, Align) and v.how == "expand" else v)
                finally:
                    self.bare = saved
                d = T(r"\operatorname{diag}\left(" + tv.tex + r"\right)")
                if vs[0] == r:
                    return T(d.tex + r"\," + wrap(tm, PRODUCT), PRODUCT), (r, c)
                if vs[0] == c:
                    return T(wrap(tm, PRODUCT) + r"\," + d.tex, PRODUCT), (r, c)
                return None
        if isinstance(e, Apply) and e.fn in ("add", "sub") and len(sig) == 2:
            a, b = self._m(e.args[0]), self._m(e.args[1])
            if a is None or b is None or a[1] != b[1] and b[1] != (None, None):
                return None
            op = " + " if e.fn == "add" else " - "
            return T(wrap(a[0], SUM) + op + wrap(b[0], SUM + (e.fn == "sub")), SUM), a[1]
        if len(sig) <= 1 and not any(isinstance(x, (Aggregate, Take, Restrict)) for x in _walk(e)):
            # a vector made elementwise (log(y / s + 0.1)): written so, its index left out
            saved, self.bare = self.bare, True
            try:
                t = self.expr(e)
            finally:
                self.bare = saved
            # parenthesized in a matrix product unless one term (Xᵀ (w ⊙ z), not Xᵀ w ⊙ z)
            return (t if t.prec >= ATOM else T(t.tex, SUM)), ((sig[0] if sig else None), None)
        if isinstance(e, Var) and len(sig) == 2:
            saved, self.bare = self.bare, True
            try:
                t = self.expr(e)
            finally:
                self.bare = saved
            return t, (sig[0], sig[1])
        return None

    def _flat_minmax(self, e: Expr) -> list[T]:
        if isinstance(e, Apply) and e.fn in ("max2", "min2"):
            first = e.args[0]
            if isinstance(first, Apply) and first.fn == e.fn:
                return [*self._flat_minmax(first), self.expr(e.args[1])]
            return [self.expr(first), self.expr(e.args[1])]
        return [self.expr(e)]

    def binary(self, fn: str, x: Expr, y: Expr) -> T:
        a, b = self.expr(x), self.expr(y)
        if fn == "add":
            return T(f"{wrap(a, SUM)} + {wrap(b, SUM)}", SUM)
        if fn == "sub":
            return T(f"{wrap(a, SUM)} - {wrap(b, SUM + 1)}", SUM)
        if fn == "mul":
            if self.bare and x.sig and y.sig:             # vectors without their indices: elementwise, so written
                return T(wrap(a, PRODUCT) + r" \odot " + wrap(b, PRODUCT + 1), PRODUCT)
            if x.sig == () or y.sig == () or self.index_tex(x.sig) is not None and self.index_tex(y.sig) is not None \
                    or isinstance(x, Const) or isinstance(y, Const):
                # a number, or values written with all their indices (n_{ig} q_{ig}): an ordinary product
                joint = r" \cdot " if b.tex[:1].isdigit() or b.tex.startswith("-") else r"\,"
            else:                                       # arrays, their indices not known: numpy's elementwise *
                joint = r" \odot "
            return T(wrap(a, PRODUCT) + joint + wrap(b, PRODUCT + 1), PRODUCT)
        if fn == "div":
            return T(r"\frac{" + a.tex + "}{" + b.tex + "}")
        if fn == "floordiv":
            return T(r"\left\lfloor\frac{" + a.tex + "}{" + b.tex + r"}\right\rfloor")
        if fn == "mod":
            return T(f"{wrap(a, PRODUCT)} \\bmod {wrap(b, PRODUCT + 1)}", PRODUCT)
        if fn == "pow":
            return T(base_tex(a) + "^{" + b.tex + "}", POWER)
        if fn == "matmul":
            return T(wrap(a, PRODUCT) + r"\," + wrap(b, PRODUCT + 1), PRODUCT)
        if fn == "and" and isinstance(x, Apply) and isinstance(y, Apply) and x.fn in ("lt", "le", "gt", "ge") \
                and y.fn in ("lt", "le", "gt", "ge") and x.args[1] == y.args[0]:
            # a < b < c, as the code wrote it
            return T(f"{wrap(self.expr(x.args[0]), COMPARE + 1)} {COMPARE_TEX[x.fn]} {wrap(self.expr(x.args[1]), COMPARE + 1)}"
                     f" {COMPARE_TEX[y.fn]} {wrap(self.expr(y.args[1]), COMPARE + 1)}", COMPARE)
        if fn == "and":
            return T(f"{wrap(a, AND + 1)} \\land {wrap(b, AND + 1)}", AND)
        if fn == "or":
            return T(f"{wrap(a, OR + 1)} \\lor {wrap(b, OR + 1)}", OR)
        raise ValueError(fn)

    def under(self, over: tuple[Any, ...], where: Expr | None) -> str:
        """What a sum or a family runs over, written under it (and above it, after "^", for a range)."""
        if len(over) == 2 and all(isinstance(i, Axis) for i in over) and isinstance(where, Apply) \
                and where.fn == "eq" and all(isinstance(a, Idx) for a in where.args):
            return f"{self.letter(over[0])}={self.letter(over[1])}"     # matched indices: Σ_{g=d}
        parts = []
        top = ""
        for i in over:
            if isinstance(i, Axis):
                parts.append(self.letter(i) or r"\text{axis}")
            elif isinstance(i, Pos):
                parts.append(r"\text{axis } " + str(i.k))
            elif isinstance(i, Binder):
                p, t = self.binder(i)
                parts.append(p)
                top = t
        out = ",".join(parts)
        if where is not None:
            out += " : " + self.expr(where).tex
        return out + ("}^{" + top if top else "")

    def binder(self, b: Binder) -> tuple[str, str]:
        """A loop variable's range: ("i=0", "n - 1") for range(n), ("x \\in xs", "") otherwise."""
        var = self.target(b.var)
        d = b.domain
        if isinstance(d, Range) and isinstance(b.var, str):
            lo = self.expr(d.lo).tex
            hi = d.hi
            top = str(hi.value - 1) if isinstance(hi, Const) and isinstance(hi.value, int) else \
                f"{wrap(self.expr(hi), SUM)} - 1"
            return f"{var}={lo}", top
        return var + r" \in " + self.expr(d).tex, ""

    def target(self, var: Any) -> str:
        if isinstance(var, str):
            if var == "_":
                return r"\_"
            n = self.names.get(var)
            return n.tex if n is not None else self.symbol(var)
        return r"\left(" + ", ".join(self.target(v) for v in var) + r"\right)"

    def count(self, e: Aggregate) -> T:
        """How many rows: |{d : cond}|, or the size of the set."""
        letters = ",".join(self.letter(i) or "" for i in e.over if isinstance(i, Axis))
        if e.where is None:
            return T(r"\left|" + letters + r"\right|")
        return T(r"\left|\left\{" + letters + " : " + self.expr(e.where).tex + r"\right\}\right|")

    def e_Lambda(self, e: Lambda) -> T:
        params = ", ".join(self.symbol(b.var) for b in e.over if isinstance(b.var, str))
        return T(params + r" \mapsto " + self.expr(e.body).tex, OR)

    def e_Optimum(self, e: Optimum) -> T:
        """argmin_β f(β), with how it is found under it (the optimizer, its start, its bounds), so that the
        point the code takes is what is written."""
        if isinstance(e.fun, Lambda) and e.fun.over and isinstance(e.fun.over[0].var, str):
            var = self.symbol(e.fun.over[0].var)
            body = self.expr(e.fun.body).tex
        else:
            var = r"x"
            body = self.expr(e.fun).tex + r"\left(x\right)"
        how = []
        if isinstance(e.method, (Const, Text)) and isinstance(e.method.value, str):
            how.append(r"\text{by " + text(e.method.value) + "}")
        elif e.method is not None:
            how.append(r"\text{by } " + self.expr(e.method).tex)
        how.append(r"\text{from } " + self.expr(e.start).tex)
        if e.bounds is not None:
            how.append(r"\text{within } " + self.expr(e.bounds).tex)
        if e.jac is not None:
            how.append(r"\text{gradient } " + (self.expr(e.jac.body).tex if isinstance(e.jac, Lambda)
                                                else self.expr(e.jac).tex))
        op = r"\operatorname*{arg\,min}" if e.op == "min" else r"\operatorname*{arg\,max}"
        return T(op + "_{" + var + r"} \left[" + body + r"\right]_{\substack{" + r" \\ ".join(how) + "}}",
                 SUM)

    def e_Aggregate(self, e: Aggregate) -> T:
        if e.op == "count":
            return self.count(e)
        # an index it combines over that is also free where it is used (v = argmax_v A_v): its own copy, v'
        saved = dict(self.relabel)
        inside = {a for x in _walk(e.body) for a in (x.sig or ())}
        for i in e.over:
            if isinstance(i, Axis) and (i in self.outer or i in self.relabel.values()):
                k = i.copy + 1                          # a copy neither free outside, nor used inside, nor taken
                while any(Axis(i.name, k) == o for o in (*self.outer, *inside, *self.relabel.values())):
                    k += 1
                self.relabel[i] = Axis(i.name, k, i.where, i.part, i.letter)
        try:
            sub = self.under(e.over, e.where)
            op = AGGREGATE_TEX.get(e.op, operator(e.op))
            body = self.expr(e.body)
        finally:
            self.relabel = saved
        return T(op + ("_{" + sub + "}" if sub else "") + " " + wrap(body, PRODUCT), PRODUCT)

    def e_Build(self, e: Build) -> T:
        sub = self.under(e.over, e.where)
        top = ""
        if "}^{" in sub:
            sub, top = sub.split("}^{")
        return T(r"\left(" + self.expr(e.body).tex + r"\right)_{" + sub + "}" + (f"^{{{top}}}" if top else ""))

    def e_Mapping(self, e: Mapping) -> T:
        if not e.pairs:
            return T(r"\varnothing")
        pairs = [f"{self.expr(k).tex} \\mapsto {self.expr(v).tex}" for k, v in e.pairs]
        body = r"\left\{" + ", ".join(pairs) + r"\right\}"
        if not e.over:
            return T(body)
        sub = self.under(e.over, None)
        top = ""
        if "}^{" in sub:
            sub, top = sub.split("}^{")
        return T(body + "_{" + sub + "}" + (f"^{{{top}}}" if top else ""))

    def e_Tup(self, e: Tup) -> T:
        if e.kind == "set":
            return T(r"\left\{" + ", ".join(self.expr(x).tex for x in e.items) + r"\right\}")
        if e.kind == "list" and not e.items:
            return T(r"\left(\right)")
        return T(r"\left(" + ", ".join(self.expr(x).tex for x in e.items) + r"\right)")

    def e_Key(self, e: Key) -> T:
        """A group's key: the key column's name, indexed by the group (plate_p)."""
        idx = self.letter(e.axis)
        return T(indexed(ident(e.name), idx) if idx else ident(e.name))

    def e_Table(self, e: Table) -> T:
        """A table, as its columns: a base table's (if it keeps them) and its own, each defined; over its rows."""
        parts = [] if e.base is None else [self.expr(e.base).tex]
        for n, x in e.cols:
            if isinstance(x, Key):
                parts.append(self.expr(x).tex)
                continue
            idx = self.index_tex(x.sig)
            parts.append(f"{indexed(ident(n), idx) if idx else ident(n)} = {self.expr(x).tex}")
        return T(r"\left(" + r",\; ".join(parts) + r"\right)")

    def e_Idx(self, e: Idx) -> T:
        return T(self.letter(e.axis) or r"\text{index}")

    def e_Extent(self, e: Extent) -> T:
        if isinstance(e.axis, Axis) and e.axis.known and e.axis.name is not None:
            a = e.axis
            if a.where is not None:                     # the part of a set where a mask holds: how many
                return T(r"\left|\left\{" + (self.letter(a) or "") + " : " + self.expr(a.where).tex + r"\right\}\right|")
            n = a.letter or a.name or ""
            return T(r"\left|" + (r"\mathit{" + text(n) + "}" if len(n) > 1 else n) + r"\right|")
        k = e.axis.k if isinstance(e.axis, Pos) else 0
        of = self.expr(e.of)
        if k == 0:
            return T(r"\operatorname{len}\left(" + of.tex + r"\right)")
        return T(r"\operatorname{shape}\left(" + of.tex + r"\right)_{" + str(k) + "}")

    def e_Range(self, e: Range) -> T:
        hi = e.hi
        top = str(hi.value - 1) if isinstance(hi, Const) and isinstance(hi.value, int) else \
            f"{wrap(self.expr(hi), SUM)} - 1"
        return T(r"\left\{" + self.expr(e.lo).tex + r", \ldots, " + top + r"\right\}")

    def e_Cases(self, e: Cases) -> T:
        rows = []
        node: Expr = e
        while isinstance(node, Cases):
            for c, v in node.branches:
                rows.append(self.expr(v).tex + r" & \text{if } " + self.condition(c))
            node = node.otherwise
        rows.append(self.expr(node).tex + r" & \text{otherwise}")
        return T(r"\begin{cases}" + r" \\ ".join(rows) + r"\end{cases}")

    def condition(self, c: Expr) -> str:
        return self.expr(c).tex

    def e_Op(self, e: Op) -> T:
        if e.qualname == "builtins.str.format" and e.args and isinstance(e.args[0], Text) and e.args[0].value:
            return self.formatted(e.args[0].value, e.args[1:])
        return self.e_Op_named(e)

    def formatted(self, template: str, values: tuple[Expr, ...]) -> T:
        """An f-string: its text, each value it formats written as math in its place (“c” (i + 1))."""
        import string
        out, k = [], 0
        for literal, field_name, _, _ in string.Formatter().parse(template):
            if literal:
                out.append(r"\text{“" + text(literal) + "”}")
            if field_name is not None and k < len(values):
                out.append(wrap(self.expr(values[k]), PRODUCT))
                k += 1
        return T(r"\,".join(out) or r"\text{“”}", PRODUCT)

    def e_Op_named(self, e: Op) -> T:
        if e.qualname in MADE:
            return T(MADE[e.qualname])
        if (e.qualname in VALUES_OF_FIRST or e.name in VALUES_OF_FIRST) and e.args:
            return self.expr(e.args[0])
        if e.qualname == "torch.autograd.grad" and len(e.args) == 2:     # ∇_θ L
            wrt = e.args[1].items if isinstance(e.args[1], Tup) else (e.args[1],)
            return T(r"\nabla_{" + ", ".join(self.expr(w).tex for w in wrt) + "} " + wrap(self.expr(e.args[0]),
                                                                                        PRODUCT), PRODUCT)
        name = self.names[e.name].tex if e.name in self.names else operator(e.name)
        parts = []
        for x in e.args:
            if isinstance(x, Op) and x.qualname == "*" and x.args:      # f(*xs): xs's items
                inner = x.args[0]
                parts += [self.expr(i).tex for i in inner.items] if isinstance(inner, Tup) else \
                    [r"\ast " + self.expr(inner).tex]
            else:
                parts.append(self.expr(x).tex)
        parts += [r"\text{" + text(k) + "} = " + self.expr(v).tex for k, v in e.kwargs]
        return T(name + r"\left(" + ", ".join(parts) + r"\right)")

    def e_Item(self, e: Item) -> T:
        b = e.base
        if isinstance(b, Var) and b.type == "list" and (isinstance(e.key, Const) and type(e.key.value) is int
                                                        and e.key.value >= 0 or isinstance(e.key, Var)):
            # an element of a family: its index first, then its own (a_{0,bg})
            n = self.aliases.get(b.name, b.name)
            name = self.names.get(n)
            base = _strip_indices(name.tex, name.dims) if name is not None and name.dims is not None else \
                self.symbol(n)
            idx = self.index_tex(e.sig) or ""
            k = self.expr(e.key).tex
            return T(indexed(base, k + ("," + idx if idx else "")))
        if isinstance(e.key, (Const, Text)) and isinstance(e.key.value, str) and e.key.value.isidentifier() \
                and isinstance(b, Field):
            # an object's slot by name (self.var["dispersions"], adata.obsm["X_pca"]): the slot as a symbol, its
            # container above it (dispersions^{(var)}), so that it is not taken for a variable of the same name
            slot_idx = self.index_tex(e.sig)
            slot = self.symbol(e.key.value) + r"^{(\mathit{" + text(b.attr) + "})}"
            return T(indexed(slot, slot_idx) if slot_idx else slot)
        return T(base_tex(self.expr(e.base)) + r"_{[" + self.key(e.key) + "]}")

    def key(self, k: Expr) -> str:
        if isinstance(k, SliceKey):
            ends = [self.expr(x).tex if x is not None else "" for x in (k.lo, k.hi)]
            return ":".join(ends + ([self.expr(k.step).tex] if k.step is not None else []))
        if isinstance(k, Tup):
            return r",\,".join(self.key(x) for x in k.items)
        return self.expr(k).tex

    def e_SliceKey(self, e: SliceKey) -> T:
        return T(self.key(e))

    def e_Align(self, e: Align) -> T:
        if e.how == "rename" and e.base.sig is not None and e.sig is not None:
            saved = dict(self.relabel)
            renames = {old: new for old, new in zip(e.base.sig, e.sig) if old != new}
            # an index the result's takes that the expression uses for another: that one moved to a copy no index
            # of it is (else (H_{cc'})^{-1} v_c, its c' named c, would read H_{cc})
            inside = {a for x in _walk(e.base) for a in (x.sig or ()) if a.known}
            for new in set(renames.values()):
                if new in inside and new not in renames:
                    k = new.copy + 1
                    while Axis(new.name, k) in inside or Axis(new.name, k) in renames.values():
                        k += 1
                    renames[new] = Axis(new.name, k, new.where, new.part, new.letter)
            for old, new in renames.items():
                self.relabel[old] = new
            try:
                return self.expr(e.base)
            finally:
                self.relabel = saved
        if e.how == "transpose" and not (known(e.base.sig) and self.index_tex(e.base.sig)):
            return T(base_tex(self.expr(e.base)) + r"^\top", POWER)
        return self.expr(e.base)

    def e_Restrict(self, e: Restrict) -> T:
        """x on a part of an axis: x_{cg}|_{learns_c}, x_{cg}|_{g < k}; parts of several axes together."""
        conds: list[str] = []
        node: Expr = e
        while isinstance(node, Restrict):
            c = self.restriction(node)
            if c is None:
                return self.raw_index(e)
            conds.insert(0, c)
            node = node.base
        return T(r"\left." + self.expr(node).tex + r"\right|_{" + r",\,".join(conds) + "}")

    def restriction(self, e: Restrict) -> str | None:
        sig = e.base.sig
        a = sig[e.pos] if sig is not None and e.pos < len(sig) else None
        d = self.letter(a) if a is not None else None
        if e.mask is not None:
            m = self.expr(e.mask)
            if a is not None and e.mask.sig is not None and len(e.mask.sig) == 1 and e.mask.sig[0].base() == a.base():
                return m.tex                            # x where learns_c
            if d is None:
                return None
            return f"{d} \\in {m.tex}"
        assert e.part is not None
        lo, hi, step = e.part
        if d is not None and lo is not None and hi is None and step is None and lo.type == "slice":
            return f"{d} \\in {self.expr(lo).tex}"     # a slice a variable holds: the indices in it
        if d is None or step is not None or any(_negative(x) for x in (lo, hi)):
            return None
        if lo is not None and hi is not None:
            return f"{self.expr(lo).tex} \\le {d} < {self.expr(hi).tex}"
        if hi is not None:
            return f"{d} < {self.expr(hi).tex}"
        if lo is not None:
            return f"{d} \\ge {self.expr(lo).tex}"
        return ""

    def raw_index(self, e: Restrict | Take) -> T:
        """x_{[:, ::-1]}: the code's own indexing, where the math would not be clearer."""
        node: Expr = e
        keys: list[tuple[int, str]] = []
        while isinstance(node, (Restrict, Take)):
            if isinstance(node, Restrict):
                if node.part is not None:
                    lo, hi, step = node.part
                    k = ":".join(self.expr(x).tex if x is not None else "" for x in (lo, hi)) + \
                        (":" + self.expr(step).tex if step is not None else "")
                else:
                    assert node.mask is not None
                    k = self.expr(node.mask).tex
                keys.append((node.pos, k))
            else:
                for p, i in zip(node.pos, node.index):
                    keys.append((p, self.expr(i).tex))
            node = node.base
        n = max(p for p, _ in keys) + 1
        parts = [":"] * n
        for p, k in keys:
            parts[p] = k
        return T(base_tex(self.expr(node)) + r"_{[" + r",\,".join(parts) + "]}")

    def e_Take(self, e: Take) -> T:
        """A value at the positions pointers give: v_{π_k, k} (the pointers written in place of their indices),
        x_{cg}|_{g = 0} (a number)."""
        sig = e.base.sig
        if sig is None or not known(sig) or any(_negative(i) for i in e.index):
            return self.raw_index(e)
        if all(isinstance(i, Idx) for i in e.index) and len({i.axis for i in e.index}) == 1:   # type: ignore
            # a diagonal: the value with its indices made one (diag(M M^T)_g = Σ_i m_{gi} m_{gi})
            saved = dict(self.relabel)
            target = e.index[0].axis                    # type: ignore[attr-defined]
            target = self.relabel.get(target, target)
            for p in e.pos:
                if sig[p] != target:
                    self.relabel[sig[p]] = target
            try:
                return self.expr(e.base)
            finally:
                self.relabel = saved
        if all(isinstance(i, Const) for i in e.index):
            conds = [f"{self.letter(sig[p])} = {self.expr(i).tex}" for p, i in zip(e.pos, e.index)]
            return T(r"\left." + self.expr(e.base).tex + r"\right|_{" + r",\,".join(conds) + "}")
        if isinstance(e.base, Var) and e.base.name not in self.names:
            base = self.symbol(e.base.name)
            subs = []
            for k, a in enumerate(sig):
                if a == NEW:
                    continue
                if k in e.pos:
                    subs.append(self.expr(e.index[e.pos.index(k)]).tex)
                else:
                    subs.append(self.letter(a) or "")
            return T(base + "_{" + ",".join(subs) + "}")
        return self.raw_index(e)

    def e_Dist(self, e: Dist) -> T:
        name = DIST_TEX.get(e.family, operator(e.family))
        params = dict(e.params)
        if e.family == "normal" and set(params) >= {"loc", "scale"}:
            s = self.expr(params["scale"])
            return T(name + r"\left(" + self.expr(params["loc"]).tex + ", " + base_tex(s) + r"^{2}\right)")
        return T(name + r"\left(" + ", ".join(self.expr(v).tex for _, v in e.params) + r"\right)")

    def e_Draw(self, e: Draw) -> T:
        return T(r"\operatorname{draw}\left(" + self.expr(e.dist).tex + r"\right)")

    def e_Density(self, e: Density) -> T:
        d = self.e_Dist(e.dist).tex
        name, _, rest = d.partition(r"\left(")
        inner = rest.removesuffix(r"\right)")
        tex = name + r"\left(" + self.expr(e.at).tex + r" \mid " + inner + r"\right)"
        return T((r"\log " if e.log else "") + tex, PRODUCT if e.log else ATOM)

    # ---- statements ------------------------------------------------------------------------------------------

    def block(self, body: list[Stmt] | tuple[Stmt, ...], fn: Any, depth: int) -> list[str]:
        pad = r"\quad " * depth
        out: list[str] = []
        for st in body:
            out += self.statement(st, fn, depth, pad)
        return out

    def statement(self, st: Stmt, fn: Any, depth: int, pad: str) -> list[str]:
        value = getattr(st, "value", None) or getattr(st, "test", None)
        self.outer = {a for a in (value.sig or ()) if a.known and a != NEW} if isinstance(value, Expr) else set()
        try:
            return self._statement(st, fn, depth, pad)
        finally:
            self.outer = set()

    def _statement(self, st: Stmt, fn: Any, depth: int, pad: str) -> list[str]:
        if isinstance(st, Unwritten):
            return [pad + r"&\;\text{(not written: line " + str(st.line) + ")}"]
        if isinstance(st, Assume):
            return [pad + ASSUMING + self.expr(st.test).tex]
        if isinstance(st, Effect):
            return [pad + "&" + self.expr(st.value).tex]
        if isinstance(st, Let):
            if len(st.targets) == 1 and st.targets[0] in self.aliases:
                return []
            value = self.expr(st.value)
            again = st.again and all(t in self.defined for t in st.targets)
            rel = r"\leftarrow" if again else "="
            lhs = self.lhs(st)
            self.defined |= set(st.targets)
            if isinstance(st.value, Draw):
                return [f"{pad}{lhs} &\\sim {self.expr(st.value.dist).tex}"]
            if self.matrix_form(st.value):              # linear algebra: the statement in matrix notation
                self.bare = True
                try:
                    lhs = self.lhs(st)
                finally:
                    self.bare = False
                return [f"{pad}{lhs} &{rel} {self.matrix(st.value).tex}"]  # type: ignore
            return [f"{pad}{lhs} &{rel} {value.tex}"]
        if isinstance(st, SetPart):
            part = st.shown if st.shown is not None and not isinstance(st.shown, Item) else st.target
            return [f"{pad}{self.expr(part).tex} &\\leftarrow {self.expr(st.value).tex}"]
        if isinstance(st, Return):
            if st.value is None:
                return []
            if isinstance(st.value, Draw):              # what it returns is a draw: f(...) ~ D
                return [f"{pad}{self.return_lhs(st.value, fn)} &\\sim {self.expr(st.value.dist).tex}"]
            if self.matrix_form(st.value):
                return [f"{pad}{self.return_lhs(st.value, fn)} &= {self.matrix(st.value).tex}"]  # type: ignore
            lhs, written = self.return_lhs(st.value, fn), self.expr(st.value).tex
            return [] if lhs == written else [f"{pad}{lhs} &= {written}"]     # β = β: said by the lines above
        if isinstance(st, For):
            return self.loop(st, fn, depth, pad)
        if isinstance(st, While):
            head = pad + r"&\text{while } " + self.expr(st.test).tex + r"\text{:}"
            return [head, *self.block(st.body, fn, depth + 1), *self.no_break(st, fn, depth, pad)]
        if isinstance(st, If):
            cases = self.cases(st, pad)
            if cases is not None:
                return [cases]
            out = [pad + r"&\text{if } " + self.expr(st.test).tex + r"\text{:}", *self.block(st.body, fn, depth + 1)]
            if st.orelse:
                out += [pad + r"&\text{else:}", *self.block(st.orelse, fn, depth + 1)]
            return out
        if isinstance(st, Jump):
            return [pad + r"&\text{" + st.kind + "}"]
        return [pad + r"&\;\text{(not written: line " + str(st.line) + ")}"]

    def lhs(self, st: Let) -> str:
        parts = [self.var(Var(t, sig=st.value.sig if len(st.targets) == 1 else _part_sig(st.value, k))).tex
                 if t != "_" else r"\_" for k, t in enumerate(st.targets)]
        return parts[0] if len(parts) == 1 else r"\left(" + ", ".join(parts) + r"\right)"

    def return_lhs(self, value: Expr, fn: Any) -> str:
        items = value.items if isinstance(value, Tup) and value.kind == "tuple" else None
        if "return" in self.names:
            return self.names["return"].tex
        if items and all(f"return.{k}" in self.names for k in range(len(items))):
            # what it returns, item by item, as the symbols the step gives them: (b̂, D) = (b, d)
            return r"\left(" + ", ".join(self.names[f"return.{k}"].tex for k in range(len(items))) + r"\right)"
        if fn is not None:                              # f(x, ...): a method's self left out
            return function_tex(fn.name) + r"\left(" + ", ".join(
                self.given[p].tex if p in self.given else self.var(Var(p, sig=s)).tex
                for p, s, _ in fn.params if p != "self") + r"\right)"
        return r"\text{result}"

    def loop(self, st: For, fn: Any, depth: int, pad: str) -> list[str]:
        d = st.domain
        if isinstance(d, Range) and isinstance(d.lo, Const) and d.lo.value == 0:
            # as an algorithm is written: "repeat n times:", or "for r = 0, ..., n - 1:" if the turn is read
            count = self.expr(d.hi)
            used = st.var is not None and any(n in st.var for b in walk_stmts(list(st.body))
                                              for e in _exprs(b) for n in free_vars(e))
            if not used:
                head = r"&\textbf{repeat } " + count.tex + r"\textbf{ times:}"
            else:
                hi = d.hi
                end = str(hi.value - 1) if isinstance(hi, Const) and isinstance(hi.value, int) else \
                    wrap(count, SUM) + " - 1"
                assert st.var is not None
                head = r"&\textbf{for } " + self.target(st.var[0] if len(st.var) == 1 else st.var) + \
                    r" = 0, \ldots, " + end + r"\textbf{:}"
            self.defined |= set(st.var or ())
            return [pad + head, *self.block(st.body, fn, depth + 1), *self.no_break(st, fn, depth, pad)]
        var = "_" if st.var is None else st.var[0] if len(st.var) == 1 else st.var
        head = pad + r"&\text{for } " + self.target(var) + r" \in " + self.expr(d).tex + r"\text{:}"
        self.defined |= set(st.var or ())
        return [head, *self.block(st.body, fn, depth + 1), *self.no_break(st, fn, depth, pad)]

    def no_break(self, st: For | While, fn: Any, depth: int, pad: str) -> list[str]:
        """A loop's else: what is done when it ends without a break."""
        if not st.orelse:
            return []
        return [pad + r"&\text{if it ended without a break:}", *self.block(st.orelse, fn, depth + 1)]

    def cases(self, st: If, pad: str) -> str | None:
        """An if that decides values: each branch assigns them (one assignment each), and a value a branch does not
        assign keeps the one it had. Written as one equation with cases, the variables the code had before written
        primed: (z', e) = {f(z) if κ; (z, ∅) otherwise}. None if the if does more than that."""
        def branch(body: tuple[Stmt, ...]) -> tuple[list[str], Expr] | None:
            if not body:
                return [], Tup(())
            if len(body) != 1 or not isinstance(b := body[0], Let):
                return None
            return list(b.targets), b.value
        if st.orelse and isinstance(st.orelse[0], If):
            return None                                 # elif: not yet
        yes, no = branch(st.body), branch(st.orelse)
        if yes is None or no is None or not yes[0]:
            return None
        names = list(dict.fromkeys([*yes[0], *no[0]]))

        def value(b: tuple[list[str], Expr]) -> str | None:
            assigned, e = b
            if assigned == names:
                return self.expr(e).tex
            parts = dict(zip(assigned, e.items)) if isinstance(e, Tup) and len(e.items) == len(assigned) else \
                ({assigned[0]: e} if len(assigned) == 1 else None)
            if parts is None:
                return None
            items = [self.expr(parts[n]).tex if n in parts else self.var(Var(n)).tex for n in names]
            return items[0] if len(items) == 1 else r"\left(" + ", ".join(items) + r"\right)"
        a, b = value(yes), value(no)
        if a is None or b is None:
            return None
        cond = self.condition(st.test)
        lhs = [self.names[n + "="].tex if n + "=" in self.names else
               primed(self.var(Var(n)).tex) if n in self.defined else self.var(Var(n)).tex for n in names]
        self.defined |= set(names)
        for n, tex in zip(names, lhs):                  # from here on, the new values
            self.given.setdefault(n, Name(self.var(Var(n)).tex))     # (a parameter, as it was given)
            self.names[n] = Name(tex, self.names[n].dims if n in self.names else None)
        left = lhs[0] if len(lhs) == 1 else r"\left(" + ", ".join(lhs) + r"\right)"
        return (f"{pad}{left} &= \\begin{{cases}}{a} & \\text{{if }} {cond} \\\\ {b} & \\text{{otherwise}}"
                f"\\end{{cases}}")


def _core(sig: Sig) -> tuple[Axis, ...] | None:
    """A signature without its axes of length one (put in to broadcast along)."""
    return None if sig is None else tuple(a for a in sig if a != NEW)


def _walk(e: Expr) -> list[Expr]:
    from headway.core.terms import walk
    return walk(e)


def _exprs(st: Stmt) -> list[Expr]:
    from headway.core.terms import stmt_exprs
    return stmt_exprs(st)


def _part_sig(e: Expr, k: int) -> Sig:
    if e.parts is not None and k < len(e.parts):
        return e.parts[k].sig
    if isinstance(e, Tup) and k < len(e.items):
        return e.items[k].sig
    return None


def _negative(e: Expr | None) -> bool:
    return isinstance(e, Const) and isinstance(e.value, (int, float)) and e.value < 0 or \
        isinstance(e, Apply) and e.fn == "neg"


def _dotted(e: Expr) -> str | None:
    if isinstance(e, Var):
        return e.name
    if isinstance(e, Field):
        head = _dotted(e.base)
        return None if head is None else f"{head}.{e.attr}"
    return None


def _strip_indices(tex: str, dims: tuple[str | None, ...]) -> str:
    """A symbol's TeX without the indices its dims give it (z_{ig}: z; p_{0,bv}: p_{0})."""
    idx = "".join(d for d in dims if d)
    for suffix in ("_{" + idx + "}", "_" + idx):
        if idx and tex.endswith(suffix):
            return tex[:-len(suffix)]
    m = re.fullmatch(r"(.*)_\{(.*)," + re.escape(idx) + r"\}", tex) if idx else None
    if m:
        return f"{m.group(1)}_{{{m.group(2)}}}"
    return tex


# ---- notation helpers ----------------------------------------------------------------------------------------

def wrap(v: T, prec: int) -> str:
    return v.tex if v.prec >= prec else r"\left(" + v.tex + r"\right)"


def base_tex(v: T) -> str:
    """v, to put a subscript or superscript on: braced if it has one already (x_i^\\top, {w_g}_{[k]})."""
    out = wrap(v, ATOM)
    return "{" + out + "}" if out == v.tex and scripted(out) else out


def scripted(tex: str) -> bool:
    """Whether tex has a subscript or superscript of its own: outside any {...} and \\left...\\right."""
    depth = 0
    k = 0
    while k < len(tex):
        if tex.startswith(r"\left", k) or tex.startswith(r"\right", k):
            depth += 1 if tex.startswith(r"\left", k) else -1
            k += 5
        elif tex[k] == "\\":
            k += 2
            continue
        elif tex[k] == "{":
            depth += 1
        elif tex[k] == "}":
            depth -= 1
        elif tex[k] in "_^" and depth == 0:
            return True
        k += 1
    return False


def indexed(base: str, indices: str) -> str:
    """A symbol written with its indices: x_{ig}; one with a subscript of its own takes them after it, p_{0,ig}."""
    if not scripted(base):
        return base + "_{" + indices + "}"
    m = re.fullmatch(r"(.*)_\{([^{}]*)\}", base) or re.fullmatch(r"(.*)_([A-Za-z0-9])", base)
    if m and not scripted(m.group(1)):
        return f"{m.group(1)}_{{{m.group(2)},{indices}}}"
    return "{" + base + "}_{" + indices + "}"


def primed(tex: str) -> str:
    """A symbol's new value: primed, before its indices (z_{cs}: z'_{cs})."""
    m = re.fullmatch(r"(.*?)(_\{[^{}]*\}|_[A-Za-z0-9])", tex)
    if m and not scripted(m.group(1)):
        return f"{m.group(1)}'{m.group(2)}"
    return tex + "'" if not scripted(tex) else "{" + tex + "}'"


def constant(v: object) -> T:
    if isinstance(v, bool) or v is None:
        return T(r"\text{" + str(v) + "}")
    if isinstance(v, int):
        return T(str(v), ATOM if v >= 0 else UNARY)
    if isinstance(v, float):
        if v != v:
            return T(r"\mathrm{NaN}")
        if v in (float("inf"), float("-inf")):
            return T(r"\infty" if v > 0 else r"-\infty", ATOM if v > 0 else UNARY)
        if v == 3.141592653589793:
            return T(r"\pi")
        if v == 2.718281828459045:
            return T("e")
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):               # 1e-3 -> 10^{-3}, 2.5e-8 -> 2.5 \cdot 10^{-8}
            mantissa, exp = f"{v:e}".split("e")
            digits, power = mantissa.rstrip("0").rstrip("."), "10^{" + str(int(exp)) + "}"
            sign = "-" if digits.startswith("-") else ""
            digits = digits.lstrip("-")
            return T(sign + power, ATOM if not sign else UNARY) if digits == "1" else \
                T(sign + digits + r" \cdot " + power, PRODUCT)
        out = repr(v).removesuffix(".0")
        return T(out, ATOM if v >= 0 else UNARY)
    if isinstance(v, str):
        return T(r"\text{“" + text(v) + "”}")
    if v is Ellipsis:
        return T(r"\ldots")
    return T(r"\text{" + text(str(v)) + "}")


def text(v: str) -> str:
    """A string, to put in \\text{}: TeX's special characters escaped, braces and backslashes dropped."""
    v = v.replace("\\", "").replace("{", "").replace("}", "")
    for c in "_&%#$":
        v = v.replace(c, "\\" + c)
    return v.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")


def ident(name: str) -> str:
    """A code name, written: greek names as letters, a trailing number as a subscript, a one-letter name in
    italics, a longer one as an italic word."""
    if name in GREEK:
        return "\\" + name
    stem, digits = name.rstrip("0123456789"), name[len(name.rstrip("0123456789")):]
    stem = stem.rstrip("_") if digits else stem
    if digits and stem:
        return ident(stem) + "_{" + digits + "}"
    if len(name) == 1:
        return name
    return r"\mathit{" + name.replace("_", r"\_") + "}"


def operator(name: str) -> str:
    return r"\operatorname{" + name.replace("_", r"\_") + "}"


def function_tex(name: str) -> str:
    """A function's name, as its calls write it: an operator (\\operatorname{bin\\_step}), or a single letter."""
    return ident(name) if len(name) == 1 else r"\operatorname{" + text(name) + "}"


def notation(name: str) -> bool:
    """A name that is a symbol already: one letter (with a number: x0), or a greek letter's name."""
    base = name.rstrip("0123456789").rstrip("_")
    return len(base) == 1 and base.isalpha() or base.lower() in GREEK_WORDS


def neighbours(name: str, greek_first: bool) -> list[str]:
    """The symbols a code name may take, nearest first: its words' initials (and their capitals), their greek
    counterparts, the first initial accented, subscripted by the second, then numbered."""
    base = name.rstrip("0123456789").rstrip("_")
    digits = name[len(base):].lstrip("_")
    words = [w for w in re.split(r"_+|(?<=[a-z])(?=[A-Z])", base) if w]
    if base.lower() in GREEK_WORDS:                     # mu, sigma: already a symbol
        letter = "\\" + base.lower()
        return [letter + (f"_{{{digits}}}" if digits else "")] + neighbours(base[0], greek_first)
    if len(base) == 1 and base.isalpha():               # x, x0, X: as written
        first = [base + (f"_{{{digits}}}" if digits else "")]
    else:
        first = []
    initials = list(dict.fromkeys(w[0].lower() for w in words if w[0].isalpha())) or ["v"]
    latin = [c for i in initials for c in (i, i.upper())]
    greek = []
    for i in initials:
        g = GREEK_OF.get(i)
        if g:
            greek += ["\\" + g] + (["\\" + g.capitalize()] if g in CAPITAL_GREEK else [])
    near = first + (greek + latin if greek_first else latin + greek)
    a = initials[0]
    near += [rf"\tilde{{{a}}}", rf"\hat{{{a}}}", rf"\bar{{{a}}}"]
    if len(initials) > 1:
        near += [f"{a}_{{{i}}}" for i in initials[1:]]
    near += [f"{a}_{{{n}}}" for n in range(1, 100)]
    return [t for t in near if t not in NOT_SYMBOLS]


__all__ = ["Printer", "Name", "T", "ASSUMING", "INDEX_LETTERS", "Draw", "Dist", "Density"]
