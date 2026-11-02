"""polars' table: data frames as tables of the calculus, and polars' expressions as expressions of a row.

A table is named columns over one index set of rows (a parameter's, named after it: `design`'s rows are the index
set design). A column is a value over the rows (`design["learns"]` is learns_d). polars' expressions (`pl.col("x")`,
`pl.col("x") > 0`, `pl.col("x").sum()`) are read as expressions of a row of the table they are used on, and put in
its rows where they are used:

  df.filter(e)                     the rows where e holds              df|_{e_d}
  df.with_columns(y=e)             a column made of the rows' e        (df, y_d = e_d)
  df.group_by("k").agg(e.sum())    a row per value of k, its e summed  (k_k, x_k = Σ_{d : k_d = k} e_d)
  df.group_by("k").len()           a row per value of k, its count     (k_k, len_k = |{d : k_d = k}|)

A verb this table does not know (sort, unique, join for now, replace_strict) is an operator.
"""
from __future__ import annotations

from dataclasses import replace

from headway.core.libs import Call, Table as Lib, register
from headway.core.terms import (Aggregate, Apply, Axis, Const, Expr, Field, Key, Op, Restrict, Shape, Table, Take,
                                Text, Tup, Var, broadcast, resig)

POLARS = register(Lib("polars"))
DF, EXPR, SERIES = "polars.DataFrame", "polars.Expr", "polars.Series"
POLARS.types.update({"polars.DataFrame": DF, "polars.Series": SERIES, "polars.Expr": EXPR,
                     "polars.LazyFrame": DF})
ROW = Axis("·row")                                     # the rows of the table an expression will be used on
ROWVAR = "·row"
AGG = {"sum": "sum", "mean": "mean", "max": "max", "min": "min", "median": "median", "std": None, "var": None,
       "len": "count", "count": "count", "n_unique": None, "first": "first", "last": "last", "any": "any", "all": "all"}


def _rows(t: Expr) -> Axis | None:
    return t.sig[0] if t.sig and len(t.sig) == 1 else None


def place(e: Expr, table: Expr, rows: Axis | None = None) -> Expr:
    """A row's expression, put in a table's rows (and its columns read from the table)."""
    target = rows or _rows(table)
    if target is None:
        return e
    return resig(e, {ROW: target}, {ROWVAR: table})


def column(table: Expr, name: str) -> Expr:
    rows = _rows(table)
    known = dict(table.columns or ())
    s = known.get(name)
    sig = (rows,) if rows is not None else None
    return Field(table, name, sig=sig, type=s.type if s else SERIES, dtype=s.dtype if s else "")


@POLARS.function("polars.col")
def _col(c: Call) -> Expr | None:
    names = [c.literal(a) for a in c.args]
    if not names or c.kwargs or not all(isinstance(n, str) for n in names):
        return None
    row = Var(ROWVAR, sig=(ROW,), type=DF)
    cols = [Field(row, n, sig=(ROW,), type=EXPR) for n in names]
    return cols[0] if len(cols) == 1 else Tup(tuple(cols), type=EXPR)


@POLARS.function("polars.lit")
def _lit(c: Call) -> Expr | None:
    return c.args[0] if len(c.args) == 1 and not c.kwargs else None


@POLARS.function("polars.len")
def _len(c: Call) -> Expr | None:
    if c.args or c.kwargs:
        return None
    return Aggregate("count", (ROW,), Const(1, sig=(ROW,), type=EXPR), sig=(), type="int")


@POLARS.function("polars.Series")
def _series(c: Call) -> Expr | None:
    """pl.Series(values) or pl.Series(name, values): the values."""
    if c.kwargs or not 1 <= len(c.args) <= 2:
        return None
    v = c.args[-1]
    return replace(v, type=SERIES) if v.sig is not None else v


@POLARS.method((SERIES,), "to_numpy", "to_list", "alias")
def _same(c: Call) -> Expr | None:
    return replace(c.args[0], type="numpy.ndarray") if c.name == "to_numpy" else c.args[0]


@POLARS.method((DF,), "__getitem__", "get_column")
def _getitem(c: Call) -> Expr | None:
    t, *keys = c.args
    if len(keys) != 1:
        return None
    k = c.literal(keys[0])
    if isinstance(k, str):
        return column(t, k)
    return None


@POLARS.method((DF,), "filter")
def _filter(c: Call) -> Expr | None:
    t = c.args[0]
    rows = _rows(t)
    if rows is None or c.kwargs or len(c.args) < 2:
        return None
    mask: Expr | None = None
    for a in c.args[1:]:
        m = place(a, t)
        mask = m if mask is None else Apply("and", (mask, m), sig=(rows,), type="", dtype="bool")
    assert mask is not None
    kept = Axis(rows.name, rows.copy, mask, rows.part, rows.letter)
    return Restrict(t, 0, mask=mask, sig=(kept,), type=t.type, columns=t.columns)


def _named(c: Call, t: Expr, first: int = 1) -> list[tuple[str, Expr]] | None:
    """A with_columns's or select's columns: name=e, or e.alias(name), or a column itself."""
    out: list[tuple[str, Expr]] = []
    for a in c.args[first:]:
        items = a.items if isinstance(a, Tup) and a.type == EXPR else (a,)
        for x in items:
            if isinstance(x, Op) and x.name == "alias" and len(x.args) == 2 and isinstance(x.args[1], Text):
                out.append((x.args[1].value or "", x.args[0]))
            elif isinstance(x, Field) and isinstance(x.base, Var) and x.base.name == ROWVAR:
                out.append((x.attr, x))
            elif isinstance(x, Text) and x.value:
                out.append((x.value, Field(Var(ROWVAR, sig=(ROW,), type=DF), x.value, sig=(ROW,), type=EXPR)))
            else:
                return None
    out += list(c.kwargs.items())
    return out


@POLARS.method((DF,), "with_columns", "select")
def _with_columns(c: Call) -> Expr | None:
    t = c.args[0]
    rows = _rows(t)
    cols = _named(c, t)
    if rows is None or cols is None:
        return None
    placed = tuple((n, place(e, t)) for n, e in cols)
    shapes = dict(t.columns or ()) if c.name == "with_columns" else {}
    shapes |= {n: Shape(e.type or SERIES, e.sig, dtype=e.dtype) for n, e in placed}
    return Table(t if c.name == "with_columns" else None, placed, sig=(rows,), type=DF,
                 columns=tuple(shapes.items()))


@POLARS.method((DF,), "group_by", "groupby")
def _group_by(c: Call) -> Expr | None:
    t = c.args[0]
    keys = [c.literal(a) for a in c.args[1:]]
    if not keys or not all(isinstance(k, str) for k in keys) or _rows(t) is None or \
            set(c.kwargs) - {"maintain_order"}:
        return None
    return Op("group_by", (t, *(Text(k) for k in keys)), known=True, qualname="polars.DataFrame.group_by",  # type: ignore
              type="polars.GroupBy", sig=None)


def _groups(g: Expr) -> tuple[Expr, list[str], Axis] | None:
    if not (isinstance(g, Op) and g.name == "group_by" and g.args):
        return None
    t, *keys = g.args
    names = [k.value for k in keys if isinstance(k, Text) and k.value]
    rows = _rows(t)
    if rows is None or len(names) != len(keys):
        return None
    return t, names, Axis("+".join(names))


def _grouped(t: Expr, names: list[str], groups: Axis, aggs: list[tuple[str, Expr]]) -> Expr | None:
    rows = _rows(t)
    assert rows is not None
    cond: Expr | None = None
    for n in names:                                     # the rows of group k: those whose keys are k's
        eq = Apply("eq", (column(t, n), Key(groups, n, sig=(groups,))), sig=(rows, groups), dtype="bool")
        cond = eq if cond is None else Apply("and", (cond, eq), sig=(rows, groups), dtype="bool")
    cols: list[tuple[str, Expr]] = [(n, Key(groups, n, sig=(groups,), type=SERIES)) for n in names]
    for name, e in aggs:
        if not (isinstance(e, Aggregate) and e.over == (ROW,)):
            return None                                 # not one value per group: not read
        body = place(e.body, t)
        if e.op in ("first", "last"):
            return None
        cols.append((name, Aggregate(e.op, (rows,), body, cond, sig=(groups,), type=SERIES)))
    return Table(None, tuple(cols), tuple((n, column(t, n)) for n in names), sig=(groups,), type=DF,
                 columns=tuple((n, Shape(SERIES, (groups,))) for n, _ in cols))


@POLARS.method(("polars.GroupBy",), "agg")
def _agg(c: Call) -> Expr | None:
    got = _groups(c.args[0])
    if got is None:
        return None
    t, names, groups = got
    aggs = _named(c, t)
    if aggs is None:
        return None
    return _grouped(t, names, groups, aggs)


@POLARS.method(("polars.GroupBy",), "len", "count")
def _group_len(c: Call) -> Expr | None:
    got = _groups(c.args[0])
    if got is None or c.kwargs or len(c.args) > 1:
        return None
    t, names, groups = got
    count = Aggregate("count", (ROW,), Const(1, sig=(ROW,), type=EXPR), sig=(), type="int")
    return _grouped(t, names, groups, [("len" if c.name == "len" else "count", count)])


# a row's expression's methods: reductions over the rows, and the rest operators of the row

@POLARS.method((EXPR,), *AGG)
def _expr_agg(c: Call) -> Expr | None:
    x = c.args[0]
    op = AGG[c.name]
    if op is None or c.kwargs or len(c.args) > 1 or x.sig != (ROW,):
        return None
    if op in ("first", "last"):
        return Take(x, (0,), (Const(0 if op == "first" else -1, sig=(), type="int"),), sig=(), type=EXPR)
    return Aggregate(op, (ROW,), x, sig=(), type=EXPR)


@POLARS.method((EXPR,), "alias")
def _alias(c: Call) -> Expr | None:
    if len(c.args) != 2 or not isinstance(c.args[1], Text):
        return None
    return Op("alias", (c.args[0], c.args[1]), known=True, qualname="polars.Expr.alias", sig=c.args[0].sig, type=EXPR)


@POLARS.method((EXPR,), "is_null", "is_not_null", "fill_null", "replace_strict", "cast", "is_in", "abs", "log",
               "exp", "sqrt", "round")
def _row_op(c: Call) -> Expr | None:
    """An operation of each row's value: elementwise, by its name (what it does to a null, to a type: polars')."""
    x = c.args[0]
    if c.name in ("abs", "log", "exp", "sqrt") and len(c.args) == 1 and not c.kwargs:
        return Apply(c.name, (x,), sig=x.sig, type=EXPR)
    return Op(c.name, tuple(c.args), tuple(c.kwargs.items()), qualname=f"polars.Expr.{c.name}",
              sig=broadcast(*(a.sig for a in c.args)) if all(a.sig is not None for a in c.args) else x.sig,
              type=EXPR, dtype="bool" if c.name.startswith("is_") else "")


__all__ = ["POLARS", "place", "column"]
