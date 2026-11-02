"""Code as a paper would write it: a Program folded into as few equations as it can be (docs/core.md §8's
substitution, and the calculus's Choose and Aggregate).

  - a variable assigned once and read once is written into where it is read (one equation, not a chain);
  - a variable kept (given a name, read more than once, declared with its axes) keeps its own line;
  - an if/else that decides values, or returns early, is cases;
  - a loop that adds up (acc += e: a sum; acc *= e: a product) or fills in a family (out[i] = e, out.append(e)) is
    a sum, a product or a family over the loop's variable.

Anything else (a loop that carries values, a change in place, a statement not written) is left as it is: the printer
writes it line by line, as an algorithm. Folding changes no value: the folded Program is run by the evaluator too
(verify.py), so a fold that did would be seen.
"""
from __future__ import annotations

from collections import Counter

from headway.core.terms import (Aggregate, Apply, Assume, Binder, Build, Cases, Const, Draw, Effect, Expr, For, If,
                                Item, Jump, Let, Op, Program, Range, Return, SetPart, Stmt, Tup, Unwritten, Var, While,
                                bound, free_vars, map_children, stmt_body, stmt_exprs, walk, walk_stmts)


class _NotOne(Exception):
    """The code cannot be one equation."""


def fold(prog: Program, keep: set[str] | None = None) -> Program:
    """The Program as one equation (and the lines of the variables kept), if it can be; else as it is."""
    body = fold_body(prog.body, set(keep or ()) | reused(prog.body))
    return Program(prog.name, prog.params, body if body is not None else prog.body, prog.qualname)


def fold_body(body: list[Stmt], keep: set[str]) -> list[Stmt] | None:
    flow = _Flow(keep)
    try:
        ret, last = flow.block(body, {})
    except _NotOne:
        return None
    out = flow.lines
    if ret is not None:
        return [*out, Return(ret)]
    if last is None or last in keep or last in flow.kept:
        return out or None
    return [*out, Let((last,), flow.final[last])]


def substitute(e: Expr, env: dict[str, Expr]) -> Expr:
    """e with each variable it reads replaced by the expression env gives it (a sum's or a family's own variables
    are not the outer ones)."""
    if not env:
        return e
    if isinstance(e, Var):
        return env.get(e.name, e)
    inner = bound(e)
    sub_env = {k: v for k, v in env.items() if k not in inner} if inner else env
    if isinstance(e, (Aggregate, Build)) and inner:
        # the domains are outside the binders, the body inside
        over = tuple(Binder(i.var, substitute(i.domain, env)) if isinstance(i, Binder) else i for i in e.over)
        body = substitute(e.body, sub_env)
        where = substitute(e.where, sub_env) if e.where is not None else None
        from dataclasses import replace
        return replace(e, over=over, body=body, where=where)
    return map_children(e, lambda c: substitute(c, sub_env))


def reused(body: list[Stmt]) -> set[str]:
    """Variables kept as names, each with its own line: assigned once, by a statement of the block itself (not in a
    loop or branch), never changed in place, and read more than once (f in log(f / (1 - f))). Written into the
    equation, each would be written out again wherever it is read."""
    stores = Counter(n for st in walk_stmts(body) if isinstance(st, Let) for n in st.targets)
    reads = Counter(n for st in walk_stmts(body) for e in stmt_exprs(st) for n in free_vars(e))
    once = set()
    for k, st in enumerate(body):
        if isinstance(st, Let) and len(st.targets) == 1:
            n = st.targets[0]
            before = [v for s in walk_stmts(body[:k]) for e in stmt_exprs(s) for v in free_vars(e)]
            if n not in before and n not in free_vars(st.value):
                once.add(n)
    return {v for v in once - changed(body) if stores[v] == 1 and reads[v] > 1}


def changed(body: list[Stmt]) -> set[str]:
    """The variables a block changes in place: out[k] = e, x.a = e, x += e, out.append(e), f(x, out=x)."""
    out: set[str] = set()
    for st in walk_stmts(body):
        if isinstance(st, Let) and st.inplace:
            out.update(st.targets)
        elif isinstance(st, SetPart):
            base = st.target
            while isinstance(base, Item) or hasattr(base, "base") and not isinstance(base, Var):
                base = base.base  # type: ignore[attr-defined]
            if isinstance(base, Var):
                out.add(base.name)
        elif isinstance(st, Effect) and isinstance(st.value, Op) and st.value.args \
                and isinstance(st.value.args[0], Var) and not st.value.qualname.startswith(("builtins.", "numpy.")):
            out.add(st.value.args[0].name)
    return out


class _Flow:
    def __init__(self, keep: set[str]):
        self.keep = keep
        self.lines: list[Stmt] = []                     # the kept variables' own lines
        self.final: dict[str, Expr] = {}
        self.kept: set[str] = set()                     # assigned together (a, b = f(x)): kept as a line

    def block(self, stmts: list[Stmt] | tuple[Stmt, ...], env: dict[str, Expr],
              named: bool = True) -> tuple[Expr | None, str | None]:
        """Run the statements over env (variable -> its expression): (what they return, if they always do; the
        variable assigned last). named: lines of kept variables may be written (not inside a branch or loop)."""
        last: str | None = None
        stmts = list(stmts)
        for k, st in enumerate(stmts):
            if isinstance(st, Assume):
                if not named:
                    raise _NotOne
                self.lines.append(Assume(substitute(st.test, env), line=st.line))
                continue
            if isinstance(st, Return):
                if st.value is None:
                    raise _NotOne
                self.final = env
                return substitute(st.value, env), last
            if isinstance(st, If):
                ret, ch = self.branch(st, stmts[k + 1:], env, named)
                if ret is not None:
                    return ret, last
                last = ch or last
                continue
            if isinstance(st, For):
                if st.orelse:
                    raise _NotOne
                last = self.loop(st, env) or last
                continue
            if not isinstance(st, Let):
                raise _NotOne                           # a change in place, a call for its effect, a while
            value = substitute(st.value, env)
            if st.inplace:                              # x += e, out=x: no other variable may hold x's value
                self.unshared(st.targets[0], env)
            if len(st.targets) > 1:                     # a, b = f(x): a line of its own
                if not named:
                    raise _NotOne
                for n in st.targets:
                    env.pop(n, None)
                self.lines.append(Let(st.targets, value, st.again, line=st.line))
                self.kept |= set(st.targets)
                last = st.targets[-1]
                continue
            name = st.targets[0]
            if name in self.kept:
                raise _NotOne
            env.pop(name, None)
            if any(isinstance(x, Draw) for x in walk(value)):          # a draw: a line of its own (x ~ D)
                if not named:
                    raise _NotOne
                self.lines.append(Let((name,), value, st.again, line=st.line))
                self.kept.add(name)
                last = name
                continue
            if name in self.keep:
                if not named:
                    raise _NotOne
                # a kept variable keeps its line; changed again, it is written as an update (x <-)
                self.lines.append(Let((name,), value if not st.inplace else substitute(st.value, {
                    k: v for k, v in env.items() if k != name}), st.again, st.inplace, line=st.line))
            else:
                env[name] = value
            last = name
        self.final = env
        return None, last

    def unshared(self, v: str, env: dict[str, Expr]) -> None:
        """Before an in-place change of v: no other variable may hold what v holds (y = x; x += 1: y would change
        too, which substitution cannot show)."""
        held = [e for k, e in env.items() if k != v] + [getattr(line, "value", None) for line in self.lines]
        if any(v in free_vars(e) for e in held if e is not None):
            raise _NotOne

    def branch(self, st: If, rest: list[Stmt], env: dict[str, Expr],
               named: bool) -> tuple[Expr | None, str | None]:
        """An if/else: cases. Both branches returning, or one returning and the code after the if: the returned
        value. Neither returning: each variable a branch changes becomes cases."""
        test = substitute(st.test, env)
        yes, no = dict(env), dict(env)
        r1, _ = self.block(st.body, yes, named=False)
        r2, _ = self.block(st.orelse, no, named=False)
        if r1 is not None and r2 is not None:
            return _cases(test, r1, r2), None
        if r1 is not None or r2 is not None:            # if c: return a, then the rest
            r3, _ = self.block(rest, no if r1 is not None else yes, named)
            if r3 is None:
                raise _NotOne
            return (_cases(test, r1, r3) if r1 is not None else _cases(test, r3, r2)), None  # type: ignore[arg-type]
        last = None
        for v in [*yes, *(v for v in no if v not in yes)]:
            a, b = yes.get(v), no.get(v)
            if a is env.get(v) and b is env.get(v):
                continue
            env[v] = _cases(test, a or Var(v), b or Var(v))
            last = v
        return None, last

    def loop(self, st: For, env: dict[str, Expr]) -> str | None:
        """A loop that adds up or fills in, element by element: sums, products and indexed families."""
        over = substitute(st.domain, env)
        # what the loop assigns is not what it held before (b = b - f(q), q = expit(b): b is the loop's own)
        assigned = {n for b in walk_stmts(list(st.body)) if isinstance(b, Let) for n in b.targets}
        local = {k: v for k, v in env.items() if k not in assigned}
        var = st.var
        for n in var or ():
            local.pop(n, None)
        if isinstance(over, Op) and over.qualname == "builtins.enumerate" and var is not None and len(var) == 2 \
                and len(over.args) == 1:
            # for i, x in enumerate(xs): over i, with x the i-th of xs
            i, x = var
            xs = over.args[0]
            local[x] = Item(xs, Var(i, sig=(), type="int"), sig=xs.elem.sig if xs.elem else None,
                            type=xs.elem.type if xs.elem else "")
            var = (i,)
            over = Range(Const(0, sig=(), type="int"), Apply("len", (xs,), sig=(), type="int"), type="range")
        if var is None or len(var) != 1:
            raise _NotOne
        index = var[0]
        sums: dict[str, tuple[str, Expr]] = {}
        families: dict[str, Expr] = {}
        turn: set[str] = set()                          # variables of one turn of the loop
        appended: set[str] = set()
        for b in st.body:
            if isinstance(b, Let) and len(b.targets) == 1 and b.again and isinstance(b.value, Apply) \
                    and b.value.fn in ("add", "sub", "mul") and len(b.value.args) == 2 \
                    and isinstance(b.value.args[0], Var) and b.value.args[0].name == b.targets[0] \
                    and b.targets[0] not in {*sums, *families, *turn}:
                acc = b.targets[0]
                term = substitute(b.value.args[1], {k: v for k, v in local.items() if k != acc})
                if acc in free_vars(term):              # acc += f(acc), or of a value made from acc: not a sum
                    raise _NotOne
                sums[acc] = (b.value.fn, term)
                continue
            if isinstance(b, SetPart) and isinstance(b.target, Item) and isinstance(b.target.base, Var) \
                    and isinstance(b.target.key, Var) and b.target.key.name == index:          # out[i] = e
                families[b.target.base.name] = substitute(b.value, local)
                continue
            if isinstance(b, Effect) and isinstance(b.value, Op) and b.value.name == "append" \
                    and len(b.value.args) == 2 and isinstance(b.value.args[0], Var) and not b.value.kwargs:
                families[b.value.args[0].name] = substitute(b.value.args[1], local)      # out.append(e)
                appended.add(b.value.args[0].name)
                continue
            if isinstance(b, Let) and len(b.targets) == 1 and b.targets[0] not in sums \
                    and b.targets[0] not in families:
                local[b.targets[0]] = substitute(b.value, local)       # a variable of one turn of the loop
                turn.add(b.targets[0])
                continue
            raise _NotOne
        ch = [*sums, *families]
        if not ch or any(v in self.keep for v in ch):
            raise _NotOne
        for v in sums:
            self.unshared(v, env)
        for v in families:                              # filled whole: an empty list, or an array made for it
            if not _filled_whole(env.get(v), v in appended, over):
                raise _NotOne
        binder = Binder(index, over)
        for v, (op, e) in sums.items():
            whole: Expr = Aggregate("prod" if op == "mul" else "sum", (binder,), e, sig=e.sig, type=e.type)
            start = env.get(v)
            empty = 1 if op == "mul" else 0
            if start is None or isinstance(start, Const) and start.value == empty and not start.sig:
                env[v] = Apply("neg", (whole,), sig=whole.sig, type=whole.type) if op == "sub" else whole
            else:
                env[v] = Apply(op, (start, whole), sig=whole.sig, type=whole.type)
        for v, e in families.items():
            env[v] = Build(e, (binder,), type="list", elem=e.shape)
        return ch[-1]


def _cases(test: Expr, a: Expr, b: Expr) -> Cases:
    from headway.core.terms import broadcast
    return Cases(((test, a),), b, sig=broadcast(a.sig, b.sig, test.sig) if a.sig is not None and b.sig is not None
                 else None, type=a.type if a.type == b.type else "")


def _filled_whole(start: Expr | None, appended: bool, over: Expr) -> bool:
    """Whether a loop fills a family whole: appended to an empty list, or every element of an array made for it
    (out = np.zeros(n); for k in range(n): out[k] = ...)."""
    if appended:
        return isinstance(start, Tup) and start.kind == "list" and not start.items
    if not isinstance(over, Range) or not (isinstance(over.lo, Const) and over.lo.value == 0):
        return False
    if isinstance(start, Op) and start.name in ("zeros", "empty", "ones", "full") and start.args:
        return start.args[0] == over.hi
    if isinstance(start, Const) and start.sig is not None and len(start.sig) == 1:
        from headway.core.terms import Extent
        return isinstance(over.hi, Extent) and over.hi.axis == start.sig[0]
    return False


__all__ = ["fold", "substitute", "reused", "changed", "stmt_body", "Jump", "Unwritten", "While"]
