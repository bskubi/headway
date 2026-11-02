r"""Python code -> KaTeX: what a step's code computes, written as math. Nothing is run.

  counts / factors[:, None]        \frac{\mathit{counts}}{\mathit{factors}}
  np.log1p(np.maximum(y, 0))       \log\left(1 + \max\left(y, 0\right)\right)
  z - z.mean(0)                    z - \operatorname{mean}_{i} z           (with z named as a symbol of dims (i, g))

This is the core's front door (headway/core, docs/core.md): the code is read into the calculus (core/lower.py, with
the libraries' tables, core/libs), folded as a paper would write it (core/fold.py) and printed (core/printer.py).
Names: a code name can be given TeX, or a symbol's TeX and its indices (`names`). Unknown functions are written as
\operatorname{name}(...), so the result is always the code, never a guess at what it means.
"""
from __future__ import annotations

import ast
import textwrap
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from headway.core.fold import fold_body, reused
from headway.core.lower import OUTPUT_ONLY, Lowerer, Module, Unsupported, _output_only
from headway.core.printer import (ASSUMING, ATOM, INDEX_LETTERS, Name, Printer, T, function_tex, ident, indexed,
                                  neighbours, notation, scripted, text)
from headway.core.terms import (NEW, Axis, Const, Expr, Let, Op, Program, Shape, Stmt, Var, stmt_exprs, walk,
                                walk_stmts)

MODULES = {"np", "numpy", "math", "scipy", "sp"}
SAME = {"asarray", "array", "asanyarray", "ascontiguousarray", "copy", "astype", "float32", "float64", "float_",
        "view", "to_numpy", "float", "list", "tuple"}       # conversions that keep the values and their axes
Dims = tuple[str | None, ...] | None

# the names other modules use
_ident, _function, _scripted, _text, _notation = ident, function_tex, scripted, text, notation
__all__ = ["Name", "transcribe", "returned", "condition", "imports", "ASSUMING", "INDEX_LETTERS", "MODULES", "SAME",
           "OUTPUT_ONLY", "_output_only", "neighbours", "Unsupported"]


def _indexed_tex(base: str, dims: tuple[str | None, ...]) -> str:
    """A symbol written with its indices: x_{ig}; one with a subscript of its own takes them after it, p_{0,ig}."""
    return indexed(base, "".join(d for d in dims if d))


def _given(names: dict[str, Name]) -> dict[str, Shape]:
    """The shapes the names a spec gives have: their indices, each an index set the printer writes as given."""
    out = {}
    for k, n in names.items():
        if n.dims is None or k.startswith("return") or k.endswith("="):
            continue
        sig = tuple(NEW if d == "1" else Axis(None) if d is None else Axis(d, letter=d if len(d) == 1 else None)
                    for d in n.dims)
        one = Shape("numpy.ndarray" if sig else "float", sig)
        out[k] = Shape("list", None, elem=one) if n.per else one
    return out


def _module(tree: ast.Module, file: str | None, imported: set[str] | frozenset[str]) -> Module:
    from headway.core.lower import DEFAULT_IMPORTS
    m = Module.read(file) if file else Module.of(tree)
    for n in imported:                                  # a module the file imports (embed.Embedding(a))
        if n not in DEFAULT_IMPORTS:
            m.imports.setdefault(n, n)
    return m


def _program(tree: ast.Module, low: Lowerer) -> tuple[Program, ast.FunctionDef | None]:
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef):
        fn = tree.body[0]
        return low.program(fn), fn
    return Program(None, [], low.block(tree.body)), None


def _in_context(file: str, first: int, last: int, module: Module, names: dict[str, Name], hints: bool,
                kinds: Callable[[int, int, str], str | None] | None,
                helpers: dict[str, ast.FunctionDef]) -> tuple[Program, Lowerer, set[str]] | None:
    """A block of lines of a function, read as part of the function (so that its values have the shapes the
    function's annotations and its earlier lines give them): the block's statements, and the lowerer that read
    them. None if the lines are not a function's statements."""
    try:
        tree = ast.parse(Path(file).read_text())
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and n.lineno < first and last <= (n.end_lineno or n.lineno)
               and not any(isinstance(m, ast.FunctionDef) and m is not n and m.lineno < first
                           and last <= (m.end_lineno or m.lineno) for m in ast.walk(n))), None)
    if fn is None:
        return None
    low = Lowerer(module, _given(names), hints=hints, kinds=kinds if hints else None, offset=0,
                  indent=0, inline=helpers, named=set(names))
    whole = low.program(fn)

    def pick(body: list[Stmt]) -> list[Stmt]:
        chosen = [st for st in body if first <= st.line <= last]
        if chosen:
            return chosen
        for st in body:                                 # the block inside a loop or a branch: look there
            from headway.core.terms import stmt_body
            inner = pick(stmt_body(st))
            if inner:
                return inner
        return []
    chosen = pick(whole.body)
    if not chosen:
        return None
    before = {p for p, _, _ in whole.params} | {n for st in walk_stmts(whole.body) if isinstance(st, Let)
                                               and st.line < first for n in st.targets}
    return Program(None, [], chosen), low, before


def _aliases(body: list[Stmt], given: set[str], params: set[str]) -> dict[str, str]:
    """x = y (or a conversion of y that keeps its values), x assigned only there: x another name for y, written as
    y, with no line of its own; never when either is changed (two names for one value would hide it)."""
    stores = Counter(n for st in walk_stmts(body) if isinstance(st, Let) for n in st.targets)
    from headway.core.fold import changed
    ch = changed(body)
    out = {}
    for st in body:
        if isinstance(st, Let) and len(st.targets) == 1 and isinstance(st.value, Var) and not st.again:
            x, y = st.targets[0], st.value.name
            if x not in given and stores[x] == 1 and stores[y] == 0 and x not in ch and y not in ch and x != y:
                out[x] = y
    return out


def transcribe(source: str, names: dict[str, Name] | None = None, max_lines: int = 40,
               imported: set[str] | frozenset[str] = frozenset(), short: bool = False,
               first_line: int = 1, hints: bool = False,
               kinds: Callable[[int, int, str], str | None] | None = None,
               axes: Callable[[ast.expr], list[str] | None] | None = None,
               indices: dict[str, str] | None = None, inline: dict[str, str] | None = None,
               flush: bool = False, define: list[tuple[str, int]] | None = None,
               where: list[str] | None = None, before: list[str] | None = None, file: str | None = None) -> str | None:
    """The TeX of a function's body or a block of statements (as a Location's source), or None if it does not
    parse or holds nothing to write. Straight-line code is one equation (core/fold.py); other code, line by line.
    short: the code's other names as short symbols too, each the first free one of its near neighbours.
    hints: the code's type hints taken as assumptions: a number's annotation makes it a scalar; jaxtyping's named
    axes (Float[np.ndarray, "cells genes"]) give it indices, so that a sum over an axis is a sum over its index.
    kinds(line, column, name): what a type checker infers each name is ("scalar"; typeinfo.py). indices: {axis
    name: its index's TeX}. inline: {function name: its source}, written where it is called. flush: every line flush
    left. define: [(source, first line)] of functions the code calls, defined under it ("where ..."), each written
    with what its first call gives for its parameters. where: lines (TeX) that say what the code's symbols are.
    before: lines (TeX) to put first, in the same block. file: the code's file (for its imports). (axes: unused,
    the core reads jaxtyping's annotations itself.)"""
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None
    names = dict(names or {})
    given = set(names)
    module = _module(tree, file, imported)
    helpers = {}
    for k, src in (inline or {}).items():
        try:
            f = ast.parse(textwrap.dedent(src)).body[0]
        except (SyntaxError, IndexError):
            continue
        if isinstance(f, ast.FunctionDef):
            helpers[k] = f
    indent = min((len(line) - len(line.lstrip()) for line in source.splitlines() if line.strip()), default=0)
    low = Lowerer(module, _given(names), hints=hints, kinds=kinds if hints else None, offset=first_line - 1,
                  indent=indent, inline=helpers, named=set(names))
    prog, fn = _program(tree, low)
    outer: set[str] = set()                             # (a block's) variables with values before it
    if fn is None and file:                             # a block of a function's lines: read in the function
        last_line = first_line + len(textwrap.dedent(source).splitlines()) - 1
        inside = _in_context(file, first_line, last_line, module, names, hints, kinds, helpers)
        if inside is not None:
            prog, low, outer = inside
    used = {n.id if isinstance(n, ast.Name) else n.arg for n in ast.walk(tree) if isinstance(n, (ast.Name, ast.arg))}
    p = Printer(names, short=short, indices=dict(indices or {}), reserved={u for u in used if len(u) == 1})
    p.defined |= outer
    if short:                                           # names that are notation already (p, x0, mu) keep it
        for node in ast.walk(tree):
            n = node.id if isinstance(node, ast.Name) else node.arg if isinstance(node, ast.arg) else None
            if n and n not in p.names and notation(n) and p.free(own := neighbours(n, False)[0]):
                p.names[n] = Name(own)
                p.taken.add(own)
    if fn is not None:
        p.defined |= {a for a, _, _ in prog.params}
        # the variable a function returns, assigned in it: written as the symbol of what it returns
        last = prog.body[-1] if prog.body else None
        from headway.core.terms import Return
        if isinstance(last, Return) and isinstance(last.value, Var) and "return" in names \
                and last.value.name not in names and last.value.name not in p.defined \
                and any(isinstance(st, Let) and last.value.name in st.targets for st in walk_stmts(prog.body)):
            p.names[last.value.name] = names["return"]
            p.fixed.add(last.value.name)
    p.prepare([a for st in walk_stmts(prog.body) for x in stmt_exprs(st) for e in walk(x) for a in (e.sig or ())]
              + [a for _, s, _ in prog.params for a in (s or ())])
    # kept as names, each with its own line: the names given, those declared with their axes, those read more
    # than once (not the short symbols: the switch between them and code names changes no line)
    keep = given | low.declared | reused(prog.body)
    body = fold_body(prog.body, keep) or prog.body
    p.aliases = _aliases(body, given, {a for a, _, _ in prog.params})
    lines = p.block(body, prog if fn is not None else None, 0)
    if not lines:
        return None
    # the conditions the code states at its top level: a block of their own, under the equations
    conditions = [line.removeprefix(ASSUMING) for line in lines if line.startswith(ASSUMING)]
    lines = [line for line in lines if not line.startswith(ASSUMING)]
    lines = lines[:max_lines] + ([r"&\;\vdots"] if len(lines) > max_lines else [])
    if not lines:
        return None
    held = [r"\text{provided that } " + conditions[0], *(r"\text{and } " + c for c in conditions[1:])] \
        if conditions else []
    defined = _definitions(define or [], prog, p, module, hints, short)
    said = list(where or [])
    first_rows = list(before or [])
    if not held and not defined and not said and not first_rows:
        if len(lines) == 1:
            return lines[0].replace("&", "", 1)
        if not flush and not any(line.startswith((r"\quad", r"&\text{", r"&\textbf{")) for line in lines):
            return r"\begin{aligned}" + r" \\ ".join(lines) + r"\end{aligned}"
    # one flat block, every row flush left (none nested in another: each would add its own margin): what comes
    # first, the code's lines (loops and ifs indented by depth, as code is), what it is provided, what it uses
    rows = [*first_rows, *(line.replace("&", "", 1) for line in lines), *held]
    gaps = [r" \\[0.6em] "] * len(first_rows) + [r" \\ "] * len(lines) + [r" \\ "] * len(held)
    if held:
        gaps[len(first_rows) + len(lines)] = r" \\[0.6em] "
    if said or defined:
        first, *rest = [*said, *defined]
        rows += [r"\text{where } " + first, *rest]
        gaps += [r" \\[0.9em] "] + [r" \\[0.4em] "] * len(rest)
    out = r"\displaystyle " + rows[0]
    for gap, row in zip(gaps[1:], rows[1:]):
        out += gap + r"\displaystyle " + row
    return r"\begin{array}{l}" + out + r"\end{array}"


def _definitions(define: list[tuple[str, int]], prog: Program, p: Printer, module: Module, hints: bool,
                 short: bool) -> list[str]:
    """The functions the code calls, defined: each written with the symbols its first call gives (bin_step's k,
    given B, is B), its own other names short symbols as they would be beside the code; a line written already
    (the same q in two definitions) left out."""
    calls = [e for st in walk_stmts(prog.body) for x in stmt_exprs(st) for e in walk(x) if isinstance(e, Op)]
    out: list[str] = []
    seen: set[str] = set()
    for source, first in define:
        try:
            f = ast.parse(textwrap.dedent(source)).body[0]
        except (SyntaxError, IndexError):
            continue
        if not isinstance(f, ast.FunctionDef):
            continue
        call = next((c for c in calls if c.name == f.name), None)
        params = [a.arg for a in (*f.args.posonlyargs, *f.args.args, *f.args.kwonlyargs)]
        args: list[tuple[str, Expr]] = [*zip(params, call.args), *call.kwargs] if call else []
        sub_names: dict[str, Name] = {}
        shapes: dict[str, Shape] = {}
        for name, a in args:
            v = p.expr(a)
            idx = p.index_tex(a.sig)
            sub_names[name] = Name(v.tex if v.prec >= ATOM else r"\left(" + v.tex + r"\right)",
                                   tuple(p.letter(x) for x in a.sig or ()) if idx or a.sig == () else None)
            shapes[name] = a.shape
        sub = Printer(sub_names, short=short, indices=p.indices, taken=set(p.taken), reserved=p.reserved)
        sub.letters = p.letters
        low = Lowerer(module, shapes, hints=hints, offset=first - 1)
        try:
            fprog = low.program(f)
        except Unsupported:
            continue
        sub.defined |= set(params)
        body = fold_body(fprog.body, set(sub_names) | low.declared | reused(fprog.body)) or fprog.body
        for line in sub.block(body, fprog, 0):
            row = line.replace("&", "", 1)
            if row not in seen:
                seen.add(row)
                out.append(row)
    return out


def returned(source: str, names: dict[str, Name] | None = None, short: bool = False) -> str | None:
    """What a function returns, as one expression (its variables written in), or None if it is not one (a line it
    keeps, a loop)."""
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(fn := tree.body[0], ast.FunctionDef):
        return None
    names = dict(names or {})
    low = Lowerer(Module.of(tree), _given(names), hints=False)
    prog = low.program(fn)
    body = fold_body(prog.body, set(names))
    from headway.core.terms import Return
    if not body or len(body) != 1 or not isinstance(body[0], Return) or body[0].value is None:
        return None
    return Printer(names, short=short).expr(body[0].value).tex


def condition(source: str, names: dict[str, Name] | None = None) -> str | None:
    """A step's condition (a Python expression of symbol keys) as math, its truth values explicit: `correct` is
    correct = True, `not correct` correct = False, `a is None` a = None; `and`, `or` as ∧, ∨. None if it does not
    parse."""
    try:
        e = ast.parse(source, mode="eval").body
    except SyntaxError:
        return None
    p = Printer(dict(names or {}))
    low = Lowerer(Module(), hints=False)

    def tex(x: ast.expr) -> str:
        return p.expr(low.expr(x)).tex

    def truth(x: ast.expr) -> str:
        if isinstance(x, ast.BoolOp):
            sym = r"\land" if isinstance(x.op, ast.And) else r"\lor"
            return f" {sym} ".join(r"\left(" + truth(v) + r"\right)" if isinstance(v, ast.BoolOp) else truth(v)
                                   for v in x.values)
        if isinstance(x, (ast.Name, ast.Attribute)):            # a flag: true
            return tex(x) + r" = \text{True}"
        if isinstance(x, ast.UnaryOp) and isinstance(x.op, ast.Not) and isinstance(x.operand, (ast.Name, ast.Attribute)):
            return tex(x.operand) + r" = \text{False}"
        return tex(x)
    return truth(e)


def imports(file: str) -> set[str]:
    """The names a Python file imports at its top level (import x, import x as y, from m import x)."""
    try:
        tree = ast.parse(Path(file).read_text())
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()
    out: set[str] = set()
    for st in tree.body:
        if isinstance(st, (ast.Import, ast.ImportFrom)):
            out.update((a.asname or a.name).split(".")[0] for a in st.names)
    return out


_unused = (Const, T)
