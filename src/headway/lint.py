"""The clean-code linter: where a clean module leaves the clean subset, or where its math stays opaque, each named
by its line, so that an AI agent (or a person) writing clean code can fix it. And how much of it reads as math.

    headway lint clean_methembed.steps

What it reports (read from the code's syntax and its translation into the core's calculus, nothing run):

  error    a statement the calculus cannot say (written "not written: line N")
  error    a value changed in place (x[k] = e, x += e on an array, a call that changes its argument): clean code
           assigns new values, so that each name holds one value
  warning  an operation no library table translates: shown by its name, its meaning opaque in the math
  warning  an array whose index sets are not known (a parameter without a jaxtyping annotation, a value made by an
           operator): its sums are over positions ("axis 0"), not indices
  warning  a function with no declaration of what it computes, among those a step stands for (equals, maximizes, ...)
  note     a loop that carries values (written as an algorithm: fine, if it is the method's)

Coverage: of the expressions the module's functions write, the share written as math (not as an operator, nor as
the code's own indexing).
"""
from __future__ import annotations

import ast
import importlib.util
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from headway.core import libs
from headway.core.fold import changed
from headway.core.lower import LIBRARIES, Lowerer, Module
from headway.core.terms import (Aggregate, Apply, Effect, Expr, For, Item, Let, Op, Program, SetPart, Unwritten,
                                Var, While, stmt_exprs, walk, walk_stmts)


@dataclass
class Finding:
    level: str                      # "error", "warning", "note"
    line: int
    function: str
    message: str


@dataclass
class Lint:
    module: str
    file: str
    findings: list[Finding] = field(default_factory=list)
    math: int = 0                   # expressions written as math
    opaque: int = 0                 # written as an operator or the code's indexing

    @property
    def coverage(self) -> float:
        total = self.math + self.opaque
        return self.math / total if total else 1.0

    def text(self) -> str:
        lines = [f"{self.module} ({self.file}): {self.coverage:.0%} of its expressions written as math"]
        order = {"error": 0, "warning": 1, "note": 2}
        for f in sorted(self.findings, key=lambda f: (order[f.level], f.line)):
            lines.append(f"  {f.level:7} line {f.line:4}  {f.function}: {f.message}")
        counts = Counter(f.level for f in self.findings)
        lines.append(f"{counts['error']} errors, {counts['warning']} warnings, {counts['note']} notes")
        return "\n".join(lines)


def lint(module_name: str, hints: bool = True) -> Lint:  # noqa: C901
    """Lint a module of clean code, by its import name or its file (its source is read, not imported)."""
    if module_name.endswith(".py") and Path(module_name).exists():
        file = module_name
        module_name = Path(module_name).stem
    else:
        spec = importlib.util.find_spec(module_name)
        if spec is None or not spec.origin:
            raise ValueError(f"no module {module_name}")
        file = spec.origin
    module = Module.read(file, module_name)
    tree = ast.parse(Path(file).read_text())
    out = Lint(module_name, file)
    declared = {fn.name for fn in module.functions.values() if _declares(fn)}
    functions: list[tuple[str, ast.FunctionDef, ast.ClassDef | None]] = \
        [(fn.name, fn, None) for fn in tree.body if isinstance(fn, ast.FunctionDef)]
    for cls in (c for c in tree.body if isinstance(c, ast.ClassDef)):
        functions += [(f"{cls.name}.{f.name}", f, cls) for f in cls.body
                      if isinstance(f, ast.FunctionDef) and not f.name.startswith("__")]
    for name, fn, owner in functions:
        low = Lowerer(module, hints=hints)
        prog = low.program(fn, owner)
        _check(out, name, fn, prog)
    del declared
    return out


def _declares(fn: ast.FunctionDef) -> bool:
    return any(isinstance(d, ast.Call) and getattr(d.func, "id", getattr(d.func, "attr", "")) in
               ("equals", "approximates", "maximizes", "minimizes", "solves") for d in fn.decorator_list)


def _check(out: Lint, name: str, fn: ast.FunctionDef, prog: Program) -> None:
    def add(level: str, line: int, message: str) -> None:
        out.findings.append(Finding(level, line, name, message))
    for p, sig, type_ in prog.params:
        if p != "self" and type_ in ("numpy.ndarray", "torch.Tensor", "jax.Array") and sig is None:
            add("warning", fn.lineno, f"parameter {p}: an array whose axes are not named (annotate it with "
                                      f"jaxtyping's Float[np.ndarray, \"...\"]), so its sums are over positions")
        if p != "self" and not type_ and not sig:
            add("note", fn.lineno, f"parameter {p} has no annotation: what it is is not known")
    in_place = changed(prog.body)
    seen_ops: set[tuple[int, str]] = set()
    for st in walk_stmts(prog.body):
        if isinstance(st, Unwritten):
            add("error", st.line, "a statement the math cannot say (it is shown as \"not written\")")
        elif isinstance(st, SetPart):
            add("error", st.line, "a value changed in place (assign a new value to a name instead)")
        elif isinstance(st, Let) and st.inplace:
            add("error", st.line, f"{', '.join(st.targets)} changed in place (x = x + e, not x += e, for arrays)")
        elif isinstance(st, Effect) and not (isinstance(st.value, Op) and st.value.name == "append"):
            add("error", st.line, "a call whose result is not used: it may change what it is given")
        elif isinstance(st, (For, While)) and any(isinstance(b, Let) and b.again for b in walk_stmts(list(st.body))):
            add("note", st.line, "a loop that carries values: written as an algorithm")
        for e in stmt_exprs(st):
            for x in walk(e):
                if isinstance(x, Op):
                    if x.local or x.known or x.qualname.startswith("builtins.") and x.name in ("enumerate", "zip") \
                            or x.qualname == "*":
                        continue
                    out.opaque += 1
                    q = x.qualname or x.name
                    key = (st.line, q)
                    if key not in seen_ops and _library(q):
                        seen_ops.add(key)
                        add("warning", st.line, f"{q}: no table translates it, so its meaning is opaque in the "
                                                "math (written as an operator)")
                elif isinstance(x, Item) and x.base.type in ("numpy.ndarray", "torch.Tensor"):
                    out.opaque += 1
                elif isinstance(x, Aggregate) and x.sig is None and any(type(i).__name__ == "Pos" for i in x.over):
                    out.opaque += 1
                    add("warning", st.line, "a sum over an axis whose index set is not known (written \"axis k\")")
                elif isinstance(x, (Apply, Aggregate)):
                    out.math += 1
    del in_place


def _library(q: str) -> bool:
    """An operation of a library (one a table could translate), not the code's own."""
    return q.split(".")[0] in LIBRARIES or q.startswith(("?.", "numpy.ndarray.", "torch.Tensor."))


__all__ = ["lint", "Lint", "Finding", "libs", "Var", "Expr"]
