"""Checks a spec against itself and against the code.

  structure    every symbol, index and group an op or symbol names is declared; every symbol has at most one
               producer; every value is produced and every non-output is used; the graph has no cycle; a step's
               condition is an expression of symbols
  references   every code reference resolves (resolve.py): renamed or deleted code fails the check
  verifiers    the spec's @verify functions: the named ops' equations reproduce the code's results
"""
from __future__ import annotations

import ast
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from headway.files import bases
from headway.resolve import Location, RefError, resolve
from headway.spec import Headway


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    locations: dict[str, Location] = field(default_factory=dict)       # ref -> location
    verified: dict[str, tuple[bool, str]] = field(default_factory=dict)  # op -> (passed, message)

    @property
    def ok(self) -> bool:
        return not self.errors

    def text(self) -> str:
        lines = [f"error: {e}" for e in self.errors] + [f"warning: {w}" for w in self.warnings]
        passed = sum(ok for ok, _ in self.verified.values())
        lines.append(f"{len(self.locations)} code references resolved; {passed} of {len(self.verified)} verified "
                     f"ops passed; {len(self.errors)} errors, {len(self.warnings)} warnings")
        return "\n".join(lines)


def structure(g: Headway, report: Report) -> None:
    e, w = report.errors, report.warnings
    for s in g.symbols.values():
        for d in (*s.dims, *([s.per] if s.per else [])):
            if d not in g.indices:
                e.append(f"symbol {s.key}: undeclared index {d!r}")
    producer: dict[str, str] = {}
    used: set[str] = set()
    for o in g.ops.values():
        if o.per and o.per not in g.indices:
            e.append(f"op {o.key}: undeclared index {o.per!r}")
        for k in o.inputs + o.outputs:
            if k not in g.symbols:
                e.append(f"op {o.key}: undeclared symbol {k!r}")
        for k in o.outputs:
            if k in producer:
                e.append(f"symbol {k}: produced by both {producer[k]} and {o.key}")
            producer[k] = o.key
            if k in g.symbols and g.symbols[k].role in ("input", "setting", "constant"):
                e.append(f"op {o.key}: produces {k}, declared as {g.symbols[k].role}")
        used.update(o.inputs)
        # a where-entry written exactly as a declared symbol (the same TeX string) is that symbol
        for tex, _ in o.where:
            for k, sym in g.symbols.items():
                if sym.tex != tex:
                    continue
                if k in o.inputs + o.outputs:
                    w.append(f"op {o.key}: where-entry {tex!r} is its {k} already, defined by the symbol")
                else:
                    e.append(f"op {o.key}: where-entry {tex!r} is the symbol {k}; list it as an input or output")
        for code_name, k in o.bind.items():
            if k not in g.symbols:
                e.append(f"op {o.key}: bind {code_name!r} to undeclared symbol {k!r}")
        for code_name, n in o.names.items():
            if "symbol" in n and n["symbol"] not in g.symbols:
                e.append(f"op {o.key}: names {code_name!r} as undeclared symbol {n['symbol']!r}")
    # `was` names earlier keys; one this spec still declares would match two objects across versions
    for kind, objects in (("op", g.ops), ("symbol", g.symbols), ("group", g.groups)):
        for k, obj in objects.items():
            for old in obj.was:
                if old in objects:
                    e.append(f"{kind} {k}: was={old!r}, but {old!r} is still declared in this spec")
    for path in g.groups:
        if not any(o.group == path or o.group.startswith(path + "/") for o in g.ops.values()):
            w.append(f"group {path!r}: no ops")
    # what a step declares it computes: one of headway.clean's kinds, of a code reference
    from headway.clean import KINDS
    for o in g.ops.values():
        for r in o.relations:
            if r.get("kind") not in KINDS:
                e.append(f"op {o.key}: relation kind {r.get('kind')!r} is not one of {', '.join(KINDS)}")
            elif not r.get("target"):
                e.append(f"op {o.key}: its {r['kind']} declaration names no function")
    # a step's condition: an expression of symbols
    for o in g.ops.values():
        if not o.condition:
            continue
        try:
            names = {n.id for n in ast.walk(ast.parse(o.condition, mode="eval")) if isinstance(n, ast.Name)}
        except SyntaxError:
            e.append(f"op {o.key}: condition {o.condition!r} is not an expression")
            continue
        used.update(names & set(g.symbols))             # a symbol deciding whether a step is done is used
        unknown = sorted(names - set(g.symbols))
        if unknown:
            w.append(f"op {o.key}: condition {o.condition!r} names {', '.join(unknown)}, not symbols: the viewer cannot "
                     "say what they are")
    for s in g.symbols.values():
        if s.role in ("value", "output") and s.key not in producer:
            e.append(f"symbol {s.key}: a {s.role} that no op produces (declare it as an input?)")
        if s.role != "output" and s.key not in used:
            w.append(f"symbol {s.key}: not used by any op")
    # cycles, over ops
    after = {o.key: {producer[s] for s in o.inputs if s in producer} for o in g.ops.values()}
    state: dict[str, int] = {}

    def visit(k: str, trail: list[str]) -> None:
        if state.get(k) == 1:
            e.append("cycle: " + " -> ".join(trail[trail.index(k):] + [k]))
        if state.get(k):
            return
        state[k] = 1
        for p in after[k]:
            visit(p, trail + [k])
        state[k] = 2
    for k in after:
        visit(k, [])


def references(g: Headway, report: Report) -> None:
    owners = [(f"op {o.key}", o.code) for o in g.ops.values()] + \
             [(f"op {o.key}: its {r.get('kind')} declaration", [x for x in (r.get("target"), *r.get("subject_to", ()))
                                                               if x]) for o in g.ops.values() for r in o.relations] + \
             [(f"symbol {s.key}", s.code) for s in g.symbols.values()] + \
             [(f"dataset {d.key}", d.code) for d in getattr(g, "datasets", {}).values()] + \
             [(f"figure {f.key}", f.code) for f in getattr(g, "figures", {}).values()]       # a Suite's
    files_ = [d.file for d in g.declared.values() if d.function == "<module>"] or [d.file for d in g.declared.values()]
    where = bases(max(set(files_), key=files_.count) if files_ else None)      # for references to files by path
    for owner, refs in owners:
        for ref in refs:
            if ref in report.locations:
                continue
            try:
                report.locations[ref] = resolve(ref, where)
            except RefError as err:
                report.errors.append(f"{owner}: {err}")
    for o in g.ops.values():
        if not o.code:
            report.warnings.append(f"op {o.key}: no code reference")
    for s in g.symbols.values():
        if s.role == "constant" and not s.code:
            report.warnings.append(f"constant {s.key}: no code reference to where it is set")


def verifiers(g: Headway, report: Report) -> None:
    for ops, fn in g.verifiers:
        for k in ops:
            if k not in g.ops:
                report.errors.append(f"verifier {fn.__name__}: undeclared op {k!r}")
        try:
            fn()
            result = (True, f"{fn.__name__} passed")
        except Exception:
            result = (False, f"{fn.__name__} failed:\n{traceback.format_exc(limit=3)}")
            report.errors.append(f"verifier {fn.__name__} ({', '.join(ops)}) failed")
        for k in ops:
            if k in report.verified and not report.verified[k][0]:
                continue                                        # a failure stays a failure
            report.verified[k] = result


def check(g: Headway, run_verifiers: bool = True) -> Report:
    from headway.bench import Suite, check_suite
    report = Report()
    structure(g, report)
    references(g, report)
    if isinstance(g, Suite):
        check_suite(g, report.errors, report.warnings)
    if g.hints:
        shapes_stated(g, report)
    if run_verifiers:
        verifiers(g, report)
    return report


def shapes_stated(g: Headway, report: Report) -> None:
    """With hints: each step's bound values (bind=, names= given symbols) against the shapes the code's type hints
    state for them (jaxtyping, shapes.py): as many axes as the symbol's dims, and each of the function's axis names
    one index of the spec within the step. A warning: the hints are not checked by Python."""
    from headway.draft import Drafter, DraftError, Package
    drafters: dict[str, Drafter | None] = {}
    for o in g.ops.values():
        loc = report.locations.get(o.code[0]) if o.code else None
        module, _, qual = o.code[0].partition(":") if o.code else ("", "", "")
        if loc is None or loc.lang != "python" or not qual or "#" in qual:
            continue
        top = module.split(".")[0]
        if top not in drafters:
            folder = Path(loc.file).parent
            while (folder.parent / "__init__.py").exists():
                folder = folder.parent
            try:
                drafters[top] = Drafter(Package(top, [folder.parent]), hints=True, infer=False)
            except DraftError:
                drafters[top] = None
        d = drafters[top]
        m = d.package.module(module) if d else None
        cls, _, name = qual.rpartition(".")
        owner = m.defs.get(cls) if m and cls else None
        node = next((n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name), None) \
            if isinstance(owner, ast.ClassDef) else m.defs.get(name) if m else None
        if d is None or m is None or not isinstance(node, ast.FunctionDef):
            continue
        annotations = {a.arg: a.annotation for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}
        returned = d.shape_items(m, node.returns)
        named = {**{c: k for c, k in o.bind.items()}, **{c: n["symbol"] for c, n in o.names.items() if "symbol" in n}}
        seen: dict[str, tuple[str, str]] = {}
        for code_name, key in named.items():
            if key not in g.symbols:
                continue
            if code_name == "return":
                axes = d.shape(m, node.returns)
            elif code_name.startswith("return.") and code_name[7:].isdigit():
                axes = returned[int(code_name[7:])] if returned and int(code_name[7:]) < len(returned) else None
            else:
                axes = d.shape(m, annotations.get(code_name))
            if axes is None:
                continue
            dims = list(g.symbols[key].dims)
            if not dims:                        # a scalar, or dims= not given yet (a draft leaves what it cannot tell)
                report.warnings.append(f"op {o.key}: {code_name} is annotated with axes ({' '.join(axes)}) in "
                                       f"{o.code[0]}, but {key} has no dims= (a scalar, or not given yet)")
                continue
            if len(dims) != len(axes):
                report.warnings.append(f"op {o.key}: {code_name} is annotated with {len(axes)} axes "
                                       f"({' '.join(axes)}) in {o.code[0]}, but {key} has {len(dims)} (dims={dims})")
                continue
            for axis, index in zip(axes, dims):
                if axis in seen and seen[axis][0] != index:
                    report.warnings.append(f"op {o.key}: the axis {axis} of {o.code[0]} is {seen[axis][0]} in "
                                           f"{seen[axis][1]} but {index} in {code_name}")
                seen.setdefault(axis, (index, code_name))
