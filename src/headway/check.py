"""Checks a spec against itself and against the code.

  structure    every symbol, index and group an op or symbol names is declared; every symbol has at most one
               producer; every value is produced and every non-output is used; the graph has no cycle
  references   every code reference resolves (resolve.py): renamed or deleted code fails the check
  verifiers    the spec's @verify functions: the named ops' equations reproduce the code's results
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field

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
    for s in g.symbols.values():
        if s.role in ("value", "output") and s.key not in producer:
            e.append(f"symbol {s.key}: a {s.role} that no op produces (declare it as an input?)")
        if s.role != "output" and s.key not in used:
            w.append(f"symbol {s.key}: not used by any op")
    # `was` names earlier keys; one this spec still declares would match two objects across versions
    for kind, objects in (("op", g.ops), ("symbol", g.symbols), ("group", g.groups)):
        for k, obj in objects.items():
            for old in obj.was:
                if old in objects:
                    e.append(f"{kind} {k}: was={old!r}, but {old!r} is still declared in this spec")
    for path in g.groups:
        if not any(o.group == path or o.group.startswith(path + "/") for o in g.ops.values()):
            w.append(f"group {path!r}: no ops")
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
             [(f"symbol {s.key}", s.code) for s in g.symbols.values()] + \
             [(f"dataset {d.key}", d.code) for d in getattr(g, "datasets", {}).values()] + \
             [(f"figure {f.key}", f.code) for f in getattr(g, "figures", {}).values()]       # a Suite's
    for owner, refs in owners:
        for ref in refs:
            if ref in report.locations:
                continue
            try:
                report.locations[ref] = resolve(ref)
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
    if run_verifiers:
        verifiers(g, report)
    return report
