"""What a trace says about the code it traced, beyond its values:

  paths(trace, function)    the distinct paths its calls took (the branches, each line to the line it went to),
                            how many calls took each, and how many times its loops ran (survey §6)
  index_sets(trace)         its arrays' axes grouped into index sets, by their sizes over the calls: two axes are
                            one set if their sizes agree on every call that has both, and the sets a size alone
                            cannot separate are said to be ambiguous (survey §7)
"""
from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from headway.tracing.record import Call, Trace, Value


# ---- paths ---------------------------------------------------------------------------------------------------

@dataclass
class Path_:
    branches: frozenset[str]            # "line>line" taken at least once
    calls: int = 0
    loops: Counter[str] = field(default_factory=Counter)    # each branch's times taken, over the calls
    example: int | None = None          # a call id that took it


def paths(trace: Trace, function: str) -> list[Path_]:
    calls = trace.calls_of(function)
    out: dict[frozenset[str], Path_] = {}
    for c in calls:
        key = frozenset({f"raised {c.raised}"}) if c.raised else frozenset(c.branches)
        p = out.setdefault(key, Path_(key))
        p.calls += 1
        for b, n in c.branches.items():
            p.loops[b] += n
        if p.example is None:
            p.example = c.id
    f = next((f for k, f in trace.functions.items() if k == function or k.split(":")[-1] == function
              or k.endswith(":" + function)), None)
    if f is not None:                                   # calls past the cap on stored calls: counted by path
        for key_text, n in f.more_paths.items():
            key = frozenset({key_text}) if key_text.startswith("(") else frozenset(b for b in key_text.split("|")
                                                                                      if b)
            p = out.setdefault(key, Path_(key))
            p.calls += n
        # their loop counts: shared out over the paths in proportion is not known; added to the paths that
        # have the branch
        for b, n in f.more_branches.items():
            having = [p for p in out.values() if b in p.branches]
            if having:
                having[0].loops[b] += n
    return sorted(out.values(), key=lambda p: -p.calls)


def _back(branch: str) -> bool:
    """A branch to an earlier line: a loop going round again (its count is the loop's iterations), not a choice."""
    a, _, b = branch.partition(">")
    return a.isdigit() and b.isdigit() and int(b) < int(a)


def describe_paths(trace: Trace, function: str) -> str:
    ps = paths(trace, function)
    if not ps:
        return f"{function}: no calls"
    total = sum(p.calls for p in ps)
    common = ps[0].branches
    source = _source(trace, function)
    lines = [f"{function}: {total} calls, {len(ps)} paths"]
    for i, p in enumerate(ps):
        differs = sorted(p.branches ^ common, key=_line_of) if i else []
        head = "the most common path" if not i else "differs at " + "; ".join(
            f"{'took' if b in p.branches else 'not'} {_branch_text(b, source)}" for b in differs[:4])
        repeated = [(b, n / p.calls) for b, n in p.loops.items() if n > p.calls]   # taken more than once a call
        loops = ", ".join(f"line {_line_of(b)}: {k:.3g} times a call" for b, k in
                          sorted(repeated, key=lambda bk: _line_of(bk[0]))[:3])
        lines.append(f"  {p.calls:6d} ({p.calls / total:.1%})  {head}" + (f"  [loops {loops}]" if loops else ""))
    return "\n".join(lines)


def _line_of(branch: str) -> int:
    a = branch.split(">")[0]
    return int(a) if a.isdigit() else 0


def _source(trace: Trace, function: str) -> dict[int, str]:
    f = next((f for k, f in trace.functions.items() if k == function or k.endswith(":" + function)
              or k.split(":")[-1] == function), None)
    if f is None or not Path(f.file).exists():
        return {}
    return dict(enumerate(Path(f.file).read_text().splitlines(), 1))


def _branch_text(branch: str, source: dict[int, str]) -> str:
    a, _, b = branch.partition(">")
    if not (a.isdigit() and b.isdigit()):
        return branch
    text = source.get(int(a), "").strip()
    return f"line {a} ({text[:60]}) to {b}" if text else f"line {a} to {b}"


# ---- index sets ----------------------------------------------------------------------------------------------

@dataclass
class Slot:
    """One axis of one parameter (or of the result) of one function."""
    function: str
    name: str                           # the parameter, or "return", or "return.0" for a tuple's first item
    axis: int

    def __str__(self) -> str:
        return f"{self.function.split(':')[-1]}.{self.name}[{self.axis}]"

    def key(self) -> tuple[str, str, int]:
        return (self.function, self.name, self.axis)


@dataclass
class IndexSet:
    name: str
    slots: list[Slot]
    sizes: list[int]                    # the distinct sizes it had
    ambiguous: list[str] = field(default_factory=list)  # other sets it could not be told from (same sizes)


def _shapes(c: Call) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}

    def add(name: str, v: Value | None, depth: int = 0) -> None:
        if v is None:
            return
        if v.kind == "array" and v.shape is not None:
            out[name] = v.shape
        elif v.kind == "seq" and v.items is not None and depth < 1:
            for i, item in enumerate(v.items):
                add(f"{name}.{i}", item, depth + 1)
        elif v.kind == "table" and v.shape:
            out[name] = v.shape
    for k, v in c.args.items():
        if k != "self":
            add(k, v)
    if c.result is not None:
        add("return", c.result)
    return out


def index_sets(trace: Trace, functions: list[str] | None = None) -> list[IndexSet]:
    """Axes grouped by the sizes they had together. Within a call, axes of equal size are candidates for one set;
    over the calls, two axes stay together only if their sizes agree every time both are present. A function's
    calls are what link its axes, and a value passed from one function to another links theirs: the same size
    sequence over the run (call by call) is not needed, only agreement where both were seen."""
    wanted = set(functions or ())
    slots: dict[tuple[str, str, int], Slot] = {}
    seen: dict[tuple[str, str, int], list[tuple[int, int]]] = defaultdict(list)    # (call id, size)
    per_call: dict[int, dict[tuple[str, str, int], int]] = {}
    for c in trace.calls:
        short = c.function.split(":")[-1]
        if wanted and c.function not in wanted and short not in wanted:
            continue
        here: dict[tuple[str, str, int], int] = {}
        for name, shape in _shapes(c).items():
            for ax, n in enumerate(shape):
                if n < 1:
                    continue
                k = (c.function, name, ax)
                slots.setdefault(k, Slot(*k))
                seen[k].append((c.id, n))
                here[k] = n
        per_call[c.id] = here
    # within one function, two slots are linked if equal in every call where both appear, and both appear
    # together at least once
    parent = {k: k for k in slots}

    def find(k: tuple[str, str, int]) -> tuple[str, str, int]:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    by_function: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for k in slots:
        by_function[k[0]].append(k)
    for ks in by_function.values():
        for i, a in enumerate(ks):
            sa = dict(seen[a])
            for b in ks[i + 1:]:
                sb = dict(seen[b])
                common = sa.keys() & sb.keys()
                if common and all(sa[c] == sb[c] for c in common):
                    parent[find(a)] = find(b)
    # across functions: a callee's slot and its caller's slot are linked when the callee's call has a parent call
    # with an equal size there and the parameter names agree, or the caller passes a value of that exact shape
    children: dict[int, list[int]] = defaultdict(list)
    for c in trace.calls:
        if c.parent is not None:
            children[c.parent].append(c.id)
    link: Counter[tuple[tuple[str, str, int], tuple[str, str, int]]] = Counter()
    clash: Counter[tuple[tuple[str, str, int], tuple[str, str, int]]] = Counter()
    for pid, kids in children.items():
        outer = per_call.get(pid, {})
        for kid in kids:
            inner = per_call.get(kid, {})
            for a, na in inner.items():
                for b, nb in outer.items():
                    if a[1] == b[1] or a[1].split(".")[0] == b[1].split(".")[0]:
                        (link if na == nb else clash)[(a, b)] += 1
    for (a, b), n in link.items():
        if not clash[(a, b)]:
            parent[find(a)] = find(b)
    # across functions: the same parameter name and axis, with the same sizes over the run (two sizes or more, so
    # that agreement is not one size by chance)
    by_name: dict[tuple[str, int], list[tuple[str, str, int]]] = defaultdict(list)
    for k in slots:
        if not k[1].startswith("return"):
            by_name[(k[1], k[2])].append(k)
    for ks in by_name.values():
        for a in ks[1:]:
            sizes_a, sizes_b = {n for _, n in seen[a]}, {n for _, n in seen[ks[0]]}
            if len(sizes_a) > 1 and sizes_a == sizes_b:
                parent[find(a)] = find(ks[0])
    groups: dict[tuple[str, str, int], list[tuple[str, str, int]]] = defaultdict(list)
    for k in slots:
        if any(n > 1 for _, n in seen[k]):              # an axis of length 1 every time: no set (broadcasting)
            groups[find(k)].append(k)
    sets = []
    for members in groups.values():
        sizes = sorted({n for m in members for _, n in seen[m]})
        sets.append(IndexSet("", [slots[m] for m in members], sizes))
    # sets that could not be told apart: the same single size everywhere they were seen
    for s in sets:
        s.ambiguous = [str(i) for i, o in enumerate(sets) if o is not s and len(s.sizes) == 1 and o.sizes == s.sizes]
    _name(sets)
    return sorted(sets, key=lambda s: (-len(s.slots), s.name))


def _name(sets: list[IndexSet]) -> None:
    """A first name for each set, from its members: the most common 1-d parameter among them (size_factors:
    'size_factors'), else its most common member; to be renamed by whoever reads the code (survey §7)."""
    used: set[str] = set()
    for s in sets:
        named = [m for m in s.slots if not m.name.startswith("return")]
        counts = Counter(m.name.split(".")[0] for m in named)
        name = counts.most_common(1)[0][0] if counts else "result"
        base, k = re.sub(r"\W", "_", name), 2
        name = base
        while name in used:
            name, k = f"{base}{k}", k + 1
        used.add(name)
        s.name = name


def describe_sets(sets: list[IndexSet], function: str | None = None) -> str:
    out = []
    for s in sets:
        members = [m for m in s.slots if function is None or m.function.split(":")[-1] == function
                   or m.function == function]
        if not members:
            continue
        sizes = ", ".join(map(str, s.sizes[:6])) + (" ..." if len(s.sizes) > 6 else "")
        tag = "  (one size only: not told apart by size)" if len(s.sizes) == 1 else ""
        out.append(f"{s.name}: sizes {sizes}{tag}\n    " + ", ".join(sorted(map(str, members))[:12])
                   + (f" ... {len(members) - 12} more" if len(members) > 12 else ""))
    return "\n".join(out)


def signatures(sets: list[IndexSet], function: str) -> dict[str, tuple[str, ...]]:
    """Each parameter's axes, as the names of their index sets, for one function (as headway reads jaxtyping's
    annotations): {"counts": ("samples", "genes")}. An axis no set holds (length 1, or never an array) is "1"."""
    axes: dict[str, dict[int, str]] = defaultdict(dict)
    for s in sets:
        for m in s.slots:
            if m.function == function or m.function.split(":")[-1] == function.split(":")[-1]:
                axes[m.name][m.axis] = s.name
    out = {}
    for name, d in axes.items():
        if "." in name:
            continue
        n = max(d) + 1
        out[name] = tuple(d.get(i, "1") for i in range(n))
    return out


def renamed(sets: list[IndexSet], names: dict[str, str]) -> list[IndexSet]:
    for s in sets:
        s.name = names.get(s.name, s.name)
    return sets


__all__ = ["paths", "describe_paths", "index_sets", "describe_sets", "signatures", "renamed", "ast"]


# ---- configuration and environment ----------------------------------------------------------------------------

def options(trace: Trace, function: str) -> dict[str, Counter[str]]:
    """The values a function's settings had over its calls: each parameter given a number, a string, a boolean or
    None (its values, counted), or an object (its type, and its plain attributes' values)."""
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for c in trace.calls_of(function):
        for k, v in c.args.items():
            if v.kind in ("scalar", "str", "none"):
                out[k][repr(v.value)] += 1
            elif v.kind in ("object", "record"):
                attrs = v.value if isinstance(v.value, dict) else {}
                plain = {a: x for a, x in attrs.items() if isinstance(x, (bool, int, float, str)) and
                         not str(x).startswith("<")}
                out[k][v.type + (f" {plain}" if plain and len(str(plain)) < 200 else "")] += 1
    return dict(out)


def describe_options(trace: Trace, function: str) -> str:
    ops = options(trace, function)
    lines = [f"{function}:"]
    for k, counts in ops.items():
        shown = ", ".join(f"{v} ({n})" for v, n in counts.most_common(6))
        more = f", ... {len(counts) - 6} more values" if len(counts) > 6 else ""
        lines.append(f"  {k}: {shown}{more}")
    return "\n".join(lines)


def describe_environment(trace: Trace) -> str:
    e = trace.environment
    used = set(e.get("modules", []))
    dists = e.get("distributions", {})
    packages = {k: v for k, v in dists.items() if k.replace("-", "_") in used or k in used}
    lines = [f"Python {e.get('python')} ({e.get('implementation')}) on {e.get('platform')}",
             "settings: " + (", ".join(f"{k}={v}" for k, v in e.get("variables", {}).items()) or "none set"),
             f"numpy errors: {e.get('numpy_errors')}",
             f"packages the run imported ({len(packages)} of {len(dists)} installed): "
             + ", ".join(f"{k} {v}" for k, v in sorted(packages.items()))]
    if e.get("changed"):
        lines.append("changed by the run: " + ", ".join(f"{k} ({v['before']} -> {v['after']})"
                                                       for k, v in e["changed"].items()))
    return "\n".join(lines)
