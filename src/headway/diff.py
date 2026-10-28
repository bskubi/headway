"""Comparing versions of a headway: which objects correspond, and what changed between them.

Versions are compared as built (the JSON `headway build --json` writes, or a built page), so a version can come
from anywhere: a build kept from earlier, a git checkout, another method's spec. Objects correspond by key, and
by the `was` keys a spec declares for renames, merges and splits; the correspondence is taken over all the
versions at once (a lineage), so a rename two versions back still matches. What counts as a change is read from
the built objects, never from a TeX parse:

  renamed     a different key (matched by `was`)
  structure   inputs or outputs (as lineages), group, `per`; a symbol's dimensions or role
  equation    a step's TeX, or a symbol's, compared as text with the whitespace collapsed
  code        the references, or the source text they resolve to
  value       a setting's or a constant's value
  check       the @verify result
  wording     labels, names, descriptions, `where=` definitions (quiet by default)

The viewer (viewer.html, "lanes") applies the same rules; keep the two in step.
"""
from __future__ import annotations

import difflib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KINDS = {"op": "ops", "symbol": "symbols", "group": "groups"}
QUIET = {"wording"}


def load(path: str | Path) -> dict[str, Any]:
    """A built headway: the JSON from `--json`, or the data inside a page from `headway build`."""
    text = Path(path).read_text()
    if text.lstrip().startswith("{"):
        d = json.loads(text)
    else:
        m = re.search(r"^let D = (\{.*\});$", text, re.M)
        if not m:
            raise ValueError(f"{path}: neither a headway JSON file nor a page built by headway")
        d = json.loads(m[1])
    if not isinstance(d, dict) or not {"ops", "symbols", "groups"} <= set(d):
        raise ValueError(f"{path}: not a headway")
    d.pop("lanes", None)
    return d


def _key(kind: str, obj: dict[str, Any]) -> str:
    return obj["path"] if kind == "group" else obj["key"]


Item = tuple[int, str, str]         # (version, kind, key)


def lineages(headways: Sequence[dict[str, Any]]) -> dict[Item, str]:
    """Each object of each version -> its lineage, named by one of its keys (the latest version's)."""
    up: dict[Item, Item] = {}

    def find(x: Item) -> Item:
        while up[x] != x:
            up[x] = up[up[x]]
            x = up[x]
        return x

    def join(a: Item, b: Item) -> None:
        up[find(a)] = find(b)
    holders: dict[tuple[str, str], list[Item]] = {}
    for v, d in enumerate(headways):
        for kind, field_ in KINDS.items():
            for obj in d[field_]:
                item = (v, kind, _key(kind, obj))
                up[item] = item
                holders.setdefault((kind, item[2]), []).append(item)
    for same in holders.values():
        for item in same[1:]:
            join(item, same[0])
    for v, d in enumerate(headways):
        for kind, field_ in KINDS.items():
            for obj in d[field_]:
                for old in obj.get("was", []):
                    for other in holders.get((kind, old), []):
                        if other[0] != v:                       # within one version, `was` is a check error
                            join((v, kind, _key(kind, obj)), other)
    members: dict[Item, list[Item]] = {}
    for item in up:
        members.setdefault(find(item), []).append(item)
    out = {}
    for group in members.values():
        name = max(group, key=lambda it: (it[0], -_order(headways, it)))[2]
        out.update({it: name for it in group})
    return out


def _order(headways: Sequence[dict[str, Any]], item: Item) -> int:
    v, kind, key = item
    return next(n for n, o in enumerate(headways[v][KINDS[kind]]) if _key(kind, o) == key)


@dataclass
class Change:
    kind: str                       # op, symbol, group
    key: str                        # in the newer version (the older one, if removed)
    status: str                     # added, removed, changed, split, merged
    what: list[str] = field(default_factory=list)       # categories, for "changed"
    details: list[tuple[str, str]] = field(default_factory=list)  # (category, what changed)
    old: list[str] = field(default_factory=list)        # the older version's keys

    @property
    def quiet(self) -> bool:
        return self.status == "changed" and set(self.what) <= QUIET


def _tex(t: str) -> str:
    return " ".join((t or "").split())


def _code(obj: dict[str, Any]) -> list[tuple[str, str]]:
    return [(c["ref"], c.get("source", "")) for c in obj.get("code", [])]


def compare(headways: Sequence[dict[str, Any]], a: int, b: int,
            lin: dict[Item, str] | None = None) -> list[Change]:
    """What changed from version a to version b, in b's order (removals last)."""
    lin = lin if lin is not None else lineages(headways)
    A, B = headways[a], headways[b]

    def by_lineage(v: int, kind: str) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for obj in headways[v][KINDS[kind]]:
            out.setdefault(lin[(v, kind, _key(kind, obj))], []).append(obj)
        return out

    def syms(v: int, keys: list[str]) -> set[str]:
        return {lin.get((v, "symbol", k), "?" + k) for k in keys}

    def group_of(v: int, path: str) -> str:
        return lin.get((v, "group", path), path) if path else ""

    out: list[Change] = []
    for kind in KINDS:
        old, new = by_lineage(a, kind), by_lineage(b, kind)
        for name, objs in new.items():
            if name not in old:
                out.extend(Change(kind, _key(kind, o), "added") for o in objs)
                continue
            olds = old[name]
            for o, n, status in _pair(kind, olds, objs):
                if status in ("split", "merged"):
                    okeys, nkeys = [_key(kind, x) for x in olds], [_key(kind, x) for x in objs]
                    out.append(Change(kind, _key(kind, n), status, ["structure"],
                                      [("structure", f"{', '.join(okeys)} -> {', '.join(nkeys)}")], okeys))
                elif change := _compare(kind, o, n, a, b, syms, group_of):
                    out.append(change)
        for name, objs in old.items():
            if name not in new:
                out.extend(Change(kind, _key(kind, o), "removed") for o in objs)
    return out


def _pair(kind: str, olds: list[dict[str, Any]], news: list[dict[str, Any]]
          ) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    """The two versions' objects of one lineage, paired: the same key, else one left on each side (a rename);
    what remains was split (more new) or merged (fewer new)."""
    keys = {_key(kind, o): o for o in olds}
    pairs = [(keys[_key(kind, n)], n, "same") for n in news if _key(kind, n) in keys]
    left_old = [o for o in olds if _key(kind, o) not in {_key(kind, n) for n in news}]
    left_new = [n for n in news if _key(kind, n) not in keys]
    if len(left_old) == 1 and len(left_new) == 1:
        return pairs + [(left_old[0], left_new[0], "same")]
    if left_old and not left_new:                       # merged into an object that kept its key
        return [(o, n, "merged" if k == 0 else s) for k, (o, n, s) in enumerate(pairs)]
    status = "split" if len(left_new) > len(left_old) else "merged"
    return pairs + [(olds[0], n, status) for n in left_new]


def _compare(kind: str, o: dict[str, Any], n: dict[str, Any], a: int, b: int, syms: Any, group_of: Any
             ) -> Change | None:
    what: list[str] = []
    details: list[tuple[str, str]] = []

    def note(category: str, text: str) -> None:
        if category not in what:
            what.append(category)
        details.append((category, text))
    if _key(kind, o) != _key(kind, n):
        note("renamed", f"renamed from {_key(kind, o)}")
    if kind == "op":
        for side in ("inputs", "outputs"):
            was, now = syms(a, o[side]), syms(b, n[side])
            if was != now:
                note("structure", f"{side}: " + ", ".join([f"-{k}" for k in sorted(was - now)]
                                                          + [f"+{k}" for k in sorted(now - was)]))
        if group_of(a, o["group"]) != group_of(b, n["group"]):
            note("structure", f"moved from group {o['group'] or '(top level)'} to {n['group'] or '(top level)'}")
        if o.get("per") != n.get("per"):
            note("structure", f"per: {o.get('per')} -> {n.get('per')}")
        if _tex(o["tex"]) != _tex(n["tex"]):
            note("equation", f"equation: {o['tex']}\n       -> {n['tex']}")
        if o.get("verified") != n.get("verified"):
            note("check", f"equation check: {_verified(o)} -> {_verified(n)}")
        for f in ("label", "desc"):
            if o.get(f) != n.get(f):
                note("wording", f"{f}: {o.get(f)!r} -> {n.get(f)!r}")
        if [list(w) for w in o.get("where", [])] != [list(w) for w in n.get("where", [])]:
            note("wording", "where= definitions")
    elif kind == "symbol":
        for f in ("dims", "per", "role"):
            if o.get(f) != n.get(f):
                note("structure", f"{f}: {o.get(f)} -> {n.get(f)}")
        if _tex(o["tex"]) != _tex(n["tex"]):
            note("equation", f"notation: {o['tex']} -> {n['tex']}")
        if o.get("value") != n.get("value"):
            note("value", f"value: {o.get('value')} -> {n.get('value')}")
        for f in ("name", "desc"):
            if o.get(f) != n.get(f):
                note("wording", f"{f}: {o.get(f)!r} -> {n.get(f)!r}")
    else:
        for f in ("label", "desc"):
            if o.get(f) != n.get(f):
                note("wording", f"{f}: {o.get(f)!r} -> {n.get(f)!r}")
    if kind != "group" and _code(o) != _code(n):
        note("code", "code:\n" + code_diff(o, n))
    return Change(kind, _key(kind, n), "changed", what, details, [_key(kind, o)]) if what else None


def _verified(o: dict[str, Any]) -> str:
    return {True: "passed", False: "failed", None: "none"}[o.get("verified")]


def code_diff(o: dict[str, Any], n: dict[str, Any], limit: int = 40) -> str:
    """A unified diff of the code two versions of an object point to."""
    def lines(obj: dict[str, Any]) -> list[str]:
        return [ln for ref, src in _code(obj) for ln in [f"# {ref}", *src.rstrip("\n").split("\n")]]
    diff = list(difflib.unified_diff(lines(o), lines(n), lineterm="", n=1))[2:]
    return "\n".join(diff[:limit] + ([f"... {len(diff) - limit} more lines"] if len(diff) > limit else []))


MARK = {"added": "+", "removed": "-", "changed": "~", "split": "<", "merged": ">"}


def text(headways: Sequence[dict[str, Any]], names: Sequence[str], wording: bool = False) -> str:
    """Each version against the one before it, as text."""
    lin = lineages(headways)
    out = []
    for b in range(1, len(headways)):
        changes = [c for c in compare(headways, b - 1, b, lin) if wording or not c.quiet]
        out.append(f"{names[b - 1]} -> {names[b]}: " + (f"{len(changes)} changes" if changes else "no changes"))
        for c in changes:
            label = _label(headways[b if c.status != "removed" else b - 1], c.kind, c.key)
            head = f"  {MARK[c.status]} {c.kind} {c.key}{f' ({label})' if label else ''}"
            if c.status in ("split", "merged"):
                head += f": {c.status} from {', '.join(c.old)}"
            elif c.what:
                head += ": " + ", ".join(w for w in c.what if wording or w not in QUIET)
            out.append(head)
            for category, d in c.details:
                if wording or category not in QUIET:
                    out.extend("      " + ln for ln in d.split("\n"))
    return "\n".join(out)


def _label(d: dict[str, Any], kind: str, key: str) -> str:
    obj: dict[str, Any] = next((o for o in d[KINDS[kind]] if _key(kind, o) == key), {})
    return obj.get("label") or obj.get("name") or ""
