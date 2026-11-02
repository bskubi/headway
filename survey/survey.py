# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml"]
# ///
"""The survey's records: check them against the traits, tabulate them, rename a trait across them.

    uv run survey.py check              records that name unknown traits, or lack required fields
    uv run survey.py table              which codebase has which trait (by category), with headway's status
    uv run survey.py trait ID           every codebase's evidence for one trait
    uv run survey.py rename OLD NEW     rename a trait id in every record (traits.yaml is edited by hand)
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
REQUIRED = ("id", "name", "url", "commit", "read", "families", "computation", "traits")


def traits() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for category, items in yaml.safe_load((HERE / "traits.yaml").read_text()).items():
        for tid, t in items.items():
            out[tid] = {**t, "category": category}
    return out


def records() -> list[tuple[Path, dict]]:
    out = []
    for p in sorted((HERE / "codebases").glob("*.yaml")):
        try:
            out.append((p, yaml.safe_load(p.read_text())))
        except yaml.YAMLError as e:
            sys.exit(f"{p.name}: {e}")
    return out


def check() -> int:
    known = traits()
    problems = []
    for path, r in records():
        problems += [f"{path.name}: no {f}" for f in REQUIRED if f not in r]
        problems += [f"{path.name}: unknown trait {t}" for t in r.get("traits", {}) if t not in known]
    print("\n".join(problems) or f"{len(records())} records, {len(known)} traits: ok")
    return 1 if problems else 0


def table() -> None:
    known = traits()
    rs = records()
    ids = [r["id"] for _, r in rs]
    width = max(map(len, known)) + 2
    print(" " * (width + 2) + "headway   " + "  ".join(f"{i[:10]:10}" for i in ids))
    category = None
    for tid, t in known.items():
        if t["category"] != category:
            category = t["category"]
            print(f"\n{category}")
        marks = ["x" if tid in r.get("traits", {}) else "." for _, r in rs]
        print(f"  {tid:{width}}{t.get('headway', ''):10}" + "  ".join(f"{m:10}" for m in marks))
    print()
    for _, r in rs:
        print(f"{r['id']}: computation {r['computation']}; tests {(r.get('tests') or {}).get('result')}")


def trait(tid: str) -> None:
    t = traits()[tid]
    print(f"{tid} ({t['category']}, headway: {t.get('headway')}): {t['description']}")
    for _, r in records():
        if tid in r.get("traits", {}):
            ev = r["traits"][tid] or {}
            print(f"  {r['id']}: {ev.get('where', '')}  {ev.get('evidence', '')}".rstrip())


def rename(old: str, new: str) -> None:
    for path, r in records():
        if old in r.get("traits", {}):
            text = path.read_text().replace(f"  {old}:", f"  {new}:")
            path.write_text(text)
            print(f"{path.name}: {old} -> {new}")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["table"]
    if cmd == "check":
        sys.exit(check())
    {"table": table, "trait": trait, "rename": rename}[cmd](*args)
