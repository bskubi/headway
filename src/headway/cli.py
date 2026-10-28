"""headway check <spec module>     the spec against itself and the code; exit 1 on errors
headway build <spec module> -o out.html [--collapse GROUP] [--hide GROUP] [--json out.json] [--no-verify]
                                   [--offline] [--compare [NAME=]VERSION ...] [--results RESULTS.json ...]
                                   [--name NAME] [--key NAME=VALUE ...]
headway serve [<spec module>] [--port 8765] [--open] [--collapse GROUP] [--hide GROUP] [--no-verify]
                                   [--compare [NAME=]VERSION ...]
                                   the library (recent headways, and the specs and built headways in any folder,
                                   starting at this one), and with a spec, that spec served live: rebuilt as the
                                   spec and the code change
headway diff VERSION VERSION [VERSION ...] [--wording]
                                   what changed from each version to the next
headway attach VERSION RESULTS.json [-o OUT]
                                   put benchmark results (and their suite, built) into a built headway (JSON or page)
headway results VERSION VERSION ... [--heldout]
                                   the methods' results on their common suite, with its objectives
headway guide                      how headway works and how to write a headway for a codebase

<spec module> is an importable module holding one Headway (or "module:attribute"). A VERSION is a built version: the
JSON from --json, or a page from build; for diff, also a spec module, built as it is now. --compare shows each
given version in its own lane above the spec's, registered on one layout."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any

from headway import api, bench, diff, page
from headway.check import check


def built(target: str) -> dict[str, Any]:
    """A VERSION argument of diff: a built file, or a spec module, built in a fresh process (two versions of one
    module cannot be imported into one)."""
    if Path(target).is_file():
        return diff.load(target)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "headway.json"
        run = subprocess.run([sys.executable, "-m", "headway.cli", "build", target, "-o", os.devnull,
                              "--json", str(out), "--no-verify"], capture_output=True, text=True)
        if run.returncode or not out.exists():
            raise SystemExit(f"could not build {target}:\n{run.stderr}")
        return json.loads(out.read_text())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="headway", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="check a spec")
    c.add_argument("spec")
    c.add_argument("--no-verify", action="store_true", help="skip the @verify functions")
    b = sub.add_parser("build", help="write the viewer")
    b.add_argument("spec")
    b.add_argument("-o", "--out", default="headway.html")
    b.add_argument("--json", help="also write the graph as JSON")
    b.add_argument("--collapse", action="append", default=[], help="a group to show collapsed (repeatable)")
    b.add_argument("--hide", action="append", default=[], help="a group to leave out (repeatable)")
    b.add_argument("--no-verify", action="store_true")
    b.add_argument("--offline", action="store_true",
                   help="write KaTeX, elkjs and highlight.js into the file (about 3 MB), so it opens without internet")
    b.add_argument("--editor", help="vscode (default), cursor, pycharm, file, or a URL template with {file}, {line}, "
                                    "{col}; default $HEADWAY_EDITOR")
    b.add_argument("--results", action="append", default=[], metavar="RESULTS.json",
                   help="benchmark results to put into the headway, with their suite (repeatable)")
    b.add_argument("--name", help="what to call this version in a comparison (e.g. v2, a tag, an experiment)")
    b.add_argument("--compare", action="append", default=[], metavar="[NAME=]VERSION",
                   help="another version (a --json file or a built page) to show in a lane of its own (repeatable)")
    b.add_argument("--key", action="append", default=[], metavar="NAME=VALUE",
                   help="a label of this version to group or pair versions by: a hyperparameter, a variant "
                        "(repeatable; numbers and true/false are read as such)")
    s = sub.add_parser("serve", help="serve the viewer and rebuild it when the spec or the code changes")
    s.add_argument("spec", nargs="?", help="a spec to serve live (the library is served either way)")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--open", action="store_true", help="open it in the browser")
    s.add_argument("--collapse", action="append", default=[], help="a group to show collapsed (repeatable)")
    s.add_argument("--hide", action="append", default=[], help="a group to leave out (repeatable)")
    s.add_argument("--no-verify", action="store_true")
    s.add_argument("--editor", help="as for build")
    s.add_argument("--compare", action="append", default=[], metavar="[NAME=]VERSION", help="as for build")
    df = sub.add_parser("diff", help="what changed between versions of a headway")
    df.add_argument("headways", nargs="+", metavar="VERSION")
    df.add_argument("--wording", action="store_true", help="also list changes to labels and descriptions")
    at = sub.add_parser("attach", help="put benchmark results into a built headway")
    at.add_argument("headway")
    at.add_argument("results")
    at.add_argument("-o", "--out", help="where to write it (default: the headway file itself)")
    rs = sub.add_parser("results", help="compare methods on their benchmark results")
    rs.add_argument("headways", nargs="+", metavar="VERSION")
    rs.add_argument("--heldout", action="store_true", help="include the held-out datasets")
    sub.add_parser("guide", help="how headway works and how to write a headway for a codebase (for people and "
                                 "for coding agents)")
    a = p.parse_args(argv)
    if a.command == "guide":
        print(resources.files("headway").joinpath("GUIDE.md").read_text())
        return 0
    if a.command == "attach":
        api.attach(a.headway, a.results, a.out, log=say)
        return 0
    if a.command == "results":
        headways = [diff.load(t) for t in a.headways]
        print(bench.table(headways, [d.get("name") or t for d, t in zip(headways, a.headways)], a.heldout))
        return 0
    if a.command == "diff":
        if len(a.headways) < 2:
            p.error("diff needs two versions or more")
        headways = [built(t) for t in a.headways]
        print(diff.text(headways, [d.get("name") or t for d, t in zip(headways, a.headways)], a.wording))
        return 0
    try:
        compared = api.lanes(a.compare) if a.command in ("build", "serve") else []
    except (OSError, ValueError) as e:
        p.error(f"--compare: {e}")
    if a.command in ("build", "serve"):
        try:
            page.editor(a.editor)                          # a bad --editor fails here, before any work
        except AssertionError as e:
            p.error(str(e))

    if a.command == "serve":
        from headway.home import Home
        build_args = [*(f"--collapse={c}" for c in a.collapse), *(f"--hide={h}" for h in a.hide),
                      *([f"--editor={a.editor}"] if a.editor else []), *(["--no-verify"] if a.no_verify else [])]
        if a.spec is None and a.compare:
            p.error("serve: --compare needs a spec (in the library, open versions side by side instead)")
        Home(os.getcwd(), build_args).run(a.port, a.open, spec=a.spec, compare=compared)
        return 0

    if a.command == "check":
        report = check(api.load(a.spec), run_verifiers=not a.no_verify)
        say(report.text())
        return 0 if report.ok else 1
    keys = {}
    for kv in a.key:
        k, eq, v = kv.partition("=")
        if not eq or not k:
            p.error(f"--key {kv}: not NAME=VALUE")
        try:
            parsed = json.loads(v)
        except ValueError:
            parsed = v
        keys[k] = parsed if isinstance(parsed, (int, float, bool)) else v
    api.build(a.spec, a.out, json=a.json, name=a.name, results=a.results, compare=a.compare, collapse=a.collapse,
              hide=a.hide, keys=keys, verify=not a.no_verify, offline=a.offline, editor=a.editor, log=say)
    return 0


def say(message: str) -> None:
    print(message, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
