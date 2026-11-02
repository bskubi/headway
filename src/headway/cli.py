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
headway init <pkg.mod:func> [-o SPEC.py] [--depth 2] [--hints] [--clean]
                                   a first draft of a spec, read from the code: a step per call to the package's
                                   functions, starting at this one; the TeX and descriptions are left as TODOs
headway redraft <spec> [--force]   a spec drafted by init, drafted again from the code with the same arguments
                                   (a spec edited since: only with --force, the edited one kept as .bak)
headway tex <code reference> [--name CODE=TEX ...] [--hints]
                                   a Python function (or block) written as math, as the side panel shows it
headway record -p PACKAGE [-p ...] -o DIR [--first N] [--every N] [--limit N] -- PYTEST ARGS
                                   run tests (or any pytest target), recording the calls of the packages' functions:
                                   their arguments, results, paths and environment, as a trace in DIR
headway paths DIR [FUNCTION ...]   the paths a trace's calls took, function by function, and how often
headway sets DIR [FUNCTION]        the index sets a trace's array sizes show, and the axes in each
headway verify DIR FUNCTION ...    the math shown for each function, checked on the calls the trace recorded
headway options DIR FUNCTION ...   the settings each function was called with (values, objects' types), counted
headway env DIR                    the environment a trace ran in (Python, packages, thread settings)
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
    i = sub.add_parser("init", help="draft a spec from the code")
    i.add_argument("entry", help="the function the method starts in, as pkg.mod:func")
    i.add_argument("-o", "--out", help="where to write it (default: print it)")
    i.add_argument("--depth", type=int, default=2,
                   help="how many levels of called functions to open up into groups of their own calls (default 2)")
    i.add_argument("--hints", action="store_true",
                   help="take the code's type hints as assumptions: settings dataclasses, methods of package "
                        "classes as steps, arrays and lists changed in place by += and .append")
    i.add_argument("--clean", action="store_true",
                   help="the code is clean code (written to be read; see the guide): its type hints taken, each "
                        "assignment between the package's calls a step of its own, every function opened up")
    rd = sub.add_parser("redraft", help="draft a drafted spec again from the code")
    rd.add_argument("spec", help="the spec: its file, or its module")
    rd.add_argument("--force", action="store_true",
                    help="even if it has been edited since it was drafted (the edited one is kept as .bak)")
    tx = sub.add_parser("tex", help="write a Python function's code as math")
    tx.add_argument("ref", help='a code reference, e.g. "pkg.mod:func" or "pkg.mod:func#from..to"')
    tx.add_argument("--name", action="append", default=[], metavar="CODE=TEX",
                    help="how to write a name of the code (repeatable; CODE may be \"return\")")
    tx.add_argument("--hints", action="store_true",
                    help="take the code's type hints as assumptions: numbers as scalars, jaxtyping's named axes as "
                         "indices")
    tx.add_argument("--trace", metavar="DIR", help="a trace (headway record) whose array sizes give the function's "
                                                   "parameters their index sets")
    tx.add_argument("--set", action="append", default=[], metavar="FOUND=NAME",
                    help="rename an index set the trace found (repeatable)")
    rc = sub.add_parser("record", help="run tests, recording the calls of the packages' functions as a trace")
    rc.add_argument("-p", "--package", dest="packages", action="append", required=True,
                    help="a package (or absolute directory) whose functions to record (repeatable)")
    rc.add_argument("-o", "--out", required=True, help="the trace's directory")
    rc.add_argument("--first", type=int, default=20, help="calls of each function recorded in full first")
    rc.add_argument("--every", type=int, default=50, help="then every n-th call")
    rc.add_argument("--limit", type=int, default=200, help="calls of each function recorded in full at most")
    rc.add_argument("--parallel", action="store_true", help="leave joblib parallel (its workers are not traced)")
    rc.add_argument("--no-jit", action="store_true",
                    help="run numba's compiled functions as Python (NUMBA_DISABLE_JIT=1), so that they are traced")
    rc.add_argument("--python", metavar="EXE", help="run the tests in this interpreter (the codebase's own, which may "
                                                    "be an older Python): the tracer runs there standalone")
    rc.add_argument("pytest_args", nargs="*", help="after --: the tests to run, as pytest takes them")
    pa = sub.add_parser("paths", help="the paths a trace's calls took")
    pa.add_argument("trace")
    pa.add_argument("functions", nargs="*")
    ve = sub.add_parser("verify", help="check the math shown for functions on the calls a trace recorded")
    ve.add_argument("trace")
    ve.add_argument("functions", nargs="*", help="module:qualname of each function (none: every function traced)")
    ve.add_argument("--limit", type=int, default=50, help="calls of each function to check")
    op = sub.add_parser("options", help="the settings a trace's functions were called with")
    op.add_argument("trace")
    op.add_argument("functions", nargs="+")
    en = sub.add_parser("env", help="the environment a trace ran in")
    en.add_argument("trace")
    se = sub.add_parser("sets", help="the index sets a trace's array sizes show")
    se.add_argument("trace")
    se.add_argument("function", nargs="?")
    ln = sub.add_parser("lint", help="where clean code leaves the clean subset, or its math stays opaque")
    ln.add_argument("module", help="the clean code's module, e.g. clean_methembed.steps")
    tr = sub.add_parser("trace", help="run the code's tests and check clean code on their calls: against the code it "
                                      "stands for, its declarations, and the math shown")
    tr.add_argument("module", help="the clean code's module, e.g. clean_methembed.steps")
    tr.add_argument("--swap", action="store_true",
                    help="put the clean versions in place of the originals and run the tests instead")
    tr.add_argument("--set", action="append", default=[], metavar="MODULE.NAME=VALUE",
                    help="change a module constant before the run (e.g. a chunk size; repeatable)")
    tr.add_argument("--limit", type=int, default=50, help="calls of each function to check (default 50)")
    tr.add_argument("pytest_args", nargs="*", help="after --: the tests to run, as pytest takes them")
    sub.add_parser("guide", help="how headway works and how to write a headway for a codebase (for people and "
                                 "for coding agents)")
    a = p.parse_args(argv)
    if a.command == "guide":
        print(resources.files("headway").joinpath("GUIDE.md").read_text())
        return 0
    if a.command == "init":
        from headway.draft import DraftError, draft
        if a.out and Path(a.out).exists():
            p.error(f"init: {a.out} exists; give another -o or remove it")
        try:
            text = draft(a.entry, a.depth, hints=a.hints, clean=a.clean)
        except DraftError as e:
            p.error(f"init: {e}")
        if a.out:
            Path(a.out).write_text(text)
            say(f"wrote {a.out}; next: fill in its TODOs, and `headway check` it")
        else:
            print(text, end="")
        return 0
    if a.command == "redraft":
        from headway.draft import DraftEdited, DraftError, redraft
        from headway.home import spec_file
        f = Path(a.spec) if Path(a.spec).is_file() else spec_file(a.spec, Path.cwd())
        if f is None:
            p.error(f"redraft: no spec {a.spec}")
        try:
            backup = redraft(f, force=a.force)
        except DraftEdited as e:
            p.error(f"redraft: {e}; --force to draft it again anyway (the edited one is kept as {f.name}.bak)")
        except DraftError as e:
            p.error(f"redraft: {e}")
        say(f"redrafted {f}" + (f"; the edited one is {backup}" if backup else ""))
        return 0
    if a.command == "lint":
        from headway.lint import lint
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        linted = lint(a.module)
        print(linted.text())
        return 1 if any(f.level == "error" for f in linted.findings) else 0
    if a.command == "trace":
        from headway import cleanrun
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        if a.swap:
            return cleanrun.swap(a.module, a.pytest_args, a.set)
        found = cleanrun.check(a.module, a.pytest_args, a.set, a.limit)
        bad = [n for n, rels in found.evidence.items() for results in rels.values() if not all(r.passed for r in
                                                                                              results)]
        return 1 if bad else 0
    if a.command == "record":
        return _record(a)
    if a.command == "paths":
        from headway.tracing import Trace, describe_paths
        t = Trace.read(a.trace)
        for fn in a.functions or sorted(t.functions):
            print(describe_paths(t, fn))
        return 0
    if a.command == "verify":
        from headway.tracing import Trace
        from headway.tracing.check import check_function
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        t = Trace.read(a.trace)
        verdicts = []
        attempted = 0
        for f in a.functions or sorted(f for f in t.functions if "<locals>" not in f):
            attempted += 1
            try:
                verdicts.append(check_function(t, f, a.limit))
            except Exception as e:                      # noqa: BLE001 - one function's failure, reported
                print(f"{f}: not checked ({type(e).__name__}: {e})")
                continue
            print(verdicts[-1].text())
        if not a.functions:
            agreed = sum(v.agreed == v.checked > 0 for v in verdicts)
            differed = [v.function for v in verdicts if v.disagreed]
            print(f"\n{attempted} functions: the math agreed on every recorded call for {agreed}; it ran and gave "
                  f"other values for {len(differed)}; the rest could not be checked (not read, not recorded, "
                  f"no calls)" + "".join(f"\n  disagreed: {f}" for f in differed))
        return 1 if any(v.disagreed for v in verdicts) else 0
    if a.command == "options":
        from headway.tracing import Trace
        from headway.tracing.analyze import describe_options
        t = Trace.read(a.trace)
        for f in a.functions:
            print(describe_options(t, f))
        return 0
    if a.command == "env":
        from headway.tracing import Trace
        from headway.tracing.analyze import describe_environment
        print(describe_environment(Trace.read(a.trace)))
        return 0
    if a.command == "sets":
        from headway.tracing import Trace, describe_sets, index_sets
        print(describe_sets(index_sets(Trace.read(a.trace)), a.function))
        return 0
    if a.command == "tex":
        from headway import codetex
        from headway.draft import axes_reader
        from headway.resolve import RefError, resolve
        names = {}
        for kv in a.name:
            k, eq, v = kv.partition("=")
            if not eq or not k:
                p.error(f"--name {kv}: not CODE=TEX")
            names[k] = codetex.Name(v)
        if os.getcwd() not in sys.path:                 # as for a spec: modules import from here
            sys.path.insert(0, os.getcwd())
        try:
            loc = resolve(a.ref)
        except RefError as e:
            p.error(f"tex: {e}")
        where: list[str] = []
        if a.trace:
            names, letters = _traced_names(a.trace, a.ref, a.set, names)
            where = [rf"{v} \in \text{{{codetex._text(k)}}}" for k, v in letters.items()]
        if loc.lang != "python":
            p.error(f"tex: {a.ref} is {loc.lang}; only Python is written as math")
        out = codetex.transcribe(loc.source, names, imported=codetex.imports(loc.file), file=loc.file,
                                 first_line=loc.start,
                                 hints=a.hints or bool(a.trace), where=where or None,
                                 axes=axes_reader(loc.file) if a.hints else None)
        if out is None:
            p.error(f"tex: nothing to write in {a.ref}")
        print(out)
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


def _record(a: Any) -> int:
    """headway record: the tests run with the tracer on, the trace written."""
    from headway.tracing import Tracer
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    args = [x for x in a.pytest_args if x != "--"]
    if a.no_jit:
        os.environ["NUMBA_DISABLE_JIT"] = "1"           # read by numba at import; recorded with the environment
    if a.python:
        script = Path(__file__).parent / "tracing" / "standalone.py"
        cmd = [a.python, str(script), *(f"-p={p}" for p in a.packages), "-o", str(Path(a.out).resolve()),
               f"--first={a.first}", f"--every={a.every}", f"--limit={a.limit}",
               *(["--parallel"] if a.parallel else []), "--", *args]
        return subprocess.run(cmd).returncode
    import pytest
    t = Tracer(a.packages, first=a.first, every=a.every, limit=a.limit, sequential=not a.parallel)
    with t:
        code = pytest.main(args)
    out = t.write(a.out)
    assert t.trace is not None
    say(f"tests: exit {int(code)}; {len(t.trace.calls)} calls of {len(t.trace.functions)} functions "
        f"({sum(c.kept for c in t.trace.calls)} recorded in full); trace in {out}")
    return int(code)


def _traced_names(trace: str, ref: str, renames: list[str], names: dict[str, Any]) -> tuple[dict[str, Any],
                                                                                              dict[str, str]]:
    """The names of a function's parameters with the index sets a trace found for their axes, and a letter for
    each set (its first letter, else the next free one)."""
    from headway import codetex
    from headway.tracing import Trace, index_sets, signatures
    from headway.tracing.analyze import renamed
    t = Trace.read(trace)
    function = ref.split("#")[0].replace(":", ".").rsplit(".", 1)
    qual = f"{function[0]}:{function[1]}" if len(function) == 2 else ref
    rename = dict(kv.split("=", 1) for kv in renames if "=" in kv)
    sets = renamed(index_sets(t, [qual]), rename)
    sigs = signatures(sets, qual)
    out = dict(names)
    letters: dict[str, str] = {}
    for param, dims in sigs.items():
        if param == "return" or param in out:
            continue
        for d in dims:
            if d != "1" and d not in letters:
                free = [ch for ch in d.lower() + "ijklmnpqrstuvwxyz" if ch.isalpha() and ch not in letters.values()]
                letters[d] = free[0]
        sub = "".join(letters[d] for d in dims if d != "1")
        tex = codetex._ident(param) + (f"_{{{sub}}}" if sub else "")
        out[param] = codetex.Name(tex, tuple(letters[d] if d != "1" else "1" for d in dims))
    return out, letters


def say(message: str) -> None:
    print(message, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
