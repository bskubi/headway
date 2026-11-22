"""headway check <spec module>     the spec against itself and the code; exit 1 on errors
headway build <spec module> -o out.html [--collapse GROUP] [--hide GROUP] [--json out.json] [--no-verify]
                                   [--offline] [--compare [NAME=]VERSION ...] [--results RESULTS.json ...]
                                   [--name NAME] [--key NAME=VALUE ...]
headway serve [<spec module> | ATLAS] [--port 8765] [--open] [--collapse GROUP] [--hide GROUP] [--no-verify]
                                   [--compare [NAME=]VERSION ...]
                                   the library (recent headways, and the specs and built headways in any folder,
                                   starting at this one); with a spec, that spec served live: rebuilt as the spec
                                   and the code change; with an atlas folder, its explorer
headway diff VERSION VERSION [VERSION ...] [--wording]
                                   what changed from each version to the next
headway attach VERSION RESULTS.json [-o OUT]
                                   put benchmark results (and their suite, built) into a built headway (JSON or page)
headway results VERSION VERSION ... [--heldout]
                                   the methods' results on their common suite, with its objectives
headway build <spec module> --atlas DIR --name NAME [--from VERSION ...] [--why TEXT] [--prediction TEXT]
                                   [--keep-old NAME] [--drop-results]
                                   add the version (with its results) to an atlas, made from the versions given
headway attach ATLAS NAME RESULTS.json | ATLAS --relation ID RESULTS.json --suite SUITE
headway relate ATLAS KIND VERSION ... [--base V] [--factors K,K] [--id ID] [--label] [--why] [--prediction]
                                   [--outcome] [--tag T ...]
                                   relate versions: step, pair, alternatives, factorial, group (update ID: change one)
headway page ATLAS [-o atlas.html] [--view V] [--full]
                                   the atlas's page: every version explored, any of them opened as a stack
headway atlas init|import|scan|check|validate|list|rename|remove|extract|merge|view ...
                                   make, check and reorganize atlases (docs/atlas.md); diff and results take an
                                   atlas too: headway diff ATLAS A B, headway results ATLAS [TERMS]
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
headway excerpt PDF [--page N] (--between FIRST LAST | --panel LETTER | --box L T R B) [-o OUT.png]
headway excerpt PDF --page N --panels
                                   preview an excerpt of a document (G.excerpt): where it is found, as a box to
                                   give, and its image; or the panel letters a page's figure has
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
from headway import atlas as atlas_mod
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


def excerpt_command(p: argparse.ArgumentParser, a: argparse.Namespace) -> int:
    from headway import media, pdftext
    from headway.spec import Media
    pdf = Path(a.pdf)
    if not pdf.is_file():
        p.error(f"excerpt: {pdf} not found")
    try:
        if a.panels:
            if a.page is None:
                p.error("excerpt: --panels needs --page")
            pages = pdftext.pages(pdf)
            if not 1 <= a.page <= len(pages):
                p.error(f"excerpt: {pdf} has pages 1 to {len(pages)}")
            width, height, words = pages[a.page - 1]
            for w in sorted(pdftext.labels(words), key=lambda w: (round(w.y0 / 12), w.x0)):
                b = pdftext.panel(pdf, a.page, w.text)
                print(f"{w.text}  box=({b[0] / width:.4f}, {b[1] / height:.4f}, {b[2] / width:.4f}, {b[3] / height:.4f})")
            return 0
        if sum(x is not None for x in (a.between, a.panel, a.box)) > 1:
            p.error("excerpt: one of --between, --panel and --box")
        m = Media("excerpt", page=a.page, box=tuple(a.box) if a.box else None,  # type: ignore[arg-type]
                  between=tuple(a.between) if a.between else None, panel=a.panel)
        if m.page is None and m.between is None:
            p.error("excerpt: give --page (or find the text with --between)")
        page, box = media.locate(pdf, m)
        shown = "the whole page" if box is None else f"box=({', '.join(f'{v:.4f}' for v in box)})"
        print(f"page {page}, {shown}")
        if a.out:
            Path(a.out).write_bytes(media.excerpt(pdf, page, box, a.dpi))
            say(f"wrote {a.out}")
    except (pdftext.FindError, RuntimeError) as e:
        p.error(f"excerpt: {e}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="headway", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="check a spec")
    c.add_argument("spec")
    c.add_argument("--no-verify", action="store_true", help="skip the @verify functions")
    b = sub.add_parser("build", help="write the viewer")
    b.add_argument("spec")
    b.add_argument("-o", "--out", help="the page to write (default headway.html; none with --atlas)")
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
    b.add_argument("--atlas", metavar="DIR", help="add this version (with its results) to an atlas, made if need be; "
                                                  "needs --name")
    b.add_argument("--from", dest="parents", action="append", default=[], metavar="VERSION",
                   help="with --atlas: a version this one was made from (repeatable); recorded, and a step")
    b.add_argument("--why", default="", help="with --from: why this version was made")
    b.add_argument("--prediction", default="", help="with --from: what was expected of it")
    b.add_argument("--keep-old", metavar="NAME", help="with --atlas, rebuilding a name: keep the earlier build as NAME")
    b.add_argument("--drop-results", action="store_true",
                   help="with --atlas, rebuilding a name: remove the earlier build's results")
    s = sub.add_parser("serve", help="serve the viewer and rebuild it when the spec or the code changes")
    s.add_argument("spec", nargs="?", help="a spec to serve live, or an atlas folder to explore (the library is served "
                                          "either way)")
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
    at = sub.add_parser("attach", help="put benchmark results into a built headway, or into an atlas")
    at.add_argument("headway", metavar="VERSION|ATLAS")
    at.add_argument("rest", nargs="+", metavar="ARG",
                    help="RESULTS.json; for an atlas: NAME RESULTS.json, or --relation ID RESULTS.json --suite KEY")
    at.add_argument("-o", "--out", help="where to write it (default: the headway file itself)")
    at.add_argument("--relation", help="atlas: the results are a relation's (differences, contrasts)")
    at.add_argument("--suite", help="with --relation: the suite (as the atlas names it) the results are on")
    rl = sub.add_parser("relate", help="relate versions in an atlas: a step, a pair, alternatives, a factorial, a group")
    rl.add_argument("atlas")
    rl.add_argument("kind", help="step, pair, alternatives, factorial, group; or update (then: the relation's id)")
    rl.add_argument("members", nargs="+", metavar="VERSION")
    rl.add_argument("--base", help="alternatives or group: the version the others are alternatives to")
    rl.add_argument("--factors", default="", help="factorial: the keys its versions differ in, comma-separated")
    rl.add_argument("--id", help="the relation's id (default: made from its kind and members)")
    rl.add_argument("--label")
    rl.add_argument("--why")
    rl.add_argument("--prediction")
    rl.add_argument("--outcome")
    rl.add_argument("--tag", action="append", default=[])
    pg = sub.add_parser("page", help="write the page of an atlas: the explorer, and the versions it opens")
    pg.add_argument("atlas")
    pg.add_argument("-o", "--out", default="atlas.html")
    pg.add_argument("--view", help="a view of the atlas to open in (and whose versions to carry in full)")
    pg.add_argument("--full", action="store_true", help="carry every version in full (default: up to 40)")
    pg.add_argument("--offline", action="store_true", help="as for build")
    al = sub.add_parser("atlas", help="make, check and reorganize atlases (see docs/atlas.md)")
    alsub = al.add_subparsers(dest="atlas_command", required=True)
    x = alsub.add_parser("init", help="make an atlas folder")
    x.add_argument("atlas")
    x.add_argument("--name")
    x = alsub.add_parser("import", help="add built versions (JSON or pages, with a page's other versions)")
    x.add_argument("atlas")
    x.add_argument("files", nargs="+")
    x = alsub.add_parser("scan", help="the manifest made again from versions/, and steps from their parents")
    x.add_argument("atlas")
    x = alsub.add_parser("check", help="results on earlier builds, provenance that differs, missing versions")
    x.add_argument("atlas")
    x = alsub.add_parser("validate", help="errors in the format (after editing the files by hand)")
    x.add_argument("atlas")
    x = alsub.add_parser("list", help="the versions, relations and views")
    x.add_argument("atlas")
    x = alsub.add_parser("rename", help="rename a version, with every reference to it")
    x.add_argument("atlas")
    x.add_argument("old")
    x.add_argument("new")
    x = alsub.add_parser("remove", help="remove a version, its results and its place in relations")
    x.add_argument("atlas")
    x.add_argument("name")
    x = alsub.add_parser("extract", help="a new atlas with some of the versions (and what they need)")
    x.add_argument("atlas")
    x.add_argument("out")
    x.add_argument("terms", nargs="+", metavar="TERM",
                   help="versions, or near:NAME~N, path:A..B, key:K=V, relation:ID, kind:K, tag:T (-TERM excludes)")
    x = alsub.add_parser("merge", help="a new atlas with both; the same name and model is one version")
    x.add_argument("a")
    x.add_argument("b")
    x.add_argument("out")
    x.add_argument("--prefix", help="rename b's versions whose name a has with a different model")
    x = alsub.add_parser("view", help="save a view: a stack (a path of versions) or a graph (a query)")
    x.add_argument("atlas")
    x.add_argument("name")
    x.add_argument("terms", nargs="*", metavar="TERM", help="as for extract; a stack's path, in this order if names")
    x.add_argument("--graph", action="store_true", help="a graph view of the matching versions (default: a stack)")
    x.add_argument("--baseline")
    x.add_argument("--suite", action="append", default=[], help="the suites to compare on (default: all)")
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
    ex = sub.add_parser("excerpt", help="preview an excerpt of a PDF (G.excerpt), or list a page's panel letters")
    ex.add_argument("pdf")
    ex.add_argument("--page", type=int)
    ex.add_argument("--between", nargs=2, metavar=("FIRST", "LAST"), help="the text from FIRST to LAST")
    ex.add_argument("--panel", metavar="LETTER", help="a figure's panel, by its letter")
    ex.add_argument("--box", nargs=4, type=float, metavar=("L", "T", "R", "B"), help="fractions of the page")
    ex.add_argument("--panels", action="store_true", help="list the page's panel letters and their boxes")
    ex.add_argument("-o", "--out", help="write the excerpt as a PNG")
    ex.add_argument("--dpi", type=int, help="resolution (default: chosen to give about 2000 px on the longer side)")
    sub.add_parser("guide", help="how headway works and how to write a headway for a codebase (for people and "
                                 "for coding agents)")
    a = p.parse_args(argv)
    if a.command == "guide":
        print(resources.files("headway").joinpath("GUIDE.md").read_text())
        return 0
    if a.command == "excerpt":
        return excerpt_command(p, a)
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
        try:
            linted = lint(a.module)
        except (ValueError, SyntaxError, OSError) as e:          # no such module or file, or one that doesn't parse
            p.error(f"lint: {e}")
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
                                 hints=a.hints or bool(a.trace), where=where or None)
        if out is None:
            p.error(f"tex: nothing to write in {a.ref}")
        print(out)
        return 0
    if a.command in ("attach", "relate", "page", "atlas") or \
            (a.command in ("results", "diff") and atlas_mod.is_atlas(a.headways[0])):
        try:
            return atlas_command(p, a)
        except atlas_mod.AtlasError as e:
            p.error(str(e))
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
        if a.compare and (a.spec is None or atlas_mod.is_atlas(a.spec)):
            p.error("serve: --compare needs a spec (in the library, or an atlas, open versions side by side instead)")
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
    if a.atlas and not a.name:
        p.error("build: --atlas needs --name (what the version is called in the atlas)")
    if not a.atlas and (a.parents or a.keep_old or a.drop_results):
        p.error("build: --from, --keep-old and --drop-results go with --atlas")
    if (a.why or a.prediction) and not a.parents:
        p.error("build: --why and --prediction go with --from (they are recorded on the step from each parent)")
    out = a.out or (None if a.atlas else "headway.html")
    try:
        api.build(a.spec, out, json=a.json, name=a.name, results=a.results, compare=a.compare, collapse=a.collapse,
                  hide=a.hide, keys=keys, verify=not a.no_verify, offline=a.offline, editor=a.editor, log=say,
                  atlas=a.atlas, parents=a.parents, why=a.why, prediction=a.prediction, keep_old=a.keep_old,
                  drop_results=a.drop_results)
    except atlas_mod.AtlasError as e:
        p.error(str(e))
    return 0


def atlas_command(p: argparse.ArgumentParser, a: argparse.Namespace) -> int:
    """attach, relate, page and atlas; results and diff given an atlas."""
    from headway.atlas import Atlas
    if a.command == "attach":
        if not atlas_mod.is_atlas(a.headway):
            if len(a.rest) != 1:
                p.error("attach VERSION RESULTS.json (an atlas: attach ATLAS NAME RESULTS.json)")
            api.attach(a.headway, a.rest[0], a.out, log=say)
            return 0
        at = Atlas.open(a.headway)
        if a.relation:
            if len(a.rest) != 1 or not a.suite:
                p.error("attach ATLAS --relation ID RESULTS.json --suite KEY")
            notes = at.attach_relation(a.relation, json.loads(Path(a.rest[0]).read_text()), a.suite)
        else:
            if len(a.rest) != 2:
                p.error("attach ATLAS NAME RESULTS.json")
            notes = at.attach(a.rest[0], api.read_results(a.rest[1], say))
        for n in notes:
            say(n)
        return 0
    if a.command == "relate":
        at = Atlas.open(a.atlas)
        fields = {k: getattr(a, k) for k in ("label", "why", "prediction", "outcome")}
        if a.kind == "update":
            if len(a.members) != 1:
                p.error("relate ATLAS update ID --outcome ... (one relation)")
            r = at.update_relation(a.members[0], **fields, tags=a.tag or None)
        else:
            r = at.relate(a.kind, a.members, rid=a.id, base=a.base,
                          factors=[f for f in a.factors.split(",") if f], **fields, tags=a.tag)
        say(f"{r['id']}: {r['kind']} of {', '.join(m['v'] for m in r['members'])}")
        return 0
    if a.command == "page":
        api.atlas_page(a.atlas, a.out, view=a.view, full=True if a.full else None, offline=a.offline, log=say)
        return 0
    if a.command in ("results", "diff"):
        at = Atlas.open(a.headways[0])
        given = a.headways[1:]
        names = given if given and all(at.has(n) for n in given) else at.select(given) if given else at.names
        if not names:
            p.error(f"{a.command}: no version of the atlas matches {' '.join(given)}")
        ds = [at.entry(n) for n in names]
        if a.command == "results":
            print(bench.table(ds, names, a.heldout))
        else:
            if len(ds) < 2:
                p.error("diff needs two versions or more")
            print(diff.text(ds, names, a.wording))
        return 0
    c = a.atlas_command
    if c == "init":
        at = Atlas.init(a.atlas, a.name)
        say(f"made the atlas {at.meta['name']} in {a.atlas}")
        return 0
    if c == "merge":
        out, notes = Atlas.merge(Atlas.open(a.a), Atlas.open(a.b), a.out, a.prefix)
        for n in notes:
            say(n)
        say(f"wrote {a.out}: {len(out.names)} versions")
        return 0
    at = Atlas.open(a.atlas)
    if c == "import":
        for n in at.import_files(a.files):
            say(n)
    elif c == "scan":
        for n in at.scan() or ["nothing to change"]:
            say(n)
    elif c == "check":
        found = at.check()
        print("\n".join(found) if found else "nothing to report")
    elif c == "validate":
        errs = at.validate()
        print("\n".join(errs) if errs else "valid")
        return 1 if errs else 0
    elif c == "list":
        print(f"{at.meta['name']}: {len(at.names)} versions, {len(at.relations)} relations")
        for n in at.names:
            v = at.meta["versions"][n]
            res = ", ".join(r["suite"] for r in at.results(n))
            print(f"  {n:24s} {v.get('built_at', '')[:16]:16s} {v['model'][7:15]}  {res}")
        for r in at.relations:
            print(f"  {r['id']}: {r['kind']} of {', '.join(m['v'] for m in r['members'])}"
                  + (f" ({r['label']})" if r.get("label") else ""))
        for vn, v in at.meta["views"].items():
            print(f"  view {vn}: {v.get('kind')} {' › '.join(v.get('path', [])) or v.get('query', '')}")
    elif c == "rename":
        at.rename(a.old, a.new)
        say(f"renamed {a.old} to {a.new}")
    elif c == "remove":
        at.remove(a.name)
        say(f"removed {a.name}")
    elif c == "extract":
        names = at.select(a.terms)
        at.extract(a.out, names)
        say(f"wrote {a.out}: {len(names)} versions ({', '.join(names)})")
    elif c == "view":
        # a graph view's query is the viewer's to read (it knows terms the command line does not: contains:, ...)
        names = [] if a.graph else list(a.terms) if all(at.has(t) for t in a.terms) else at.select(a.terms)
        view: dict[str, Any] = {"kind": "graph", "query": " ".join(a.terms)} if a.graph else \
            {"kind": "stack", "path": names}
        if a.baseline:
            view["baseline"] = a.baseline
        if a.suite:
            view["suites"] = a.suite
        at.set_view(a.name, view)
        say(f"view {a.name}: {view.get('query') or ' › '.join(names)}")
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
    head = ref.split("#")[0]                    # pkg.mod:Cls.meth as the trace names it; pkg.mod.func -> pkg.mod:func
    qual = head if ":" in head else ":".join(head.rsplit(".", 1))
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
