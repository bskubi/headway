"""A spec -> the viewer: one HTML file holding the graph, the legend and the code it points to."""
from __future__ import annotations

import ast
import datetime
import functools
import importlib.metadata
import inspect
import json
import os
import re
import subprocess
import textwrap
import urllib.parse
import urllib.request
from base64 import b64encode
from collections import Counter
from collections.abc import Callable, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from headway import codedoc, codetex, draft, typeinfo
from headway.check import Report
from headway.resolve import Location, shortpath
from headway.spec import Decl, Headway, Op

# links that open a file at a line: {file} is the absolute path with forward slashes, starting with "/"
EDITORS = {
    "vscode": ("VS Code", "vscode://file{file}:{line}:{col}"),
    "cursor": ("Cursor", "cursor://file{file}:{line}:{col}"),
    "pycharm": ("PyCharm", "pycharm://open?file={file}&line={line}"),
    "file": ("browser", "file://{file}"),
}


def _version() -> str:
    try:
        return importlib.metadata.version("headway")
    except importlib.metadata.PackageNotFoundError:
        return ""


def editor(name: str | None = None) -> dict[str, str]:
    """An editor preset, or a URL template with {file}, {line} and {col}; by default $HEADWAY_EDITOR, else VS Code."""
    name = name or os.environ.get("HEADWAY_EDITOR") or "vscode"
    if name in EDITORS:
        label, url = EDITORS[name]
        return {"name": label, "url": url}
    assert "{file}" in name, f"--editor: one of {', '.join(EDITORS)}, or a URL template with {{file}} (and {{line}}, {{col}})"
    return {"name": "editor", "url": name}


def data(g: Headway, report: Report, collapse: Sequence[str] = (), hide: Sequence[str] = (), *, spec: str = "",
         editor_: dict[str, str] | None = None) -> dict[str, Any]:
    def code(refs: list[str]) -> list[dict[str, Any]]:
        out = []
        for ref in refs:
            loc = report.locations.get(ref)
            out.append({"ref": ref, "ok": loc is not None} | ({} if loc is None else
                       {"file": loc.file, "relpath": loc.relpath, "start": loc.start, "end": loc.end,
                        "source": loc.source, "lang": loc.lang}))
        return out

    def decl(name: str) -> dict[str, Any] | None:
        if name not in g.declared:
            return None
        d = g.declared[name]
        return {"file": d.file, "relpath": shortpath(d.module, d.file), "function": d.function,
                "line": d.line, "col": _column(d.file, d.line, d.col), "fields": _fields(d)}

    from headway.bench import Suite, suite_data
    groups = g.all_groups()
    coded = code_names(g)
    for path in [*collapse, *hide]:
        assert path in groups, f"no group {path!r}; groups: {', '.join(groups)}"
    return {
        "title": g.title,
        "draft": g.draft,
        "hints": g.hints,
        "spec": spec,
        "editor": editor_ or editor(),
        "desc": g.desc,
        "built": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "headway": _version(),                                  # the headway that built it
        "built_at": datetime.datetime.now().isoformat(),        # to order versions by when they were built
        "source": source(g),
        "indices": [vars(ix) | {"decl": decl("index:" + ix.key)} for ix in g.indices.values()],
        "symbols": [vars(s) | {"kind": s.kind, "code": code(s.code), "value": None if s.value is None else str(s.value),
                               "decl": decl("symbol:" + s.key), "code_name": coded[s.key]}
                    for s in g.symbols.values()],
        "ops": [{k: v for k, v in vars(o).items() if k not in ("bind", "names")} | {
                    "code": code(o.code),
                    "code_tex": code_tex(g, o, report),
                    "code_tex_names": code_tex(g, o, report, code_names=True),
                    "code_doc": code_doc(o, report),
                    "code_about": code_about(g, o, report),
                    "relation_views": relation_views(g, o, report),
                    "math_check": math_check(o, report),
                    **condition_of(g, o),
                    "decl": decl("op:" + o.key),
                    "verified": None if o.key not in report.verified else report.verified[o.key][0],
                    "verify_msg": report.verified.get(o.key, (None, ""))[1]}
                for o in g.ops.values()],
        "groups": [vars(gr) | {"collapsed": gr.collapsed or path in collapse, "decl": decl("group:" + path)}
                   for path, gr in groups.items()],
        "hidden": list(hide),
        "check": {"errors": report.errors, "warnings": report.warnings},
    } | ({"suite": suite_data(g, code, decl)} if isinstance(g, Suite) else {})


def code_names(g: Headway) -> dict[str, str]:
    """What the code calls each symbol (the viewer's "Code names"). A drafted spec's keys are its variables' names
    (scale_2, a repeated call's, is scale again). Otherwise: the name the steps' code most often has for it (names=,
    bind=), else a setting's or constant's field (Settings.pcs: pcs), else its key."""
    seen: dict[str, Counter[str]] = {}
    for o in g.ops.values():
        pairs = [*o.bind.items(), *((c, n["symbol"]) for c, n in o.names.items() if "symbol" in n)]
        for code_name, k in pairs:
            if not code_name.startswith("return"):
                seen.setdefault(k, Counter())[code_name] += 1
    out = {}
    for k, s in g.symbols.items():
        stem, _, n = k.rpartition("_")
        if g.draft:
            out[k] = stem if n.isdigit() and stem in g.symbols else k
        elif seen.get(k) and k not in seen[k]:
            out[k] = seen[k].most_common(1)[0][0]
        else:
            out[k] = (s.code and _field(s.code[0])) or k
    return out


def _field(ref: str) -> str | None:
    """The name a code reference gives a value: a field ("pkg.mod:Settings.pcs": pcs) or a module constant
    ("pkg.mod:SD_FLOOR"); not a function, a class or an anchor into one."""
    _, colon, qual = ref.partition(":")
    if not colon or "#" in qual or "/" in ref.partition(":")[0]:
        return None
    last = qual.rsplit(".", 1)[-1]
    return last if "." in qual or last.isupper() else None


def condition_of(g: Headway, o: Op) -> dict[str, Any]:
    """A step's condition as math (its truth values explicit: correct = True), and the symbols it is about."""
    if not o.condition:
        return {"condition_tex": None, "condition_on": []}
    try:
        on = sorted({n.id for n in ast.walk(ast.parse(o.condition, mode="eval")) if isinstance(n, ast.Name)
                     and n.id in g.symbols})
    except SyntaxError:
        return {"condition_tex": None, "condition_on": []}
    names = {k: codetex.Name(g.symbols[k].tex) for k in on}
    return {"condition_tex": codetex.condition(o.condition, names), "condition_on": on}


def code_tex(g: Headway, o: Op, report: Report, code_names: bool = False) -> str | None:
    """The step's first Python code reference, written as math, with its names (bind, then names) written as
    their symbols or TeX and its other names as short symbols; None if it cannot be. code_names: every name as
    the code has it instead (still with the indices its symbol has, for the sums)."""
    loc = next((report.locations.get(r) for r in o.code if report.locations.get(r)), None)
    if loc is None or loc.lang != "python":
        return None
    names = _symbol_names(g, o)
    if code_names:                                      # the same lines: only the names differ, as the code has them
        names = {c: codetex.Name(codetex._ident(c), n.dims) for c, n in names.items() if not c.startswith("return")}
    clean = bool((g.draft or {}).get("clean"))
    inline, defined = _called(loc) if clean else ({}, [])

    try:
        # clean code: its own small loops written where they are called (inline); the other functions of its own
        # it calls defined under it ("where ..."), with what decides whether it is done, as one block
        about = [tex for tex, _ in _deciders(g, o, loc, names)]
        guard = condition_of(g, o)["condition_tex"] if o.condition else None
        main = codetex.transcribe(loc.source, names, imported=codetex.imports(loc.file), file=loc.file,
                                  short=not code_names,
                                  first_line=loc.start, hints=g.hints, kinds=_kinds(loc.file) if g.hints else None,
                                  axes=draft.axes_reader(loc.file) if g.hints else None,
                                  indices={k: ix.tex for k, ix in g.indices.items()}, inline=inline,
                                  flush=bool(defined or about or guard), define=defined, where=about)
        if main is None or not guard:
            return main
        return r"\begin{array}{l}\displaystyle " + main + r" \\ \displaystyle \text{only if } " + guard + \
            r"\end{array}"
    except Exception:                                   # noqa: BLE001 (a construct it does not know: no panel)
        return None


def relation_views(g: Headway, o: Op, report: Report) -> list[dict[str, Any]]:
    """What a step's code declares it computes (headway.clean), for the viewer: the statement as math (an
    objective's argmax under its constraints, an equation's solution) with the target's definition and description,
    and what a run's checks found (.headway-checks.json beside the code, written by the run that checked them)."""
    from headway.resolve import RefError, resolve
    loc = next((report.locations.get(r) for r in o.code if report.locations.get(r)), None)
    names = _symbol_names(g, o)
    out: list[dict[str, Any]] = []
    evidence = _evidence(loc, o.code[0]) if loc is not None and o.code else []
    for r in o.relations:
        view: dict[str, Any] = {"kind": r.get("kind"), "target": r.get("target"), "tex": None, "definition": None,
                                "block": None, "about": None}
        over = list(r.get("over", []))
        given: dict[str, codetex.Name] = {k: v for k, v in names.items() if not k.startswith("return")}
        for k, (v, k2) in enumerate(zip(over, o.outputs)):     # the results it declares, as the step's symbols
            n = names.get("return." + str(k)) or (names.get("return") if len(over) == 1 else None)
            if n is None and k2 in g.symbols:
                n = codetex.Name(g.symbols[k2].tex)
            if n is not None:
                given[v] = n
        try:
            target = resolve(r["target"]) if r.get("target") else None
        except RefError:
            target = None
        if target is not None and target.lang == "python" and r.get("kind") in ("maximizes", "minimizes", "solves"):
            fn = ast.parse(textwrap.dedent(target.source)).body[0]
            if isinstance(fn, ast.FunctionDef):
                params = [a.arg for a in fn.args.args]
                call = codetex._function(fn.name) + r"\left(" + ", ".join(
                    (given.get(p) or codetex.Name(codetex._ident(p))).tex for p in params) + r"\right)"
                variables = [(given.get(v) or codetex.Name(codetex._ident(v))).tex for v in over]
                lhs = variables[0] if len(variables) == 1 else r"\left(" + ", ".join(variables) + r"\right)"
                constraints = []
                for c in r.get("subject_to", []):
                    try:
                        cl = resolve(c)
                    except RefError:
                        continue
                    e = codetex.returned(cl.source, given)
                    constraints.append(f"{e} = 0" if e else codetex._function(c.rpartition(':')[2]) + " = 0")
                under = r",\, ".join(variables) + (r" \,:\, " + r",\ ".join(constraints) if constraints else "")
                if r["kind"] == "solves":
                    view["tex"] = f"{lhs} " + r"\text{ solves } " + call + " = 0"
                else:
                    op = r"\operatorname*{arg\,max}" if r["kind"] == "maximizes" else r"\operatorname*{arg\,min}"
                    view["tex"] = f"{lhs} = {op}_{{{under}}} {call}"
                view["definition"] = codetex.transcribe(target.source, given, short=True, first_line=target.start,
                                                        flush=True, file=target.file)
                # one block, flush left as the algorithm is: the statement, then where its objective is defined
                view["block"] = codetex.transcribe(target.source, given, short=True, first_line=target.start,
                                                   flush=True, before=[view["tex"], r"\text{where}"], file=target.file)
                view["about"] = codedoc.describe(target)
        out.append(view | {"evidence": next((e for e in evidence if e.get("kind") == r.get("kind")
                                             and _same_ref(e.get("target", ""), r.get("target") or "")), None)})
    return out


def _symbol_names(g: Headway, o: Op) -> dict[str, codetex.Name]:
    """How a step's code names are written: as the symbols they hold (names=, bind=), with their indices."""
    names: dict[str, codetex.Name] = {}
    for code_name, n in [*((c, {"symbol": k}) for c, k in o.bind.items()), *o.names.items()]:
        s = g.symbols.get(n.get("symbol", ""))
        if s is not None:
            dims = tuple(g.indices[d].tex if d in g.indices else d for d in s.dims)
            bare = not codetex._scripted(s.tex) or re.fullmatch(r".*_\{?[0-9]+\}?", s.tex) is not None
            names[code_name] = codetex.Name(codetex._indexed_tex(s.tex, dims) if dims and bare else s.tex, dims,
                                            s.per or None)
        elif "tex" in n:
            names[code_name] = codetex.Name(n["tex"])
    for k, out in enumerate(o.outputs):                 # its results, in order, for a declaration's over=
        s = g.symbols.get(out)
        if s is not None:
            dims = tuple(g.indices[d].tex if d in g.indices else d for d in s.dims)
            names["return." + str(k)] = codetex.Name(codetex._indexed_tex(s.tex, dims) if dims and not
                                                     codetex._scripted(s.tex) else s.tex, dims)
    return names


def math_check(o: Op, report: Report) -> dict[str, Any] | None:
    """Whether the math the viewer shows for a step's code computes what the code computes, as the last checking
    run found (headway trace): the run's calls of the step's function, its math run on each (core/verify.py). For a
    block of a function's lines: the function's, whose math the block's is part of."""
    loc = next((report.locations.get(r) for r in o.code if report.locations.get(r)), None)
    if loc is None or loc.lang != "python" or not o.code:
        return None
    e = next((e for e in _evidence(loc, o.code[0]) if e.get("kind") == "math"), None)
    if e is None:
        return None
    return e | {"part": "#" in o.code[0], "function": o.code[0].split("#")[0].rpartition(":")[2]}


def _evidence(loc: Location, ref: str) -> list[dict[str, Any]]:
    """What the last checking run found of a function's declarations (.headway-checks.json beside its code)."""
    try:
        data = json.loads((Path(loc.file).parent / ".headway-checks.json").read_text())
    except (OSError, ValueError):
        return []
    entries = data.get("checks", {}).get(ref.split("#")[0], [])
    return [e | {"run": data.get("run", ""), "when": data.get("when", "")} for e in entries]


def _same_ref(a: str, b: str) -> bool:
    """Two references to one function: "pkg.mod:func" and "pkg.mod.func" alike."""
    return a.replace(":", ".") == b.replace(":", ".")


def code_about(g: Headway, o: Op, report: Report) -> list[dict[str, str]]:
    """What the settings and inputs that decide a step (its condition, or its if's test) are: their symbol, name,
    role and description, shown under its math."""
    loc = next((report.locations.get(r) for r in o.code if report.locations.get(r)), None)
    if loc is None or loc.lang != "python":
        return []
    return [{"tex": tex, "text": text} for tex, text in _deciders(g, o, loc, {})]


def _deciders(g: Headway, o: Op, loc: Location, names: dict[str, codetex.Name]) -> list[tuple[str, str]]:
    """What decides whether a step is done, or which of its cases is: the settings and inputs its condition (or
    its if's test) names, each as (TeX, "κ = correct"; text, "correct: a setting, default True: remove plate
    effects")."""
    keys: list[str] = []
    if o.condition:
        keys += condition_of(g, o)["condition_on"]
    try:
        tree = ast.parse(textwrap.dedent(loc.source))
    except SyntaxError:
        tree = None
    if tree is not None and len(tree.body) == 1 and isinstance(tree.body[0], ast.If):
        for n in ast.walk(tree.body[0].test):
            code = n.id if isinstance(n, ast.Name) else ast.unparse(n) if isinstance(n, ast.Attribute) else None
            k = o.names.get(code or "", {}).get("symbol")
            if k:
                keys.append(k)
    out = []
    for k in dict.fromkeys(keys):
        s = g.symbols.get(k)
        if s is None or s.role not in ("setting", "input", "constant"):
            continue
        tex = names.get(next((c for c, n in o.names.items() if n.get("symbol") == k), ""), codetex.Name(s.tex)).tex
        kind = f"a {s.role}" + (f", default {s.value}" if s.value is not None else "")
        said = "" if not s.desc or s.desc.startswith("TODO") else ": " + s.desc
        out.append((f"{tex} = " + r"\textit{" + codetex._text(s.name) + "}", f"{s.name}: {kind}{said}"))
    return out


def _called(loc: Location, limit: int = 8) -> tuple[dict[str, str], list[tuple[str, int]]]:
    """For clean code: the functions of the same file the referenced code calls (only what its lines call, for a
    block in a function), and those they call, each once in order of first call. Those that iterate (a loop that
    carries values) are written where they are called: ({name: source}); the others are defined under the step's
    equation: [(source, first line)]. Decorators are not calls the code makes."""
    try:
        text = Path(loc.file).read_text()
        tree = ast.parse(text)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return {}, []
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def calls(fn: ast.FunctionDef, lines: tuple[int, int] | None = None) -> list[str]:
        return [n.func.id for st in fn.body for n in ast.walk(st) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and (lines is None or lines[0] <= n.lineno <= lines[1])]
    start = next((n for n in defs.values()
                  if min([n.lineno, *(d.lineno for d in n.decorator_list)]) <= loc.start <= (n.end_lineno or n.lineno)),
                 None)
    if start is None:
        return {}, []
    whole = loc.start <= min([start.lineno, *(d.lineno for d in start.decorator_list)])
    found: list[ast.FunctionDef] = []
    queue = calls(start, None if whole else (loc.start, loc.end))
    while queue and len(found) < limit:
        callee = defs.get(queue.pop(0))
        if callee is not None and callee is not start and callee not in found:
            found.append(callee)
            queue += calls(callee)
    lines = text.splitlines(keepends=True)

    def source(f: ast.FunctionDef) -> str:
        return "".join(lines[f.lineno - 1:f.end_lineno])
    iterates = {f.name for f in found if any(isinstance(n, (ast.For, ast.While)) and draft._carried(n)
                                             for n in ast.walk(f))}
    return ({f.name: source(f) for f in found if f.name in iterates},
            [(source(f), f.lineno) for f in found if f.name not in iterates])


@functools.cache
def _types(folder: Path) -> typeinfo.Types:
    return typeinfo.Types.of_package(folder)


def _kinds(file: str) -> Callable[[int, int, str], str | None] | None:
    """What a type checker (mypy) infers each name of a file is, for the code written as math (with hints): from
    its whole package, the topmost folder of __init__.py files holding it."""
    folder = Path(file).parent
    if not (folder / "__init__.py").exists():
        return None
    while (folder.parent / "__init__.py").exists():
        folder = folder.parent
    types = _types(folder)
    return lambda line, column, name: types.kind(file, line, column, name)


def code_doc(o: Op, report: Report) -> str | None:
    """What the step's first code reference says about itself (its docstring or comments, codedoc.py)."""
    loc = next((report.locations.get(r) for r in o.code if report.locations.get(r)), None)
    try:
        return codedoc.describe(loc) if loc else None
    except Exception:                                   # noqa: BLE001 (an odd file: no description from it)
        return None


def source(g: Headway) -> dict[str, Any]:
    """Where the spec was built from: its file and, if it is in a git repository, the commit, branch and whether
    the work tree had changes. Shown in the viewer and in its prompts; headway does not need git."""
    files = [d.file for d in g.declared.values() if d.function == "<module>"] or [d.file for d in g.declared.values()]
    if not files:
        return {}
    file = max(set(files), key=files.count)
    out: dict[str, Any] = {"file": file}

    def git(*args: str) -> str | None:
        try:
            run = subprocess.run(["git", "-C", str(Path(file).parent), *args], capture_output=True, text=True,
                                 timeout=5)
        except (OSError, subprocess.SubprocessError):
            return None
        return run.stdout.strip() if run.returncode == 0 else None
    commit = git("rev-parse", "HEAD")
    if commit:
        out["git"] = {"commit": commit, "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                      "describe": git("describe", "--tags", "--always", "--dirty"),
                      "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    return out


@functools.cache
def _parsed(file: str, mtime: float) -> tuple[ast.Module | None, list[str]]:
    try:
        text = Path(file).read_text()
        return ast.parse(text), text.splitlines()
    except (OSError, SyntaxError, ValueError):
        return None, []


def _column(file: str, line: int, col: int) -> int:
    """ast's 0-based byte offset -> the 1-based character column editors use."""
    lines = _parsed(file, _mtime(file))[1]
    if not 0 < line <= len(lines):
        return col + 1
    return len(lines[line - 1].encode()[:col].decode(errors="ignore")) + 1


def _mtime(file: str) -> float:
    try:
        return os.stat(file).st_mtime
    except OSError:
        return 0.0


def _fields(d: Decl) -> dict[str, list[int]]:
    """Each argument of the declaring call (tex, desc, inputs, code, ...) -> [line, column] where it starts."""
    tree, _ = _parsed(d.file, _mtime(d.file))
    if tree is None:
        return {}
    span = (d.line, d.col, d.end_line, d.end_col)
    call = next((n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and (n.lineno, n.col_offset, n.end_lineno, n.end_col_offset) == span), None)
    owner = next((c for c in (Headway, *Headway.__subclasses__()) if call and isinstance(call.func, ast.Attribute)
                  and hasattr(c, call.func.attr)), None)      # a Headway method, or a Suite's (metric, ...)
    if call is None or owner is None or not isinstance(call.func, ast.Attribute):
        return {}
    params = [p for p in inspect.signature(getattr(owner, call.func.attr)).parameters if p != "self"]
    args: dict[str, ast.expr] = {}
    for name, arg in zip(params, call.args):
        if isinstance(arg, ast.Starred):
            break
        args[name] = arg
    args.update({kw.arg: kw.value for kw in call.keywords if kw.arg})
    return {name: [a.lineno, _column(d.file, a.lineno, a.col_offset)] for name, a in args.items()}


def html(d: dict[str, Any] | None, offline: bool = False, library: dict[str, Any] | None = None) -> str:
    """The viewer with a built headway in it; with d None, the library (served by home.py, which passes
    `library`: what the viewer needs to list and open headways)."""
    template = resources.files("headway").joinpath("viewer.html").read_text()
    if offline:
        template = inline_assets(template)
    payload = json.dumps(d).replace("</", "<\\/")
    return (template.replace("/*HEADWAY_DATA*/null", payload, 1)
            .replace("/*HEADWAY_LIBRARY*/null", json.dumps(library).replace("</", "<\\/"), 1)
            .replace("<!--TITLE-->", _escape(d["title"] if d else "headway")))


def inline_assets(page: str) -> str:
    """The page with its CDN scripts, stylesheets and KaTeX's fonts written into it, so it works offline.

    Scripts go in as data: URLs rather than inline text: highlight.js holds `<!--` and `<script` in its regexes,
    which would throw the HTML parser off. Only the woff2 fonts are kept; every current browser reads them."""
    def script(m: re.Match[str]) -> str:
        return f'<script src="data:text/javascript;base64,{b64encode(_fetch(m[1])).decode()}"></script>'

    def style(m: re.Match[str]) -> str:
        url, media = m[1], m[2]
        css = _fetch(url).decode()

        def font(f: re.Match[str]) -> str:
            data = b64encode(_fetch(urllib.parse.urljoin(url, f[1]))).decode()
            return f'src:url(data:font/woff2;base64,{data}) format("woff2")'
        css = re.sub(r'src:url\(([^)]+\.woff2)\)[^;}]*', font, css)
        assert "</style" not in css and "url(fonts/" not in css, url
        return f'<style media="{media}">{css}</style>' if media else f"<style>{css}</style>"
    page = re.sub(r'<script src="(https://[^"]+)"></script>', script, page)
    return re.sub(r'<link rel="stylesheet"(?: crossorigin="anonymous")? href="(https://[^"]+)"(?: media="([^"]+)")?>', style, page)


def _fetch(url: str) -> bytes:
    """A pinned CDN file, downloaded once and kept in the user's cache directory."""
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "headway" / "cdn"
    path = cache / url.split("://", 1)[1]
    if not path.exists():
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                body = r.read()
        except OSError as e:
            raise RuntimeError(f"--offline needs {url} once, to cache it, and could not fetch it: {e}") from e
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        part.write_bytes(body)
        part.replace(path)
    return path.read_bytes()


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
