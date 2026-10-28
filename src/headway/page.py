"""A spec -> the viewer: one HTML file holding the graph, the legend and the code it points to."""
from __future__ import annotations

import ast
import datetime
import functools
import inspect
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from base64 import b64encode
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from headway.check import Report
from headway.resolve import shortpath
from headway.spec import Decl, Headway

# links that open a file at a line: {file} is the absolute path with forward slashes, starting with "/"
EDITORS = {
    "vscode": ("VS Code", "vscode://file{file}:{line}:{col}"),
    "cursor": ("Cursor", "cursor://file{file}:{line}:{col}"),
    "pycharm": ("PyCharm", "pycharm://open?file={file}&line={line}"),
    "file": ("browser", "file://{file}"),
}


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
    for path in [*collapse, *hide]:
        assert path in groups, f"no group {path!r}; groups: {', '.join(groups)}"
    return {
        "title": g.title,
        "spec": spec,
        "editor": editor_ or editor(),
        "desc": g.desc,
        "built": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "built_at": datetime.datetime.now().isoformat(),        # to order versions by when they were built
        "source": source(g),
        "indices": [vars(ix) | {"decl": decl("index:" + ix.key)} for ix in g.indices.values()],
        "symbols": [vars(s) | {"kind": s.kind, "code": code(s.code), "value": None if s.value is None else str(s.value),
                               "decl": decl("symbol:" + s.key)}
                    for s in g.symbols.values()],
        "ops": [{k: v for k, v in vars(o).items() if k != "bind"} | {
                    "code": code(o.code),
                    "decl": decl("op:" + o.key),
                    "verified": None if o.key not in report.verified else report.verified[o.key][0],
                    "verify_msg": report.verified.get(o.key, (None, ""))[1]}
                for o in g.ops.values()],
        "groups": [vars(gr) | {"collapsed": gr.collapsed or path in collapse, "decl": decl("group:" + path)}
                   for path, gr in groups.items()],
        "hidden": list(hide),
        "check": {"errors": report.errors, "warnings": report.warnings},
    } | ({"suite": suite_data(g, code, decl)} if isinstance(g, Suite) else {})


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
    return re.sub(r'<link rel="stylesheet" href="(https://[^"]+)"(?: media="([^"]+)")?>', style, page)


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
