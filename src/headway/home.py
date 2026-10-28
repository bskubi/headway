"""headway serve with no spec: a home page for a directory.

It lists what is under the directory: the specs (modules that make a Headway or a Suite at their top level), to
serve one live (rebuilt as it and its code change, as `headway serve SPEC` does), and the built headways (pages
from `headway build`, and --json files), grouped by method, to open one or compare several. Built files open in the
current viewer, so an old page gets the viewer's newer features.

    /                           the home page (the directory is looked through again on each visit)
    /view?file=F&compare=G...   a built headway, with others in lanes beside it
    /live/PATH/...              the spec in the .py file PATH, served live (started on the first visit)
"""
from __future__ import annotations

import html
import json
import os
import re
import threading
import urllib.parse
from pathlib import Path
from typing import Any

from headway import api, page
from headway.serve import Handler, Server, run

SKIP = {".git", ".hg", ".venv", "venv", "env", "node_modules", "__pycache__", "site-packages", "dist", "build",
        ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
MAX_FILES = 5000            # files looked at, at most: a home directory is not what this is for
MAX_SPEC = 1_000_000        # bytes; a larger .py file is not read for a spec
# a module-level assignment of a Headway or a Suite: `G = headway.Headway(`, `B = Suite(`
SPEC_LINE = re.compile(r"^\w+\s*(?::[^=\n]+)?=\s*(?:headway\.)?(?:Headway|Suite)\(", re.M)


class Home:
    def __init__(self, root: str | Path, build_args: list[str] | None = None):
        self.root = Path(root).resolve()
        self.build_args = build_args or []
        self.live: dict[str, Server] = {}                   # by the spec file's path under the root
        self.lock = threading.Lock()
        self.seen: dict[Path, tuple[int, dict[str, Any] | None]] = {}     # a file's mtime -> what it is

    # ---------- what is under the root ----------
    def files(self) -> list[Path]:
        out: list[Path] = []
        for d, dirs, names in os.walk(self.root):
            dirs[:] = sorted(x for x in dirs if x not in SKIP and not x.startswith(".") and not x.endswith(".egg-info"))
            out += [Path(d) / n for n in sorted(names) if n.endswith((".html", ".htm", ".json", ".py"))]
            if len(out) > MAX_FILES:
                break
        return out[:MAX_FILES]

    def what(self, f: Path) -> dict[str, Any] | None:
        """A built headway's summary, or a spec's module, or None; remembered until the file changes."""
        try:
            st = f.stat()
        except OSError:
            return None
        if f in self.seen and self.seen[f][0] == st.st_mtime_ns:
            return self.seen[f][1]
        info = self._spec(f, st.st_size) if f.suffix == ".py" else self._built(f)
        self.seen[f] = (st.st_mtime_ns, info)
        return info

    def _spec(self, f: Path, size: int) -> dict[str, Any] | None:
        if size > MAX_SPEC or f.name.startswith("test_") or f.name.endswith("_test.py") or f.name == "conftest.py":
            return None
        try:
            text = f.read_text()
        except (OSError, UnicodeDecodeError):
            return None
        if "headway" not in text or not SPEC_LINE.search(text):
            return None
        module, cwd = module_of(f)
        return {"kind": "spec", "path": self.rel(f), "module": module, "cwd": str(cwd),
                "title": _title(text)}

    def _built(self, f: Path) -> dict[str, Any] | None:
        try:
            with f.open("rb") as fh:
                head = fh.read(4096)
            if f.suffix == ".json" and not head.lstrip().startswith(b"{"):
                return None
            if f.suffix != ".json" and b"<!doctype html>" not in head.lower():
                return None
            d, is_page, _ = api.read(f)
        except (OSError, ValueError, UnicodeDecodeError):
            return None
        return {"kind": "built", "path": self.rel(f), "page": is_page, "title": d.get("title", ""),
                "name": d.get("name") or f.stem, "built": d.get("built", ""), "built_at": d.get("built_at", ""),
                "git": (d.get("source") or {}).get("git"), "keys": d.get("keys") or {},
                "results": [r["id"] for r in d.get("results", [])], "lanes": len(d.get("lanes", [])),
                "suite": "suite" in d, "errors": len(d.get("check", {}).get("errors", []))}

    def rel(self, f: Path) -> str:
        return f.relative_to(self.root).as_posix()

    def under(self, rel: str, suffixes: tuple[str, ...]) -> Path | None:
        """A path given in a request -> the file under the root, if it is one (and nothing outside the root)."""
        f = (self.root / rel).resolve()
        return f if f.is_relative_to(self.root) and f.is_file() and f.suffix in suffixes else None

    # ---------- serving ----------
    def get(self, h: Handler, target: str) -> None:
        url = urllib.parse.urlsplit(target)
        q = urllib.parse.parse_qs(url.query)
        path = urllib.parse.unquote(url.path)
        if path == "/":
            h.send(self.page(), "text/html")
        elif path == "/favicon.ico":
            h.send(b"", "image/x-icon", 204)
        elif path == "/view":
            self.view(h, q.get("file", [""])[0], q.get("compare", []))
        elif path.startswith("/live/"):
            rel, dot, rest = path[len("/live/"):].partition(".py")
            if not dot or not rest:                         # a relative URL in the page needs the slash
                h.send_response(301)
                h.send_header("Location", f"/live/{urllib.parse.quote(rel + dot)}/")
                h.end_headers()
                return
            server = self.server(rel + dot)
            if server is None:
                h.send(f"no spec {rel + dot} under {self.root}", "text/plain", 404)
            else:
                server.get(h, rest)
        else:
            h.send("not found", "text/plain", 404)

    def view(self, h: Handler, rel: str, compare: list[str]) -> None:
        files = [self.under(r, (".html", ".htm", ".json")) for r in [rel, *compare]]
        if not rel or None in files:
            h.send(f"not a built headway under {self.root}: {', '.join([rel, *compare])}", "text/plain", 404)
            return
        try:
            d, _, _ = api.read(files[0])                    # type: ignore[arg-type]
            lanes = d.get("lanes", []) + api.lanes(str(f) for f in files[1:])
        except (OSError, ValueError) as e:
            h.send(str(e), "text/plain", 400)
            return
        h.send(page.html(d | ({"lanes": lanes} if lanes else {})), "text/html")

    def server(self, rel: str) -> Server | None:
        """The live server of the spec in the file `rel`, started on the first visit."""
        with self.lock:
            if rel not in self.live:
                f = self.under(rel, (".py",))
                info = f and self.what(f)
                if not info or info["kind"] != "spec":
                    return None
                server = Server(info["module"], self.build_args, cwd=info["cwd"])
                server.start()
                self.live[rel] = server
            return self.live[rel]

    def handler(self) -> type[Handler]:
        home = self

        class Many(Handler):
            def do_GET(self) -> None:
                home.get(self, self.path)
        return Many

    def run(self, port: int = 8765, open_browser: bool = False) -> None:
        run(self.handler(), port, open_browser, f"serving the headways under {self.root} at {{url}} (ctrl-c to stop)",
            self.close)

    def close(self) -> None:
        for server in self.live.values():
            server.close()

    # ---------- the home page ----------
    def page(self) -> str:
        found = [x for x in map(self.what, self.files()) if x]
        specs = [x for x in found if x["kind"] == "spec"]
        built = [x for x in found if x["kind"] == "built"]
        by_title: dict[str, list[dict[str, Any]]] = {}
        for b in sorted(built, key=lambda b: b["built_at"], reverse=True):
            by_title.setdefault(b["title"] or "(untitled)", []).append(b)
        e = html.escape
        body = ""
        if specs:
            body += ("<h2>Specs</h2><p class=mu>Serve one live: its page is rebuilt whenever the spec or the code it "
                     "points to changes.</p><table><tr><th>module</th><th>file</th><th></th></tr>")
            for s in specs:
                running = "being served" if s["path"] in self.live else ""
                title = f"<div class=mu>{e(s['title'])}</div>" if s["title"] else ""
                body += (f"<tr><td><a href='/live/{_q(s['path'])}/'><b>{e(s['module'])}</b></a>{title}</td>"
                         f"<td class=mu>{e(s['path'])}</td><td class=mu>{running}</td></tr>")
            body += "</table>"
        if by_title:
            body += ("<h2>Built headways</h2><p class=mu>Click a name to open it. Tick several to compare them: the "
                     "most recently built opens with the others in lanes beside it.</p>")
            for title, items in by_title.items():
                body += f"<h3>{e(title)}</h3><table><tr><th></th><th>name</th><th>file</th><th>built</th><th>results</th><th>keys</th></tr>"
                for b in items:
                    git = b["git"] or {}
                    note = [f"{b['lanes']} other versions in it" if b["lanes"] else "",
                            "a benchmark suite" if b["suite"] else "",
                            f"{b['errors']} check errors" if b["errors"] else ""]
                    body += (f"<tr><td><input type=checkbox data-file='{e(b['path'])}' data-at='{e(b['built_at'])}'></td>"
                             f"<td><a href='/view?file={_q(b['path'])}'><b>{e(b['name'])}</b></a>"
                             f"<div class=mu>{e(' · '.join(x for x in note if x))}</div></td>"
                             f"<td class=mu>{e(b['path'])}</td>"
                             f"<td class=mu>{e(b['built'])}{' · ' + e(git.get('describe') or '') if git else ''}</td>"
                             f"<td class=mu>{e(', '.join(b['results']))}</td>"
                             f"<td class=mu>{e(', '.join(f'{k}={_value(v)}' for k, v in b['keys'].items()))}</td></tr>")
                body += "</table>"
        if not found:
            body = (f"<p>Nothing here yet: no spec (a module making a <code>headway.Headway</code>) and no built "
                    f"headway under {e(str(self.root))}.</p><p><code>headway build SPEC -o headway.html</code> builds "
                    "one; <code>headway guide</code> says how to write a spec.</p>")
        return HOME.replace("<!--ROOT-->", e(str(self.root))).replace("<!--BODY-->", body)


def module_of(f: Path) -> tuple[str, Path]:
    """A .py file -> its module name and the directory it imports from: up through the packages it is in."""
    parts, d = [f.stem], f.parent
    while (d / "__init__.py").exists() and d.parent != d:
        parts.insert(0, d.name)
        d = d.parent
    if parts[-1] == "__init__" and len(parts) > 1:
        parts.pop()
    return ".".join(parts), d


def _title(text: str) -> str:
    """The title given to the Headway or Suite, if it is a plain string."""
    m = re.search(r"(?:Headway|Suite)\(\s*(['\"])(.*?)\1", text)
    return m[2] if m else ""


def _value(v: Any) -> str:
    """A key's value as the viewer writes it: true, not True."""
    return v if isinstance(v, str) else json.dumps(v)


def _q(s: str) -> str:
    return html.escape(urllib.parse.quote(s), quote=True)


HOME = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>headway</title>
<style>
:root { --bg: #fbfaf7; --panel: #fff; --fg: #1d1d1f; --muted: #6b6b70; --border: #d6d3cc; --accent: #2f5fd0; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #17171a; --panel: #1f1f23; --fg: #e8e6e1; --muted: #9a9890; --border: #3a3a40; --accent: #7fa2f0; }
}
body { margin: 0; background: var(--bg); color: var(--fg); font: 14px/1.45 system-ui, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 80px; }
h1 { font-size: 20px; margin: 0 0 2px; }
h2 { font-size: 13px; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin: 28px 0 4px; }
h3 { font-size: 15px; margin: 18px 0 6px; }
.mu { color: var(--muted); font-size: 12px; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
table { width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--border); border-radius: 6px; }
th { text-align: left; font-weight: 500; font-size: 12px; color: var(--muted); padding: 6px 10px; border-bottom: 1px solid var(--border); }
td { padding: 6px 10px; border-top: 1px solid var(--border); vertical-align: top; overflow-wrap: anywhere; }
td:first-child:has(input) { width: 20px; }
code { font-size: 13px; }
#bar { position: fixed; left: 0; right: 0; bottom: 0; background: var(--panel); border-top: 1px solid var(--border); padding: 10px 16px; display: none; justify-content: center; gap: 12px; align-items: center; }
#bar.on { display: flex; }
button { font: inherit; padding: 5px 14px; border-radius: 5px; border: 1px solid var(--accent); background: var(--accent); color: #fff; cursor: pointer; }
</style></head>
<body><main>
<h1>headway</h1><div class="mu"><!--ROOT--></div>
<!--BODY-->
</main>
<div id="bar"><span id="count"></span><button id="go">Compare</button></div>
<script>
const boxes = [...document.querySelectorAll("input[data-file]")];
const picked = () => boxes.filter((b) => b.checked).sort((a, b) => b.dataset.at.localeCompare(a.dataset.at));
const refresh = () => {
  const n = picked().length;
  document.getElementById("bar").classList.toggle("on", n > 1);
  document.getElementById("count").textContent = `${n} selected; ${picked()[0]?.closest("tr").querySelector("b").textContent} opens with the others beside it`;
};
boxes.forEach((b) => b.addEventListener("change", refresh));
document.getElementById("go").onclick = () => {
  const [main, ...rest] = picked().map((b) => b.dataset.file);
  location.href = "/view?file=" + encodeURIComponent(main) + rest.map((f) => "&compare=" + encodeURIComponent(f)).join("");
};
</script></body></html>
"""
