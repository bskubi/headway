"""headway serve: the library, and the headways opened from it.

The library lists what can be opened: the headways opened recently (kept in the user's state directory, so they
are there in every session), and what is in a folder (the launch directory at first; any folder can be browsed):
the specs (modules that make a Headway or a Suite at their top level), to serve one live, and the built headways
(pages from `headway build`, and --json files), to open one or several side by side. The viewer draws it
(viewer.html): the whole page when nothing is open, over the page when something is. Built files open in the
current viewer, so an old page gets the viewer's newer features.

    /                           the library
    /?open=F&open=G...          built headways, the first in the page and the others in lanes beside it
    /live/PATH/...              the spec in the .py file PATH (absolute, or under the launch directory), served
                                live (started on the first visit); ?open=F adds built headways in lanes
    /api/list?dir=D             what is in folder D, and its subfolders
    /api/file?path=F            a built headway, as it is on disk (to open beside the page)
    /api/recent                 what was opened recently

Only requests to 127.0.0.1 or localhost are answered, so another web page cannot use the server to read files.
"""
from __future__ import annotations

import datetime
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
MAX_FILES = 5000            # candidate files under a folder, at most
MAX_ENTRIES = 40_000        # directory entries walked through, at most: a home directory is not what this is for
MAX_DEPTH = 6               # folders below the listed one
MAX_SPEC = 1_000_000        # bytes; a larger .py file is not read for a spec
MAX_RECENT = 60
# a module-level assignment of a Headway or a Suite: `G = headway.Headway(`, `B = Suite(`
SPEC_LINE = re.compile(r"^\w+\s*(?::[^=\n]+)?=\s*(?:headway\.)?(?:Headway|Suite)\(", re.M)
BUILT = (".html", ".htm", ".json")


def cache_file() -> Path:
    state = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(state) / "headway" / "library.json"


def recent_file() -> Path:
    """Where the recently opened headways are kept: $XDG_STATE_HOME/headway/recent.json."""
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state) / "headway" / "recent.json"


class Home:
    def __init__(self, root: str | Path, build_args: list[str] | None = None):
        self.root = Path(root).resolve()
        self.build_args = build_args or []
        self.live: dict[str, Server] = {}                   # by the spec file's absolute path
        self.lock = threading.Lock()
        self.seen: dict[Path, tuple[int, dict[str, Any] | None]] = {}     # a file's mtime -> what it is
        self.cached = 0                                     # entries of `seen` not yet saved
        self._load_seen()

    # ---------- what is in a folder ----------
    def files(self, folder: Path | None = None) -> tuple[list[Path], bool]:
        """The candidate files under a folder (specs and built headways, by their names), and whether the walk
        stopped short (too many files or entries, or folders deeper than MAX_DEPTH)."""
        top = folder or self.root
        out: list[Path] = []
        entries, short = 0, False
        for d, dirs, names in os.walk(top):
            depth = len(Path(d).relative_to(top).parts)
            dirs[:] = sorted(x for x in dirs if _shown(x))
            if depth >= MAX_DEPTH:
                short, dirs[:] = short or bool(dirs), []
            entries += len(names) + len(dirs)
            out += [Path(d) / n for n in sorted(names) if n.endswith((*BUILT, ".py"))]
            if len(out) > MAX_FILES or entries > MAX_ENTRIES:
                return out[:MAX_FILES], True
        return out, short

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
        self.cached += 1
        return info

    # what each file is, kept between runs ($XDG_CACHE_HOME/headway/library.json), so a new server does not
    # parse every file again
    def _load_seen(self) -> None:
        try:
            got = json.loads(cache_file().read_text())
            self.seen = {Path(k): (v[0], v[1]) for k, v in got.items() if isinstance(v, list) and len(v) == 2}
        except (OSError, ValueError, TypeError, IndexError):
            pass

    def save_seen(self) -> None:
        if not self.cached:
            return
        with self.lock:
            self.cached = 0
            try:
                f = cache_file()
                f.parent.mkdir(parents=True, exist_ok=True)
                tmp = f.with_name(f"{f.name}.{os.getpid()}.part")
                tmp.write_text(json.dumps({str(k): list(v) for k, v in list(self.seen.items())}))
                tmp.replace(f)
            except OSError:
                pass

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
        return {"kind": "spec", "abs": str(f), "module": module, "cwd": str(cwd), "title": _title(text),
                "name": module, "built_at": _mtime(f)}

    def _built(self, f: Path) -> dict[str, Any] | None:
        # most .json and .html files are not headways: tell from their bytes before parsing (the slow part)
        try:
            with f.open("rb") as fh:
                head = fh.read(4096)
            if f.suffix == ".json" and not (head.lstrip().startswith(b"{") and b'"title"' in head):
                return None
            if f.suffix != ".json" and (b"<!doctype html>" not in head.lower() or b"\nlet D = {" not in f.read_bytes()):
                return None
            d, is_page, _ = api.read(f)
        except (OSError, ValueError, UnicodeDecodeError):
            return None
        return {"kind": "built", "abs": str(f), "page": is_page, "title": d.get("title", ""),
                "name": d.get("name") or f.stem, "built": d.get("built", ""), "built_at": d.get("built_at", ""),
                "git": (d.get("source") or {}).get("git"), "keys": d.get("keys") or {},
                "results": [r["id"] for r in d.get("results", [])], "lanes": len(d.get("lanes", [])),
                "suite": "suite" in d, "errors": len(d.get("check", {}).get("errors", []))}

    def listing(self, folder: str) -> dict[str, Any]:
        """What the library shows of a folder: its specs and built headways (with their paths from the folder),
        and its subfolders."""
        d = self.path(folder or str(self.root))
        if d is None or not d.is_dir():
            raise FileNotFoundError(f"no folder {folder}")
        found, short = self.files(d)
        items = [x | {"path": Path(x["abs"]).relative_to(d).as_posix()} for x in map(self.what, found) if x]
        self.save_seen()
        try:
            subs = sorted(e.name for e in os.scandir(d) if e.is_dir() and _shown(e.name))
        except OSError:
            subs = []
        return {"dir": str(d), "parent": str(d.parent) if d.parent != d else None, "home": str(Path.home()),
                "root": str(self.root), "folders": subs, "items": items, "short": short}

    def path(self, p: str) -> Path | None:
        """A path from a request: absolute, `~/...`, or relative to the launch directory."""
        if not p:
            return None
        q = Path(p).expanduser()
        return (q if q.is_absolute() else self.root / q).resolve()

    # ---------- recently opened ----------
    def remember(self, files: list[Path]) -> None:
        f = recent_file()
        with self.lock:
            now = datetime.datetime.now().isoformat(timespec="seconds")
            old = [r for r in self._recent() if r["abs"] not in {str(x) for x in files}]
            new = [{"abs": str(x), "at": now} for x in files]
            try:
                f.parent.mkdir(parents=True, exist_ok=True)
                tmp = f.with_name(f.name + ".part")
                tmp.write_text(json.dumps((new + old)[:MAX_RECENT], indent=1))
                tmp.replace(f)
            except OSError:
                pass                                        # recents are a convenience; never fail a page on them

    def _recent(self) -> list[dict[str, Any]]:
        try:
            got = json.loads(recent_file().read_text())
            return [r for r in got if isinstance(r, dict) and isinstance(r.get("abs"), str)]
        except (OSError, ValueError):
            return []

    def recent(self) -> list[dict[str, Any]]:
        """The recently opened headways and specs that are still there, most recent first."""
        out = []
        for r in self._recent():
            info = self.what(Path(r["abs"]))
            if info:
                out.append(info | {"path": _tilde(r["abs"]), "opened_at": r["at"]})
        return out

    # ---------- serving ----------
    def get(self, h: Handler, target: str) -> None:
        url = urllib.parse.urlsplit(target)
        q = urllib.parse.parse_qs(url.query)
        path = urllib.parse.unquote(url.path)
        try:
            if path == "/":
                self.view(h, q.get("open", []))
            elif path == "/favicon.ico":
                h.send(b"", "image/x-icon", 204)
            elif path == "/view":                                   # the old form of /?open=
                h.redirect("/?" + urllib.parse.urlencode([("open", f) for f in q.get("file", []) + q.get("compare", [])]))
            elif path == "/api/list":
                h.send(json.dumps(self.listing(q.get("dir", [""])[0])), "application/json")
            elif path == "/api/recent":
                h.send(json.dumps({"recent": self.recent()}), "application/json")
            elif path == "/api/file":
                f = self.path(q.get("path", [""])[0])
                info = f and f.suffix in BUILT and self.what(f)
                if not f or not info or info["kind"] != "built":
                    raise FileNotFoundError(f"not a built headway: {q.get('path', [''])[0]}")
                self.remember([f])
                h.send(f.read_bytes(), "application/json" if f.suffix == ".json" else "text/html")
            elif path.startswith("/live/"):
                self.serve_live(h, path[len("/live/"):], q.get("open", []))
            else:
                h.send("not found", "text/plain", 404)
        except FileNotFoundError as e:
            h.send(str(e), "text/plain", 404)
        except (OSError, ValueError) as e:
            h.send(str(e), "text/plain", 400)

    def library(self) -> dict[str, Any]:
        """What the viewer is told about the library it is served from."""
        return {"root": str(self.root)}

    def view(self, h: Handler, opened: list[str]) -> None:
        """Built headways in the viewer, the first in the page and the others in lanes; none: the library."""
        if not opened:
            h.send(page.html(None, library=self.library()), "text/html")
            return
        files = [self.path(f) for f in opened]
        bad = [o for o, f in zip(opened, files)
               if not f or f.suffix not in BUILT or (self.what(f) or {}).get("kind") != "built"]
        if bad:
            raise FileNotFoundError(f"not a built headway: {', '.join(bad)}")
        good = [f for f in files if f]
        d, _, _ = api.read(good[0])
        lanes = d.get("lanes", []) + api.lanes(str(f) for f in good[1:])
        self.remember(good)
        h.send(page.html(d | {"opened": str(good[0])} | ({"lanes": lanes} if lanes else {}),
                         library=self.library()), "text/html")

    def serve_live(self, h: Handler, rest: str, opened: list[str]) -> None:
        rel, dot, sub = rest.partition(".py")
        if not dot or not sub:                                  # a relative URL in the page needs the slash
            h.redirect("/live/" + urllib.parse.quote(rel + dot) + "/")
            return
        f = self.path(rel + dot)
        server = self.server(f) if f else None
        if f is None or server is None:
            raise FileNotFoundError(f"no spec {rel + dot}")
        if sub == "/":
            self.remember([f])
            lanes = api.lanes(str(p) for p in map(self.path, opened) if p)
            h.send(server.page(lanes, library=self.library()), "text/html")
        else:
            server.get(h, sub)

    def server(self, f: Path, module: str | None = None, compare: list[dict[str, Any]] | None = None) -> Server | None:
        """The live server of the spec in file `f`, started on the first visit."""
        with self.lock:
            if str(f) not in self.live:
                info = self.what(f)
                if not info or info["kind"] != "spec":
                    return None
                server = Server(module or info["module"], self.build_args, cwd=info["cwd"], compare=compare or ())
                server.start()
                self.live[str(f)] = server
            return self.live[str(f)]

    def live_url(self, f: Path) -> str:
        """The URL of a spec served live: its path under the launch directory, or its absolute path."""
        p = f.relative_to(self.root).as_posix() if f.is_relative_to(self.root) else str(f)
        return "/live/" + urllib.parse.quote(p) + "/"

    def handler(self) -> type[Handler]:
        home = self

        class Library(Handler):
            def do_GET(self) -> None:
                if self.local():
                    home.get(self, self.path)
        return Library

    def run(self, port: int = 8765, open_browser: bool = False, spec: str | None = None,
            compare: list[dict[str, Any]] | None = None) -> None:
        """Serve the library; with a spec (a module name), start it live and open on it."""
        start, message = "", f"the headway library at {{url}}, starting in {self.root} (ctrl-c to stop)"
        if spec:
            f = spec_file(spec, self.root)
            if f is None or self.server(f, module=spec, compare=compare) is None:
                raise SystemExit(f"serve: no spec module {spec} found from {self.root}")
            start = self.live_url(f)[1:]
            message = f"serving {spec} at {{url}}{start} (the library at {{url}}; ctrl-c to stop)"
        run(self.handler(), port, open_browser, message, self.close, start)

    def close(self) -> None:
        for server in self.live.values():
            server.close()


def spec_file(spec: str, cwd: Path) -> Path | None:
    """The file of a spec module ("pkg.mod" or "pkg.mod:attr"), imported from `cwd`."""
    import importlib.util
    import sys
    name = spec.partition(":")[0]
    sys.path.insert(0, str(cwd))
    try:
        found = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    finally:
        sys.path.remove(str(cwd))
    return Path(found.origin).resolve() if found and found.origin else None


def module_of(f: Path) -> tuple[str, Path]:
    """A .py file -> its module name and the directory it imports from: up through the packages it is in."""
    parts, d = [f.stem], f.parent
    while (d / "__init__.py").exists() and d.parent != d:
        parts.insert(0, d.name)
        d = d.parent
    if parts[-1] == "__init__" and len(parts) > 1:
        parts.pop()
    return ".".join(parts), d


def _shown(name: str) -> bool:
    return name not in SKIP and not name.startswith(".") and not name.endswith(".egg-info")


def _title(text: str) -> str:
    """The title given to the Headway or Suite, if it is a plain string."""
    m = re.search(r"(?:Headway|Suite)\(\s*(['\"])(.*?)\1", text)
    return m[2] if m else ""


def _mtime(f: Path) -> str:
    try:
        return datetime.datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return ""


def _tilde(p: str) -> str:
    home = str(Path.home())
    return "~" + p[len(home):] if p == home or p.startswith(home + os.sep) else p
