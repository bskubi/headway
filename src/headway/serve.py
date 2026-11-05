"""headway serve: the viewer on a local web server, rebuilt whenever the spec or the code it describes changes.

Each rebuild runs `headway build` in a fresh Python process (so edited modules are imported anew, and a broken
spec cannot take the server down), then tells the open pages, which swap in the new graph and keep their view. A
failed rebuild leaves the last good graph up, with the error over it.

Watched: every .py file under the spec's top-level package (or the spec file alone, outside a package), and every
file the last build's code references and declarations point to.

headway's own source is watched too (an editable install, while headway itself is being changed): when it changes,
the server restarts itself in place (same command, same port), once the new code imports, and the open pages reload.
"""
from __future__ import annotations

import importlib.util
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import webbrowser
from collections.abc import Callable, Iterable, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from headway import media, page

POLL = 0.4          # seconds between looks at the files
QUIET = 0.3         # seconds of no further change before rebuilding
BOOT = f"{os.getpid()}-{time.time_ns()}"   # this server process: a page that finds another one reloads


class Server:
    def __init__(self, spec: str, build_args: Sequence[str] = (), cwd: str | None = None,
                 compare: Sequence[dict[str, Any]] = ()):
        self.spec, self.build_args = spec, list(build_args)
        self.compare = list(compare)                        # other versions, shown in lanes of their own
        self.cwd = cwd or os.getcwd()
        self.data: dict[str, Any] | None = None             # the last good build
        self.error = ""                                     # the last build's failure, if it failed
        self.version = 0
        self.changed = threading.Condition()
        self.stopping = False
        self.base = self._base_files()
        # the builds' own bytecode cache. Python takes a cached module as current when the source's size and
        # whole-second mtime match, so a same-length edit within a second (p_f -> q_f) would build from the old
        # code; the server drops the cached bytecode of every file it sees change
        self.pycache = tempfile.mkdtemp(prefix="headway-pycache-")

    # ---------- building ----------
    def _base_files(self) -> list[Path]:
        module_name = self.spec.partition(":")[0]
        sys.path.insert(0, self.cwd)
        try:
            top = importlib.util.find_spec(module_name.split(".")[0])
            found = importlib.util.find_spec(module_name)
        except (ImportError, ValueError):
            return []
        finally:
            sys.path.remove(self.cwd)
        if top and top.submodule_search_locations:
            return [p for d in top.submodule_search_locations for p in Path(d).rglob("*.py")]
        return [Path(found.origin)] if found and found.origin else []

    def watched(self) -> set[Path]:
        files = set(self.base)
        if self.data:
            for item in [*self.data["symbols"], *self.data["ops"], *self.data["indices"], *self.data["groups"]]:
                if item.get("decl"):
                    files.add(Path(item["decl"]["file"]))
                files.update(Path(c["file"]) for c in item.get("code", []) if c.get("ok"))
        return files

    def rebuild(self, changed: Iterable[Path] = ()) -> None:
        started = time.monotonic()
        for f in changed:
            for cached in Path(self.pycache).rglob(f"{f.stem}.*.pyc"):
                cached.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "headway.json"
            run = subprocess.run([sys.executable, "-m", "headway.cli", "build", self.spec, "-o", os.devnull,
                                  "--json", str(out), *self.build_args],
                                 cwd=self.cwd, capture_output=True, text=True,
                                 env=os.environ | {"PYTHONPYCACHEPREFIX": self.pycache})
            ok = run.returncode == 0 and out.exists()
            data = json.loads(out.read_text()) if ok else None
        report = run.stderr.strip()
        with self.changed:
            if data is not None:
                self.data, self.error = data | {"live": True, "version": self.version + 1}, ""
                if self.compare:
                    self.data["lanes"] = self.compare
            else:
                self.error = report[-4000:] or f"build exited with {run.returncode}"
            self.version += 1
            self.changed.notify_all()
        took = time.monotonic() - started
        summary = report.splitlines()[0] if ok and report else ("ok" if ok else "FAILED")
        print(f"[{time.strftime('%H:%M:%S')}] build {self.version} in {took:.1f}s: {summary}", file=sys.stderr)
        if not ok:
            print(report, file=sys.stderr)

    def watch(self) -> None:
        def stamps() -> dict[Path, tuple[int, int]]:
            out = {}
            for f in self.watched():
                try:
                    st = f.stat()
                    out[f] = (st.st_mtime_ns, st.st_size)
                except OSError:
                    out[f] = (-1, -1)
            return out
        seen = stamps()
        while not self.stopping:
            time.sleep(POLL)
            now = stamps()
            if now == seen:
                continue
            while True:                                     # let a burst of saves settle
                time.sleep(QUIET)
                settled = stamps()
                if settled == now:
                    break
                now = settled
            self.base = self._base_files()                  # a file may have been added
            self.rebuild([f for f in now if seen.get(f) != now[f]])
            # what was built; a save during the build is seen on the next look. Files the build newly points
            # to are taken as they are.
            seen = now | {f: st for f, st in stamps().items() if f not in now}

    def state(self) -> dict[str, Any]:
        return {"version": self.version, "ok": not self.error, "error": self.error, "boot": BOOT}

    # ---------- serving ----------
    def get(self, h: Handler, target: str) -> None:
        """Answer a GET of `target` (the page, its data, the build state, the stream of rebuilds, a file the page
        shows: one of its documents or linked media, /asset?path=...)."""
        url = urllib.parse.urlsplit(target)
        path = urllib.parse.unquote(url.path)
        if path == "/asset":
            h.send_asset(urllib.parse.parse_qs(url.query).get("path", [""])[0], media.assets(self.data or {}))
        elif path == "/":
            h.send(self.page(), "text/html")
        elif path == "/data.json":
            h.send(json.dumps(self.data), "application/json")
        elif path == "/state.json":
            h.send(json.dumps(self.state()), "application/json")
        elif path == "/events":
            self.events(h)
        else:
            h.send("not found", "text/plain", 404)

    def post(self, h: Handler, path: str) -> None:
        """Answer a POST: /redraft, the spec drafted again from the code with its headway init arguments (the
        rebuild follows, as for any change). A spec edited since it was drafted: 409, unless {"force": true}."""
        if path != "/redraft":
            h.send("not found", "text/plain", 404)
            return
        from headway.draft import DraftEdited, DraftError, redraft
        try:
            force = bool(json.loads(h.body() or "{}").get("force"))
            file = self.file()
            backup = redraft(file, force=force)
        except DraftEdited as e:
            h.send(json.dumps({"edited": True, "message": str(e)}), "application/json", 409)
            return
        except (DraftError, OSError, ValueError) as e:
            h.send(json.dumps({"error": str(e)}), "application/json", 400)
            return
        kept = f" (the edited one kept as {backup})" if backup else ""
        print(f"[{time.strftime('%H:%M:%S')}] redrafted {file}{kept}", file=sys.stderr)
        h.send(json.dumps({"ok": True, "file": str(file), "backup": str(backup) if backup else None}),
               "application/json")

    def file(self) -> Path:
        """The spec's file."""
        sys.path.insert(0, self.cwd)
        try:
            found = importlib.util.find_spec(self.spec.partition(":")[0])
        finally:
            sys.path.remove(self.cwd)
        if not found or not found.origin:
            raise ValueError(f"no file for {self.spec}")
        return Path(found.origin)

    def page(self, lanes: Sequence[dict[str, Any]] = (), library: dict[str, Any] | None = None) -> str:
        """The live page: the last good build, with more versions in lanes (and, served from the library, what
        the viewer needs to show it); before a first good build, the error."""
        if self.data is None:
            return _waiting(self.error)
        d = self.data | {"boot": BOOT} | ({"lanes": [*self.data.get("lanes", []), *lanes]} if lanes else {})
        return page.html(d, library=library)

    def events(self, h: Handler) -> None:
        # server-sent events: the build state now, then on every rebuild (and a heartbeat)
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Cache-Control", "no-store")
        h.end_headers()
        version = -1
        try:
            while not self.stopping:
                with self.changed:
                    if self.version == version:
                        self.changed.wait(15)
                    state = self.state()
                if state["version"] != version:
                    version = state["version"]
                    h.wfile.write(f"data: {json.dumps(state)}\n\n".encode())
                else:
                    h.wfile.write(b": still here\n\n")
                h.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def handler(self) -> type[Handler]:
        server = self

        class One(Handler):
            def do_GET(self) -> None:
                if self.local():
                    server.get(self, self.path)

            def do_POST(self) -> None:
                if self.writable():
                    server.post(self, self.path.split("?")[0])
        return One

    def start(self) -> None:
        """The first build, then watching for changes (in a thread)."""
        self.rebuild()
        threading.Thread(target=self.watch, daemon=True).start()

    def run(self, port: int = 8765, open_browser: bool = False) -> None:
        self.start()
        run(self.handler(), port, open_browser, f"serving {self.spec} at {{url}} (watching {len(self.watched())} "
            "files; ctrl-c to stop)", self.close)

    def close(self) -> None:
        self.stopping = True
        with self.changed:
            self.changed.notify_all()
        shutil.rmtree(self.pycache, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass

    def local(self) -> bool:
        """Is the request addressed to this machine by name (127.0.0.1, localhost)? Another web page could
        reach the server through a name of its own that resolves here (DNS rebinding); it is refused."""
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        if host in ("127.0.0.1", "localhost", "::1"):
            return True
        self.send("only requests to 127.0.0.1 or localhost are answered", "text/plain", 403)
        return False

    def writable(self) -> bool:
        """May this request change files (a POST)? Only from headway's own pages: addressed here by name, with
        headway's header (which a page of another site cannot add without asking, and is not asked), and, if the
        browser says where it comes from, from here."""
        if not self.local():
            return False
        origin = self.headers.get("Origin")
        port = self.headers.get("Host", "").rpartition(":")[2]           # local() vouched for the host
        here = {f"http://{h}:{port}" for h in ("127.0.0.1", "localhost", "[::1]")}
        if self.headers.get("X-Headway") != "1" or origin is not None and origin not in here:
            self.send("only headway's own pages may change files", "text/plain", 403)
            return False
        return True

    def body(self) -> str:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n).decode() if n else ""

    def redirect(self, to: str) -> None:
        self.send_response(302)
        self.send_header("Location", to)
        self.end_headers()

    def send_asset(self, path: str, allowed: set[str]) -> None:
        """A file a page shows (a document, a linked image, sound or video), if the page declares it; with byte
        ranges, so that a video can be sought in."""
        if not path or path not in allowed or not os.path.isfile(path):
            self.send("not a file of this headway", "text/plain", 404)
            return
        size = os.path.getsize(path)
        start, end = 0, size - 1
        ranged = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range") or "")
        if ranged and (ranged.group(1) or ranged.group(2)):
            if ranged.group(1):
                start, end = int(ranged.group(1)), min(int(ranged.group(2) or end), end)
            else:
                start = max(size - int(ranged.group(2)), 0)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        self.send_response(206 if ranged else 200)
        self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        if ranged:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(left, 1 << 20))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def send(self, body: str | bytes, kind: str, status: int = 200) -> None:
        raw = body.encode() if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", kind + "; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)


def run(handler: type[Handler], port: int, open_browser: bool, message: str, close: Callable[[], None],
        start: str = "") -> None:
    """Serve on 127.0.0.1 until ctrl-c; `message` is printed with {url} filled in. The browser opens at `start`."""
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    httpd.daemon_threads = True
    url = f"http://127.0.0.1:{httpd.server_port}/"
    print(message.format(url=url), file=sys.stderr)
    if open_browser:
        webbrowser.open(url + start)
    restart = threading.Event()
    threading.Thread(target=_watch_self, args=(httpd, restart), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        close()
        httpd.server_close()
    if restart.is_set():                                # the same command, on the same port, with the new code
        args = [a for a in sys.argv[1:] if a != "--open"]
        if "--port" not in args and not any(a.startswith("--port=") for a in args):
            args.append(f"--port={httpd.server_port}")
        os.execv(sys.executable, [sys.executable, "-m", "headway.cli", *args])


def boot_events(h: Handler) -> None:
    """Server-sent events: this server process's id, then a heartbeat, for as long as the page listens."""
    h.send_response(200)
    h.send_header("Content-Type", "text/event-stream")
    h.send_header("Cache-Control", "no-store")
    h.end_headers()
    try:
        h.wfile.write(f"data: {json.dumps({'boot': BOOT})}\n\n".encode())
        while True:
            h.wfile.flush()
            time.sleep(15)
            h.wfile.write(b": still here\n\n")
    except (BrokenPipeError, ConnectionResetError, ValueError):
        pass


def _own_files() -> dict[Path, tuple[int, int]]:
    """headway's own source (its modules and the viewer), with their modification stamps."""
    out = {}
    for f in Path(__file__).parent.iterdir():
        if f.suffix in (".py", ".html"):
            try:
                st = f.stat()
                out[f] = (st.st_mtime_ns, st.st_size)
            except OSError:
                pass
    return out


def _watch_self(httpd: ThreadingHTTPServer, restart: threading.Event) -> None:
    """When headway's own source changes and the new code imports, stop serving so that run() restarts."""
    seen = _own_files()
    while True:
        time.sleep(POLL)
        now = _own_files()
        if now == seen:
            continue
        time.sleep(QUIET)
        now = _own_files()
        seen = now
        check = subprocess.run([sys.executable, "-c", "import headway.cli, headway.home, headway.page"],
                               capture_output=True, text=True)
        stamp = time.strftime("%H:%M:%S")
        if check.returncode:
            print(f"[{stamp}] headway's own code changed but does not import; still serving the old one:\n"
                  f"{check.stderr.strip()[-2000:]}", file=sys.stderr)
            continue
        print(f"[{stamp}] headway's own code changed: restarting", file=sys.stderr)
        restart.set()
        httpd.shutdown()
        return


def _waiting(error: str) -> str:
    """The page before the first good build: the error, and a reload once a build succeeds."""
    text = error.replace("&", "&amp;").replace("<", "&lt;")
    return f"""<!doctype html><meta charset="utf-8"><title>headway</title>
<body style="font:14px system-ui;margin:24px"><h2>The spec does not build yet</h2>
<p>Fix it and save; this page reloads when it builds.</p><pre style="white-space:pre-wrap">{text}</pre>
<script>new EventSource("events").onmessage = (e) => {{ if (JSON.parse(e.data).ok) location.reload(); }};</script>"""
