"""headway serve: the viewer on a local web server, rebuilt whenever the spec or the code it describes changes.

Each rebuild runs `headway build` in a fresh Python process (so edited modules are imported anew, and a broken
spec cannot take the server down), then tells the open pages, which swap in the new graph and keep their view. A
failed rebuild leaves the last good graph up, with the error over it.

Watched: every .py file under the spec's top-level package (or the spec file alone, outside a package), and every
file the last build's code references and declarations point to.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from collections.abc import Iterable, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from headway import page

POLL = 0.4          # seconds between looks at the files
QUIET = 0.3         # seconds of no further change before rebuilding


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
        return {"version": self.version, "ok": not self.error, "error": self.error}

    # ---------- serving ----------
    def handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def send(self, body: str, kind: str, status: int = 200) -> None:
                raw = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", kind + "; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:
                path = self.path.split("?")[0]
                if path == "/":
                    if server.data is None:
                        self.send(_waiting(server.error), "text/html")
                    else:
                        self.send(page.html(server.data), "text/html")
                elif path == "/data.json":
                    self.send(json.dumps(server.data), "application/json")
                elif path == "/state.json":
                    self.send(json.dumps(server.state()), "application/json")
                elif path == "/events":
                    self.events()
                else:
                    self.send("not found", "text/plain", 404)

            def events(self) -> None:
                # server-sent events: the build state now, then on every rebuild (and a heartbeat)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                version = -1
                try:
                    while not server.stopping:
                        with server.changed:
                            if server.version == version:
                                server.changed.wait(15)
                            state = server.state()
                        if state["version"] != version:
                            version = state["version"]
                            self.wfile.write(f"data: {json.dumps(state)}\n\n".encode())
                        else:
                            self.wfile.write(b": still here\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        return Handler

    def run(self, port: int = 8765, open_browser: bool = False) -> None:
        self.rebuild()
        httpd = ThreadingHTTPServer(("127.0.0.1", port), self.handler())
        httpd.daemon_threads = True
        threading.Thread(target=self.watch, daemon=True).start()
        url = f"http://127.0.0.1:{httpd.server_port}/"
        print(f"serving {self.spec} at {url} (watching {len(self.watched())} files; ctrl-c to stop)", file=sys.stderr)
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.close()
            httpd.server_close()

    def close(self) -> None:
        self.stopping = True
        with self.changed:
            self.changed.notify_all()
        shutil.rmtree(self.pycache, ignore_errors=True)


def _waiting(error: str) -> str:
    """The page before the first good build: the error, and a reload once a build succeeds."""
    text = error.replace("&", "&amp;").replace("<", "&lt;")
    return f"""<!doctype html><meta charset="utf-8"><title>headway</title>
<body style="font:14px system-ui;margin:24px"><h2>The spec does not build yet</h2>
<p>Fix it and save; this page reloads when it builds.</p><pre style="white-space:pre-wrap">{text}</pre>
<script>new EventSource("events").onmessage = (e) => {{ if (JSON.parse(e.data).ok) location.reload(); }};</script>"""
