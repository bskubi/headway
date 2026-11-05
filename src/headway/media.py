"""Documents and media: what a spec's G.document, G.image, G.table, G.audio, G.video and G.excerpt become in the
built headway.

A document is found where the spec says, or else looked for in the spec's project (by name, and by its sha256 when
the spec gives one), and recorded with its content's sha256, so the viewer can recognize the file wherever it is
found later. A medium is written into the page (an image or sound up to EMBED bytes, a table, an excerpt of a PDF
page rendered here) or linked by its path (the viewer loads it from the file, or through the server that serves the
page) or its URL.
"""
from __future__ import annotations

import csv
import hashlib
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
from base64 import b64encode
from pathlib import Path
from typing import TYPE_CHECKING, Any

from headway import pdftext
from headway.spec import Document, Headway, Media

if TYPE_CHECKING:
    from headway.check import Report

EMBED = 8_000_000                   # media up to this size are written into the page (embed=None)
ROWS = 2000                         # a table's rows shown at most
SEARCH = 200_000                    # files looked at, at most, for a moved document
SKIP = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", ".tox"}
KINDS = {"image": {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"},
         "audio": {".mp3", ".wav", ".ogg", ".oga", ".m4a", ".flac", ".aac"},
         "video": {".mp4", ".webm", ".ogv", ".mov", ".m4v"},
         "table": {".csv", ".tsv", ".tab", ".txt"}}
URL = re.compile(r"^(https?|data):", re.I)


def owners(g: Headway) -> list[tuple[str, list[Media]]]:
    """Every object that carries media, by its id in the built data ("op:key", "sym:key", "group:path")."""
    return [(f"op:{o.key}", o.media) for o in g.ops.values()] + \
        [(f"sym:{s.key}", s.media) for s in g.symbols.values()] + \
        [(f"group:{p}", gr.media) for p, gr in g.groups.items()]


def resolve(g: Headway, report: Report) -> None:
    """The documents and every object's media, into report.documents and report.media (errors and warnings too)."""
    report.documents = {d.key: document(d, report) for d in g.documents.values()}
    for owner, items in owners(g):
        if items:
            report.media[owner] = [m for n, item in enumerate(items)
                                   if (m := medium(item, f"{owner.replace(':', ' ', 1)}: medium {n + 1}",
                                                   report.documents, report)) is not None]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def project(base: Path) -> Path:
    """Where to look for a moved file: the git work tree holding `base`, else `base` itself."""
    try:
        top = subprocess.run(["git", "-C", str(base), "rev-parse", "--show-toplevel"], capture_output=True,
                             text=True, timeout=10)
        if top.returncode == 0 and top.stdout.strip():
            return Path(top.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return base


def search(root: Path, name: str, digest: str) -> Path | None:
    """A file named `name` under `root` (with this sha256, when given); else, given a sha256, any file with it and
    the same suffix."""
    seen, same_suffix = 0, []
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP)
        for f in sorted(files):
            seen += 1
            if seen > SEARCH:
                return None
            p = Path(folder) / f
            if f == name and (not digest or sha256(p) == digest):
                return p
            if digest and p.suffix.lower() == Path(name).suffix.lower():
                same_suffix.append(p)
    return next((p for p in same_suffix if sha256(p) == digest), None)


def document(d: Document, report: Report) -> dict[str, Any]:
    out: dict[str, Any] = {"key": d.key, "title": d.title, "desc": d.desc, "url": d.url, "path": "", "found": False,
                           "declared": d.path}
    if not d.path:
        return out
    path = Path(d.path).expanduser()
    path = path if path.is_absolute() else Path(d.base) / path
    if not path.is_file():
        moved = search(project(Path(d.base)), path.name, d.sha256)
        if moved is None:
            report.warnings.append(f"document {d.key}: {path} not found" + (", nor a file with its sha256 in the "
                                   "project" if d.sha256 else " (nor a file of that name in the project)") +
                                   ("; the viewer links its url" if d.url else "; the viewer can be pointed at it"))
            out["sha256"] = d.sha256
            return out
        report.warnings.append(f"document {d.key}: not at {path}; found at {moved} (update path= in the spec)")
        path = moved
        out["moved"] = True
    digest = sha256(path)
    if d.sha256 and d.sha256 != digest:
        report.warnings.append(f"document {d.key}: {path} is not the file of the sha256 the spec gives (its own: "
                               f"{digest})")
    out |= {"path": str(path.resolve()), "found": True, "sha256": digest, "size": path.stat().st_size,
            "kind": mimetypes.guess_type(path.name)[0] or ""}
    if path.suffix.lower() == ".pdf":
        out["pages"] = pdf_pages(path)
    return out


def pdf_pages(path: Path) -> int | None:
    info = _pdfinfo(path)
    m = re.search(r"^Pages:\s+(\d+)", info, re.M)
    return int(m.group(1)) if m else None


def _pdfinfo(path: Path, *args: str) -> str:
    exe = shutil.which("pdfinfo")
    if not exe:
        return ""
    r = subprocess.run([exe, *args, str(path)], capture_output=True, text=True, timeout=60)
    return r.stdout if r.returncode == 0 else ""


def medium(m: Media, where: str, docs: dict[str, dict[str, Any]], report: Report) -> dict[str, Any] | None:
    """One medium as the viewer takes it: {kind, caption, doc, page, and src (written in or a URL) or path (linked)
    or rows}. None, with an error, if it cannot be shown."""
    out: dict[str, Any] = {"kind": m.kind, "caption": m.caption, "doc": m.doc, "page": m.page}
    doc = None
    if m.doc is not None:
        doc = docs.get(m.doc)
        if doc is None:
            report.errors.append(f"{where}: no document {m.doc!r} (declare it with G.document)")
            return None
        if m.page is not None and doc.get("pages") and not 1 <= m.page <= doc["pages"]:
            report.errors.append(f"{where}: page {m.page} of document {m.doc}, which has {doc['pages']}")
            return None
    if m.kind == "cite":
        if doc is None:
            report.errors.append(f"{where}: a citation needs a document")
            return None
        return out
    if m.kind == "excerpt":
        if doc is None or (m.page is None and m.between is None):
            report.errors.append(f"{where}: an excerpt needs a document and a page")
            return None
        if not doc["found"]:
            report.errors.append(f"{where}: document {m.doc}'s file is needed to take an excerpt of it, and it is "
                                 "not found")
            return None
        pdf = Path(doc["path"])
        try:
            page, box = locate(pdf, m)
            png = excerpt(pdf, page, box, m.dpi)
        except (RuntimeError, pdftext.FindError) as e:
            report.errors.append(f"{where}: {e}")
            return None
        return out | {"kind": "image", "excerpt": True, "box": box, "page": page,
                      "src": "data:image/png;base64," + b64encode(png).decode()}
    if m.kind == "table":
        if m.rows is not None:
            rows = [[_cell(c) for c in r] for r in m.rows]
        else:
            found = _file(m, where, report)
            if found is None:
                return None
            rows = read_table(found)
            out["file"] = str(found)                             # written in: not a file the page loads
        if len(rows) > ROWS:
            report.warnings.append(f"{where}: {len(rows)} rows; the first {ROWS} shown")
            rows = rows[:ROWS]
        return out | {"rows": rows, "header": m.header}
    if URL.match(m.src):
        return out | {"src": m.src}
    path = _file(m, where, report)
    if path is None:
        return None
    size = path.stat().st_size
    if m.embed or (m.embed is None and size <= EMBED):
        kind = "image/svg+xml" if path.suffix.lower() == ".svg" else mimetypes.guess_type(path.name)[0] or ""
        if m.embed and size > EMBED:
            report.warnings.append(f"{where}: {path.name} is {size / 1e6:.1f} MB, written into the page all the same")
        return out | {"src": f"data:{kind};base64,{b64encode(path.read_bytes()).decode()}", "path": str(path)}
    return out | {"path": str(path), "mime": mimetypes.guess_type(path.name)[0] or ""}


def _file(m: Media, where: str, report: Report) -> Path | None:
    path = Path(m.src).expanduser()
    path = (path if path.is_absolute() else Path(m.base) / path).resolve()
    if not path.is_file():
        report.errors.append(f"{where}: {path} not found")
        return None
    if path.suffix.lower() not in KINDS[m.kind]:
        report.errors.append(f"{where}: {path.name} is not a {m.kind} file ({', '.join(sorted(KINDS[m.kind]))})")
        return None
    return path


def _cell(c: Any) -> str:
    return "" if c is None else str(c)


def read_table(path: Path) -> list[list[str]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    return [row for row in csv.reader(text.splitlines(), delimiter=delimiter)]


def page_size(path: Path, page: int) -> tuple[float, float, int]:
    """A PDF page's width and height in points, and its rotation."""
    info = _pdfinfo(path, "-f", str(page), "-l", str(page))
    size = re.search(rf"^Page\s+{page} size:\s+([\d.]+) x ([\d.]+)", info, re.M)
    rot = re.search(rf"^Page\s+{page} rot:\s+(\d+)", info, re.M)
    if not size:
        raise RuntimeError(f"could not read the size of page {page} of {path} (poppler's pdfinfo)")
    w, h = float(size.group(1)), float(size.group(2))
    r = int(rot.group(1)) if rot else 0
    return (h, w, r) if r % 180 else (w, h, r)


def excerpt(path: Path, page: int, box: tuple[float, float, float, float] | None, dpi: int) -> bytes:
    """A region of a PDF page as a PNG (poppler's pdftoppm), `box` as fractions of the page as it is shown."""
    exe = shutil.which("pdftoppm")
    if not exe:
        raise RuntimeError("an excerpt of a PDF needs poppler's pdftoppm (apt install poppler-utils, brew install "
                           "poppler)")
    crop: list[str] = []
    if box is not None:
        w, h, _ = page_size(path, page)
        px = [round(f * s * dpi / 72) for f, s in zip(box, (w, h, w, h))]
        crop = ["-x", str(px[0]), "-y", str(px[1]), "-W", str(max(px[2] - px[0], 1)), "-H", str(max(px[3] - px[1], 1))]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "page"
        r = subprocess.run([exe, "-png", "-r", str(dpi), "-f", str(page), "-l", str(page), "-singlefile", *crop,
                            str(path), str(out)], capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or not out.with_suffix(".png").is_file():
            raise RuntimeError(f"pdftoppm could not render page {page} of {path}: {r.stderr.strip()}")
        return out.with_suffix(".png").read_bytes()


def locate(path: Path, m: Media) -> tuple[int, tuple[float, float, float, float] | None]:
    """An excerpt's page and box (fractions of the page): given, or found by its text or its panel letter."""
    if m.between is None and m.panel is None:
        assert m.page is not None
        return m.page, m.box
    if m.between is not None:
        page, pts = pdftext.between(path, *m.between, page=m.page)
    else:
        assert m.page is not None and m.panel is not None
        page, pts = m.page, pdftext.panel(path, m.page, m.panel)
    width, height, _ = pdftext.pages(path)[page - 1]
    return page, (pts[0] / width, pts[1] / height, pts[2] / width, pts[3] / height)


def assets(d: dict[str, Any]) -> set[str]:
    """The files a built headway may have its server send: its documents and its linked media."""
    out = {doc["path"] for doc in d.get("documents", []) if doc.get("path")}
    for kind in ("ops", "symbols", "groups"):
        for x in d.get(kind, []):
            out.update(m["path"] for m in x.get("media", []) if m.get("path") and not m.get("src"))
    for lane in d.get("lanes", []):
        out |= assets(lane)
    return out
