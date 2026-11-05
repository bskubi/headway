"""Documents and media (media.py): documents found where the spec says or where they moved, media written into the
page or linked, excerpts of PDF pages, tables, the errors, and the server sending only a page's own files."""
from __future__ import annotations

import base64
import hashlib
import shutil
import struct
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import zlib
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

import headway
from headway import media
from headway.check import Report

POPPLER = shutil.which("pdftoppm") and shutil.which("pdfinfo")


def pdf(path: Path, pages: int = 2) -> Path:
    """A small PDF (200 x 100 pt pages): the left half of each page black, the right half white."""
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [" + b" ".join(b"%d 0 R" % (3 + 2 * k) for k in range(pages)) + b"] /Count %d >>"
            % pages]
    for k in range(pages):
        stream = b"0 0 0 rg 0 0 100 100 re f"
        objs.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents %d 0 R >>" % (4 + 2 * k))
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out, offsets = b"%PDF-1.4\n", []
    for n, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % x for x in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    path.write_bytes(out)
    return path


def png(path: Path) -> Path:
    raw = b"\x00\xff\x00\x00"                                  # one red pixel
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return path


def size_of(data_url: str) -> tuple[int, int]:
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return struct.unpack(">II", raw[16:24])


def spec(tmp_path: Path, body: str, name: str = "mediaspec") -> headway.Headway:
    (tmp_path / f"{name}.py").write_text("from headway import Headway\nG = Headway('m')\n" + body)
    sys.path.insert(0, str(tmp_path))
    try:
        sys.modules.pop(name, None)
        return __import__(name).G
    finally:
        sys.path.remove(str(tmp_path))


def built(g: headway.Headway) -> tuple[Report, dict]:
    report = headway.check(g)
    from headway import page
    return report, page.data(g, report)


def test_media_written_in_or_linked(tmp_path) -> None:
    png(tmp_path / "fig.png")
    (tmp_path / "t.csv").write_text("a,b\n1,$x$\n")
    (tmp_path / "s.wav").write_bytes(b"RIFF....WAVE")
    g = spec(tmp_path, """
s = G.symbol("s", "s", "s", role="input", media=[G.image("fig.png", "a figure"), G.table("t.csv", "a table"),
    G.audio("s.wav", embed=False), G.video("https://example.org/v.mp4")])
G.group("grp", media=G.table(rows=[["k", "v"], [1, None]], header=True))
G.op("o", "o", inputs=[s], code=[], group="grp")
""")
    report, d = built(g)
    m = d["symbols"][0]["media"]
    assert [x["kind"] for x in m] == ["image", "table", "audio", "video"]
    assert m[0]["src"].startswith("data:image/png;base64,") and m[0]["caption"] == "a figure"
    assert m[1]["rows"] == [["a", "b"], ["1", "$x$"]]
    assert "src" not in m[2] and m[2]["path"] == str(tmp_path / "s.wav")        # linked: the server sends it
    assert m[3]["src"] == "https://example.org/v.mp4"
    assert d["groups"][0]["media"][0]["rows"] == [["k", "v"], ["1", ""]]
    assert media.assets(d) == {str(tmp_path / "s.wav")}                          # what a server may send


def test_media_errors(tmp_path) -> None:
    png(tmp_path / "fig.png")
    g = spec(tmp_path, """
G.op("o", "o", media=[G.image("missing.png"), G.audio("fig.png"), G.image("fig.png", doc="nodoc")])
""")
    errors = headway.check(g).errors
    assert any("missing.png not found" in e for e in errors)
    assert any("fig.png is not a audio file" in e for e in errors)
    assert any("no document 'nodoc'" in e for e in errors)


@pytest.mark.skipif(not POPPLER, reason="needs poppler (pdftoppm, pdfinfo)")
def test_documents_and_excerpts(tmp_path) -> None:
    (tmp_path / "papers").mkdir()
    pdf(tmp_path / "papers" / "paper.pdf")
    digest = hashlib.sha256((tmp_path / "papers" / "paper.pdf").read_bytes()).hexdigest()
    g = spec(tmp_path, f"""
paper = G.document("p", "A paper", path="papers/paper.pdf", url="https://doi.org/x")
moved = G.document("m", "Moved", path="old/paper.pdf")
by_hash = G.document("h", "Renamed", path="gone/renamed.pdf", sha256="{digest}")
lost = G.document("l", "Lost", path="nowhere.pdf")
G.op("o", "o", media=[G.excerpt(paper, 2, (0, 0, 0.5, 1), "left half"), G.excerpt(moved, 1, caption="whole"),
                      G.image("https://example.org/f.png", doc=paper, page=1)])
""")
    report, d = built(g)
    docs = {x["key"]: x for x in d["documents"]}
    assert docs["p"]["found"] and docs["p"]["pages"] == 2 and docs["p"]["sha256"] == digest
    assert docs["m"]["found"] and docs["m"]["moved"] and docs["m"]["path"] == str(tmp_path / "papers" / "paper.pdf")
    assert docs["h"]["found"]                                     # another name: found by its sha256
    assert not docs["l"]["found"]
    assert any("document m: not at" in w for w in report.warnings)
    assert any("document l:" in w and "not found" in w for w in report.warnings)
    assert not any(w.startswith("op o: no code reference") for w in report.warnings)   # a document is its reference
    ex = d["ops"][0]["media"]
    assert ex[0]["excerpt"] and size_of(ex[0]["src"]) == (round(100 * 150 / 72), round(100 * 150 / 72))
    w, h = size_of(ex[1]["src"])                                   # the whole page (pdftoppm rounds up)
    assert abs(w - 200 * 150 / 72) <= 1 and abs(h - 100 * 150 / 72) <= 1
    assert media.assets(d) == {str(tmp_path / "papers" / "paper.pdf")}
    # a page the document does not have; an excerpt of a document not found
    g = spec(tmp_path, """
paper = G.document("p", "A paper", path="papers/paper.pdf")
lost = G.document("l", "Lost", path="nowhere.pdf", url="https://example.org/l.pdf")
G.op("o", "o", media=[G.excerpt(paper, 3), G.excerpt(lost, 1)])
""", "mediaspec2")
    errors = headway.check(g).errors
    assert any("page 3 of document p, which has 2" in e for e in errors)
    assert any("document l's file is needed" in e for e in errors)


def test_excerpt_box_checked() -> None:
    G = headway.Headway("x")
    d = G.document("d", "D", url="https://example.org/d.pdf")
    with pytest.raises(AssertionError):
        G.excerpt(d, 1, (0.5, 0, 0.2, 1))


def test_server_sends_only_its_files(tmp_path) -> None:
    png(tmp_path / "fig.png")
    (tmp_path / "secret.txt").write_text("no")
    (tmp_path / "v.webm").write_bytes(bytes(range(256)) * 4)
    (tmp_path / "srvspec.py").write_text("from headway import Headway\nG = Headway('m')\n"
                                         "G.op('o', 'o', media=[G.video('v.webm', embed=False)])\n")
    from headway.serve import Server
    server = Server("srvspec", cwd=str(tmp_path))
    server.rebuild()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.handler())
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def get(path: Path, headers: dict | None = None) -> tuple[int, bytes]:
        req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_port}/asset?path={path}", headers=headers or {})
        try:
            r = urllib.request.urlopen(req)
            return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, b""
    try:
        assert get(tmp_path / "v.webm") == (200, bytes(range(256)) * 4)
        assert get(tmp_path / "v.webm", {"Range": "bytes=10-19"}) == (206, bytes(range(10, 20)))
        assert get(tmp_path / "secret.txt")[0] == 404 and get(tmp_path / "fig.png")[0] == 404
    finally:
        server.close()
        httpd.shutdown()


def test_render_needs_poppler(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="pdftoppm"):
        media.excerpt(tmp_path / "x.pdf", 1, None, 72)
    _ = subprocess                                               # (poppler is called through subprocess)


def test_citations(tmp_path) -> None:
    g = spec(tmp_path, """
d = G.document("d", "A paper", url="https://doi.org/x")
G.op("o", "o", media=[G.cite(d, 4, "where it says so"), G.cite("nodoc")])
""", "citespec")
    report, data = built(g)
    assert data["ops"][0]["media"] == [{"kind": "cite", "caption": "where it says so", "doc": "d", "page": 4}]
    assert any("no document 'nodoc'" in e for e in report.errors)
    assert not any(w.startswith("op o: no code reference") for w in report.warnings)


# ---------- finding excerpts by their text (pdftext.py) ----------
def text_pdf(path: Path) -> Path:
    """A 400 x 500 pt page: two figure panels (a, b) side by side at the top, each a filled square with a title;
    below, two columns of running text; a second page with one more paragraph."""
    def text(x: float, y: float, s: str, size: float = 9) -> bytes:
        return b"BT /F1 %g Tf %g %g Td (%s) Tj ET\n" % (size, x, 500 - y, s.encode())
    left = ["Alpha beta gamma delta epsilon zeta", "Iota kappa lambda mu nu xi omicron",
            "Sigma tau upsilon phi chi psi omega", "The first column ends here with this"]
    right = ["Second column starts with these here", "More running text in this column now",
             "Gamma delta epsilon zeta eta iota kappa", "And the column ends right here at last"]
    page1 = (b"0 0 0 rg 30 330 120 120 re f 220 330 120 120 re f\n"          # the panels' squares
             + text(20, 40, "a", 14) + text(210, 40, "b", 14)
             + text(40, 44, "Panel title one", 6) + text(230, 44, "Panel title two", 6)
             + b"".join(text(20, 230 + 12 * n, s) for n, s in enumerate(left))
             + b"".join(text(210, 230 + 12 * n, s) for n, s in enumerate(right)))
    page2 = text(20, 60, "Elsewhere a lone sentence about identiﬁcation sits.".replace("ﬁ", "fi"))
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [4 0 R 6 0 R] /Count 2 >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    for content in (page1, page2):
        objs.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 500] /Resources << /Font << /F1 3 0 R >> >> "
                    b"/Contents %d 0 R >>" % (len(objs) + 2))
        objs.append(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
    out, offsets = b"%PDF-1.4\n", []
    for n, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % x for x in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    path.write_bytes(out)
    return path


@pytest.mark.skipif(not (POPPLER and shutil.which("pdftotext")), reason="needs poppler")
def test_find_text_and_panels(tmp_path) -> None:
    from headway import pdftext
    pdf = text_pdf(tmp_path / "t.pdf")
    # anchors: case, spaces and punctuation ignored; once, or an error that says so
    page, first, last = pdftext.find(pdf, "iota KAPPA, lambda")
    assert page == 1 and first.text == "Iota" and last.text == "lambda"
    assert pdftext.find(pdf, "lone sentence")[0] == 2
    with pytest.raises(pdftext.FindError, match="2 times"):
        pdftext.find(pdf, "gamma delta epsilon")
    with pytest.raises(pdftext.FindError, match="not in the document"):
        pdftext.find(pdf, "no such words")
    # between: whole lines of one column
    page, (x0, y0, x1, y1) = pdftext.between(pdf, "Iota kappa lambda", "column ends here with")
    assert page == 1 and x1 < 200 and 230 < y0 < 238 and 262 < y1 < 272     # baselines at 242 and 266
    page, (x0, y0, x1, y1) = pdftext.between(pdf, "More running text", "right here at last")
    assert x0 > 200 and 230 < y0 < 238
    # both columns over the same lines (as a two-column legend): one box across them
    page, (x0, y0, x1, y1) = pdftext.between(pdf, "Alpha beta gamma delta", "right here at last")
    assert x0 < 25 and x1 > 300
    # panels: each square with its letter and title, not its neighbor's
    width, height, words = pdftext.pages(pdf)[0]
    assert sorted(w.text for w in pdftext.labels(words)) == ["a", "b"]
    a = pdftext.panel(pdf, 1, "a")
    b = pdftext.panel(pdf, 1, "b")
    assert a[0] < 25 and 150 <= a[2] < 210 and a[1] < 30 and 165 < a[3] < 200     # the square ends at y = 170
    assert 200 < b[0] < 220 and b[2] >= 340
    with pytest.raises(pdftext.FindError, match="no panel letter 'c'"):
        pdftext.panel(pdf, 1, "c")


@pytest.mark.skipif(not (POPPLER and shutil.which("pdftotext")), reason="needs poppler")
def test_excerpts_found_in_a_build(tmp_path, capsys) -> None:
    text_pdf(tmp_path / "t.pdf")
    g = spec(tmp_path, """
d = G.document("d", "Doc", path="t.pdf")
G.op("o", "o", media=[G.excerpt(d, between=("lone sentence", "sits"), caption="found on page 2"),
                      G.excerpt(d, 1, panel="b"), G.excerpt(d, 1, panel="z"), G.excerpt(d, between=("nowhere", "x"))])
""", "findspec")
    report, data = built(g)
    m = data["ops"][0]["media"]
    assert m[0]["page"] == 2 and m[0]["box"][0] < 0.1 and m[0]["src"].startswith("data:image/png")
    assert m[1]["page"] == 1 and m[1]["box"][0] > 0.5
    assert any("no panel letter 'z'" in e for e in report.errors)
    assert any("'nowhere' is not in the document" in e for e in report.errors)
    with pytest.raises(AssertionError):
        headway.Headway("x").excerpt("d", 1, (0, 0, 1, 1), panel="a")    # one way of saying where
    from headway.cli import main
    assert main(["excerpt", str(tmp_path / "t.pdf"), "--page", "1", "--panels"]) == 0
    assert capsys.readouterr().out.splitlines()[0].startswith("a  box=(")
    out = tmp_path / "x.png"
    assert main(["excerpt", str(tmp_path / "t.pdf"), "--between", "Iota kappa lambda", "column ends here", "-o", str(out)]) == 0
    assert out.read_bytes()[:4] == b"\x89PNG" and "page 1, box=(" in capsys.readouterr().out
