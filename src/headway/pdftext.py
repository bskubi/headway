"""Finding regions of a PDF's pages by what they say: the text between two anchors, a figure's panel by its letter.

An excerpt of a document (G.excerpt) can be given as a box, or found:

  between=("Successful identification", "(Fig. 3a–c).")
      the lines from the one where the first anchor starts to the one where the second ends, in that column (both
      columns, where the text starts in one and ends lower in the other, as a two-column legend does). An anchor is
      matched ignoring case, spaces, punctuation, ligatures and line-break hyphens, so "PBMC25" matches "PBMC" and a
      superscript "25", and "identification" matches "identiﬁcation". The first anchor must occur once (on the page
      given, or in the whole document); the second is its first occurrence after the first.
  panel="c"
      a figure's panel, by its letter: from the letter to the next letters right of it and below it (or where the
      body text starts), with each edge moved to the blank gutter nearest it and off any word it would cut (a word
      mostly inside is kept, a neighbor's is left out), then trimmed to the ink. Panels whose text interleaves can
      still take a strip of a neighbor; give a box for those.

Words and their boxes come from poppler's pdftotext (-bbox), the page's ink from pdftoppm.
"""
from __future__ import annotations

import html
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

Box = tuple[float, float, float, float]                 # x0, y0, x1, y1 in points from the page's top left


class FindError(Exception):
    pass


@dataclass(frozen=True)
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def h(self) -> float:
        return self.y1 - self.y0


def _tool(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise FindError(f"finding text in a PDF needs poppler's {name} (apt install poppler-utils, brew install poppler)")
    return exe


@lru_cache(maxsize=64)
def _pages(path: str, mtime: float) -> tuple[tuple[float, float, tuple[Word, ...]], ...]:
    out = subprocess.run([_tool("pdftotext"), "-bbox", path, "-"], capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        raise FindError(f"pdftotext could not read {path}: {out.stderr.strip()}")
    pages = []
    for m in re.finditer(r'<page width="([\d.]+)" height="([\d.]+)">(.*?)</page>', out.stdout, re.S):
        words = tuple(Word(float(a), float(b), float(c), float(d), html.unescape(t)) for a, b, c, d, t in
                      re.findall(r'xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)<', m.group(3)))
        pages.append((float(m.group(1)), float(m.group(2)), words))
    return tuple(pages)


def pages(path: Path) -> tuple[tuple[float, float, tuple[Word, ...]], ...]:
    """Each page's width, height and words."""
    return _pages(str(path), path.stat().st_mtime)


def norm(s: str) -> str:
    """What anchors are compared by: lower case, letters and digits only (ligatures and accents unfolded)."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s.lower() if c.isalnum())


def _stream(words: tuple[Word, ...]) -> tuple[str, list[int]]:
    """The page's words run together, normalized, and which word each character comes from."""
    text, owner = [], []
    for n, w in enumerate(words):
        t = norm(w.text)
        text.append(t)
        owner += [n] * len(t)
    return "".join(text), owner


def find(path: Path, anchor: str, page: int | None = None) -> tuple[int, Word, Word]:
    """(page, first word, last word) of the anchor's one occurrence."""
    key = norm(anchor)
    if not key:
        raise FindError(f"anchor {anchor!r} has no letters or digits")
    hits = []
    for p, (_, _, words) in enumerate(pages(path), 1):
        if page is not None and p != page:
            continue
        text, owner = _stream(words)
        start = text.find(key)
        while start >= 0:
            hits.append((p, words[owner[start]], words[owner[start + len(key) - 1]]))
            start = text.find(key, start + 1)
    where = f"page {page}" if page else "the document"
    on = "on" if page else "in"
    if not hits:
        raise FindError(f"{anchor!r} is not {on} {where}")
    if len(hits) > 1:
        raise FindError(f"{anchor!r} is {on} {where} {len(hits)} times (pages {', '.join(str(h[0]) for h in hits)}); "
                        "give more of the text")
    return hits[0]


def _after(path: Path, page: int, anchor: str, first: Word) -> Word:
    """The last word of the anchor's first occurrence at or after `first` in reading order."""
    words = pages(path)[page - 1][2]
    text, owner = _stream(words)
    begin = next(n for n, c in enumerate(owner) if words[c] == first)
    at = text.find(norm(anchor), begin)
    if at < 0:
        raise FindError(f"{anchor!r} is not on page {page} after the first anchor")
    return words[owner[at + len(norm(anchor)) - 1]]


def between(path: Path, start: str, end: str, page: int | None = None, pad: float = 1.5) -> tuple[int, Box]:
    """(page, box) of the lines from the one where `start` begins to the one where `end` ends: the passage's words
    in reading order, whole lines of them, in their column; in both columns where it fills both over the same
    height (a two-column legend)."""
    page, s, _ = find(path, start, page)
    e = _after(path, page, end, s)
    width, height, words = pages(path)[page - 1]
    i, j = words.index(s), words.index(e)
    # the passage's own type: reading order can run through a figure's labels on the way
    passage = [w for w in words[i:j + 1] if abs(w.h - s.h) < 0.3 * s.h]
    mid = width / 2
    full = any(w.x0 < mid < w.x1 for w in passage if w.x1 - w.x0 < width / 2)  # a line across the middle: one column
    parts = [passage] if full else [p for p in ([w for w in passage if w.cx < mid], [w for w in passage if w.cx >= mid]) if p]
    spans = [(min(w.y0 for w in p), max(w.y1 for w in p)) for p in parts]
    if len(parts) == 2:
        (a0, a1), (b0, b1) = spans
        overlap = min(a1, b1) - max(a0, b0)
        if overlap < 0.7 * max(a1 - a0, b1 - b0):
            raise FindError(f"the text from {start!r} to {end!r} runs from one column into the other at another "
                            "height: excerpt each column on its own (end the first at its column's last words)")
    lo, hi = (0.0, width) if full or len(parts) == 2 else ((0.0, mid) if parts[0][0].cx < mid else (mid, width))
    top, bottom = min(y0 for y0, _ in spans), max(y1 for _, y1 in spans)
    # whole lines: every word of the column(s) on the lines the passage covers (an anchor can end on a superscript)
    inside = [w for w in words if lo <= w.cx <= hi and top - 0.5 <= (w.y0 + w.y1) / 2 <= bottom + 0.5]
    box = (min(w.x0 for w in inside) - pad, min(w.y0 for w in inside) - pad, max(w.x1 for w in inside) + pad,
           max(w.y1 for w in inside) + pad)
    return page, _clip(box, width, height)


def _clip(b: Box, width: float, height: float) -> Box:
    return (max(b[0], 0.0), max(b[1], 0.0), min(b[2], width), min(b[3], height))


# ---------- panels ----------
DPI = 100


@lru_cache(maxsize=32)
def _ink(path: str, mtime: float, page: int) -> np.ndarray:
    out = subprocess.run([_tool("pdftoppm"), "-gray", "-r", str(DPI), "-f", str(page), "-l", str(page), "-singlefile",
                          path], capture_output=True, timeout=300).stdout
    m = re.match(rb"P5\s+(\d+)\s+(\d+)\s+255\s", out)
    if not m:
        raise FindError(f"pdftoppm could not render page {page} of {path}")
    w, h = int(m.group(1)), int(m.group(2))
    return np.frombuffer(out[m.end():], np.uint8)[:w * h].reshape(h, w) < 225


def ink(path: Path, page: int) -> np.ndarray:
    return _ink(str(path), path.stat().st_mtime, page)


def _px(v: float) -> int:
    return int(round(v * DPI / 72))


def _pt(v: float) -> float:
    return v * 72 / DPI


def _lines(words: tuple[Word, ...]) -> list[list[Word]]:
    rows: list[list[Word]] = []
    for w in sorted(words, key=lambda w: (round(w.y0), w.x0)):
        if rows and abs(rows[-1][0].y0 - w.y0) < 1.5:
            rows[-1].append(w)
        else:
            rows.append([w])
    return rows


def _texty(line: list[Word]) -> bool:
    """A line of prose: several words, packed along a wide line (a figure's rows of ticks and labels are sparse)."""
    extent = max(w.x1 for w in line) - min(w.x0 for w in line)
    return len(line) >= 6 and extent > 120 and sum(w.x1 - w.x0 for w in line) > 0.7 * extent


def _body_height(words: tuple[Word, ...]) -> float:
    """The height of the page's running text: the most common height of words in lines of prose."""
    hs = [round(w.h, 1) for line in _lines(words) if _texty(line) for w in line]
    return max(set(hs), key=hs.count) if hs else 0.0


def _body_lines(words: tuple[Word, ...]) -> list[list[Word]]:
    body = _body_height(words)
    return [line for line in _lines(words) if body and _texty(line) and sum(abs(w.h - body) < 0.4 for w in line) >= 4]


def labels(words: tuple[Word, ...]) -> list[Word]:
    """The words that can be panel letters: one lowercase letter with no word of its size next to it on its line
    (a panel's title, smaller, may follow it), not in the running text, and as large as the largest such letters
    on the page (a figure's letters share a size; axis and legend letters are smaller)."""
    running = [(min(w.x0 for w in line), max(w.x1 for w in line), min(w.y0 for w in line), max(w.y1 for w in line))
               for line in _body_lines(words)]
    alone = []
    for w in words:
        if len(w.text) != 1 or not w.text.islower():
            continue
        near = [o for o in words if o is not w and abs(o.y0 - w.y0) < 2 and o.h > w.h * 0.8
                and (abs(o.x0 - w.x1) < 4 or abs(w.x0 - o.x1) < 4)]
        in_text = any(x0 - 2 <= w.x0 <= x1 + 2 and y0 - 4 <= w.y0 <= y1 + 2 for x0, x1, y0, y1 in running)
        if not near and not in_text:
            alone.append(w)
    if not alone:
        return []
    top = max(w.h for w in alone)
    return [w for w in alone if w.h >= top * 0.85]


def _gutter(lines: np.ndarray, at: int, lo: int, hi: int) -> int:
    idx = np.arange(max(lo, 0), min(hi, len(lines)))
    if not len(idx):
        return at
    blank = idx[lines[idx] == 0]
    if len(blank):
        return int(blank[np.argmin(abs(blank - at))])
    return int(idx[np.argmin(lines[idx] + abs(idx - at) * 1e-3)])


def snap(path: Path, page: int, box: Box, reach: float = 12, limit: Box | None = None) -> Box:
    """Each edge to the blank gutter nearest it (within `reach` points, and inside `limit`), then trimmed to the ink
    inside."""
    m = ink(path, page)
    x0, y0, x1, y1 = map(_px, box)
    lx0, ly0, lx1, ly1 = map(_px, limit or (0, 0, 1e9, 1e9))
    s = _px(reach)
    cols = m[max(y0, 0):y1].sum(0)
    x0, x1 = _gutter(cols, x0, max(x0 - s, lx0), x0 + s), _gutter(cols, x1, x1 - s, min(x1 + s, lx1))
    rows = m[:, max(x0, 0):x1].sum(1)
    y0, y1 = _gutter(rows, y0, max(y0 - s, ly0), y0 + s), _gutter(rows, y1, y1 - s, min(y1 + s, ly1))
    ys, xs = np.nonzero(m[y0:y1, x0:x1])
    if len(xs):
        x0, x1, y0, y1 = x0 + int(xs.min()), x0 + int(xs.max()) + 1, y0 + int(ys.min()), y0 + int(ys.max()) + 1
    return (_pt(x0), _pt(y0), _pt(x1), _pt(y1))


def phrases(words: tuple[Word, ...]) -> list[Word]:
    """Runs of words on one line with word spaces between them (a heading, an axis title), each as one box."""
    out: list[Word] = []
    for line in _lines(words):
        run = [line[0]]
        for w in line[1:]:
            if w.x0 - run[-1].x1 < max(run[-1].h, w.h) * 0.4 and abs(w.h - run[-1].h) < 1.5:
                run.append(w)
            else:
                out.append(Word(run[0].x0, min(r.y0 for r in run), run[-1].x1, max(r.y1 for r in run),
                                " ".join(r.text for r in run)))
                run = [w]
        out.append(Word(run[0].x0, min(r.y0 for r in run), run[-1].x1, max(r.y1 for r in run),
                        " ".join(r.text for r in run)))
    return out


def unclip(words: tuple[Word, ...], box: Box, rounds: int = 60, keep: Word | None = None,
           limit: Box | None = None) -> Box:
    """No phrase cut by an edge: one mostly inside is taken in, one mostly outside left out (by the move that loses
    the least area). The phrase holding `keep` (a panel's letter, with its title) is always taken in; nothing is
    taken in past `limit`."""
    words = tuple(phrases(words))
    lx0, ly0, lx1, ly1 = limit or (-1e9, -1e9, 1e9, 1e9)
    x0, y0, x1, y1 = box
    mine = None
    if keep is not None:
        mine = next((w for w in words if w.x0 <= keep.x0 + 0.1 and keep.x1 <= w.x1 + 0.1 and abs(w.y0 - keep.y0) < 2), keep)
        x0, y0 = max(min(x0, mine.x0 - 0.5), lx0), max(min(y0, mine.y0 - 0.5), ly0)
        x1, y1 = min(max(x1, mine.x1 + 0.5), lx1), min(max(y1, mine.y1 + 0.5), ly1)
        words = tuple(w for w in words if w is not mine)
    ignored: set[int] = set()
    for _ in range(rounds):
        cut = next((n for n, w in enumerate(words) if n not in ignored and w.x0 < x1 and w.x1 > x0 and w.y0 < y1
                    and w.y1 > y0 and not (x0 <= w.x0 and w.x1 <= x1 and y0 <= w.y0 and w.y1 <= y1)), None)
        if cut is None:
            break
        w = words[cut]
        area = (w.x1 - w.x0) * (w.y1 - w.y0)
        inside = (min(w.x1, x1) - max(w.x0, x0)) * (min(w.y1, y1) - max(w.y0, y0))
        hh = y1 - y0
        heading = (w.x0 < x0 - 1 or w.x1 > x1 + 1) and (w.y0 > y0 + 0.75 * hh or w.y1 < y0 + 0.25 * hh)
        if inside > area / 2:                           # ours: taken in, as far as the limit lets it
            g = (max(min(x0, w.x0 - 0.5), lx0), max(min(y0, w.y0 - 0.5), ly0),
                 min(max(x1, w.x1 + 0.5), lx1), min(max(y1, w.y1 + 0.5), ly1))
            whole = g[0] <= w.x0 and g[2] >= w.x1
            if whole and (g[2] - g[0]) <= 1.15 * (x1 - x0) + 4 and (g[3] - g[1]) <= 1.15 * hh + 4:
                x0, y0, x1, y1 = g
                ignored.add(cut)
                continue
            if not heading:
                ignored.add(cut)
                continue                                # (a heading running past the box: cut off below)
        ww, hh = x1 - x0, y1 - y0                       # a neighbor's: left out by moving an edge it crosses,
        moves: list[tuple[float, int, float, float]] = []   # a little (else the cut is left)
        if w.x0 < x0 < w.x1:
            moves.append((w.x1 + 0.5 - x0, 0, w.x1 + 0.5, ww))
        if w.x0 < x1 < w.x1:
            moves.append((x1 - w.x0 + 0.5, 2, w.x0 - 0.5, ww))
        if w.y0 < y0 < w.y1:
            moves.append((w.y1 + 0.5 - y0, 1, w.y1 + 0.5, hh))
        if w.y0 < y1 < w.y1:
            moves.append((y1 - w.y0 + 0.5, 3, w.y0 - 0.5, hh))
        if (w.x0 < x0 or w.x1 > x1) and w.y0 >= y0 and w.y1 <= y1:   # a heading across the box's top or bottom
            if w.y0 > y0 + 0.75 * hh:
                moves.append((y1 - w.y0 + 0.5, 3, w.y0 - 0.5, hh))
            if w.y1 < y0 + 0.25 * hh:
                moves.append((w.y1 + 0.5 - y0, 1, w.y1 + 0.5, hh))
        small = [(d, e, v) for d, e, v, size in moves if d <= max(6.0, 0.08 * size)]
        if mine is not None:                            # never past the panel's own letter and title
            small = [(d, e, v) for d, e, v in small if (e == 0 and v <= mine.x0) or (e == 2 and v >= mine.x1)
                     or (e == 1 and v <= mine.y0) or (e == 3 and v >= mine.y1)]
        if not small:
            ignored.add(cut)
            continue
        _, edge, v = min(small)
        x0, y0, x1, y1 = [v if n == edge else c for n, c in enumerate((x0, y0, x1, y1))]
    return (x0, y0, x1, y1)


def panel(path: Path, page: int, letter: str, pad: float = 2.0) -> Box:
    """The box of a figure's panel on a page, by its letter (see the top)."""
    width, height, words = pages(path)[page - 1]
    marks = labels(words)
    mine = [w for w in marks if w.text == letter]
    if not mine:
        raise FindError(f"no panel letter {letter!r} on page {page} (letters there: "
                        f"{', '.join(sorted({w.text for w in marks})) or 'none'})")
    if len(mine) > 1:
        raise FindError(f"panel letter {letter!r} is on page {page} {len(mine)} times; give a box")
    me = mine[0]
    # where the running text (or the figure's legend) starts below the panel
    legend = [line for line in _lines(words) if len(line) > 1 and re.fullmatch(r"(Fig\.?|Figure|Table)", line[0].text)
              and line[1].text[:1].isdigit()]
    text_top = min((min(w.y0 for w in line) for line in [*_body_lines(words), *legend] if line[0].y0 > me.y1),
                   default=height)

    m = ink(path, page)

    def row_right(a: Word) -> float:                    # the next letter right of a, on its row
        return min((b.x0 for b in marks if b.x0 > a.x1 + 10 and abs(b.y0 - a.y0) < 12), default=width)

    def ends_above(a: Word, y: float) -> bool:          # a's panel stops above y: a blank band across its column
        x0, x1 = _px(a.x0), _px(row_right(a))
        rows = m[max(_px(y - 8), 0):_px(y + 2), x0:x1].sum(1)
        return bool(len(rows)) and bool((rows == 0).any())

    def reaches(a: Word, b: Word) -> bool:              # a's panel goes down past b's letter: no other letter under
        under = min((c.y0 for c in marks if c is not b and c.y0 > a.y1 + 10 and c.x0 < row_right(a) - 10
                     and row_right(c) > a.x0 + 10), default=text_top)   # a (its row overlapping a's) comes first,
        return under > b.y0 + 5 and not ends_above(a, b.y0)              # and no blank band ends it above b

    def right_of(a: Word) -> float:                     # a letter on a's row, or one above whose panel reaches past
        return min((b.x0 for b in marks if b.x0 > a.x1 + 10 and b.y0 < a.y0 + 12
                    and (abs(b.y0 - a.y0) < 12 or reaches(b, a))), default=width)

    right = right_of(me)
    bottom = min((b.y0 for b in marks if b.y0 > me.y1 + 10 and b.x0 < right - 10
                  and (b.x0 >= me.x0 - 10 or right_of(b) > me.x0 + 10)), default=text_top)
    bottom = min(bottom, text_top)
    limit = (me.x0 - 15, me.y0 - 15, right - 0.5, bottom - 0.5)
    box = snap(path, page, (me.x0 - 3, me.y0 - 3, right - 2, bottom - 2), limit=limit)
    box = (box[0] - pad, box[1] - pad, min(box[2] + pad, limit[2]), min(box[3] + pad, limit[3]))
    return _clip(unclip(words, box, keep=me, limit=limit), width, height)
