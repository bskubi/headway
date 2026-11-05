"""The notation layer: indices, symbols, operations and groups, declared in a spec module next to the code.

A spec describes a method as a bipartite graph. Symbols are the values (scalars, vectors, matrices, settings, data);
operations turn input symbols into output symbols. Operations point into the code with references
("package.module:qualname", optionally "#from text..to text" for lines inside a function), which the checks
resolve. Groups ("round 1/model") nest operations so the viewer can collapse them.
"""
from __future__ import annotations

import itertools
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

ROLES = ("input", "value", "output", "setting", "constant")
MEDIA = ("image", "table", "audio", "video", "excerpt", "cite")


@dataclass
class Document:
    """A document the headway draws on (a paper, its supplement, a protocol): where it is, to open it at a page."""
    key: str
    title: str
    path: str = ""              # a file, relative to the spec that declares it (or absolute)
    url: str = ""               # where it is online (a DOI link, the PDF's URL)
    desc: str = ""
    sha256: str = ""            # its content, to find and recognize it wherever it has moved (headway build prints it)
    base: str = ""              # the folder of the file that declared it: relative paths start there


@dataclass
class Media:
    """An image, a table, a sound, a video or an excerpt of a document's page, shown in the side panel of the step,
    value or group it is attached to (media=)."""
    kind: str                   # one of MEDIA
    src: str = ""               # a file (relative to the declaring file) or a URL
    caption: str = ""
    doc: str | None = None      # the document it comes from (its key)
    page: int | None = None     # the page of that document (1-based)
    box: tuple[float, float, float, float] | None = None   # an excerpt's region: left, top, right, bottom (0-1)
    rows: list[list[Any]] | None = None                  # a table given inline
    header: bool = True         # a table's first row is its header
    embed: bool | None = None   # written into the page (True), linked (False), or by size (None)
    dpi: int = 150              # an excerpt's resolution
    base: str = ""
    between: tuple[str, str] | None = None               # an excerpt found by its text: first words, last words
    panel: str | None = None                             # an excerpt found by a figure panel's letter


@dataclass
class Index:
    key: str
    tex: str
    name: str
    desc: str = ""
    varies: bool = False        # its size may differ between calls (e.g. a subset of cells); not checked across calls


@dataclass
class Symbol:
    key: str
    tex: str
    name: str
    desc: str = ""
    dims: tuple[str, ...] = ()  # index keys
    per: str | None = None      # an index the symbol has one copy of per value (e.g. per fold)
    role: str = "value"
    value: Any = None           # a setting's default, a constant's value
    code: list[str] = field(default_factory=list)
    was: list[str] = field(default_factory=list)            # earlier keys, for comparing versions (diff.py)
    media: list[Media] = field(default_factory=list)

    @property
    def kind(self) -> str:
        return {0: "scalar", 1: "vector", 2: "matrix"}.get(len(self.dims), "array")


@dataclass
class Op:
    key: str
    label: str
    tex: str = ""
    desc: str = ""
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    code: list[str] = field(default_factory=list)
    group: str = ""
    per: str | None = None
    bind: dict[str, str] = field(default_factory=dict)     # code name -> symbol key, for the shape trace
    where: list[tuple[str, str]] = field(default_factory=list)  # (TeX, definition): symbols used only in this step
    names: dict[str, dict[str, str]] = field(default_factory=dict)  # code name -> {"symbol": key} or {"tex": TeX}
    was: list[str] = field(default_factory=list)            # earlier keys, for comparing versions (diff.py)
    condition: str = ""                 # when it is done: a Python expression of symbols ("correct"); "": always
    relations: list[dict[str, Any]] = field(default_factory=list)   # what it computes: headway.clean's declarations
    media: list[Media] = field(default_factory=list)


@dataclass
class Decl:
    """Where a declaring call (G.op(...) etc.) is in the spec."""
    file: str
    module: str
    function: str               # "<module>", or the spec's helper function the call is in
    line: int                   # the call's span; 1-based lines, 0-based columns (as in ast)
    col: int
    end_line: int
    end_col: int


@dataclass
class Group:
    path: str
    label: str
    desc: str = ""
    collapsed: bool = False
    was: list[str] = field(default_factory=list)            # earlier paths, for comparing versions (diff.py)
    media: list[Media] = field(default_factory=list)


SymbolLike: TypeAlias = "Symbol | str"
MediaLike: TypeAlias = "Media | Sequence[Media]"
IndexLike: TypeAlias = "Index | str"


def _key(x: Symbol | Index | str) -> str:
    return x if isinstance(x, str) else x.key


class Headway:
    """A method's notation and computation graph. Declare with index(), symbol(), setting(), op(), group() and
    @verify; build with `headway build <module>` and check with `headway check <module>`.

    `was` (on symbols, settings, constants, steps and groups) names the keys an object had in earlier versions of
    the spec, so that comparing versions (`headway diff`, the viewer's lanes) matches it across a rename. Several
    keys: renamed more than once, or several earlier objects merged into this one; the same earlier key on several
    objects: one object split into these.

    `draft`: for a spec drafted from the code by `headway init`, its arguments ({"entry": "pkg.mod:func", "depth":
    2}), so that it can be drafted again (`headway redraft`, the live viewer's Redraft).

    `hints`: take the code's type hints as assumptions where the code is written as math ("From the code"): a
    parameter or variable annotated as a number is a scalar (k p, not k ⊙ p)."""

    def __init__(self, title: str, desc: str = "", *, draft: dict[str, Any] | None = None, hints: bool = False):
        self.title, self.desc = title, desc
        self.draft = draft
        self.hints = hints
        self.indices: dict[str, Index] = {}
        self.symbols: dict[str, Symbol] = {}
        self.ops: dict[str, Op] = {}
        self.groups: dict[str, Group] = {}
        self.documents: dict[str, Document] = {}
        self.verifiers: list[tuple[tuple[str, ...], Callable[[], object]]] = []
        self.declared: dict[str, Decl] = {}                     # "op:key" etc. -> where the declaring call is

    def index(self, key: str, tex: str, name: str, desc: str = "", *, varies: bool = False) -> Index:
        assert key not in self.indices, f"index {key!r} declared twice"
        self.indices[key] = ix = Index(key, tex, name, desc, varies)
        self.declared["index:" + key] = _caller()
        return ix

    def symbol(self, key: str, tex: str, name: str, desc: str = "", *, dims: Sequence[IndexLike] = (),
               per: IndexLike | None = None, role: str = "value", value: Any = None,
               code: str | Iterable[str] = (), was: str | Iterable[str] = (), media: MediaLike = ()) -> Symbol:
        assert key not in self.symbols, f"symbol {key!r} declared twice"
        assert role in ROLES, f"symbol {key!r}: role {role!r} not one of {ROLES}"
        self.symbols[key] = s = Symbol(key, tex, name, desc, tuple(map(_key, dims)),
                                       None if per is None else _key(per), role, value, _refs(code), _refs(was),
                                       _media(media))
        self.declared["symbol:" + key] = _caller()
        return s

    def setting(self, key: str, tex: str, name: str, desc: str = "", *, value: Any = None,
                code: str | Iterable[str] = (), was: str | Iterable[str] = (), media: MediaLike = ()) -> Symbol:
        return self.symbol(key, tex, name, desc, role="setting", value=value, code=code, was=was, media=media)

    def constant(self, key: str, tex: str, name: str, desc: str = "", *, value: Any,
                 code: str | Iterable[str] = (), was: str | Iterable[str] = (), media: MediaLike = ()) -> Symbol:
        """A number fixed in the code (a clip bound, a tolerance, an iteration limit), where a setting is one the
        user can change. Point `code` at the line that sets it."""
        return self.symbol(key, tex, name, desc, role="constant", value=value, code=code, was=was, media=media)

    def op(self, key: str, label: str, tex: str = "", desc: str = "", *, inputs: Sequence[SymbolLike] = (),
           outputs: Sequence[SymbolLike] = (), code: str | Iterable[str] = (), group: str = "",
           per: IndexLike | None = None, bind: dict[str, SymbolLike] | None = None,
           where: dict[str, str] | None = None, names: dict[str, Symbol | str] | None = None,
           was: str | Iterable[str] = (), condition: str = "",
           relations: Sequence[dict[str, Any]] = (), media: MediaLike = ()) -> Op:
        r"""`where` defines symbols the equation uses that are not values of the graph, as a paper would after the
        equation: {TeX: definition}, e.g. {r"\mu_f": "mean of $x_{if}$ over the learning cells"}. A value another
        step also uses is better declared as a symbol and listed in both steps' inputs or outputs.
        `names` says how to write the code's names when its code is written as math ("From the code", codetex.py):
        {code name: a symbol (its TeX and indices) or TeX}, e.g. {"z": z, "pseudo": r"\kappa", "return": out}.
        The names in `bind` are written as their symbols too.
        `condition`: when the step is done, if not always: a Python expression whose names are symbol keys,
        "correct" (a setting that is True), "not correct", "mode == 'exact'"; written as math in the viewer.
        `relations`: what the step's code declares it computes (headway.clean: {"kind": "maximizes", "target":
        "pkg.mod:objective", "over": ["b", "d"], "subject_to": [...]}, or "solves", "equals", "approximates"); drafted
        from the code's decorators. The viewer writes the target as math, beside the algorithm, with what a run's
        checks found.
        `media`: images, tables, sounds, videos and excerpts of documents shown in the step's side panel
        (G.image, G.table, G.audio, G.video, G.excerpt); symbols and groups take them too."""
        assert key not in self.ops, f"op {key!r} declared twice"
        self.ops[key] = o = Op(key, label, tex, desc, list(map(_key, inputs)), list(map(_key, outputs)),
                               _refs(code), group, None if per is None else _key(per),
                               {k: _key(v) for k, v in (bind or {}).items()}, list((where or {}).items()),
                               {k: {"tex": v} if isinstance(v, str) else {"symbol": _key(v)}
                                for k, v in (names or {}).items()},
                               _refs(was), condition, [dict(r) for r in relations], _media(media))
        self.declared["op:" + key] = _caller()
        return o

    def group(self, path: str, label: str | None = None, desc: str = "", *, collapsed: bool = False,
              was: str | Iterable[str] = (), media: MediaLike = ()) -> Group:
        self.groups[path] = g = Group(path, label or path.rsplit("/", 1)[-1], desc, collapsed, _refs(was),
                                      _media(media))
        self.declared["group:" + path] = _caller()
        return g

    # ---------- documents and media ----------
    def document(self, key: str, title: str, *, path: str = "", url: str = "", desc: str = "",
                 sha256: str = "") -> Document:
        """A document the headway draws on: a paper, a supplement, a protocol. `path`: its file, relative to this
        spec (or absolute); `url`: where it is online. The viewer opens it (at a page, from a figure taken from it);
        where the file is not at `path`, headway build looks for it in the spec's project (by name, and by `sha256`
        when given), and the viewer can be pointed at it (Locate…) and remembers where it is."""
        assert key not in self.documents, f"document {key!r} declared twice"
        assert path or url, f"document {key!r}: needs a path or a url"
        self.documents[key] = d = Document(key, title, path, url, desc, sha256, _here())
        self.declared["document:" + key] = _caller()
        return d

    def image(self, src: str, caption: str = "", *, doc: Document | str | None = None, page: int | None = None,
              embed: bool | None = None) -> Media:
        """An image (png, jpg, gif, svg, webp): a file, relative to this spec, or a URL. `doc`, `page`: the document
        (and page) it comes from. `embed`: written into the page (True), linked (False), or by size (None: up to
        8 MB written in)."""
        return Media("image", src, caption, _doc(doc), page, embed=embed, base=_here())

    def table(self, src: str = "", caption: str = "", *, rows: Sequence[Sequence[Any]] | None = None,
              header: bool = True, doc: Document | str | None = None, page: int | None = None) -> Media:
        """A table: a CSV or TSV file (relative to this spec), or `rows` given inline (lists of cells; the first row
        its header, unless header=False)."""
        assert bool(src) != (rows is not None), "a table needs a file or rows, not both"
        return Media("table", src, caption, _doc(doc), page, rows=None if rows is None else [list(r) for r in rows],
                     header=header, base=_here())

    def audio(self, src: str, caption: str = "", *, doc: Document | str | None = None, page: int | None = None,
              embed: bool | None = None) -> Media:
        """A sound (mp3, wav, ogg, m4a, flac): a file or a URL. By size (None): up to 8 MB written in."""
        return Media("audio", src, caption, _doc(doc), page, embed=embed, base=_here())

    def video(self, src: str, caption: str = "", *, doc: Document | str | None = None, page: int | None = None,
              embed: bool | None = None) -> Media:
        """A video (mp4, webm, ogv, mov): a file or a URL. By size (None): up to 8 MB written in, else linked."""
        return Media("video", src, caption, _doc(doc), page, embed=embed, base=_here())

    def cite(self, doc: Document | str, page: int | None = None, note: str = "") -> Media:
        """A citation of a document (at a page): listed under Sources in the side panel, with a link that opens it.
        For what a step takes from a document without showing a part of it."""
        return Media("cite", "", note, _doc(doc), page, base=_here())

    def excerpt(self, doc: Document | str, page: int | None = None, box: Sequence[float] | None = None,
                caption: str = "", *, between: tuple[str, str] | None = None, panel: str | None = None,
                dpi: int = 150) -> Media:
        """A region of a page of a PDF document, taken from it as an image when the headway is built (a figure, a
        table, a paragraph of a paper). Say which region by one of:
          box=(left, top, right, bottom)   fractions of the page's width and height from its top left corner
          between=("first words", "last words")
                                           the lines from where the first words start to where the last ones end,
                                           in their column; found anywhere in the document if no page is given
          panel="c"                        a figure's panel, found by its letter on the page
        or none of them: the whole page. Text is matched ignoring case, spaces, punctuation and ligatures (see
        headway.pdftext). `headway excerpt` previews one. Needs poppler (pdftoppm, pdftotext, pdfinfo)."""
        given = [x is not None for x in (box, between, panel)]
        assert sum(given) <= 1, "an excerpt takes one of box=, between= and panel="
        assert page is not None or between is not None, "an excerpt needs a page (unless found by between=)"
        b = None if box is None else tuple(float(x) for x in box)
        assert b is None or (len(b) == 4 and 0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1), \
            f"excerpt box {box}: (left, top, right, bottom), fractions of the page from 0 to 1"
        assert between is None or (len(between) == 2 and all(isinstance(t, str) and t for t in between)), \
            "between=(first words, last words)"
        return Media("excerpt", "", caption, _doc(doc), page, b, dpi=dpi, base=_here(),  # type: ignore[arg-type]
                     between=None if between is None else (between[0], between[1]), panel=panel)

    def verify(self, *ops: str) -> Callable[[Callable[[], object]], Callable[[], object]]:
        """Register a check that the named ops' equations match the code (raise AssertionError if not). Run by
        `headway check` and `headway build`; the viewer marks the ops verified or failed."""
        def register(fn: Callable[[], object]) -> Callable[[], object]:
            self.verifiers.append((ops, fn))
            return fn
        return register

    def all_groups(self) -> dict[str, Group]:
        """The declared groups and every ancestor path of an op's group, parents before children."""
        paths = set(self.groups)
        for o in self.ops.values():
            parts = o.group.split("/") if o.group else []
            paths.update("/".join(parts[:n]) for n in range(1, len(parts) + 1))
        return {p: self.groups.get(p) or Group(p, p.rsplit("/", 1)[-1])
                for p in sorted(paths, key=lambda p: (p.count("/"), list(self.groups).index(p)
                                                      if p in self.groups else len(self.groups), p))}


def _caller() -> Decl:
    """Where a declaration is: the call in the first frame outside headway (in a helper of the spec, the helper's)."""
    f = sys._getframe(1)
    while f.f_back and f.f_globals.get("__name__", "").split(".")[0] == "headway":
        f = f.f_back
    # the exact span of the call being made in that frame
    line, end_line, col, end_col = next(itertools.islice(f.f_code.co_positions(), f.f_lasti // 2, None))
    return Decl(f.f_code.co_filename, f.f_globals.get("__name__", ""), f.f_code.co_name,
                line or f.f_lineno, col or 0, end_line or f.f_lineno, end_col or 0)


def _here() -> str:
    """The folder of the file making the declaration (outside headway): relative paths start there."""
    return os.path.dirname(os.path.abspath(_caller().file))


def _doc(doc: Document | str | None) -> str | None:
    return doc.key if isinstance(doc, Document) else doc


def _media(media: MediaLike) -> list[Media]:
    items = [media] if isinstance(media, Media) else list(media)
    assert all(isinstance(m, Media) for m in items), "media: G.image, G.table, G.audio, G.video, G.excerpt or G.cite"
    return items


def _refs(code: str | Iterable[str]) -> list[str]:
    return [code] if isinstance(code, str) else list(code)
