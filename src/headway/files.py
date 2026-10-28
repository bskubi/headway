"""Code references into files, in any language: what resolve.py does for Python modules, for a path.

  "R/score.R"                        the whole file
  "R/score.R:score_cells"            a definition (a function, class, method, struct, rule, ...), found with
                                     tree-sitter; "Model.score" for one inside another (a method of a class, a
                                     function in a Rust impl)
  "R/score.R:12-40", "R/score.R:12"  lines; for any file, but not checked: they go stale silently when the file
                                     changes, where a name or an anchor fails the check
  "...#text", "...#from text..to text"   as for Python references, within the file or the definition. On a line
                                     of code (not a comment), "#text" takes the whole construct that starts on that
                                     line: tree-sitter's largest node there, or, for a language it lacks, the line
                                     with the lines indented under it and a closing line ("}", "end", ...)

A path is relative to the working directory, else to the git work tree holding the spec, else to the folder
above the spec's outermost package, else to the spec's folder; or absolute (bases()). Parsers come from tree-sitter-language-pack, which downloads each language's on first use (into its
cache); without one, names do not resolve, and lines and anchors still do.
"""
from __future__ import annotations

import functools
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from headway.resolve import Location, RefError, _anchored

FILENAMES = {"Snakefile": "snakemake"}
EXTENSIONS = {".smk": "snakemake", ".nf": "nextflow"}            # nextflow: no tree-sitter grammar; anchors, lines
COMMENTS = ("#", "//", "--", "%", "/*", ";")                    # a comment line, for languages without a parser
DEFINING = re.compile(r"definition|declaration|item|rule|function|class|method|struct|module|impl|trait|enum|"
                      r"interface|process|macro|type")      # node types a "name" field names a definition in
CLOSERS = ("}", ")", "]", "end", "fi", "done", "esac")


@dataclass
class Definition:
    name: str
    start: int          # 1-based lines, inclusive
    end: int
    rank: int = 0       # where it was found: 0 the tags query, 1 the outline, 2 a "name" field


def bases(spec_file: str | None) -> list[Path]:
    """Where relative paths are looked for: the working directory, the git work tree holding the spec, the folder
    above the spec's outermost package, the spec's own folder."""
    out = [Path.cwd()]
    if spec_file:
        folder = Path(spec_file).parent
        try:
            run = subprocess.run(["git", "-C", str(folder), "rev-parse", "--show-toplevel"], capture_output=True,
                                 text=True, timeout=5)
            if run.returncode == 0 and run.stdout.strip():
                out.append(Path(run.stdout.strip()))
        except (OSError, subprocess.SubprocessError):
            pass
        top = folder
        while (top / "__init__.py").exists():
            top = top.parent
        out += [top, folder]
    return list(dict.fromkeys(p.resolve() for p in out))


def _split(ref: str) -> tuple[str, str, str]:
    """ref -> (path, name or lines, anchor). The anchor goes first: its text may hold ":"."""
    head, _, anchor = ref.partition("#")
    path, _, name = head.partition(":")
    return path, name, anchor


def find(ref: str, where: list[Path]) -> tuple[Path, Path] | None:
    """(the file, the folder it was found from) if the reference names an existing file."""
    path = Path(_split(ref)[0]).expanduser()
    if path.is_absolute():
        return (path, path.parent) if path.is_file() else None
    return next(((b / path, b) for b in where if (b / path).is_file()), None)


def looks_like_file(ref: str) -> bool:
    """Is the reference meant as a path (for the message when it is not found), not a Python module?"""
    path = _split(ref)[0]
    p = Path(path)
    return "/" in path or "\\" in path or p.name in FILENAMES or p.suffix in EXTENSIONS or \
        (p.suffix not in ("", ".py") and _detect(p) is not None)


def language(path: Path) -> str:
    """The file's language, as tree-sitter names it ("" if unknown)."""
    return FILENAMES.get(path.name) or EXTENSIONS.get(path.suffix) or _detect(path) or ""


def _detect(path: Path) -> str | None:
    try:
        import tree_sitter_language_pack as tsl
    except ImportError:
        return None
    try:
        return tsl.detect_language_from_path(str(path))
    except Exception:
        return None


@functools.cache
def _parser(lang: str) -> tuple[Any, Any] | None:
    """(tree-sitter Language, Parser) for a language, or None if there is none (or it cannot be downloaded)."""
    if not lang:
        return None
    try:
        import tree_sitter_language_pack as tsl
        if not tsl.has_language(lang):
            return None
        return tsl.get_language(lang), tsl.get_parser(lang)
    except Exception:
        return None


@functools.cache
def _parsed(file: str, mtime: float, lang: str) -> tuple[Any, list[Definition]] | None:
    """The file's syntax tree and definitions, or None without a parser."""
    lp = _parser(lang)
    if lp is None:
        return None
    text = Path(file).read_bytes()
    tree = lp[1].parse(text)
    return tree, _definitions(text, tree, lang, lp[0])


def _definitions(text: bytes, tree: Any, lang: str, language_: Any) -> list[Definition]:
    """Named definitions, from three sources tree-sitter gives (no grammar-specific code of ours): the grammar's
    tags query (what code navigation uses), the language pack's structure outline, and any node with a "name"
    field of a node whose type says it defines something. The first that names a span wins."""
    import tree_sitter as ts
    import tree_sitter_language_pack as tsl
    found: list[Definition] = []
    spans: set[tuple[int, int]] = set()

    def add(name: str | None, start: int, end: int, rank: int) -> None:
        if name and (start, end) not in spans:
            spans.add((start, end))
            found.append(Definition(name, start, end, rank))
    query = tsl.get_tags_query(lang)
    if query:
        try:
            for _, caps in ts.QueryCursor(ts.Query(language_, query)).matches(tree.root_node):
                names = caps.get("name") or []
                for k, nodes in caps.items():
                    if k.startswith("definition") and names and names[0].text:
                        add(names[0].text.decode(), nodes[0].start_point[0] + 1, nodes[0].end_point[0] + 1, 0)
        except Exception:
            pass
    try:
        result = tsl.process(text.decode(errors="replace"), tsl.ProcessConfig(language=lang, structure=True))

        def outline(items: list[Any]) -> None:
            for it in items:
                add(it.name, it.span.start_line + 1, it.span.end_line + 1, 1)
                outline(it.children)
        outline(result.structure)
    except Exception:
        pass
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        name = node.child_by_field_name("name") if node.is_named else None
        if name is not None and name.text and DEFINING.search(node.type) and "call" not in node.type:
            add(name.text.decode(), node.start_point[0] + 1, node.end_point[0] + 1, 2)
        stack.extend(reversed(node.named_children))
    return sorted(found, key=lambda d: (d.start, -d.end))


def resolve(ref: str, file: Path, base: Path) -> Location:
    path, name, anchor = _split(ref)
    lang = language(file)
    try:
        text = file.read_text()
    except (OSError, UnicodeDecodeError) as e:
        raise RefError(f"{ref}: cannot read {file} ({e})") from e
    lines = text.splitlines(keepends=True)
    parsed = _parsed(str(file), file.stat().st_mtime, lang)
    start, end = 1, len(lines)
    if name and name.replace("-", "").isdigit():
        lo, _, hi = name.partition("-")
        start, end = int(lo), int(hi or lo)
        if not 1 <= start <= end <= len(lines):
            raise RefError(f"{ref}: lines {name}, but {path} has {len(lines)}")
    elif name:
        if parsed is None:
            raise RefError(f"{ref}: no parser for {path} ({lang or 'an unknown language'}), so no names; "
                           f"point to it with \"{path}#text\" or lines (\"{path}:12-40\")")
        d = _named(ref, name, parsed[1])
        start, end = d.start, d.end
    if anchor:
        comment = _comment(parsed, lines)
        part, a = _anchored(ref, lines[start - 1:end], start, anchor, lambda n: comment(start - 1 + n))
        if ".." not in anchor and not comment(a - 1):
            end = min(end, _construct(parsed, lines, a))
            part = lines[a - 1:end]
        start, end = a, a + len(part) - 1
    try:
        relpath = str(file.resolve().relative_to(base))
    except ValueError:
        relpath = str(file)
    return Location(ref, str(file.resolve()), relpath, start, end, "".join(lines[start - 1:end]), lang)


def _named(ref: str, name: str, defs: list[Definition]) -> Definition:
    """The definition a (dotted) name points to: "Model.score" is a "score" inside a "Model"."""
    *outer, last = name.split(".")
    hits = [d for d in defs if d.name == last]
    for o in reversed(outer):
        hits = [d for d in hits if any(p.name == o and p.start <= d.start and d.end <= p.end and p != d for p in defs)]
    hits = [d for d in hits if not any(p is not d and p.start <= d.start and d.end <= p.end for p in hits)]  # outermost
    hits = [d for d in hits if d.rank == min(h.rank for h in hits)]    # a struct (tagged) over its impl (outlined)
    if not hits:
        known = sorted({d.name for d in defs})
        raise RefError(f"{ref}: no definition {name!r}" + (f"; the file defines {', '.join(known[:20])}"
                                                            + (" ..." if len(known) > 20 else "") if known else ""))
    if len(hits) > 1:
        raise RefError(f"{ref}: {len(hits)} definitions named {name!r} (lines "
                       f"{', '.join(f'{d.start}-{d.end}' for d in hits)}); point to one by its lines or a #text anchor")
    return hits[0]


def _comment(parsed: tuple[Any, list[Definition]] | None, lines: list[str]) -> Callable[[int], bool]:
    """Is line n (0-based) a comment? From the syntax tree if there is one, else by its first characters."""
    if parsed is None:
        return lambda n: lines[n].strip().startswith(COMMENTS)
    root = parsed[0].root_node

    def comment(n: int) -> bool:
        if not lines[n].strip():
            return False
        col = len(lines[n]) - len(lines[n].lstrip())
        node = root.descendant_for_point_range((n, col), (n, col + 1))
        return node is not None and "comment" in node.type and node.start_point[0] == n
    return comment


def _construct(parsed: tuple[Any, list[Definition]] | None, lines: list[str], line: int) -> int:
    """The last line of what starts on a line of code (1-based)."""
    row, text = line - 1, lines[line - 1]
    col = len(text) - len(text.lstrip())
    if parsed is not None:
        root = parsed[0].root_node
        node = root.named_descendant_for_point_range((row, col), (row, col))
        while node is not None and node.parent is not None and node.parent != root and \
                tuple(node.parent.start_point) == (row, col):
            node = node.parent
        if node is not None and node != root:
            return int(node.end_point[0]) + 1
    end = line
    for n in range(line, len(lines)):
        s = lines[n].strip()
        indent = len(lines[n]) - len(lines[n].lstrip())
        if not s:
            continue
        if indent > col:
            end = n + 1
            continue
        if indent == col and s.startswith(CLOSERS):
            end = n + 1
        break
    return end
