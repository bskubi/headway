"""How easy headway's text is to read: the measures and limits of SKILL.md, over the text people see.

    python .claude/skills/plain-text/readability.py            # the table, for every part
    python .claude/skills/plain-text/readability.py viewer -v  # and every string over a limit, in one part

The parts: viewer (src/headway/viewer.html: tooltips, placeholders, panel text, the tour), cli (strings in the
Python package other than docstrings: help, errors, reports) and guide (src/headway/GUIDE.md, without code).
Standard library only. Syllables are counted by a rule of thumb, so the grade is approximate.
"""
import ast
import html
import re
import statistics
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src" / "headway"
# the limits of SKILL.md: words in a sentence (a tooltip or message; the guide), colons and semicolons in one
LIMIT = {"viewer": 25, "cli": 25, "guide": 35}
JOINS = 1


def syllables(w):
    w = w.lower().strip("'’-")
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    return max(1, len(re.findall(r"[aeiouy]{1,2}", w)))


def words(t):
    return re.findall(r"[A-Za-z][A-Za-z'’]*", t)


def sentences(t):
    """Split at the end of each sentence, and between the items of a list separated by middle dots."""
    return [s for s in re.split(r"(?<=[.!?])\s+|\s+·\s+", t) if words(s)]


# ---------- the text, by part ----------

def _flat(s):
    s = html.unescape(re.sub(r"<[^>]+>", " ", s))
    return re.sub(r"\s+", " ", s).strip()


def templates(src):
    """The text of every JavaScript template literal in src, nested ones included, each ${...} written as X.
    A pattern cannot pair the backticks of a template inside another template's ${...}; this reads them in order."""
    out = []

    def quoted(i, q):                           # past the closing quote of a string that starts at i
        i += 1
        while i < len(src) and src[i] != q:
            i += 2 if src[i] == "\\" else 1
        return i + 1

    def template(i):                            # i: just past the opening backtick; returns the index past the closing one
        text = []
        while i < len(src):
            c = src[i]
            if c == "\\":
                text.append(src[i:i + 2]); i += 2
            elif c == "`":
                out.append("".join(text)); return i + 1
            elif src.startswith("${", i):
                i, depth = i + 2, 1
                while i < len(src) and depth:
                    c = src[i]
                    if c == "`": i = template(i + 1)
                    elif c in "\"'": i = quoted(i, c)
                    else:
                        depth += {"{": 1, "}": -1}.get(c, 0); i += 1
                text.append(" X ")
            else:
                text.append(c); i += 1
        return i

    for script in re.findall(r"<script[^>]*>(.*?)</script>", src, flags=re.S):
        src, i = script, 0
        while i < len(src):
            c = src[i]
            if c == "`": i = template(i + 1)
            elif c in "\"'": i = quoted(i, c)
            elif src.startswith("//", i): i = src.find("\n", i) % (len(src) + 1) or len(src)
            elif src.startswith("/*", i): i = src.find("*/", i) + 2 if "*/" in src[i:] else len(src)
            else: i += 1
    return out


# the prompts the viewer writes for an AI agent: written for a model, so held to precision rather than these limits
# (from the first marker to the second; the short double-quoted strings there, their buttons' tooltips, stay in)
PROMPTS = [("const PRESETS = [", "const ask = {"), ("const wholePrompt", "/* ---------- legend"),
           ("const RESULT_ASKS = [", "function resultsAskHTML"), ("const RESULTS_CONTEXT", "function setView")]


def viewer():
    v = re.sub(r"<style>.*?</style>", "", (SRC / "viewer.html").read_text(), flags=re.S)
    for a, b in PROMPTS:
        i, j = v.index(a), v.index(b)
        keep = re.findall(r'"[^"\n]{12,89}"', v[i:j])         # the buttons' tooltips; the prompts go
        v = v[:i] + "\n".join(keep) + v[j:]
    out = [m.group(1) for m in re.finditer(r'(?:title|placeholder|aria-label)="([^"$]{3,})"', v)]
    for t in templates(v) + [m.group(1) for m in re.finditer(r'"((?:[^"\\\n]|\\.){12,})"', v)]:
        t = _flat(t)
        if len(words(t)) >= 4 and " " in t and not re.search(r"[{};=]|=>|\(\)|\[data-", t):
            out.append(t)
    return list(dict.fromkeys(html.unescape(t) for t in out))


def cli():
    out = []
    for f in sorted(SRC.rglob("*.py")):
        tree = ast.parse(f.read_text())
        docs = {id(n.body[0].value) for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        for n in ast.walk(tree):
            if isinstance(n, ast.JoinedStr):
                t = " ".join(p.value if isinstance(p, ast.Constant) else "X" for p in n.values)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
                t = n.value
            else:
                continue
            t = re.sub(r"\s+", " ", t).strip()
            if len(words(t)) >= 5 and " " in t and not re.search(r"[{}<>]|^\w+\(|def |import ", t):
                out.append(t)
    return list(dict.fromkeys(out))


def guide():
    g = re.sub(r"```.*?```", "", (SRC / "GUIDE.md").read_text(), flags=re.S)
    paras = (re.sub(r"`[^`]*`", "X", p) for p in re.split(r"\n\s*\n", g))
    return [re.sub(r"\s+", " ", p).strip() for p in paras if p.strip() and not p.lstrip().startswith(("#", "|"))]


PARTS = {"viewer": viewer, "cli": cli, "guide": guide}


# ---------- the measures ----------

def measure(items, limit):
    S = [s for t in items for s in sentences(t)]
    W = [w for s in S for w in words(s)]
    n = [len(words(s)) for s in S]
    wps, spw = len(W) / len(S), sum(map(syllables, W)) / len(W)
    return {
        "strings": len(items), "words": len(W),
        "grade": .39 * wps + 11.8 * spw - 15.59,                  # Flesch-Kincaid
        "words/sentence": wps,
        "over limit": sum(k > limit for k in n) / len(n),
        "joins/sentence": sum(len(re.findall(r"[:;]", s)) for s in S) / len(S),
        "brackets/sentence": sum(s.count("(") for s in S) / len(S),
    }


def over(items, limit):
    """The sentences over a limit: too long, or joined by more than JOINS colons and semicolons."""
    for t in items:
        for s in sentences(t):
            k, j = len(words(s)), len(re.findall(r"[:;]", s))
            if k > limit or j > JOINS:
                yield k, j, s


def main(argv):
    parts = [a for a in argv if a in PARTS] or list(PARTS)
    print(f"{'part':8} {'strings':>7} {'words':>6} {'grade':>5} {'w/sent':>6} {'over':>5} {'joins':>5} {'()':>5}")
    for p in parts:
        m = measure(PARTS[p](), LIMIT[p])
        print(f"{p:8} {m['strings']:7} {m['words']:6} {m['grade']:5.1f} {m['words/sentence']:6.1f} "
              f"{100 * m['over limit']:4.0f}% {m['joins/sentence']:5.2f} {m['brackets/sentence']:5.2f}")
    if "-v" in argv:
        for p in parts:
            bad = sorted(over(PARTS[p](), LIMIT[p]), key=lambda x: -x[0])
            print(f"\n== {p}: {len(bad)} sentences over a limit ({LIMIT[p]} words, {JOINS} colon or semicolon)")
            for k, j, s in bad:
                print(f"  [{k}w {j}j] {s}")
    if "--median" in argv:
        for p in parts:
            g = [measure([t], LIMIT[p])["grade"] for t in PARTS[p]() if len(words(t)) >= 12]
            print(f"{p}: median grade of strings of 12+ words {statistics.median(g):.1f}")


if __name__ == "__main__":
    main(sys.argv[1:])
