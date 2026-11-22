"""The atlas: many versions of a method, each one's results on benchmark suites, and the relations among them.

A version file describes one model; everything involving more than one version lives here (docs/atlas.md). An atlas
is a folder:

    atlas.json          the manifest: its name, the versions (file, model and page ids), the suites, the views
    versions/NAME.json  a version as `headway build --json` writes it, without results or other versions
    suites/SUITE.json   a built suite, stored once
    results/NAME/SUITE.json        one version's results on one suite (bench.entry, the suite referred to)
    relations.jsonl     one relation per line: {"id", "kind", "members": [{"v", "role"?, "levels"?}], ...}
    relresults/REL/SUITE.json      a relation's results on one suite: paired differences, contrasts
    media/              images from results, named by content hash

Headway computes none of the numbers and decides nothing for the user: it stores what it is given, reports what it
notices (results computed on an earlier build, provenance that differs between a version and the atlas), and does
the mechanical parts of reorganizing (rename, extract, merge). Every file is plain JSON an agent may edit by hand;
`validate` checks the format afterwards.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import mimetypes
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from headway import bench, diff

FORMAT = 1
MANIFEST = "atlas.json"
KINDS = bench.RELATION_KINDS
# what a version's page id leaves out: what changes without the version changing (the editor its links open, the
# headway release that built it)
VOLATILE = {"name", "built", "built_at", "results", "lanes", "snapshot", "opened", "live", "provenance", "editor",
            "headway"}


class AtlasError(Exception):
    pass


def _plain(s: str) -> str:
    """A name with what cannot be in a file name replaced (the file names before slug() added its hash)."""
    return re.sub(r"[^A-Za-z0-9._@+=-]+", "_", s).strip("._") or "_"


def slug(s: str) -> str:
    """A file name for a version name, suite id or relation id. A name that had to be changed to be one gets a hash of
    itself added, so that two names never share a file ("x/y" and "x_y")."""
    out = _plain(s)
    return out if out == s else f"{out}-{hashlib.sha256(s.encode()).hexdigest()[:8]}"


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _code(o: Mapping[str, Any]) -> list[Any]:
    return [[c.get("ref"), c.get("source", "")] for c in o.get("code", [])]


def model_id(d: Mapping[str, Any]) -> str:
    """What determines the numbers: the steps' structure, conditions and code, the symbols' shapes and values.
    Descriptions, equations as written and documents are left out (a prose edit keeps the model id)."""
    ops = sorted([o["key"], sorted(o["inputs"]), sorted(o["outputs"]), o.get("per"), o.get("condition", ""), _code(o)]
                 for o in d["ops"])
    syms = sorted([s["key"], s.get("dims"), s.get("per"), s.get("role"), s.get("value"), _code(s)]
                  for s in d["symbols"])
    return "sha256:" + _sha([ops, syms])


# what a suite's declarations hold that does not decide whether results on it are comparable: wording, where each
# thing is declared, and code references (the metrics' code is in model_id, the datasets' is added back in
# suite_hash; the figures' is left out: how an image is drawn does not change the numbers)
SUITE_WORDING = {"name", "desc", "tex", "equation", "source", "decl", "code"}


def suite_hash(suite: Mapping[str, Any]) -> str:
    """What makes results on a suite comparable: its model (model_id: the metrics' structure and code), the code
    that makes or loads each dataset, and its declarations (datasets, metrics, objectives, ...) without their
    wording. A reworded description, a line added to the suite's module or another headway release keeps the
    hash."""
    meta = {k: [{f: v for f, v in x.items() if f not in SUITE_WORDING} if isinstance(x, Mapping) else x for x in v]
            if isinstance(v, list) else v for k, v in (suite.get("suite") or {}).items()}
    data_code = [[x.get("key"), _code(x)] for x in (suite.get("suite") or {}).get("datasets", [])
                 if isinstance(x, Mapping)]
    return "sha256:" + _sha([model_id(suite), meta, data_code])


def page_id(d: Mapping[str, Any]) -> str:
    """Everything a version shows, apart from VOLATILE (its name, when it was built, its provenance, results and other
    versions) and the git fields that change without the code changing (dirty, describe)."""
    rest = {k: v for k, v in d.items() if k not in VOLATILE}
    src = copy.deepcopy(rest.get("source") or {})
    if isinstance(src.get("git"), dict):
        src["git"] = {k: v for k, v in src["git"].items() if k not in ("dirty", "describe")}
    rest["source"] = src
    return "sha256:" + _sha(rest)


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text())


def _write_json(p: Path, data: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(p)


def _map_images(figures: Mapping[str, Any], fn: Any) -> dict[str, Any]:
    """Figures ({dataset: {figure: [image, ...]}}) with fn applied to each image (a dict)."""
    return {dk: {fk: [fn(dict(img)) for img in imgs] for fk, imgs in by_fig.items()} for dk, by_fig in figures.items()}


def _result_models(rr: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The models each result in a relation's results file was computed on ({version: model id}): each difference's
    and contrast's own, else (a file written before they were kept) the file's "members"."""
    members = rr.get("members", {})
    return [x.get("models", members) for x in rr.get("differences", [])] + \
        [rr.get("contrast_models", {}).get(k, members) for k in rr.get("contrasts", {})]


def _rename_models(rr: dict[str, Any], fn: Any) -> None:
    """A relation's results file with the versions it names renamed by fn."""
    rr["members"] = {fn(k): v for k, v in rr.get("members", {}).items()}
    for x in rr.get("differences", []):
        x["of"] = [fn(n) for n in x["of"]]
        if "models" in x:
            x["models"] = {fn(k): v for k, v in x["models"].items()}
    for k, models in rr.get("contrast_models", {}).items():
        rr["contrast_models"][k] = {fn(v): m for v, m in models.items()}


def _default_id(r: Mapping[str, Any]) -> str | None:
    """The id a step or a pair is given from its members' names (Atlas.relate, attach, add_version); None for
    other kinds, whose default ids do not name their members."""
    ms = [m["v"] for m in r["members"]]
    if len(ms) != 2 or r["kind"] not in ("step", "pair"):
        return None
    return f"{ms[0]}->{ms[1]}" if r["kind"] == "step" else f"pair:{ms[0]}|{ms[1]}"


def requery(query: str, versions: Mapping[str, str], relations: Mapping[str, str]) -> str:
    """A view's query with versions and relations renamed (old -> new): the names in near:, path:, in: and bare
    name terms, and the ids in relation: terms; a term may start with - (excludes)."""
    def term(t: str) -> str:
        neg = "-" if t.startswith("-") and len(t) > 1 else ""
        op, colon, arg = t[len(neg):].partition(":")
        v = lambda n: versions.get(n, n)                     # noqa: E731
        if not colon:
            return neg + v(op)
        if op == "relation":
            arg = relations.get(arg, arg)
        elif op == "near":
            name, tilde, k = arg.partition("~")
            arg = v(name) + tilde + k
        elif op == "path":
            a, dots, b = arg.partition("..")
            arg = v(a) + dots + (v(b) if dots else "")
        elif op == "in":
            arg = ",".join(v(n) for n in arg.split(","))
        return f"{neg}{op}:{arg}"
    return " ".join(term(t) if t else t for t in query.split(" "))


def is_atlas(path: str | Path) -> bool:
    return (Path(path) / MANIFEST).is_file()


class Atlas:
    """An atlas folder, read into memory; changes are written by save() (and by the methods that say so)."""

    def __init__(self, root: str | Path, meta: dict[str, Any], relations: list[dict[str, Any]]):
        self.root = Path(root)
        self.meta = meta
        self.relations = relations

    # ---------- opening and saving ----------
    @classmethod
    def init(cls, root: str | Path, name: str | None = None) -> Atlas:
        root = Path(root)
        if is_atlas(root):
            raise AtlasError(f"{root} is an atlas already")
        root.mkdir(parents=True, exist_ok=True)
        a = cls(root, {"atlas": FORMAT, "name": name or root.resolve().name, "versions": {}, "suites": {},
                       "views": {}}, [])
        a.save()
        return a

    @classmethod
    def open(cls, root: str | Path) -> Atlas:
        root = Path(root)
        if not is_atlas(root):
            raise AtlasError(f"{root}: not an atlas (no {MANIFEST}); make one with `headway atlas init {root}`")
        meta = _read_json(root / MANIFEST)
        if meta.get("atlas") != FORMAT:
            raise AtlasError(f"{root}: atlas format {meta.get('atlas')!r}; this headway reads format {FORMAT}")
        for k in ("versions", "suites", "views"):
            meta.setdefault(k, {})
        rels = []
        f = root / "relations.jsonl"
        if f.is_file():
            for n, line in enumerate(f.read_text().splitlines(), 1):
                if line.strip():
                    try:
                        rels.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise AtlasError(f"{f}:{n}: not JSON ({e})") from None
        return cls(root, meta, rels)

    def save(self) -> None:
        _write_json(self.root / MANIFEST, self.meta)
        tmp = self.root / "relations.jsonl.tmp"
        tmp.write_text("".join(json.dumps(r) + "\n" for r in self.relations))
        tmp.replace(self.root / "relations.jsonl")

    # ---------- versions ----------
    @property
    def names(self) -> list[str]:
        """The versions, in the order they were built."""
        return sorted(self.meta["versions"], key=lambda n: (self.meta["versions"][n].get("built_at") or "", n))

    def has(self, name: str) -> bool:
        return name in self.meta["versions"]

    def _need(self, name: str) -> dict[str, Any]:
        if name not in self.meta["versions"]:
            raise AtlasError(f"no version {name!r} in the atlas")
        return self.meta["versions"][name]

    def version(self, name: str) -> dict[str, Any]:
        """A version as stored (no results)."""
        return _read_json(self.root / self._need(name)["file"])

    def add_version(self, given: Mapping[str, Any], name: str | None = None, *, parents: Sequence[str] = (),
                    why: str = "", prediction: str = "", keep_old: str | None = None, drop_results: bool = False,
                    ) -> list[str]:
        """Add a built version, or replace the one of that name (a rebuild). Results it carries are attached;
        its parents (made from) are recorded in it and as steps. A rebuild keeps the name's results, relations and
        recorded parents (unless it is given others); results computed on an earlier model are marked as such where
        they are shown. keep_old keeps the old build as its own version under that name, with a step from it;
        drop_results removes the old results. Returns notes on what was done."""
        d: dict[str, Any] = copy.deepcopy(dict(given))
        name = name or d.get("name")
        if not name:
            raise AtlasError("a version in an atlas needs a name (--name)")
        if not {"ops", "symbols", "groups"} <= set(d):
            raise AtlasError(f"{name}: not a built headway")
        notes: list[str] = []
        results = d.pop("results", []) or []
        d.pop("lanes", None)
        d["name"] = name
        prov = d.get("provenance") or {}
        old = self.meta["versions"].get(name)
        if old and keep_old:
            if self.has(keep_old):
                raise AtlasError(f"--keep-old {keep_old}: there is a version of that name already")
            self.rename(name, keep_old)
            notes.append(f"kept the earlier build of {name} as {keep_old}")
            parents = [*parents, keep_old] if keep_old not in parents else parents
            old = None
        elif keep_old:
            notes.append(f"--keep-old {keep_old}: {name} is not in the atlas yet, so there is no earlier build to keep")
        if old and not parents and not prov.get("parents"):
            # a rebuild that names no parents keeps the ones the earlier build recorded (with their why, prediction)
            earlier = (self.version(name).get("provenance") or {}).get("parents")
            if earlier:
                prov["parents"] = earlier
        if parents:
            declared = {p["name"]: p for p in prov.get("parents", [])}
            prov["parents"] = [{**{k: v for k, v in declared.get(p, {}).items() if k not in ("name", "model")},
                                "name": p, **({"model": self.meta["versions"][p]["model"]} if self.has(p) else {}),
                                **({"why": why} if why else {}), **({"prediction": prediction} if prediction else {})}
                               for p in parents]
            for p in parents:
                if not self.has(p):
                    notes.append(f"parent {p} is not in the atlas (yet); the step from it is made when it is added")
        if prov:
            d["provenance"] = prov
        entry = {"file": f"versions/{slug(name)}.json", "model": model_id(d), "page": page_id(d),
                 "built_at": d.get("built_at") or str(d.get("built", ""))}
        if old:
            history = old.get("history", [])
            if old.get("model") != entry["model"] or old.get("page") != entry["page"]:
                history = [*history, {k: old.get(k) for k in ("model", "page", "built_at")}][-20:]
                notes.append(f"replaced {name}" + ("" if old.get("model") != entry["model"] else
                                                   " (the same model: its results stay current)"))
            if history:
                entry["history"] = history
            if drop_results:
                n = self._drop_results(name)
                notes.append(f"removed {n} earlier result{'s' if n != 1 else ''} of {name}")
        else:
            notes.append(f"added {name}")
        _write_json(self.root / entry["file"], d)
        self.meta["versions"][name] = entry
        # steps from the parents in the atlas, and to the versions already in it that declare this one a parent
        # (a parent added after its child), so that the order versions are added in does not matter
        for p in prov.get("parents", []):
            if self.has(p["name"]):
                self._ensure_step(p["name"], name, p)
        for n in self.meta["versions"]:
            if n != name:
                for p in (self.version(n).get("provenance") or {}).get("parents", []):
                    if p["name"] == name:
                        self._ensure_step(name, n, p)
        for r in results:
            notes += self.attach(name, r)
        self.save()
        return notes

    def _ensure_step(self, parent: str, child: str, p: Mapping[str, Any]) -> None:
        for r in self.relations:
            if r["kind"] == "step" and [m["v"] for m in r["members"]] == [parent, child]:
                return
        step = {"id": self._new_id(f"{parent}->{child}"), "kind": "step",
                "members": [{"v": parent, "role": "from"}, {"v": child, "role": "to"}],
                **{k: p[k] for k in ("why", "prediction") if p.get(k)}}
        self.relations.append(step)
        self._absorb_pairs(step)

    def _absorb_pairs(self, step: dict[str, Any]) -> None:
        """A pair of the step's two versions (made for their differences) becomes the step: its fields and results
        move onto it."""
        two = {m["v"] for m in step["members"]}
        for p in [x for x in self.relations if x["kind"] == "pair" and {m["v"] for m in x["members"]} == two]:
            for k, v in p.items():
                step.setdefault(k, v)
            for sid, rr in self.relresults(p["id"]).items():
                rr["relation"] = step["id"]
                _write_json(self.root / "relresults" / slug(step["id"]) / f"{slug(sid)}.json", rr)
            shutil.rmtree(self.root / "relresults" / slug(p["id"]), ignore_errors=True)
            self.relations.remove(p)
            self._requery_views({}, {p["id"]: step["id"]})

    def _requery_views(self, versions: Mapping[str, str], relations: Mapping[str, str]) -> None:
        """The views' queries with versions and relations renamed (requery)."""
        if not versions and not relations:
            return
        for view in self.meta["views"].values():
            if isinstance(view.get("query"), str):
                view["query"] = requery(view["query"], versions, relations)

    def _drop_results(self, name: str) -> int:
        folder = self.root / "results" / slug(name)
        if not folder.is_dir():
            return 0
        n = len(list(folder.glob("*.json")))
        shutil.rmtree(folder)
        return n

    def remove(self, name: str) -> None:
        """A version, its results, its place in relations (a relation left with fewer than two members goes) and in
        the parents other versions record."""
        self._need(name)
        (self.root / self.meta["versions"][name]["file"]).unlink(missing_ok=True)
        self._drop_results(name)
        del self.meta["versions"][name]
        for n, e in self.meta["versions"].items():
            vd = self.version(n)
            prov = vd.get("provenance") or {}
            if any(p["name"] == name for p in prov.get("parents", [])):
                prov["parents"] = [p for p in prov["parents"] if p["name"] != name]
                _write_json(self.root / e["file"], vd)
        kept = []
        for r in self.relations:
            r["members"] = [m for m in r["members"] if m["v"] != name]
            if len(r["members"]) >= 2:
                kept.append(r)
            else:
                shutil.rmtree(self.root / "relresults" / slug(r["id"]), ignore_errors=True)
        self.relations = kept
        self.save()

    # ---------- suites and results ----------
    def suite(self, key: str) -> dict[str, Any]:
        return _read_json(self.root / self.meta["suites"][key]["file"])

    def _store_suite(self, sid: str, suite: Mapping[str, Any], notes: list[str]) -> str:
        h = suite_hash(suite)
        have = self.meta["suites"].get(sid)
        # the stored suite is hashed again, so that a hash recorded by an earlier headway does not count
        if have and "hash" not in have and suite_hash(self.suite(sid)) == h:
            have["hash"] = h                            # an entry written by hand, without its hash
        if have and have.get("hash") != h and suite_hash(self.suite(sid)) != h:
            key = f"{sid}#{h[:8]}"
            if key not in self.meta["suites"]:
                notes.append(f"warning: suite {sid} differs from the one stored under that id; both are kept "
                             f"(this one as {key}): bump the suite's version= when a change makes results incomparable")
            sid = key
        if sid not in self.meta["suites"]:
            f = f"suites/{slug(sid)}.json"
            _write_json(self.root / f, dict(suite))
            self.meta["suites"][sid] = {"file": f, "hash": h}
        return sid

    def _media_out(self, figures: Mapping[str, Any]) -> dict[str, Any]:
        """Figures with their written-in images moved to media/ (by content hash)."""
        def out(img: dict[str, Any]) -> dict[str, Any]:
            m = re.match(r"data:([^;,]+);base64,(.*)$", img.get("src", ""), re.S)
            if not m:
                return img
            data = base64.b64decode(m[2])
            ext = ".svg" if m[1] == "image/svg+xml" else mimetypes.guess_extension(m[1]) or ".bin"
            name = hashlib.sha256(data).hexdigest()[:32] + ext
            f = self.root / "media" / name
            if not f.exists():
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(data)
            img = {**img, "media": name, "type": m[1]}
            del img["src"]
            return img
        return _map_images(figures, out)

    def _media_in(self, figures: Mapping[str, Any]) -> dict[str, Any]:
        """Figures with their images from media/ written in again (data URLs), as a built headway has them."""
        def back(img: dict[str, Any]) -> dict[str, Any]:
            f = self.root / "media" / img["media"] if "media" in img else None
            if f is None or not f.is_file():
                return img
            return {**img, "src": f"data:{img.get('type') or 'application/octet-stream'};base64,"
                                  + base64.b64encode(f.read_bytes()).decode()}
        return _map_images(figures, back)

    def _copy_media(self, out: Atlas, result: Mapping[str, Any]) -> None:
        """The media a result's figures use, copied into another atlas."""
        def copy_one(img: dict[str, Any]) -> dict[str, Any]:
            if "media" in img and (self.root / "media" / img["media"]).is_file():
                (out.root / "media").mkdir(exist_ok=True)
                shutil.copy2(self.root / "media" / img["media"], out.root / "media" / img["media"])
            return img
        _map_images(result.get("figures") or {}, copy_one)

    def attach(self, name: str, entry: Mapping[str, Any]) -> list[str]:
        """A version's results on one suite (as api.read_results makes them: bench.entry, with the built suite).
        Differences from other versions go onto the relation between the two (a step, else a pair, made if
        need be). Replaces earlier results of the version on that suite."""
        v = self._need(name)
        notes: list[str] = []
        e = dict(entry)
        sid = self._store_suite(e.pop("id"), e.pop("suite"), notes)
        differences = e.pop("differences", {}) or {}
        if e.get("figures"):
            e["figures"] = self._media_out(e["figures"])
        stored = {"version": name, "model": v["model"], "suite": sid, **e}
        _write_json(self.root / "results" / slug(name) / f"{slug(sid)}.json", stored)
        notes.append(f"results of {name} on {sid}")
        for other, data in differences.items():
            rel = self.between(name, other)
            if rel is None:                     # made here, even if `other` is not in the atlas yet (it may come)
                rel = {"id": self._new_id(f"pair:{name}|{other}"), "kind": "pair", "members": [{"v": name}, {"v": other}]}
                self.relations.append(rel)
            self._put_relresult(rel["id"], sid, {"differences": [{"of": [name, other], "values": data}]})
            notes.append(f"differences {name} − {other} on the relation {rel['id']}")
        self.save()
        return notes

    def results(self, name: str) -> list[dict[str, Any]]:
        """A version's stored results (each refers to its suite by key)."""
        folder = self.root / "results" / slug(name)
        return [_read_json(f) for f in sorted(folder.glob("*.json"))] if folder.is_dir() else []

    def relresults(self, rid: str) -> dict[str, dict[str, Any]]:
        folder = self.root / "relresults" / slug(rid)
        return {r["suite"]: r for r in (_read_json(f) for f in sorted(folder.glob("*.json")))} if folder.is_dir() else {}

    def _put_relresult(self, rid: str, sid: str, add: Mapping[str, Any]) -> None:
        f = self.root / "relresults" / slug(rid) / f"{slug(sid)}.json"
        r = self.rel(rid)
        cur: dict[str, Any] = _read_json(f) if f.is_file() else {"relation": rid, "suite": sid}
        model = lambda v: self.meta["versions"].get(v, {}).get("model")     # noqa: E731
        # the models each result was computed on, kept with it: a later result in the file does not vouch for an
        # earlier one. "members" (the models when the file was last written) is what atlases before this had.
        cur["members"] = {m["v"]: model(m["v"]) for m in r["members"]}
        for d in add.get("differences", []):
            d = {**d, "models": {v: model(v) for v in d["of"]}}
            cur["differences"] = [x for x in cur.get("differences", []) if x["of"] != d["of"]] + [d]
        if add.get("contrasts"):
            cur["contrasts"] = {**cur.get("contrasts", {}), **add["contrasts"]}
            cur["contrast_models"] = {**cur.get("contrast_models", {}),
                                      **{k: dict(cur["members"]) for k in add["contrasts"]}}
        _write_json(f, cur)

    def attach_relation(self, rid: str, results: Mapping[str, Any], suite_key: str) -> list[str]:
        """Results of a relation on a suite: {"differences": [{"of": [a, b], "values": {dataset: {metric:
        [...]}}}], "contrasts": {key: {dataset: {metric: [...]}}}, "replicates": [ids] (optional)}, the suite
        already in the atlas. The differences are checked against the suite as a version's are (bench.entry), and
        stored in the same form ({values, intervals, ids, pooled}); what does not fit is left out, with a note."""
        self.rel(rid)
        if suite_key not in self.meta["suites"]:
            raise AtlasError(f"suite {suite_key!r} is not in the atlas (attach a version's results on it first)")
        meta = self.suite(suite_key)["suite"]
        datasets = {d["key"]: d for d in meta["datasets"]}
        metrics = {m["key"]: m for m in meta["metrics"]}
        warnings: list[str] = []
        reps = bench._replicates(results.get("replicates"), datasets, warnings)
        checked: list[dict[str, Any]] = []
        for x in results.get("differences", []):
            if not (isinstance(x, Mapping) and isinstance(x.get("of"), list) and len(x["of"]) == 2
                    and isinstance(x.get("values"), Mapping)):
                warnings.append("a difference needs \"of\": [a, b] and \"values\"; left out")
                continue
            a, b = map(str, x["of"])
            values = bench._differences({b: x["values"]}, datasets, metrics, reps, warnings).get(b)
            if values:
                checked.append({"of": [a, b], "values": values})
        self._put_relresult(rid, suite_key, {**results, "differences": checked})
        return [f"results of the relation {rid} on {suite_key}"] + [f"warning: {w}" for w in warnings]

    def entry(self, name: str) -> dict[str, Any]:
        """A version as the viewer and `headway diff`/`results` read one: with its results, each carrying its
        suite and the differences from other versions the atlas holds."""
        d = self.version(name)
        model = self.meta["versions"][name]["model"]
        diffs: dict[str, dict[str, Any]] = {}
        for r in self.relations:
            if not any(m["v"] == name for m in r["members"]):
                continue
            for sid, rr in self.relresults(r["id"]).items():
                for x in rr.get("differences", []):
                    if x["of"][0] == name:
                        diffs.setdefault(sid, {})[x["of"][1]] = x["values"]
        out = []
        for res in self.results(name):
            sid = res["suite"]
            e = {k: v for k, v in res.items() if k not in ("version", "model", "suite")}
            e["id"] = sid
            e["suite"] = self.suite(sid)
            if e.get("figures"):
                e["figures"] = self._media_in(e["figures"])
            if diffs.get(sid):
                e["differences"] = diffs[sid]
            if res.get("model") and res["model"] != model:
                e["earlier_build"] = True
            out.append(e)
        if out:
            d["results"] = out
        return d

    # ---------- relations ----------
    def rel(self, rid: str) -> dict[str, Any]:
        for r in self.relations:
            if r["id"] == rid:
                return r
        raise AtlasError(f"no relation {rid!r}")

    def between(self, a: str, b: str) -> dict[str, Any] | None:
        """The step between two versions (either way), else a pair of them."""
        found = [r for r in self.relations if r["kind"] in ("step", "pair") and {m["v"] for m in r["members"]} == {a, b}]
        return next((r for r in found if r["kind"] == "step"), found[0] if found else None)

    def _new_id(self, base: str) -> str:
        ids = {r["id"] for r in self.relations}
        if base not in ids:
            return base
        n = 2
        while f"{base}.{n}" in ids:
            n += 1
        return f"{base}.{n}"

    def relate(self, kind: str, members: Sequence[str | Mapping[str, Any]], *, rid: str | None = None,
               base: str | None = None, factors: Sequence[str] = (), **fields: Any) -> dict[str, Any]:
        """A relation among versions. step: [from, to]; pair: two; alternatives: any, with an optional base;
        factorial: the cells, levels read from each version's keys named in factors; group: any."""
        if kind not in KINDS:
            raise AtlasError(f"kind {kind!r}: one of {', '.join(KINDS)}")
        ms: list[dict[str, Any]] = [dict(m) if isinstance(m, Mapping) else {"v": m} for m in members]
        if base and not any(m["v"] == base for m in ms):
            ms.insert(0, {"v": base})
        for m in ms:
            if base and m["v"] == base:
                m["role"] = "base"
        if kind == "step":
            if len(ms) != 2:
                raise AtlasError("a step has two members: from and to")
            ms[0]["role"], ms[1]["role"] = "from", "to"
        if kind == "pair" and len(ms) != 2:
            raise AtlasError("a pair has two members")
        if len(ms) < 2:
            raise AtlasError(f"a {kind} needs two members or more")
        for m in ms:
            if not self.has(m["v"]):
                raise AtlasError(f"no version {m['v']!r} in the atlas")
        if kind == "factorial":
            if not factors:
                raise AtlasError("a factorial needs its factors (keys of its versions)")
            for m in ms:
                if "levels" not in m:
                    keys = self.version(m["v"]).get("keys", {})
                    missing = [f for f in factors if f not in keys]
                    if missing:
                        raise AtlasError(f"{m['v']} has no key {', '.join(missing)}: give its levels")
                    m["levels"] = {f: keys[f] for f in factors}
            fields["factors"] = list(factors)
        default = f"{ms[0]['v']}->{ms[1]['v']}" if kind == "step" else f"{kind}:{ms[0]['v']}|{ms[1]['v']}" \
            if kind == "pair" else f"{kind}-{sum(r['kind'] == kind for r in self.relations) + 1}"
        if rid and any(r["id"] == rid for r in self.relations):
            raise AtlasError(f"there is a relation {rid!r} already")
        r = {"id": rid or self._new_id(default), "kind": kind, "members": ms,
             **{k: v for k, v in fields.items() if v not in (None, "", [], ())}}
        self.relations.append(r)
        if kind == "step":
            self._absorb_pairs(r)
        self.save()
        return r

    def update_relation(self, rid: str, **fields: Any) -> dict[str, Any]:
        r = self.rel(rid)
        for k, v in fields.items():
            if v is None:
                continue
            if k == "tags":
                r["tags"] = sorted(set(r.get("tags", [])) | set(v))
            else:
                r[k] = v
        self.save()
        return r

    # ---------- views ----------
    def set_view(self, name: str, view: Mapping[str, Any]) -> None:
        self.meta["views"][name] = dict(view)
        self.save()

    # ---------- reorganizing ----------
    def rename(self, old: str, new: str) -> None:
        """A version renamed, with every reference to it."""
        entry = self._need(old)
        if self.has(new):
            raise AtlasError(f"there is a version {new!r} already")
        d = self.version(old)
        d["name"] = new
        (self.root / entry["file"]).unlink()
        entry["file"] = f"versions/{slug(new)}.json"
        _write_json(self.root / entry["file"], d)
        self.meta["versions"] = {new if k == old else k: v for k, v in self.meta["versions"].items()}
        src, dst = self.root / "results" / slug(old), self.root / "results" / slug(new)
        if src.is_dir():
            src.rename(dst)
            for f in dst.glob("*.json"):
                r = _read_json(f)
                r["version"] = new
                _write_json(f, r)
        renamed_ids: dict[str, str] = {}
        for r in self.relations:
            if not any(m["v"] == old for m in r["members"]):
                continue
            was = _default_id(r)
            for m in r["members"]:
                if m["v"] == old:
                    m["v"] = new
            rid = r["id"]
            # an id made from the members' names ("v3->v4", "pair:v3|v4", with any ".2") is made again from the new
            # ones, so that it does not name a version the relation no longer has
            if was and (rid == was or re.fullmatch(re.escape(was) + r"\.\d+", rid)):
                r["id"] = self._new_id(_default_id(r) or rid)
                renamed_ids[rid] = r["id"]
            for sid, rr in self.relresults(rid).items():
                _rename_models(rr, lambda n: new if n == old else n)
                rr["relation"] = r["id"]
                _write_json(self.root / "relresults" / slug(r["id"]) / f"{slug(sid)}.json", rr)
            if r["id"] != rid:
                shutil.rmtree(self.root / "relresults" / slug(rid), ignore_errors=True)

        self._requery_views({old: new}, renamed_ids)
        for n in self.meta["versions"]:
            if n == new:
                continue
            vd = self.version(n)
            changed = False
            for p in (vd.get("provenance") or {}).get("parents", []):
                if p["name"] == old:
                    p["name"], changed = new, True
            if changed:
                _write_json(self.root / self.meta["versions"][n]["file"], vd)
        for view in self.meta["views"].values():
            for k in ("path", "hidden"):
                if isinstance(view.get(k), list):
                    view[k] = [new if x == old else x for x in view[k]]
            if view.get("baseline") == old:
                view["baseline"] = new
        self.save()

    def scan(self) -> list[str]:
        """The manifest made again from versions/ (ids recomputed, files found), and a step made for each parent a
        version declares that has none and is in the atlas."""
        notes = []
        seen: dict[str, dict[str, Any]] = {}
        for f in sorted((self.root / "versions").glob("*.json")):
            try:
                d = _read_json(f)
            except json.JSONDecodeError as e:
                notes.append(f"{f.name}: not JSON ({e}); skipped")
                continue
            name = d.get("name") or f.stem
            if name in seen:
                # two files of one name (a copy not yet renamed): the one at the name's own file name is kept
                other = seen[name]["file"]
                if f"versions/{f.name}" != f"versions/{slug(name)}.json":
                    notes.append(f"{f.name}: also named {name} (as {Path(other).name} is); left out: give it a "
                                 "name of its own")
                    continue
                notes.append(f"{Path(other).name}: also named {name} (as {f.name} is); left out: give it a name of "
                             "its own")
            old = self.meta["versions"].get(name, {})
            seen[name] = {**old, "file": f"versions/{f.name}", "model": model_id(d), "page": page_id(d),
                          "built_at": d.get("built_at") or str(d.get("built", ""))}
            if not old:
                notes.append(f"found {name}")
            elif old.get("model") != seen[name]["model"]:
                notes.append(f"{name}: its model changed (edited or rebuilt)")
        for n in set(self.meta["versions"]) - set(seen):
            notes.append(f"{n}: its file is gone; left out of the manifest")
        self.meta["versions"] = seen
        moved = 0                       # folders named before slug() added a hash to names it changes
        for top, keys in (("results", list(seen)), ("relresults", [r["id"] for r in self.relations])):
            for k in keys:
                was, now = self.root / top / _plain(k), self.root / top / slug(k)
                if was != now and was.is_dir() and not now.exists():
                    was.rename(now)
                    moved += 1
        if moved:
            notes.append(f"moved {moved} folders to their current names")
        before = len(self.relations)
        for n in seen:
            for p in (self.version(n).get("provenance") or {}).get("parents", []):
                if p["name"] in seen:
                    self._ensure_step(p["name"], n, p)
        if len(self.relations) > before:
            notes.append(f"made {len(self.relations) - before} steps from the versions' parents")
        self.save()
        return notes

    def check(self) -> list[str]:
        """What is worth knowing: results and relation results computed on an earlier build of a version,
        provenance that differs between a version and the atlas, references to versions not in it. Nothing is
        changed."""
        out = []
        vs = self.meta["versions"]
        for n, e in vs.items():
            for r in self.results(n):
                if r.get("model") and r["model"] != e["model"]:
                    out.append(f"{n}: its results on {r['suite']} were computed on an earlier build")
                if r.get("suite") not in self.meta["suites"]:
                    out.append(f"{n}: results on {r.get('suite')}, a suite not in the atlas")
        for r in self.relations:
            for m in r["members"]:
                if m["v"] not in vs:
                    out.append(f"relation {r['id']}: member {m['v']} is not in the atlas")
            for sid, rr in self.relresults(r["id"]).items():
                stale = {v for models in _result_models(rr) for v, model in models.items()
                         if v in vs and model and model != vs[v]["model"]}
                for v in sorted(stale):
                    out.append(f"relation {r['id']}: its results on {sid} were computed on an earlier build of {v}")
        for n in vs:
            parents = {p["name"]: p for p in (self.version(n).get("provenance") or {}).get("parents", [])}
            steps = [r for r in self.relations if r["kind"] == "step" and r["members"][1]["v"] == n]
            for s in steps:
                frm = s["members"][0]["v"]
                if parents and frm not in parents:
                    out.append(f"step {s['id']}: {n} declares other parents ({', '.join(parents)})")
                elif frm in parents:
                    for k in ("why", "prediction"):
                        if parents[frm].get(k) and s.get(k) and parents[frm][k] != s[k]:
                            out.append(f"step {s['id']}: its {k} differs from the one {n} was built with")
        for vn, view in self.meta["views"].items():
            for x in [*view.get("path", []), *([view["baseline"]] if view.get("baseline") else [])]:
                if x not in vs:
                    out.append(f"view {vn}: {x} is not in the atlas")
        return out

    def validate(self) -> list[str]:
        """Errors in the format (for files edited by hand)."""
        errs = []
        m = self.meta
        for k in ("name", "versions", "suites", "views"):
            if k not in m:
                errs.append(f"{MANIFEST}: no {k!r}")
        for n, e in m.get("versions", {}).items():
            f = self.root / e.get("file", "")
            if not f.is_file():
                errs.append(f"version {n}: no file {e.get('file')}")
                continue
            try:
                d = _read_json(f)
            except json.JSONDecodeError as ex:
                errs.append(f"version {n}: {e['file']} is not JSON ({ex})")
                continue
            if not {"ops", "symbols", "groups"} <= set(d):
                errs.append(f"version {n}: {e['file']} is not a built headway")
            for k in ("model", "page"):
                if k not in e:
                    errs.append(f"version {n}: no {k} id (headway atlas scan makes it)")
        for sid, e in m.get("suites", {}).items():
            if not (self.root / e.get("file", "")).is_file():
                errs.append(f"suite {sid}: no file {e.get('file')}")
            if "hash" not in e:
                errs.append(f"suite {sid}: no hash (suite_hash of its file)")
        ids = set()
        for r in self.relations:
            rid = r.get("id")
            if not rid:
                errs.append(f"a relation without an id: {json.dumps(r)[:80]}")
                continue
            if rid in ids:
                errs.append(f"relation {rid}: the id is used twice")
            ids.add(rid)
            if r.get("kind") not in KINDS:
                errs.append(f"relation {rid}: kind {r.get('kind')!r} is not one of {', '.join(KINDS)}")
            ms = r.get("members")
            if not isinstance(ms, list) or len(ms) < 2 or not all(isinstance(x, dict) and "v" in x for x in ms):
                errs.append(f"relation {rid}: members must be a list of two or more {{\"v\": name}}")
            elif r.get("kind") == "step" and len(ms) != 2:
                errs.append(f"relation {rid}: a step has two members")
        for f in sorted((self.root / "relresults").glob("*/*.json")) if (self.root / "relresults").is_dir() else []:
            try:
                rr = _read_json(f)
            except json.JSONDecodeError as ex:
                errs.append(f"{f.relative_to(self.root)}: not JSON ({ex})")
                continue
            if rr.get("relation") not in ids:
                errs.append(f"{f.relative_to(self.root)}: relation {rr.get('relation')!r} is not in relations.jsonl")
            for x in rr.get("differences", []):
                if not (isinstance(x.get("of"), list) and len(x["of"]) == 2 and "values" in x):
                    errs.append(f"{f.relative_to(self.root)}: a difference needs \"of\": [a, b] and \"values\"")
        for n in m.get("versions", {}):
            for r in self.results(n):
                if r.get("suite") not in m.get("suites", {}):
                    errs.append(f"results of {n}: suite {r.get('suite')!r} is not in the manifest")
        for vn, view in m.get("views", {}).items():
            if view.get("kind") not in ("stack", "graph"):
                errs.append(f"view {vn}: kind must be stack or graph")
        return errs

    def extract(self, dst: str | Path, names: Iterable[str]) -> Atlas:
        """A new atlas with these versions, their results, the suites and media those use, the relations among
        them (a relation keeps the members included, if two or more), and the views restricted to them."""
        keep = [n for n in names if self.has(n)]
        out = Atlas.init(dst, self.meta["name"])
        self._copy_into(out, keep, {n: n for n in keep})
        out.meta["views"] = {}
        for vn, view in self.meta["views"].items():
            v = copy.deepcopy(view)
            if "path" in v:
                v["path"] = [x for x in v["path"] if x in keep]
            if v.get("baseline") not in keep:
                v.pop("baseline", None)
            out.meta["views"][vn] = v
        out.save()
        return out

    def _copy_into(self, out: Atlas, names: Sequence[str], rename: Mapping[str, str],
                   related: Sequence[str] | None = None, notes: list[str] | None = None) -> dict[str, str]:
        """Versions (renamed) copied into another atlas with their results and media, and the relations among
        `related` (default: the same versions), each keeping the members it has among them. A suite stored in
        `out` under another key (one of the same id but other content is there) is referred to by that key; the
        warning goes into `notes`. Returns the relations' ids in `out` (by their ids here) that differ."""
        notes = [] if notes is None else notes
        suites: dict[str, str] = {}                     # this atlas's suite key -> out's

        def suite_in_out(sid: str) -> str:
            if sid not in suites:
                suites[sid] = out._store_suite(sid, self.suite(sid), notes)
            return suites[sid]
        for n in names:
            e = self.meta["versions"][n]
            d = self.version(n)
            nn = rename[n]
            d["name"] = nn
            for p in (d.get("provenance") or {}).get("parents", []):
                p["name"] = rename.get(p["name"], p["name"])
            ne = {**e, "file": f"versions/{slug(nn)}.json"}
            _write_json(out.root / ne["file"], d)
            out.meta["versions"][nn] = ne
            for r in self.results(n):
                sid = suite_in_out(r["suite"])
                self._copy_media(out, r)
                _write_json(out.root / "results" / slug(nn) / f"{slug(sid)}.json", {**r, "version": nn, "suite": sid})
        inside = set(names if related is None else related)
        ids: dict[str, str] = {}                        # this atlas's relation ids -> out's
        for r in self.relations:
            ms = [m for m in r["members"] if m["v"] in inside]
            if len(ms) < 2 or (r["kind"] in ("step", "pair") and len(ms) != 2):
                continue
            nr = copy.deepcopy({**r, "members": ms})
            was = _default_id(nr)
            for m in nr["members"]:
                m["v"] = rename[m["v"]]
            # an id made from the members' names is made again from their new names (as rename does)
            now = _default_id(nr)
            if was and now and was != now and (r["id"] == was or re.fullmatch(re.escape(was) + r"\.\d+", r["id"])):
                nr["id"] = now
            if any(x["id"] == nr["id"] for x in out.relations):
                if any(x["id"] == nr["id"] and x["kind"] == nr["kind"] and x["members"] == nr["members"]
                       for x in out.relations):
                    ids[r["id"]] = nr["id"]
                    continue
                nr["id"] = out._new_id(nr["id"])
            ids[r["id"]] = nr["id"]
            out.relations.append(nr)
            for sid, rr in self.relresults(r["id"]).items():
                if sid not in self.meta["suites"] or (sid not in suites and sid not in out.meta["suites"]):
                    continue                    # results on a suite none of the versions copied has results on
                osid = suite_in_out(sid)
                rr = copy.deepcopy(rr)
                rr["relation"], rr["suite"] = nr["id"], osid
                rr["differences"] = [x for x in rr.get("differences", []) if set(x["of"]) <= inside]
                rr["members"] = {k: v for k, v in rr.get("members", {}).items() if k in inside}
                for k, models in rr.get("contrast_models", {}).items():
                    rr["contrast_models"][k] = {v: m for v, m in models.items() if v in inside}
                _rename_models(rr, lambda v: rename.get(v, v))
                _write_json(out.root / "relresults" / slug(nr["id"]) / f"{slug(osid)}.json", rr)
        out.save()
        return {k: v for k, v in ids.items() if k != v}

    @staticmethod
    def merge(a: Atlas, b: Atlas, dst: str | Path, prefix: str | None = None) -> tuple[Atlas, list[str]]:
        """A new atlas with both. Versions of one name and one model are one (b's results added where a has
        none on that suite); one name, different models: b's is renamed with prefix, or, without one, nothing
        is written and the conflicts are returned. Relations of one id and the same members are one."""
        conflicts = [n for n in b.meta["versions"] if a.has(n) and a.meta["versions"][n]["model"] !=
                     b.meta["versions"][n]["model"]]
        if conflicts and not prefix:
            raise AtlasError("the same name, different models: " + ", ".join(conflicts) +
                             "; give a prefix to rename b's, or rename them first")
        rename = {n: (prefix + n if n in conflicts else n) for n in b.meta["versions"]}
        new = [n for n in b.meta["versions"] if not (a.has(n) and n not in conflicts)]
        taken = [rename[n] for n in conflicts if a.has(rename[n]) or rename[n] in b.meta["versions"]]
        if taken:
            raise AtlasError(f"the prefix {prefix!r} gives names already taken: {', '.join(taken)}; give another")
        out = a.extract(dst, a.names)
        out.meta["name"] = f"{a.meta['name']} + {b.meta['name']}"
        notes = [f"renamed {n} to {rename[n]}" for n in conflicts]
        same = [n for n in b.meta["versions"] if n not in new]
        ids = b._copy_into(out, new, rename, related=list(b.meta["versions"]), notes=notes)
        for n in same:
            have = {r["suite"] for r in out.results(n)}
            for r in b.results(n):
                if r["suite"] not in have:
                    sid = out._store_suite(r["suite"], b.suite(r["suite"]), notes)
                    b._copy_media(out, r)
                    _write_json(out.root / "results" / slug(n) / f"{slug(sid)}.json", {**r, "suite": sid})
                    notes.append(f"{n}: added b's results on {r['suite']}")
        for vn, view in b.meta["views"].items():
            v = copy.deepcopy(view)
            for k in ("path", "hidden"):
                if isinstance(v.get(k), list):
                    v[k] = [rename.get(x, x) for x in v[k]]
            if v.get("baseline"):
                v["baseline"] = rename.get(v["baseline"], v["baseline"])
            if isinstance(v.get("query"), str):
                v["query"] = requery(v["query"], {n: m for n, m in rename.items() if n != m}, ids)
            out.meta["views"][vn if vn not in out.meta["views"] else f"{vn} (b)"] = v
        out.save()
        return out, notes

    def import_files(self, paths: Iterable[str | Path]) -> list[str]:
        """Built versions (JSON or pages, with a page's other versions) added: results attached, differences moved
        to relations; a page with other versions also becomes a stack view."""
        notes = []
        from headway.api import read
        for path in map(Path, paths):
            try:
                d, _, _ = read(path)
            except ValueError as e:
                raise AtlasError(str(e)) from None
            if "atlas" in d:
                raise AtlasError(f"{path}: an atlas page; extract or merge atlases instead")
            lanes = d.pop("lanes", []) or []
            items = [(ln.get("name") or ln["data"].get("name"), ln["data"]) for ln in lanes] + \
                    [(d.get("name") or path.stem, d)]
            names = []
            for name, data in items:
                if self.has(name) and self.meta["versions"][name]["page"] == page_id(data):
                    names.append(name)
                    continue
                notes += self.add_version(data, name)
                names.append(name)
            if len(items) > 1:
                order = sorted(names, key=lambda n: self.meta["versions"][n].get("built_at") or "")
                self.meta["views"][path.stem] = {"kind": "stack", "path": order}
                notes.append(f"view {path.stem}: {' › '.join(order)}")
        self.save()
        return notes

    # ---------- what the viewer reads ----------
    def summary(self) -> dict[str, Any]:
        """Everything the atlas explorer shows without opening a version in full: per version its keys, provenance,
        a fingerprint of its steps (lineage -> hash of what is not wording) and its results' values; the suites'
        declarations; the objectives' scores; the relations, with structural diffs of steps and pairs."""
        names = self.names
        ds = [self.version(n) for n in names]
        lin = diff.lineages(ds) if ds else {}
        order: list[str] = []
        labels: dict[str, dict[str, str]] = {}
        versions: dict[str, Any] = {}
        # the lineages in the newest version's order (its labels and groups), then those only older versions have
        for i in reversed(range(len(ds))):
            for o in ds[i]["ops"]:
                ln = lin[(i, "op", o["key"])]
                if ln not in labels:
                    order.append(ln)
                    labels[ln] = {"label": o.get("label") or o["key"], "group": o.get("group") or ""}
        groups = sorted({labels[x]["group"] for x in order})
        order.sort(key=lambda x: groups.index(labels[x]["group"]))
        for i, (n, d) in enumerate(zip(names, ds)):
            syms = lambda keys: sorted(lin.get((i, "symbol", k), "?" + k) for k in keys)  # noqa: E731
            # a lineage can hold several ops of one version (a step split in two): its fingerprint covers them all
            parts: dict[str, list[str]] = {}
            for o in d["ops"]:
                parts.setdefault(lin[(i, "op", o["key"])], []).append(_sha(
                    [syms(o["inputs"]), syms(o["outputs"]), o.get("per"), o.get("condition", ""),
                     " ".join((o.get("tex") or "").split()), o.get("verified"), _code(o)]))
            fp = {ln: (ps[0] if len(ps) == 1 else _sha(sorted(ps)))[:12] for ln, ps in parts.items()}
            values: dict[str, list[Any]] = {}
            for s in d["symbols"]:
                if s.get("value") is not None:
                    values.setdefault(lin[(i, "symbol", s["key"])], []).append(s["value"])
            settings = {ln: vs[0] if len(vs) == 1 else sorted(vs, key=str) for ln, vs in values.items()}
            e = self.meta["versions"][n]
            git = (d.get("source") or {}).get("git") or {}
            versions[n] = {"model": e["model"], "page": e["page"], "built_at": e.get("built_at", ""),
                           "title": d.get("title", ""), "keys": d.get("keys", {}),
                           "source": {k: git.get(k) for k in ("commit", "branch", "describe", "dirty") if k in git},
                           "parents": [p["name"] for p in (d.get("provenance") or {}).get("parents", [])],
                           "history": e.get("history", []), "ops": fp, "settings": settings,
                           "desc": (d.get("desc") or "")[:400]}
        suites: dict[str, Any] = {}
        results: dict[str, dict[str, Any]] = {}
        scores: dict[str, dict[str, Any]] = {}
        for sid in self.meta["suites"]:
            s = self.suite(sid)
            meta = s["suite"]
            suites[sid] = {"title": s.get("title", sid), "version": meta.get("version"),
                           "datasets": [{k: x.get(k) for k in ("key", "name", "heldout")} for x in meta["datasets"]],
                           "metrics": [{**{k: x.get(k) for k in ("key", "better", "range", "interval")},
                                        "name": next((y.get("name") for y in s["symbols"] if y["key"] == x["key"]),
                                                     x["key"]) or x["key"]}
                                       for x in meta["metrics"]],
                           "objectives": [{k: x.get(k) for k in ("key", "name", "how", "weights", "datasets",
                                                                 "categories")} for x in meta["objectives"]],
                           "categories": meta.get("categories", []),
                           "contrasts": meta.get("contrasts", [])}
            have = {}
            for n in names:
                r = next((r for r in self.results(n) if r["suite"] == sid), None)
                if r:
                    have[n] = r
                    results.setdefault(sid, {})[n] = {"values": r["values"], "replicates": r.get("replicates", {}),
                                                      "intervals": r.get("intervals", {}),
                                                      "earlier": bool(r.get("model")) and
                                                      r["model"] != self.meta["versions"][n]["model"]}
            for o in meta["objectives"]:
                vals = bench.scores(list(have.values()), o, s)
                scores.setdefault(sid, {})[o["key"]] = dict(zip(have, vals))
        relres: dict[str, Any] = {}
        diffs: dict[str, Any] = {}
        index = {n: i for i, n in enumerate(names)}
        for r in self.relations:
            rr = self.relresults(r["id"])
            if rr:
                relres[r["id"]] = rr
            if r["kind"] in ("step", "pair") and all(m["v"] in index for m in r["members"]):
                a, b = (index[m["v"]] for m in r["members"])
                changes = [c for c in diff.compare(ds, a, b, lin) if not c.quiet]
                diffs[r["id"]] = {
                    "added": sum(c.status == "added" for c in changes),
                    "changed": sum(c.status in ("changed", "split", "merged") for c in changes),
                    "removed": sum(c.status == "removed" for c in changes),
                    "items": [[c.status, c.kind, c.key, diff._label(ds[b if c.status != "removed" else a], c.kind, c.key),
                               [w for w in c.what if w not in diff.QUIET]] for c in changes][:60]}
        return {"name": self.meta["name"], "format": FORMAT, "order": names, "versions": versions,
                "lineages": [{"key": x, **labels[x]} for x in order], "suites": suites, "results": results,
                "scores": scores, "relations": list(self.relations), "relresults": relres, "diffs": diffs,
                "views": self.meta["views"]}

    def page_data(self, view: str | None = None, full: bool | None = None) -> dict[str, Any]:
        """The data of an atlas page: the summary, and the versions in full that a static page carries (all of
        them for a small atlas or with full=True; else those of the view, or the latest)."""
        if view is not None and view not in self.meta["views"]:
            raise AtlasError(f"no view {view!r} (views: {', '.join(self.meta['views']) or 'none'})")
        s = self.summary()
        names = self.names
        if full is None:
            full = len(names) <= 40
        if full:
            carry = names
        else:
            v = self.meta["views"].get(view or "", {})
            carry = [n for n in v.get("path", []) if self.has(n)] or names[-1:]
        return {"title": self.meta["name"], "atlas": s, "atlas_view": view,
                "atlas_versions": {n: self.entry(n) for n in carry}}

    # ---------- choosing versions (the explorer's query terms that make sense on the command line) ----------
    def _adjacent(self) -> dict[str, set[str]]:
        adj: dict[str, set[str]] = {n: set() for n in self.meta["versions"]}
        for r in self.relations:
            ms = [m["v"] for m in r["members"] if m["v"] in adj]
            if r["kind"] in ("step", "pair"):
                if len(ms) == 2:
                    adj[ms[0]].add(ms[1])
                    adj[ms[1]].add(ms[0])
            else:
                base = next((m["v"] for m in r["members"] if m.get("role") == "base"), None)
                for x in ms:
                    if base and x != base:
                        adj[x].add(base)
                        adj[base].add(x)
        return adj

    def path(self, a: str, b: str) -> list[str]:
        adj, todo = self._adjacent(), [a]
        prev: dict[str, str | None] = {a: None}
        while todo:
            x = todo.pop(0)
            if x == b:
                break
            for y in sorted(adj.get(x, ())):
                if y not in prev:
                    prev[y] = x
                    todo.append(y)
        if b not in prev:
            return []
        out: list[str] = []
        cur: str | None = b
        while cur is not None:
            out.insert(0, cur)
            cur = prev[cur]
        return out

    def select(self, terms: Sequence[str]) -> list[str]:
        """Versions matching every term: near:NAME~N, path:A..B, key:K=V (or key:K), relation:ID, kind:K, tag:T; a
        term starting with - excludes; the names given are one term (any of them). In the order they were built."""
        alls = self.names
        chosen = set(alls)
        named = {t for t in terms if t in self.meta["versions"]}          # names given are one set: any of them
        if named:
            chosen &= named
        adj = self._adjacent()
        for t in (t for t in terms if t not in named):
            neg = t.startswith("-") and len(t) > 1
            t = t[1:] if neg else t
            op, _, arg = t.partition(":")
            if op == "near":
                v, _, k = arg.partition("~")
                if not (k or "1").isdigit():
                    raise AtlasError(f"{t}: near:NAME~N, N a number of steps")
                dist, todo = {v: 0}, [v] if v in adj else []
                while todo:
                    x = todo.pop(0)
                    if dist[x] < int(k or 1):
                        for y in adj[x]:
                            if y not in dist:
                                dist[y] = dist[x] + 1
                                todo.append(y)
                got = set(dist) & set(alls)
            elif op == "path":
                a, _, b = arg.partition("..")
                got = set(self.path(a, b))
            elif op == "key":
                k, eq, v = arg.partition("=")
                got = set()
                for n in alls:
                    keys = self.version(n).get("keys", {})
                    if k in keys and (not eq or str(keys[k]).lower() == v.lower()):
                        got.add(n)
            elif op == "relation":
                got = {m["v"] for r in self.relations if r["id"] == arg for m in r["members"]}
            elif op == "kind":
                got = {m["v"] for r in self.relations if r["kind"] == arg for m in r["members"]}
            elif op == "tag":
                got = {m["v"] for r in self.relations if arg in r.get("tags", []) for m in r["members"]}
            else:
                got = {t} & set(alls)
                if not got:
                    raise AtlasError(f"{t}: not a version or a term (near:, path:, key:, relation:, kind:, tag:)")
            chosen = chosen - got if neg else chosen & got
        return [n for n in alls if n in chosen]
