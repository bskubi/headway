"""A trace: what a run of some code did, at the boundaries of the functions traced, written as files that any
language's tracer can write and headway's checks read (survey/tracing-design.md §1).

    <dir>/manifest.json     the run: its environment, the functions traced, and their calls
    <dir>/arrays.npz        the arrays the calls took and gave, by key ("c12.counts", "c12.return")

A call records its function, its arguments and result (each a Value: an array's dtype and shape and its key in
arrays.npz; a number, string or None itself; another object's type and its plain attributes), the call it was made
from, and its path: the branches it took (each as a line and the line it went to) with how many times. Calls past a
function's limit keep only their path and sizes, so that every call is counted and the ones recorded are a sample.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = "headway-trace/1"


@dataclass
class Value:
    kind: str                           # "array", "scalar", "str", "none", "seq", "map", "object", "table"
    type: str = ""                      # the value's type (module.qualname)
    dtype: str = ""
    shape: list[int] | None = None
    key: str | None = None              # an array's key in arrays.npz (None: not kept, only described)
    value: Any = None                   # a scalar's, a string's value; an object's plain attributes
    items: list[Value] | None = None    # a tuple's or list's items (short ones)
    names: dict[str, Value] | None = None   # a dict's items, a table's columns

    @staticmethod
    def of(d: dict[str, Any]) -> Value:
        items = [Value.of(i) for i in d["items"]] if d.get("items") is not None else None
        names = {k: Value.of(v) for k, v in d["names"].items()} if d.get("names") is not None else None
        return Value(d["kind"], d.get("type", ""), d.get("dtype", ""), d.get("shape"), d.get("key"),
                     d.get("value"), items, names)


@dataclass
class Call:
    id: int
    function: str                       # module:qualname
    seq: int                            # the n-th call of its function (from 0)
    parent: int | None = None
    args: dict[str, Value] = field(default_factory=dict)
    result: Value | None = None
    raised: str | None = None
    kept: bool = True                   # its values recorded (False: past its function's limit; sizes only)
    branches: dict[str, int] = field(default_factory=dict)  # "line>line": times taken
    seconds: float = 0.0
    globals: dict[str, Value] = field(default_factory=dict)  # the module's constants it read, as they were

    @staticmethod
    def of(d: dict[str, Any]) -> Call:
        return Call(d["id"], d["function"], d["seq"], d.get("parent"),
                    {k: Value.of(v) for k, v in d.get("args", {}).items()},
                    Value.of(d["result"]) if d.get("result") else None, d.get("raised"), d.get("kept", True),
                    d.get("branches", {}), d.get("seconds", 0.0),
                    {k: Value.of(v) for k, v in d.get("globals", {}).items()})


@dataclass
class Function:
    name: str                           # module:qualname
    file: str
    line: int
    calls: int = 0                      # every call, recorded or not
    # calls past the cap on stored calls: only counted, by their path ("line>line|line>line": calls) and by
    # the loops' total iterations (branch: times taken)
    more_paths: dict[str, int] = field(default_factory=dict)
    more_branches: dict[str, int] = field(default_factory=dict)


@dataclass
class Trace:
    environment: dict[str, Any]
    run: dict[str, Any]
    functions: dict[str, Function]
    calls: list[Call]
    directory: Path | None = None
    _arrays: Any = None

    def write(self, directory: str | Path, arrays: dict[str, np.ndarray]) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        manifest = {"format": FORMAT, "environment": self.environment, "run": self.run,
                    "functions": {k: asdict(f) for k, f in self.functions.items()},
                    "calls": [_plain(asdict(c)) for c in self.calls]}
        (d / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
        np.savez_compressed(d / "arrays.npz", **arrays)  # type: ignore[arg-type]
        self.directory = d
        return d

    @staticmethod
    def read(directory: str | Path) -> Trace:
        d = Path(directory)
        m = json.loads((d / "manifest.json").read_text())
        if m.get("format") != FORMAT:
            raise ValueError(f"{d}: not a {FORMAT} trace")
        return Trace(m["environment"], m["run"], {k: Function(**f) for k, f in m["functions"].items()},
                     [Call.of(c) for c in m["calls"]], d)

    def array(self, key: str) -> np.ndarray:
        if self._arrays is None:
            assert self.directory is not None
            self._arrays = np.load(self.directory / "arrays.npz", allow_pickle=False)
        return self._arrays[key]

    def value(self, v: Value | None) -> Any:
        """A recorded value, rebuilt: an array from arrays.npz, a number or string itself, a tuple or list of
        them; None for what was only described."""
        if v is None or v.kind == "none":
            return None
        if v.kind == "array":
            return self.array(v.key) if v.key else None
        if v.kind in ("scalar", "str"):
            return v.value
        if v.kind == "seq" and v.items is not None:
            items = [self.value(i) for i in v.items]
            return tuple(items) if v.type == "builtins.tuple" else items
        if v.kind == "map" and v.names is not None:
            return {k: self.value(x) for k, x in v.names.items()}
        if v.kind == "slice" and v.items is not None:
            return slice(*(self.value(i) for i in v.items))
        if v.kind == "record":                          # (no fields: an object with no state of its own)
            return _record(v.type, {k: self.value(x) for k, x in (v.names or {}).items()})
        if v.kind == "table" and v.names is not None and v.type.startswith("polars."):
            import polars as pl
            cols = {k: self.value(x) for k, x in v.names.items()}
            return None if any(c is None for c in cols.values()) else pl.DataFrame(cols)
        if v.kind == "table" and v.names is not None and not v.type.startswith("polars."):
            import pandas as pd  # type: ignore[import-untyped]
            cols = {k: self.value(x) for k, x in v.names.items()}
            if any(c is None for c in cols.values()):
                return None
            if set(cols) == {"values", "index"} and "Series" in v.type:
                return pd.Series(cols["values"], index=cols["index"])
            return pd.DataFrame(cols)
        return None

    def calls_of(self, function: str) -> list[Call]:
        return [c for c in self.calls if c.function == function or c.function.endswith(":" + function)
                or c.function.split(":")[-1] == function]


def _record(type_: str, fields: dict[str, Any]) -> Any:
    """A recorded record, rebuilt as its class's instance (without calling its __init__: its fields as recorded).
    None if its class does not import."""
    import importlib
    module, _, qual = type_.rpartition(".")
    try:
        obj: Any = importlib.import_module(module)
    except ImportError:
        for _ in range(module.count(".")):              # a nested qualname: module.Class.Inner
            module, _, outer = module.rpartition(".")
            qual = f"{outer}.{qual}"
            try:
                obj = importlib.import_module(module)
                break
            except ImportError:
                continue
        else:
            return None
    for part in qual.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    try:
        out = object.__new__(obj)
    except TypeError:                                   # a class with its own __new__ (a subclass of a builtin)
        try:
            out = obj.__new__(obj)
        except TypeError:
            return None
    try:
        for k, v in fields.items():
            object.__setattr__(out, k, v)
    except (TypeError, AttributeError):                 # a class that keeps its state elsewhere: not rebuilt
        return None
    return out


def _plain(d: Any) -> Any:
    """Drop empty fields, so that the manifest stays readable."""
    if isinstance(d, dict):
        return {k: _plain(v) for k, v in d.items()
                if not (v is None or isinstance(v, (dict, list)) and not v) or k == "value"}
    if isinstance(d, list):
        return [_plain(x) for x in d]
    if isinstance(d, np.ndarray):
        return f"array{list(d.shape)} {d.dtype}"
    if isinstance(d, np.generic):
        return d.item()
    return d
