"""The Python tracer: records the calls of chosen functions while any code runs (a test suite, a script), with
sys.monitoring (Python 3.12), so that code not traced costs one event per function, once.

    with Tracer(["pydeseq2"]) as t:
        run()
    t.trace.write("trace/")

Which functions: those whose module is one of the packages named (or whose file is under a directory named).
Each function's first `first` calls are recorded in full, then every `every`-th up to `limit`; every call, recorded
or not, keeps its path (the branches it took, with how many times) and its arrays' shapes, so that paths are counted
over all of them (survey/tracing-design.md §3, §6).

Work sent to other processes is not seen by this tracer: joblib's default backend runs in other processes, so while
tracing joblib runs sequentially (the run's record says so).
"""
from __future__ import annotations

import contextlib
import dis
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType
from typing import Any, Iterator

import numpy as np

try:
    from headway.tracing import env
    from headway.tracing.record import Call, Function, Trace, Value
except ImportError:                                     # run standalone in another interpreter (standalone.py)
    import env  # type: ignore[no-redef, import-not-found]
    from record import Call, Function, Trace, Value  # type: ignore[no-redef, import-not-found]

M: Any = getattr(sys, "monitoring", None)            # Python 3.12+; earlier ones use sys.settrace
MAX_ELEMENTS = 20_000_000                # arrays larger than this are described, not kept
MAX_ITEMS = 32                          # a tuple's or list's items recorded one by one, up to this many


@dataclass
class _Frame:
    call: Call
    code: CodeType
    started: float
    branches: Counter[tuple[int, int]] = field(default_factory=Counter)


class Tracer:
    def __init__(self, packages: list[str], first: int = 20, every: int = 50, limit: int = 200, stored: int = 2000,
                 functions: list[str] | None = None, sequential: bool = True, legacy: bool = False):
        self.packages = [p for p in packages if not p.startswith("/")]
        self.directories = [str(Path(p).resolve()) for p in packages if p.startswith("/")]
        self.functions = set(functions or ())   # qualnames to trace (all of the packages' if empty)
        self.first, self.every, self.limit = first, every, limit
        self.stored = stored                # calls of a function kept as calls (sizes, path); past it, counted
        self.sequential = sequential
        self.force_legacy = legacy          # sys.settrace even where sys.monitoring exists (to test it)
        self.tool: int | None = None
        self.targets: dict[CodeType, str] = {}
        self.skipped: set[CodeType] = set()
        self.stack: list[_Frame] = []
        self.calls: list[Call] = []
        self.arrays: dict[str, np.ndarray] = {}
        self.counts: Counter[str] = Counter()
        self.kept: Counter[str] = Counter()
        self.info: dict[str, Function] = {}
        self.lines: dict[CodeType, dict[int, int]] = {}
        self.trace: Trace | None = None
        self._before: dict[str, Any] = {}
        self._exit: contextlib.ExitStack | None = None

    # ---- which code ------------------------------------------------------------------------------------------

    def _target(self, code: CodeType, frame: Any = None) -> str | None:
        if code in self.targets:
            return self.targets[code]
        if code in self.skipped:
            return None
        name = None
        frame = frame if frame is not None else sys._getframe(2)
        module = frame.f_globals.get("__name__", "")
        file = code.co_filename
        parts = module.split(".")
        test_code = any(x in ("tests", "test", "conftest") or x.startswith("test_") for x in parts)
        if not test_code and (any(module == p or module.startswith(p + ".") for p in self.packages)
                or any(file.startswith(d) for d in self.directories)) and not code.co_name.startswith("<") \
                and not file.startswith("<") and code.co_flags & 0x01:  # a function (not a class body, nor
            # code a library generated, as a dataclass's __init__)
            qualname = getattr(code, "co_qualname", code.co_name)     # Python < 3.11: its name only
            qual = f"{module}:{qualname}"
            if not self.functions or qualname in self.functions or qual in self.functions:
                name = qual
        if name is None:
            self.skipped.add(code)
            return None
        self.targets[code] = name
        self.info[name] = Function(name, file, code.co_firstlineno)
        if self.tool is not None:                       # sys.monitoring: branches by instruction offset
            self.lines[code] = {i.offset: (i.positions.lineno or 0) for i in dis.get_instructions(code)
                                if i.positions is not None}
            M.set_local_events(self.tool, code, M.events.BRANCH)
        return name

    # ---- events ----------------------------------------------------------------------------------------------

    def _start(self, code: CodeType, offset: int) -> Any:
        name = self._target(code)
        if name is None:
            return M.DISABLE
        self._begin(code, name, sys._getframe(1))
        return None

    def _begin(self, code: CodeType, name: str, frame: Any) -> None:
        seq = self.counts[name]
        self.counts[name] += 1
        keep = seq < self.first or (seq % self.every == 0 and self.kept[name] < self.limit)
        parent = self.stack[-1].call.id if self.stack else None
        call = Call(len(self.calls), name, seq, parent, kept=keep)
        if not keep and seq >= self.stored:             # past the cap: only its path will be counted
            call.id = -1
            self.stack.append(_Frame(call, code, time.perf_counter()))
            return
        n = code.co_argcount + code.co_kwonlyargcount + bool(code.co_flags & 0x04) + bool(code.co_flags & 0x08)
        for arg in code.co_varnames[:n]:
            if arg in frame.f_locals:
                call.args[arg] = self._value(frame.f_locals[arg], f"c{call.id}.{arg}", keep)
        if keep:
            self.kept[name] += 1
            if "numpy" in sys.modules:                  # numpy's global random state: draws reproducible
                st = np.random.get_state()
                call.globals["__numpy_random__"] = self._value(
                    [st[0], st[1], st[2], st[3], st[4]], f"c{call.id}.__numpy_random__", True)
            g = frame.f_globals                         # the module's constants it reads, as they are now
            for n_ in code.co_names:
                v_ = g.get(n_)
                if isinstance(v_, (bool, int, float, str)) and not n_.startswith("__"):
                    call.globals[n_] = self._value(v_, "", False)
        self.calls.append(call)
        self.stack.append(_Frame(call, code, time.perf_counter()))

    def _return(self, code: CodeType, offset: int, value: Any) -> Any:
        if code not in self.targets:
            return M.DISABLE
        f = self._pop(code)
        if f is not None:
            f.call.result = self._value(value, f"c{f.call.id}.return", f.call.kept)
        return None

    def _unwind(self, code: CodeType, offset: int, exc: BaseException) -> Any:
        if code not in self.targets:
            return None
        f = self._pop(code)
        if f is not None:
            f.call.raised = type(exc).__name__
        return None

    def _branch(self, code: CodeType, offset: int, destination: int) -> Any:
        for f in reversed(self.stack):
            if f.code is code:
                f.branches[(offset, destination)] += 1
                break
        return None

    def _pop(self, code: CodeType) -> _Frame | None:
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i].code is code:
                f = self.stack.pop(i)
                if f.call.id == -1:                     # past the cap: its path counted, the call dropped
                    info = self.info[f.call.function]
                    lines = self.lines.get(code)
                    keys = {(f"{lines.get(a, 0)}>{lines.get(b, 0)}" if lines is not None else f"{a}>{b}"): n
                            for (a, b), n in f.branches.items()}
                    path = "|".join(sorted(keys)) if not getattr(self, "_legacy", False) else "(not followed)"
                    info.more_paths[path] = info.more_paths.get(path, 0) + 1
                    for k, n in keys.items():
                        info.more_branches[k] = info.more_branches.get(k, 0) + n
                    return None
                f.call.seconds = time.perf_counter() - f.started
                if code in self.lines:                  # offsets: their lines
                    lines = self.lines[code]
                    f.call.branches = {f"{lines.get(a, 0)}>{lines.get(b, 0)}": n for (a, b), n in
                                       f.branches.items()}
                else:                                   # lines already (sys.settrace)
                    f.call.branches = {f"{a}>{b}": n for (a, b), n in f.branches.items()}
                return f
        return None

    # ---- values ----------------------------------------------------------------------------------------------

    def _value(self, x: Any, key: str, keep: bool, depth: int = 0) -> Value:
        t = f"{type(x).__module__}.{type(x).__qualname__}"
        if x is None:
            return Value("none")
        if isinstance(x, (bool, int, float, complex, np.generic)) and not isinstance(x, np.ndarray):
            v = x.item() if isinstance(x, np.generic) else x
            return Value("scalar", t, value=v if not isinstance(v, complex) else str(v))
        if isinstance(x, str):
            return Value("str", t, value=x if len(x) < 1000 else x[:1000])
        arr = _as_array(x)
        if arr is not None and arr.dtype == object:     # labels: kept as text
            try:
                arr = arr.astype(str)
            except (TypeError, ValueError):
                pass
        if arr is not None:
            v = Value("array", t, str(arr.dtype), list(arr.shape))
            if keep and arr.dtype != object and arr.size <= MAX_ELEMENTS:
                self.arrays[key] = np.array(arr, copy=True)
                v.key = key
            return v
        if depth < 3 and isinstance(x, (tuple, list)):
            items = [self._value(i, f"{key}.{k}", keep, depth + 1) for k, i in enumerate(x[:MAX_ITEMS])]
            return Value("seq", t, shape=[len(x)], items=items)
        if depth < 3 and isinstance(x, dict) and all(isinstance(k, str) for k in x) and len(x) <= MAX_ITEMS:
            return Value("map", t, names={k: self._value(i, f"{key}.{k}", keep, depth + 1) for k, i in x.items()})
        if isinstance(x, slice):
            return Value("slice", t, items=[self._value(i, f"{key}.{k}", keep, depth + 1)
                                           for k, i in enumerate((x.start, x.stop, x.step))])
        table = _columns(x) if depth < 3 else None
        fields = _fields(x) if table is None else None
        if fields is not None and depth < 3:            # a record (a dataclass, a small plain object)
            return Value("record", t, names={k: self._value(v, f"{key}.{k}", keep, depth + 1)
                                             for k, v in fields.items()})
        if table is not None:
            return Value("table", t, shape=[table[0]],
                         names={str(c): self._value(col, f"{key}.{c}", keep, depth + 1)
                                for c, col in list(table[1].items())[:MAX_ITEMS]})
        return Value("object", t, value=_attributes(x) if depth < 1 else None)

    # ---- running ---------------------------------------------------------------------------------------------

    def __enter__(self) -> Tracer:
        self._before = env.environment()
        self._exit = contextlib.ExitStack()
        if self.sequential:
            self._exit.enter_context(_in_process())
        self._started = time.time()
        if M is None or self.force_legacy:              # Python < 3.12: sys.settrace
            self._legacy = True
            sys.settrace(self._trace)
            return self
        self._legacy = False
        for tool in range(6):
            if M.get_tool(tool) is None:
                self.tool = tool
                break
        if self.tool is None:
            raise RuntimeError("no free sys.monitoring tool id")
        M.use_tool_id(self.tool, "headway")
        M.register_callback(self.tool, M.events.PY_START, self._start)
        M.register_callback(self.tool, M.events.PY_RETURN, self._return)
        M.register_callback(self.tool, M.events.PY_UNWIND, self._unwind)
        M.register_callback(self.tool, M.events.BRANCH, self._branch)
        M.set_events(self.tool, M.events.PY_START | M.events.PY_RETURN | M.events.PY_UNWIND)
        return self

    def _trace(self, frame: Any, event: str, arg: Any) -> Any:
        """sys.settrace's global function (Python < 3.12): a traced function's frame gets a local one, which sees
        its lines (a jump between lines not one after another is a branch taken) and its return."""
        if event != "call":
            return None
        code = frame.f_code
        name = self._target(code, frame)
        if name is None:
            return None
        self._begin(code, name, frame)
        last = [frame.f_lineno]
        if not self.stack[-1].call.kept:                # its path not followed (line events are costly here):
            frame.f_trace_lines = False                 # on older Pythons, paths are the recorded calls' only

        def local(fr: Any, ev: str, a: Any) -> Any:
            if ev == "line":
                prev, last[0] = last[0], fr.f_lineno
                if fr.f_lineno != prev + 1 and self.stack and self.stack[-1].code is code:
                    self.stack[-1].branches[(prev, fr.f_lineno)] += 1
            elif ev == "return":
                f = self._pop(code)
                if f is not None:
                    f.call.result = self._value(a, f"c{f.call.id}.return", f.call.kept)
            elif ev == "exception":
                pass
            return local
        return local

    def __exit__(self, *exc: Any) -> None:
        if getattr(self, "_legacy", False):
            sys.settrace(None)
        else:
            assert self.tool is not None
            M.set_events(self.tool, 0)
            for code in self.targets:
                M.set_local_events(self.tool, code, 0)
            for event in (M.events.PY_START, M.events.PY_RETURN, M.events.PY_UNWIND, M.events.BRANCH):
                M.register_callback(self.tool, event, None)
            M.free_tool_id(self.tool)
        if self._exit is not None:
            self._exit.close()
        after = env.environment()
        for name, f in self.info.items():
            f.calls = self.counts[name]
        environment = {**self._before, "modules": env.modules(), "changed": env.changes(self._before, after)}
        run = {"started": self._started, "seconds": time.time() - self._started, "packages": self.packages,
               "tracer": "sys.settrace (paths of recorded calls only)" if getattr(self, "_legacy", False)
               else "sys.monitoring",
               "sequential": self.sequential, "first": self.first, "every": self.every, "limit": self.limit,
               "argv": sys.argv}
        self.trace = Trace(environment, run, self.info, self.calls)

    def write(self, directory: str | Path) -> Path:
        assert self.trace is not None
        return self.trace.write(directory, self.arrays)


def _as_array(x: Any) -> np.ndarray | None:
    if isinstance(x, np.ndarray):
        return x
    mod = type(x).__module__.split(".")[0]
    try:
        if mod == "torch":
            return x.detach().cpu().numpy()             # type: ignore[no-any-return]
        if mod in ("jax", "jaxlib"):
            return np.asarray(x)
        if mod == "scipy" and hasattr(x, "toarray") and hasattr(x, "nnz"):
            return x.toarray() if np.prod(x.shape) <= MAX_ELEMENTS else None  # type: ignore[no-any-return]
    except Exception:                                   # noqa: BLE001 - a value that will not convert: described
        return None
    return None


def _columns(x: Any) -> tuple[int, dict[str, Any]] | None:
    """A data frame's rows and columns (pandas, polars), or a Series as one column."""
    mods = {c.__module__.split(".")[0] for c in type(x).__mro__}     # a subclass of pandas' (formulaic's ModelMatrix)
    mod = "pandas" if "pandas" in mods else "polars" if "polars" in mods else type(x).__module__.split(".")[0]
    try:
        if mod == "pandas":
            if hasattr(x, "columns"):
                return len(x), {str(c): x[c].to_numpy() for c in x.columns}
            if hasattr(x, "to_numpy"):
                return len(x), {"values": x.to_numpy(), "index": np.array([str(i) for i in x.index])}
        if mod == "polars" and hasattr(x, "columns"):
            return x.height, {c: x[c].to_numpy() for c in x.columns}
    except Exception:                                   # noqa: BLE001
        return None
    return None


_FUNCTIONS: tuple[type, ...] = (type(len), type(lambda: 0), type(object().__str__), type([].append), type(sys.exit),
              type(Tracer.write))


def _fields(x: Any) -> dict[str, Any] | None:
    """A record's fields: a dataclass's, or a plain object's attributes when there are few and each is a number,
    a string, an array or a short sequence (not a framework's object, whose attributes are its machinery)."""
    import dataclasses
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: getattr(x, f.name) for f in dataclasses.fields(x) if hasattr(x, f.name)}
    d = getattr(x, "__dict__", None)
    if not isinstance(d, dict) or len(d) > MAX_ITEMS or type(x).__module__.split(".")[0] in (
            "builtins", "numpy", "pandas", "polars", "anndata", "torch", "scipy", "sklearn"):
        return None
    if all(v is None or isinstance(v, (bool, int, float, str, np.ndarray, np.generic, tuple, list, dict))
           or getattr(v, "__dict__", None) is not None and not isinstance(v, (type, _FUNCTIONS))
           for v in d.values()):
        return dict(d)                                  # nested objects recorded as records in turn (to depth 2)
    return None


def _attributes(x: Any) -> dict[str, Any] | None:
    """An object's plain attributes (numbers, strings, the types of others): what configures it (a family, a
    link, an inference backend)."""
    d = getattr(x, "__dict__", None)
    if not isinstance(d, dict):
        return None
    out: dict[str, Any] = {}
    for k, v in list(d.items())[:64]:
        if isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
        elif isinstance(v, np.ndarray):
            out[k] = f"array{list(v.shape)} {v.dtype}"
        else:
            out[k] = f"<{type(v).__module__}.{type(v).__qualname__}>"
    return out


@contextlib.contextmanager
def _in_process() -> Iterator[None]:
    """joblib's work run in this process (n_jobs=1), even where the code sets its own backend, so that its calls
    are traced."""
    try:
        import joblib.parallel as jp
    except ImportError:
        yield
        return
    original = jp.Parallel.__init__

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        kwargs["n_jobs"] = 1
        if args:
            args = (1, *args[1:])
        original(self, *args, **kwargs)
    jp.Parallel.__init__ = init                         # type: ignore[method-assign]
    try:
        yield
    finally:
        jp.Parallel.__init__ = original                 # type: ignore[method-assign]


@contextlib.contextmanager
def tracing(packages: list[str], directory: str | Path, **kwargs: Any) -> Iterator[Tracer]:
    t = Tracer(packages, **kwargs)
    with t:
        yield t
    t.write(directory)


__all__ = ["Tracer", "tracing", "defaultdict"]
