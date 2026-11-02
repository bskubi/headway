"""Python, read into the calculus: a function's syntax tree to a Program of terms, each with its signature.

What a name is comes from the module: its imports (np is numpy; expit is scipy.special.expit), its own functions and
classes (records), and its type hints, which give values their types and, in jaxtyping's form
(Float[np.ndarray, "bins groups"]), their index sets. A call is translated by the table of the library it belongs
to (libs/), a method by the table of its receiver's type; anything no table translates is an operator (Op), shown by
its name. A statement the calculus cannot say is Unwritten, named by its line. Nothing is run.
"""
from __future__ import annotations

import ast
import builtins
import copy
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from headway import shapes
from headway.core import libs
from headway.core.libs import Call
from headway.core.terms import (NEW, OPTIMUM, SCALAR, UNKNOWN, Align, Apply, Assume, Axis, Binder, Build, Cases, Const,
                                Effect, Extent, Lambda, Optimum,
                                Expr, Field, For, If, Item, Jump, Let, Mapping, Op, Program, Range, Return,
                                SetPart, Shape, Sig, SliceKey, Stmt, Text, Tup, Unwritten, Var, While, broadcast,
                                shaped, walk_stmts)

DEFAULT_IMPORTS = {"np": "numpy", "numpy": "numpy", "math": "math", "scipy": "scipy", "sp": "scipy",
                   "torch": "torch", "pd": "pandas", "pl": "polars", "jnp": "jax.numpy", "random": "random",
                   "NDArray": "numpy.typing.NDArray",
                   **{d: "jaxtyping." + d for d in shapes.DTYPES}}
OUTPUT_ONLY = {"print", "logging", "warnings", "logger", "log", "tqdm"}     # calls that change no value
SCALAR_TYPES = {"float", "int", "bool", "complex", "float64", "float32", "float16", "int64", "int32", "int16",
                "int8", "uint64", "uint32", "uint16", "uint8", "floating", "integer", "number", "bool_"}
ARRAY_TYPES = {"numpy.ndarray": "numpy.ndarray", "numpy.typing.NDArray": "numpy.ndarray",
               "numpy.typing.ArrayLike": "numpy.ndarray", "torch.Tensor": "torch.Tensor",
               "jax.Array": "jax.Array", "jax.numpy.ndarray": "jax.Array"}
SEQUENCES = {"list", "typing.List", "Sequence", "typing.Sequence", "collections.abc.Sequence", "Iterable",
             "typing.Iterable", "collections.abc.Iterable", "List"}
BINOPS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.Pow: "pow", ast.Mod: "mod",
          ast.FloorDiv: "floordiv", ast.MatMult: "matmul", ast.BitAnd: "and", ast.BitOr: "or",
          ast.LShift: "lshift", ast.RShift: "rshift"}
COMPARES = {ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge",
            ast.Is: "is", ast.IsNot: "isnot", ast.In: "in", ast.NotIn: "notin"}


class Unsupported(Exception):
    """A construct the calculus cannot say."""


_RETURNS: dict[tuple[object, ...], Shape | None] = {}


@dataclass
class Module:
    """What a module's names are: its imports, functions, classes and type aliases."""
    name: str = ""
    imports: dict[str, str] = field(default_factory=dict)      # local name -> qualified ("np" -> "numpy")
    functions: dict[str, ast.FunctionDef] = field(default_factory=dict)
    classes: dict[str, ast.ClassDef] = field(default_factory=dict)
    aliases: dict[str, ast.expr] = field(default_factory=dict)
    constants: dict[str, object] = field(default_factory=dict)    # NAME = a number (a module's constant)
    file: str = ""

    @classmethod
    def read(cls, file: str | Path, name: str = "") -> Module:
        try:
            tree = ast.parse(Path(file).read_text())
        except (OSError, SyntaxError, UnicodeDecodeError):
            return cls(name, file=str(file))
        return cls.of(tree, name, str(file))

    @classmethod
    def of(cls, tree: ast.Module, name: str = "", file: str = "") -> Module:
        m = cls(name, file=file)
        package = name.rpartition(".")[0]
        for st in tree.body:
            if isinstance(st, ast.FunctionDef):
                m.functions[st.name] = st
            elif isinstance(st, ast.ClassDef):
                m.classes[st.name] = st
            elif isinstance(st, ast.Import):
                for a in st.names:
                    m.imports[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
            elif isinstance(st, ast.ImportFrom):
                base = st.module or ""
                if st.level:
                    parts = package.split(".") if package else []
                    parts = parts[:len(parts) - st.level + 1] if st.level > 1 else parts
                    base = ".".join([*parts, *([st.module] if st.module else [])])
                for a in st.names:
                    m.imports[a.asname or a.name] = f"{base}.{a.name}" if base else a.name
            elif isinstance(st, (ast.Assign, ast.AnnAssign)):
                t = st.targets[0] if isinstance(st, ast.Assign) and len(st.targets) == 1 else getattr(st, "target",
                                                                                                        None)
                if isinstance(t, ast.Name) and st.value is not None and _type_like(st.value):
                    m.aliases[t.id] = st.value
                elif isinstance(t, ast.Name) and isinstance(st.value, (ast.Constant, ast.UnaryOp)):
                    try:
                        v = ast.literal_eval(st.value)
                    except ValueError:
                        continue
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        m.constants[t.id] = v
        return m

    def full(self, dotted: str) -> str:
        """What a dotted name stands for, by the module's imports: np.log -> numpy.log."""
        head, _, rest = dotted.partition(".")
        base = self.imports.get(head) or DEFAULT_IMPORTS.get(head) or head
        return base + ("." + rest if rest else "")


def _uncopied(value: Expr) -> Expr:
    """A value named by itself has each index set once: a copy (k', from Xᵀ X's second index, or a solve's) is the
    set itself, unless the set is there twice (a covariance's k and k')."""
    if value.sig is None or not any(a.copy for a in value.sig):
        return value
    names = [a.name for a in value.sig if a.known and a != NEW]
    if len(names) != len(set(names)):
        return value
    sig = tuple(Axis(a.name, 0, a.where, a.part, a.letter) if a.copy else a for a in value.sig)
    return Align(value, "rename", sig=sig, type=value.type, dtype=value.dtype)


OPTIMIZERS = {"scipy.optimize.minimize"}


def _parallel_map(e: ast.Call, qualify: Callable[[ast.expr], str | None]) -> ast.ListComp | None:
    """Parallel(n_jobs=...)(delayed(f)(args) for x in xs) (joblib): the list [f(args) for x in xs] it computes,
    however the work is shared out."""
    if not (isinstance(e.func, ast.Call) and qualify(e.func.func) in ("joblib.Parallel", "joblib.parallel.Parallel")
            and len(e.args) == 1 and not e.keywords and isinstance(e.args[0], (ast.GeneratorExp, ast.ListComp))):
        return None
    gen = e.args[0]
    elt = gen.elt
    if not (isinstance(elt, ast.Call) and isinstance(elt.func, ast.Call)
            and qualify(elt.func.func) in ("joblib.delayed", "joblib.parallel.delayed") and len(elt.func.args) == 1):
        return None
    inner = ast.copy_location(ast.Call(elt.func.args[0], elt.args, elt.keywords), elt)
    return ast.fix_missing_locations(ast.copy_location(ast.ListComp(inner, gen.generators), gen))


def _slot(e: ast.Subscript) -> str | None:
    """x.attr["name"] (or x["name"]), x a name: the slot's key in the environment ("self.var['name']")."""
    if not (isinstance(e.slice, ast.Constant) and isinstance(e.slice.value, str)):
        return None
    base = _dotted(e.value) if isinstance(e.value, ast.Attribute) else e.value.id if isinstance(e.value, ast.Name) \
        else None
    return f"{base}[{e.slice.value!r}]" if base else None


def _returns_named(body: list[ast.stmt], inline: set[str]) -> list[ast.stmt]:
    """A function's body with each call in its return of a function to write in place (g(...)) taken out to an
    assignment before it (`_r0 = g(...)`), so that g is written in place as an assignment is."""
    if not body or not isinstance(body[-1], ast.Return) or body[-1].value is None:
        return body
    ret = body[-1]
    hoisted: list[ast.stmt] = []

    class Hoist(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id in inline:
                name = f"_r{len(hoisted)}"
                hoisted.append(ast.fix_missing_locations(ast.copy_location(
                    ast.Assign([ast.Name(name, ast.Store())], node), ret)))
                return ast.copy_location(ast.Name(name, ast.Load()), node)
            return node
    value = Hoist().visit(ret.value) if ret.value is not None else None
    if not hoisted:
        return body
    back = ast.fix_missing_locations(ast.copy_location(ast.Return(value), ret))
    return [*body[:-1], *hoisted, back]


EXTENT = "extent"                       # a Shape part marking an integer as the size of an index set


def _axes_spec(ann: ast.expr) -> list[str] | None:
    """The axis names of a jaxtyping annotation ("bins groups"), read without its module."""
    if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
        try:
            ann = ast.parse(ann.value, mode="eval").body
        except SyntaxError:
            return None
    if isinstance(ann, ast.BinOp):
        ann = ann.left
    if isinstance(ann, ast.Subscript) and isinstance(ann.slice, ast.Tuple) and len(ann.slice.elts) == 2 \
            and isinstance(ann.slice.elts[1], ast.Constant) and isinstance(ann.slice.elts[1].value, str):
        names = ann.slice.elts[1].value.split()
        return names if all(n.isidentifier() for n in names) else None
    return None


def _type_like(e: ast.expr) -> bool:
    """A module-level assignment that is a type alias (Matrix = Float[np.ndarray, "a b"])."""
    return isinstance(e, ast.Subscript) and isinstance(e.slice, ast.Tuple) and len(e.slice.elts) == 2 \
        and isinstance(e.slice.elts[1], ast.Constant) and isinstance(e.slice.elts[1].value, str)


class Lowerer:
    """Reads one function (or block) into terms. given: the shapes of names known from outside (a spec's symbols,
    with their indices); hints: take type hints as true (clean code's are checked, evaluate.py); kinds: what a
    type checker says a name is at a place in the file ("scalar"; typeinfo.py)."""

    def __init__(self, module: Module | None = None, given: dict[str, Shape] | None = None, hints: bool = True,
                 kinds: Callable[[int, int, str], str | None] | None = None, offset: int = 0, indent: int = 0,
                 inline: dict[str, ast.FunctionDef] | None = None, named: set[str] | None = None):
        self.module = module or Module()
        self.env: dict[str, Shape] = dict(given or {})
        self.given = set(given or {})
        self.hints = hints
        self.kinds = kinds
        self.offset = offset                # the file's line of the source's line 1, less 1
        self.indent = indent                # columns the source was dedented by
        self.inline = dict(inline or {})
        self.axes: dict[str, Axis] = {}     # index set name -> its Axis (one object per set, for the letters)
        self.declared: set[str] = set()     # variables an annotation gives axes (kept as names, with their lines)
        self.named = set(named or ())       # names the spec gives a symbol (a constant so named stays a name)
        self.members_made: dict[str, Expr] = {}     # self.x, as the class's __init__ made it
        self.depth = 0                      # functions read within functions (returns())
        self.local_defs: dict[str, ast.FunctionDef] = {}    # functions defined inside the one being read
        self.inline_constants = True        # a module's constants written as their values (False: as names, their
        #                                     values given at run time, as a trace records them)
        # names holding a part of another array (part = x[:, block]): (x, the key, whether it is a view of x:
        # True for basic indexing, False for a copy, None if not known)
        self.views: dict[str, tuple[ast.expr, ast.expr, bool | None]] = {}
        self.within: set[str] = set()

    # ---- types ------------------------------------------------------------------------------------------------

    def shape_of(self, ann: ast.expr | None, depth: int = 0) -> Shape:
        """What an annotation says a value is."""
        if ann is None or depth > 8:
            return UNKNOWN
        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            try:
                return self.shape_of(ast.parse(ann.value, mode="eval").body, depth + 1)
            except SyntaxError:
                return UNKNOWN
        if isinstance(ann, ast.Name) and ann.id in self.module.aliases:
            return self.shape_of(self.module.aliases[ann.id], depth + 1)
        if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):        # X | None: X
            sides = [s for s in (ann.left, ann.right) if not (isinstance(s, ast.Constant) and s.value is None)]
            return self.shape_of(sides[0], depth + 1) if len(sides) == 1 else UNKNOWN
        dotted = _dotted(ann.value if isinstance(ann, ast.Subscript) else ann)
        if dotted is None:
            return UNKNOWN
        full = self.module.full(dotted)
        head, _, last = full.rpartition(".")
        if isinstance(ann, ast.Subscript):
            args = list(ann.slice.elts) if isinstance(ann.slice, ast.Tuple) else [ann.slice]
            if head.split(".")[0] == "jaxtyping" and last in shapes.DTYPES and len(args) == 2:
                inner = self.shape_of(args[0], depth + 1)
                type_ = inner.type or "numpy.ndarray"
                spec = args[1].value if isinstance(args[1], ast.Constant) and isinstance(args[1].value, str) else None
                dtype = "bool" if last == "Bool" else "int" if last.startswith(("Int", "UInt", "Integer")) else \
                    "float" if last.startswith(("Float", "BFloat")) else ""
                return Shape(type_, self.axes_of(spec) if spec is not None else None, dtype=dtype)
            if last in ("tuple", "Tuple") and not any(isinstance(a, ast.Constant) and a.value is Ellipsis
                                                      for a in args):
                return Shape("tuple", None, parts=tuple(self.shape_of(a, depth + 1) for a in args))
            if full in SEQUENCES or last in ("list", "List", "Sequence", "Iterable"):
                return Shape("list", None, elem=self.shape_of(args[0], depth + 1))
            if full in ARRAY_TYPES:
                return Shape(ARRAY_TYPES[full], None)
            if last in ("Optional",):
                return self.shape_of(args[0], depth + 1)
            return Shape(full)
        if last in SCALAR_TYPES and head in ("", "numpy", "builtins"):
            return Shape("bool" if last in ("bool", "bool_") else "int" if "int" in last else "float", ())
        if full in ARRAY_TYPES:
            return Shape(ARRAY_TYPES[full], None)
        if dotted in self.module.classes:
            return Shape("record:" + dotted)
        return Shape(libs.known_type(full) or full)

    def axes_of(self, spec: str) -> Sig:
        """The axes a jaxtyping shape string names ("cells genes"), each its index set; None if any is not a name
        (a fixed size, "...", "*batch", "#n", an expression)."""
        names = spec.split()
        if not all(n.isidentifier() and n != "_" for n in names):
            return None
        out: list[Axis] = []                            # "genes genes": a square matrix, its second index a copy
        for n in names:
            a = self.axis(n)
            k = sum(1 for b in out if b.name == a.name)
            out.append(Axis(a.name, a.copy + k, a.where, a.part, a.letter) if k else a)
        return tuple(out)

    def axis(self, name: str) -> Axis:
        if name not in self.axes:
            self.axes[name] = Axis(name)
        return self.axes[name]

    def record_fields(self, cls: str) -> dict[str, Shape]:
        c = self.module.classes.get(cls)
        if c is None:
            return {}
        return {st.target.id: self.shape_of(st.annotation) for st in c.body
                if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)}

    # ---- functions --------------------------------------------------------------------------------------------

    def members(self, cls: ast.ClassDef) -> None:
        """A method's class: what its __init__ assigns to self (self.layer1 = nn.Linear(features, hidden)), read so
        that the method's self.layer1 is that value."""
        init = next((st for st in cls.body if isinstance(st, ast.FunctionDef) and st.name == "__init__"), None)
        if init is None:
            return
        sub = Lowerer(self.module, hints=self.hints)
        sub.axes = self.axes
        for a in (*init.args.posonlyargs, *init.args.args, *init.args.kwonlyargs):
            sub.env[a.arg] = sub.shape_of(a.annotation) if self.hints else UNKNOWN
        for st in init.body:
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(t := st.targets[0], ast.Attribute) \
                    and isinstance(t.value, ast.Name) and t.value.id == "self":
                try:
                    value = sub.expr(st.value)
                except Unsupported:
                    continue
                self.env[f"self.{t.attr}"] = value.shape
                self.members_made[f"self.{t.attr}"] = value

    def program(self, fn: ast.FunctionDef, cls: ast.ClassDef | None = None) -> Program:
        if cls is None:                                 # a method: its class, by where it is in the module
            cls = next((c for c in self.module.classes.values() if fn in c.body or any(
                isinstance(s, ast.FunctionDef) and s.name == fn.name and s.lineno == fn.lineno + self.offset
                for s in c.body)), None)
        if cls is not None:
            self.members(cls)
        args = [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]
        if self.hints:                      # a parameter given indices and annotated with axes: those axes are its
            for a in args:
                s = self.env.get(a.arg) if a.arg in self.given else None
                if s is not None and s.type == "list" and s.elem is not None and a.annotation is not None \
                        and _axes_spec(a.annotation) is not None:
                    s = s.elem
                if s is None or s.sig is None or a.annotation is None:
                    continue
                spec = _axes_spec(a.annotation)
                if spec is not None and len(spec) == len(s.sig):
                    for n, ax in zip(spec, s.sig):
                        if ax.known and ax != NEW:
                            self.axes.setdefault(n, ax)
        params = []
        for a in args:
            s = self.env.get(a.arg) if a.arg in self.given else None
            if s is None and self.hints:
                hinted0 = self.shape_of(a.annotation)
                if hinted0.type in TABLES and hinted0.sig is None:      # a table's rows: an index set of its own
                    self.env[a.arg] = Shape(hinted0.type, (self.axis(a.arg),), columns=hinted0.columns)
                    self.given.add(a.arg)
                    s = self.env[a.arg]
            hinted = self.shape_of(a.annotation) if self.hints and a.annotation is not None else UNKNOWN
            if s is not None and s.type == "list" and s.elem is not None and hinted.type != "list":
                s = s.elem                  # a family's symbol, given to a function of one of its elements
            if s is not None and hinted.type:                   # given: its kind from its annotation
                s = Shape(hinted.type or s.type, s.sig, hinted.parts, hinted.elem, hinted.dtype or s.dtype)
            if s is None:
                s = self.shape_of(a.annotation) if self.hints else UNKNOWN
            self.env[a.arg] = s
            params.append((a.arg, s.sig, s.type))
        body = fn.body[1:] if _docstring(fn.body) else fn.body
        return Program(fn.name, params, self.block(body), qualname=f"{self.module.name}.{fn.name}".lstrip("."))

    def block(self, stmts: list[ast.stmt]) -> list[Stmt]:
        out: list[Stmt] = []
        for st in stmts:
            try:
                out += self.stmt(st)
            except Unsupported:
                out.append(Unwritten(line=self.line(st)))
        return out

    def line(self, node: ast.AST) -> int:
        return getattr(node, "lineno", 0) + self.offset

    # ---- statements ------------------------------------------------------------------------------------------

    def stmt(self, st: ast.stmt) -> list[Stmt]:
        line = self.line(st)
        if isinstance(st, ast.Expr):
            if isinstance(st.value, ast.Constant):
                return []                                   # a docstring, a bare constant
            if not isinstance(st.value, ast.Call):
                return []
            if _output_only(st.value):
                return []                                   # printing, logging: not math
            dest = next((k.value for k in st.value.keywords if k.arg == "out"), None)
            if isinstance(dest, ast.Name):                  # np.clip(x, a, b, out=x): x <- ...
                call = ast.Call(st.value.func, st.value.args, [k for k in st.value.keywords if k.arg != "out"])
                value = self.expr(call)
                held = self.env.get(dest.id)
                if held is not None and held.sig is not None:
                    value = shaped(value, held)         # into what x holds: its axes stay what they were
                self.env[dest.id] = value.shape
                return self.changed_in_place(dest.id, Let((dest.id,), value, again=True, inplace=True, line=line),
                                             line)
            return [Effect(self.expr(st.value), line=line)]
        if isinstance(st, ast.Assert):
            return [Assume(self.expr(c), line=line) for c in _holds(st.test)]
        if isinstance(st, ast.FunctionDef):             # a function defined here: read where it is used
            self.local_defs[st.name] = st
            self.inline.setdefault(st.name, st)
            return []
        if isinstance(st, (ast.Pass, ast.Import, ast.ImportFrom, ast.ClassDef, ast.Delete,
                           ast.Raise, ast.Global, ast.Nonlocal)):
            return []
        if isinstance(st, (ast.Assign, ast.AnnAssign)) and st.value is not None and isinstance(st.value, ast.Call) \
                and isinstance(st.value.func, ast.Name) and st.value.func.id in self.inline \
                and (lines := self.inlined(st)) is not None:
            return lines
        if isinstance(st, ast.Assign):
            if len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                self.note_view(st.targets[0].id, st.value)
            value = self.expr(st.value)
            out: list[Stmt] = []
            for t in st.targets:
                out += self.assign(t, value, None, line)
            return out
        if isinstance(st, ast.AnnAssign):
            if st.value is None:
                return []
            declared = self.shape_of(st.annotation) if self.hints else UNKNOWN
            return self.assign(st.target, self.expr(st.value), declared, line)
        if isinstance(st, ast.AugAssign):
            target = self.expr(st.target)
            value = self.binop(st.op, target, self.expr(st.value))
            if target.sig is not None and value.sig != target.sig and target.type in ("numpy.ndarray", "torch.Tensor"):
                value = shaped(value, target.shape)     # changed in place: its axes stay what they were
            if isinstance(st.target, ast.Name):
                self.env[st.target.id] = value.shape
                return self.changed_in_place(st.target.id, Let((st.target.id,), value, again=True, inplace=True,
                                                               line=line), line)
            return self.assign(st.target, value, None, line)
        if isinstance(st, ast.Return):
            return [Return(self.expr(st.value) if st.value is not None else None, line=line)]
        if isinstance(st, (ast.For, ast.AsyncFor)):
            domain = self.expr(st.iter)
            names = _names(st.target)
            if names is None:
                raise Unsupported
            before = dict(self.env)
            self.bind_loop(st.target, domain)
            body = self.block(st.body)
            orelse = self.block(st.orelse)
            self.merge(before, dict(self.env))
            var = None if names == ("_",) else names
            return [For(var, domain, tuple(body), tuple(orelse), line=line)]
        if isinstance(st, ast.While):
            before = dict(self.env)
            test = self.expr(st.test)
            body = self.block(st.body)
            orelse = self.block(st.orelse)
            self.merge(before, dict(self.env))
            return [While(test, tuple(body), tuple(orelse), line=line)]
        if isinstance(st, ast.If):
            test = self.expr(st.test)
            before = dict(self.env)
            body = self.block(st.body)
            after_body = dict(self.env)
            self.env = dict(before)
            orelse = self.block(st.orelse)
            self.merge(after_body, dict(self.env))
            return [If(test, tuple(body), tuple(orelse), line=line)]
        if isinstance(st, (ast.Continue, ast.Break)):
            return [Jump("continue" if isinstance(st, ast.Continue) else "break", line=line)]
        if isinstance(st, (ast.With, ast.Try)):
            return self.block(st.body)
        raise Unsupported

    def merge(self, a: dict[str, Shape], b: dict[str, Shape]) -> None:
        """After branches (or a loop and the code before it): a variable's shape where they agree, else unknown."""
        self.env = {}
        for k in {*a, *b}:
            x, y = a.get(k), b.get(k)
            if x is None or y is None:
                self.env[k] = x or y or UNKNOWN
            else:
                self.env[k] = x if x == y else Shape(x.type if x.type == y.type else "", x.sig if x.sig == y.sig
                                                     else None)

    def assign(self, t: ast.expr, value: Expr, declared: Shape | None, line: int) -> list[Stmt]:
        if isinstance(t, ast.Name):
            again = t.id in self.env
            if declared is not None and (declared.sig is not None or declared.type):
                value = self.declare(value, declared)
                if declared.sig:
                    self.declared.add(t.id)
            value = _uncopied(value)
            if value.type in TABLES and value.sig is None:  # a table read from elsewhere: its rows a set of its own
                value = replace(value, sig=(self.axis(t.id),))
            self.env[t.id] = value.shape
            if isinstance(value, Extent) and isinstance(value.axis, Axis) and value.axis.known:
                # n = x.shape[1]: n is that index set's size, so that np.zeros(n) is indexed by it
                self.env[t.id] = Shape(value.type or "int", (), parts=(Shape(EXTENT, (value.axis,)),))
            return [Let((t.id,), value, again=again, line=line)]
        if isinstance(t, (ast.Tuple, ast.List)) and all(isinstance(e, ast.Name) for e in t.elts):
            names = tuple(e.id for e in t.elts if isinstance(e, ast.Name))
            again = all(n in self.env for n in names)
            parts = value.parts if value.parts is not None and len(value.parts) == len(names) else \
                tuple(i.shape for i in value.items) if isinstance(value, Tup) and len(value.items) == len(names) \
                else (UNKNOWN,) * len(names)
            for n, s in zip(names, parts):
                self.env[n] = s
            return [Let(names, value, again=again, line=line)]
        if isinstance(t, (ast.Tuple, ast.List)) and isinstance(value, Tup) and len(value.items) == len(t.elts):
            # a[k], b[k] = e, f: each its own assignment (the values are computed before any is assigned)
            out: list[Stmt] = []
            for elt, item in zip(t.elts, value.items):
                out += self.assign(elt, item, None, line)
            return out
        if isinstance(t, ast.Subscript) and (slot := _slot(t)) is not None:
            # obj["name"] = e (an AnnData's var, a dict of results): the slot holds e from here on
            self.env[slot] = value.shape
        if isinstance(t, ast.Subscript):              # out[k] = e: as written
            items = list(t.slice.elts) if isinstance(t.slice, ast.Tuple) else [t.slice]
            keys = [self.key(s) for s in items]
            try:
                shown: Expr | None = self.expr(t)
            except Unsupported:
                shown = None
            return [SetPart(Item(self.expr(t.value), keys[0] if len(keys) == 1 else Tup(tuple(keys))), value,
                            shown, line=line)]
        if isinstance(t, ast.Attribute):
            return [SetPart(Field(self.expr(t.value), t.attr), value, line=line)]
        raise Unsupported

    def changed_in_place(self, name: str, change: Stmt, line: int) -> list[Stmt]:
        """A name's value changed in place: if it is a view of another array (part = x[:, b]), that array changes
        too, written as its part set to the name's new value; if it may be a view (its key not known to be basic
        or advanced indexing), the line is not written (what changes is not known)."""
        if name not in self.views:
            return [change]
        base, key, view = self.views[name]
        if view is False:
            return [change]
        if view is None:
            raise Unsupported
        target = ast.copy_location(ast.Subscript(base, key, ast.Store()), base)
        return [change, *self.assign(target, Var(name, sig=self.env.get(name, UNKNOWN).sig,
                                                 type=self.env.get(name, UNKNOWN).type), None, line)]

    def note_view(self, name: str, value: ast.expr) -> None:
        """name = x[...]: whether name is a view of x (numpy's basic indexing: slices, integers), a copy (an
        array of positions or a mask), or not known."""
        self.views.pop(name, None)
        if not (isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name)):
            return
        if self.env.get(value.value.id, UNKNOWN).type not in ("numpy.ndarray", "torch.Tensor", ""):
            return
        keys = list(value.slice.elts) if isinstance(value.slice, ast.Tuple) else [value.slice]
        kinds: list[bool | None] = []
        for k in keys:
            if isinstance(k, ast.Slice) or isinstance(k, ast.Constant) and (k.value is Ellipsis or k.value is None
                                                                            or type(k.value) is int):
                kinds.append(True)
            elif isinstance(k, ast.Name):
                t = self.env.get(k.id, UNKNOWN)
                kinds.append(True if t.type in ("slice", "int") and t.sig in (None, ()) else
                             False if t.sig else None)
            else:
                kinds.append(False if isinstance(k, (ast.List, ast.Compare)) else None)
        view = False if False in kinds else None if None in kinds else True
        whole = all(isinstance(k, ast.Name) or isinstance(k, ast.Constant) and type(k.value) is int for k in keys)
        if view and whole:                              # integers only: a view if fewer than the array's axes,
            ndim = self.env.get(value.value.id, UNKNOWN).sig    # else a number (a copy); not known without them
            view = None if ndim is None else (True if len(keys) < len(ndim) else False)
        self.views[name] = (value.value, value.slice, view)

    def declare(self, value: Expr, declared: Shape) -> Expr:
        """A value given an annotation: its type and index sets as declared (an annotation names a subset's index
        set: learning: "learners columns" = x[learns]). A declaration of another number of axes is not taken."""
        if declared.sig is not None and value.sig is not None and len(declared.sig) != len(value.sig):
            return value
        if declared.sig is not None and value.sig is not None and declared.sig != value.sig:
            # the same values, their axes' index sets named as declared (a part of one set, given its own name)
            return Align(value, "rename", sig=declared.sig, type=declared.type or value.type, dtype=value.dtype)
        return shaped(value, Shape(declared.type or value.type, declared.sig if declared.sig is not None
                                   else value.sig, value.parts, declared.elem or value.elem, value.dtype))

    def bind_loop(self, target: ast.expr, domain: Expr) -> None:
        """A loop's (or comprehension's) variables: their shapes, from what it runs over."""
        names = _names(target) or ()
        if isinstance(domain, Range):
            for n in names:
                self.env[n] = Shape("int", ())
            return
        if isinstance(domain, Op) and domain.qualname == "builtins.enumerate" and isinstance(target, ast.Tuple) \
                and len(target.elts) == 2 and domain.args:
            self.bind(target.elts[0], Shape("int", ()))
            self.bind_loop(target.elts[1], domain.args[0])
            return
        if isinstance(domain, Op) and domain.qualname == "builtins.zip" and isinstance(target, ast.Tuple) \
                and len(target.elts) == len(domain.args):
            for t, d in zip(target.elts, domain.args):
                self.bind_loop(t, d)
            return
        elem = domain.elem or UNKNOWN
        if isinstance(target, ast.Tuple) and elem.parts is not None and len(elem.parts) == len(target.elts):
            for t, s in zip(target.elts, elem.parts):
                self.bind(t, s)
            return
        self.bind(target, elem if not isinstance(target, ast.Tuple) else UNKNOWN)

    def bind(self, t: ast.expr, s: Shape) -> None:
        for n in _names(t) or ():
            self.env[n] = s if isinstance(t, ast.Name) else UNKNOWN

    def inlined(self, st: ast.Assign | ast.AnnAssign) -> list[Stmt] | None:
        """x = f(a, b), f one of the functions to write in place: f's own statements, its parameters replaced by
        what the call gives, its return assigned to x. None if it cannot be (a parameter f reassigns given other
        than a variable, a return before the end, an argument it does not take)."""
        call = st.value
        assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        helper = self.inline[call.func.id]
        if not helper.body or not isinstance(helper.body[-1], ast.Return) or helper.body[-1].value is None \
                or any(isinstance(n, ast.Return) for b in helper.body[:-1] for n in ast.walk(b)):
            return None
        a = helper.args
        params = [*a.posonlyargs, *a.args]
        given: dict[str, ast.expr] = {}
        for p, d in zip(params[len(params) - len(a.defaults):], a.defaults):
            given[p.arg] = d
        for p, kd in zip(a.kwonlyargs, a.kw_defaults):
            if kd is not None:
                given[p.arg] = kd
        if len(call.args) > len(params) or any(isinstance(x, ast.Starred) for x in call.args):
            return None
        given |= {p.arg: x for p, x in zip(params, call.args)}
        for k in call.keywords:
            if k.arg is None or k.arg not in {p.arg for p in (*params, *a.kwonlyargs)}:
                return None
            given[k.arg] = k.value
        stored = {n.id for b in helper.body for n in ast.walk(b) if isinstance(n, ast.Name)
                  and isinstance(n.ctx, ast.Store)}
        if any(p in stored and not isinstance(x, ast.Name) for p, x in given.items()):
            return None

        class Rename(ast.NodeTransformer):
            def visit_Name(self, n: ast.Name) -> ast.AST:
                x = given.get(n.id)
                if x is None:
                    return n
                if isinstance(n.ctx, ast.Store):
                    return ast.Name(x.id, ast.Store()) if isinstance(x, ast.Name) else n
                return copy.deepcopy(x)
        body = [Rename().visit(copy.deepcopy(b)) for b in helper.body]
        for b in body:
            ast.copy_location(b, st)
            for n in ast.walk(b):
                if hasattr(n, "lineno"):
                    n.lineno = st.lineno
        ret = body[-1]
        assert isinstance(ret, ast.Return) and ret.value is not None
        out = self.block(body[:-1])
        target = st.targets[0] if isinstance(st, ast.Assign) else st.target
        if isinstance(target, ast.Name) and isinstance(ret.value, ast.Name) and ret.value.id == target.id:
            return out                                  # return x, into x: nothing more to write
        last: ast.stmt = ast.Assign([target], ret.value) if isinstance(st, ast.Assign) else ast.AnnAssign(
            target=st.target, annotation=st.annotation, value=ret.value, simple=st.simple)
        ast.copy_location(last, st)
        return out + self.stmt(last)

    # ---- expressions -----------------------------------------------------------------------------------------

    def expr(self, e: ast.expr) -> Expr:
        if isinstance(e, ast.Name):
            return self.name(e)
        if isinstance(e, ast.Constant):
            v = e.value
            if v is Ellipsis:
                return Const(..., type="ellipsis")
            if isinstance(v, str):
                return Text(v, type="str")
            t = "bool" if isinstance(v, bool) else "int" if isinstance(v, int) else "float" \
                if isinstance(v, float) else "none" if v is None else "complex" if isinstance(v, complex) else ""
            return Const(v, sig=() if v is not None else None, type=t)
        return self.compound(e)

    def compound(self, e: ast.expr) -> Expr:
        if isinstance(e, ast.Lambda):                   # a function as a value: its body in its parameters
            lam = self.function_value(e, [])
            if lam is None:
                raise Unsupported
            return lam
        if isinstance(e, ast.BinOp):
            return self.binop(e.op, self.expr(e.left), self.expr(e.right))
        if isinstance(e, ast.UnaryOp):
            u = self.expr(e.operand)
            if isinstance(e.op, ast.UAdd):
                return u
            if isinstance(e.op, ast.USub):
                if isinstance(u, Const) and isinstance(u.value, (int, float)) and not isinstance(u.value, bool):
                    return Const(-u.value, sig=(), type=u.type)
                return Apply("neg", (u,), sig=u.sig, type=u.type)
            if isinstance(e.op, ast.Not):
                return Apply("not", (u,), sig=u.sig, type=u.type if u.sig else "bool", dtype="bool")
            return self.call_entry("numpy.invert", [u], {}, e) or Op("invert", (u,), sig=u.sig, type=u.type)
        if isinstance(e, ast.BoolOp):
            vals = [self.expr(x) for x in e.values]
            out = vals[0]
            for w in vals[1:]:
                out = Apply("and" if isinstance(e.op, ast.And) else "or", (out, w), sig=broadcast(out.sig, w.sig),
                            type="bool", dtype="bool")
            return out
        if isinstance(e, ast.Compare):
            left = self.expr(e.left)
            parts = []
            for op, right in zip(e.ops, e.comparators):
                r = self.expr(right)
                parts.append(Apply(COMPARES[type(op)], (left, r), sig=broadcast(left.sig, r.sig),
                                   type=_array_type(left, r) or "bool", dtype="bool"))
                left = r
            out = parts[0]
            for p in parts[1:]:                         # a < b < c: a < b and b < c
                out = Apply("and", (out, p), sig=broadcast(out.sig, p.sig), type=_array_type(out, p) or "bool",
                            dtype="bool")
            return out
        if isinstance(e, ast.IfExp):
            branches = []
            node: ast.expr = e
            while isinstance(node, ast.IfExp):
                branches.append((self.expr(node.test), self.expr(node.body)))
                node = node.orelse
            other = self.expr(node)
            sig = broadcast(*(v.sig for _, v in branches), other.sig)
            same = all(v.type == other.type for _, v in branches)
            return Cases(tuple(branches), other, sig=sig, type=other.type if same else "")
        if isinstance(e, ast.Attribute):
            return self.attribute(e)
        if isinstance(e, ast.Subscript):
            return self.subscript(e)
        if isinstance(e, ast.Call):
            return self.call(e)
        if isinstance(e, (ast.Tuple, ast.List, ast.Set)):
            items = tuple(self.expr(x) for x in e.elts)
            kind = "tuple" if isinstance(e, ast.Tuple) else "list" if isinstance(e, ast.List) else "set"
            same = bool(items) and all(i.shape == items[0].shape for i in items)
            return Tup(items, kind, type=kind, parts=tuple(i.shape for i in items),
                       elem=items[0].shape if same else None)
        if isinstance(e, ast.Dict):
            if any(k is None for k in e.keys):
                raise Unsupported
            return Mapping(tuple((self.expr(k), self.expr(v)) for k, v in zip(e.keys, e.values) if k is not None),
                           type="dict")
        if isinstance(e, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
            over, where, before = self.generators(e.generators)
            body = self.expr(e.elt)
            self.env = before
            return Build(body, over, where, type="list", elem=body.shape)
        if isinstance(e, ast.DictComp):
            over, where, before = self.generators(e.generators)
            pair = (self.expr(e.key), self.expr(e.value))
            self.env = before
            return Mapping((pair,), over, type="dict")
        if isinstance(e, ast.JoinedStr):              # f"c{i + 1}": the text it makes, from its values
            template, values = "", []
            for part in e.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    template += part.value.replace("{", "{{").replace("}", "}}")
                elif isinstance(part, ast.FormattedValue):
                    spec = part.format_spec
                    if spec is not None and not (isinstance(spec, ast.JoinedStr) and all(
                            isinstance(v, ast.Constant) for v in spec.values)):
                        return Text(None, type="str")
                    conv = "!" + chr(part.conversion) if part.conversion not in (-1, None) else ""
                    fmt = ":" + "".join(str(v.value) for v in spec.values) if spec is not None else ""  # type: ignore[attr-defined]
                    template += "{" + conv + fmt + "}"
                    values.append(self.expr(part.value))
                else:
                    return Text(None, type="str")
            return Op("format", (Text(template, type="str"), *values), qualname="builtins.str.format", known=True,
                      type="str", sig=broadcast(*(v.sig for v in values)) if values and all(
                          v.sig is not None for v in values) else ())
        if isinstance(e, ast.Starred):
            raise Unsupported
        raise Unsupported

    def generators(self, gens: list[ast.comprehension]) -> tuple[tuple[Binder, ...], Expr | None, dict[str, Shape]]:
        before = dict(self.env)
        over: list[Binder] = []
        where: Expr | None = None
        for g in gens:
            domain = self.expr(g.iter)
            var = _target_names(g.target)
            if var is None:
                raise Unsupported
            self.bind_loop(g.target, domain)
            over.append(Binder(var, domain))
            for c in g.ifs:
                t = self.expr(c)
                where = t if where is None else Apply("and", (where, t), sig=(), type="bool")
        return tuple(over), where, before

    def name(self, e: ast.Name) -> Expr:
        n = e.id
        if n in self.env:
            s = self.env[n]
            if s.sig is None and self.kinds is not None and n not in self.given and hasattr(e, "lineno") \
                    and self.kinds(e.lineno + self.offset, e.col_offset + self.indent, n) == "scalar":
                s = Shape(s.type or "float", ())
            return shaped(Var(n), s)
        q = self.qualify(e)
        if q is not None and (c := libs.constant(q)) is not None:
            return c
        if n in self.module.constants and n not in self.named and self.inline_constants:  # RIDGE = 1e-6
            v = self.module.constants[n]
            return Const(v, sig=(), type="int" if isinstance(v, int) else "float")
        if n in self.module.classes or n in self.module.functions:
            return Var(n, type="function")
        s = UNKNOWN
        if self.kinds is not None and hasattr(e, "lineno") \
                and self.kinds(e.lineno + self.offset, e.col_offset + self.indent, n) == "scalar":
            s = Shape("float", ())
        return shaped(Var(n), s)

    def qualify(self, e: ast.expr) -> str | None:
        """What a name or dotted name stands for, if it is not a variable: a module's function, a builtin, one of
        the module's own functions or classes."""
        d = _dotted(e)
        if d is None:
            return None
        head, _, rest = d.partition(".")
        if head in self.env:
            return None
        if head in self.module.imports or head in DEFAULT_IMPORTS:
            return self.module.full(d)
        if head in self.module.functions or head in self.module.classes:
            return f"{self.module.name}.{d}".lstrip(".")
        if not rest and hasattr(builtins, head):
            return "builtins." + head
        return None

    def optimum(self, e: ast.Call, q: str) -> Expr:
        """scipy.optimize.minimize(fun, x0, jac=, method=, bounds=): the point where fun is least, its function read
        as math in its parameter, which has x0's shape."""
        kw = {k.arg: k.value for k in e.keywords if k.arg}
        fun_node = e.args[0] if e.args else kw.get("fun")
        x0_node = e.args[1] if len(e.args) > 1 else kw.get("x0")
        if fun_node is None or x0_node is None or set(kw) - {"fun", "x0", "jac", "method", "bounds", "tol", "options"}:
            return self.opaque(e)
        start = self.expr(x0_node)
        fun = self.function_value(fun_node, [start.shape])
        jac_node = kw.get("jac")
        jac = self.function_value(jac_node, [start.shape]) if jac_node is not None else None
        method_node = kw.get("method")
        method = self.expr(method_node) if method_node is not None else None
        bounds = self.expr(kw["bounds"]) if "bounds" in kw else None
        if fun is None:
            return self.opaque(e)
        return Optimum("min", fun, start, method, bounds, jac, type=OPTIMUM,
                       parts=(start.shape, Shape("float", ()), Shape("bool", ())))

    def function_value(self, node: ast.expr, shapes: list[Shape]) -> Expr | None:
        """A function passed as a value (a lambda, a function defined in this one, a function of the module): its
        body as an expression in its parameters, the first of which has the shape given. None if it cannot be read
        as one expression (an Op naming it is used instead by the caller)."""
        if isinstance(node, ast.Lambda):
            fn = ast.FunctionDef(name="", args=node.args, body=[ast.Return(node.body)], decorator_list=[],
                                 returns=None, type_params=[])
            ast.copy_location(fn, node)
        elif isinstance(node, ast.Name) and node.id in self.local_defs:
            fn = self.local_defs[node.id]
        elif isinstance(node, ast.Name) and node.id in self.module.functions:
            fn = self.module.functions[node.id]
        else:
            return None
        params = [a.arg for a in (*fn.args.posonlyargs, *fn.args.args)]
        if not params:
            return None
        given = dict(self.env)
        for p_, s_ in zip(params, shapes):
            given[p_] = s_
        sub = Lowerer(self.module, given, hints=self.hints, kinds=None, offset=self.offset, indent=self.indent,
                      inline={**self.inline, **self.local_defs}, named=self.named)
        sub.local_defs = dict(self.local_defs)
        sub.axes = self.axes
        sub.inline_constants = self.inline_constants
        body = _returns_named(fn.body, set(sub.inline))
        stmts = sub.block(body[1:] if _docstring(body) else body)
        from headway.core.fold import fold_body
        folded = fold_body(stmts, set())
        if not folded or len(folded) != 1 or not isinstance(folded[0], Return) or folded[0].value is None:
            return None
        binders = tuple(Binder(p_, Const(None))
                        for p_ in params)
        return Lambda(binders, folded[0].value, fn.name, type="function")

    def attribute(self, e: ast.Attribute) -> Expr:
        q = self.qualify(e)
        if q is not None:
            c = libs.constant(q)
            if c is not None:
                return c
            return Var(e.attr, type="module")           # a module, a function: named by itself
        base = self.expr(e.value)
        dotted = _dotted(e)
        if dotted is not None and dotted in self.env:   # a name given for x.attr (s.pcs)
            return shaped(Field(base, e.attr), self.env[dotted])
        entry = libs.attribute(base.type or "?", e.attr)
        if entry is not None and (out := entry(Call(e.attr, [base], {}, self, receiver=True, node=e))) is not None:
            return out
        if base.type.startswith("record:"):
            s = self.record_fields(base.type.removeprefix("record:")).get(e.attr, UNKNOWN)
            return shaped(Field(base, e.attr), s)
        if base.type == OPTIMUM and base.parts is not None and e.attr in ("x", "fun", "success"):
            return shaped(Field(base, e.attr), base.parts[("x", "fun", "success").index(e.attr)])
        return Field(base, e.attr)

    def subscript(self, e: ast.Subscript) -> Expr:
        slot = _slot(e)
        if slot is not None and slot in self.env:       # a slot written earlier in this function: what it holds
            return shaped(Item(self.expr(e.value), Const(e.slice.value, type="str")), self.env[slot])  # type: ignore[attr-defined]
        base = self.expr(e.value)
        items = list(e.slice.elts) if isinstance(e.slice, ast.Tuple) else [e.slice]
        keys = [self.key(s) for s in items]
        if isinstance(base, Tup) and len(keys) == 1 and isinstance(keys[0], Const) and type(keys[0].value) is int \
                and -len(base.items) <= keys[0].value < len(base.items):
            return base.items[keys[0].value]           # (a, b)[0], x.shape[1]: that item
        entry = libs.method(base.type or "?", "__getitem__")
        if entry is not None and (out := entry(Call("__getitem__", [base, *keys], {}, self, receiver=True,
                                                    node=e))) is not None:
            return out
        if base.type == "tuple" and base.parts is not None and len(keys) == 1 and isinstance(keys[0], Const) \
                and isinstance(keys[0].value, int) and -len(base.parts) <= keys[0].value < len(base.parts):
            return shaped(Item(base, keys[0]), base.parts[keys[0].value])
        elem = base.elem if base.type == "list" and len(keys) == 1 and keys[0].sig == () else None
        key = keys[0] if len(keys) == 1 else Tup(tuple(keys))
        return shaped(Item(base, key), elem) if elem is not None else Item(base, key)

    def key(self, s: ast.expr) -> Expr:
        if isinstance(s, ast.Slice):
            return SliceKey(*(self.expr(x) if x is not None else None for x in (s.lower, s.upper, s.step)),
                            type="slice")
        return self.expr(s)

    def binop(self, op: ast.operator, a: Expr, b: Expr) -> Expr:
        name = BINOPS.get(type(op))
        if name is None:
            raise Unsupported
        typ = _array_type(a, b) or ("int" if a.type == b.type == "int" and name not in ("div", "pow") else
                                    "float" if a.sig == () and b.sig == () else "")
        if name in ("and", "or"):
            return Apply(name, (a, b), sig=broadcast(a.sig, b.sig), type=typ or "bool", dtype="bool")
        if name == "matmul":
            entry = libs.function(_library_of(a, b) + ".matmul")
            if entry is not None and (out := entry(Call("matmul", [a, b], {}, self))) is not None:
                return out
            return Apply("matmul", (a, b), sig=None, type=typ)
        return Apply(name, (a, b), sig=broadcast(a.sig, b.sig), type=typ)

    def call(self, e: ast.Call) -> Expr:
        f = e.func
        family = _parallel_map(e, self.qualify)
        if family is not None:                          # joblib's Parallel(...)(delayed(f)(x) for ...): f's family
            return self.expr(family)
        if any(isinstance(a, ast.Starred) for a in e.args) or any(k.arg is None for k in e.keywords):
            return self.opaque(e)
        if isinstance(f, ast.Name) and f.id in self.local_defs and not e.keywords:
            # a function defined in this one, called in an expression: its body, its parameters the arguments
            args = [self.expr(a) for a in e.args]
            lam = self.function_value(f, [a.shape for a in args])
            if isinstance(lam, Lambda) and len(lam.over) == len(args):
                from headway.core.fold import substitute
                return substitute(lam.body, {b.var: a for b, a in zip(lam.over, args) if isinstance(b.var, str)})
        q = self.qualify(f)
        if q in OPTIMIZERS:
            return self.optimum(e, q)
        if q is not None:                               # a function of a module, a builtin, the module's own
            args = [self.expr(a) for a in e.args]
            kwargs = {k.arg: self.expr(k.value) for k in e.keywords if k.arg}
            local_name = q.rpartition(".")[2]
            if isinstance(f, ast.Name) and f.id in self.module.functions:
                fn = self.module.functions[f.id]
                s = self.shape_of(fn.returns) if self.hints else UNKNOWN
                if s.sig is None and s.parts is None:
                    s = self.returns(fn, args, kwargs) or s
                return shaped(Op(f.id, tuple(args), tuple(kwargs.items()), qualname=q, local=True), s)
            if isinstance(f, ast.Name) and f.id in self.module.classes:
                return Op(f.id, tuple(args), tuple(kwargs.items()), qualname=q, local=True, type="record:" + f.id)
            out = self.call_entry(q, args, kwargs, e)
            if out is not None:
                return out
            written = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else local_name
            op = Op(written, tuple(args), tuple(kwargs.items()), qualname=q)
            found = _foreign(q)                         # a function of another module of the code: its shape
            if found is not None:
                module, fn = found
                s = self.shape_of(fn.returns) if self.hints else UNKNOWN
                if s.sig is None and s.parts is None:
                    saved = self.module
                    self.module = module
                    try:
                        s = self.returns(fn, args, kwargs) or s
                    finally:
                        self.module = saved
                return shaped(op, s)
            return op
        if isinstance(f, ast.Attribute) and (_dotted(f) or "") in self.env:
            # a value that is called (self.layer1(x), a layer the class made): its type's __call__
            callee = self.attribute(f)
            args = [self.expr(a) for a in e.args]
            kwargs = {k.arg: self.expr(k.value) for k in e.keywords if k.arg}
            entry = libs.method(callee.type or "?", "__call__")
            if entry is not None and (out := entry(Call("__call__", [callee, *args], kwargs, self, receiver=True,
                                                        node=e))) is not None:
                return out
            return Op(f.attr, tuple(args), tuple(kwargs.items()), qualname=f"{callee.type or '?'}.__call__")
        if isinstance(f, ast.Attribute):                # a method of a value
            receiver = self.expr(f.value)
            args = [self.expr(a) for a in e.args]
            kwargs = {k.arg: self.expr(k.value) for k in e.keywords if k.arg}
            entry = libs.method(receiver.type or "?", f.attr)
            if entry is None and receiver.type in ("float", "int", "bool"):
                entry = libs.method("?", f.attr)        # a number's .astype, .item: numpy's (a numpy scalar's)
            if entry is not None and (out := entry(Call(f.attr, [receiver, *args], kwargs, self, receiver=True,
                                                        node=e))) is not None:
                return out
            return Op(f.attr, (receiver, *args), tuple(kwargs.items()), qualname=f"{receiver.type or '?'}.{f.attr}")
        if isinstance(f, ast.Name):                     # a variable that is a function
            args = [self.expr(a) for a in e.args]
            kwargs = {k.arg: self.expr(k.value) for k in e.keywords if k.arg}
            return Op(f.id, tuple(args), tuple(kwargs.items()))
        return self.opaque(e)

    def returns(self, fn: ast.FunctionDef, args: list[Expr], kwargs: dict[str, Expr]) -> Shape | None:
        """What one of the module's functions returns, given these arguments: its own code read with them (the
        shape of its return, if it has one return and its shape is known); None if not known. Not deeper than a few
        calls, and never into itself."""
        if self.depth > 3 or fn.name in self.within:
            return None
        params = [a.arg for a in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)]
        given = dict(zip(params, (a.shape for a in args))) | {k: v.shape for k, v in kwargs.items() if k in params}
        key = (self.module.file, self.module.name, fn.name, fn.lineno,
               tuple(sorted((k, repr(v.sig), v.type) for k, v in given.items())))
        if key in _RETURNS:
            return _RETURNS[key]
        sub = Lowerer(self.module, given, hints=self.hints)
        sub.axes = self.axes
        sub.depth = self.depth + 1
        sub.within = self.within | {fn.name}
        try:
            prog = sub.program(fn)
        except Exception:                               # noqa: BLE001 (a function it cannot read: not known)
            prog = None
        rets = [st for st in walk_stmts(prog.body) if isinstance(st, Return)] if prog is not None else []
        out = rets[0].value.shape if len(rets) == 1 and rets[0].value is not None else None
        if out is not None and out.sig is None and out.parts is None:
            out = None
        _RETURNS[key] = out
        return out

    def call_entry(self, q: str, args: list[Expr], kwargs: dict[str, Expr], node: ast.AST) -> Expr | None:
        entry = libs.function(q)
        if entry is None:
            return None
        return entry(Call(q, args, kwargs, self, node=node))

    def opaque(self, e: ast.Call) -> Expr:
        """f(*a, **k): as written, its arguments as far as they can be."""
        f = e.func
        name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ast.unparse(f)
        if any(k.arg is None for k in e.keywords) or isinstance(f, ast.Attribute) and self.qualify(f) is None:
            raise Unsupported                           # f(**k), x.m(*a): not written
        args = tuple(Op("*", (self.expr(a.value),), qualname="*") if isinstance(a, ast.Starred) else self.expr(a)
                     for a in e.args)
        kwargs = tuple((k.arg, self.expr(k.value)) for k in e.keywords if k.arg)
        return Op(name, args, kwargs, qualname=self.qualify(f) or "")


# ---- helpers ---------------------------------------------------------------------------------------------------

TABLES = ("polars.DataFrame", "pandas.DataFrame")
LIBRARIES = ("numpy", "scipy", "torch", "jax", "pandas", "polars", "sklearn", "math", "builtins", "random",
             "statsmodels", "matplotlib", "anndata", "scanpy")
_FOREIGN: dict[str, tuple[Module, ast.FunctionDef] | None] = {}


def _foreign(q: str) -> tuple[Module, ast.FunctionDef] | None:
    """A function of another module of the code (not a library's), read from its source: (its module, its
    definition); None if it is not one, or its source is not found."""
    if q.split(".")[0] in LIBRARIES or "." not in q:
        return None
    if q in _FOREIGN:
        return _FOREIGN[q]
    import importlib.util
    module_name, _, name = q.rpartition(".")
    out = None
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError, AttributeError):
        spec = None
    if spec is not None and spec.origin and spec.origin.endswith(".py"):
        m = Module.read(spec.origin, module_name)
        fn = m.functions.get(name)
        out = (m, fn) if fn is not None else None
    _FOREIGN[q] = out
    return out


def _docstring(body: list[ast.stmt]) -> bool:
    return bool(body) and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
        and isinstance(body[0].value.value, str)


def _output_only(call: ast.Call) -> bool:
    """print(...), logging.info(...), warnings.warn(...): calls that change no value."""
    root = call.func
    while isinstance(root, ast.Attribute):
        root = root.value
    return isinstance(root, ast.Name) and root.id in OUTPUT_ONLY


def _holds(test: ast.expr, every: bool = False) -> list[ast.expr]:
    """An assert's condition, one per part that must hold: (x > 0).all() and np.all(x > 0) as x > 0, which an
    indexed inequality means; `a and b` as a and b; in an all(), `a & b` as a and b too."""
    if isinstance(test, ast.Call) and not test.keywords:
        if isinstance(test.func, ast.Attribute) and test.func.attr == "all" and not test.args:
            return _holds(test.func.value, True)
        if _dotted(test.func) in ("np.all", "numpy.all") and len(test.args) == 1:
            return _holds(test.args[0], True)
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        return [c for v in test.values for c in _holds(v, every)]
    if every and isinstance(test, ast.BinOp) and isinstance(test.op, ast.BitAnd):
        return [*_holds(test.left, True), *_holds(test.right, True)]
    return [test]


def _names(t: ast.expr) -> tuple[str, ...] | None:
    if isinstance(t, ast.Name):
        return (t.id,)
    if isinstance(t, (ast.Tuple, ast.List)):
        out: list[str] = []
        for e in t.elts:
            n = _names(e)
            if n is None:
                return None
            out += n
        return tuple(out)
    return None


def _target_names(t: ast.expr) -> str | tuple | None:
    """A comprehension's target, its structure kept: x, (i, x), (fold, (m, n))."""
    if isinstance(t, ast.Name):
        return t.id
    if isinstance(t, (ast.Tuple, ast.List)):
        parts = [_target_names(e) for e in t.elts]
        return None if any(p is None for p in parts) else tuple(parts)
    return None


def _dotted(e: ast.expr) -> str | None:
    if isinstance(e, ast.Name):
        return e.id
    if isinstance(e, ast.Attribute):
        head = _dotted(e.value)
        return None if head is None else f"{head}.{e.attr}"
    return None


def _array_type(*xs: Expr) -> str:
    """The array type of an elementwise result: that of its array operands (numpy's, torch's)."""
    for x in xs:
        if x.type in ("numpy.ndarray", "torch.Tensor", "jax.Array"):
            return x.type
    return ""


def _library_of(*xs: Expr) -> str:
    t = _array_type(*xs)
    return {"torch.Tensor": "torch", "jax.Array": "jax.numpy"}.get(t, "numpy")


def lower_source(source: str, module: Module | None = None, **kw: object) -> Program:
    """A function's source (or a block of statements) as a Program."""
    tree = ast.parse(textwrap.dedent(source))
    low = Lowerer(module or Module.of(tree), **kw)  # type: ignore[arg-type]
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef):
        return low.program(tree.body[0])
    return Program(None, [], low.block(tree.body))


def same(a: Expr, b: Expr) -> bool:
    return a == b


__all__ = ["Lowerer", "Module", "Unsupported", "lower_source", "NEW", "SCALAR", "replace"]
