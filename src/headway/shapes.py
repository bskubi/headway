"""Array shapes the code states in its type hints, in jaxtyping's form: `Float[np.ndarray, "cells genes"]` is an
array whose axes are named cells and genes. Read from the syntax tree (jaxtyping need not be installed), with the
module's imports saying which names are jaxtyping's.

Only named axes are read: an annotation with an axis of fixed size (3), any number of axes (`...`, `*batch`), a
broadcastable one (`#n`), an anonymous one (`_`) or an expression (`n+1`) gives no shape at all, rather than a part.
"""
from __future__ import annotations

import ast
import keyword
from collections.abc import Callable

# jaxtyping's array annotations: the dtype families (Float, Float32, ...) and Shaped (any dtype)
DTYPES = {"Shaped", "Bool", "Num", "Real", "Inexact", "Integer", "Int", "UInt", "Float", "Complex", "Key",
          *(f"{base}{bits}" for base in ("Int", "UInt") for bits in (8, 16, 32, 64)),
          *(f"Float{bits}" for bits in (16, 32, 64)), "BFloat16", *(f"Complex{bits}" for bits in (64, 128))}


def dims(ann: ast.expr | None, full: Callable[[str], str], alias: Callable[[str], ast.expr | None]) -> list[str] | None:
    """The named axes an annotation gives an array, or None. full(dotted name) -> what it stands for by the
    module's imports ("Float" -> "jaxtyping.Float"); alias(name) -> the annotation a type alias stands for."""
    ann = _unwrapped(ann, alias)
    if not isinstance(ann, ast.Subscript) or not isinstance(ann.slice, ast.Tuple) or len(ann.slice.elts) != 2:
        return None
    name = _dotted(ann.value)
    if name is None:
        return None
    head, _, last = full(name).rpartition(".")
    if head.split(".")[0] != "jaxtyping" or last not in DTYPES:
        return None
    shape = ann.slice.elts[1]
    if not isinstance(shape, ast.Constant) or not isinstance(shape.value, str):
        return None
    names = shape.value.split()
    return names if all(n.isidentifier() and n != "_" and not keyword.iskeyword(n) for n in names) else None


def items(ann: ast.expr | None, alias: Callable[[str], ast.expr | None]) -> list[ast.expr] | None:
    """The annotations of a tuple's items (tuple[A, B]), or None if the annotation is not a tuple of fixed size."""
    ann = _unwrapped(ann, alias)
    if isinstance(ann, ast.Subscript) and _dotted(ann.value) in ("tuple", "typing.Tuple", "Tuple") \
            and isinstance(ann.slice, ast.Tuple) and not any(isinstance(e, ast.Constant) and e.value is Ellipsis
                                                             for e in ann.slice.elts):
        return list(ann.slice.elts)
    return None


def _unwrapped(ann: ast.expr | None, alias: Callable[[str], ast.expr | None], depth: int = 0) -> ast.expr | None:
    """An annotation as it is meant: a string parsed, an alias expanded, `X | None` as X."""
    if ann is None or depth > 10:
        return ann
    if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
        try:
            return _unwrapped(ast.parse(ann.value, mode="eval").body, alias, depth + 1)
        except SyntaxError:
            return None
    if isinstance(ann, ast.Name) and (target := alias(ann.id)) is not None:
        return _unwrapped(target, alias, depth + 1)
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        sides = [s for s in (ann.left, ann.right) if not (isinstance(s, ast.Constant) and s.value is None)]
        return _unwrapped(sides[0], alias, depth + 1) if len(sides) == 1 else None
    return ann


def _dotted(e: ast.expr) -> str | None:
    if isinstance(e, ast.Name):
        return e.id
    if isinstance(e, ast.Attribute):
        head = _dotted(e.value)
        return None if head is None else f"{head}.{e.attr}"
    return None
