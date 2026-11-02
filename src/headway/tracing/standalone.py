"""The tracer run in a codebase's own interpreter (which may be an older Python than headway's, without headway
installed): it needs only the standard library, numpy and pytest. headway record --python PATH runs it.

    PYTHON standalone.py -p PACKAGE [-p ...] -o DIR [--first N] [--every N] [--limit N] [--parallel] -- PYTEST ARGS
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracer import Tracer  # type: ignore[import-not-found]  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-p", "--package", dest="packages", action="append", required=True)
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--first", type=int, default=20)
    p.add_argument("--every", type=int, default=50)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--parallel", action="store_true")
    p.add_argument("pytest_args", nargs="*")
    a = p.parse_args()
    sys.path.remove(os.path.dirname(os.path.abspath(__file__)))     # the code's own modules first
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    import pytest
    t = Tracer(a.packages, first=a.first, every=a.every, limit=a.limit, sequential=not a.parallel)
    with t:
        code = pytest.main([x for x in a.pytest_args if x != "--"])
    t.write(a.out)
    print(f"tests: exit {int(code)}; {len(t.calls)} calls recorded; trace in {a.out}", file=sys.stderr)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
