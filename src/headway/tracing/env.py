"""The environment a run had: what can change what code computes without changing the code
(survey/tracing-design.md §5). Read at the start of a run and again at its end."""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import sys
from typing import Any

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES", "JAX_ENABLE_X64", "NUMBA_DISABLE_JIT")


def environment() -> dict[str, Any]:
    """The interpreter, the platform, every installed distribution's version, the thread and seed settings in the
    environment, and numpy's state (its error handling and the global random generator's)."""
    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
        "variables": {k: os.environ[k] for k in THREAD_VARIABLES if k in os.environ},
        "distributions": dict(sorted({(d.metadata["Name"] or "?").lower(): d.version
                                      for d in importlib.metadata.distributions()}.items())),
    }
    out.update(numpy_state())
    return out


def numpy_state() -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError:
        return {}
    state = np.random.get_state()
    return {"numpy_errors": np.geterr(),
            "numpy_random": hashlib.sha1(state[1].tobytes() + str(state[2:]).encode()).hexdigest()[:16]}


def modules() -> list[str]:
    """The top-level packages imported so far (which of the optional ones a run used)."""
    return sorted({m.split(".")[0] for m in sys.modules if not m.startswith("_")})


def changes(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """What a run changed in the global state it does not own (numpy's random generator reseeded, its error
    handling set): the before and after of each."""
    out = {}
    for k in ("numpy_errors", "numpy_random"):
        if before.get(k) != after.get(k):
            out[k] = {"before": before.get(k), "after": after.get(k)}
    return out
