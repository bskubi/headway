"""headway: see how a method works, and how much each change to it gains. A spec module declares the method's
steps (equations, inputs, outputs, references into the code); headway checks it against the code and draws it as
an interactive computation graph, compares versions of it, and shows their benchmark results.
`headway guide` (or GUIDE.md in this package) describes it all."""
from headway.api import attach, build, load, read_results
from headway.bench import Suite
from headway.check import Report, check
from headway.spec import Group, Headway, Index, Op, Symbol
from headway.trace import Trace, trace

__all__ = ["Group", "Headway", "Index", "Op", "Report", "Suite", "Symbol", "Trace", "attach", "build", "check", "load",
           "read_results", "trace"]
