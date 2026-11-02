"""Tracing: what code did when it ran, recorded at the boundaries of the functions traced, and what that says
about the code (its paths, its arrays' index sets). See survey/tracing-design.md."""
from headway.tracing.analyze import describe_paths, describe_sets, index_sets, paths, signatures
from headway.tracing.record import Call, Trace, Value
from headway.tracing.tracer import Tracer, tracing

__all__ = ["Call", "Trace", "Tracer", "Value", "describe_paths", "describe_sets", "index_sets", "paths",
           "signatures", "tracing"]
