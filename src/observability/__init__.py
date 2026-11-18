"""Structured JSON-Lines observability with hardware KV-Cache token tracking."""

from .tracer import SpanEvent, StructuredTracer, new_span_id, new_trace_id

__all__ = ["SpanEvent", "StructuredTracer", "new_span_id", "new_trace_id"]
