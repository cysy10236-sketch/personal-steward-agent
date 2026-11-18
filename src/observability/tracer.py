"""Thread-safe JSON-Lines tracer recording spans, token usage, cached tokens, and quantiles."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Any, Iterator
import uuid

import numpy as np
from pydantic import BaseModel, Field


def new_trace_id() -> str:
    return f"tr_{uuid.uuid4().hex[:16]}"


def new_span_id() -> str:
    return f"sp_{uuid.uuid4().hex[:12]}"


class SpanEvent(BaseModel):
    """Single JSON-Lines trace span with input/output/cached token telemetry."""

    timestamp: str
    trace_id: str
    span_id: str
    component: str
    operation: str
    model_id: str = ""
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    status_code: int = 200
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class _MutableSpan:
    def __init__(self) -> None:
        self.model_id: str = ""
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cached_input_tokens: int = 0
        self.status_code: int = 200
        self.metadata: dict[str, Any] = {}


class StructuredTracer:
    """Appends structured JSONL spans to disk and computes latency quantiles."""

    def __init__(self, log_path: Path | str = Path("logs/trace_events.jsonl")) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._events: list[SpanEvent] = []

    @contextmanager
    def span(self, trace_id: str, component: str, operation: str) -> Iterator[_MutableSpan]:
        span_id = new_span_id()
        bag = _MutableSpan()
        t0 = time.perf_counter()
        err_str: str | None = None
        try:
            yield bag
        except Exception as exc:
            bag.status_code = 500
            err_str = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            ev = SpanEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                trace_id=trace_id,
                span_id=span_id,
                component=component,
                operation=operation,
                model_id=bag.model_id,
                input_tokens=bag.input_tokens,
                output_tokens=bag.output_tokens,
                cached_input_tokens=bag.cached_input_tokens,
                latency_ms=elapsed_ms,
                status_code=bag.status_code,
                error=err_str,
                metadata=bag.metadata,
            )
            line = json.dumps(ev.model_dump(mode="json"), ensure_ascii=False)
            with self._lock:
                self._events.append(ev)
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")

    @property
    def events(self) -> list[SpanEvent]:
        with self._lock:
            return list(self._events)

    @staticmethod
    def latency_summary(values_ms: list[float]) -> dict[str, float]:
        if not values_ms:
            return {"mean_ms": 0.0, "std_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
        arr = np.asarray(values_ms, dtype=np.float64)
        return {
            "mean_ms": round(float(np.mean(arr)), 2),
            "std_ms": round(float(np.std(arr)), 2),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
        }
