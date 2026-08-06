"""
Enterprise tracing utilities.

Currently backed by LangSmith.

Later this same interface will also emit:
- Prometheus metrics
- OpenTelemetry spans
- Cost metrics
"""

from __future__ import annotations

import time
from contextlib import contextmanager

from langsmith import trace


@contextmanager
def trace_node(
    name: str,
    **metadata,
):
    """
    Create a traced span for a graph node.
    """

    start = time.perf_counter()

    with trace(
        name=name,
        metadata=metadata,
    ):

        try:
            yield

        finally:

            elapsed = time.perf_counter() - start

            print(
                f"[TRACE] {name} "
                f"{elapsed:.3f}s"
            )