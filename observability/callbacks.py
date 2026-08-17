"""
Enterprise callback handler.
"""

from __future__ import annotations

import time

from langchain_core.callbacks import BaseCallbackHandler

from observability.metrics import metrics

from observability.cost_tracker import cost_tracker


class EnterpriseCallback(
    BaseCallbackHandler,
):

    def __init__(self):

        self.start = None

    def on_llm_start(

        self,

        serialized,

        prompts,

        **kwargs,

    ):

        self.start = time.perf_counter()

    def on_llm_end(
        self,
        response,
        **kwargs,
    ):

        if self.start is not None:

            latency = (
                time.perf_counter()
                - self.start
            )

            metrics.record_latency(
                "llm",
                latency,
            )

        usage = getattr(
            response,
            "llm_output",
            None,
        ) or {}

        token_usage = (
            usage.get(
                "token_usage",
                {},
            )
            or {}
        )

        metrics.record_tokens(

            token_usage.get(
                "prompt_tokens",
                0,
            ) or 0,

            token_usage.get(
                "completion_tokens",
                0,
            ) or 0,

            token_usage.get(
                "total_tokens",
                0,
            ) or 0,
        )

        self.start = None