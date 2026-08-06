"""
metrics.py

Central metrics recorder.
"""

from __future__ import annotations

from collections import defaultdict

from utils.logger import get_logger

logger = get_logger(__name__)


class MetricsRecorder:

    def __init__(self):

        self.counters = defaultdict(int)

    def increment(
        self,
        name: str,
        value: int = 1,
    ):

        self.counters[name] += value

    def record_latency(
        self,
        name: str,
        seconds: float,
    ):

        logger.info(
            "[METRIC] %s latency %.3fs",
            name,
            seconds,
        )

    def record_tokens(

        self,

        prompt_tokens: int,

        completion_tokens: int,

        total_tokens: int,
    ):

        logger.info(

            "[TOKENS] prompt=%d completion=%d total=%d",

            prompt_tokens,

            completion_tokens,

            total_tokens,
        )


metrics = MetricsRecorder()