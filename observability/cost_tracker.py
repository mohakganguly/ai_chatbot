"""
Temporary cost tracker.

Later this will use
real pricing.
"""

from __future__ import annotations


class CostTracker:

    def estimate(

        self,

        model: str,

        prompt_tokens: int,

        completion_tokens: int,

    ) -> float:

        return 0.0


cost_tracker = CostTracker()