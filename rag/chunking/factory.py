"""
Factory for selecting the chunking strategy.
"""

from __future__ import annotations

from config import CHUNKING_STRATEGY

from rag.chunking.chunker import HybridChunker

from utils.logger import get_logger

logger = get_logger(__name__)


class ChunkingFactory:

    @staticmethod
    def create():

        strategy = CHUNKING_STRATEGY.lower()

        logger.info(
            "Chunking Strategy : %s",
            strategy
        )

        if strategy == "hybrid":

            return HybridChunker()

        raise ValueError(
            f"Unknown chunking strategy: {strategy}"
        )