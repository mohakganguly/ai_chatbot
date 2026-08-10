"""
reranker.py

Fast enterprise reranker.

Pipeline:

Hybrid Retrieval
        ↓
RRF Top-K candidates
        ↓
MiniLM Cross-Encoder
        ↓
Top-N relevant documents
"""

from __future__ import annotations
import threading
import time
from typing import List, Optional

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import RERANK_TOP_K, RERANKER_MODEL

from utils.logger import get_logger


logger = get_logger(__name__)


class Reranker:
    """
    Fast cross-encoder reranker.

    The model is initialized lazily so importing/creating the
    Reranker does not immediately incur model loading cost.
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        max_length: int = 256,
        batch_size: int = 8,
    ):
        self.model: Optional[CrossEncoder] = None
        self.model_lock = threading.Lock()
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size


        logger.info(
            "Reranker service created. "
            "Model will be initialized lazily: %s",
            self.model_name,
        )

    # =========================================================
    # Lazy Model Initialization
    # =========================================================

    def _get_model(self) -> CrossEncoder:

        if self.model is not None:
            return self.model

        with self.model_lock:

            if self.model is not None:
                return self.model

            logger.info(
                "Initializing reranker model '%s' on CPU.",
                self.model_name,
            )

            start = time.perf_counter()

            try:

                self.model = CrossEncoder(
                    self.model_name,
                    max_length=self.max_length,
                    device="cpu",
                )

                elapsed = time.perf_counter() - start

                logger.info(
                    "Reranker model initialized in %.3f sec.",
                    elapsed,
                )

                return self.model

            except Exception:

                logger.exception(
                    "Failed to initialize reranker model."
                )

                raise

    # =========================================================
    # Rerank
    # =========================================================

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = RERANK_TOP_K,
    ) -> List[Document]:
        """
        Rerank retrieved documents using a cross-encoder.

        Parameters
        ----------
        query:
            User query.

        documents:
            Candidate documents returned by hybrid retrieval.

        top_k:
            Number of documents returned after reranking.

        Returns
        -------
        List[Document]
            Reranked documents.
        """

        if not documents:

            logger.info(
                "No documents supplied for reranking."
            )

            return []

        # -----------------------------------------------------
        # Avoid unnecessary inference
        # -----------------------------------------------------

        if len(documents) <= 1:

            logger.info(
                "Only %d document supplied. "
                "Skipping reranking.",
                len(documents),
            )

            return documents[:top_k]

        logger.info(
            "Reranking %d candidate document(s).",
            len(documents),
        )

        start = time.perf_counter()

        try:

            model = self._get_model()

            # -------------------------------------------------
            # Create query-document pairs
            # -------------------------------------------------

            pairs = [
                (
                    query,
                    document.page_content,
                )
                for document in documents
            ]

            # -------------------------------------------------
            # Cross-encoder inference
            # -------------------------------------------------

            inference_start = time.perf_counter()

            scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )

            inference_time = (
                time.perf_counter()
                - inference_start
            )

            # -------------------------------------------------
            # Rank documents
            # -------------------------------------------------

            ranked = sorted(
                zip(documents, scores),
                key=lambda item: float(item[1]),
                reverse=True,
            )

            reranked_documents = []

            for document, score in ranked[:top_k]:

                metadata = document.metadata.copy()

                metadata["reranker_score"] = float(
                    score
                )

                document.metadata = metadata

                reranked_documents.append(
                    document
                )

            total_time = (
                time.perf_counter()
                - start
            )

            logger.info(
                "Reranking inference completed in %.3f sec.",
                inference_time,
            )

            logger.info(
                "Reranking completed in %.3f sec.",
                total_time,
            )

            logger.info(
                "Selected top-%d document(s).",
                len(reranked_documents),
            )

            return reranked_documents

        except Exception:

            logger.exception(
                "Reranking failed. "
                "Returning original retrieval results."
            )

            return documents[:top_k]

    # =========================================================
    # Warm-up
    # =========================================================

    def warm_up(self) -> None:
        """
        Initialize the model and perform one tiny inference.

        Useful for application startup/background warming so
        the first real user query does not pay the model
        initialization cost.
        """

        logger.info(
            "Warming up reranker."
        )

        start = time.perf_counter()

        try:

            model = self._get_model()

            model.predict(
                [
                    (
                        "warmup query",
                        "warmup document",
                    )
                ],
                batch_size=1,
                show_progress_bar=False,
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            logger.info(
                "Reranker warm-up completed in %.3f sec.",
                elapsed,
            )

        except Exception:

            logger.exception(
                "Reranker warm-up failed."
            )

            raise


# =============================================================
# Singleton
# =============================================================

_reranker = Reranker()


def get_reranker() -> Reranker:

    return _reranker


# =============================================================
# Testing
# =============================================================

if __name__ == "__main__":

    from rag.retrieval.retriever import Retriever

    logger.info("=" * 80)
    logger.info("Starting reranker test.")
    logger.info("=" * 80)

    query = "What is a Zombie Process?"

    # ---------------------------------------------------------
    # Retrieve hybrid candidates
    # ---------------------------------------------------------

    retriever = Retriever()

    documents = retriever.retrieve(
        query=query,
        thread_id=None,
        top_k=8,
    )

    logger.info(
        "Hybrid retrieval returned %d candidate(s).",
        len(documents),
    )

    # ---------------------------------------------------------
    # Rerank
    # ---------------------------------------------------------

    reranker = get_reranker()

    start = time.perf_counter()

    results = reranker.rerank(
        query=query,
        documents=documents,
        top_k=5,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    logger.info(
        "End-to-end reranking test completed in %.3f sec.",
        elapsed,
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    print("=" * 80)

    for index, document in enumerate(
        results,
        start=1,
    ):

        print(f"\nResult {index}")

        print(
            "Reranker score:",
            document.metadata.get(
                "reranker_score"
            ),
        )

        print(
            document.page_content[:500]
        )

        print("-" * 80)

    logger.info(
        "Reranker test completed successfully."
    )