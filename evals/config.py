"""
config.py

Configuration for RAGAS evaluation.
"""

from __future__ import annotations

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from core.llm import (
    evaluation_llm,
    get_evaluation_embeddings,
)


# ==========================================================
# Wrapped Evaluation Models
# ==========================================================

RAGAS_LLM = LangchainLLMWrapper(
    evaluation_llm
)


evaluation_embeddings = (
    get_evaluation_embeddings()
)


RAGAS_EMBEDDINGS = (
    LangchainEmbeddingsWrapper(
        evaluation_embeddings
    )
)


# ==========================================================
# RAGAS Metrics
# ==========================================================

RAGAS_METRICS = [

    Faithfulness(
        llm=RAGAS_LLM,
    ),

    AnswerRelevancy(
        llm=RAGAS_LLM,
        embeddings=RAGAS_EMBEDDINGS,
    ),

    ContextPrecision(
        llm=RAGAS_LLM,
    ),

    ContextRecall(
        llm=RAGAS_LLM,
    ),

]