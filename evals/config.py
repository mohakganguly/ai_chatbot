"""
config.py

Configuration for RAGAS evaluation.
"""

from __future__ import annotations

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
    ContextRecall,
)

from core.llm import (
    evaluation_llm,
    get_evaluation_embeddings,
)

# ----------------------------------------------------------
# Wrapped models
# ----------------------------------------------------------

RAGAS_LLM = LangchainLLMWrapper(
    evaluation_llm
)
evaluation_embeddings = get_evaluation_embeddings()
RAGAS_EMBEDDINGS = LangchainEmbeddingsWrapper(
    evaluation_embeddings
)

# ----------------------------------------------------------
# Metrics
# ----------------------------------------------------------

RAGAS_METRICS = [

    Faithfulness(),

    ResponseRelevancy(),

    ContextPrecision(),

    ContextRecall(),

]