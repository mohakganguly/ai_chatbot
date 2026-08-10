"""
llm.py

Centralized initialization of all LLMs used in the project.
"""

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from rag.embedding.embedding_model import get_embedding_model
from config import *
from observability.callbacks import EnterpriseCallback
callback = EnterpriseCallback()

planner_llm = ChatGroq(
    model=CHAT_MODEL,
    temperature=0,
    callbacks=[callback],

)

answer_llm = ChatGroq(
    model=CHAT_MODEL,
    temperature=CHAT_TEMPERATURE,
    callbacks=[callback],
)


router_llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0,
    callbacks=[callback],
)

observation_llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0,
    callbacks=[callback],
)


rewriter_llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0,
)

title_llm = ChatMistralAI(
    model="ministral-3b-latest",
    temperature=0,
    callbacks=[callback],
)
evaluation_llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0,
    callbacks=[callback],
)
def get_evaluation_embeddings():
    """
    Get the embedding model for evaluation.
    """
    return get_embedding_model()

from graph.router_schema import RouterOutput

structured_router_llm = router_llm.with_structured_output(
    RouterOutput
)

from graph.schemas.rewrite_decision_schema import RewriteDecision

structured_rewrite_llm = rewriter_llm.with_structured_output(
    RewriteDecision
)