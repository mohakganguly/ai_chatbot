"""
Central configuration for the chatbot project.
"""

import os

from dotenv import load_dotenv

load_dotenv()


#############################################
# LLM Configuration
#############################################

CHAT_MODEL = "llama-3.1-8b-instant"
CHAT_TEMPERATURE=0.2


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
MISTRAL_API_KEY=os.getenv("MISTRAL_API_KEY")
#############################################
# Embedding Configuration
#############################################

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

EMBEDDING_DEVICE = "cpu"

NORMALIZE_EMBEDDINGS = True


#############################################
# Chunking Configuration
#############################################
CHUNKING_STRATEGY = "recursive"
CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200


#############################################
# Vector Database
#############################################

QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "enterprise_rag"

QDRANT_HOST = "localhost"

QDRANT_PORT = 6333

VECTOR_SIZE = 384

DISTANCE_METRIC = "COSINE"
#############################################
# PostgreSQL
#############################################

DATABASE_URL = os.getenv("DATABASE_URL")


#############################################
# Retrieval
#############################################

RETRIEVAL_TOP_K = 20




#############################################
# Reranking
#############################################
RERANKER_MODEL = "BAAI/bge-reranker-base" #cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_TOP_K = 5
MIN_RELEVANCE_SCORE = 0.45

# ==========================================================
# Context Filter
# ==========================================================

MIN_RERANK_SCORE = 3.0

MAX_CONTEXT_DOCUMENTS = 5

MAX_CONTEXT_CHARACTERS = 12000
#############################################
# Miscellaneous
#############################################

LOG_LEVEL = "INFO"

MAX_CHAT_HISTORY = 20