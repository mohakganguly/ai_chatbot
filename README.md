# Enterprise AI Assistant

> **A production-inspired AI assistant built using LangGraph,
> Retrieval-Augmented Generation (RAG), Multi-Agent Tool Orchestration,
> LangSmith Observability, and RAGAS Evaluation.**

------------------------------------------------------------------------

## Overview

Enterprise AI Assistant is a modular conversational AI platform designed
around modern enterprise AI architecture.

Instead of relying on a single LLM call, the assistant dynamically
routes requests to either a general conversational workflow or an
iterative agent workflow capable of planning, tool execution, retrieval,
observation, and grounded answer generation.

Core capabilities include:

-   Intelligent query routing
-   Enterprise Retrieval-Augmented Generation (RAG)
-   Multi-agent orchestration
-   Tool execution
-   Context-aware retrieval
-   Automatic citations
-   Persistent conversations
-   LangSmith observability
-   RAGAS evaluation

------------------------------------------------------------------------

# Features

## Intelligent Query Routing

-   General conversation routing
-   Agent workflow routing
-   Conversation-aware decision making

## Enterprise RAG Pipeline

-   Query rewrite decision
-   Query rewriting
-   Semantic retrieval
-   Cross-Encoder reranking
-   Context filtering
-   Citation generation

## Multi-Agent Workflow

-   Planner Agent
-   Executor Agent
-   Observation Agent
-   Final Answer Generator

## Modular Tool Framework

Supports enterprise tools such as:

-   Document Retrieval
-   Web Search
-   Calculator
-   Weather
-   Custom plugins

## Observability

Integrated with **LangSmith** for:

-   Node tracing
-   Prompt tracing
-   Tool execution
-   Latency analysis
-   Token usage
-   Workflow visualization

## Evaluation

Integrated with **RAGAS** to measure:

-   Faithfulness
-   Answer Relevancy
-   Context Precision
-   Context Recall
-   Response Latency

------------------------------------------------------------------------

# LangGraph Workflow

``` text
                    User
                      │
                      ▼
                Streamlit UI
                      │
                      ▼
                 Router Node
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
   General Chat             Planner Agent
                                  │
                                  ▼
                             Executor
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
             Document Retrieval           External Tools
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                          Observation Agent
                                  │
                          Continue Planning?
                           │              │
                          Yes            No
                           │              │
                           ▼              ▼
                       Planner      Final Answer
```

------------------------------------------------------------------------

# Enterprise RAG Pipeline

``` text
User Query
    │
    ▼
Conversation History
    │
    ▼
Rewrite Decision
    │
 ┌──┴─────────┐
 │            │
 ▼            ▼
Original   Query Rewrite
 Query          │
 └──────┬───────┘
        ▼
 Semantic Retrieval
        │
        ▼
 Cross Encoder Reranker
        │
        ▼
 Context Filtering
        │
        ▼
 Citation Builder
        │
        ▼
 Context to LLM
        │
        ▼
 Final Response
```

------------------------------------------------------------------------

# Document Ingestion Pipeline

``` text
Documents
    │
    ▼
Loaders
    │
    ▼
Text Extraction
    │
    ▼
Recursive Text Splitter
    │
    ▼
BGE Embeddings
    │
    ▼
Qdrant Vector Database
```

------------------------------------------------------------------------

# Tech Stack

### Backend

-   Python
-   FastAPI
-   LangGraph
-   LangChain

### Models

-   Groq
-   Mistral AI

### Retrieval

-   Qdrant
-   BAAI/bge-small-en-v1.5
-   BAAI/bge-reranker-base

### Database

-   PostgreSQL
-   SQLAlchemy

### Frontend

-   Streamlit

### Observability

-   LangSmith

### Evaluation

-   RAGAS

------------------------------------------------------------------------

# Project Structure

``` text
.
├── agents/
├── graph/
├── rag/
├── tools/
├── observability/
├── evals/
├── db/
├── frontend/
├── utils/
├── config.py
└── app.py
```

------------------------------------------------------------------------

# Installation

``` bash
git clone <repository-url>
cd enterprise-ai-assistant

python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file:

``` env
GROQ_API_KEY=
MISTRAL_API_KEY=
TAVILY_API_KEY=
LANGSMITH_API_KEY=

LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=Enterprise AI Assistant
```

Run:

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

# Evaluation

``` bash
python -m evals.run --suite rag
```

Reports include:

-   Faithfulness
-   Answer Relevancy
-   Context Precision
-   Context Recall
-   Latency

------------------------------------------------------------------------

# Future Roadmap

-   Guardrails
-   Redis Caching
-   Celery Workers
-   Hybrid Search
-   Docker Deployment
-   CI/CD
-   Authentication & RBAC
-   Multi-format document ingestion

------------------------------------------------------------------------

# License

MIT License
