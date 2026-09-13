<div align="center">

# Enterprise AI Assistant

### Production-Grade Agentic AI Assistant with Hybrid RAG, Tool Calling, Reranking, Citations, Observability & DeepEval

**LangGraph · LangChain · Qdrant · PostgreSQL · Groq · Mistral AI · Streamlit · LangSmith · DeepEval**

</div>

---

# 📌 Table of Contents

- [Overview](#-overview)
- [Key Capabilities](#-key-capabilities)
- [System Architecture](#-system-architecture)
- [End-to-End Runtime Workflow](#-end-to-end-runtime-workflow)
- [Agent Architecture](#-agent-architecture)
- [Router Workflow](#-router-workflow)
- [Planner Workflow](#-planner-workflow)
- [Executor Workflow](#-executor-workflow)
- [Observation Workflow](#-observation--agent-loop)
- [Document Ingestion](#-document-ingestion-workflow)
- [Document Parsing](#-document-parsing-workflow)
- [Chunking](#-chunking-workflow)
- [Embeddings](#-embedding-workflow)
- [Qdrant Storage](#-qdrant-storage-workflow)
- [Complete RAG Pipeline](#-complete-rag-workflow)
- [Conversation History](#-conversation-history-workflow)
- [Query Rewriting](#-query-rewrite-decision-workflow)
- [Dense Retrieval](#-dense-retrieval-workflow)
- [BM25 Retrieval](#-bm25-retrieval-workflow)
- [Hybrid Retrieval](#-hybrid-retrieval-workflow)
- [RRF Fusion](#-rrf-fusion-workflow)
- [Cross Encoder Reranking](#-cross-encoder-reranking-workflow)
- [Context Filtering](#-context-filtering-workflow)
- [RetrievalResult](#-retrievalresult-workflow)
- [Citation Builder](#-citation-builder-workflow)
- [PromptBuilder](#-promptbuilder-workflow)
- [Generation](#-generation-workflow)
- [Persistence](#-persistence-workflow)
- [Observability](#-langsmith-observability-workflow)
- [Evaluation Architecture](#-complete-deepeval-evaluation-architecture)
- [Evaluation Dataset](#-evaluation-dataset-workflow)
- [EvaluationSample](#-evaluationsample-workflow)
- [Evaluation Collector](#-evaluation-collector-workflow)
- [Production Retrieval Evaluation](#-production-retrieval-evaluation-workflow)
- [Production Generation Evaluation](#-production-generation-evaluation-workflow)
- [DeepEval RAG Triad](#-deepeval-rag-triad-workflow)
- [Contextual Relevancy](#-contextual-relevancy-workflow)
- [Contextual Recall](#-contextual-recall-workflow)
- [Contextual Precision](#-contextual-precision-workflow)
- [Faithfulness](#-faithfulness-workflow)
- [Answer Relevancy](#-answer-relevancy-workflow)
- [Answer Correctness](#-answer-correctness-workflow)
- [Judge Model](#-deepeval-judge-model-workflow)
- [Rate Limit & Retry](#-rate-limit--retry-workflow)
- [DeepEval Runner](#-deepeval-runner-workflow)
- [Evaluation Report](#-evaluation-report-workflow)
- [Failure Diagnosis](#-retrieval-vs-generation-failure-diagnosis)
- [Complete Production + Evaluation Flow](#-complete-production--evaluation-workflow)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the Application](#-running-the-application)
- [Running Evaluation](#-running-evaluation)
- [Evaluation Metrics](#-evaluation-metrics)
- [Engineering Principles](#-engineering-principles)
- [Roadmap](#-roadmap)

---

# 📖 Overview

The **Enterprise AI Assistant** is a production-oriented agentic AI system designed to answer user queries using enterprise knowledge, execute external tools when required, maintain conversational context, and generate grounded responses with citations.

The system combines:

- Stateful agent orchestration using **LangGraph**
- LLM and tool abstractions using **LangChain**
- Dense semantic retrieval
- BM25 lexical retrieval
- Hybrid retrieval
- Reciprocal Rank Fusion
- Cross-encoder reranking
- Context filtering
- Conversation-aware query rewriting
- Thread-scoped document retrieval
- Citation generation
- Persistent conversations
- PostgreSQL persistence
- Qdrant vector storage
- LangSmith tracing
- DeepEval-based evaluation

The central design principle is:

```text
User Query
    ↓
Agentic Reasoning
    ↓
Retrieval / Tool Execution
    ↓
Observation
    ↓
Grounded Generation
    ↓
Citations
    ↓
Final Answer