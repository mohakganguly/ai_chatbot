<div align="center">

# 🤖 Enterprise AI Assistant Chatbot

### AI Assistant with Hybrid RAG · Tool Calling · Reranking · Citations · Observability · DeepEval

<p>
  <img src="https://img.shields.io/badge/LangGraph-Orchestration-1C3C3C?style=for-the-badge&logo=graphql&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white" />
  <img src="https://img.shields.io/badge/Qdrant-VectorDB-DC244C?style=for-the-badge&logo=databricks&logoColor=white" />
  <img src="https://img.shields.io/badge/PostgreSQL-Storage-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" />
</p>
<p>
  <img src="https://img.shields.io/badge/Groq-Inference-F55036?style=for-the-badge&logo=lightning&logoColor=white" />
  <img src="https://img.shields.io/badge/Mistral_AI-LLM-FF7000?style=for-the-badge&logo=mistralai&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/LangSmith-Observability-1C3C3C?style=for-the-badge" />
  <img src="https://img.shields.io/badge/DeepEval-Testing-6E56CF?style=for-the-badge" />
</p>

<p><i>A production-grade agentic RAG system — plan, act, observe, cite, and evaluate.</i></p>

</div>

---

## 📌 Table of Contents

<table>
<tr>
<td valign="top" width="33%">

**Core**
- [Overview](#-overview)
- [Key Capabilities](#-key-capabilities)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)

**Agent**
- [Agent Architecture](#-agent-architecture)
- [Main Agentic Chat Loop](#-main-agentic-chat-loop)
- [Router / Planner / Executor](#-router-planner--executor)
- [Observation & Agent Loop](#-observation--agent-loop)

</td>
<td valign="top" width="33%">

**Ingestion & RAG**
- [Document Ingestion](#-document-ingestion-pipeline)
- [Chunking & Embedding](#-chunking--embedding)
- [Qdrant Storage](#-qdrant-storage-workflow)
- [Complete RAG Pipeline](#-complete-rag-pipeline)
- [Hybrid Retrieval (Dense + BM25 + RRF)](#-hybrid-retrieval-workflow)
- [Cross-Encoder Reranking](#-cross-encoder-reranking)
- [Citation Builder](#-citation-builder-workflow)

</td>
<td valign="top" width="33%">

**Evaluation**
- [Evaluation Architecture](#-deepeval-evaluation-architecture)
- [RAG Triad Metrics](#-deepeval-rag-triad)
- [Failure Diagnosis](#-retrieval-vs-generation-failure-diagnosis)

**Operate**
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the App](#-running-the-application)
- [Running Evaluation](#-running-evaluation)
- [Roadmap](#-roadmap)

</td>
</tr>
</table>

---

## 📖 Overview

The **Enterprise AI Assistant** is a production-oriented agentic AI system designed to answer user queries using enterprise knowledge, execute external tools when required, maintain conversational context, and generate grounded responses with citations.

```text
User Query → Agentic Reasoning → Retrieval / Tool Execution → Observation → Grounded Generation → Citations → Final Answer
```

The system combines stateful agent orchestration (**LangGraph**), LLM/tool abstractions (**LangChain**), dense + lexical + hybrid retrieval, reciprocal rank fusion, cross-encoder reranking, context filtering, conversation-aware query rewriting, thread-scoped document retrieval, citation generation, persistent conversations (**PostgreSQL**), vector storage (**Qdrant**), tracing (**LangSmith**), and evaluation (**DeepEval**).

---

## ✨ Key Capabilities

| Category | Capability |
|---|---|
| 🧠 **Agentic Reasoning** | Plan → Execute → Observe loop with iterative tool calling until the goal is achieved |
| 🔍 **Hybrid Retrieval** | Dense (semantic) + BM25 (lexical) fused via Reciprocal Rank Fusion |
| 🎯 **Reranking** | Cross-encoder reranking + context filtering for high-precision passages |
| 🛠️ **Tool Calling** | Calculator, Python execution, web search, and retriever tools routed dynamically |
| 📎 **Citations** | Every generated answer is grounded with traceable source citations |
| 💾 **Persistence** | Thread-scoped conversation history in PostgreSQL |
| 📊 **Observability** | Full run tracing via LangSmith |
| ✅ **Evaluation** | DeepEval RAG Triad — faithfulness, relevancy, recall, precision, correctness |

---

## 🏗️ System Architecture

```mermaid
flowchart TB
    subgraph Client["🖥️ Client Layer"]
        UI[Streamlit UI]
    end

    subgraph Agent["🧠 Agentic Core — LangGraph"]
        Planner[Planner Agent]
        Executor[Executor]
        Tools[Tool Adapters]
        Observer[Observation Agent]
        Answer[Answer Agent]
    end

    subgraph Data["💾 Data Layer"]
        Qdrant[(Qdrant Vector DB)]
        SQL[(PostgreSQL)]
    end

    subgraph Obs["📊 Observability & Eval"]
        LangSmith[LangSmith Tracing]
        DeepEval[DeepEval Runner]
    end

    UI --> Planner
    Planner --> Executor
    Executor --> Tools
    Tools --> Qdrant
    Tools --> SQL
    Tools --> Observer
    Observer --> Planner
    Observer --> Answer
    Answer --> UI

    Agent -.traces.-> LangSmith
    Agent -.samples.-> DeepEval

    style Client fill:#0f172a,color:#fff,stroke:#38bdf8
    style Agent fill:#1e1b3a,color:#fff,stroke:#a78bfa
    style Data fill:#1a0f1f,color:#fff,stroke:#f472b6
    style Obs fill:#0f1f16,color:#fff,stroke:#34d399
```

---

## 🔁 Agent Architecture

### Main Agentic Chat Loop

> Mirrors the runtime graph: the **Planner** issues a tool call, the **Executor** dispatches it to the right adapter, results are **observed**, and the loop continues until the goal is achieved.

```mermaid
flowchart TD
    Start(["👤 User Query"]) --> Planner["🧭 Planner Agent"]

    Planner -->|"1. Generates ToolCall"| Executor["⚙️ Executor"]
    Executor -->|"2. Fetches tool instance"| Registry[("📋 registry.py")]
    Executor -->|"3. Passes ToolContext"| Adapters{{"🔀 Tool Adapters"}}

    Adapters -->|Route| Calc["🧮 calculator_tool"]
    Adapters -->|Route| Py["🐍 python_tool"]
    Adapters -->|Route| Web["🌐 web_search_tool"]
    Adapters -->|Route| Retr["📚 retriever_tool"]

    Calc -->|Delegate| CalcSvc["calculator_service"]
    Py -->|Delegate| PySvc["python_service"]
    Web -->|Delegate| WebSvc["web_search_service"]
    Retr -->|Delegate| RetrSvc["retrieval_service"]

    CalcSvc -->|"4. Raw Data"| Adapters
    PySvc -->|"4. Raw Data"| Adapters
    WebSvc -->|"4. Raw Data"| Adapters
    RetrSvc -->|"4. Raw Data"| Adapters

    Adapters -->|"5. Packages ToolResult"| Executor
    Executor -->|"6. Passes ToolResult"| Observer["🔎 Observation Agent"]
    Observer -->|"7. Evaluates Result"| Complete{"✅ Is the answer complete?"}

    Complete -->|"Yes — keep searching"| Planner
    Complete -->|"No — goal achieved"| AnswerAgent["📝 Answer Agent"]
    AnswerAgent --> Final(["🏁 Final Response"])

    style Start fill:#0f172a,color:#fff,stroke:#38bdf8,stroke-width:2px
    style Planner fill:#2e1065,color:#fff,stroke:#a78bfa,stroke-width:2px
    style Executor fill:#0c4a6e,color:#fff,stroke:#38bdf8,stroke-width:2px
    style Adapters fill:#7c2d12,color:#fff,stroke:#fb923c,stroke-width:2px
    style Observer fill:#2e1065,color:#fff,stroke:#a78bfa,stroke-width:2px
    style Complete fill:#312e81,color:#fff,stroke:#818cf8,stroke-width:2px
    style AnswerAgent fill:#2e1065,color:#fff,stroke:#a78bfa,stroke-width:2px
    style Final fill:#0f172a,color:#fff,stroke:#38bdf8,stroke-width:2px
    style Registry fill:#082f49,color:#fff,stroke:#38bdf8
    style Calc fill:#78350f,color:#fff,stroke:#fbbf24
    style Py fill:#78350f,color:#fff,stroke:#fbbf24
    style Web fill:#78350f,color:#fff,stroke:#fbbf24
    style Retr fill:#78350f,color:#fff,stroke:#fbbf24
    style CalcSvc fill:#064e3b,color:#fff,stroke:#34d399
    style PySvc fill:#064e3b,color:#fff,stroke:#34d399
    style WebSvc fill:#064e3b,color:#fff,stroke:#34d399
    style RetrSvc fill:#064e3b,color:#fff,stroke:#34d399
```

### Router · Planner · Executor

| Stage | Responsibility |
|---|---|
| **Router** | Classifies the incoming query and determines whether retrieval, a tool, or a direct answer is required |
| **Planner** | Decomposes the goal into a sequence of tool calls, one step at a time |
| **Executor** | Resolves the concrete tool instance from the registry and dispatches with full context |
| **Observation** | Evaluates whether the accumulated tool results satisfy the user's goal, looping back to the planner if not |

---

## 📥 Document Ingestion Pipeline

```mermaid
flowchart TD
    Upload(["📤 User Uploads File"]) --> DocSvc["📄 document_service.py"]
    DocSvc --> IngestSvc["⚙️ ingestion_service.py"]
    IngestSvc -->|"Load → Parse → Chunk → Embed"| Qdrant[("🟥 Qdrant Vector DB")]
    IngestSvc --> SQLMeta[("🟥 SQL Metadata DB")]
    Qdrant -.Searches.-> IngestSvc

    style Upload fill:#0f172a,color:#fff,stroke:#38bdf8,stroke-width:2px
    style DocSvc fill:#1e293b,color:#fff,stroke:#94a3b8,stroke-width:2px
    style IngestSvc fill:#1e293b,color:#fff,stroke:#94a3b8,stroke-width:2px
    style Qdrant fill:#4c0519,color:#fff,stroke:#fb7185,stroke-width:2px
    style SQLMeta fill:#4c0519,color:#fff,stroke:#fb7185,stroke-width:2px
```

### Chunking & Embedding

```mermaid
flowchart LR
    Doc["📄 Raw Document"] --> Parse["🔍 Parse"] --> Chunk["✂️ Chunk"] --> Embed["🧬 Embed"] --> Store[("Qdrant")]

    style Doc fill:#0f172a,color:#fff,stroke:#38bdf8
    style Parse fill:#1e293b,color:#fff,stroke:#94a3b8
    style Chunk fill:#1e293b,color:#fff,stroke:#94a3b8
    style Embed fill:#1e293b,color:#fff,stroke:#94a3b8
    style Store fill:#4c0519,color:#fff,stroke:#fb7185
```

---

## 🔀 Hybrid Retrieval Workflow

```mermaid
flowchart TD
    Query["❓ Rewritten Query"] --> Dense["🧬 Dense Retrieval"]
    Query --> BM25["🔤 BM25 Retrieval"]
    Dense --> RRF["🔗 Reciprocal Rank Fusion"]
    BM25 --> RRF
    RRF --> Rerank["🎯 Cross-Encoder Reranking"]
    Rerank --> Filter["🧹 Context Filtering"]
    Filter --> Result["📦 RetrievalResult"]
    Result --> Cite["📎 Citation Builder"]
    Cite --> Prompt["🧱 PromptBuilder"]
    Prompt --> Gen["✍️ Generation"]
    Gen --> Persist[("🗄️ Persistence")]

    style Query fill:#0f172a,color:#fff,stroke:#38bdf8
    style Dense fill:#1e1b3a,color:#fff,stroke:#a78bfa
    style BM25 fill:#1e1b3a,color:#fff,stroke:#a78bfa
    style RRF fill:#312e81,color:#fff,stroke:#818cf8
    style Rerank fill:#7c2d12,color:#fff,stroke:#fb923c
    style Filter fill:#7c2d12,color:#fff,stroke:#fb923c
    style Result fill:#064e3b,color:#fff,stroke:#34d399
    style Cite fill:#064e3b,color:#fff,stroke:#34d399
    style Prompt fill:#0c4a6e,color:#fff,stroke:#38bdf8
    style Gen fill:#0c4a6e,color:#fff,stroke:#38bdf8
    style Persist fill:#4c0519,color:#fff,stroke:#fb7185
```

---

## ✅ DeepEval Evaluation Architecture

```mermaid
flowchart TD
    Prod["📡 Production Traffic"] --> Collector["🗃️ Evaluation Collector"]
    Collector --> Sample["📋 EvaluationSample"]
    Sample --> RetrEval["🔍 Retrieval Evaluation"]
    Sample --> GenEval["✍️ Generation Evaluation"]

    RetrEval --> Triad{{"⚖️ RAG Triad"}}
    GenEval --> Triad

    Triad --> CR["Contextual Relevancy"]
    Triad --> CRec["Contextual Recall"]
    Triad --> CP["Contextual Precision"]
    Triad --> Faith["Faithfulness"]
    Triad --> AR["Answer Relevancy"]
    Triad --> AC["Answer Correctness"]

    CR & CRec & CP & Faith & AR & AC --> Judge["👨‍⚖️ Judge Model"]
    Judge --> Runner["🏃 DeepEval Runner"]
    Runner --> Report["📊 Evaluation Report"]
    Report --> Diagnosis["🩺 Retrieval vs Generation Failure Diagnosis"]

    style Prod fill:#0f172a,color:#fff,stroke:#38bdf8
    style Collector fill:#1e293b,color:#fff,stroke:#94a3b8
    style Sample fill:#1e293b,color:#fff,stroke:#94a3b8
    style Triad fill:#312e81,color:#fff,stroke:#818cf8
    style Judge fill:#7c2d12,color:#fff,stroke:#fb923c
    style Runner fill:#064e3b,color:#fff,stroke:#34d399
    style Report fill:#064e3b,color:#fff,stroke:#34d399
    style Diagnosis fill:#4c0519,color:#fff,stroke:#fb7185
```

### DeepEval RAG Triad

| Metric | Measures |
|---|---|
| **Contextual Relevancy** | Are retrieved chunks relevant to the query? |
| **Contextual Recall** | Does retrieved context cover the expected answer? |
| **Contextual Precision** | Are the most relevant chunks ranked highest? |
| **Faithfulness** | Is the answer grounded in retrieved context (no hallucination)? |
| **Answer Relevancy** | Does the answer address the actual question? |
| **Answer Correctness** | Does the answer match the ground truth? |

---

## 🧰 Technology Stack

<div align="center">

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM Framework | LangChain |
| LLM Inference | Groq, Mistral AI |
| Vector Store | Qdrant |
| Relational Store | PostgreSQL |
| UI | Streamlit |
| Tracing | LangSmith |
| Evaluation | DeepEval |

</div>

---

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
