# Databricks RAG with Mosaic AI — Production Deployment Layer

> **Author:** Ravi Amaraweera · Senior Data Architect / Analytics Engineer  
> **Part of the portfolio series:** `databricks_rag_with_vectorSearch` (V1) → this repo (V2)

---

## What This Project Is

This project demonstrates a **production-grade RAG (Retrieval-Augmented Generation) pipeline** built entirely on the Databricks Lakehouse Platform using the Mosaic AI stack.

It answers the question:  
> *"I've built a RAG chain in a notebook. How do I turn it into a versioned, monitored, deployed product?"*

The answer is this notebook.

---

## Architecture: V1 → V2 Pipeline

```
V1 (databricks_rag_vector_search.ipynb)
──────────────────────────────────────────
  Apache Airflow Docs
       │
       ▼
  Delta Table (Unity Catalog)   ←── CDF-enabled, append-safe
       │
       ▼
  Databricks Vector Search      ←── Delta Sync Index (BGE Large embeddings)
       │
       ▼
  LangChain QA Chain            ←── In-notebook prototype
       │
       ▼ (V2 picks up here)

V2 (databricks_rag_mosaic_ai_v2.ipynb)
──────────────────────────────────────────
  mlflow.pyfunc.ChatModel       ←── Standardised OpenAI-compatible interface
       │
       ▼
  MLflow Experiment Run         ←── Every iteration tracked with run_id
       │
       ▼
  Unity Catalog Model Registry  ←── Version 1, Version 2 … (rollback-safe)
       │
       ▼
  agents.deploy()               ←── One-liner: serverless endpoint + Review App
       │
       ├──► /invocations REST API   ←── Any app can call this over HTTP
       ├──► Review App Chat UI      ←── Shareable, no Databricks login needed
       └──► MLflow Traces           ←── RETRIEVER + LLM spans for debugging
                │
                ▼
           Agent Evaluation        ←── LLM judges: groundedness, relevance, precision
```

---

## What V2 Adds Over V1

| Capability | Tool | Business value |
|---|---|---|
| Model versioning | MLflow + Unity Catalog | Rollback to any previous chain version |
| Live REST API | `agents.deploy()` → Serverless endpoint | Any application can call the RAG chain |
| Scale-to-zero serving | Databricks Serverless | Zero idle cost (unlike always-on VS endpoint) |
| Shareable chat UI | Mosaic AI Review App | Stakeholder demos with no Databricks login |
| Observability | MLflow Tracing | Debug retrieval vs generation failures |
| Quality scoring | Agent Evaluation | Objective groundedness and relevance scores |

---

## Prerequisites

### 1 — Run V1 first
This notebook **builds on top of** `databricks_rag_vector_search.ipynb`.  
Before starting V2, confirm these V1 resources exist:

| Resource | Expected name |
|---|---|
| Vector Search endpoint | `rag-portfolio-endpoint` |
| Vector Search index | `rag_portfolio.doc_search.airflow_docs_index` |
| Unity Catalog | `rag_portfolio.doc_search` |

### 2 — Databricks workspace
| Requirement | Minimum |
|---|---|
| Databricks Runtime | 15.4 LTS ML or above |
| Cluster | Single-node, 16 GB RAM (i3.xlarge or equivalent) |
| Unity Catalog | Enabled (`CREATE MODEL`, `USE CATALOG` permissions) |
| Foundation Model APIs | Enabled (Workspace Settings → Machine Learning) |

### 3 — Permissions
`CREATE MODEL` on `rag_portfolio.doc_search`  
`USE CATALOG` on `rag_portfolio`  
`CAN USE` on `rag-portfolio-endpoint` (Vector Search)

---

## Repository Structure

```
databricks_rag_mosaic_ai/
├── databricks_rag_mosaic_ai_v2.ipynb    ← Enhanced notebook (this repo)
├── README.md                            ← Developer reference (this file)
└── GETTING_STARTED.md                   ← Beginner walkthrough
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data store | Delta Lake (Unity Catalog) | Source-of-truth for document chunks |
| Embeddings | BGE Large EN (Databricks Foundation Model API) | Semantic vector encoding |
| Vector store | Databricks Vector Search (Delta Sync) | Low-latency semantic retrieval |
| LLM | `databricks-gpt-oss-20b` (Foundation Model API) | Generation (~$1.00/1M tokens) |
| Chain | LangChain + `SimpleRetrievalQA` | Retrieve → Prompt → Generate |
| Model packaging | `mlflow.pyfunc.ChatModel` | OpenAI-compatible interface |
| Model registry | MLflow + Unity Catalog | Version history and governance |
| Deployment | `databricks.agents.deploy()` | Serverless endpoint + Review App |
| Observability | MLflow Tracing | RETRIEVER and LLM spans |
| Evaluation | Mosaic AI Agent Evaluation | Groundedness, relevance, precision |

---

## Quickstart

```bash
# 1. Clone this repo
git clone https://github.com/ramaraweera/databricks_rag_mosaic_ai.git

# 2. Import the notebook into Databricks
#    Workspace → Import → From URL or file
#    Select: databricks_rag_mosaic_ai_v2.ipynb

# 3. Attach to a 15.4 LTS ML cluster

# 4. Run cells top-to-bottom
#    Cell 1 installs packages (kernel restarts — this is expected)
#    Cell 2 sets configuration constants — update if your names differ
#    Cells 3–9 build, log, register, and deploy the chain
#    Cell 18 (cleanup) is commented out by design
```

### Timing

| Cell range | Step | Typical time |
|---|---|---|
| 1 | Package install + kernel restart | 1–2 min |
| 2–4 | Config + chain rebuild | < 1 min |
| 5–6 | Model file + local test | < 1 min |
| 7 | `mlflow.pyfunc.log_model` | 1–3 min |
| 8 | Unity Catalog model registration | < 1 min |
| 9–10 | `agents.deploy()` + wait for READY | 5–10 min |
| 11 | Live API test | < 1 min |
| 12 | MLflow Tracing | < 1 min |
| 13–15 | Evaluation dataset + scoring | 2–4 min |
| 16–17 | Review App + V2 improvement | demo only |

---

## Customising for Your Own Data

The only required changes to run this on a different knowledge base:

| Cell | Variable | Change to |
|---|---|---|
| 2 | `CATALOG`, `SCHEMA` | Your Unity Catalog names |
| 2 | `VS_ENDPOINT`, `VS_INDEX` | Your V1 Vector Search names |
| 2 | `MODEL_NAME` | Your desired Unity Catalog model path |
| 4 | `columns=["text", "source", "doc_type"]` | Your Delta table column names |
| 4 | Prompt template | Domain-specific instructions |
| 13 | `eval_dataset` | QA pairs from your own knowledge domain |

---

## Cost Reference

All figures are approximate. Costs depend on Databricks pricing tier and region.

| Resource | Idle cost | Active cost |
|---|---|---|
| Vector Search endpoint (V1) | ~$0.28/hr always-on | Same |
| Model Serving endpoint (V2) | **$0.00** (scale-to-zero) | ~$0.07–$0.15/1K tokens |
| `databricks-gpt-oss-20b` | — | ~$1.00/1M input tokens |
| `databricks-bge-large-en` | — | ~$1.43/1M tokens |

**Total estimated cost for this notebook: $5–$15 within Databricks $400 free trial.**

> Delete the Vector Search endpoint (V1, Cell 19) when done. It is the only always-on resource.

---

## Key Design Decisions

### Why `mlflow.pyfunc.ChatModel` instead of `mlflow.pyfunc.PythonModel`?
`ChatModel` generates an OpenAI-compatible `/invocations` schema automatically.  
This means the endpoint works with any OpenAI SDK client — no custom parsing needed.

### Why `agents.deploy()` instead of standard `mlflow.deployments`?
`agents.deploy()` does three things in one call: creates the serving endpoint, enables  
scale-to-zero, and provisions the Review App. Standard deployment requires three separate API calls.

### Why `SimpleRetrievalQA` instead of `langchain.chains.RetrievalQA`?
`RetrievalQA` was removed in LangChain 1.x. `SimpleRetrievalQA` is a self-contained  
drop-in replacement that avoids the deprecated import and produces identical behaviour.

### Why code-based model logging (`python_model="/tmp/rag_chain_model.py"`)?
Code-based logging (MLflow 2.12+) serialises the Python file rather than pickling a  
class instance. This is more reproducible and avoids dependency conflicts between  
notebook and serving endpoint environments.

---

## Evaluation Output Schema

Each row in the evaluation results DataFrame contains:

| Column | Type | Description |
|---|---|---|
| `question` | str | The question sent to the chain |
| `answer` | str | The chain's generated answer |
| `groundedness_score` | int (1–5) | Word-overlap proxy for answer groundedness |
| `sources` | list[str] | Source document names used for the answer |

**Target:** average `groundedness_score` ≥ 4.0. Scores below 3 indicate retrieval or generation problems.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: rag_chain_model` | Cell 5 not yet run | Re-run Cell 5 |
| `ENDPOINT_NOT_FOUND` in Cell 4 | V1 Vector Search endpoint deleted | Re-run Cells 7–10 in V1 notebook |
| `mlflow.register_model` permission error | Missing `CREATE MODEL` | Add permission in Unity Catalog → rag_portfolio |
| `agents.deploy()` timeout | Workspace tier limitation | Check Workspace Settings → Serving |
| Review App URL 404 | Endpoint still provisioning | Wait for Cell 10 to print `READY` |
| Low groundedness scores (< 3) | Wrong chunks retrieved | Increase `k` in retriever or improve prompt |
| `FutureWarning: ChatModel is deprecated` | MLflow 3.x | Safe to ignore; use `ResponsesAgent` in MLflow 3+ |

---

## Portfolio Context

This repository is part of a two-notebook portfolio series demonstrating end-to-end  
AI engineering on Databricks by **Ravi Amaraweera**, Senior Data Architect / Analytics Engineer.

| Notebook | What it shows |
|---|---|
| `databricks_rag_vector_search.ipynb` | Data ingestion → Delta → Vector Search → LangChain prototype |
| `databricks_rag_mosaic_ai_v2.ipynb` | MLflow packaging → Unity Catalog → Deployed REST API → Evaluation |

Together, these notebooks demonstrate the full spectrum from raw data to a monitored,  
versioned, production AI service — a rare combined skill set across data engineering and AI engineering.

---

## Licence
Data: Apache Airflow documentation (Apache 2.0)  
Notebook: MIT
