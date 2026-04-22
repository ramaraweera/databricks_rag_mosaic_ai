# Databricks RAG with Mosaic AI — Portfolio Project

> **Author:** Ravi Amaraweera · Senior Data Architect / Analytics Engineer  
> **Stack:** Databricks · Unity Catalog · Mosaic AI · Vector Search · LangChain · MLflow  
> **Data:** Apache Airflow open-source documentation (Apache 2.0 licence — free to use)

A two-notebook, end-to-end Retrieval-Augmented Generation (RAG) system built entirely on the Databricks Lakehouse platform. It ingests technical documentation, builds a semantic vector index, and exposes the resulting agent as a live REST API with a shareable chat UI — all within the Databricks free trial ($400 credits).

---

## Architecture

```
 Raw docs (web)                      Databricks Lakehouse
 ┌──────────────┐     ingest      ┌──────────────────────────────────────────────┐
 │ Airflow docs │ ──────────────► │  Unity Catalog (rag_portfolio.doc_search)    │
 │ (HTML pages) │                 │  ┌───────────────────────────────────────┐   │
 └──────────────┘                 │  │  Delta Table: airflow_docs_chunks     │   │
                                  │  │  (CDF enabled, source of truth)       │   │
                                  │  └────────────────┬──────────────────────┘   │
                                  │                   │ Delta Sync               │
                                  │  ┌────────────────▼──────────────────────┐   │
                                  │  │  Vector Search Index                  │   │
                                  │  │  (BGE Large embeddings, HNSW)         │   │
                                  │  └────────────────┬──────────────────────┘   │
                                  │                   │ semantic search           │
                                  │  ┌────────────────▼──────────────────────┐   │
                                  │  │  RAG Chain (LangChain + Databricks)   │   │
                                  │  │  Retriever → Prompt → LLM             │   │
                                  │  └────────────────┬──────────────────────┘   │
                                  │                   │ MLflow ChatModel          │
                                  │  ┌────────────────▼──────────────────────┐   │
                                  │  │  Model Serving Endpoint (scale-to-0)  │   │
                                  │  │  REST API + Review App chat UI        │   │
                                  │  └───────────────────────────────────────┘   │
                                  └──────────────────────────────────────────────┘
```

---

## Repository Structure

```
├── databricks_rag_vector_search.ipynb    # V1 — data ingestion, vector index, LangChain chain
├── databricks_rag_mosaic_ai_v2.ipynb     # V2 — MLflow, Model Serving, Agent Evaluation
├── README.md                             # Developer reference (this file)
└── GETTING_STARTED.md                    # Beginner-friendly walkthrough
```

---

## Prerequisites

| Requirement | Detail |
|---|---|
| Databricks Runtime | DBR 14.3 LTS ML or higher |
| Databricks cluster | Single-node, 14GB RAM minimum (e.g. i3.xlarge or equivalent) |
| Unity Catalog | Enabled on your workspace (default on all new workspaces) |
| Permissions | `CREATE CATALOG`, `CREATE SCHEMA`, `CREATE TABLE` on Unity Catalog |
| Vector Search | Databricks Vector Search enabled on workspace |
| Foundation Model APIs | Enabled — confirm in Serving tab |
| Credits | ~$5-$15 within the Databricks $400 free trial |

---

## Quickstart

### Step 1 — Clone the repo into your Databricks workspace

```bash
# In a Databricks terminal or Repos tab:
git clone https://github.com/ramaraweera/databricks_rag_mosaic_ai.git
```

Or: **Workspace → Repos → Add Repo → paste this URL**.

### Step 2 — Run V1: Data Ingestion + Vector Index

Open `databricks_rag_vector_search.ipynb` and run cells top-to-bottom.

| Cell range | What happens | Approx. time |
|---|---|---|
| 1–3 | Install packages, set config | 3 min |
| 4–7 | Scrape and chunk Airflow docs | 5 min |
| 8–10 | Create Unity Catalog + Delta Table | 2 min |
| 11–13 | Create Vector Search endpoint + index | 8–15 min |
| 14–16 | Build LangChain chain, test queries | 2 min |
| 17–18 | Batch evaluation, save to Delta | 3 min |
| 19 | *(Optional)* Cleanup | — |

> **Wait for Vector Search index status = ONLINE before proceeding to V2.**

### Step 3 — Run V2: Mosaic AI Production Layer

Open `databricks_rag_mosaic_ai_v2.ipynb` and run cells top-to-bottom.

| Cell | What happens | Approx. time |
|---|---|---|
| 1 | Install Mosaic AI packages | 3 min |
| 2–3 | Set variables, create MLflow experiment | 1 min |
| 4 | Reconnect Vector Search + rebuild chain | 1 min |
| 5–6 | Wrap chain as MLflow PyFunc, local test | 1 min |
| 7–8 | Log + register model in Unity Catalog | 2 min |
| 9–10 | Deploy to Model Serving endpoint | 5–8 min |
| 11 | Test live REST API | 1 min |
| 12 | Add MLflow Tracing | 1 min |
| 13–15 | Agent Evaluation with LLM judges | 3 min |
| 16 | Explore Review App | — |
| 17 | Improve prompt, re-deploy V2 | 5 min |
| 18 | *(Commented out)* Cleanup | — |

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Storage | Delta Lake + Unity Catalog | Versioned source-of-truth for document chunks |
| Change tracking | Change Data Feed (CDF) | Incremental sync from Delta to Vector Search |
| Embeddings | `databricks-bge-large-en` | Text → vector representation |
| Vector store | Databricks Vector Search (Delta Sync) | ANN similarity search |
| LLM | `databricks-gpt-oss-20b` | Response generation (cheapest Foundation Model) |
| Chain | LangChain + `DatabricksVectorSearch` | Retrieve → Prompt → Generate pipeline |
| Model packaging | `mlflow.pyfunc.ChatModel` | OpenAI-compatible model interface |
| Model registry | MLflow + Unity Catalog | Versioned model artefacts with access control |
| Serving | Mosaic AI Model Serving (serverless) | Scale-to-zero REST API |
| Review UI | Mosaic AI Review App | Shareable chat UI (no Databricks login needed) |
| Observability | MLflow Tracing | Per-request RETRIEVER + LLM span recording |
| Evaluation | Mosaic AI Agent Evaluation | Groundedness, relevance, retrieval precision |

---

## Customising for Your Own Data

To adapt this system to a different knowledge base:

1. **Replace the data source** — edit the scraping/loading cells in V1 with your own URLs or file paths.
2. **Update chunk size** — adjust `chunk_size` and `chunk_overlap` in the splitter cell based on your document length.
3. **Rename the catalog/schema** — update `CATALOG` and `SCHEMA` in Cell 2 of both notebooks.
4. **Update the system prompt** — replace "Apache Airflow" references in the prompt template with your domain.
5. **Update the evaluation dataset** — replace the 5 QA pairs in V2 Cell 13 with domain-specific questions.

No other changes are required.

---

## Key Design Decisions

**Why no LlamaIndex or Hugging Face?**  
Pure Databricks-native stack (Vector Search, Foundation Model APIs, MLflow, agents.deploy) keeps all components inside one governance boundary — Unity Catalog access controls apply to the data, the model, and the endpoint simultaneously.

**Why Delta Sync index (not Direct Access)?**  
Delta Sync automatically propagates chunk-level changes from the Delta Table to the vector index using CDF. Incremental re-indexing is free and requires no code.

**Why Apache Airflow docs?**  
Apache 2.0 licence — no copyright or attribution issues. The content is deeply technical, which creates a meaningful retrieval challenge (shallow embedding matches on "Airflow" alone won't produce correct answers).

**Why `databricks-gpt-oss-20b` as the LLM?**  
The cheapest Foundation Model API option — typically $0.001-$0.003 per 1K tokens. Stays well within the $400 free trial for development and evaluation.

---

## Evaluation Output Schema

V1 persists evaluation results to `rag_portfolio.doc_search.eval_results`:

```
Column               Type       Description
----                 ----       -----------
question             STRING     Input question
expected_answer      STRING     Human-verified correct answer
actual_answer        STRING     RAG chain output
relevance_score      DOUBLE     LLM judge: 1-5, relevance to question
groundedness_score   DOUBLE     LLM judge: 1-5, supported by retrieved context
retrieval_source     ARRAY      Source file paths of retrieved chunks
evaluation_date      TIMESTAMP  When the evaluation row was generated
```

---

## Teardown (Cost Control)

The **Vector Search endpoint** charges ~$0.28/hr while running — the only always-on cost.
The **Model Serving endpoint** has scale-to-zero and costs $0 when idle.

To stop all charges:
1. Run the cleanup cell in V2 (Cell 18, uncomment first) to delete the serving endpoint.
2. Run the cleanup cell in V1 (Cell 19, uncomment first) to delete the Vector Search endpoint.
3. Optionally drop the Unity Catalog tables with the SQL commands shown in the cleanup cells.

---

## About the Author

**Ravi Amaraweera** is a Senior Data Architect and Analytics Engineer specialising in lakehouse architecture, data platform engineering, and AI/ML integration on cloud-native data platforms. This project demonstrates end-to-end Databricks capability from raw data ingestion through to a production-deployed RAG agent.

🔗 [LinkedIn](https://www.linkedin.com/in/ravindra-amaraweera) | [GitHub](https://github.com/ramaraweera)

---

## Licence

Code: MIT. Documentation data (Apache Airflow): Apache 2.0.
