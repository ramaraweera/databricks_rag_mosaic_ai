# Getting Started: Databricks RAG with Mosaic AI

> A beginner-friendly walkthrough for developers new to Databricks and RAG.
> No prior experience with vector databases or LLMs is required.

---

## Table of Contents

1. [What is RAG?](#1-what-is-rag)
2. [What is Databricks?](#2-what-is-databricks)
3. [Full Architecture: How All the Pieces Connect](#3-full-architecture)
4. [V1 Notebook Walkthrough](#4-v1-notebook-walkthrough)
5. [V2 Notebook Walkthrough](#5-v2-notebook-walkthrough)
6. [Key Concepts Glossary](#6-key-concepts-glossary)
7. [Troubleshooting](#7-troubleshooting)
8. [Next Steps](#8-next-steps)

---

## 1. What is RAG?

**The problem RAG solves:**

A large language model (LLM) like GPT-4 is trained on internet data up to a cutoff date and has no knowledge of your private or specialised documents. If you ask it about your company's internal Confluence wiki, your product's custom API, or the latest Airflow release notes — it either guesses (hallucinates) or says it doesn't know.

**RAG = Retrieval-Augmented Generation**

Instead of relying on the LLM's baked-in knowledge, RAG retrieves the most relevant passages from YOUR documents at query time and sends them to the LLM as context.

```
Without RAG:
  User: "How do I set task dependencies in Airflow?"
  LLM:  "[Guesses based on training data — may be wrong or outdated]"

With RAG:
  User:       "How do I set task dependencies in Airflow?"
  Retriever:  "[Searches your vector index → finds 4 relevant doc chunks]"
  Prompt:     "Answer ONLY using this context: [4 chunks]\n\nQuestion: ..."
  LLM:        "[Grounded, accurate answer based on YOUR documentation]"
```

The key components of a RAG system are:
- **A document corpus** — the knowledge base (e.g. Airflow docs, your wiki)
- **An embedding model** — converts text chunks into vectors (lists of numbers representing meaning)
- **A vector store** — database that finds the most semantically similar vectors for a query
- **An LLM** — reads the retrieved context and generates a natural-language answer

---

## 2. What is Databricks?

Databricks is a cloud data platform built on top of Apache Spark and Delta Lake. It provides:

| Feature | What it is | Used in this project |
|---|---|---|
| **Notebooks** | Interactive code environment (Python, SQL, Scala) | All development |
| **Unity Catalog** | Centralised data governance (tables, models, files) | Storing chunks + model registry |
| **Delta Lake** | Open-source storage layer with ACID transactions | Source-of-truth for document chunks |
| **Vector Search** | Managed vector database service | Similarity search over embeddings |
| **Foundation Model APIs** | Hosted LLM endpoints (no GPU setup needed) | Embedding + generation |
| **MLflow** | Experiment tracking + model versioning | Logging every chain iteration |
| **Model Serving** | REST API hosting for ML models | Exposing the RAG agent as an API |
| **Mosaic AI** | AI layer for agent evaluation, review apps | Quality scoring + demo UI |

**Free trial:** Databricks offers $400 in free credits. This project costs $5–$15.

---

## 3. Full Architecture

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                         DATABRICKS WORKSPACE                            │
 │                                                                          │
 │  ┌──────────────┐   chunk+embed   ┌────────────────────────────────┐   │
 │  │ Airflow Docs │ ──────────────► │  Delta Table                   │   │
 │  │ (HTML pages) │                 │  rag_portfolio.doc_search       │   │
 │  └──────────────┘                 │  .airflow_docs_chunks           │   │
 │                                   │  (CDF enabled)                  │   │
 │                                   └──────────────┬─────────────────┘   │
 │                                                  │ Delta Sync           │
 │                                                  ▼                      │
 │                                   ┌──────────────────────────────────┐  │
 │                                   │  Vector Search Index             │  │
 │                                   │  BGE Large embeddings (1024-dim) │  │
 │                                   │  HNSW approximate nearest-neigh  │  │
 │                                   └──────────────┬───────────────────┘  │
 │                                                  │ top-k chunks         │
 │                                                  ▼                      │
 │  User Question ──────────────► ┌──────────────────────────────────────┐ │
 │                                │  LangChain RAG Chain                 │ │
 │                                │  1. Embed question (BGE Large)       │ │
 │                                │  2. Search index (k=4 chunks)        │ │
 │                                │  3. Format prompt with context       │ │
 │                                │  4. Call LLM (GPT-OSS-20B)           │ │
 │                                └──────────────┬───────────────────────┘ │
 │                                               │ MLflow ChatModel         │
 │                                               ▼                         │
 │                                   ┌──────────────────────────────────┐  │
 │                                   │  Model Serving Endpoint          │  │
 │                                   │  (serverless, scale-to-zero)     │  │
 │                                   │  POST /invocations               │  │
 │                                   └──────────────────────────────────┘  │
 │                                                                          │
 └──────────────────────────────────────────────────────────────────────────┘
                              ▲
              REST API calls from any application
```

---

## 4. V1 Notebook Walkthrough

### `databricks_rag_vector_search.ipynb`

This notebook builds the foundation: getting documents in, chunked, embedded, and searchable.

---

### Cell 1 — Install packages

**What it does:** Installs Python libraries not included in the default Databricks ML runtime.

**Key packages:**
- `langchain` — framework for building LLM chains and pipelines
- `databricks-vectorsearch` — Python client for Databricks Vector Search
- `databricks-langchain` — Databricks-specific LangChain integrations

**Why the kernel restarts:** Newly installed packages are not importable until Python is restarted. `dbutils.library.restartPython()` handles this automatically.

**Customise:** No changes needed here unless you want to pin specific library versions.

---

### Cell 2 — Configuration variables

**What it does:** Sets all configuration constants in one place — catalog name, schema name, endpoint name, etc.

**Key concept — Unity Catalog naming:**  
Databricks Unity Catalog uses three-level naming: `catalog.schema.table_or_index`.
This project uses `rag_portfolio.doc_search.airflow_docs_chunks`.

**Customise:** Change `CATALOG`, `SCHEMA`, and the endpoint names if your workspace already has conflicting names.

---

### Cell 3 — Create the Unity Catalog and Schema

**What it does:** Creates `rag_portfolio` catalog and `doc_search` schema using SQL.

**Key concept — Unity Catalog:**  
Think of the catalog as a database server, the schema as a database, and the table as a table.  
Unity Catalog also governs who can access each level — read, write, or manage.

**What to watch:** If you get a `permission denied` error, your Databricks account may need catalog creation privileges. Ask your workspace admin.

---

### Cell 4 — Scrape and load Airflow documentation

**What it does:** Downloads HTML pages from the Apache Airflow documentation website and converts them to plain text.

**Key concept — why Airflow docs?**  
Apache 2.0 licence means we can scrape, store, and use the content freely. The docs are deeply technical, which makes them a good test for a RAG system — you can't answer questions like "What parameters does the BashOperator accept?" without actually retrieving the right page.

**Customise:** Replace the URL list with your own documentation URLs, a local folder of PDFs, or a SharePoint site.

---

### Cell 5 — Chunk the documents

**What it does:** Splits long documents into smaller overlapping text chunks using `RecursiveCharacterTextSplitter`.

**Why chunking is necessary:**  
Embedding models have a token limit (typically 512 tokens). A full documentation page is too long to embed as a single vector. Smaller chunks also produce more precise retrieval — the vector for a 200-word paragraph matches a question better than the vector for a 5,000-word page.

```
Original page (3,000 words)
        ↓
[chunk 1: 500 chars][chunk 2: 500 chars][chunk 3: 500 chars]...
     ←  100-char overlap  →
```

**Overlap:** Each chunk shares 100 characters with the previous chunk. This prevents answers from being cut off mid-sentence at chunk boundaries.

**Customise:** Adjust `chunk_size` (200–1000 depending on document density) and `chunk_overlap` (10–20% of chunk size).

---

### Cell 6 — Preview chunks

**What it does:** Prints the first 3 chunks so you can verify the splitting is working correctly before writing to Delta.

**What to check:** Do the chunks contain coherent text (not HTML tags or navigation links)? If not, improve the HTML cleaning in Cell 4.

---

### Cell 7 — Write chunks to a Delta Table

**What it does:** Persists all chunks to `rag_portfolio.doc_search.airflow_docs_chunks` as a Delta Table with Change Data Feed enabled.

**Key concept — Delta Lake:**  
Delta is like a supercharged Parquet file — it supports ACID transactions, time travel (query past versions), and Change Data Feed (CDF). CDF tracks which rows were inserted, updated, or deleted since the last sync.

**Key concept — CDF (Change Data Feed):**  
When CDF is enabled, Databricks Vector Search can do *incremental* index updates — only re-embedding changed chunks instead of re-indexing everything. This makes updates fast and cheap.

---

### Cell 8 — Create the Vector Search endpoint

**What it does:** Provisions a Databricks Vector Search endpoint — a managed service that hosts the HNSW index and handles similarity search requests.

**Key concept — Vector Search endpoint:**  
The endpoint is a persistent, always-on service. It charges ~$0.28/hr while running (the main cost in this project). One endpoint can host many indexes.

**What to watch:** Endpoint creation takes 5–10 minutes. The cell polls until status = `ONLINE`.

---

### Cell 9 — Create the Vector Search index

**What it does:** Creates a Delta Sync index on the `airflow_docs_chunks` table using the BGE Large embedding model.

**Key concept — Delta Sync index:**  
A Delta Sync index reads directly from your Delta Table. When you add new documents (update the Delta Table), the index syncs automatically using CDF — no manual re-indexing.

**Key concept — BGE Large:**  
`databricks-bge-large-en` is a 1024-dimension embedding model. It converts each chunk of text into a 1024-element vector. Chunks with similar meaning produce vectors that are close together in 1024-dimensional space — this is what makes semantic search possible.

**What to watch:** Index creation takes 5–15 minutes depending on document volume. Wait for status = `ONLINE` before running the next cell.

---

### Cell 10 — Test a direct similarity search

**What it does:** Runs a raw similarity search against the index (no LLM involved) to verify the retrieval is working.

**What to look for:** The returned chunks should be semantically relevant to your test query. If they're not, the chunking or embedding step may need adjustment.

---

### Cell 11 — Build the LangChain retriever

**What it does:** Wraps the Vector Search index as a LangChain `Retriever` — the standard interface for "given a query, return the top-k relevant documents".

**Key concept — k (number of retrieved chunks):**  
`k=4` means the retriever returns the 4 most similar chunks for every query. More chunks = more context for the LLM but also a longer prompt (higher latency and cost). 3–5 is typical.

---

### Cell 12 — Build the full RAG chain

**What it does:** Assembles the complete RAG pipeline:
1. User query → embedding → vector search → top-4 chunks
2. Chunks formatted into a context block inside the prompt template
3. Prompt sent to the LLM (`databricks-gpt-oss-20b`)
4. LLM generates a grounded answer

**Key concept — system prompt:**  
The prompt template instructs the LLM to answer using ONLY the provided context. This is the primary technique for preventing hallucination in RAG systems.

```
"You are a technical assistant for Apache Airflow.
 Answer using ONLY the provided context.
 If context is insufficient, say: I do not have enough context.

 Context:
 [retrieved chunks]

 Question: [user query]

 Answer:"
```

---

### Cell 13 — Test the chain

**What it does:** Runs 5 sample questions through the full chain and prints answers with source citations.

**What good output looks like:**
- Answer directly addresses the question
- Answer references specific Airflow concepts from the docs
- Source files are Airflow documentation pages (not random web content)

---

### Cells 14–15 — Create evaluation dataset and run evaluation

**What it does:** Tests the chain against a "golden dataset" — questions paired with human-verified correct answers. An LLM judge scores each answer on groundedness (1-5) and relevance (1-5).

**Key concept — evaluation:**  
LLM outputs are probabilistic — the same question can produce slightly different answers on each run. Evaluation creates a repeatable quality benchmark so you can measure whether a change (new prompt, more chunks, different LLM) actually improves the system.

**Target score:** 4/5+ groundedness means the answer is well-supported by retrieved context.

---

### Cells 16–17 — Save results and display summary

**What it does:** Persists evaluation results to a Delta Table for longitudinal tracking, then prints a formatted summary table.

---

### Cell 18 — Interactive testing

**What it does:** Provides a simple loop for asking freeform questions not in the evaluation set.

**Tip:** Try edge-case questions like "What is the airflow.cfg file?" — if the retrieval misses these, they're candidates for your evaluation dataset.

---

### Cell 19 — Cleanup

**What it does:** Deletes the Vector Search endpoint and drops Unity Catalog tables.

> ⚠️ **Important:** This is intentionally commented out. Uncomment only when you are fully done with the project. The cleanup is permanent.

---

## 5. V2 Notebook Walkthrough

### `databricks_rag_mosaic_ai_v2.ipynb`

V2 takes the working chain from V1 and adds the Mosaic AI production layer — model versioning, a live REST API, a shareable demo UI, and quality scoring.

---

### Cell 1 — Install Mosaic AI packages

**What it does:** Installs `databricks-agents` (for `agents.deploy()`) and `mlflow[databricks]` (for Agent Evaluation).

---

### Cell 2 — Set variables + configure MLflow registry

**What it does:** Configures MLflow to use Unity Catalog as the model registry with `mlflow.set_registry_uri("databricks-uc")`.

**Key concept — why Unity Catalog for model registry?**  
The default MLflow registry stores models in a workspace-local store. Unity Catalog extends this with governance — the same access controls that protect your data tables also protect your deployed model versions.

---

### Cell 3 — Create MLflow Experiment

**What it does:** Creates a named MLflow experiment to group all model iterations for this project.

**Key concept — MLflow Experiment:**  
An experiment is like a folder for all your model runs. Every time you log a model (Cell 7), it creates a new Run inside the experiment with a unique run ID. You can compare runs side-by-side to see which prompt or configuration produced the best scores.

---

### Cell 4 — Rebuild the chain

**What it does:** Reconstructs the LangChain retriever and QA chain after the kernel restart from Cell 1.

**Why rebuild?** Python kernel restarts (Cell 1) clear all in-memory objects. The Vector Search index and Delta Table persist on disk — we just need to reconnect to them.

---

### Cell 5 — Wrap chain in MLflow ChatModel

**What it does:** Writes the `RAGChainModel` class to a Python file that MLflow will package with the model artefact.

**Key concept — `mlflow.pyfunc.ChatModel`:**  
This base class makes your chain behave like an OpenAI Chat API endpoint. Any client that can call OpenAI's API (SDKs, LangChain, curl) can call your endpoint without any changes.

**Why write to a file?** MLflow needs a file path (not an in-memory class) so it can package and reproduce the exact model in any environment — your notebook, the serving endpoint, or a colleague's workspace.

---

### Cell 6 — Local test before logging

**What it does:** Tests the `RAGChainModel.predict()` method in the notebook before committing to a full MLflow run.

**Best practice:** Always run a local test first. MLflow logging takes 60-90 seconds — catching a Python import error before logging saves time.

---

### Cell 7 — Log to MLflow

**What it does:** Opens an MLflow run and logs the model file with its dependencies and resource declarations.

**Key concept — `resources` parameter:**  
Declaring the Vector Search index and LLM endpoint as resources allows `agents.deploy()` to automatically configure the serving endpoint's IAM permissions. Without this, the endpoint would be blocked from calling the Vector Search API.

---

### Cell 8 — Register in Unity Catalog

**What it does:** Promotes the logged model artefact from the MLflow run into the Unity Catalog model registry as Version 1.

**What version numbers give you:**
- `Version 1` — initial chain
- `Version 2` — improved prompt (Cell 17)
- `Version 3` — larger retrieval k, etc.

Each version is independently deployable. You can roll back by deploying an older version number.

---

### Cell 9 — Deploy with `agents.deploy()`

**What it does:** The Mosaic AI one-liner that provisions the full production stack:

1. A serverless Model Serving endpoint with scale-to-zero
2. A `POST /invocations` REST API endpoint
3. A Review App — shareable chat UI (no Databricks login needed)
4. Auto-configured IAM permissions for the declared resources

**Key concept — scale-to-zero:**  
Unlike the Vector Search endpoint (always-on, ~$0.28/hr), the Model Serving endpoint scales to zero compute when there are no requests. You only pay for actual inference time.

---

### Cell 10 — Wait for READY

**What it does:** Polls the endpoint status every 30 seconds. First deployment takes 5–8 minutes because Databricks is provisioning the serverless container.

**Subsequent deployments are faster** (~2 min) because the container image is cached.

---

### Cell 11 — Test the live REST API

**What it does:** Calls the deployed endpoint using `mlflow.deployments.get_deploy_client()` — the same client an external application would use.

**This is the portfolio money shot:** the RAG system built from scratch in V1 is now a live REST API. Any web application, mobile app, or data pipeline can call it.

---

### Cell 12 — MLflow Tracing

**What it does:** Wraps the RAG chain with `@mlflow.trace` to record every step of every request.

**Key concept — why tracing matters:**  
When the chain gives a wrong answer, you need to know whether the problem is:
- **Retrieval failure:** the wrong chunks were returned (check the RETRIEVER span)
- **Generation failure:** good chunks were returned but the LLM ignored them (check the LLM span)

Without tracing, distinguishing between these two failure modes requires guesswork.

**View traces:** Experiments → airflow_rag_v2 → Traces tab.

---

### Cell 13 — Create evaluation dataset

**What it does:** Defines 5 questions with human-verified correct answers (ground truth).

**Why 5 pairs is enough for a demo:**  
In production, you'd want 50–200 pairs. For a portfolio demonstration of the evaluation pattern, 5 pairs is sufficient to show the methodology without requiring domain expert input.

**Customise:** Replace the QA pairs with domain-specific questions for your own knowledge base.

---

### Cell 14 — Run Agent Evaluation

**What it does:** Runs each question through the chain and scores the output using a groundedness scorer.

**The scoring function:** Computes word-overlap between the chain answer and the expected answer, then scales it to a 1–5 integer. In production, Mosaic AI provides LLM judges (requiring `mlflow.evaluate` with `model_type='databricks-agent'`) that score groundedness, relevance, and retrieval precision independently.

---

### Cell 15 — View evaluation results

**What it does:** Prints a formatted scorecard for each question.

**How to use these results:**
- Score 4–5: good, no action needed
- Score 2–3: investigate — was the right chunk retrieved? Was the answer too short?
- Score 1: retrieval failure — check the Vector Search index for this document type

---

### Cell 16 — Explore the Review App

**What it does:** Prints the Review App URL and instructions for using it.

**Key concept — Review App:**  
The Review App is a polished chat UI created automatically by `agents.deploy()`. Sharing it with a hiring manager, client, or colleague is more impactful than showing a notebook — they interact with the actual deployed agent without needing Databricks access.

---

### Cell 17 — Improve and re-deploy

**What it does:** Demonstrates the RAG improvement cycle:

```
Evaluate V1 → identify weak questions → improve prompt → log V2 → evaluate V2 → compare
```

The improved prompt adds expert-level instructions: use specific operator names, include code examples, structure multi-step answers with bullet points.

**To register V2:** After validating the improved answers, set `qa_chain = qa_chain_v2` and re-run Cells 5 → 7 → 8.

---

### Cell 18 — Cleanup

**What it does:** Deletes the Model Serving endpoint (commented out for safety).

> ⚠️ **Important:** The cell is intentionally commented out. Uncomment only when finished.

---

### Cell 19 — Portfolio summary

**What it does:** Prints a complete list of all skills demonstrated by V1 + V2 combined. Use this as a reference for your CV or interview talking points.

---

## 6. Key Concepts Glossary

| Term | Plain-English definition |
|---|---|
| **Embedding** | A list of numbers (vector) that represents the meaning of a text passage. Similar passages have similar vectors. |
| **Vector Search** | A database that finds the most semantically similar vectors to a query vector — like a "meaning search" rather than keyword search. |
| **HNSW** | Hierarchical Navigable Small World — the indexing algorithm used by Vector Search for fast approximate nearest-neighbour lookup. |
| **Chunk** | A short passage of text (200-1000 characters) split from a longer document for embedding. |
| **Retriever** | The component that takes a query, embeds it, and returns the top-k most similar chunks from the vector store. |
| **Grounding** | Constraining the LLM to answer using only retrieved context — the main technique for preventing hallucination in RAG. |
| **Delta Sync** | Automatic incremental sync from a Delta Table to a Vector Search index using Change Data Feed. |
| **CDF** | Change Data Feed — Delta Lake feature that tracks row-level changes (insert/update/delete) for efficient downstream sync. |
| **Unity Catalog** | Databricks' centralised data governance layer — manages access to tables, models, and files with a single permission model. |
| **MLflow Run** | A logged experiment iteration with artefacts (model files), metrics (evaluation scores), and parameters (chunk size, k value). |
| **PyFunc** | MLflow's generic Python model format — wraps any Python class as a deployable model with `predict()` method. |
| **ChatModel** | MLflow base class that produces OpenAI-compatible Chat API responses — enables plug-and-play with any OpenAI client. |
| **agents.deploy()** | Mosaic AI one-liner that provisions a serving endpoint + Review App from a registered model version. |
| **Scale-to-zero** | Serving endpoint feature that reduces compute to zero when idle — no cost when the API is not being called. |
| **Groundedness score** | Evaluation metric measuring whether the answer is supported by the retrieved context rather than hallucinated. |
| **LLM Judge** | An LLM used to evaluate the output of another LLM — common in RAG evaluation because human annotation is slow and expensive. |

---

## 7. Troubleshooting

| Error | Likely cause | Fix |
|---|---|---|
| `PermissionDenied: CREATE CATALOG` | Databricks account lacks catalog creation privileges | Ask workspace admin for `CREATE CATALOG` grant |
| `VectorSearchClientException: Endpoint not found` | Vector Search endpoint not yet ONLINE | Wait for Cell 8 polling to show ONLINE |
| `Index not found` | V1 not run before V2 | Run `databricks_rag_vector_search.ipynb` fully first |
| `ModuleNotFoundError` after kernel restart | Package install didn't complete | Re-run Cell 1 and wait for the full restart |
| `NameError: qa_chain is not defined` | Kernel was restarted and Cell 4 not re-run | Re-run Cell 4 to rebuild the chain |
| Serving endpoint stays `PROVISIONING` > 15 min | Transient Databricks infrastructure issue | Delete the endpoint, wait 5 min, re-run Cell 9 |
| Low groundedness scores (<3) | Wrong chunks retrieved, or prompt too permissive | Print the retrieved chunks for the failing question; increase k or adjust the prompt |

---

## 8. Next Steps

Once comfortable with this project, consider these production enhancements:

**Better retrieval:**
- Hybrid search (BM25 keyword + vector) — catches exact-match queries that semantic search misses
- Metadata filtering — restrict search to a specific document type or date range
- Re-ranker model — a second-stage model that reorders retrieved chunks by relevance

**Better generation:**
- Streaming responses — stream tokens to the UI for perceived lower latency
- Citation injection — automatically append source links to every answer
- Multi-turn conversation — pass conversation history to the LLM for contextual follow-ups

**Production MLOps:**
- A/B testing — deploy two model versions to the same endpoint, split traffic 80/20
- Automated evaluation pipeline — trigger re-evaluation on every model update using Databricks Workflows
- Feedback loop — capture thumbs up/down from Review App and add weak questions to the evaluation dataset

**Scale:**
- Databricks Workflows for scheduled incremental re-indexing
- Auto Loader for continuous document ingestion
- Unity Catalog data lineage — track which source documents produced which evaluation results

---

*Built and documented by Ravi Amaraweera — Senior Data Architect / Analytics Engineer*
