# Getting Started: Databricks RAG with Mosaic AI (V2)

> **Written for:** Beginners to Databricks and MLOps  
> **Prerequisite:** Have run `databricks_rag_vector_search.ipynb` (V1) successfully  
> **Time to read:** ~20 minutes  
> **Time to run:** ~30 minutes

---

## Before You Begin: What You Need to Know

### What is MLOps?
MLOps (Machine Learning Operations) is the discipline of treating AI models like  
software products — with versioning, testing, deployment, and monitoring.  

Without MLOps:  
- Your chain lives only inside a notebook
- Nobody outside Databricks can use it
- You cannot track whether your changes improved or degraded quality
- There is no rollback if something breaks

With MLOps (what this notebook adds):  
- The chain is a versioned product registered in a catalogue
- Anyone can call it via a REST API
- Every iteration is tracked and comparable
- You can roll back to a previous version at any time

### What is MLflow?
MLflow is an open-source platform for managing the machine learning lifecycle.  
Think of it as version control + packaging + deployment for AI models.

Key concepts you will encounter:
- **Experiment**: a folder that groups all runs for a project
- **Run**: one execution that produces a logged model, metrics, and metadata
- **Model URI**: a unique address pointing to a specific logged model
- **Registry**: a catalogue of models with version numbers

### What is Unity Catalog?
Unity Catalog is Databricks' centralised governance layer.  
It manages databases, tables, volumes, *and* AI models — all in one place with:
- Fine-grained access control (who can see/use each asset)
- Data lineage (where did this data come from?)
- Model versioning (Version 1, 2, 3 … just like a package)

Think of it as the address book for everything in your Databricks workspace.

### What is Mosaic AI?
Mosaic AI is Databricks' umbrella brand for its AI engineering tools:
- **Foundation Model APIs**: hosted LLMs and embedding models (no GPU setup needed)
- **Vector Search**: serverless semantic search over your Delta Tables
- **Agent Framework** (`databricks-agents`): tooling to deploy, evaluate, and monitor RAG chains
- **Model Serving**: serverless HTTP endpoints for any MLflow model

---

## The Big Picture: What This Notebook Does

```
Your LangChain QA chain (built in V1)
         │
         ▼
Wrap it in a standard MLflow "box"        ← Cell 5
         │
         ▼
Test the box locally                      ← Cell 6
         │
         ▼
Log the box to MLflow (create a Run)      ← Cell 7
         │
         ▼
Register it in Unity Catalog (Version N)  ← Cell 8
         │
         ▼
Deploy → live REST endpoint + Review App  ← Cell 9
         │
         ├── Any app calls the API         ← Cell 11
         ├── MLflow records every call     ← Cell 12
         └── LLM judges score quality      ← Cell 14
```

---

## Cell-by-Cell Walkthrough

---

### Cell 1 — Install Mosaic AI Packages

**What it does:** Installs two packages not present in V1:
- `databricks-agents`: provides `agents.deploy()` — the one-liner that deploys your chain
- `mlflow[databricks]`: adds Databricks-specific tracing and evaluation on top of base MLflow

After installation, the kernel (Python process) restarts automatically.  
**This is expected.** All variables from before the restart are gone. Start from Cell 2.

---

### Cell 2 — Set Shared Variables

**What it does:** Defines all names in one place so you only need to change them here.

```python
CATALOG      = "rag_portfolio"
SCHEMA       = "doc_search"
MODEL_NAME   = f"{CATALOG}.{SCHEMA}.airflow_rag_agent"
VS_ENDPOINT  = "rag-portfolio-endpoint"
VS_INDEX     = f"{CATALOG}.{SCHEMA}.airflow_docs_index"
LLM_ENDPOINT = "databricks-gpt-oss-20b"
EMBED_MODEL  = "databricks-bge-large-en"
```

**Key concept — Unity Catalog three-level naming:**  
Everything in Databricks follows the pattern `CATALOG.SCHEMA.object_name`.  
`MODEL_NAME = "rag_portfolio.doc_search.airflow_rag_agent"` means:  
- Catalog: `rag_portfolio` (your project's top-level namespace)  
- Schema: `doc_search` (like a folder inside the catalog)  
- Model name: `airflow_rag_agent` (the specific asset)

**`mlflow.set_registry_uri("databricks-uc")`**: this line tells MLflow to use  
Unity Catalog as the model registry, instead of the legacy MLflow registry.  
Without this, `mlflow.register_model()` in Cell 8 would fail.

---

### Cell 3 — Create an MLflow Experiment

**What it does:** Creates a named folder (Experiment) in MLflow for this project.

**Analogy:** An experiment is like a Git repository for model runs.  
Each run is a commit — it captures what you did and what result you got.

**Where to view it:** Left sidebar → Experiments → `airflow_rag_v2`

You will see columns like: Run name, Date, Duration, and any logged metrics.  
As you iterate (V1 chain → improved prompt → V2 chain), each becomes a new row here.

---

### Cell 4 — Reconnect to Vector Search and Rebuild the Chain

**What it does:** Re-creates the Python objects that the kernel restart in Cell 1 erased.

**Why do we need to do this again?**  
A Python kernel restart clears all in-memory variables.  
The `llm`, `retriever`, and `qa_chain` objects from V1 are gone.  
This cell creates identical objects using the same configuration.

**New class — `SimpleRetrievalQA`:**  
LangChain 1.x removed the old `RetrievalQA` chain class.  
`SimpleRetrievalQA` is a lightweight replacement we wrote ourselves.  
Pipeline:
1. `retriever.invoke(query)` → returns top-4 most similar document chunks
2. `prompt.format(context=..., question=...)` → builds the full prompt string
3. `llm.invoke(prompt)` → sends prompt to LLM and gets a response

**Key parameter — `temperature=0`:**  
Setting temperature to 0 makes the LLM deterministic — it always picks the most  
probable token rather than sampling. This is best practice for factual Q&A where  
you want consistent, repeatable answers rather than creative variation.

---

### Cell 5 — Wrap the Chain in an MLflow PyFunc Model Class

**What it does:** Writes a Python file (`/tmp/rag_chain_model.py`) containing the  
`RAGChainModel` class. This file is the "shipping box" MLflow uses to package and deploy your chain.

**Why a file and not a class instance?**  
MLflow serialises models for portability. A Python class instance might reference  
objects that don't exist in the serving environment. A Python *file* that reconstructs  
everything from scratch (in `load_context`) is fully portable.

**`RAGChainModel` methods:**

| Method | When called | What it does |
|---|---|---|
| `load_context(context)` | Once at endpoint startup | Builds the chain from scratch using stored config |
| `predict(context, messages, params)` | On every request | Extracts the user question, runs the chain, returns an answer |

**Why inherit from `ChatModel`?**  
`mlflow.pyfunc.ChatModel` is a subclass of `mlflow.pyfunc.PythonModel` that adds  
an OpenAI-compatible interface. The serving endpoint will automatically accept and  
return messages in the same format as OpenAI's Chat Completions API.

**`mlflow.models.set_model(RAGChainModel)`:**  
This line at the bottom of the file tells MLflow which class is the entry point.  
Without it, MLflow does not know which class to instantiate when loading the model.

---

### Cell 6 — Test the Wrapper Locally Before Logging

**What it does:** Calls `predict()` directly in the notebook before committing to a full MLflow run.

**Why test first?**  
Logging a model to MLflow (Cell 7) takes 1–3 minutes and uses cluster compute.  
Running a quick local test here catches errors in 5 seconds — before you waste time  
on a failed logging run.

If this cell produces a coherent Airflow answer, the wrapper is correct.  
If it raises an error, fix the issue in Cell 5 before proceeding.

---

### Cell 7 — Log the Model to MLflow

**What it does:** Opens an MLflow run and packages the model with all its dependencies.

**Breaking down `mlflow.pyfunc.log_model()`:**

```python
mlflow.pyfunc.log_model(
    artifact_path="rag_chain",          # folder name inside the run artefacts
    python_model="/tmp/rag_chain_model.py", # the file (not a class!)
    resources=[...],                    # declares external service dependencies
    pip_requirements=[...],             # exact packages the serving env needs
    input_example={...},               # sample request → auto-generates API schema
)
```

**`resources` — why is this important?**  
When `agents.deploy()` creates a serving endpoint, it needs to know which Databricks  
services the model depends on so it can automatically configure IAM permissions.  
Declaring `DatabricksVectorSearchIndex` and `DatabricksServingEndpoint` here means  
the endpoint gets access to those services automatically — no credentials needed in code.

**`input_example` — why include it?**  
MLflow uses the input example to infer the model's input/output schema.  
The serving endpoint uses this schema to generate its interactive API documentation.

After this cell, look in the Experiments tab — you will see a new Run with a green checkmark.

---

### Cell 8 — Register the Model in Unity Catalog

**What it does:** Takes the logged model from Cell 7 and registers it in Unity Catalog  
with a version number. Each call adds Version N+1.

**Difference between logging and registering:**

| | Logging (Cell 7) | Registering (Cell 8) |
|---|---|---|
| Where stored | MLflow experiment run (ephemeral) | Unity Catalog (permanent, governed) |
| Versioned? | By run ID only | Version 1, 2, 3 … (human-readable) |
| Access control? | Experiment-level | Unity Catalog RBAC |
| Can deploy from? | No | Yes (`agents.deploy()` requires a version) |

**Where to view it:**  
Catalog (left sidebar) → rag_portfolio → doc_search → airflow_rag_agent → Versions tab

---

### Cell 9 — Deploy to a Model Serving Endpoint

**What it does:** Calls `agents.deploy()` — Mosaic AI's one-liner that does three things:

1. **Creates a serverless Model Serving endpoint**  
   This is a live HTTPS endpoint that runs your `RAGChainModel.predict()` method  
   on every POST request. No server management needed — Databricks handles scaling.

2. **Enables scale-to-zero**  
   Unlike the Vector Search endpoint (always-on, ~$0.28/hr), the serving endpoint  
   drops to zero compute when no requests are coming in. You pay only during active inference.

3. **Creates a Review App**  
   A shareable chat UI at a unique URL. Recipients do not need a Databricks login.  
   This is the "show don't tell" feature for portfolio and client demos.

**Wait time:** Cell 10 polls every 30 seconds until the endpoint shows `READY`.  
Typical wait: 5–10 minutes for a new endpoint.

---

### Cell 10 — Wait for the Endpoint to Be Ready

**What it does:** Polls `w.serving_endpoints.get()` every 30 seconds.

You can also watch progress in the UI:  
Serving tab (left sidebar) → your endpoint name → Status column

The endpoint goes through states: `NOT_READY → UPDATING → READY`

---

### Cell 11 — Call the Live REST API

**What it does:** Queries the deployed endpoint using `mlflow.deployments.get_deploy_client()`.

**Why is this significant for the portfolio?**  
This is the same client call an external application would make.  
The chain is no longer a notebook experiment — it is a real service.

Any of these can now call it:
- A web app built in React or FastAPI
- Another Databricks job
- An Apache Airflow DAG
- A Slack bot or Teams integration

---

### Cell 12 — Add MLflow Tracing

**What it does:** Wraps the RAG function with `@mlflow.trace` so every call is recorded.

**Why tracing matters:**  
When the chain gives a wrong answer, you need to know *why*.  
Was it a retrieval problem (wrong chunks returned)?  
Was it a generation problem (good chunks, but LLM misread them)?

Without tracing: you see only the wrong answer.  
With tracing: you see every step.

**What each span captures:**

| Span | Contents |
|---|---|
| `RETRIEVER` | Query text, how many chunks returned, source file names |
| `LLM` | Query text, how many context documents provided, answer length |

**Where to view traces:**  
Experiments → airflow_rag_v2 → Traces tab  
Click any trace to expand it into a tree of spans with timestamps and durations.

---

### Cell 13 — Create the Evaluation Dataset

**What it does:** Creates a small "golden dataset" — questions paired with known-correct answers.

**What is a golden dataset?**  
In evaluation, a golden dataset is a set of inputs where the correct answer is  
already known (verified by a domain expert). The evaluation framework compares  
the chain's actual answers against these expected answers using automated judges.

**Why only 5 pairs?**  
5 pairs is enough to demonstrate the evaluation pattern. In production you would  
use 50–200 pairs for statistically meaningful quality scores.

---

### Cell 14 — Run Agent Evaluation with LLM Judges

**What it does:** Scores the chain's answers against the golden dataset.

**What the groundedness score means:**  
Groundedness measures whether the answer is supported by the retrieved context —  
i.e., the chain is not hallucinating information from outside the knowledge base.

| Score | Interpretation |
|---|---|
| 5 | Fully grounded: every claim in the answer is supported by the context |
| 4 | Mostly grounded: minor gaps but acceptable for production |
| 3 | Partially grounded: some claims lack context support |
| 1–2 | Low groundedness: likely hallucinating or retrieved wrong chunks |

**What to do with a low score:**
- Score 1–2 on most questions → retriever issue: try increasing `k` (more chunks) or re-check your V1 index
- Score 3 on specific questions → prompt issue: add more specific instructions for that question type
- Score 4–5 → chain is performing well; try harder questions

---

### Cell 15 — View and Understand the Evaluation Results

**What it does:** Prints a formatted summary of per-question scores with a visual bar.

**How to read the output:**
```
Q: How do I set task dependencies in Airflow?...
   Score  : [****-] 4/5
   Source : best-practices.html
```

The bar shows filled (`*`) vs empty (`-`) segments out of 5.  
The source shows which document file contributed to the answer.  
If the source looks wrong (e.g., a monitoring doc for a DAG design question), that  
explains a low score — the retriever fetched an irrelevant chunk.

---

### Cell 16 — Explore the Review App

**What it does:** Prints the Review App URL from Cell 9 and explains how to use it.

**What the Review App looks like:**  
A chat interface similar to ChatGPT or Claude.  
On the right panel, it shows the source document chunks used to generate the answer.  
This "show your work" feature builds trust with non-technical stakeholders.

**Portfolio tip:** Take a screenshot of the Review App answering a question correctly.  
Add it to your GitHub README under a "Live Demo" section. This is the most impactful  
visual you can add to a portfolio project.

---

### Cell 17 — Version 2: Improve and Re-deploy

**What it does:** Demonstrates the iterative improvement cycle.

**The MLOps cycle in practice:**
```
Evaluate V1 → identify weak questions → improve prompt → run Cells 5→7→8 → evaluate V2 → compare
```

**What changed in the V2 prompt:**
- Added: "Mention specific Airflow operators, parameters, or code patterns when relevant"
- Added: "Use bullet points for multi-step answers"

These instructions guide the LLM to be more precise and actionable — which improves  
groundedness scores for questions about specific Airflow features.

After validating the improved answers, re-run Cells 5 → 7 → 8 to register Version 2  
in Unity Catalog alongside Version 1.

---

### Cell 18 — Cleanup

**What it does:** Provides commented-out code to delete the serving endpoint.

**When to run it:**  
After your demo session is complete and you no longer need the endpoint.

**What you do NOT need to delete:**  
- Delta Tables (no ongoing cost)
- Unity Catalog model registry entries (no ongoing cost)
- MLflow experiment runs (no ongoing cost)

**What you DO need to delete to stop charges:**  
- Vector Search endpoint (V1, Cell 19) — ~$0.28/hr always-on
- Model Serving endpoint — this one (scale-to-zero, effectively free when idle)

---

### Cell 19 — Portfolio Summary

**What it does:** Prints a summary of the complete V1 + V2 skill set.

**How to use this in interviews:**  
When asked "walk me through a project", this output gives you a structured answer  
covering both data engineering (V1) and AI engineering/MLOps (V2).

---

## Concepts Glossary

| Term | Plain-English meaning |
|---|---|
| **MLflow Run** | One experiment iteration — a snapshot of a model and its metadata |
| **Model URI** | The unique address of a logged model, e.g. `models:/rag_portfolio...` |
| **PyFunc** | MLflow's generic model format — wraps any Python object into a deployable model |
| **ChatModel** | A PyFunc subclass with an OpenAI-compatible chat interface |
| **`load_context()`** | The method called once when a serving endpoint loads a model — like `__init__` |
| **`predict()`** | The method called on every inference request — like a web handler |
| **scale-to-zero** | The serving endpoint stops consuming compute when idle — zero cost when unused |
| **Review App** | A shareable chat UI Databricks creates alongside a deployed agent endpoint |
| **Groundedness** | A quality measure: is the answer supported by the retrieved context, or hallucinated? |
| **LLM judge** | An LLM used to automatically score another LLM's outputs |
| **Resources declaration** | Listing the Databricks services a model depends on, so IAM permissions are auto-configured |
| **Span** | One step inside a traced function call (e.g., RETRIEVER span, LLM span) |

---

## Troubleshooting Common Errors

| Error | What it means | Fix |
|---|---|---|
| `ModuleNotFoundError: rag_chain_model` | Cell 5 not yet run in this session | Re-run Cell 5 |
| `ENDPOINT_NOT_FOUND` | V1 Vector Search endpoint was deleted | Re-run Cells 7–10 in `databricks_rag_vector_search.ipynb` |
| Kernel restarted unexpectedly after Cell 1 | Normal behaviour — `restartPython()` is intentional | Start from Cell 2 |
| `CreateModelVersion` permission denied | Missing `CREATE MODEL` privilege | Add via: Catalog → rag_portfolio → Permissions |
| `agents.deploy()` returns immediately but endpoint is NOT_READY | Provisioning takes time | Cell 10 handles this — just wait |
| Review App returns 404 | Endpoint still provisioning | Wait until Cell 10 prints `READY` |
| Evaluation scores all 1–2 | Wrong chunks being retrieved | Check VS index is ONLINE in Vector Search UI |
| `FutureWarning: ChatModel deprecated` | MLflow 3.x renamed the API | Safe to ignore for now; use `ResponsesAgent` for MLflow 3+ projects |

---

## What to Explore Next

Once this notebook is running, here are natural next steps to extend the project:

1. **Add more data sources** — ingest dbt docs, Spark docs, or your own internal knowledge base alongside Airflow docs
2. **Improve chunking strategy** — experiment with different `chunk_size` and `chunk_overlap` values in V1 and measure the impact on evaluation scores
3. **Use Mosaic AI's built-in LLM judges** — replace the word-overlap scorer with `mlflow.evaluate()` using `model_type='databricks-agent'` for production-grade scoring
4. **Add production monitoring** — use `agents.enable_trace_reviews()` to collect user thumbs up/down feedback through the Review App
5. **Multi-turn conversation** — extend `RAGChainModel.predict()` to pass full conversation history to the LLM rather than just the last message
6. **CI/CD pipeline** — add a Databricks Job that re-runs evaluation automatically whenever new documents are added to the Delta Table

---

*Built by Ravi Amaraweera — Senior Data Architect / Analytics Engineer*
