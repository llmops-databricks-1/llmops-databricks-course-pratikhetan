# Databricks Architecture Designer Agent — Knowledge Base Pipeline Guide

This document explains the full knowledge base pipeline for the Databricks Architecture Designer Agent: ingestion, chunking, and vector search indexing.

---

## Overview

The pipeline runs in two stages managed by `resources/kb_pipeline_job.yml`:

```
Stage 1 — Ingestion  (notebooks/1.3_databricks_knowledge_ingestion.py)
   Sources: GitHub accelerators + OSS docs sites + GitHub repos
       ↓ append-only
   Delta table: {catalog}.{schema}.databricks_knowledge_base

Stage 2 — Chunk + VS Sync  (notebooks/run_chunk_and_sync.py)
   KBProcessor.chunk_and_save() — two-stage Markdown-aware chunking
       ↓ append-only, CDF enabled
   Delta table: {catalog}.{schema}.kb_chunks
       ↓ Delta Sync Index (TRIGGERED)
   Databricks Vector Search Index: {catalog}.{schema}.kb_chunks_index
```

### Ingestion source types

1. **Solution Accelerators** — Databricks industry accelerator repos (GitHub), `source_type = "accelerator"`
2. **OSS Docs** — Delta Lake, Spark, MLflow, Databricks docs (official sites & GitHub repos), `source_type = "oss_docs"`

Both stages are **incremental (append-only)**: each run skips documents already present in the respective table.

---

## Step-by-Step Ingestion Process

### 1. Solution Accelerators (Section 1)
- **Source:** All public repos in `databricks-industry-solutions` GitHub org
- **Filters:** not archived, not a fork, pushed within 4 years, README word count ≥ 100
- **How:**
  - List all repos via GitHub API (paginated, 100/page)
  - For each repo, fetch README (`README.md` / `readme.md` / `Readme.md`, tries `main` then `master`)
  - Skip if `doc_id` already in table (incremental)
- **`doc_id` key:** `acc:{repo_name}`
- **Output:** `source_type = "accelerator"`

### 2. OSS Docs (Section 2)

#### Strategy A — Direct URL fetch
Uses `trafilatura` with a browser-header fallback for sites that block bots.

| Source | Pages | Notes |
|---|---|---|
| Delta Lake | 12 pages | `docs.delta.io` — Sphinx HTML |
| Apache Spark | 16 pages | `spark.apache.org` — server-rendered |
| MLflow | 16 Python API pages | `mlflow.org/docs/latest/python_api/` — static Sphinx |
| Databricks Docs | 28 pages | `docs.databricks.com` — Delta, ML, data engineering, Unity Catalog, compute |

- **`doc_id` key:** `oss:{url}`
- **Output:** `source_type = "oss_docs"`

#### Strategy B — GitHub traversal
Traverses GitHub repos for narrative Markdown/RST docs, filtered by filename keywords.

| Repo | Branch | Scan dirs | Focus |
|---|---|---|---|
| `mlflow/mlflow` | `master` | `docs/`, `docs/source/` | Tracking, registry, serving, LLM integrations |
| `databricks/databricks-sdk-py` | `main` | `docs/` | Jobs, clusters, model serving, vector search |
| `databricks/genai-cookbook` | `main` | `rag_app_sample_code/`, `agent_app_sample_code/`, `.` | RAG patterns, agent architecture, chunking, eval |
| `databricks/mlops-stacks` | `main` | `.` | MLOps reference architecture, CI/CD, project structure |
| `databricks/databricks-ml-examples` | `master` | `.` | LLM fine-tuning, inference, RAG, and serving patterns |

- **`doc_id` key:** `oss:{repo}:{path}`
- **Output:** `source_type = "oss_docs"`

### 3. Write to Delta Table (Section 3)
- Combines `acc_docs + oss_docs`
- If nothing new: logs "up to date" and skips write
- If new docs exist: appends to Delta table with `.mode("append")`

---

## Stage 2 — Chunking (`KBProcessor`)

`src/arch_designer_agent/kb_processor.py` handles chunking via `KBProcessor.chunk_and_save()`.

### Incremental logic
```
Run 1: kb_chunks doesn't exist → chunk all docs → create table + enable CDF
Run 2+: load source_doc_ids already in kb_chunks → skip those → chunk only new docs → append
```

### Two-stage Markdown-aware splitting

| Stage | Splitter | Triggered when |
|---|---|---|
| 1A | `MarkdownHeaderTextSplitter` | Doc has `#` headers in first 50 lines |
| 1B | `RecursiveCharacterTextSplitter` | Plain text / RST (no headers) |
| 2 | `RecursiveCharacterTextSplitter` (1500 chars, 150 overlap) | Always (sub-splits Stage 1A sections) |

`section_header` is built as a breadcrumb from the heading hierarchy: `h1 > h2 > h3`.

`chunk_id` is a deterministic MD5 of `"{doc_id}:chunk:{index}"` — same content always produces the same ID.

Chunks with fewer than 20 words are dropped.

---

## Stage 3 — Vector Search Sync (`VectorSearchManager`)

`src/arch_designer_agent/vector_search.py` manages the VS endpoint and index.

### Index type: Delta Sync
`create_delta_sync_index` — Databricks reads `kb_chunks` via Change Data Feed (CDF), calls the embedding model, and updates the vector index. You do **not** pre-compute embeddings.

The alternative (`create_direct_access_index`) requires you to push pre-computed vectors manually — not used here.

### `pipeline_type = "TRIGGERED"`
The index only syncs when `index.sync()` is explicitly called (end of every job run). The alternative `"CONTINUOUS"` auto-syncs within minutes but runs always-on compute — more expensive and unnecessary for a batch pipeline.

### Sync flow in `sync_index()`
```
create_or_get_index()        # creates endpoint + index if missing
_wait_for_index_online()     # polls describe() every 20s until state = ONLINE
index.sync()                 # reads CDF delta, embeds only new/changed chunks
```
The wait step prevents the `400 Bad Request` that occurs when calling `.sync()` on a newly-created index still in `PROVISIONING` state.

---

## Incremental Logic (Stage 1)

```
Run 1 (first time): table doesn't exist → EXISTING_DOC_IDS = {} → fetch all → append all
Run 2+:             load doc_ids from table → skip existing → fetch only new → append new
```

`doc_id` is a deterministic MD5 hash of a stable key (URL or repo+path), so the same document always produces the same `doc_id` across runs.

---

## Full Refresh (Ingestion Table)

To wipe `databricks_knowledge_base` and re-ingest everything:

1. Open `notebooks/1.3_databricks_knowledge_ingestion.py` in Databricks
2. Find the **"Optional: full refresh"** cell
3. Set `FORCE_RESET = True` and run **that cell alone**:
   ```python
   FORCE_RESET = True   # ← change this
   # cell will TRUNCATE the table and clear EXISTING_DOC_IDS
   ```
4. **Flip it back to `False` immediately** before running the rest of the notebook

> **Why a flag and not a comment?** Commented-out code can be accidentally executed if cells run via "Run All". The `FORCE_RESET = False` default makes an accidental truncation impossible.

To also re-chunk everything, truncate `kb_chunks` separately:
```sql
TRUNCATE TABLE {catalog}.{schema}.kb_chunks
```
then re-run Stage 2 (`run_chunk_and_sync.py`).

---

## How to Run the Pipeline

1. Open `notebooks/1.3_databricks_knowledge_ingestion.py` in Databricks
2. Run all cells (or run as a Databricks job via `resources/`)
3. Monitor logs for counts and extraction status
4. Inspect the output table: `{catalog}.{schema}.databricks_knowledge_base`

---

## Extending the Pipeline

- **Add new direct URLs:** append to the relevant list (`_DELTA_URLS`, `_SPARK_URLS`, `_DATABRICKS_DOCS_URLS`, etc.) as `(title, url)` tuples
- **Add new GitHub repos:** add an entry to `_GH_SOURCES` with `branch`, `scan_dirs`, `keywords`, and `exts`
- **Adjust filters:** change `_ACC_MIN_WORDS`, `_ACC_MAX_AGE_YRS`, `_OSS_MIN_WORDS`, `_OSS_MAX_WORDS` as needed
- **Debug extraction failures:** check logs for `✗` lines; sites that are JS-rendered will return `None` from trafilatura

---

## Key Files

| File | Purpose |
|---|---|
| `notebooks/1.3_databricks_knowledge_ingestion.py` | Stage 1: ingestion notebook |
| `notebooks/run_chunk_and_sync.py` | Stage 2: chunk + VS sync job notebook |
| `notebooks/2.2_kb_chunking.py` | Teaching notebook: chunking walkthrough |
| `notebooks/2.4_embeddings_vector_search.py` | Teaching notebook: VS setup and search |
| `src/arch_designer_agent/kb_processor.py` | `KBProcessor` class — chunking logic |
| `src/arch_designer_agent/vector_search.py` | `VectorSearchManager` class — VS lifecycle |
| `resources/kb_pipeline_job.yml` | Databricks job: 2-task pipeline (ingest → chunk+sync) |
| `project_config.yml` | Catalog, schema, endpoint config |

---

## Troubleshooting

### Ingestion
- **Doc returns `✗`:** site may be JavaScript-rendered or blocking bots — try adding it to `_DATABRICKS_DOCS_URLS` (browser-header fallback is already built in)
- **GitHub traversal returns 0 docs:** check `branch` and `scan_dirs` are correct; branches tried in order: configured → `master` → `main`
- **Rate limit warnings:** add a GitHub token via `dbutils.secrets` (scope `llmops_course`, key `github_token`) to raise limit from 60 to 5,000 req/hr

### Vector Search
- **`400 Bad Request` on `.sync()`:** index was still `PROVISIONING` — `sync_index()` now waits automatically; if it still fails, increase `wait_timeout` (default 600s)
- **`TimeoutError` from `_wait_for_index_online`:** endpoint or index provisioning is taking longer than expected — check Databricks VS console for errors, then retry
- **Index shows stale data:** confirm `delta.enableChangeDataFeed = true` is set on `kb_chunks` (`SHOW TBLPROPERTIES {catalog}.{schema}.kb_chunks`)

---

For questions or to extend the pipeline, see comments in the notebooks or the source classes.
