# Databricks Architecture Designer Agent — Knowledge Base Ingestion Guide

This document explains how the knowledge base ingestion pipeline works for the Databricks Architecture Designer Agent. It covers the sources, extraction strategies, and step-by-step instructions to run and extend the pipeline.

---

## Overview

The pipeline ingests documentation from three source types into a single Delta table (`{catalog}.{schema}.databricks_knowledge_base`):

1. **Solution Accelerators** — Databricks industry accelerator repos (GitHub), `source_type = "accelerator"`
2. **OSS Docs** — Delta Lake, Spark, MLflow, Databricks docs (official sites & GitHub repos), `source_type = "oss_docs"`

The pipeline is **incremental (append-only)**: on each run it loads existing `doc_id` values from the table and skips any document already present, so only new documents are fetched and written.

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

## Incremental Logic

```
Run 1 (first time): table doesn't exist → EXISTING_DOC_IDS = {} → fetch all → append all
Run 2+:             load doc_ids from table → skip existing → fetch only new → append new
```

`doc_id` is a deterministic MD5 hash of a stable key (URL or repo+path), so the same document always produces the same `doc_id` across runs.

---

## Full Refresh

To wipe the table and re-ingest everything:

1. Open the notebook in Databricks
2. Find the **"Optional: full refresh"** cell
3. Uncomment both lines and run that cell alone:
   ```python
   spark.sql(f"TRUNCATE TABLE {FULL_TABLE}")
   EXISTING_DOC_IDS.clear()
   ```
4. Then run the rest of the notebook normally

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
- `notebooks/1.3_databricks_knowledge_ingestion.py` — Main pipeline notebook
- `resources/arxiv_data_ingestion_job.yml` — Example Databricks job definition
- `project_config.yml` — Catalog/schema config

---

## Troubleshooting
- **Doc returns `✗`:** site may be JavaScript-rendered or blocking bots — try adding it to `_DATABRICKS_DOCS_URLS` with browser headers fallback already built in
- **GitHub traversal returns 0 docs:** check `branch` and `scan_dirs` are correct; branches tried in order: configured → `master` → `main`
- **Rate limit warnings:** add a GitHub token via `dbutils.secrets` (scope `llmops_course`, key `github_token`) to raise limit from 60 to 5,000 req/hr

---

For questions or to extend the pipeline, see comments in the notebook or contact the project maintainer.
