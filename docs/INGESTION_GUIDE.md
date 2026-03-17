# Databricks Architecture Designer Agent — Knowledge Base Ingestion Guide

This document explains how the knowledge base ingestion pipeline works for the Databricks Architecture Designer Agent. It covers the sources, extraction strategies, and step-by-step instructions to run and extend the pipeline.

---

## Overview

The pipeline ingests documentation from three main sources into a single Delta table (`mlops_dev.pratikhe.databricks_knowledge_base`):

1. **Solution Accelerators** — Databricks industry accelerator repos (GitHub)
2. **OSS Docs** — Delta Lake, Spark, MLflow, Databricks SDK (official docs & GitHub)
3. **Databricks Blog** — Blog posts via RSS feeds

All data is merged into a single Delta table for RAG and search.

---

## Step-by-Step Ingestion Process

### 1. Solution Accelerators (Section 1)
- **Source:** All public repos in `databricks-industry-solutions` GitHub org
- **How:**
  - List all repos (skip archived/forks)
  - For each repo, fetch README and docs files
  - Filter by word count, last update, etc.
  - Extract text using `trafilatura`
- **Output:** List of accelerator docs with metadata

### 2. OSS Docs (Section 2)
- **Source:**
  - **Delta Lake:** 12 key pages from https://docs.delta.io/latest/
  - **Spark:** 16 key pages from https://spark.apache.org/docs/latest/
  - **MLflow:** 16 Python API pages from https://mlflow.org/docs/latest/python_api/
  - **Databricks SDK:** Sphinx docs from `databricks/databricks-sdk-py` GitHub repo
- **How:**
  - For Delta, Spark, MLflow: fetch each URL, extract text with `trafilatura` (with browser headers fallback)
  - For SDK: traverse GitHub repo, filter `.md`/`.rst` docs by keywords
- **Output:** List of OSS docs with metadata

### 3. Databricks Blog (Section 3)
- **Source:**
  - RSS feeds from https://www.databricks.com/blog and subcategories
- **How:**
  - Parse RSS, fetch each post, extract text with `trafilatura`
  - Filter by keywords, word count, and recency
- **Output:** List of blog posts with metadata

### 4. Merge and Write to Delta Table (Section 4)
- **How:**
  - Combine all docs from above
  - Compute a unique `doc_id` (MD5 hash)
  - Merge into Delta table using `MERGE` (idempotent)

---

## How to Run the Pipeline

1. Open `notebooks/1.3_databricks_knowledge_ingestion.py` in Databricks or VS Code
2. Run all cells (or run as a Databricks job)
3. Monitor logs for counts and extraction status
4. Inspect the output table: `mlops_dev.pratikhe.databricks_knowledge_base`

---

## Extending the Pipeline

- **Add new sources:**
  - For new direct URLs, add to the relevant URL list
  - For new GitHub repos, add to `_GH_SOURCES` with scan dirs/keywords
- **Adjust filters:**
  - Change min/max word count, keywords, or date filters as needed
- **Debug extraction:**
  - Use logs to see which docs fail extraction and why
  - Update `_fetch_url_doc` for new site patterns if needed

---

## Key Files
- `notebooks/1.3_databricks_knowledge_ingestion.py` — Main pipeline notebook
- `resources/databricks_knowledge_ingestion_job.yml` — Databricks job definition
- `project_config.yml` — Table/volume config

---

## Troubleshooting
- If a doc fails extraction, check if the site is JavaScript-rendered or blocks bots
- For GitHub traversal, ensure correct branch and scan_dirs
- Use browser headers and `favor_recall=True` for stubborn sites

---

For questions or to extend the pipeline, see comments in the notebook or contact the project maintainer.
