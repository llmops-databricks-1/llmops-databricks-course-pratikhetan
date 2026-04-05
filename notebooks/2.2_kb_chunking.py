# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.2: Knowledge Base Chunking
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Why we chunk Markdown/RST/web text (not PDFs)
# MAGIC - Two-stage chunking: MarkdownHeaderTextSplitter → RecursiveCharacterTextSplitter
# MAGIC - Preserving metadata through the chunking pipeline
# MAGIC - Writing chunks to Delta with Change Data Feed enabled

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Chunking Strategy for Markdown & Web Text
# MAGIC
# MAGIC The knowledge base contains **Markdown, RST, and web-extracted text** — not PDFs.
# MAGIC `ai_parse_document` (used for arXiv PDFs) does not apply here.
# MAGIC
# MAGIC ### Two-Stage Approach
# MAGIC
# MAGIC **Stage 1 — `MarkdownHeaderTextSplitter`**
# MAGIC - Splits on `#`, `##`, `###` headers
# MAGIC - Each section header becomes **chunk metadata** (not lost in the text)
# MAGIC - Preserves document hierarchy
# MAGIC
# MAGIC **Stage 2 — `RecursiveCharacterTextSplitter`**
# MAGIC - Splits with separators: `\n\n` → `\n` → ` `
# MAGIC - Keeps chunks within the embedding model's token window
# MAGIC - `chunk_size=1500` chars ≈ 375 tokens (well within the 512-token limit of
# MAGIC   `databricks-gte-large-en`)
# MAGIC - `chunk_overlap=150` chars ensures context continuity across boundaries
# MAGIC
# MAGIC Documents with no Markdown headers (web text, RST) skip Stage 1 and go
# MAGIC directly to Stage 2.
# MAGIC
# MAGIC The `KBProcessor` class in `arch_designer_agent.kb_processor` implements
# MAGIC this pipeline. The same class is used here for interactive demos and by
# MAGIC the scheduled job (`resources/deployment_scripts/sync_vs_index.py`).

# COMMAND ----------

# MAGIC %pip install ../dist/*.whl --force-reinstall --quiet

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from arch_designer_agent.config import get_env, load_config
from arch_designer_agent.kb_processor import KBProcessor

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

processor = KBProcessor(spark=spark, config=cfg)

logger.info(f"Source table : {processor.kb_table}")
logger.info(f"Output table : {processor.chunks_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Run Chunking
# MAGIC
# MAGIC `KBProcessor.chunk_and_save()` handles:
# MAGIC 1. Load existing chunked `doc_id`s — skip already-chunked docs (incremental)
# MAGIC 2. Two-stage chunk each new document
# MAGIC 3. Append to `kb_chunks` with `mergeSchema=true`
# MAGIC 4. Enable Change Data Feed so Vector Search syncs incrementally

# COMMAND ----------

new_chunk_count = processor.chunk_and_save()
logger.info(f"chunk_and_save() wrote {new_chunk_count:,} new chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Chunk Quality Statistics

# COMMAND ----------

from pyspark.sql.functions import avg, count  # noqa: E402
from pyspark.sql.functions import max as _max  # noqa: E402
from pyspark.sql.functions import min as _min  # noqa: E402

chunks_tbl = spark.table(processor.chunks_table)

stats = (
    chunks_tbl.groupBy("source_type")
    .agg(
        count("*").alias("chunk_count"),
        avg(F.length("text")).alias("avg_chars"),
        _min(F.length("text")).alias("min_chars"),
        _max(F.length("text")).alias("max_chars"),
    )
    .orderBy("source_type")
)

logger.info("KB Chunks breakdown:")
for row in stats.collect():
    logger.info(
        f"  [{row['source_type']:<12}]  chunks={row['chunk_count']:>5}  "
        f"avg_chars={int(row['avg_chars']):>5}  "
        f"min={row['min_chars']:>4}  max={row['max_chars']:>5}"
    )

total_chunks = chunks_tbl.count()
logger.info(f"Total chunks in {processor.chunks_table}: {total_chunks:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Inspect Sample Chunks

# COMMAND ----------

logger.info("Sample chunks by source_type:")
chunks_tbl.select("source_type", "source_repo", "title", "section_header", "text").show(10, truncate=60)
