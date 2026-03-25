"""
KB Chunk + Vector Search Sync Script
=====================================
Scheduled via kb_pipeline_job.yml (Task 2 — chunk_and_sync).
Runs after 1.3_databricks_knowledge_ingestion.py has written new docs.

What it does:
  1. Loads config for the current environment
  2. Calls KBProcessor.chunk_and_save() — chunks new KB docs into kb_chunks
  3. Creates the VS endpoint if it doesn't exist
  4. Creates the kb_chunks_index if it doesn't exist
  5. Triggers a TRIGGERED pipeline sync so new chunks get embedded

Usage (local / Databricks job):
    python sync_vs_index.py --env dev
"""

import argparse
import sys

from loguru import logger
from pyspark.sql import SparkSession

from arch_designer_agent.config import get_env, load_config
from arch_designer_agent.kb_processor import KBProcessor
from arch_designer_agent.vector_search import VectorSearchManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync KB Vector Search index")
    parser.add_argument(
        "--env",
        default=None,
        help="Environment override (dev/acc/prd). Defaults to spark tag.",
    )
    args = parser.parse_args()

    spark = SparkSession.builder.getOrCreate()
    env = args.env or get_env(spark)
    cfg = load_config("project_config.yml", env)

    logger.info(f"Environment : {env}")
    logger.info(f"Catalog     : {cfg.catalog}")
    logger.info(f"Schema      : {cfg.schema}")
    logger.info(f"VS Endpoint : {cfg.vector_search_endpoint}")

    # Step 1 — Chunk new KB documents
    processor = KBProcessor(spark=spark, config=cfg)
    new_chunks = processor.chunk_and_save()
    logger.info(f"Chunking complete: {new_chunks:,} new chunks written")

    # Step 2 — Sync Vector Search index
    vs_manager = VectorSearchManager(config=cfg)

    # Create endpoint (no-op if already exists)
    vs_manager.create_endpoint_if_not_exists()

    # Create or get the index (no-op if already exists)
    index = vs_manager.create_or_get_index()

    # Trigger sync — new rows appended to kb_chunks since last sync
    # will be embedded and added to the index.
    logger.info(f"Triggering sync on index: {vs_manager.index_name}")
    index.sync()
    logger.info("✓ Vector Search index sync triggered successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error(f"sync_vs_index failed: {exc}")
        sys.exit(1)
