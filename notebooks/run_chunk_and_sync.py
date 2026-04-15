# Databricks notebook source
"""
KB Chunk + Vector Search Sync — Job Notebook

Run by kb_pipeline_job (Task 2: chunk_and_sync) after the ingestion
notebook has written new docs to databricks_knowledge_base.

Steps:
  1. KBProcessor.chunk_and_save()  — chunk new KB docs into kb_chunks
  2. VectorSearchManager           — create endpoint/index if missing,
                                     trigger TRIGGERED pipeline sync
"""

from loguru import logger
from pyspark.sql import SparkSession

from arch_designer_agent.config import get_env, load_config
from arch_designer_agent.kb_processor import KBProcessor
from arch_designer_agent.vector_search import VectorSearchManager

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

logger.info(f"Environment : {env}")
logger.info(f"Catalog     : {cfg.catalog}")
logger.info(f"Schema      : {cfg.schema}")

# COMMAND ----------
# -- Step 1: Chunk new KB documents ----------------------------------------

processor = KBProcessor(spark=spark, config=cfg)
new_chunks = processor.chunk_and_save()
logger.info(f"Chunking complete: {new_chunks:,} new chunks written")

# COMMAND ----------
# -- Step 2: Sync Vector Search index --------------------------------------

vs_manager = VectorSearchManager(config=cfg)
vs_manager.sync_index()
logger.info("✓ Vector Search index sync triggered successfully")

# COMMAND ----------
# -- Step 3 (optional): Agent orchestration demo ---------------------------
# Keep disabled for job runs. Enable only for interactive notebook testing.

RUN_AGENT_DEMO = False

if RUN_AGENT_DEMO:
    from arch_designer_agent.agent import DatabricksExpertAgent

    demo_query = (
        "Design a low-latency Databricks architecture for near real-time fraud "
        "detection with strong governance controls."
    )

    agent = DatabricksExpertAgent(spark=spark, config=cfg)
    answer = agent.chat(demo_query)

    logger.info("=== Agent Recommendation ===")
    logger.info(answer)
