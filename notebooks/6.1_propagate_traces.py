# Databricks notebook source
# DBTITLE 1,Lecture 6.1 Header
# MAGIC %md
# MAGIC # Lecture 6.1: Propagate Traces — Arch Designer Agent
# MAGIC
# MAGIC ## What this notebook does
# MAGIC
# MAGIC Sends 30 representative architecture queries to the deployed
# MAGIC `arch-designer-agent-dev` endpoint to populate the MLflow traces table.
# MAGIC These traces are then evaluated by `update_traces_aggregated.py`.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Agent deployed via notebook 5.1 / `register_deploy_agent` job
# MAGIC - Endpoint status: `READY`

# COMMAND ----------

# DBTITLE 1,Setup
import random
import time
from datetime import datetime

from databricks.sdk import WorkspaceClient
from openai import OpenAI

workspace = WorkspaceClient()
host = workspace.config.host
token = workspace.tokens.create(lifetime_seconds=3600).token_value

endpoint_name = "arch-designer-agent-dev"

client = OpenAI(
    api_key=token,
    base_url=f"{host}/serving-endpoints",
)

# COMMAND ----------

# DBTITLE 1,Architecture queries
queries = [
    # Factual — should answer directly from KB
    "What is the difference between Delta Sync Index and Direct Access Index in Vector Search?",
    "What is Delta Live Tables and when should I use it on Databricks?",
    "What is the medallion architecture and how should I implement it?",
    "How does Auto Loader differ from COPY INTO for incremental data ingestion on Databricks?",
    "What monitoring and observability tools should I use for Databricks pipelines in production?",
    "What is Unity Catalog and how does it handle data governance?",
    "How does Databricks Model Serving work and what are the deployment options?",
    "What is the difference between a Delta Live Tables pipeline and a Databricks Job?",
    "How does Photon accelerate queries in Databricks SQL?",
    "What are the benefits of using MLflow for experiment tracking on Databricks?",
    # Design — should trigger workspace scan + table profile
    "Design a real-time fraud detection architecture on Databricks with low latency requirements.",
    "Design a machine learning feature store architecture on Databricks.",
    "How do I architect a cost-efficient data lakehouse for a small analytics team?",
    "Design a CDC-based ingestion pipeline from an operational database into Databricks.",
    "How should I implement a recommendation system on Databricks at scale?",
    "Design a multi-hop data pipeline for a retail analytics use case on Databricks.",
    "How do I build a real-time event processing architecture using Structured Streaming?",
    "Design an enterprise RAG application using Databricks Vector Search and Model Serving.",
    "How should I architect a batch ML scoring pipeline for 100M records daily?",
    "Design a data quality monitoring system for a production lakehouse.",
    # Design with specific constraints
    "Design a Databricks architecture for a healthcare data platform with strict PII controls.",
    "How do I build a cost-optimised Databricks pipeline for a startup with limited budget?",
    "Design a multi-cloud data ingestion architecture feeding into a Databricks lakehouse.",
    "How should I architect a Databricks solution for a team with limited Spark expertise?",
    "Design a streaming anomaly detection system on Databricks with sub-second latency.",
    # Follow-up / multi-turn style (standalone)
    "What changes if I need to support 10x more data volume in a lakehouse architecture?",
    "How do I add observability and cost tracking to an existing Databricks data platform?",
    "What is the best way to handle schema evolution in a Delta Lake medallion pipeline?",
    "How do I implement row-level and column-level security in Unity Catalog?",
    "How should I migrate an existing Spark on EMR workload to Databricks?",
]

# COMMAND ----------

# DBTITLE 1,Send queries to endpoint
for i, query in enumerate(queries):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_id = f"s-propagate-{timestamp}-{random.randint(100000, 999999)}"
    request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

    print(f"[{i + 1}/{len(queries)}] {query[:70]}...")
    response = client.responses.create(
        model=endpoint_name,
        input=[{"role": "user", "content": query}],
        extra_body={
            "custom_inputs": {
                "session_id": session_id,
                "request_id": request_id,
            }
        },
    )
    time.sleep(3)

print("Done — all queries sent.")

# COMMAND ----------
