# Databricks notebook source
# DBTITLE 1,Lecture 5.1 Header
# MAGIC %md
# MAGIC # Lecture 5.1: Agent Deployment & Testing — Arch Designer Agent
# MAGIC
# MAGIC ## Topics Covered
# MAGIC - Deploying agents using `agents.deploy()`
# MAGIC - Configuring environment variables
# MAGIC - Testing deployed endpoints via OpenAI-compatible client
# MAGIC - Multi-turn conversation with session IDs
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Model registered to Unity Catalog via notebook 4.4
# MAGIC - `@champion` alias set on the target version

# COMMAND ----------

# DBTITLE 1,Setup and configuration
import os
import mlflow
from databricks import agents
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow import MlflowClient
from pyspark.sql import SparkSession

from arch_designer_agent.config import get_env, load_config

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

mlflow.set_experiment(cfg.experiment_name)

model_name = f"{cfg.catalog}.{cfg.schema}.arch_agent"
endpoint_name = "arch-agent-endpoint-dev"

model_version = MlflowClient().get_model_version_by_alias(
    model_name, "champion"
).version

workspace = WorkspaceClient()
experiment = MlflowClient().get_experiment_by_name(cfg.experiment_name)

logger.info(f"Model      : {model_name} v{model_version}")
logger.info(f"Endpoint   : {endpoint_name}")
logger.info(f"Experiment : {cfg.experiment_name}")

# COMMAND ----------

# DBTITLE 1,Deploy section header
# MAGIC %md
# MAGIC ## 1. Deploy Agent
# MAGIC
# MAGIC The `agents.deploy()` API handles:
# MAGIC - Endpoint creation and configuration
# MAGIC - Inference tables for monitoring
# MAGIC - Environment variables
# MAGIC - Model versioning and scale-to-zero

# COMMAND ----------

# DBTITLE 1,Deploy agent to serving endpoint
git_sha = "local"

# Delete the existing endpoint if it's in a broken state (config=None after
# a failed first deployment), since agents.deploy() cannot update it.
try:
    ep = workspace.serving_endpoints.get(endpoint_name)
    if ep.config is None:
        workspace.serving_endpoints.delete(endpoint_name)
        logger.info(f"Deleted broken endpoint '{endpoint_name}' (config was None)")
except Exception:
    pass  # endpoint doesn't exist — agents.deploy() will create it

agents.deploy(
    model_name=model_name,
    model_version=int(model_version),
    endpoint_name=endpoint_name,
    scale_to_zero=True,
    workload_size="Small",
    deploy_feedback_model=False,
    environment_vars={
        "GIT_SHA": git_sha,
        "MODEL_VERSION": model_version,
        "MODEL_SERVING_ENDPOINT_NAME": endpoint_name,
        "MLFLOW_EXPERIMENT_ID": experiment.experiment_id,
    },
)

# COMMAND ----------

# DBTITLE 1,Test section header
# MAGIC %md
# MAGIC ## 2. Test the Deployed Endpoint
# MAGIC
# MAGIC Wait for deployment to complete (5-10 minutes), then test the endpoint
# MAGIC using the OpenAI-compatible Responses API.

# COMMAND ----------

# DBTITLE 1,Test deployed endpoint
import random
from datetime import datetime
from openai import OpenAI

host = workspace.config.host
token = workspace.tokens.create(lifetime_seconds=2000).token_value

client = OpenAI(
    api_key=token,
    base_url=f"{host}/serving-endpoints",
)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

response = client.responses.create(
    model=endpoint_name,
    input=[
        {"role": "user", "content": "Design a real-time fraud detection pipeline on Databricks for streaming transaction data"}
    ],
    extra_body={"custom_inputs": {
        "session_id": session_id,
        "request_id": request_id,
    }}
)

logger.info(f"Response ID : {response.id}")
logger.info(f"Session ID  : {response.custom_outputs.get('session_id')}")
logger.info(f"Request ID  : {response.custom_outputs.get('request_id')}")
logger.info("\nAssistant Response:")
logger.info("-" * 80)
logger.info(response.output[0].content[0].text)
logger.info("-" * 80)