# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 4.4: Evaluation, Log & Register — Arch Designer Agent
# MAGIC
# MAGIC ## What this notebook does
# MAGIC
# MAGIC 1. **Evaluate** — runs all 5 scorers (2 custom + 3 LLM-as-judge) against `eval_inputs.txt`
# MAGIC 2. **Log** — packages `agent.py` as an MLflow pyfunc model with resource declarations
# MAGIC 3. **Register** — versions the model in Unity Catalog under `{catalog}.{schema}.arch_agent`
# MAGIC 4. **Alias** — sets `@champion` so serving endpoints can reference it by alias
# MAGIC
# MAGIC ## Scorers
# MAGIC
# MAGIC | Scorer | Type | Checks |
# MAGIC |---|---|---|
# MAGIC | `response_length_check` | Custom Python | >= 100 words |
# MAGIC | `cites_databricks_service` | Custom Python | mentions DLT, Delta, UC, VS, etc. |
# MAGIC | `architectural_clarity` | LLM-as-judge (Maverick) | concrete components, actionable |
# MAGIC | `stays_in_databricks_scope` | LLM-as-judge (Maverick) | no AWS Glue / GCP Dataflow |
# MAGIC | `grounded_in_evidence` | LLM-as-judge (Maverick) | cites real Databricks features |

# COMMAND ----------

import os
import subprocess

import mlflow
import nest_asyncio
from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import SparkSession

from arch_designer_agent.agent import log_register_agent
from arch_designer_agent.config import get_env, load_config

nest_asyncio.apply()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)
w = WorkspaceClient()

mlflow.set_experiment(cfg.experiment_name)

logger.info(f"Environment : {env}")
logger.info(f"Catalog     : {cfg.catalog}")
logger.info(f"Schema      : {cfg.schema}")
logger.info(f"Experiment  : {cfg.experiment_name}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Run evaluation
# MAGIC
# MAGIC Loads `eval_inputs.txt` (10 architecture questions), runs the agent on each,
# MAGIC then scores every response with all 5 scorers.
# MAGIC Results are logged automatically to the active MLflow experiment.

# COMMAND ----------

from arch_designer_agent.evaluation import evaluate_agent

eval_results = evaluate_agent(cfg, eval_inputs_path="../eval_inputs.txt")

logger.info("Evaluation complete. Metrics:")
for metric, value in eval_results.metrics.items():
    logger.info(f"  {metric}: {value:.3f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Log model + register to Unity Catalog
# MAGIC
# MAGIC `log_register_agent()`:
# MAGIC - Opens an MLflow run tagged with `git_sha` and `run_id`
# MAGIC - Logs `agent.py` as a pyfunc model with resource declarations (endpoints, VS index, tables)
# MAGIC - Logs the evaluation metrics from step 2 alongside the model
# MAGIC - Registers the model to UC as `{catalog}.{schema}.arch_agent`
# MAGIC - Sets the `@champion` alias on the new version

# COMMAND ----------

# Derive git SHA for lineage tracking (falls back to "local" outside git)
try:
    git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    git_sha = os.getenv("GIT_SHA", "local")

run_id = os.getenv("DATABRICKS_RUN_ID", "local")
model_name = f"{cfg.catalog}.{cfg.schema}.arch_agent"
agent_code_path = "../src/arch_designer_agent/agent.py"

logger.info(f"git_sha    : {git_sha}")
logger.info(f"run_id     : {run_id}")
logger.info(f"model_name : {model_name}")

registered_model = log_register_agent(
    cfg=cfg,
    git_sha=git_sha,
    run_id=run_id,
    agent_code_path=agent_code_path,
    model_name=model_name,
    evaluation_metrics=eval_results.metrics,
)

logger.info(f"Registered: {model_name} v{registered_model.version} (@champion)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Verify registration
# MAGIC
# MAGIC Confirms the model version and alias are visible in Unity Catalog.

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()
model_version = client.get_model_version(
    name=model_name,
    version=registered_model.version,
)
champion_version = client.get_model_version_by_alias(
    name=model_name,
    alias="champion",
)

logger.info(f"Latest version  : {model_version.version} — {model_version.status}")
logger.info(f"@champion points to version: {champion_version.version}")
logger.info(f"Model URI: models:/{model_name}@champion")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. (Optional) Smoke-test predict() locally
# MAGIC
# MAGIC Loads the registered model back via MLflow pyfunc and calls `predict()` once
# MAGIC to confirm the serialised agent works end-to-end before deploying to serving.

# COMMAND ----------

loaded = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")

test_request = {"input": [{"role": "user", "content": "What is Delta Live Tables and when should I use it?"}]}

response = loaded.predict(test_request)
logger.info("Smoke test response:")
logger.info(response)
