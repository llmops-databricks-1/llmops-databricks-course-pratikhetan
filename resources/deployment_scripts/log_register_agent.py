# Databricks notebook source
"""
Log and register arch-designer-agent to Unity Catalog via MLflow.

Parameters (set via Databricks Job base_parameters):
  env     - target environment: dev | acc | prd  (default: dev)
  git_sha - Git commit SHA for traceability       (default: local)
  run_id  - Databricks job run ID                 (default: local)
"""

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import dbutils

from arch_designer_agent.agent import log_register_agent
from arch_designer_agent.config import ProjectConfig
from arch_designer_agent.evaluation import evaluate_agent

# COMMAND ----------


def get_widget(name: str, default: str) -> str:
    try:
        return dbutils.widgets.get(name)
    except Exception:
        return default


env = get_widget("env", "dev")
git_sha = get_widget("git_sha", "local")
run_id = get_widget("run_id", "local")

cfg = ProjectConfig.from_yaml(config_path="../../project_config.yml", env=env)

mlflow.set_experiment(cfg.experiment_name)

# Grant CAN_MANAGE on the experiment to the configured user so they can view traces
w = WorkspaceClient()
try:
    experiment = w.experiments.get_by_name(cfg.experiment_name).experiment
    if experiment and cfg.experiment_permissions_user:
        w.experiments.set_experiment_permissions(
            experiment_id=experiment.experiment_id,
            access_control_list=[
                {
                    "user_name": cfg.experiment_permissions_user,
                    "permission_level": "CAN_MANAGE",
                }
            ],
        )
except Exception as e:
    print(f"Warning: could not set experiment permissions: {e}")

model_name = f"{cfg.catalog}.{cfg.schema}.arch_designer_agent"

# COMMAND ----------

# Run evaluation against eval_inputs.txt
results = evaluate_agent(cfg, eval_inputs_path="../../eval_inputs.txt")

# COMMAND ----------

# Log model to MLflow and register in Unity Catalog with alias "champion"
registered_model = log_register_agent(
    cfg=cfg,
    git_sha=git_sha,
    run_id=run_id,
    agent_code_path="../../agent_model.py",
    model_name=model_name,
    evaluation_metrics=results.metrics,
)
