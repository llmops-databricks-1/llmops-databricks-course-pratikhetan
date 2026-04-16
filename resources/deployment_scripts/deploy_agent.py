# Databricks notebook source
"""
Deploy arch-designer-agent to a Databricks Model Serving endpoint.

Reads the model version pinned to the "champion" alias set by log_register_agent,
then calls agents.deploy() with Lakebase service-principal secrets injected as
environment variables so the serving container can authenticate to Lakebase.

Parameters (set via Databricks Job base_parameters):
  env     - target environment: dev | acc | prd  (default: dev)
  git_sha - Git commit SHA for traceability       (default: local)
"""

from databricks import agents
from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import dbutils
from loguru import logger
from mlflow import MlflowClient

from arch_designer_agent.config import ProjectConfig

# COMMAND ----------

# Get parameters (passed via base_parameters in job YAML)
git_sha = dbutils.widgets.get("git_sha")
env = dbutils.widgets.get("env")

# Secret scope follows the <env>_SPN convention (dev_SPN, acc_SPN, prd_SPN).
# Each scope exposes two keys: client_id and client_secret.
secret_scope = f"{env}_SPN"

# Load configuration
cfg = ProjectConfig.from_yaml("../../project_config.yml", env=env)

model_name = f"{cfg.catalog}.{cfg.schema}.arch_designer_agent"
endpoint_name = f"arch-designer-agent-{env}"

# COMMAND ----------

# Resolve the registered model version pinned to "champion"
client = MlflowClient()
model_version = client.get_model_version_by_alias(model_name, "champion").version

# Resolve experiment ID for logging
experiment = client.get_experiment_by_name(cfg.experiment_name)

logger.info("Deploying agent:")
logger.info(f"  Model:    {model_name}  (version {model_version})")
logger.info(f"  Endpoint: {endpoint_name}")
logger.info(f"  Env:      {env}")

# COMMAND ----------

# Deploy agent to Model Serving.
# Lakebase credentials are injected as environment variables so the container
# can create a connection without embedding secrets in model artifacts.
agents.deploy(
    model_name=model_name,
    model_version=int(model_version),
    endpoint_name=endpoint_name,
    usage_policy_id=cfg.usage_policy_id,
    scale_to_zero=True,
    workload_size="Small",
    deploy_feedback_model=False,
    environment_vars={
        "GIT_SHA": git_sha,
        "MODEL_VERSION": model_version,
        "MODEL_SERVING_ENDPOINT_NAME": endpoint_name,
        "MLFLOW_EXPERIMENT_ID": experiment.experiment_id,
        # Course-code-hub used LAKEBASE_SP_* names — our memory.py reads DATABRICKS_* instead:
        # "LAKEBASE_SP_CLIENT_ID": f"{{{{secrets/{secret_scope}/client_id}}}}",
        # "LAKEBASE_SP_CLIENT_SECRET": f"{{{{secrets/{secret_scope}/client_secret}}}}",
        # "LAKEBASE_SP_HOST": WorkspaceClient().config.host,
        "DATABRICKS_CLIENT_ID": f"{{{{secrets/{secret_scope}/client_id}}}}",
        "DATABRICKS_CLIENT_SECRET": f"{{{{secrets/{secret_scope}/client_secret}}}}",
        "DATABRICKS_HOST": WorkspaceClient().config.host,
    },
)

logger.info("Deployment complete!")

# COMMAND ----------

# Grant CAN_QUERY to all workspace users and CAN_MANAGE to the deploying user
# so the endpoint is visible in the UI regardless of which SPN deployed it.
w = WorkspaceClient()
w.serving_endpoints.set_permissions(
    serving_endpoint_id=endpoint_name,
    access_control_list=[
        {"group_name": "users", "permission_level": "CAN_QUERY"},
        {"user_name": "pratikhetan@gmail.com", "permission_level": "CAN_MANAGE"},
    ],
)
logger.info(f"Permissions set on endpoint {endpoint_name}")
