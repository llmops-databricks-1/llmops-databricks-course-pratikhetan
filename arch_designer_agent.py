"""MLflow entry-point file for DatabricksExpertAgent.

This file is the ``python_model`` artifact logged by ``log_register_agent``.
MLflow serialises the *class* during ``log_model`` and deserialises it at
serving time by importing this module — so the agent's ``__init__`` receives no
arguments and bootstraps itself from the ``ModelConfig`` embedded in the run
artifact.
"""

import mlflow
from mlflow.models import ModelConfig

from arch_designer_agent.agent import DatabricksExpertAgent
from arch_designer_agent.config import ProjectConfig

# development_config is a local-only fallback used when running this file
# directly (e.g. mlflow.pyfunc.load_model on a local path).
# At serving time MLflow uses the model_config snapshot baked into the artifact
# by log_register_agent — project_config.yml is NOT available in the container.
try:
    _cfg = ProjectConfig.from_yaml("project_config.yml", env="dev")
    _dev_config = {
        "catalog": _cfg.catalog,
        "schema": _cfg.schema,
        "llm_endpoint": _cfg.llm_endpoint,
        "lakebase_instance": _cfg.lakebase_instance,
        "genie_space_id": _cfg.genie_space_id,
        "warehouse_id": _cfg.warehouse_id or None,
        "system_prompt": _cfg.system_prompt,
    }
except Exception:
    _dev_config = {}

config = ModelConfig(development_config=_dev_config)

agent = DatabricksExpertAgent()

mlflow.models.set_model(agent)
