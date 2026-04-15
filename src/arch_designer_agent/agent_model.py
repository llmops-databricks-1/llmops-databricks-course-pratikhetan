"""MLflow code-model entry point."""

import mlflow

from arch_designer_agent.agent import DatabricksExpertAgent

agent = DatabricksExpertAgent()
mlflow.models.set_model(agent)
