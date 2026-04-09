import mlflow
from mlflow.genai.scorers import Guidelines
from loguru import logger

from arch_designer_agent.agent import DatabricksExpertAgent
from arch_designer_agent.config import ProjectConfig

# ---------------------------------------------------------------------------
# LLM-as-judge guidelines
# ---------------------------------------------------------------------------

architectural_clarity_guideline = Guidelines(
    name="architectural_clarity",
    guidelines=[
        "The response must describe a concrete architecture with named components",
        "The response should explain why each component was chosen",
        "The response must be actionable — a reader should know what to build",
    ],
    model="databricks:/databricks-llama-4-maverick",
)

databricks_scope_guideline = Guidelines(
    name="stays_in_databricks_scope",
    guidelines=[
        "The response must only recommend Databricks-native services and patterns",
        "The response must not recommend services from other cloud vendors (AWS Glue, GCP Dataflow, etc.)",
        "If a third-party tool is mentioned, it must be framed as an ingestion source, not a replacement",
    ],
    model="databricks:/databricks-llama-4-maverick",
)

grounded_in_evidence_guideline = Guidelines(
    name="grounded_in_evidence",
    guidelines=[
        "The response must cite at least one specific Databricks feature or documentation concept",
        "The response must not invent Databricks product names or features that do not exist",
        "Claims about capabilities should be specific, not vague generalisations",
    ],
    model="databricks:/databricks-llama-4-maverick",
)


# ---------------------------------------------------------------------------
# Custom scorers
# ---------------------------------------------------------------------------

@mlflow.genai.scorer
def response_length_check(outputs: list) -> bool:
    """Architecture answers must have enough detail — at least 100 words.

    Unlike a short Q&A answer, an architecture recommendation needs substance.
    """
    text = _extract_text(outputs)
    return len(text.split()) >= 100


@mlflow.genai.scorer
def cites_databricks_service(outputs: list) -> bool:
    """Check that the response mentions at least one real Databricks service.

    Prevents the agent from giving generic cloud advice without Databricks grounding.
    """
    text = _extract_text(outputs).lower()
    services = [
        "delta live tables", "dlt", "delta lake", "unity catalog",
        "vector search", "model serving", "databricks jobs", "mlflow",
        "automl", "feature store", "lakebase", "genie", "warehouse",
        "databricks sql", "serverless", "photon", "medallion",
    ]
    return any(s in text for s in services)


def _extract_text(outputs: list) -> str:
    """Safely extract text string from scorer output argument."""
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        if isinstance(first, dict) and "text" in first:
            return first["text"]
        if isinstance(first, str):
            return first
        return str(first)
    return str(outputs)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_agent(
    cfg: ProjectConfig,
    eval_inputs_path: str,
) -> mlflow.models.EvaluationResult:
    """Run evaluation on DatabricksExpertAgent.

    Args:
        cfg: Project configuration.
        eval_inputs_path: Path to evaluation inputs file (one question per line).

    Returns:
        MLflow EvaluationResult with metrics.
    """
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()

    agent = DatabricksExpertAgent(spark=spark, config=cfg)

    with open(eval_inputs_path) as f:
        eval_data = [
            {"inputs": {"question": line.strip()}}
            for line in f if line.strip()
        ]

    def predict_fn(question: str) -> str:
        return agent.chat(question)

    return mlflow.genai.evaluate(
        predict_fn=predict_fn,
        data=eval_data,
        scorers=[
            response_length_check,
            cites_databricks_service,
            architectural_clarity_guideline,
            databricks_scope_guideline,
            grounded_in_evidence_guideline,
        ],
    )


def create_eval_data_from_file(eval_inputs_path: str) -> list[dict]:
    """Load evaluation data from a file.

    Args:
        eval_inputs_path: Path to file with one question per line.

    Returns:
        List of evaluation data dictionaries.
    """
    with open(eval_inputs_path) as f:
        return [
            {"inputs": {"question": line.strip()}}
            for line in f if line.strip()
        ]
