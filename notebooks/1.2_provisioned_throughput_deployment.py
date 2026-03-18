# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 1.2: Provisioned Throughput Deployment
# MAGIC - Your own fine-tuned models
# MAGIC - Custom models registered in Unity Catalog
# MAGIC - Models that need dedicated capacity
# COMMAND ----------

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayConfig,
    AiGatewayInferenceTableConfig,
    AiGatewayUsageTrackingConfig,
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from loguru import logger
from openai import OpenAI

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Understanding Provisioned Throughput
# MAGIC
# MAGIC ### Model Units and Throughput
# MAGIC
# MAGIC - **Model Unit**: A unit of compute capacity for serving models
# MAGIC - **Throughput**: Measured in tokens per second
# MAGIC - **Example**: For DeepSeek/Llama models:
# MAGIC   - 1 model unit ≈ 65 tokens/second
# MAGIC   - 50 model units ≈ 3,250 tokens/second

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Advanced Configuration Parameters
# MAGIC
# MAGIC ### Available Parameters for Provisioned Throughput:
# MAGIC
# MAGIC #### Core Parameters:
# MAGIC - **workload_size**: `Small`, `Medium`, `Large` - Compute capacity per instance
# MAGIC - **scale_to_zero_enabled**: `True`/`False` - Auto-scale to zero when idle
# MAGIC - **min_provisioned_throughput**: Minimum model units (must be 0 if scale_to_zero enabled)  # noqa: E501
# MAGIC - **max_provisioned_throughput**: Maximum model units for auto-scaling
# MAGIC
# MAGIC #### Monitoring & Observability:
# MAGIC - **Inference Tables**: Log all requests/responses to Delta table
# MAGIC   - Must be enabled via Databricks UI (Serving -> Endpoint -> Configuration)
# MAGIC   - Or via REST API (not available in SDK 0.78.0)
# MAGIC   - Creates table: `{catalog}.{schema}.{endpoint_name}_payload`
# MAGIC   - Includes: request_id, timestamp, request, response, status_code, latency
# MAGIC   - Useful for: debugging, auditing, model monitoring, fine-tuning data
# MAGIC
# MAGIC #### Safety & Compliance:
# MAGIC - **guardrails**: Configure input/output validation and filtering
# MAGIC   - PII detection and redaction
# MAGIC
# MAGIC #### Environment Variables:
# MAGIC - **environment_vars**: Pass custom environment variables to the model
# MAGIC   - API keys for external services
# MAGIC   - Custom configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Deploying LLama with Provisioned Throughput

# COMMAND ----------

# Configuration - Using a real model from system.ai catalog
ENDPOINT_NAME = "llama-3-2-1b-provisioned"  # Your endpoint name
MODEL_NAME = "system.ai.llama_v3_2_1b_instruct"  # Model from system.ai catalog
WORKLOAD_SIZE = "Small"  # Options: Small, Medium, Large
SCALE_TO_ZERO = True  # Set to True to save costs when not in use
MIN_PROVISIONED_THROUGHPUT = 0  # Must be 0 when scale_to_zero is enabled
MAX_PROVISIONED_THROUGHPUT = 20  # Max capacity for auto-scaling

catalog = "mlops_dev"
schema = "pratikhe"
BUDGET_POLICY_ID = None  # e.g. "my-budget-policy-id"
# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Check if endpoint exists

# COMMAND ----------

def endpoint_exists(endpoint_name: str) -> bool:
    """Check if serving endpoint exists."""
    try:
        w.serving_endpoints.get(endpoint_name)
        return True
    except Exception:
        return False

if endpoint_exists(ENDPOINT_NAME):
    logger.info(f"Endpoint '{ENDPOINT_NAME}' already exists")
    logger.info("To update, delete the existing endpoint first or use a different name")
else:
    logger.info(f"Endpoint '{ENDPOINT_NAME}' does not exist. Ready to create.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Create the endpoint with provisioned throughput
# MAGIC
# MAGIC **Important Notes:**
# MAGIC - Provisioned throughput endpoints take 15-30 minutes to deploy
# MAGIC - You'll be charged for the provisioned capacity while it's running
# MAGIC - Make sure to delete the endpoint when not in use for the course

# COMMAND ----------

ai_gateway_cfg = AiGatewayConfig(
    inference_table_config=AiGatewayInferenceTableConfig(
        enabled=True,
        catalog_name=catalog,
        schema_name=schema,
        table_name_prefix="provisioned_throughput_monitoring",
    ),
    usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
)

endpoint_config = EndpointCoreConfigInput(
    name=ENDPOINT_NAME,
    served_entities=[
        ServedEntityInput(
            entity_name=MODEL_NAME,
            entity_version="1",
            workload_size=WORKLOAD_SIZE,
            scale_to_zero_enabled=SCALE_TO_ZERO,
            min_provisioned_throughput=MIN_PROVISIONED_THROUGHPUT,
            max_provisioned_throughput=MAX_PROVISIONED_THROUGHPUT,
        )
    ],
)

# Create the endpoint
logger.info(f"Creating endpoint '{ENDPOINT_NAME}'...")

w.serving_endpoints.create(
    name=ENDPOINT_NAME,
    config=endpoint_config,
    ai_gateway=ai_gateway_cfg,
    budget_policy_id=BUDGET_POLICY_ID,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Monitor endpoint deployment

# COMMAND ----------

def wait_for_endpoint(endpoint_name: str, timeout_minutes: int = 30) -> bool | None:
    """Wait for endpoint to be ready."""
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60

    while True:
        try:
            endpoint = w.serving_endpoints.get(endpoint_name)
            config_state = endpoint.state.config_update
            ready_state = endpoint.state.ready

            logger.info(f"Status: config_update={config_state}, ready={ready_state}")

            # Check for failure state
            if hasattr(endpoint.state, 'config_update_message'):
                msg = endpoint.state.config_update_message
                if msg:
                    logger.info(f"Message: {msg}")

            # Endpoint is ready when config is NOT_UPDATING and ready state is READY
            if config_state.value == "NOT_UPDATING" and ready_state.value == "READY":
                logger.info(f"Endpoint '{endpoint_name}' is ready!")
                return True

            # Check if endpoint creation failed
            if config_state.value == "UPDATE_FAILED":
                logger.error("Endpoint creation failed!")
                if hasattr(endpoint.state, 'config_update_message'):
                    logger.error(f"Error: {endpoint.state.config_update_message}")
                return False

            if time.time() - start_time > timeout_seconds:
                logger.warning("Timeout waiting for endpoint")
                return False

            time.sleep(30)  # Check every 30 seconds

        except Exception as e:
            logger.error(f"Error checking endpoint: {e}")
            logger.info(
                "Tip: Check the Databricks UI -> Machine Learning"
                " -> Serving for detailed error messages"
            )
            return False

# Uncomment to monitor deployment
wait_for_endpoint(ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Calling the Provisioned Endpoint

# COMMAND ----------

# Example: Call the endpoint once it's ready
host = w.config.host
token = w.tokens.create(lifetime_seconds=1200).token_value

client = OpenAI(
    api_key=token,
    base_url=f"{host}/serving-endpoints"
)

response = client.chat.completions.create(
    model=ENDPOINT_NAME,
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {
            "role": "user",
            "content": "Explain the benefits of provisioned throughput for LLMs.",
        },
    ],
    max_tokens=500,
    temperature=0.7
)

logger.info("Response:")
logger.info(response.choices[0].message.content)
logger.info(f"Tokens used: {response.usage.total_tokens}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Performance Monitoring

# COMMAND ----------

def get_endpoint_metrics(endpoint_name: str) -> object | None:
    """Get endpoint metrics and status."""
    try:
        endpoint = w.serving_endpoints.get(endpoint_name)

        logger.info(f"Endpoint: {endpoint_name}")
        logger.info(f"State: {endpoint.state.config_update}")
        logger.info("Configuration:")

        for entity in endpoint.config.served_entities:
            logger.info(f"  Model: {entity.entity_name}")
            logger.info(f"  Workload Size: {entity.workload_size}")
            min_tp = entity.min_provisioned_throughput
            max_tp = entity.max_provisioned_throughput
            logger.info(f"  Min Throughput: {min_tp} model units")
            logger.info(f"  Max Throughput: {max_tp} model units")
            logger.info(f"  Scale to Zero: {entity.scale_to_zero_enabled}")

        return endpoint
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return None

get_endpoint_metrics(ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Cost Estimation

# COMMAND ----------

def estimate_provisioned_cost(
    model_units: int,
    hours_per_day: int,
    days: int,
    cost_per_unit_hour: float = 2.0,  # Approximate cost
) -> float:
    """Estimate cost for provisioned throughput."""
    total_hours = hours_per_day * days
    total_cost = model_units * total_hours * cost_per_unit_hour

    logger.info("Cost Estimation:")
    logger.info(f"  Model Units: {model_units}")
    logger.info(f"  Hours per day: {hours_per_day}")
    logger.info(f"  Days: {days}")
    logger.info(f"  Total hours: {total_hours}")
    logger.info(f"  Cost per unit-hour: ${cost_per_unit_hour}")
    logger.info(f"  Total cost: ${total_cost:.2f}")

    # Calculate throughput
    tokens_per_second = model_units * 65
    total_tokens = tokens_per_second * 3600 * total_hours
    cost_per_million_tokens = (total_cost / total_tokens) * 1_000_000

    logger.info(f"  Throughput: {tokens_per_second:,} tokens/second")
    logger.info(f"  Total tokens: {total_tokens:,}")
    logger.info(f"  Effective cost: ${cost_per_million_tokens:.2f} per 1M tokens")

    return total_cost

# Example: 50 model units, 8 hours/day, 30 days
estimate_provisioned_cost(50, 8, 30)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Cleanup
# MAGIC
# MAGIC **Important**: Delete the endpoint when not in use to avoid charges!

# COMMAND ----------

# Uncomment to delete the endpoint
"""
def delete_endpoint(endpoint_name: str):
    try:
        w.serving_endpoints.delete(endpoint_name)
        logger.info(f"Endpoint '{endpoint_name}' deleted successfully")
    except Exception as e:
        logger.error(f"Error deleting endpoint: {e}")

# delete_endpoint(ENDPOINT_NAME)
"""

logger.info("Remember to delete your provisioned endpoint when done!")
logger.info("Uncomment the code above to delete the endpoint.")
