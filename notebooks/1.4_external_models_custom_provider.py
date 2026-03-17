# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 1.4: External Models with Custom Provider

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve OpenAI API Key from Secrets
# MAGIC
# MAGIC Your OpenAI API key is stored in Databricks Secrets:
# MAGIC - **Scope**: `llmops_course`
# MAGIC - **Key**: `openai_key`
# MAGIC
# MAGIC ### How to Access Secrets:
# MAGIC
# MAGIC **Using Databricks SDK** (recommended - works everywhere):
# MAGIC ```python
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC w = WorkspaceClient()
# MAGIC openai_api_key = w.secrets.get_secret(scope="llmops_course", key="openai_key").value
# MAGIC ```
# MAGIC
# MAGIC **Using dbutils** (Databricks notebooks only):
# MAGIC ```python
# MAGIC openai_api_key = dbutils.secrets.get(scope="llmops_course", key="openai_key")
# MAGIC ```
# MAGIC
# MAGIC **For External Model Endpoints** (UI configuration):
# MAGIC - Use secret reference format: `{{secrets/llmops_course/openai_key}}`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create External Model Endpoint with Custom Provider

# COMMAND ----------

import mlflow.deployments
from loguru import logger

# Get MLflow Deployments client
client = mlflow.deployments.get_deploy_client("databricks")

# Configuration
ENDPOINT_NAME = "openai-dalle-custom"

# Check if endpoint already exists
try:
    existing = client.get_endpoint(ENDPOINT_NAME)
    logger.info(f"Endpoint '{ENDPOINT_NAME}' already exists")
    logger.info(f"Status: {existing}")
except Exception:
    logger.info(f"Creating External Model endpoint: {ENDPOINT_NAME}")

    # Create External Model endpoint for OpenAI DALL-E
    endpoint = client.create_endpoint(
        name=ENDPOINT_NAME,
        config={
            "served_entities": [{
                "name": "dalle-image-generation",
                "external_model": {
                    "name": "dall-e-3",
                    "provider": "openai",
                    "task": "llm/v1/images",  # Image generation task type
                    "openai_config": {
                        "openai_api_key": "{{secrets/llmops_course/openai_key}}",
                        "openai_api_base": "https://api.openai.com/v1",
                        "openai_api_type": "openai"
                    }
                }
            }]
        }
    )

    logger.info(f"Endpoint created successfully: {ENDPOINT_NAME}")
    logger.info(f"Configuration: {endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Query the Custom Provider Endpoint
# MAGIC
# MAGIC Once your endpoint is deployed, you can query it using the OpenAI SDK:

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image
import json

w = WorkspaceClient()

# Authenticate using Databricks SDK
host = w.config.host
token = w.tokens.create(lifetime_seconds=1200).token_value

# Create OpenAI client pointing to Databricks endpoint
client = OpenAI(
    api_key=token,
    base_url=f"{host.rstrip('/')}/serving-endpoints"
)

ENDPOINT_NAME = "openai-dalle-custom"

logger.info(f"Client configured to use endpoint: {ENDPOINT_NAME}")
logger.info(f"Base URL: {host}/serving-endpoints")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate an Image (Base64 Response)
# MAGIC
# MAGIC ### Response Format Options:
# MAGIC - **`b64_json`**: Returns image as base64-encoded string (recommended)

# COMMAND ----------

# Generate image with base64 response
response = client.images.generate(
    model=ENDPOINT_NAME,
    prompt="Two cats wearing superhero capes in a sunny garden",
    n=1,  # Number of images to generate
    style="vivid",  # Options: "vivid" or "natural"
    quality="standard",  # Options: "standard" or "hd"
    response_format="b64_json"  # Returns base64-encoded image
)

logger.info("Image generated successfully!")
logger.info(f"Prompt: {response.data[0].revised_prompt if hasattr(response.data[0], 'revised_prompt') else 'N/A'}")
logger.info("Response format: b64_json")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Display the Generated Image

# COMMAND ----------

# Decode and display the image
image_data = response.data[0].b64_json
image_bytes = base64.b64decode(image_data)
image = Image.open(BytesIO(image_bytes))

# Display in notebook
display(image)

# Optionally save to file
# image.save("/dbfs/tmp/generated_image.png")
logger.info(f"Image size: {image.size}")
logger.info(f"Image format: {image.format}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Image with URL Response

# COMMAND ----------

# Generate image with URL response (temporary access)
response_url = client.images.generate(
    model=ENDPOINT_NAME,
    prompt="A futuristic data center with glowing servers",
    n=1,
    style="vivid",
    quality="standard",
    response_format="url"  # Returns temporary URL
)

image_url = response_url.data[0].url
logger.info("Image generated!")
logger.info("Temporary URL (expires in 2 hours):")
logger.info(image_url)
