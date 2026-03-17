# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 1.1: Foundation Models Overview
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Foundation models on Databricks
# MAGIC - Databricks hosted models
# MAGIC - External models
# MAGIC - Pricing comparison

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Foundation Models on Databricks
# MAGIC
# MAGIC Databricks provides several ways to access foundation models:
# MAGIC
# MAGIC 1. **Foundation Model APIs (Pay-per-token)**
# MAGIC    - Serverless, fully managed
# MAGIC    - Pay only for what you use
# MAGIC    - No infrastructure management
# MAGIC    - Examples: Meta Llama, Mistral, DBRX, DeepSeek
# MAGIC
# MAGIC 2. **Provisioned Throughput**
# MAGIC    - Dedicated capacity
# MAGIC    - Predictable performance
# MAGIC    - Better for high-volume workloads
# MAGIC    - Can be fine-tuned
# MAGIC
# MAGIC 3. **External Models**
# MAGIC    - OpenAI, Anthropic, Cohere, etc.
# MAGIC    - Unified interface
# MAGIC    - Centralized governance

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI

# Initialize workspace client
w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Databricks Hosted Models (Foundation Model APIs)

# COMMAND ----------

# List available foundation models
endpoints = w.serving_endpoints.list()

logger.info("Available Foundation Model Endpoints:")
logger.info("-" * 80)
for endpoint in endpoints:
    if endpoint.name and "databricks" in endpoint.name:
        logger.info(f"Name: {endpoint.name}")
        logger.info(f"State: {endpoint.state}")
        logger.info("-" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example: Using Meta Llama

# COMMAND ----------

# Using OpenAI SDK with Databricks
# Authenticate using Databricks SDK
host = w.config.host
token = w.tokens.create(lifetime_seconds=1200).token_value

client = OpenAI(
    api_key=token,
    base_url=f"{host.rstrip('/')}/serving-endpoints"
)
model_name = "databricks-llama-4-maverick"

# Call the model
response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain LLMOps in 3 sentences."}
    ],
    max_tokens=200,
     # Temperature ontrols randomness: 0.0 = deterministic, 1.0 = more creative/random
    temperature=0.7
)

logger.info("Response:")
logger.info(response.choices[0].message.content)
logger.info(f"Tokens used: {response.usage.total_tokens}")
logger.info(f"Input tokens: {response.usage.prompt_tokens}")
logger.info(f"Output tokens: {response.usage.completion_tokens}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Pricing Comparison
# MAGIC
# MAGIC ### Foundation Model APIs (Pay-per-token)
# MAGIC
# MAGIC Pricing is in **DBUs (Databricks Units)** per million tokens:
# MAGIC
# MAGIC | Model | Input (DBU/1M tokens) | Output (DBU/1M tokens) |
# MAGIC |-------|----------------------|------------------------|
# MAGIC | Llama 4 Maverick | 7.143 | 21.429 |
# MAGIC | Llama 3.3 70B | 7.143 | 21.429 |
# MAGIC | GPT OSS 120B | 2.143 | 8.571 |
# MAGIC | Gemma 3 12B | 2.143 | 7.143 |
# MAGIC | Llama 3.1 8B | 2.143 | 6.429 |
# MAGIC | GPT OSS 20B | 1.000 | 4.286 |
# MAGIC
# MAGIC ### Provisioned Throughput
# MAGIC
# MAGIC Priced in **DBU per hour** for dedicated capacity:
# MAGIC
# MAGIC | Model | Entry Capacity (DBU/hour) | Scaling Capacity (DBU/hour) |
# MAGIC |-------|---------------------------|----------------------------|
# MAGIC | Llama 4 Maverick | 85.714 | 85.714 |
# MAGIC | Llama 3.3 70B | 85.714 | 342.857 |
# MAGIC | Llama 3.1 8B | 53.571 | 106.000 |
# MAGIC | Llama 3.2 3B | 46.429 | 92.857 |
# MAGIC | Llama 3.2 1B | 42.857 | 85.714 |
# MAGIC
# MAGIC **Note**: DBU pricing varies by region and contract. Check [Databricks Pricing](https://www.databricks.com/product/pricing) for current rates.
# MAGIC
# MAGIC ### Cost Calculation Example

# COMMAND ----------

def calculate_api_cost(input_tokens: int, output_tokens: int,
                       input_dbu_per_1m: float, output_dbu_per_1m: float) -> float:
    """Calculate DBU cost for pay-per-token API."""
    input_cost = (input_tokens / 1_000_000) * input_dbu_per_1m
    output_cost = (output_tokens / 1_000_000) * output_dbu_per_1m
    return input_cost + output_cost

def calculate_provisioned_cost(hours: int, dbu_per_hour: float) -> float:
    """Calculate DBU cost for provisioned throughput."""
    return hours * dbu_per_hour

# Example: 1M input tokens, 500K output tokens with Llama 3.3 70B
api_cost = calculate_api_cost(1_000_000, 500_000, 7.143, 21.429)
logger.info(f"Pay-per-token cost: {api_cost:.2f} DBUs")

# Example: 24 hours with Llama 3.2 1B provisioned (entry capacity)
provisioned_cost = calculate_provisioned_cost(24, 42.857)
logger.info(f"Provisioned throughput cost (24h): {provisioned_cost:.2f} DBUs")

# Break-even analysis
# Assume 10M tokens processed in 24h (mix of input/output)
input_tokens = 6_000_000
output_tokens = 4_000_000

api_cost_equivalent = calculate_api_cost(input_tokens, output_tokens, 7.143, 21.429)
logger.info(f"For {input_tokens + output_tokens:,} tokens in 24h:")
logger.info(f"API cost (Llama 3.3 70B): {api_cost_equivalent:.2f} DBUs")
logger.info(f"Provisioned cost (Llama 3.2 1B): {provisioned_cost:.2f} DBUs")
logger.info(f"Difference: {api_cost_equivalent - provisioned_cost:.2f} DBUs")
logger.info("Provisioned throughput becomes cost-effective at high, predictable volumes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. External Models
# MAGIC
# MAGIC Databricks allows you to integrate external model providers through a unified interface.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Key Differences Summary
# MAGIC
# MAGIC | Aspect | Foundation APIs | Provisioned Throughput | External Models |
# MAGIC |--------|----------------|------------------------|-----------------|
# MAGIC | **Pricing** | Pay-per-token | Pay-per-model-unit-hour | Pay-per-token (external) |
# MAGIC | **Performance** | Shared capacity | Dedicated capacity | Varies by provider |
# MAGIC | **Latency** | Variable | Predictable | Variable |
# MAGIC | **Fine-tuning** | No | Yes | Limited |
# MAGIC | **Best For** | Variable workloads | High-volume, predictable | Specific model requirements |
# MAGIC | **Setup** | Instant | Requires provisioning | Requires credentials |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Choosing the Right Option
# MAGIC
# MAGIC **Use Foundation Model APIs when:**
# MAGIC - You have variable or unpredictable workloads
# MAGIC - You're prototyping or experimenting
# MAGIC - You want zero infrastructure management
# MAGIC - Your volume is low to medium
# MAGIC
# MAGIC **Use Provisioned Throughput when:**
# MAGIC - You have high, predictable volume
# MAGIC - You need consistent low latency
# MAGIC - You want to fine-tune models
# MAGIC - Cost predictability is important
# MAGIC
# MAGIC **Use External Models when:**
# MAGIC - You need specific models not available on Databricks
# MAGIC - You have existing contracts with providers
# MAGIC - You need specific model capabilities
