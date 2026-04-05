# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.4: Session Memory with Lakebase
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Lakebase (Databricks PostgreSQL) for session persistence
# MAGIC - Managing conversation history
# MAGIC - Connection pooling and authentication
# MAGIC - Building stateful agents

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import DatabaseInstance, DatabaseInstanceState
from uuid import uuid4
from loguru import logger

from arxiv_curator.memory import LakebaseMemory
from arxiv_curator.config import load_config, get_env

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create Lakebase Instance
# MAGIC
# MAGIC **Lakebase** is Databricks' managed PostgreSQL service:
# MAGIC - Fully managed and serverless
# MAGIC - Integrated with Databricks authentication
# MAGIC - Supports standard PostgreSQL features
# MAGIC - Ideal for session state, caching, and metadata

# COMMAND ----------

w = WorkspaceClient()
cfg = load_config("../project_config.yml")

instance_name = "arxiv-agent-instance"

usage_policy_id = cfg.usage_policy_id  # TODO: replace with your usage policy ID

# Create or get existing instance
try:
    instance = w.database.get_database_instance(instance_name)
    logger.info(f"Using existing instance: {instance_name}")
    if instance.state == DatabaseInstanceState.STOPPED:
        logger.info("Instance is stopped, starting...")
        instance = w.database.update_database_instance(
            name=instance_name,
            database_instance=DatabaseInstance(name=instance_name,
                                               stopped=False),
            update_mask="stopped",
        )
        instance = w.database.wait_get_database_instance_database_available(instance_name)
        logger.info("Instance started")
    lakebase_host = instance.read_write_dns
except Exception:
    logger.info(f"Creating new instance: {instance_name}")
    instance = w.database.create_database_instance(
        DatabaseInstance(
            name=instance_name, capacity="CU_1",
            usage_policy_id=usage_policy_id
        ),
    )
    lakebase_host = instance.response.read_write_dns

logger.info(f"Lakebase host: {lakebase_host}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize Memory Manager
# MAGIC
# MAGIC The `LakebaseMemory` class handles:
# MAGIC - Connection pooling
# MAGIC - Authentication (SPN or user credentials)
# MAGIC - Table creation
# MAGIC - Message persistence

# COMMAND ----------

memory = LakebaseMemory(
    host=lakebase_host,
    instance_name=instance_name,
)

logger.info("✓ Memory manager initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Save and Load Messages
# MAGIC
# MAGIC Messages are stored per session ID:
# MAGIC - Each session has a unique ID
# MAGIC - Messages are stored in order
# MAGIC - Sessions can be resumed later

# COMMAND ----------

# Create a test session
session_id = f"test-session-{uuid4()}"

# Save some messages
test_messages = [
    {"role": "user", "content": "What are recent papers on transformers?"},
    {"role": "assistant", "content": "Here are some recent papers on transformer architectures..."},
    {"role": "user", "content": "Tell me more about the first one"},
]

memory.save_messages(session_id, test_messages)
logger.info(f"✓ Saved {len(test_messages)} messages to session: {session_id}")

# COMMAND ----------

# Load messages back
loaded_messages = memory.load_messages(session_id)

logger.info(f"✓ Loaded {len(loaded_messages)} messages:")
for i, msg in enumerate(loaded_messages, 1):
    logger.info(f"  {i}. [{msg['role']}] {msg['content'][:50]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Multi-Turn Conversation
# MAGIC
# MAGIC Demonstrate a stateful conversation:

# COMMAND ----------

# Start a new session
conversation_id = f"conversation-{uuid4()}"

# Turn 1
turn1_messages = [
    {"role": "user", "content": "I'm interested in LLM evaluation metrics"}
]
memory.save_messages(conversation_id, turn1_messages)

# Simulate agent response
turn1_response = [
    {"role": "assistant", "content": "Common LLM evaluation metrics include BLEU, ROUGE, and BERTScore..."}
]
memory.save_messages(conversation_id, turn1_response)

# Turn 2 - reference to previous context
turn2_messages = [
    {"role": "user", "content": "Which one is best for summarization?"}
]
memory.save_messages(conversation_id, turn2_messages)

# Load full conversation
full_conversation = memory.load_messages(conversation_id)

logger.info(f"✓ Full conversation ({len(full_conversation)} messages):")
for msg in full_conversation:
    logger.info(f"  [{msg['role']}] {msg['content']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Using Memory with an LLM
# MAGIC
# MAGIC Integrate memory with LLM calls for stateful conversations:

# COMMAND ----------

from openai import OpenAI
from arxiv_curator.config import load_config, get_env
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

# Create OpenAI client for Databricks
client = OpenAI(
    api_key=w.tokens.create(lifetime_seconds=1200).token_value,
    base_url=f"{w.config.host}/serving-endpoints"
)

def chat_with_memory(session_id: str, user_message: str, memory: LakebaseMemory) -> str:
    """Chat with LLM using session memory for context."""
    # Load previous messages
    previous_messages = memory.load_messages(session_id)
    
    # Build messages with system prompt
    messages = [
        {"role": "system", "content": "You are a helpful research assistant."}
    ] + previous_messages + [
        {"role": "user", "content": user_message}
    ]
    
    # Call LLM
    response = client.chat.completions.create(
        model=cfg.llm_endpoint,
        messages=messages,
    )
    
    assistant_response = response.choices[0].message.content
    
    # Save new messages to memory
    memory.save_messages(session_id, [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_response},
    ])
    
    return assistant_response

logger.info("✓ Chat function with memory created")

# COMMAND ----------

# Create a new session with memory
agent_session_id = f"agent-session-{uuid4()}"

# First query
response1 = chat_with_memory(agent_session_id, "What is RAG in the context of LLMs?", memory)
logger.info(f"Response 1: {response1[:200]}...")

# COMMAND ----------

# Follow-up query with context (memory is automatically loaded)
response2 = chat_with_memory(agent_session_id, "What are the main components?", memory)
logger.info(f"Response 2: {response2[:200]}...")

# COMMAND ----------

# View full conversation
full_agent_conversation = memory.load_messages(agent_session_id)

logger.info(f"✓ Full agent conversation ({len(full_agent_conversation)} messages):")
for i, msg in enumerate(full_agent_conversation, 1):
    role = msg["role"]
    content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
    logger.info(f"  {i}. [{role}] {content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook, we learned:
# MAGIC
# MAGIC 1. ✅ How to create and manage Lakebase instances
# MAGIC 2. ✅ How to use `LakebaseMemory` for session persistence
# MAGIC 3. ✅ How to save and load conversation history
# MAGIC 4. ✅ How to build stateful multi-turn conversations
# MAGIC 5. ✅ How to integrate memory with agents
# MAGIC
# MAGIC **Next Steps:**
# MAGIC - Implement session expiration
# MAGIC - Add conversation summarization
# MAGIC - Build a chatbot UI with session management
