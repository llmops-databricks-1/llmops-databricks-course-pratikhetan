# Databricks notebook source
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import DatabaseInstance, DatabaseInstanceState
from uuid import uuid4
from loguru import logger

from arxiv_curator.memory import LakebaseMemory
from arxiv_curator.config import load_config, get_env


scope_name = "arxiv-agent-scope"
os.environ["DATABRICKS_CLIENT_ID"] = dbutils.secrets.get(scope_name, "client_id")
os.environ["DATABRICKS_CLIENT_SECRET"] = dbutils.secrets.get(scope_name, "client_secret")
 

w = WorkspaceClient()
os.environ["DATABRICKS_HOST"] = w.config.host

# COMMAND ----------
instance_name = "arxiv-agent-instance"
instance = w.database.get_database_instance(instance_name)
lakebase_host = instance.read_write_dns

memory = LakebaseMemory(
    host=lakebase_host,
    instance_name=instance_name,
)

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
