# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.2: Databricks Architecture Designer Agent
# MAGIC
# MAGIC ## What this notebook shows
# MAGIC
# MAGIC How to turn the vector-search knowledge base you built in weeks 1–2 into a
# MAGIC **true agentic assistant** where the **LLM decides which tools to call** based on
# MAGIC your query — not a hardcoded pipeline.
# MAGIC
# MAGIC ## Tool pool
# MAGIC
# MAGIC | Tool | Type | When LLM calls it |
# MAGIC |---|---|---|
# MAGIC | `{catalog}__{schema}__kb_chunks_index` | **MCP** (Vector Search) | Grounding / evidence retrieval |
# MAGIC | `check_workspace_state` | Custom Python | Design questions — fetches live inventory filtered by query keywords |
# MAGIC | `profile_table` | Custom Python | After finding relevant tables — schema, row count, nulls, sample rows |
# MAGIC | `clarify_requirements` | Custom Python | Design requested but zero constraints given |
# MAGIC | `health_check` | Custom Python | Agent suspects missing KB data |
# MAGIC
# MAGIC ## Agent loop (same pattern as SimpleAgent in lecture 3.1)
# MAGIC
# MAGIC ```
# MAGIC User query
# MAGIC   ↓
# MAGIC LLM sees all tool specs → decides which tool(s) to call
# MAGIC   ↓
# MAGIC check_workspace_state(focus_keywords) → live inventory filtered to query topic
# MAGIC   ↓
# MAGIC KB search (1 or more times) → grounded documentation evidence
# MAGIC   ↓
# MAGIC LLM reasons from inventory + evidence → drafts options → picks best
# MAGIC   ↓
# MAGIC no tool_calls in response → return final answer citing what already exists
# MAGIC ```

# COMMAND ----------

import asyncio

import nest_asyncio
from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import SparkSession

from arch_designer_agent.agent import DatabricksExpertAgent
from arch_designer_agent.agent_tools import DatabricksExpertTools
from arch_designer_agent.config import get_env, load_config
from arch_designer_agent.mcp import ToolRegistry, create_mcp_tools

# Enable nested async (required in Databricks notebooks)
nest_asyncio.apply()

# COMMAND ----------
# -- Setup ------------------------------------------------------------------

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)
w = WorkspaceClient()

w = WorkspaceClient()
for inst in w.database.list_database_instances():
    print(inst.name, inst.state, inst.read_write_dns)

logger.info(f"Environment : {env}")
logger.info(f"Catalog     : {cfg.catalog}")
logger.info(f"Schema      : {cfg.schema}")
logger.info(f"LLM endpoint: {cfg.llm_endpoint}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Load MCP tools from the Vector Search server
# MAGIC
# MAGIC Databricks exposes every vector search index in a catalog/schema as an MCP tool
# MAGIC automatically — no code required to define it.
# MAGIC
# MAGIC ### MCP URL format
# MAGIC ```
# MAGIC {workspace_host}/api/2.0/mcp/vector-search/{catalog}/{schema}
# MAGIC ```
# MAGIC Each index becomes a tool named `{catalog}__{schema}__{index_name}`.

# COMMAND ----------

host = w.config.host
vs_mcp_url = f"{host}/api/2.0/mcp/vector-search/{cfg.catalog}/{cfg.schema}"
logger.info(f"Vector Search MCP URL: {vs_mcp_url}")

# Load tools from MCP server
mcp_tools = asyncio.run(create_mcp_tools(w, [vs_mcp_url]))

logger.info(f"MCP tools loaded ({len(mcp_tools)}):")
for tool in mcp_tools:
    logger.info(f"  - {tool.name}")
    logger.info(f"    {tool.spec['function']['description'][:100]}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Create custom tools
# MAGIC
# MAGIC Each custom tool is a `ToolInfo` with:
# MAGIC - `name` — how the LLM refers to it
# MAGIC - `spec` — OpenAI function-calling JSON (description + parameter schema)
# MAGIC - `exec_fn` — Python function that runs when called

# COMMAND ----------

custom_tool_infos = DatabricksExpertTools(spark=spark, config=cfg).build_tool_infos()

logger.info(f"Custom tools ({len(custom_tool_infos)}):")
for tool in custom_tool_infos:
    logger.info(f"  - {tool.name}")
    logger.info(f"    {tool.spec['function']['description'][:100]}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Register all tools into one ToolRegistry
# MAGIC
# MAGIC `ToolRegistry` holds every tool the agent can call, regardless of whether it
# MAGIC came from MCP or was a custom Python function.

# COMMAND ----------

registry = ToolRegistry()
registry.register_many(mcp_tools)
registry.register_many(custom_tool_infos)

logger.info(f"Total tools registered: {len(registry.list_tools())}")
logger.info(f"Tools: {registry.list_tools()}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Show tool specs (what the LLM sees)
# MAGIC
# MAGIC These JSON specs are passed to the LLM on every call.
# MAGIC The LLM reads `name`, `description`, and `parameters` to decide which tool to use.

# COMMAND ----------

logger.info("Tool specifications sent to LLM:")
logger.info("=" * 80)
for spec in registry.get_all_specs():
    fn = spec["function"]
    logger.info(f"\nTool  : {fn['name']}")
    logger.info(f"Desc  : {fn['description'][:120]}")
    params = list(fn.get("parameters", {}).get("properties", {}).keys())
    logger.info(f"Params: {params}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Build the agent
# MAGIC
# MAGIC `DatabricksExpertAgent` wraps:
# MAGIC - The OpenAI client (pointing at Databricks serving endpoint)
# MAGIC - The tool registry
# MAGIC - A system prompt that tells the LLM *when* to use each tool

# COMMAND ----------

agent = DatabricksExpertAgent(
    spark=spark,
    config=cfg,
    workspace_client=w,
    system_prompt=DatabricksExpertAgent.DEFAULT_SYSTEM_PROMPT,
)

logger.info("Agent ready.")
logger.info(f"Tools registered: {agent.registry.list_tools()}")
logger.info(f"System prompt preview: {agent.system_prompt[:200]}...")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Test 1 — factual question (agent uses only vector search MCP tool)
# MAGIC
# MAGIC For a factual question the LLM should call the MCP vector-search tool to
# MAGIC retrieve grounded evidence, then answer with citations.

# COMMAND ----------

query1 = "What is Delta Live Tables and when should I use it on Databricks?"

logger.info(f"Query: {query1}")
logger.info("=" * 80)

answer1 = agent.chat(query1)
logger.info(f"\nAnswer:\n{answer1}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Test 2 — architecture design question (agent uses multiple tools)
# MAGIC
# MAGIC For this query the LLM should chain:
# MAGIC 1. Vector search (grounding)
# MAGIC 2. generate_architecture_options
# MAGIC 3. analyze_tradeoffs
# MAGIC 4. validate_design
# MAGIC ...before returning a final recommendation.

# COMMAND ----------

query2 = (
    "Design a real-time Databricks architecture for fraud detection with strict compliance controls and low latency."
)

logger.info(f"Query: {query2}")
logger.info("=" * 80)

answer2 = agent.chat(query2)
logger.info(f"\nAnswer:\n{answer2}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Test 3 — cost-focused question (different constraint, different tool priority)
# MAGIC
# MAGIC Keywords like "budget" or "cheap" should lead the agent to rank cost-optimized
# MAGIC options higher even though the design toolchain is the same.

# COMMAND ----------

query3 = (
    "We need a budget-friendly batch architecture on Databricks for a small "
    "analytics team. Simplicity matters more than speed."
)

logger.info(f"Query: {query3}")
logger.info("=" * 80)

answer3 = agent.chat(query3)
logger.info(f"\nAnswer:\n{answer3}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Direct tool execution (optional — for debugging)
# MAGIC
# MAGIC You can call any registered tool directly without going through the agent loop.

# COMMAND ----------

# Call the MCP vector search tool directly
vs_tool_name = f"{cfg.catalog}__{cfg.schema}__kb_chunks_index"

if vs_tool_name in registry.list_tools():
    direct_result = registry.execute(
        vs_tool_name,
        {"query": "medallion architecture best practices"},
    )
    logger.info("Direct MCP tool result (first 500 chars):")
    logger.info(str(direct_result)[:500])
else:
    logger.warning(f"Tool '{vs_tool_name}' not found. Available: {registry.list_tools()}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Lakebase connectivity test (optional — run before multi-turn)
# MAGIC
# MAGIC Verifies the full stack: SDK auth → token generation → PostgreSQL connection →
# MAGIC table DDL → INSERT → SELECT.  Skip if `lakebase_instance` is not set in config.

# COMMAND ----------

if cfg.lakebase_instance:
    from uuid import uuid4

    import psycopg

    from arch_designer_agent.memory import LakebaseMemory

    # Resolve host (reuses the instance the agent already started)
    lakebase_host = agent._get_or_start_lakebase(cfg.lakebase_instance)
    logger.info(f"Lakebase host: {lakebase_host}")

    mem = LakebaseMemory(host=lakebase_host, instance_name=cfg.lakebase_instance)

    # Round-trip via LakebaseMemory API
    test_session = f"connectivity-test-{uuid4()}"
    mem.save_messages(
        test_session,
        [
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong"},
        ],
    )
    rows = mem.load_messages(test_session)
    assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"
    logger.info(f"✓ Round-trip OK — {len(rows)} messages stored and retrieved")

    # Verify via raw psycopg query
    conn_str = mem._get_connection_string()
    with psycopg.connect(conn_str) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM session_messages WHERE session_id = %s",
            (test_session,),
        ).fetchone()[0]
    logger.info(f"✓ Direct psycopg query: {count} rows for session '{test_session}'")
else:
    logger.info("Skipping Lakebase test — lakebase_instance not set in config")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 11. Multi-turn conversation (requires lakebase_instance in config)
# MAGIC
# MAGIC Pass the same `conversation_id` across calls.  Each turn loads prior messages
# MAGIC from Lakebase and appends only the new turns — so the LLM always sees the full
# MAGIC context window but writes are cheap (just the delta).

# COMMAND ----------

if cfg.lakebase_instance:
    session_id = "fraud-design-session-001"

    # Turn 1 — initial design question
    logger.info("=== Turn 1 ===")
    answer_t1 = agent.chat(
        "Design a real-time fraud detection architecture on Databricks.",
        conversation_id=session_id,
    )
    logger.info(f"Answer T1:\n{answer_t1[:500]}...")

    # Turn 2 — follow-up; LLM has full context from turn 1
    logger.info("\n=== Turn 2 ===")
    answer_t2 = agent.chat(
        "Focus more on the compliance and audit trail aspects.",
        conversation_id=session_id,
    )
    logger.info(f"Answer T2:\n{answer_t2[:500]}...")

    # Turn 3 — another follow-up
    logger.info("\n=== Turn 3 ===")
    answer_t3 = agent.chat(
        "What Databricks features specifically help with GDPR requirements here?",
        conversation_id=session_id,
    )
    logger.info(f"Answer T3:\n{answer_t3[:500]}...")
else:
    logger.info("Skipping multi-turn test — set lakebase_instance in project_config.yml to enable")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Concept | What we built |
# MAGIC |---|---|
# MAGIC | MCP tools | Vector Search index exposed as agent-callable tool |
# MAGIC | Custom tools | check_workspace_state, profile_table, clarify_requirements, health_check |
# MAGIC | ToolRegistry | Single registry for all tools regardless of source |
# MAGIC | LLM-driven routing | LLM reads tool descriptions, decides which to call |
# MAGIC | Agentic loop | Keeps calling tools until LLM returns final answer |
# MAGIC | Lakebase memory | Multi-turn conversation history in PostgreSQL (millisecond writes) |
# MAGIC
# MAGIC **Key difference from RAG**: in pure RAG you always call vector search once and
# MAGIC answer. Here the **LLM plans its own tool chain** per query — different questions
# MAGIC trigger different sequences of tool calls.
