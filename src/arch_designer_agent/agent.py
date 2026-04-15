"""Databricks Expert Architecture Agent — LLM-driven tool loop."""

import asyncio
import json
import re
import warnings
from collections.abc import Generator
from datetime import datetime
from typing import Any
from uuid import uuid4

import backoff
import mlflow
import nest_asyncio
import openai
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.database import DatabaseInstance, DatabaseInstanceState
from loguru import logger
from mlflow import MlflowClient
from mlflow.entities import SpanType
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksTable,
    DatabricksVectorSearchIndex,
)
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from pyspark.sql import SparkSession

from arch_designer_agent.agent_tools import DatabricksExpertTools
from arch_designer_agent.config import ProjectConfig
from arch_designer_agent.mcp import ToolRegistry, create_mcp_tools
from arch_designer_agent.memory import LakebaseMemory


class DatabricksExpertAgent(ResponsesAgent):
    """Architecture assistant that lets the LLM decide which tools to call.

    Tool pool (registered at startup):
    - Vector Search MCP tool   → searches the live kb_chunks_index (grounded evidence)
    - check_workspace_state    → live inventory of endpoints, pipelines, tables, models, jobs
    - profile_table            → schema, row count, null rates, sample rows for a Delta table
    - clarify_requirements     → asks user for missing constraints (only when none given)
    - health_check             → verifies kb Delta tables exist

    The LLM reads all tool descriptions and decides on every turn which
    tool(s) to call, in what order, and when it has enough context to
    return a final answer.  No hard-coded sequence.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a Databricks Architecture Design & Optimization Agent. "
        "You help users design, evaluate, and optimize data and AI architectures "
        "on the Databricks Lakehouse Platform.\n\n"
        "## STEP 1 — CLASSIFY THE QUERY\n\n"
        "Before doing anything, classify the query as exactly one of:\n\n"
        "**DESIGN** — user wants you to design, architect, build, or recommend a solution. "
        "Trigger words: 'design', 'architect', 'build', 'implement', 'set up', "
        "'how should I', 'what architecture', 'recommend', 'create a pipeline'.\n\n"
        "**FACTUAL** — user wants an explanation or definition of a concept. "
        "Trigger words: 'what is', 'explain', 'how does', 'when should I use', "
        "'what is the difference'.\n\n"
        "---\n\n"
        "## STEP 2 — CALL TOOLS (before writing any answer)\n\n"
        "Databricks documentation has already been retrieved and is shown above as KB results. "
        "DO NOT call the KB search tool again under any circumstances — it has already run.\n\n"
        "**IF DESIGN — follow these steps in strict order, one tool call per iteration:**\n\n"
        "  STEP 2a. Check if the query mentions ANY of: latency, speed, real-time, batch, "
        "streaming, compliance, cost, budget, team size, scale. "
        "If none → call clarify_requirements, then stop. "
        "If at least one → go to STEP 2b.\n\n"
        "  STEP 2b. Call check_workspace_state with relevant keywords. "
        "This is MANDATORY. Output ONLY this tool call — no text, no answer.\n\n"
        "  STEP 2c. After check_workspace_state returns, read the existing_relevant field:\n"
        "     - If existing_relevant is non-empty → your ONLY output for this iteration "
        "is a profile_table tool call on the most relevant entry. "
        "DO NOT write any answer text yet. The answer comes after profile_table returns.\n"
        "     - If existing_relevant is empty → skip to STEP 3 and write your answer.\n\n"
        "  STEP 2d. After profile_table returns → write your answer (STEP 3).\n\n"
        "**IF FACTUAL:**\n"
        "  Do NOT call any tools. Answer directly from the KB results shown above.\n\n"
        "---\n\n"
        "## STEP 3 — WRITE THE ANSWER\n\n"
        "**ALL answers — CITATIONS ARE MANDATORY:**\n"
        "  The KB results shown above contain a 'url' field on every entry. "
        "You MUST cite ONLY URLs that appear in the KB results above. "
        "NEVER generate, guess, or hallucinate URLs — if a URL is not in the KB results, "
        "do NOT use it. Every citation must come from the 'url' field of a KB result.\n"
        "  Write [description](url) inline in the text. "
        "You MUST end every answer with a '## References' section that lists every URL used. "
        "FORMAT: '- [Page title](url)' — one line per source. "
        "An answer missing inline citations OR missing the ## References section is WRONG.\n\n"
        "  - Ground every claim in the KB results. Never invent Databricks features.\n"
        "  - Cite ALL relevant KB results, especially solution accelerators and industry "
        "examples that match the use case (e.g. fraud detection accelerators for fraud queries).\n"
        "  - NEVER write: 'let me check', 'I would like to inspect', 'I can call', "
        "'before proceeding', 'would you like', 'to further optimize ... I would like to'. "
        "NEVER end with a follow-up question. NEVER ask the user for more information.\n\n"
        "**DESIGN answers only — constraint-aware, strict structure:**\n\n"
        "  CONSTRAINT FILTERING (MANDATORY — do this before writing anything):\n"
        "  1. Extract the user's explicit constraints from their query.\n"
        "  2. Use the constraint map below as a HARD FILTER on every recommendation.\n"
        "     Every component you recommend MUST be compatible with ALL stated constraints.\n"
        "     If a KB result mentions a technology that conflicts with a constraint,\n"
        "     do NOT include it — even if the KB result is otherwise relevant.\n\n"
        "  Constraint map — when the user says → you MUST / MUST NOT:\n\n"
        "  | User says | MUST use | MUST NOT use |\n"
        "  |---|---|---|\n"
        "  | batch, scheduled, weekly, daily"
        " | Databricks Jobs, spark.read, COPY INTO, scheduled SQL"
        " | Autoloader, Structured Streaming, Kafka, real-time, streaming triggers |\n"
        "  | real-time, streaming, low latency"
        " | Structured Streaming, Autoloader, Delta Live Tables, Kafka"
        " | Scheduled Jobs-only batch, COPY INTO for real-time |\n"
        "  | simplicity, small team"
        " | SQL Warehouse, Databricks SQL, Jobs UI, single notebooks"
        " | Multi-service streaming, Kafka, complex orchestration, microservices |\n"
        "  | budget, cost, cheap"
        " | Serverless SQL, spot/photon, auto-terminate, Jobs (no always-on)"
        " | Provisioned throughput, always-on endpoints, premium tiers |\n"
        "  | compliance, governance, audit"
        " | Unity Catalog, row/column security, audit logs, lineage"
        " | Open-access patterns without governance |\n"
        "  | scale, high volume, millions"
        " | Auto-scaling clusters, Photon, Z-order, liquid clustering"
        " | Single-node, pandas-only approaches |\n\n"
        "  If multiple constraints are present, apply ALL of them (intersection).\n"
        "  Example: 'batch + simplicity + budget' → use scheduled Databricks Jobs +\n"
        "  SQL Warehouse + spark.read; NEVER mention Autoloader, streaming, Kafka,\n"
        "  or always-on endpoints.\n\n"
        "  PART 1 — ## Architecture\n"
        "  Write a DETAILED architecture respecting ALL user constraints from the table above. "
        "DO NOT mention any existing table, pipeline, or model names anywhere in Part 1. "
        "Every claim must cite a KB url. Every component must reference a Databricks service. "
        "If a KB result mentions a technology that conflicts with user constraints, "
        "ignore it — do NOT include conflicting recommendations.\n\n"
        "  Structure the architecture with NUMBERED SECTIONS, one per pipeline stage. "
        "For each section provide:\n"
        "  1. **Service & purpose** — which Databricks service and why it fits.\n"
        "  2. **Technical details** — key configurations, formats, partitioning strategy, "
        "checkpoint or trigger intervals, scaling behaviour.\n"
        "  3. **Data flow** — what data enters, what transformations occur, what exits.\n"
        "  4. **Medallion layer** — explicitly label Bronze / Silver / Gold where applicable.\n"
        "  Aim for 5-8 architecture sections covering ingestion → processing → feature "
        "engineering → model training → serving → monitoring. "
        "Each section should be 3-5 sentences minimum.\n\n"
        "  CITATION REMINDER FOR ARCHITECTURE:\n"
        "  - EVERY URL you cite MUST come from the KB results above (the url field). "
        "Do NOT generate or guess URLs like docs.databricks.com/... — if a URL is not in the "
        "KB results, do NOT use it.\n"
        "  - Cite ALL relevant KB results, especially solution accelerators and industry "
        "examples (e.g. fraud detection accelerators for fraud queries, ML solution "
        "accelerators for ML queries). These are the MOST valuable references.\n"
        "  - Each architecture section should cite at least one KB result URL.\n\n"
        "  PART 2 — ## Existing Resources in Your Workspace\n"
        "  Read the resource_summary field from check_workspace_state output and follow "
        "its directive exactly.\n"
        "  If resource_summary says resources were FOUND:\n"
        "  - Include this section with a data-informed summary for each resource.\n"
        "  - For each profiled table:\n"
        "    - **<name>** (<row_count> rows, last modified <date>): can be utilized as [role].\n"
        "    - Key columns: list the columns most relevant to the use case from profile_table.\n"
        "    - Data quality: note any columns with null rates >10% from null_rates_pct.\n"
        "    - Insight: one sentence connecting a profile_table finding to an architecture decision\n"
        "      (e.g. 'row_count of 10M+ supports the streaming ingestion recommendation above').\n"
        "  - For resources that were NOT profiled (endpoints, pipelines, models, jobs), list as:\n"
        "    - **<name>**: already exists — can be leveraged as [role].\n\n"
        "  If resource_summary says NO resources matched:\n"
        "  - DO NOT include this section at all — no heading, no sentence, no mention.\n"
        "  - DO NOT write about the absence of resources (e.g. never say 'no existing "
        "resources were found', 'since no resources matched', etc.).\n"
        "  - NEVER reference internal JSON field names like 'has_relevant_resources' in your answer.\n"
        "  - Simply skip from ## Architecture directly to ## References.\n\n"
        "  ## References\n"
        "  List every URL cited above. This section is REQUIRED in every answer."
    )

    def __init__(
        self,
        spark: SparkSession | None = None,
        config: ProjectConfig | None = None,
        workspace_client: WorkspaceClient | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize the agent.

        Args:
            spark: SparkSession instance (optional, auto-created if not provided).
            config: ProjectConfig instance (optional, bootstrapped from ModelConfig).
            workspace_client: WorkspaceClient instance (optional, auto-created).
            system_prompt: Override for the default system prompt (optional).
        """
        # --- ModelConfig bootstrap (serving-time: no args passed) ---
        if spark is None or config is None:
            logger.info("No args passed — bootstrapping from MLflow ModelConfig (serving mode)")
            # Spark is created lazily by tools when they first need it.
            # This avoids SparkSession/DatabricksConnect issues at startup.
            mc = mlflow.models.ModelConfig()
            config = ProjectConfig(
                catalog=mc.get("catalog"),
                schema=mc.get("schema"),
                llm_endpoint=mc.get("llm_endpoint"),
                experiment_name="",  # not needed at serving time
            )
            # Overlay optional runtime fields
            for attr in ("system_prompt", "lakebase_instance", "genie_space_id"):
                val = mc.get(attr) if mc.get(attr) else None
                if val:
                    setattr(config, attr, val)

            # Load warehouse_id for live SQL access in serving mode
            wh_id = mc.get("warehouse_id") if mc.get("warehouse_id") else None
            if wh_id:
                config._warehouse_id = wh_id
                logger.info(f"  SQL warehouse configured: {wh_id}")

            # Load pre-scanned workspace snapshot from model_config (if present)
            ws_snap = mc.get("workspace_snapshot") if mc.get("workspace_snapshot") else None
            if ws_snap:
                config._workspace_snapshot = ws_snap
                logger.info(f"  Loaded workspace snapshot: {len(ws_snap.get('tables', []))} tables")

        self.spark = spark
        self.cfg = config
        self.w = workspace_client or WorkspaceClient()
        self.system_prompt = system_prompt or (config.system_prompt or self.DEFAULT_SYSTEM_PROMPT)

        # LLM client via SDK helper — handles auth automatically using whatever
        # credentials WorkspaceClient was configured with (SPN M2M OAuth when
        # DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET are set, PAT otherwise).
        # Token refresh is managed internally by the SDK; safe to reuse across calls.
        self._llm_client = self.w.serving_endpoints.get_open_ai_client()

        # Lakebase memory (optional — degrades gracefully if credentials fail)
        self._memory: LakebaseMemory | None = None
        if getattr(self.cfg, "lakebase_instance", None):
            try:
                lakebase_host = self._get_or_start_lakebase(self.cfg.lakebase_instance)
                self._memory = LakebaseMemory(
                    host=lakebase_host,
                    instance_name=self.cfg.lakebase_instance,
                )
                logger.info(f"  Lakebase memory enabled (instance={self.cfg.lakebase_instance})")
            except Exception as exc:
                logger.warning(f"  Lakebase init failed ({exc}) — running stateless")
        else:
            logger.info("  Lakebase not configured — running stateless (single-turn)")

        # Build and register all tools
        self.registry = ToolRegistry()
        self._load_tools()

    # ------------------------------------------------------------------
    # Lakebase instance management
    # ------------------------------------------------------------------

    def _get_or_start_lakebase(self, instance_name: str) -> str:
        """Return the read_write_dns for the Lakebase instance, starting it if stopped.

        Creates the instance if it does not yet exist (requires usage_policy_id in config).
        """
        try:
            instance = self.w.database.get_database_instance(instance_name)
            if instance.state == DatabaseInstanceState.STOPPED:
                logger.info(f"  Lakebase instance '{instance_name}' is stopped — starting...")
                self.w.database.update_database_instance(
                    name=instance_name,
                    database_instance=DatabaseInstance(name=instance_name, stopped=False),
                    update_mask="stopped",
                )
                instance = self.w.database.wait_get_database_instance_database_available(instance_name)
                logger.info("  Lakebase instance started")
            return instance.read_write_dns
        except NotFound:
            # Instance does not exist — create it
            logger.info(f"  Creating Lakebase instance '{instance_name}'...")
            usage_policy_id = getattr(self.cfg, "usage_policy_id", None)
            instance_spec = DatabaseInstance(name=instance_name, capacity="CU_1")
            if usage_policy_id:
                instance_spec = DatabaseInstance(
                    name=instance_name,
                    capacity="CU_1",
                    usage_policy_id=usage_policy_id,
                )
            instance = self.w.database.create_database_instance(instance_spec).result()
            logger.info(f"  Lakebase instance '{instance_name}' created")
            return instance.read_write_dns

    # ------------------------------------------------------------------
    # Tool loading
    # ------------------------------------------------------------------

    def _load_tools(self) -> None:
        """Register MCP tools + custom tools into the registry."""
        # Allow nested asyncio.run() inside gunicorn's event loop (serving mode)
        nest_asyncio.apply()

        # --- MCP: Vector Search index (kb_chunks_index) ---
        vs_mcp_url = f"{self.w.config.host}/api/2.0/mcp/vector-search/{self.cfg.catalog}/{self.cfg.schema}"
        try:
            mcp_tools = asyncio.run(create_mcp_tools(self.w, [vs_mcp_url]))
            self.registry.register_many(mcp_tools)
            logger.info(f"  Loaded {len(mcp_tools)} MCP tool(s): {[t.name for t in mcp_tools]}")
            if not mcp_tools:
                logger.warning("  MCP returned 0 tools; registering fallback KB search")
                self._register_fallback_kb_tool()
        except Exception as exc:
            logger.warning(f"  Could not load MCP tools ({exc}); registering fallback KB search")
            self._register_fallback_kb_tool()

        # --- Genie MCP (if configured) ---
        if getattr(self.cfg, "genie_space_id", None):
            genie_url = f"{self.w.config.host}/api/2.0/mcp/genie/{self.cfg.genie_space_id}"
            try:
                genie_tools = asyncio.run(create_mcp_tools(self.w, [genie_url]))
                self.registry.register_many(genie_tools)
                logger.info(f"  Loaded {len(genie_tools)} Genie MCP tool(s)")
            except Exception as exc:
                logger.warning(f"  Could not load Genie MCP tools ({exc})")

        # --- Custom local tools ---
        # Pass pre-scanned workspace snapshot and warehouse_id if available (serving mode).
        ws_snapshot = getattr(self.cfg, "_workspace_snapshot", None)
        warehouse_id = getattr(self.cfg, "_warehouse_id", None) or getattr(self.cfg, "warehouse_id", None)
        custom_tools = DatabricksExpertTools(
            spark=self.spark,
            config=self.cfg,
            workspace_client=self.w,
            workspace_snapshot=ws_snapshot,
            warehouse_id=warehouse_id,
        ).build_tool_infos()
        self.registry.register_many(custom_tools)
        logger.info(f"  Registered {len(custom_tools)} custom tool(s): {[t.name for t in custom_tools]}")

    def _register_fallback_kb_tool(self) -> None:
        """Register a direct Vector Search tool as fallback when MCP is unavailable.

        Uses the databricks-vectorsearch SDK directly, which works in the serving
        container because DatabricksVectorSearchIndex is declared as a resource.
        The tool name follows MCP naming convention (catalog__schema__index) so the
        KB pre-fetch logic in chat() still detects it via the "__" pattern.
        """
        from arch_designer_agent.mcp import ToolInfo

        index_name = f"{self.cfg.catalog}.{self.cfg.schema}.kb_chunks_index"
        tool_name = f"{self.cfg.catalog}__{self.cfg.schema}__kb_chunks_index"

        def _kb_search(query: str) -> str:
            from databricks.vector_search.client import VectorSearchClient

            vs_client = VectorSearchClient()
            index = vs_client.get_index(index_name=index_name)
            results = index.similarity_search(
                query_text=query,
                columns=["chunk_id", "text", "title", "source_type", "source_repo", "section_header", "url"],
                num_results=5,
                query_type="hybrid",
            )
            # Format results as list of dicts (same structure MCP returns)
            rows = results.get("result", {}).get("data_array", [])
            cols = [c["name"] for c in results.get("manifest", {}).get("columns", [])]
            formatted = [dict(zip(cols, row, strict=False)) for row in rows]
            return json.dumps(formatted, default=str)

        fallback_tool = ToolInfo(
            name=tool_name,
            spec={
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": (
                        "Search the Databricks knowledge base for architecture patterns, "
                        "best practices, and solution accelerators. Returns relevant "
                        "documentation chunks with URLs for citation."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for the knowledge base.",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
            exec_fn=_kb_search,
        )
        self.registry.register(fallback_tool)
        logger.info(f"  Registered fallback KB tool: {tool_name}")

    @mlflow.trace(span_type=SpanType.RETRIEVER, name="memory_load")
    def _load_memory(self, conversation_id: str) -> list[dict[str, Any]]:
        """Load prior messages from Lakebase for a session."""
        if self._memory:
            return self._memory.load_messages(conversation_id)
        return []

    @mlflow.trace(span_type=SpanType.CHAIN, name="memory_save")
    def _save_memory(self, conversation_id: str, messages: list[dict[str, Any]]) -> None:
        """Save new messages to Lakebase for a session."""
        if self._memory:
            self._memory.save_messages(conversation_id, messages)

    # Regex patterns for inline sentences about absence of resources.
    # These catch cases where the LLM ignores the system prompt and writes
    # about missing resources outside of the ## Existing Resources heading.
    _NO_RESOURCES_PATTERNS = [
        re.compile(r"(?i)since\s+has_relevant_resources\s+is\s+false[^.]*\.\s*"),
        re.compile(r"(?i)since\s+no\s+(existing\s+)?resources?\s+(were|was|is|are)\s+[^.]*\.\s*"),
        re.compile(r"(?i)there\s+are\s+no\s+existing\s+resources[^.]*\.\s*"),
        re.compile(r"(?i)no\s+existing\s+resources\s+(were|was)\s+(found|detected|identified|discovered)[^.]*\.\s*"),
        re.compile(r"(?i)has_relevant_resources[^.]*\.\s*"),
    ]

    @staticmethod
    def _strip_empty_resources_section(answer: str) -> str:
        """Remove Existing Resources section when it has no real content.

        Also strips inline sentences about absence of resources that the LLM
        may write despite being told not to (e.g. "Since has_relevant_resources
        is false..." or "No existing resources were found...").
        """
        # --- Pass 1: strip the ## Existing Resources heading + empty body ---
        lines = answer.split("\n")
        start = None
        end = None
        for i, line in enumerate(lines):
            if line.strip().startswith("## Existing Resources"):
                start = i
            elif start is not None and line.strip().startswith("## "):
                end = i
                break
        if start is not None:
            if end is None:
                end = len(lines)
            section = "\n".join(lines[start:end]).lower()
            has_real_data = any(kw in section for kw in ["rows,", "key columns", "null rate", "can be utilized as"])
            if not has_real_data:
                answer = "\n".join(lines[:start] + lines[end:])

        # --- Pass 2: strip inline "no resources" sentences ---
        for pattern in DatabricksExpertAgent._NO_RESOURCES_PATTERNS:
            answer = pattern.sub("", answer)

        # Clean up any leftover double blank lines from removals
        answer = re.sub(r"\n{3,}", "\n\n", answer)
        return answer.strip()

    @mlflow.trace(span_type=SpanType.TOOL)
    def _execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> object:
        """Execute a registered tool with MLflow tracing."""
        return self.registry.execute(tool_name, tool_args)

    # ------------------------------------------------------------------
    # Agent loop
    # ------------------------------------------------------------------

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def _call_llm(self, messages: list[dict[str, Any]]) -> openai.types.chat.ChatCompletion:
        """Call the LLM with exponential backoff on 429s and an MLflow LLM span.

        Why backoff?  Production serving endpoints occasionally return RateLimitError
        (HTTP 429) under burst load.  Backoff retries with exponential delay so the
        caller never has to handle transient errors manually.

        Why LLM span?  MLflow Tracing surfaces token-usage and latency per LLM call
        in the Databricks UI, enabling cost attribution and latency profiling across
        multiple tool-call rounds in the same agent turn.
        """
        tool_specs = self.registry.get_all_specs()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
            with mlflow.start_span(name="call_llm", span_type=SpanType.LLM) as span:
                span.set_inputs({"model": self.cfg.llm_endpoint, "num_messages": len(messages)})
                response = self._llm_client.chat.completions.create(
                    model=self.cfg.llm_endpoint,
                    messages=messages,
                    tools=tool_specs if tool_specs else None,
                )
                span.set_outputs(
                    {
                        "model": response.model,
                        "usage": response.usage.model_dump() if response.usage else {},
                    }
                )
                return response

    @mlflow.trace(span_type=SpanType.AGENT)
    def chat(
        self,
        user_message: str,
        conversation_id: str | None = None,
        max_iterations: int = 10,
    ) -> str:
        """Run an agentic conversation turn.

        The LLM sees all tool specs and decides which ones to call.
        The loop continues until the LLM returns a final text answer.

        Args:
            user_message: The user's query.
            conversation_id: Optional session key for multi-turn memory (requires
                lakebase_instance to be configured).  When provided, prior turns
                are loaded from Lakebase before the LLM call and the new turns
                are persisted afterward.
            max_iterations: Safety cap on tool-call rounds.

        Returns:
            Final text answer from the LLM.
        """
        prior_messages: list[dict[str, Any]] = []
        if conversation_id and self._memory:
            prior_messages = self._load_memory(conversation_id)
            if prior_messages:
                logger.info(f"  Loaded {len(prior_messages)} prior message(s) for conversation '{conversation_id}'")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            *prior_messages,
            {"role": "user", "content": user_message},
        ]
        save_from_index = 1 + len(prior_messages)

        # Pre-fetch KB results before the LLM loop so the LLM always starts with
        # grounded documentation context — no tool-ordering guard needed.
        # MCP KB tool names contain "__" (catalog__schema__index naming convention).
        mcp_tool_names = {t for t in self.registry.list_tools() if "__" in t}
        if mcp_tool_names:
            kb_tool = sorted(mcp_tool_names)[0]
            try:
                kb_context = self._execute_tool(kb_tool, {"query": user_message})
                prefetch_id = f"prefetch_{uuid4().hex[:8]}"
                # Inject as a synthetic assistant → tool pair so the LLM sees KB
                # results exactly as if it had called the tool itself first.
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": prefetch_id,
                                "type": "function",
                                "function": {
                                    "name": kb_tool,
                                    "arguments": json.dumps({"query": user_message}),
                                },
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": prefetch_id,
                        "content": str(kb_context),
                    }
                )
                logger.info(f"  → Tool called (pre-fetch): {kb_tool}(['query'])")
                logger.debug(f"  KB pre-fetch result (first 500 chars): {str(kb_context)[:500]}")
            except Exception as exc:
                logger.warning(f"  KB pre-fetch failed ({exc}); LLM can still call KB via tools")

        for iteration in range(max_iterations):
            logger.debug(f"Agent loop iteration {iteration + 1}/{max_iterations}")

            response = self._call_llm(messages)
            message = response.choices[0].message

            if not message.tool_calls:
                final_answer = message.content or ""

                # Guard: empty response — re-prompt once.
                if not final_answer.strip():
                    logger.warning("  LLM returned empty content — re-prompting for answer.")
                    messages.append({"role": "assistant", "content": ""})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response was empty. "
                                "Based on the KB results and workspace state above, "
                                "please write your complete answer now."
                            ),
                        }
                    )
                    continue

                # Guard: LLM wrote a tool call as plain text instead of using the
                # structured interface.  Use the structured tool-calling interface
                # to actually run it so the loop continues properly.
                tool_names = self.registry.list_tools()
                _plain_text_tool = next(
                    (name for name in tool_names if re.search(rf"\b{re.escape(name)}\s*\(", final_answer)),
                    None,
                )
                if _plain_text_tool:
                    logger.warning(
                        f"  LLM wrote plain-text tool call ({_plain_text_tool}) — executing via structured interface."
                    )
                    messages.append({"role": "assistant", "content": final_answer})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"You tried to call {_plain_text_tool} as plain text. "
                                "Call it now using the structured tool-calling interface."
                            ),
                        }
                    )
                    continue

                # LLM is done — return the final answer.
                final_answer = self._strip_empty_resources_section(final_answer)
                messages.append({"role": "assistant", "content": final_answer})
                if conversation_id and self._memory:
                    self._save_memory(conversation_id, messages[save_from_index:])
                return final_answer

            # LLM made structured tool calls — execute them all.
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in message.tool_calls
                    ],
                }
            )

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                logger.info(f"  → Tool called: {tool_name}({list(tool_args.keys())})")

                try:
                    result = self._execute_tool(tool_name, tool_args)
                except Exception as exc:
                    cause = exc
                    if hasattr(exc, "exceptions") and exc.exceptions:
                        cause = exc.exceptions[0]
                        logger.warning(
                            f"Tool {tool_name} raised ExceptionGroup. Sub-exception: {type(cause).__name__}: {cause}"
                        )
                    result = f"Error executing {tool_name}: {cause}"
                    logger.warning(result)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )

                # After check_workspace_state: auto-call profile_table if relevant
                # tables exist.  The LLM inconsistently skips this step, so we
                # enforce it here in Python.
                if tool_name == "check_workspace_state":
                    try:
                        ws_state = json.loads(str(result)) if isinstance(result, str) else result
                        relevant_tables = (
                            ws_state.get("tables", {}).get("existing_relevant", [])
                            if isinstance(ws_state, dict)
                            else []
                        )
                        logger.info(f"  → check_workspace_state: {len(relevant_tables)} relevant table(s)")
                        if relevant_tables:
                            top_table = relevant_tables[0]["name"]
                            full_table_name = f"{self.cfg.catalog}.{self.cfg.schema}.{top_table}"
                            logger.info(f"  → Auto-calling profile_table for '{full_table_name}'")
                            profile_id = f"auto_profile_{uuid4().hex[:8]}"
                            profile_args = {"table_name": full_table_name}
                            try:
                                profile_result = self._execute_tool("profile_table", profile_args)
                            except Exception as pexc:
                                profile_result = f"Error profiling table: {pexc}"
                                logger.warning(str(profile_result))
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": profile_id,
                                            "type": "function",
                                            "function": {
                                                "name": "profile_table",
                                                "arguments": json.dumps(profile_args),
                                            },
                                        }
                                    ],
                                }
                            )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": profile_id,
                                    "content": str(profile_result),
                                }
                            )
                    except Exception as parse_exc:
                        logger.info(f"  Auto profile_table skipped: {parse_exc}")

        if conversation_id and self._memory:
            self._save_memory(conversation_id, messages[save_from_index:])
        return "Max iterations reached without a final answer."

    # ------------------------------------------------------------------
    # MLflow pyfunc / ResponsesAgent serving interface
    # ------------------------------------------------------------------

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """MLflow pyfunc entrypoint — makes this agent deployable to Databricks Model Serving.

        Why ResponsesAgent?
        Subclassing ``mlflow.pyfunc.ResponsesAgent`` means ``mlflow.pyfunc.log_model``
        automatically knows how to call the agent at serving time.  Without it you need
        hand-written Python wrapper logic in the serving image.  With it, Model Serving
        calls ``predict()`` directly and handles request/response serialisation.

        The request ``custom_inputs`` dict accepts:
        - ``session_id``: str — optional session key for multi-turn Lakebase memory.
        """
        custom = request.custom_inputs or {}
        session_id = custom.get("session_id")
        user_message = next(
            (item.content for item in request.input if item.role == "user"),
            "",
        )
        answer = self.chat(user_message, conversation_id=session_id)
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(answer, str(uuid4()))],
            custom_outputs=request.custom_inputs,
        )

    def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Streaming variant of predict() required by the ResponsesAgent interface.

        Our agent runs a non-streaming internal loop (simpler, easier to debug).
        The full answer is buffered and emitted as a single done event — enough for
        Databricks Model Serving to stream one final chunk back to the client.
        """
        custom = request.custom_inputs or {}
        session_id = custom.get("session_id")
        user_message = next(
            (item.content for item in request.input if item.role == "user"),
            "",
        )
        answer = self.chat(user_message, conversation_id=session_id)
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(answer, str(uuid4())),
        )


# Internal tables that are agent infrastructure — excluded from workspace snapshots.
_SNAPSHOT_INTERNAL_TABLES = {"kb_chunks", "databricks_knowledge_base", "kb_chunks_index"}
_SNAPSHOT_INTERNAL_SUBSTRINGS = {
    "kb-pipeline",
    "kb_pipeline",
    "kb_chunks",
    "knowledge_base",
    "knowledge-base",
    "llmops_course",
    "online_index_view",
    "event_log_",
}


def _is_snapshot_internal(name: str) -> bool:
    """Return True if the resource name matches internal agent infrastructure."""
    name_lower = name.lower()
    return any(pat in name_lower for pat in _SNAPSHOT_INTERNAL_SUBSTRINGS)


def _prescan_workspace_state(w: WorkspaceClient, cfg: ProjectConfig) -> dict[str, Any]:
    """Scan workspace resources at registration time using full notebook permissions.

    Returns a snapshot dict that gets embedded in model_config for serving-time use.
    """
    snapshot: dict[str, Any] = {}

    # Tables
    try:
        all_tables = list(w.tables.list(catalog_name=cfg.catalog, schema_name=cfg.schema))
        snapshot["tables"] = [
            {"name": t.name, "comment": t.comment or ""}
            for t in all_tables
            if t.name not in _SNAPSHOT_INTERNAL_TABLES and not _is_snapshot_internal(t.name)
        ]
    except Exception as exc:
        logger.warning(f"Pre-scan tables failed: {exc}")
        snapshot["tables"] = []

    # Serving endpoints
    try:
        all_se = list(w.serving_endpoints.list())
        snapshot["serving_endpoints"] = [
            {"name": e.name, "state": str(e.state.config_update) if e.state else "unknown"}
            for e in all_se
            if not _is_snapshot_internal(e.name)
        ]
    except Exception as exc:
        logger.warning(f"Pre-scan serving endpoints failed: {exc}")
        snapshot["serving_endpoints"] = []

    # Pipelines
    try:
        all_dlt = list(w.pipelines.list_pipelines())
        snapshot["pipelines"] = [
            {
                "name": p.name,
                "state": str(p.latest_updates[0].state) if p.latest_updates else "no_runs",
            }
            for p in all_dlt
            if p.name and not _is_snapshot_internal(p.name)
        ]
    except Exception as exc:
        logger.warning(f"Pre-scan pipelines failed: {exc}")
        snapshot["pipelines"] = []

    # Registered models
    try:
        all_models = list(w.registered_models.list(catalog_name=cfg.catalog, schema_name=cfg.schema))
        snapshot["models"] = [{"name": m.name} for m in all_models if not _is_snapshot_internal(m.name)]
    except Exception as exc:
        logger.warning(f"Pre-scan models failed: {exc}")
        snapshot["models"] = []

    # Jobs
    try:
        all_jobs = list(w.jobs.list())
        snapshot["jobs"] = [
            {
                "name": j.settings.name,
                "schedule": str(j.settings.schedule.quartz_cron_expression)
                if j.settings and j.settings.schedule
                else "manual",
            }
            for j in all_jobs
            if j.settings and j.settings.name and not _is_snapshot_internal(j.settings.name)
        ]
    except Exception as exc:
        logger.warning(f"Pre-scan jobs failed: {exc}")
        snapshot["jobs"] = []

    # Vector search endpoints
    try:
        all_vs = list(w.vector_search_endpoints.list_endpoints().get("endpoints", []))
        snapshot["vector_search_endpoints"] = [
            {"name": e.get("name", ""), "state": e.get("endpoint_status", {}).get("state", "unknown")}
            for e in all_vs
            if not _is_snapshot_internal(e.get("name", ""))
        ]
    except Exception as exc:
        logger.warning(f"Pre-scan VS endpoints failed: {exc}")
        snapshot["vector_search_endpoints"] = []

    return snapshot


def log_register_agent(
    cfg: ProjectConfig,
    git_sha: str,
    run_id: str,
    agent_code_path: str,
    model_name: str,
    evaluation_metrics: dict | None = None,
) -> mlflow.entities.model_registry.RegisteredModel:
    """Log and register DatabricksExpertAgent to Unity Catalog via MLflow.

    Placed here (alongside the agent class) following the same convention as the
    course-code-hub repo, where log_register_agent lives at the bottom of agent.py.
    Keeping deployment logic next to the agent makes it easy to find and keeps
    evaluation.py focused purely on scoring.

    Args:
        cfg: Project configuration.
        git_sha: Git commit SHA for reproducibility tracking.
        run_id: Pipeline run identifier (set automatically by Databricks Jobs).
        agent_code_path: Path to this agent Python file (logged as pyfunc).
        model_name: Full Unity Catalog path e.g. mlops_dev.pratikhe.arch_agent.
        evaluation_metrics: Optional dict of eval metrics to log alongside the model.

    Returns:
        RegisteredModel from Unity Catalog.
    """
    w = WorkspaceClient()

    # Build resources list — only include optional endpoints if configured
    resources = [
        DatabricksServingEndpoint(endpoint_name=cfg.llm_endpoint),
        DatabricksVectorSearchIndex(index_name=f"{cfg.catalog}.{cfg.schema}.kb_chunks_index"),
        DatabricksTable(table_name=f"{cfg.catalog}.{cfg.schema}.kb_chunks"),
        DatabricksTable(table_name=f"{cfg.catalog}.{cfg.schema}.databricks_knowledge_base"),
    ]
    if cfg.embedding_endpoint:
        resources.append(DatabricksServingEndpoint(endpoint_name=cfg.embedding_endpoint))
    if cfg.warehouse_id:
        resources.append(DatabricksSQLWarehouse(warehouse_id=cfg.warehouse_id))

    # ------------------------------------------------------------------
    # Pre-scan workspace state at registration time (full notebook perms).
    # The serving container's scoped token cannot list UC tables, jobs, or
    # pipelines.  By scanning here and embedding the snapshot in model_config,
    # check_workspace_state reads from config instead of live SDK calls.
    # All discovered tables are also declared as DatabricksTable resources
    # so profile_table can read them at serving time.
    # ------------------------------------------------------------------
    workspace_snapshot = _prescan_workspace_state(w, cfg)
    logger.info(
        f"Pre-scanned workspace: {len(workspace_snapshot.get('tables', []))} tables, "
        f"{len(workspace_snapshot.get('jobs', []))} jobs, "
        f"{len(workspace_snapshot.get('pipelines', []))} pipelines, "
        f"{len(workspace_snapshot.get('models', []))} models, "
        f"{len(workspace_snapshot.get('serving_endpoints', []))} serving endpoints"
    )

    # Declare all user tables as resources so the serving token can read them
    declared_tables = {r.name for r in resources if isinstance(r, DatabricksTable)}
    for t in workspace_snapshot.get("tables", []):
        fqn = f"{cfg.catalog}.{cfg.schema}.{t['name']}"
        if fqn not in declared_tables:
            resources.append(DatabricksTable(table_name=fqn))
            declared_tables.add(fqn)

    # Only include fields the agent actually needs at serving time.
    model_config = {
        "catalog": cfg.catalog,
        "schema": cfg.schema,
        "llm_endpoint": cfg.llm_endpoint,
        "system_prompt": cfg.system_prompt,
        "lakebase_instance": cfg.lakebase_instance,
        "genie_space_id": getattr(cfg, "genie_space_id", None),
        "warehouse_id": cfg.warehouse_id or None,
        "workspace_snapshot": workspace_snapshot,
    }

    test_request = {"input": [{"role": "user", "content": "What is Delta Live Tables and when should I use it?"}]}

    mlflow.set_experiment(cfg.experiment_name)
    ts = datetime.now().strftime("%Y-%m-%d")

    with mlflow.start_run(
        run_name=f"arch-agent-{ts}",
        tags={"git_sha": git_sha, "run_id": run_id},
    ):
        # Explicit pip requirements — prevents MLflow from auto-inferring the
        # local project package (llmops-databricks-course-pratikhetan) which
        # is not on PyPI and would break container builds.
        # env_pack in register_model() captures the full notebook environment
        # (including the local package), so these are only a fallback.
        pip_reqs = [
            "cffi==1.17.1",
            "cloudpickle==3.1.1",
            "numpy==2.4.0",
            "pandas==2.3.0",
            "pyarrow==22.0.0",
            "databricks-sdk==0.85.0",
            "pydantic==2.11.7",
            "loguru==0.7.3",
            "python-dotenv==1.1.1",
            "databricks-vectorsearch==0.63",
            "openai==2.8.0",
            "databricks-mcp==0.4.0",
            "backoff==2.2.1",
            "mlflow==3.10.1",
            "nest-asyncio==1.6.0",
            "requests==2.32.3",
            "trafilatura==2.0.0",
            "langchain-text-splitters==0.3.8",
            "databricks-agents==1.8.2",
            "psycopg==3.3.2",
            "psycopg-pool==3.3.0",
            "psycopg[binary]==3.3.2",
        ]

        model_info = mlflow.pyfunc.log_model(
            name="agent",
            python_model=agent_code_path,
            resources=resources,
            input_example=test_request,
            model_config=model_config,
            pip_requirements=pip_reqs,
        )
        if evaluation_metrics:
            mlflow.log_metrics(evaluation_metrics)

    logger.info(f"Registering model: {model_name}")
    registered_model = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=model_name,
        env_pack="databricks_model_serving",
        tags={"git_sha": git_sha, "run_id": run_id},
    )
    logger.info(f"Registered version: {registered_model.version}")

    client = MlflowClient()
    logger.info("Setting alias 'champion'")
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=registered_model.version,
    )
    return registered_model
