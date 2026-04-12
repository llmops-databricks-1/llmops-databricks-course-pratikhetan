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
import openai
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.database import DatabaseInstance, DatabaseInstanceState
from loguru import logger
from mlflow import MlflowClient
from mlflow.entities import SpanType
from mlflow.models.resources import (
    DatabricksServingEndpoint,
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
        "You MUST cite every source you use. Write [description](url) inline in the text. "
        "You MUST end every answer with a '## References' section that lists every URL used. "
        "FORMAT: '- [Page title](url)' — one line per source. "
        "An answer missing inline citations OR missing the ## References section is WRONG.\n\n"
        "  - Ground every claim in the KB results. Never invent Databricks features.\n"
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
        "  | batch, scheduled, weekly, daily | Databricks Jobs, spark.read, COPY INTO, scheduled SQL | Autoloader, Structured Streaming, Kafka, real-time, streaming triggers |\n"
        "  | real-time, streaming, low latency | Structured Streaming, Autoloader, Delta Live Tables, Kafka | Scheduled Jobs-only batch, COPY INTO for real-time |\n"
        "  | simplicity, small team | SQL Warehouse, Databricks SQL, Jobs UI, single notebooks | Multi-service streaming, Kafka, complex orchestration, microservices |\n"
        "  | budget, cost, cheap | Serverless SQL, spot/photon, auto-terminate, Jobs (no always-on) | Provisioned throughput, always-on endpoints, premium tiers |\n"
        "  | compliance, governance, audit | Unity Catalog, row/column security, audit logs, lineage | Open-access patterns without governance |\n"
        "  | scale, high volume, millions | Auto-scaling clusters, Photon, Z-order, liquid clustering | Single-node, pandas-only approaches |\n\n"
        "  If multiple constraints are present, apply ALL of them (intersection).\n"
        "  Example: 'batch + simplicity + budget' → use scheduled Databricks Jobs +\n"
        "  SQL Warehouse + spark.read; NEVER mention Autoloader, streaming, Kafka,\n"
        "  or always-on endpoints.\n\n"
        "  PART 1 — ## Architecture\n"
        "  Write the architecture respecting ALL user constraints from the table above. "
        "DO NOT mention any existing table, pipeline, or model names anywhere in Part 1. "
        "Every claim must cite a KB url. Every component must reference a Databricks service. "
        "If a KB result mentions a technology that conflicts with user constraints, "
        "ignore it — do NOT include conflicting recommendations.\n\n"
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
        spark: SparkSession,
        config: ProjectConfig,
        workspace_client: WorkspaceClient | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.spark = spark
        self.cfg = config
        self.w = workspace_client or WorkspaceClient()
        self.system_prompt = system_prompt or (config.system_prompt or self.DEFAULT_SYSTEM_PROMPT)

        # LLM client via SDK helper — handles auth automatically using whatever
        # credentials WorkspaceClient was configured with (SPN M2M OAuth when
        # DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET are set, PAT otherwise).
        # Token refresh is managed internally by the SDK; safe to reuse across calls.
        self._llm_client = self.w.serving_endpoints.get_open_ai_client()

        # Lakebase memory (optional — only if lakebase_instance is configured)
        self._memory: LakebaseMemory | None = None
        if getattr(self.cfg, "lakebase_instance", None):
            lakebase_host = self._get_or_start_lakebase(self.cfg.lakebase_instance)
            self._memory = LakebaseMemory(
                host=lakebase_host,
                instance_name=self.cfg.lakebase_instance,
            )
            logger.info(f"  Lakebase memory enabled (instance={self.cfg.lakebase_instance})")
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
        # --- MCP: Vector Search index (kb_chunks_index) ---
        vs_mcp_url = f"{self.w.config.host}/api/2.0/mcp/vector-search/{self.cfg.catalog}/{self.cfg.schema}"
        try:
            mcp_tools = asyncio.run(create_mcp_tools(self.w, [vs_mcp_url]))
            self.registry.register_many(mcp_tools)
            logger.info(f"  Loaded {len(mcp_tools)} MCP tool(s): {[t.name for t in mcp_tools]}")
        except Exception as exc:
            logger.warning(f"  Could not load MCP tools ({exc}); continuing with custom tools only")

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
        custom_tools = DatabricksExpertTools(
            spark=self.spark, config=self.cfg, workspace_client=self.w
        ).build_tool_infos()
        self.registry.register_many(custom_tools)
        logger.info(f"  Registered {len(custom_tools)} custom tool(s): {[t.name for t in custom_tools]}")

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
        lines = answer.split('\n')
        start = None
        end = None
        for i, line in enumerate(lines):
            if line.strip().startswith('## Existing Resources'):
                start = i
            elif start is not None and line.strip().startswith('## '):
                end = i
                break
        if start is not None:
            if end is None:
                end = len(lines)
            section = '\n'.join(lines[start:end]).lower()
            has_real_data = any(kw in section for kw in [
                "rows,", "key columns", "null rate", "can be utilized as"
            ])
            if not has_real_data:
                answer = '\n'.join(lines[:start] + lines[end:])

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
    resources = [
        DatabricksServingEndpoint(endpoint_name=cfg.llm_endpoint),
        DatabricksServingEndpoint(endpoint_name=cfg.embedding_endpoint),
        DatabricksVectorSearchIndex(index_name=f"{cfg.catalog}.{cfg.schema}.kb_chunks_index"),
        DatabricksTable(table_name=f"{cfg.catalog}.{cfg.schema}.kb_chunks"),
        DatabricksTable(table_name=f"{cfg.catalog}.{cfg.schema}.databricks_knowledge_base"),
    ]

    model_config = {
        "catalog": cfg.catalog,
        "schema": cfg.schema,
        "llm_endpoint": cfg.llm_endpoint,
        "embedding_endpoint": cfg.embedding_endpoint,
        "vector_search_endpoint": cfg.vector_search_endpoint,
        "system_prompt": cfg.system_prompt,
        "lakebase_instance": cfg.lakebase_instance,
    }

    test_request = {"input": [{"role": "user", "content": "What is Delta Live Tables and when should I use it?"}]}

    mlflow.set_experiment(cfg.experiment_name)
    ts = datetime.now().strftime("%Y-%m-%d")

    with mlflow.start_run(
        run_name=f"arch-agent-{ts}",
        tags={"git_sha": git_sha, "run_id": run_id},
    ):
        model_info = mlflow.pyfunc.log_model(
            name="agent",
            python_model=agent_code_path,
            resources=resources,
            input_example=test_request,
            model_config=model_config,
        )
        if evaluation_metrics:
            mlflow.log_metrics(evaluation_metrics)

    logger.info(f"Registering model: {model_name}")
    registered_model = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=model_name,
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


# Required by MLflow when this file is used as a pyfunc code model
# (mlflow.pyfunc.log_model(python_model="agent.py")).
# MLflow needs to know which class to instantiate at serving time.
mlflow.models.set_model(DatabricksExpertAgent)
