"""Databricks Expert Architecture Agent — LLM-driven tool loop."""

import asyncio
import json
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.database import DatabaseInstance, DatabaseInstanceState
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

from arch_designer_agent.agent_tools import DatabricksExpertTools
from arch_designer_agent.config import ProjectConfig
from arch_designer_agent.mcp import DatabricksOAuth, ToolRegistry, create_mcp_tools
from arch_designer_agent.memory import LakebaseMemory


class DatabricksExpertAgent:
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
        "You have access to these tools — call them whenever they provide information "
        "you cannot derive from context alone:\n"
        "  • KB search (MCP)         — grounded evidence from Databricks documentation\n"
        "  • check_workspace_state   — live inventory of the user's workspace (endpoints, tables, models, jobs)\n"
        "  • profile_table           — schema, row count, nulls, sample rows for a Delta table\n"
        "  • clarify_requirements    — ask the user for constraints (ONLY if none were given)\n"
        "  • health_check            — verify KB Delta tables exist\n\n"
        "For DESIGN questions, your answer must be grounded in:\n"
        "  - What Databricks documentation says about the applicable patterns (KB search)\n"
        "  - What already exists in the workspace (relevant endpoints, tables, models, jobs)\n"
        "  - What the relevant data actually looks like (schema, volume, freshness, quality)\n"
        "Use your tools to gather whichever of these you need, in whatever order makes sense. "
        "Call tools multiple times if needed. Skip tools that would not add information. "
        "Your recommendation must explicitly reference what you found — "
        "actual resource names, column names, row counts, and KB sources.\n\n"
        "For FACTUAL questions: search the KB and answer with citations.\n\n"
        "Always cite specific sources. Never invent Databricks features."
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

        # Production-grade token provider:
        # - SPN path: DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET env vars
        #   set via load_spn_credentials().  M2M OAuth, auto-refreshes silently.
        # - PAT fallback: short-lived PAT created per chat() call (dev/notebook).
        self._oauth = DatabricksOAuth(self.w)

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

    # ------------------------------------------------------------------
    # Agent loop
    # ------------------------------------------------------------------

    def _make_llm_client(self) -> OpenAI:
        """Create an OpenAI client with a fresh, valid token.

        Called once per chat() turn so SPN tokens are always current.
        With SPN, DatabricksOAuth returns a cached + auto-refreshed OAuth token
        (no network call unless the token is expiring).
        With PAT fallback, creates a new 1-hour PAT.
        """
        return OpenAI(
            api_key=self._oauth.token(),
            base_url=f"{self.w.config.host}/serving-endpoints",
        )

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
        # Fresh client per chat() call — token is cheap to obtain for SPN
        # (SDK caches and auto-refreshes internally) and correct by design.
        llm_client = self._make_llm_client()

        prior_messages: list[dict[str, Any]] = []
        if conversation_id and self._memory:
            prior_messages = self._memory.load_messages(conversation_id)
            if prior_messages:
                logger.info(f"  Loaded {len(prior_messages)} prior message(s) for conversation '{conversation_id}'")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            *prior_messages,
            {"role": "user", "content": user_message},
        ]
        save_from_index = 1 + len(prior_messages)
        tool_specs = self.registry.get_all_specs()

        for iteration in range(max_iterations):
            logger.debug(f"Agent loop iteration {iteration + 1}/{max_iterations}")

            response = llm_client.chat.completions.create(
                model=self.cfg.llm_endpoint,
                messages=messages,
                tools=tool_specs if tool_specs else None,
            )
            message = response.choices[0].message

            if not message.tool_calls:
                final_answer = message.content or ""

                # Guard: detect when the LLM writes a tool call as plain text
                # instead of using the structured tool-calling interface.
                # e.g. 'profile_table(table_name="...")' returned as text content.
                tool_names = self.registry.list_tools()
                if any(final_answer.strip().startswith(name + "(") for name in tool_names):
                    logger.warning(
                        "  LLM returned a tool call as plain text — re-prompting to use "
                        "the structured tool-calling interface."
                    )
                    messages.append({"role": "assistant", "content": final_answer})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You must call tools using the structured tool-calling interface, "
                                "not by writing the call as plain text. "
                                "Please call the tool now using the proper tool-calling mechanism."
                            ),
                        }
                    )
                    continue

                # LLM is done — return the final answer
                messages.append({"role": "assistant", "content": final_answer})
                if conversation_id and self._memory:
                    self._memory.save_messages(conversation_id, messages[save_from_index:])
                return final_answer

            # LLM wants to call one or more tools — execute them
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
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
                    result = self.registry.execute(tool_name, tool_args)
                except Exception as exc:
                    # Unwrap ExceptionGroup (Python 3.11+ TaskGroup errors) so the
                    # real cause is visible in logs rather than "unhandled errors in
                    # a TaskGroup (1 sub-exception)".
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

        if conversation_id and self._memory:
            self._memory.save_messages(conversation_id, messages[save_from_index:])
        return "Max iterations reached without a final answer."
