"""Custom tool handlers and ToolInfo wrappers for the Databricks Expert Assistant."""

import json
from typing import Any

from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

from arch_designer_agent.config import ProjectConfig
from arch_designer_agent.mcp import ToolInfo


class DatabricksExpertTools:
    """Custom tool handlers for the Databricks Architecture Assistant."""

    def __init__(
        self,
        spark: SparkSession,
        config: ProjectConfig,
        workspace_client: WorkspaceClient | None = None,
    ) -> None:
        self.spark = spark
        self.cfg = config
        self.w = workspace_client or WorkspaceClient()
        self.kb_table = f"{config.catalog}.{config.schema}.databricks_knowledge_base"
        self.chunks_table = f"{config.catalog}.{config.schema}.kb_chunks"

    def check_workspace_state(self, args: dict[str, Any]) -> dict[str, Any]:
        """Return live workspace resources that match the user's focus keywords.

        Each section returns:
          relevant: resources whose name contains at least one focus keyword.
          total_scanned: total count of that resource type in the workspace.

        If focus_keywords is empty, all resources are returned.
        The LLM interprets relevance and uses this data to tailor recommendations.
        """
        focus = [kw.lower() for kw in args.get("focus_keywords", [])]

        def matches(name: str) -> bool:
            if not focus:
                return True
            name_lower = name.lower()
            return any(kw in name_lower for kw in focus)

        state: dict[str, Any] = {
            "focus_keywords_used": focus,
            "catalog": self.cfg.catalog,
            "schema": self.cfg.schema,
        }

        try:
            all_vs = list(self.w.vector_search_endpoints.list())
            state["vector_search_endpoints"] = {
                "relevant": [
                    {"name": e.name, "state": str(e.endpoint_status.state) if e.endpoint_status else "unknown"}
                    for e in all_vs if matches(e.name)
                ],
                "total_scanned": len(all_vs),
            }
        except Exception as exc:
            state["vector_search_endpoints"] = {"error": str(exc)}

        try:
            all_se = list(self.w.serving_endpoints.list())
            state["serving_endpoints"] = {
                "relevant": [
                    {"name": e.name, "state": str(e.state.config_update) if e.state else "unknown"}
                    for e in all_se if matches(e.name)
                ],
                "total_scanned": len(all_se),
            }
        except Exception as exc:
            state["serving_endpoints"] = {"error": str(exc)}

        try:
            all_dlt = list(self.w.pipelines.list_pipelines())
            state["dlt_pipelines"] = {
                "relevant": [
                    {
                        "name": p.name,
                        "state": str(p.latest_updates[0].state) if p.latest_updates else "no_runs",
                    }
                    for p in all_dlt if p.name and matches(p.name)
                ],
                "total_scanned": len(all_dlt),
            }
        except Exception as exc:
            state["dlt_pipelines"] = {"error": str(exc)}

        try:
            # Use SDK (Unity Catalog REST API) instead of Spark — avoids Spark Connect
            # DATA_SOURCE_NOT_FOUND errors on tables with unsupported formats.
            all_tables = list(self.w.tables.list(
                catalog_name=self.cfg.catalog,
                schema_name=self.cfg.schema,
            ))
            table_infos = [
                {"name": t.name, "comment": t.comment or ""}
                for t in all_tables
            ]

            all_table_names = [t["name"] for t in table_infos]
            state["tables"] = {
                "relevant": [
                    {"name": t["name"], "comment": t["comment"]}
                    for t in table_infos
                    if matches(t["name"]) or matches(t["comment"])
                ],
                "all": all_table_names,
                "total_scanned": len(table_infos),
            }
        except Exception as exc:
            state["tables"] = {"error": str(exc)}

        try:
            all_models = list(
                self.w.registered_models.list(
                    catalog_name=self.cfg.catalog, schema_name=self.cfg.schema
                )
            )
            state["registered_models"] = {
                "relevant": [
                    {"name": m.name}
                    for m in all_models if matches(m.name)
                ],
                "total_scanned": len(all_models),
            }
        except Exception as exc:
            state["registered_models"] = {"error": str(exc)}

        try:
            all_jobs = list(self.w.jobs.list())
            state["jobs"] = {
                "relevant": [
                    {
                        "name": j.settings.name,
                        "schedule": str(j.settings.schedule.quartz_cron_expression)
                        if j.settings and j.settings.schedule else "manual",
                    }
                    for j in all_jobs if j.settings and matches(j.settings.name)
                ],
                "total_scanned": len(all_jobs),
            }
        except Exception as exc:
            state["jobs"] = {"error": str(exc)}

        return state

    def health_check(self, args: dict[str, Any]) -> dict[str, Any]:
        _ = args
        kb_exists = self.spark.catalog.tableExists(self.kb_table)
        chunks_exists = self.spark.catalog.tableExists(self.chunks_table)
        return {
            "kb_table_exists": kb_exists,
            "chunks_table_exists": chunks_exists,
            "status": "ok" if kb_exists and chunks_exists else "degraded",
        }

    def profile_table(self, args: dict[str, Any]) -> dict[str, Any]:
        """Return schema, size, freshness, null rates, and sample rows for a Delta table.

        The LLM uses this to ground recommendations in actual data characteristics:
        - column names/types  → what transformations are needed
        - row_count           → streaming vs batch threshold
        - last_modified       → is the pipeline healthy / data fresh
        - null_rates          → data quality issues to address in Silver layer
        - sample_rows         → domain understanding (currency, IDs, labels)
        """
        table_name = args.get("table_name", "")
        if not table_name:
            return {"error": "table_name is required"}

        result: dict[str, Any] = {"table_name": table_name}

        try:
            df = self.spark.table(table_name)
            result["columns"] = [
                {"name": f.name, "type": str(f.dataType), "nullable": f.nullable}
                for f in df.schema.fields
            ]
        except Exception as exc:
            return {"error": f"Could not read table '{table_name}': {exc}"}

        try:
            result["row_count"] = df.count()
        except Exception as exc:
            result["row_count"] = {"error": str(exc)}

        try:
            history = self.spark.sql(f"DESCRIBE HISTORY {table_name} LIMIT 2")
            rows = history.collect()
            result["last_modified"] = str(rows[0]["timestamp"]) if rows else "unknown"
            result["recent_operations"] = [r["operation"] for r in rows]
        except Exception as exc:
            result["last_modified"] = {"error": str(exc)}

        try:
            total = result.get("row_count", 0)
            if isinstance(total, int) and total > 0:
                null_rates = {}
                for col in df.columns:
                    null_count = df.filter(df[col].isNull()).count()
                    null_rates[col] = round(null_count / total * 100, 1)
                result["null_rates_pct"] = null_rates
        except Exception as exc:
            result["null_rates_pct"] = {"error": str(exc)}

        try:
            result["sample_rows"] = [
                row.asDict() for row in df.limit(3).collect()
            ]
        except Exception as exc:
            result["sample_rows"] = {"error": str(exc)}

        return result

    @staticmethod
    def clarify_requirements(args: dict[str, Any]) -> dict[str, Any]:
        missing = args.get("missing_constraints", [])
        query = args.get("query", "your architecture")

        questions = []
        if "latency" in missing:
            questions.append(
                "• Latency: Do you need real-time (sub-second), near-real-time "
                "(seconds to minutes), or is a batch refresh (hourly/daily) acceptable?"
            )
        if "ingestion" in missing:
            questions.append(
                "• Ingestion: Is your source data a continuous stream (Kafka, Kinesis, "
                "CDC), scheduled files/exports, or a mix of both?"
            )
        if "governance" in missing:
            questions.append(
                "• Governance: Do you have strict data access controls, audit logging, "
                "or regulatory compliance requirements (GDPR, HIPAA, SOX)?"
            )
        if "cost" in missing:
            questions.append(
                "• Cost: Is minimising cloud spend a primary concern, or is performance "
                "the priority even at higher cost?"
            )
        if not questions:
            questions.append(
                "• Could you share more about your latency, ingestion pattern, "
                "governance needs, or cost constraints?"
            )

        clarification = (
            f"To design the best architecture for {query}, I need a few more details:\n\n"
            + "\n".join(questions)
            + "\n\nPlease answer whichever are relevant — you can skip any that don't apply."
        )
        return {"clarification_needed": True, "question": clarification}

    # ------------------------------------------------------------------
    # ToolInfo wrappers — each function accepts **kwargs (OpenAI tool call
    # format) and returns a JSON string so the LLM can read the result.
    # ------------------------------------------------------------------

    def build_tool_infos(self) -> list[ToolInfo]:
        """Return all custom tools wrapped as ToolInfo objects."""

        def _check_workspace_state(focus_keywords: list | None = None) -> str:
            result = self.check_workspace_state(
                {"focus_keywords": focus_keywords or []}
            )
            return json.dumps(result, indent=2)

        def _profile_table(table_name: str) -> str:
            result = self.profile_table({"table_name": table_name})
            return json.dumps(result, indent=2)

        def _health_check() -> str:
            result = self.health_check({})
            return json.dumps(result, indent=2)

        def _clarify_requirements(
            missing_constraints: list | None = None, query: str = ""
        ) -> str:
            result = self.clarify_requirements(
                {"missing_constraints": missing_constraints or [], "query": query}
            )
            return json.dumps(result, indent=2)

        return [
            ToolInfo(
                name="check_workspace_state",
                spec={
                    "type": "function",
                    "function": {
                        "name": "check_workspace_state",
                        "description": (
                            "Fetch live Databricks workspace inventory filtered to resources "
                            "relevant to the user's query. "
                            "YOU extract focus_keywords from the user's query (e.g. for "
                            "'fraud detection streaming' use ['fraud', 'streaming', 'detection']). "
                            "Returns, for each resource type (VS endpoints, serving endpoints, "
                            "DLT pipelines, tables, registered models, jobs): "
                            "  relevant: resources whose name contains a focus keyword, "
                            "  total_scanned: total count of that type in the workspace. "
                            "Use this for design questions so your recommendation builds on "
                            "what already exists rather than starting from scratch. "
                            "Call BEFORE drafting architecture options."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "focus_keywords": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        "Keywords extracted from the user query to filter "
                                        "workspace resources by name. "
                                        "E.g. ['fraud', 'streaming', 'model'] for a fraud "
                                        "detection streaming query."
                                    ),
                                },
                            },
                            "required": ["focus_keywords"],
                        },
                    },
                },
                exec_fn=_check_workspace_state,
            ),
            ToolInfo(
                name="profile_table",
                spec={
                    "type": "function",
                    "function": {
                        "name": "profile_table",
                        "description": (
                            "Inspect a Delta table and return its schema (column names and types), "
                            "row count, last modified timestamp, null rates per column, "
                            "and 3 sample rows. "
                            "Call this on tables returned by check_workspace_state that look "
                            "relevant to the user's query. "
                            "Use the results to make data-specific recommendations: "
                            "row_count drives streaming vs batch choice, "
                            "null_rates_pct reveals Silver-layer transformation needs, "
                            "sample_rows reveal domain context (currency, IDs, labels), "
                            "last_modified reveals whether the pipeline is healthy."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "table_name": {
                                    "type": "string",
                                    "description": (
                                        "Fully-qualified Delta table name "
                                        "(catalog.schema.table or schema.table)."
                                    ),
                                },
                            },
                            "required": ["table_name"],
                        },
                    },
                },
                exec_fn=_profile_table,
            ),
            ToolInfo(
                name="health_check",
                spec={
                    "type": "function",
                    "function": {
                        "name": "health_check",
                        "description": (
                            "Check whether required Delta tables (knowledge_base, kb_chunks) "
                            "exist. Use when the agent suspects missing data or pipeline issues."
                        ),
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                exec_fn=_health_check,
            ),
            ToolInfo(
                name="clarify_requirements",
                spec={
                    "type": "function",
                    "function": {
                        "name": "clarify_requirements",
                        "description": (
                            "Ask the user for missing constraints before designing an architecture. "
                            "Call this ONLY when the user asks for a design and has NOT specified "
                            "ANY constraints at all — no latency, no ingestion pattern, no governance "
                            "needs, and no cost priority. "
                            "Do NOT call this if the user has mentioned even one constraint "
                            "(e.g. 'low latency', 'real-time', 'compliance', 'cheap', 'streaming'). "
                            "Do NOT call this for factual questions. "
                            "EXAMPLES where you must NOT call this tool: "
                            "  - 'real-time fraud detection with compliance' → has latency + governance "
                            "  - 'batch pipeline, budget-friendly' → has ingestion + cost "
                            "  - 'low latency ML serving' → has latency "
                            "EXAMPLE where you MUST call this tool: "
                            "  - 'design me a Databricks architecture' → zero constraints. "
                            "When this tool is called, the agent will return the clarifying question "
                            "directly to the user instead of proceeding with the design."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "missing_constraints": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        'List of constraint keys that are missing. '
                                        'Valid values: "latency", "ingestion", "governance", "cost".'
                                    ),
                                },
                                "query": {
                                    "type": "string",
                                    "description": "The user's original query, used to personalise the question.",
                                },
                            },
                            "required": ["missing_constraints"],
                        },
                    },
                },
                exec_fn=_clarify_requirements,
            ),
        ]
