"""Custom tool handlers and ToolInfo wrappers for the Databricks Expert Assistant."""

import json
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import SparkSession

from arch_designer_agent.config import ProjectConfig
from arch_designer_agent.mcp import ToolInfo

# ---------------------------------------------------------------------------
# Domain synonym expansion (static fallback)
# ---------------------------------------------------------------------------
# Maps a canonical use-case keyword to related domain terms so that
# check_workspace_state catches resources even when names use different
# vocabulary.  E.g. a "fraud detection" query also matches tables named
# "transactions" or "payments".
#
# This dictionary is used as a FAST FALLBACK when the LLM-powered keyword
# expansion is unavailable or fails.

DOMAIN_SYNONYMS: dict[str, list[str]] = {
    "fraud": ["transaction", "payment", "anomaly", "risk", "suspicious", "alert", "chargeback"],
    "forecasting": ["sales", "demand", "inventory", "orders", "historical", "forecast"],
    "forecast": ["sales", "demand", "inventory", "orders", "historical"],
    "recommendation": ["product", "user", "rating", "click", "engagement", "personalization"],
    "churn": ["retention", "customer", "subscription", "cancellation", "attrition"],
    "etl": ["pipeline", "ingestion", "transform", "load", "extract"],
    "streaming": ["kafka", "kinesis", "event", "realtime", "stream"],
    "compliance": ["audit", "lineage", "governance", "access", "policy", "security"],
    "ml": ["model", "training", "inference", "feature", "scoring"],
    "analytics": ["reporting", "dashboard", "warehouse", "query", "insight"],
    "inventory": ["stock", "supply", "product", "sku", "warehouse"],
    "customer": ["user", "client", "account", "profile", "segment"],
    "financial": ["revenue", "cost", "payment", "transaction", "ledger"],
    "healthcare": ["patient", "clinical", "medical", "diagnosis", "treatment"],
    "supply_chain": ["supplier", "logistics", "procurement", "shipment", "vendor"],
}

# Tables that are internal agent infrastructure and must never appear in
# workspace state results returned to the LLM.
_INTERNAL_TABLES = {"kb_chunks", "databricks_knowledge_base", "kb_chunks_index"}

# Substrings that identify internal agent infrastructure resources across ALL
# resource types (pipelines, jobs, endpoints, etc.).  Any resource whose name
# contains one of these (case-insensitive) is silently excluded from results.
_INTERNAL_RESOURCE_SUBSTRINGS = {
    "kb-pipeline",
    "kb_pipeline",
    "kb_chunks",
    "knowledge_base",
    "knowledge-base",
    "llmops_course",
    "online_index_view",
    "event_log_",
}

# Keywords that are too generic for substring matching and cause false
# positives.  E.g. "pipeline" matches every pipeline name, "model" matches
# every registered model.  These are removed from the expanded keyword set
# BEFORE the matches() function runs.
_MATCHING_STOP_WORDS = {
    "pipeline",
    "model",
    "table",
    "data",
    "job",
    "endpoint",
    "notebook",
    "cluster",
    "warehouse",
    "schema",
    "catalog",
    "query",
    "ingestion",
    "transform",
    "load",
    "extract",
    "training",
    "inference",
    "scoring",
    "reporting",
    "dashboard",
    "batch",
    "etl",
    "ml",
    "ai",
    "dev",
    "prod",
    "test",
}

# Prompt template for LLM-powered keyword expansion
_KEYWORD_EXPANSION_PROMPT = """\
You are a keyword expansion engine for searching Databricks workspace resources.

Given a user's use-case description and initial keywords, generate a flat JSON \
array of **single-word, lowercase** search terms that are likely to appear in \
table names, pipeline names, model names, or job names for this use case.

Include:
- Domain-specific entity names (e.g. "transactions", "patients", "orders")
- Common abbreviations (e.g. "txn", "inv", "sku")
- Related data concepts (e.g. "revenue" for a sales forecasting use case)
- Singular AND plural forms if commonly used in naming

Do NOT include:
- Multi-word phrases
- Generic Databricks terms (e.g. "delta", "spark", "notebook", "pipeline",
  "model", "table", "data", "job", "endpoint", "cluster", "warehouse",
  "batch", "etl", "ml", "training", "inference", "ingestion")
- Stop words (e.g. "the", "and", "for")

User query: {user_query}
Initial keywords: {keywords}

Return ONLY a valid JSON array of strings. No explanation, no markdown fences."""


def _safe_json_default(obj: object) -> str:
    """Fallback serialiser for json.dumps — handles datetime, date, Decimal, etc."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    return str(obj)


def _is_internal_resource(name: str) -> bool:
    """Return True if the resource name matches internal agent infrastructure."""
    name_lower = name.lower()
    return any(pat in name_lower for pat in _INTERNAL_RESOURCE_SUBSTRINGS)


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

    # ------------------------------------------------------------------
    # Keyword expansion — LLM-powered with static fallback
    # ------------------------------------------------------------------

    def _llm_expand_keywords(self, raw_keywords: list[str], user_query: str = "") -> list[str]:
        """Use the LLM to dynamically generate relevant search keywords.

        Calls the configured LLM endpoint with a structured prompt to produce
        domain-aware, single-word search terms.  Falls back to an empty list
        on any failure (caller should combine with static expansion).
        """
        try:
            llm_client = self.w.serving_endpoints.get_open_ai_client()
            prompt = _KEYWORD_EXPANSION_PROMPT.format(
                user_query=user_query,
                keywords=json.dumps(raw_keywords),
            )
            response = llm_client.chat.completions.create(
                model=self.cfg.llm_endpoint,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            content = response.choices[0].message.content.strip()
            # Strip markdown fences if the model wraps the response
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            keywords = json.loads(content)
            if isinstance(keywords, list):
                result = [str(k).lower().strip() for k in keywords if isinstance(k, str)]
                logger.info(f"  LLM keyword expansion: {raw_keywords} → {result}")
                return result
        except Exception as exc:
            logger.warning(f"  LLM keyword expansion failed ({exc}); using static fallback")
        return []

    @staticmethod
    def _static_expand_keywords(keywords: list[str]) -> set[str]:
        """Expand keywords using the static DOMAIN_SYNONYMS dictionary."""
        expanded: set[str] = set()
        for kw in keywords:
            if kw in DOMAIN_SYNONYMS:
                expanded.update(DOMAIN_SYNONYMS[kw])
            for canonical, synonyms in DOMAIN_SYNONYMS.items():
                if kw in synonyms:
                    expanded.add(canonical)
                    expanded.update(synonyms)
        return expanded

    def _expand_keywords(self, keywords: list[str], user_query: str = "") -> list[str]:
        """Expand user keywords with LLM-powered + static synonym expansion.

        Pipeline:
        1. Split multi-word keywords into individual words
           ("demand forecasting" → ["demand", "forecasting"])
        2. Apply static DOMAIN_SYNONYMS as a fast baseline
        3. Call the LLM for dynamic, context-aware expansion
        4. Combine, deduplicate, and remove stop words
        """
        # Step 1: split compound keywords into individual words
        split_keywords: set[str] = set()
        for kw in keywords:
            split_keywords.update(word.lower() for word in kw.split() if word)
        all_words = list(split_keywords)

        # Step 2: static expansion (fast, no network call)
        expanded: set[str] = set(all_words)
        expanded.update(self._static_expand_keywords(all_words))

        # Step 3: LLM expansion (dynamic, handles any domain)
        llm_keywords = self._llm_expand_keywords(all_words, user_query=user_query)
        expanded.update(llm_keywords)

        # Step 4: remove stop words that cause false-positive matches
        before_count = len(expanded)
        expanded -= _MATCHING_STOP_WORDS
        if before_count != len(expanded):
            logger.info(f"  Removed {before_count - len(expanded)} stop word(s) from expanded keywords")

        logger.info(
            f"  Keyword expansion: raw={keywords} → split={all_words} "
            f"→ final={sorted(expanded)} ({len(expanded)} terms)"
        )
        return list(expanded)

    def check_workspace_state(self, args: dict[str, Any]) -> dict[str, Any]:
        """Return existing workspace resources that may be relevant to the user's use case.

        Results are SUPPLEMENTARY CONTEXT only — existing resources the user can
        leverage if applicable.  They must never drive the core architecture
        recommendation, which must come from Databricks best-practice patterns.

        Keywords are automatically expanded using LLM-powered synonym generation
        and a static domain dictionary, so that a 'demand forecasting' query also
        matches 'sales' or 'inventory' tables.

        Internal agent resources (kb_chunks, kb-pipeline, etc.) are always
        excluded from results.
        """
        raw_keywords = [kw.lower() for kw in args.get("focus_keywords", [])]
        user_query = args.get("user_query", "")
        focus = self._expand_keywords(raw_keywords, user_query=user_query)

        def matches(name: str) -> bool:
            if not focus:
                return True
            name_lower = name.lower()
            return any(kw in name_lower for kw in focus)

        state: dict[str, Any] = {
            "note": (
                "SUPPLEMENTARY CONTEXT: these are EXISTING resources in the workspace. "
                "Mention them as 'already available and can be leveraged if applicable'. "
                "Do NOT change your core architecture recommendation based on these names "
                "or their medallion layer labels (bronze/silver/gold)."
            ),
            "focus_keywords_raw": raw_keywords,
            "focus_keywords_expanded": focus,
            "catalog": self.cfg.catalog,
            "schema": self.cfg.schema,
        }

        try:
            all_vs = list(self.w.vector_search_endpoints.list())
            state["vector_search_endpoints"] = {
                "relevant": [
                    {
                        "name": e.name,
                        "state": str(e.endpoint_status.state) if e.endpoint_status else "unknown",
                    }
                    for e in all_vs
                    if matches(e.name) and not _is_internal_resource(e.name)
                ],
                "total_scanned": len(all_vs),
            }
        except Exception as exc:
            state["vector_search_endpoints"] = {"error": str(exc)}

        try:
            all_se = list(self.w.serving_endpoints.list())
            state["serving_endpoints"] = {
                "relevant": [
                    {
                        "name": e.name,
                        "state": str(e.state.config_update) if e.state else "unknown",
                    }
                    for e in all_se
                    if matches(e.name) and not _is_internal_resource(e.name)
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
                    for p in all_dlt
                    if p.name and matches(p.name) and not _is_internal_resource(p.name)
                ],
                "total_scanned": len(all_dlt),
            }
        except Exception as exc:
            state["dlt_pipelines"] = {"error": str(exc)}

        try:
            # Use SDK (Unity Catalog REST API) instead of Spark — avoids Spark Connect
            # DATA_SOURCE_NOT_FOUND errors on tables with unsupported formats.
            all_tables = list(
                self.w.tables.list(
                    catalog_name=self.cfg.catalog,
                    schema_name=self.cfg.schema,
                )
            )
            table_infos = [{"name": t.name, "comment": t.comment or ""} for t in all_tables]

            user_tables = [
                t for t in table_infos if t["name"] not in _INTERNAL_TABLES and not _is_internal_resource(t["name"])
            ]
            state["tables"] = {
                "existing_relevant": [
                    {"name": t["name"], "comment": t["comment"]}
                    for t in user_tables
                    if matches(t["name"]) or matches(t["comment"])
                ],
                "total_scanned": len(user_tables),
            }
        except Exception as exc:
            state["tables"] = {"error": str(exc)}

        try:
            all_models = list(self.w.registered_models.list(catalog_name=self.cfg.catalog, schema_name=self.cfg.schema))
            state["registered_models"] = {
                "relevant": [
                    {"name": m.name} for m in all_models if matches(m.name) and not _is_internal_resource(m.name)
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
                        if j.settings and j.settings.schedule
                        else "manual",
                    }
                    for j in all_jobs
                    if j.settings and matches(j.settings.name) and not _is_internal_resource(j.settings.name)
                ],
                "total_scanned": len(all_jobs),
            }
        except Exception as exc:
            state["jobs"] = {"error": str(exc)}

        # ------------------------------------------------------------------
        # Build a natural-language resource summary for the LLM.
        # The LLM reads "resource_summary" directly — it should NEVER
        # reference internal field names like "has_relevant_resources" in
        # its answer to the user.
        # ------------------------------------------------------------------
        def _has_items(key: str, subkey: str) -> bool:
            section = state.get(key, {})
            return bool(isinstance(section, dict) and section.get(subkey))

        has_resources = any(
            [
                _has_items("vector_search_endpoints", "relevant"),
                _has_items("serving_endpoints", "relevant"),
                _has_items("dlt_pipelines", "relevant"),
                _has_items("tables", "existing_relevant"),
                _has_items("registered_models", "relevant"),
                _has_items("jobs", "relevant"),
            ]
        )

        # Keep the boolean for programmatic use (e.g. _strip_empty_resources_section)
        state["has_relevant_resources"] = has_resources

        if has_resources:
            # Collect names of all matched resources for a quick summary
            found_names = []
            for key, subkey in [
                ("tables", "existing_relevant"),
                ("serving_endpoints", "relevant"),
                ("dlt_pipelines", "relevant"),
                ("vector_search_endpoints", "relevant"),
                ("registered_models", "relevant"),
                ("jobs", "relevant"),
            ]:
                section = state.get(key, {})
                if isinstance(section, dict):
                    for item in section.get(subkey, []):
                        found_names.append(item.get("name", "unknown"))
            state["resource_summary"] = (
                f"FOUND {len(found_names)} existing resource(s) that may be relevant: "
                f"{', '.join(found_names)}. "
                "Include an '## Existing Resources in Your Workspace' section in your "
                "answer describing how these can be leveraged."
            )
        else:
            state["resource_summary"] = (
                "NO existing resources matched this use case. "
                "Do NOT include any 'Existing Resources' section in your answer. "
                "Do NOT mention the absence of resources — simply skip to ## References. "
                "Write only the ## Architecture and ## References sections."
            )

        return state

    def health_check(self, args: dict[str, Any]) -> dict[str, Any]:
        _ = args
        kb_exists = self.spark.catalog.tableExists(self.kb_table)
        chunks_exists = self.spark.catalog.tableExists(self.chunks_table)
        return {
            "kb_ready": kb_exists and chunks_exists,
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

        # Block internal agent tables — they are never relevant to user use cases.
        bare_name = table_name.split(".")[-1].lower()
        if bare_name in _INTERNAL_TABLES or _is_internal_resource(bare_name):
            return {
                "error": f"'{table_name}' is an internal agent table and cannot be profiled. "
                "Only profile tables that were returned in check_workspace_state existing_relevant results."
            }

        result: dict[str, Any] = {"table_name": table_name}

        try:
            df = self.spark.table(table_name)
            result["columns"] = [
                {"name": f.name, "type": str(f.dataType), "nullable": f.nullable} for f in df.schema.fields
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
            result["sample_rows"] = [row.asDict() for row in df.limit(3).collect()]
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
                "• Could you share more about your latency, ingestion pattern, governance needs, or cost constraints?"
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
            result = self.check_workspace_state({"focus_keywords": focus_keywords or []})
            return json.dumps(result, indent=2, default=_safe_json_default)

        def _profile_table(table_name: str) -> str:
            result = self.profile_table({"table_name": table_name})
            return json.dumps(result, indent=2, default=_safe_json_default)

        def _health_check() -> str:
            result = self.health_check({})
            return json.dumps(result, indent=2, default=_safe_json_default)

        def _clarify_requirements(missing_constraints: list | None = None, query: str = "") -> str:
            result = self.clarify_requirements({"missing_constraints": missing_constraints or [], "query": query})
            return json.dumps(result, indent=2, default=_safe_json_default)

        return [
            ToolInfo(
                name="check_workspace_state",
                spec={
                    "type": "function",
                    "function": {
                        "name": "check_workspace_state",
                        "description": (
                            "CALL THIS FOR DESIGN QUERIES ONLY. "
                            "Discovers existing Databricks workspace resources (tables, pipelines, "
                            "models, endpoints, jobs) relevant to the user's use case. "
                            "This is MANDATORY for any query where the user asks you to design, "
                            "architect, build, or recommend a solution — you need to know what "
                            "already exists before recommending what to build. "
                            "DO NOT call this for factual/explanatory queries ('what is X', "
                            "'explain Y', 'how does Z work'). "
                            "Results are supplementary — mention existing resources as "
                            "'already available and can be leveraged as X'. "
                            "Your core architecture pattern must come from KB search, not from "
                            "what tables happen to exist. "
                            "Keywords are expanded AUTOMATICALLY using LLM-powered synonym "
                            "generation — pass the most relevant individual words from the "
                            "user's query. Use SINGLE WORDS, not phrases: "
                            "e.g. ['demand', 'forecasting', 'retail'] not ['demand forecasting']."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "focus_keywords": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        "Individual keywords extracted from the user query to "
                                        "filter workspace resources by name. Use SINGLE WORDS "
                                        "only — multi-word phrases will be split automatically. "
                                        "E.g. ['fraud', 'detection', 'streaming'] for a fraud "
                                        "detection streaming query. "
                                        "E.g. ['demand', 'forecasting', 'sales', 'retail'] for "
                                        "a retail demand forecasting query."
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
                            "ONLY call this on tables that appeared in check_workspace_state "
                            "existing_relevant results AND are directly relevant to the user's "
                            "use case data (e.g. a transactions table for fraud detection). "
                            "NEVER call this on internal tables: kb_chunks, "
                            "databricks_knowledge_base, kb_chunks_index. "
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
                                        "Fully-qualified Delta table name (catalog.schema.table or schema.table)."
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
                            "Call this ONLY when the query contains ZERO constraints — no mention of "
                            "latency, speed, ingestion type, compliance, cost, team size, or scale. "
                            "If the query contains ANY of the following words or phrases, "
                            "do NOT call this tool and proceed directly with check_workspace_state:\n"
                            "  Latency: real-time, near-real-time, low latency, batch, hourly, daily, weekly\n"
                            "  Ingestion: streaming, Kafka, CDC, files, scheduled, incremental\n"
                            "  Governance: compliance, GDPR, HIPAA, SOX, audit, access control\n"
                            "  Cost: budget, cheap, cost-effective, affordable, small team, simple\n"
                            "  Scale: large, small, high volume, millions, thousands\n"
                            "Examples where you must NOT call this:\n"
                            "  'real-time fraud detection with compliance' → real-time + compliance present\n"
                            "  'batch pipeline, budget-friendly' → batch + budget present\n"
                            "  'low latency ML serving' → low latency present\n"
                            "  'small analytics team, simplicity matters' → small team + simplicity present\n"
                            "Only call this for: 'design me a Databricks architecture' with nothing else."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "missing_constraints": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        "List of constraint keys that are missing. "
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
