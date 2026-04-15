# Databricks Expert Agent — Architecture Guide

A reference for how every part of the agent works: authentication, tool loading,
tool behaviour, routing decisions, multi-turn memory, and the end-to-end conversation flow.

---

## 1. Authentication — production-grade approach

### Why `w.tokens.create()` is not production-safe

| Problem | Detail |
|---|---|
| Tied to a human user | If that user leaves or their account is disabled, the agent breaks |
| Created once at construction | Token can expire mid-session (default lifetime ≤ 1 hour) |
| Not automatically refreshed | No retry/renewal logic |
| Audit trail points to a person | Hard to attribute automated actions correctly |

### What we use instead: `DatabricksOAuth` (in `src/arch_designer_agent/mcp.py`)

```
DatabricksOAuth
   │
   ├── SPN path (production)
   │     DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET env vars set
   │     Uses databricks.sdk.oauth.ClientCredentials (M2M OAuth)
   │     Token is cached + auto-refreshed by the SDK silently
   │     Audit trail shows the Service Principal, not a human
   │
   └── PAT fallback (notebook / dev)
         No SPN env vars found
         Creates a short-lived PAT via w.tokens.create(lifetime_seconds=3600)
         Fine for interactive development
```

### Setup for production (Databricks Job)

Step 1 — Create a Service Principal and grant it:
- `CAN USE` on the serving endpoint
- `SELECT` on the catalog/schema holding `kb_chunks`
- `CAN USE` on the vector search endpoint

Step 2 — Store credentials in Databricks Secrets:
```
secret scope:  llmops_course
  key: client_id      → value: <SPN application ID>
  key: client_secret  → value: <SPN OAuth secret>
```

Step 3 — At the top of your notebook or job:
```python
from arch_designer_agent.mcp import load_spn_credentials

load_spn_credentials("llmops_course")   # reads from Databricks Secrets
w = WorkspaceClient()                   # auto-picks up SPN credentials
agent = DatabricksExpertAgent(spark=spark, config=cfg, workspace_client=w)
```

### How the token flows through the agent

**LLM calls** — a fresh `OpenAI` client is created once per `chat()` call via `_make_llm_client()`:
```python
OpenAI(
    api_key=self._oauth.token(),        # SPN: cached OAuth token, auto-refreshes
    base_url=f"{host}/serving-endpoints",  # PAT: new 1-hour PAT
)
```

**MCP tool calls** — auth is handled transparently:
```python
DatabricksMCPClient(server_url=..., workspace_client=w)
```
`DatabricksMCPClient` uses `WorkspaceClient`'s credential chain internally.

---

## 2. Tool loading — what gets registered and where it comes from

Entry point: `_load_tools()` in `agent.py`. Runs at construction time, fills a single
`ToolRegistry`. Three sources:

### Source 1: Vector Search MCP (remote, managed by Databricks)

```python
vs_mcp_url = f"{host}/api/2.0/mcp/vector-search/{catalog}/{schema}"
mcp_tools = asyncio.run(create_mcp_tools(w, [vs_mcp_url]))
```

`create_mcp_tools()` in `mcp.py`:
1. Connects to the Databricks MCP server at that URL.
2. Calls `mcp_client.list_tools()` — Databricks scans all VS indexes in your
   catalog/schema and returns one tool per index.
3. For each index it creates:
   - a `spec` dict (OpenAI function JSON) using the schema the MCP server provided
   - an `exec_fn` closure that calls `DatabricksMCPClient.call_tool()` at runtime

Tool name becomes `{catalog}__{schema}__kb_chunks_index`.

If any MCP server fails to load, it is skipped with a WARNING — the agent continues
with the remaining tools. This prevents a broken Genie space or unavailable VS endpoint
from crashing the agent.

### Source 2: Genie MCP (remote, optional)

Same pattern, URL is `/api/2.0/mcp/genie/{genie_space_id}`. Lets the LLM query your
Delta tables in plain English via Genie. Only registered if `genie_space_id` is set
(non-null) in `project_config.yml`.

### Source 3: Custom Python tools (local, in-process)

```python
custom_tools = DatabricksExpertTools(spark, cfg, workspace_client=w).build_tool_infos()
```

`build_tool_infos()` in `agent_tools.py` returns four `ToolInfo` objects.

---

## 3. What each tool does

### Tool 1 — `{catalog}__{schema}__kb_chunks_index` (MCP)

| Property | Value |
|---|---|
| **Source** | Databricks Vector Search MCP server (remote, managed) |
| **Input** | `query` — plain text search string |
| **What it does** | Embeds your query using `databricks-gte-large-en`, finds the most similar chunks in `kb_chunks` Delta table via cosine similarity, returns raw text chunks with metadata |
| **Output to LLM** | JSON string with matching docs, titles, URLs, section headers |
| **When LLM uses it** | Any time it needs factual grounding from Databricks documentation |

### Tool 2 — `check_workspace_state` (custom)

| Property | Value |
|---|---|
| **Source** | `agent_tools.py` → `check_workspace_state` |
| **Input** | `focus_keywords` — list of keywords extracted from the user query |
| **What it does** | Calls Databricks SDK REST APIs (no Spark) to list VS endpoints, serving endpoints, DLT pipelines, tables (via `w.tables.list()` with UC comments), registered models, and jobs. Filters each to resources whose name or UC comment contains a focus keyword |
| **Output to LLM** | JSON with `relevant` + `total_scanned` per resource type |
| **When LLM uses it** | Design questions — so recommendations build on what already exists |

### Tool 3 — `profile_table` (custom)

| Property | Value |
|---|---|
| **Source** | `agent_tools.py` → `profile_table` |
| **Input** | `table_name` — fully qualified Delta table name |
| **What it does** | Uses Spark to read schema, row count, last modified timestamp (from `DESCRIBE HISTORY`), null rates per column, and 3 sample rows |
| **Output to LLM** | JSON with `columns`, `row_count`, `last_modified`, `recent_operations`, `null_rates_pct`, `sample_rows` |
| **When LLM uses it** | After `check_workspace_state` identifies relevant tables — grounds design in actual data shape |

### Tool 4 — `health_check` (custom)

| Property | Value |
|---|---|
| **Source** | `agent_tools.py` → `health_check` |
| **Input** | None |
| **What it does** | Calls `spark.catalog.tableExists()` on `databricks_knowledge_base` and `kb_chunks` |
| **Output to LLM** | `{kb_table_exists, chunks_table_exists, status: "ok" or "degraded"}` |
| **When LLM uses it** | When it suspects missing data or retrieval returned nothing |

### Tool 5 — `clarify_requirements` (custom)

| Property | Value |
|---|---|
| **Source** | `agent_tools.py` → `clarify_requirements` |
| **Input** | `missing_constraints` list, `query` string |
| **What it does** | Returns a formatted clarifying question for latency, ingestion, governance, or cost |
| **Output to LLM** | JSON with `clarification_needed: true` and `question` text |
| **When LLM uses it** | ONLY when no constraints at all are present in the user query. Must NOT be called if even one constraint is mentioned (e.g. "real-time", "compliance", "cheap") |

---

## 4. How routing works — how the LLM decides which tool to call

In `chat()` in `agent.py`:

```python
response = llm_client.chat.completions.create(
    model=self.cfg.llm_endpoint,
    messages=messages,
    tools=tool_specs,   # ALL tool specs sent here on every call
)
```

Every LLM call includes all tool specs. The LLM reads the `description`, `name`, and
`parameters` fields and decides which tool to call — no hardcoded sequence.

### When the LLM stops calling tools

```python
if not message.tool_calls:
    return message.content or ""
```

If the LLM response has no `tool_calls` in the structured interface, it means the LLM
has enough context to compose a final answer.

### Guard: LLM writes tool call as plain text

Occasionally a model will write a tool call as a text string (e.g.
`profile_table(table_name="...")`) instead of using the structured tool-calling
interface. The agent detects this pattern and re-prompts:

```python
if any(final_answer.strip().startswith(name + "(") for name in tool_names):
    # inject corrective user message and continue the loop
```

---

## 5. Multi-turn conversation memory (Lakebase)

### Why Lakebase instead of Delta

| | Delta (`ConversationMemory`) | Lakebase (`LakebaseMemory`) |
|---|---|---|
| Write latency | Seconds (Spark + Delta log commit) | Milliseconds (direct PostgreSQL INSERT) |
| Connection | Requires active Spark session | Direct psycopg connection pool |
| Auth | Spark credentials | SDK `generate_database_credential()` → short-lived PG password |

### How it works

```
Agent.__init__
  ↓
_get_or_start_lakebase(instance_name)
  ├── w.database.get_database_instance() → AVAILABLE → return read_write_dns
  ├── state == STOPPED → update_database_instance(stopped=False) → wait → return dns
  └── NotFound → create_database_instance().result() → return dns

LakebaseMemory(host, instance_name)
  └── _get_connection_string()
       ├── SPN: username = DATABRICKS_CLIENT_ID
       └── User: username = current_user.me().user_name
       w.database.generate_database_credential() → short-lived PG token
       → postgresql://{username}:{token}@{host}:5432/databricks_postgres?sslmode=require
```

### Storage schema

```sql
CREATE TABLE IF NOT EXISTS session_messages (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    message_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Message flow across turns

```
Turn 1  chat("Design fraud detection", conversation_id="s1")
        prior_messages = []      ← nothing in PG yet
        sends to LLM: [system, user_1]
        saves to PG:  [user_1, tool_calls..., assistant_1]

Turn 2  chat("Focus on compliance", conversation_id="s1")
        prior_messages = [user_1, ..., assistant_1]   ← loaded from PG
        sends to LLM: [system, user_1, ..., assistant_1, user_2]
        saves to PG:  [user_2, tool_calls..., assistant_2]   ← only new rows
```

`save_from_index = 1 + len(prior_messages)` ensures only the delta is written each
turn — not the full history again.

### Configuration

```yaml
# project_config.yml
dev:
  lakebase_instance: arch-agent-memory   # instance name
  usage_policy_id: null                  # optional: needed only if auto-creating
```

If `lakebase_instance` is `null`, the agent runs stateless (single-turn) and logs
`Lakebase not configured — running stateless (single-turn)`.

---

## 6. How tool calls and results flow through the conversation

The `messages` list is the state. It grows with every tool call:

```
Turn 0
  messages = [system_prompt, user_query]

LLM → call check_workspace_state(focus_keywords=["fraud", "streaming"])
Turn 1
  messages = [..., assistant(tool_call), tool_result("{endpoints:[...], tables:[...]}")]

LLM → call kb_chunks_index(query="real-time fraud detection compliance Databricks")
Turn 2
  messages = [..., assistant(tool_call), tool_result("...docs...")]

LLM → call profile_table(table_name="mlops_dev.pratikhe.txn_events")
Turn 3
  messages = [..., assistant(tool_call), tool_result("{columns:[...], row_count:...}")]

LLM → text answer (no tool_calls)
  → messages.append(assistant(final_answer))
  → save_messages(session_id, messages[save_from_index:])
  → return final_answer
```

The LLM sees the full `messages` history on every call. There is no explicit state
object — the `messages` list is the state.

---

## 7. Complete end-to-end flow for a real query

**Query:** `"Design a real-time Databricks architecture for fraud detection with strict compliance controls and low latency."`

```
Step 1 — Agent construction
  _get_or_start_lakebase("arch-agent-memory") → read_write_dns
  LakebaseMemory(host, instance_name) initialised
  _load_tools():
    MCP: kb_chunks_index registered
    Custom: check_workspace_state, profile_table, health_check,
            clarify_requirements registered

Step 2 — chat() called with conversation_id="fraud-session-001"
  load_messages("fraud-session-001") → [] (first turn)
  messages = [system, user("Design real-time fraud...")]

Step 3 — iteration 1
  LLM reads constraints: "real-time", "low latency", "strict compliance"
  → does NOT call clarify_requirements (constraints present)
  LLM picks: check_workspace_state(focus_keywords=["fraud","streaming","compliance"])
  → SDK REST calls: VS endpoints, serving endpoints, tables (w.tables.list()),
    models, jobs — filtered by keywords
  → returns: {tables: {relevant: [{name:"txn_events", comment:"fraud events"}]}}

Step 4 — iteration 2
  LLM sees relevant table txn_events
  LLM picks: profile_table(table_name="mlops_dev.pratikhe.txn_events")
  → row_count, schema, null_rates, sample_rows

Step 5 — iteration 3
  LLM picks: kb_chunks_index(query="real-time fraud detection streaming medallion compliance")
  → top KB chunks with citations

Step 6 — iteration 4
  LLM has: live inventory + table profile + KB evidence
  → composess final recommendation citing actual resource names, schema,
    row counts, and KB sources
  → no tool_calls in response
  → save_messages("fraud-session-001", [user, tool_calls..., assistant])
  → return final answer
```

---

## 8. File map

| File | Role |
|---|---|
| `src/arch_designer_agent/agent.py` | Agent class, `chat()` loop, Lakebase init, tool loading |
| `src/arch_designer_agent/mcp.py` | `ToolInfo`, `ToolRegistry`, `DatabricksOAuth`, MCP client utilities |
| `src/arch_designer_agent/agent_tools.py` | `check_workspace_state`, `profile_table`, `health_check`, `clarify_requirements` |
| `src/arch_designer_agent/memory.py` | `LakebaseMemory` — PostgreSQL-backed multi-turn session memory |
| `src/arch_designer_agent/config.py` | `ProjectConfig` — catalog, schema, endpoints, lakebase_instance, system prompt |
| `notebooks/3.2_arch_designer_agent.py` | Runnable walkthrough: agent setup, test queries, Lakebase connectivity test, multi-turn demo |
| `docs/PIPELINE_GUIDE.md` | Data pipeline (ingestion → chunking → VS index) that feeds the agent |


---

## 1. Authentication — production-grade approach

### Why `w.tokens.create()` is not production-safe

| Problem | Detail |
|---|---|
| Tied to a human user | If that user leaves or their account is disabled, the agent breaks |
| Created once at construction | Token can expire mid-session (default lifetime ≤ 1 hour) |
| Not automatically refreshed | No retry/renewal logic |
| Audit trail points to a person | Hard to attribute automated actions correctly |

### What we use instead: `DatabricksOAuth` (in `src/arch_designer_agent/mcp.py`)

```
DatabricksOAuth
   │
   ├── SPN path (production)
   │     DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET env vars set
   │     Uses databricks.sdk.oauth.ClientCredentials (M2M OAuth)
   │     Token is cached + auto-refreshed by the SDK silently
   │     Audit trail shows the Service Principal, not a human
   │
   └── PAT fallback (notebook / dev)
         No SPN env vars found
         Creates a short-lived PAT via w.tokens.create(lifetime_seconds=3600)
         Fine for interactive development
```

### Setup for production (Databricks Job)

Step 1 — Create a Service Principal and grant it:
- `CAN USE` on the serving endpoint
- `SELECT` on the catalog/schema holding `kb_chunks`
- `CAN USE` on the vector search endpoint

Step 2 — Store credentials in Databricks Secrets:
```
secret scope:  llmops_course
  key: client_id      → value: <SPN application ID>
  key: client_secret  → value: <SPN OAuth secret>
```

Step 3 — At the top of your notebook or job:
```python
from arch_designer_agent.mcp import load_spn_credentials

load_spn_credentials("llmops_course")   # reads from Databricks Secrets
w = WorkspaceClient()                   # auto-picks up SPN credentials
agent = DatabricksExpertAgent(spark=spark, config=cfg, workspace_client=w)
```

This mirrors `notebooks/3.5.spn_authentication_in_action.py`.

### How the token flows through the agent

**LLM calls** — a fresh `OpenAI` client is created once per `chat()` call via `_make_llm_client()`:
```python
OpenAI(
    api_key=self._oauth.token(),        # SPN: cached OAuth token, auto-refreshes
    base_url=f"{host}/serving-endpoints",  # PAT: new 1-hour PAT
)
```
This ensures the token is never stale for long-running or repeated sessions.

**MCP tool calls** — auth is handled transparently:
```python
DatabricksMCPClient(server_url=..., workspace_client=w)
```
`DatabricksMCPClient` uses `WorkspaceClient`'s credential chain internally.
For SPN this is M2M OAuth with auto-refresh. No manual token handling needed.

---

## 2. Tool loading — what gets registered and where it comes from

Entry point: `_load_tools()` in `agent.py`. Runs at construction time, fills a single
`ToolRegistry`. Three sources:

### Source 1: Vector Search MCP (remote, managed by Databricks)

```python
vs_mcp_url = f"{host}/api/2.0/mcp/vector-search/{catalog}/{schema}"
mcp_tools = asyncio.run(create_mcp_tools(w, [vs_mcp_url]))
```

`create_mcp_tools()` in `mcp.py`:
1. Connects to the Databricks MCP server at that URL.
2. Calls `mcp_client.list_tools()` — Databricks scans all VS indexes in your
   catalog/schema and returns one tool per index.
3. For each index it creates:
   - a `spec` dict (OpenAI function JSON) using the schema the MCP server provided
   - an `exec_fn` closure that calls `DatabricksMCPClient.call_tool()` at runtime

Tool name becomes `{catalog}__{schema}__kb_chunks_index`. The LLM sees it as just
another callable function.

### Source 2: Genie MCP (remote, optional)

Same pattern, URL is `/api/2.0/mcp/genie/{genie_space_id}`. Lets the LLM query your
Delta tables in plain English via Genie. Only registered if `genie_space_id` is set
in `project_config.yml`.

### Source 3: Custom Python tools (local, in-process)

```python
custom_tools = DatabricksExpertTools(spark, cfg).build_tool_infos()
```

`build_tool_infos()` in `agent_tools.py` returns four `ToolInfo` objects — each wraps
a Python function with a hand-written OpenAI spec.

---

## 3. What each tool does

### Tool 1 — `{catalog}__{schema}__kb_chunks_index` (MCP)

| Property | Value |
|---|---|
| **Source** | Databricks Vector Search MCP server (remote, managed) |
| **Input** | `query` — plain text search string |
| **What it does** | Embeds your query using `databricks-gte-large-en`, finds the most similar chunks in `kb_chunks` Delta table via cosine similarity, returns raw text chunks with metadata |
| **Output to LLM** | JSON string with matching docs, titles, URLs, section headers |
| **When LLM uses it** | Any time it needs factual grounding before answering |

### Tool 2 — `generate_architecture_options` (custom)

| Property | Value |
|---|---|
| **Source** | `agent_tools.py` → `_generate_options` |
| **Input** | `query` (text), optional `constraints` dict (latency/cost/governance/ingestion keys) |
| **What it does** | Returns three architecture templates: `streaming_medallion`, `batch_plus_serving`, `hybrid_event_batch`. The constraint dict re-orders them, giving the LLM a hint about which best fits the requirements |
| **Output to LLM** | JSON with name, summary, fit tags, risk label per option |
| **When LLM uses it** | User asks to design or build an architecture |

### Tool 3 — `analyze_tradeoffs` (custom)

| Property | Value |
|---|---|
| **Source** | `agent_tools.py` → `_analyze_tradeoffs` |
| **Input** | `options` array from Tool 2, same `constraints` dict |
| **What it does** | Scores each option on latency/cost/governance (1–5), multiplied by constraint weights (1× default, 3× if that constraint was stated in the query). Ranks by total score |
| **Scoring example** | User said "low latency" → latency weight = 3. `streaming_medallion` gets latency=5 × 3 = 15. `batch_plus_serving` gets latency=3 × 3 = 9. Streaming wins |
| **Output to LLM** | JSON with ranked list and `selected_option` (winner) |
| **When LLM uses it** | After generating options when it needs to pick one |

### Tool 4 — `validate_design` (custom)

| Property | Value |
|---|---|
| **Source** | `agent_tools.py` → `_validate_design` |
| **Input** | `selected_option` (winner from Tool 3), `constraints` dict |
| **What it does** | Hard rule checks: if governance was strict but winner lacks Unity Catalog controls → add issue. If latency was low but winner is batch-first → add conflict warning |
| **Output to LLM** | JSON with `valid` boolean and `issues` list |
| **When LLM uses it** | After selecting a winner to surface risks before presenting to user |

### Tool 5 — `health_check` (custom)

| Property | Value |
|---|---|
| **Source** | `agent_tools.py` → `_health_check` |
| **Input** | None |
| **What it does** | Calls `spark.catalog.tableExists()` on `databricks_knowledge_base` and `kb_chunks` |
| **Output to LLM** | `{kb_table_exists, chunks_table_exists, status: "ok" or "degraded"}` |
| **When LLM uses it** | When it suspects missing data or retrieval returned nothing |

---

## 4. How routing works — how the LLM decides which tool to call

This is the critical part. In `chat()` in `agent.py`:

```python
response = self._client.chat.completions.create(
    model=self.cfg.llm_endpoint,
    messages=messages,
    tools=tool_specs,   # ALL tool specs sent here on every call
)
```

Every LLM call includes `tools=tool_specs` — the full list of all OpenAI
function-calling JSON specs from the registry. The decision algorithm lives entirely
inside the LLM. Your code only tells it what is available. The LLM reads the
`description`, `name`, and `parameters` fields and reasons about which tool is
appropriate for the current conversation state.

### Why tool descriptions matter so much

The description text in each spec is the LLM's routing signal:

| Tool | Key phrase in description | Routing effect |
|---|---|---|
| `kb_chunks_index` | "grounding / evidence retrieval" | Called first almost always |
| `generate_architecture_options` | "Call when user asks to DESIGN or BUILD" | Only called on design queries |
| `analyze_tradeoffs` | "Use AFTER generate_architecture_options" | Chained after options are generated |
| `validate_design` | "Call AFTER analyze_tradeoffs to validate the winner" | Always chained after ranking |
| `health_check` | "Use when agent suspects missing data" | Only called on error or empty results |

### When the LLM stops calling tools

```python
if not message.tool_calls:
    return message.content or ""
```

If the LLM response has no `tool_calls`, it means the LLM decided it has enough
context to compose a final answer. The loop exits and returns that text immediately.

---

## 5. How tool calls and results flow through the conversation

The `messages` list is the state. It grows with every tool call:

```
Turn 0
  messages = [system_prompt, user_query]

LLM → call kb_chunks_index(query="fraud detection")
Turn 1
  messages = [..., assistant(tool_call), tool_result("...docs...")]

LLM → call generate_architecture_options(query="...", constraints={latency:low})
Turn 2
  messages = [..., assistant(tool_call), tool_result("{options:[...]}")]

LLM → call analyze_tradeoffs(options=[...], constraints={...})
Turn 3
  messages = [..., assistant(tool_call), tool_result("{selected:streaming_medallion}")]

LLM → call validate_design(selected_option=..., constraints={...})
Turn 4
  messages = [..., assistant(tool_call), tool_result("{valid:true, issues:[]}")]

LLM → text answer (no tool_calls)
  → agent returns final answer
```

The LLM sees the full `messages` history on every call. That is how it knows what the
previous tool returned when deciding the next step. There is no explicit state object —
the `messages` list is the state.

---

## 6. Complete end-to-end flow for a real query

**Query:** `"Design a real-time Databricks architecture for fraud detection with strict compliance controls."`

```
Step 1 — Authentication
  WorkspaceClient reads credentials from env / .databrickscfg
  w.tokens.create() → short-lived PAT
  PAT used by OpenAI client for LLM calls
  Same WorkspaceClient used by DatabricksMCPClient for tool calls

Step 2 — Tool loading (at agent construction)
  MCP server /api/2.0/mcp/vector-search/{catalog}/{schema}
    → lists kb_chunks_index
    → creates ToolInfo with spec + exec_fn closure
  Custom tools: generate_architecture_options, analyze_tradeoffs,
                validate_design, health_check
  All added to ToolRegistry
  All specs passed to LLM on every chat() call

Step 3 — chat() loop, iteration 1
  LLM reads: "real-time", "fraud detection", system prompt says "ground first"
  LLM picks: kb_chunks_index
  Agent executes:
    DatabricksMCPClient.call_tool(kb_chunks_index, {query:"real-time fraud detection"})
    → MCP server embeds query with databricks-gte-large-en
    → cosine similarity search on kb_chunks Delta table
    → returns top chunks with titles, text, URLs
  Result string appended to messages as role=tool

Step 4 — iteration 2
  LLM reads preceding chunks (evidence)
  LLM picks: generate_architecture_options
    with constraints={latency:"low", governance:"strict", ingestion:"streaming"}
  Agent executes: _generate_options()
    → returns 3 options, streaming_medallion first (matches constraints)
  Result JSON appended to messages

Step 5 — iteration 3
  LLM reads 3 options
  LLM picks: analyze_tradeoffs(options=[...], constraints={latency:"low", governance:"strict"})
  Agent executes: _analyze_tradeoffs()
    latency weight = 3 (user said "real-time")
    governance weight = 3 (user said "strict compliance")
    streaming_medallion: latency=5×3 + governance=4×3 = 27
    hybrid_event_batch:  latency=3×3 + governance=5×3 = 24
    batch_plus_serving:  latency=3×3 + governance=4×3 = 21
    → streaming_medallion wins
  Result JSON appended to messages

Step 6 — iteration 4
  LLM reads winner = streaming_medallion
  LLM picks: validate_design(selected_option=streaming_medallion, constraints={governance:"strict"})
  Agent executes: _validate_design()
    governance strict + streaming_medallion does not contain "hybrid"
    → issue added: "include Unity Catalog policies, row/column controls..."
    valid = False, issues = [governance warning]
  Result JSON appended to messages

Step 7 — iteration 5
  LLM has: evidence + options + ranking + validation issues
  LLM composes: final text recommendation surfacing
    - selected architecture
    - why it was chosen
    - governance issue to fix
    - alternatives considered
    - citations from KB chunks
  LLM returns text (no tool_calls)
  chat() returns final answer to user
```

---

## 7. File map

| File | Role |
|---|---|
| `src/arch_designer_agent/agent.py` | Agent class, `chat()` loop, auth, tool loading |
| `src/arch_designer_agent/mcp.py` | `ToolInfo`, `ToolRegistry`, MCP client utilities |
| `src/arch_designer_agent/agent_tools.py` | Custom tool logic + `ToolInfo` wrappers |
| `src/arch_designer_agent/vector_search.py` | `VectorSearchManager` used by the custom retrieve tool |
| `src/arch_designer_agent/config.py` | `ProjectConfig` — catalog, schema, endpoints, system prompt |
| `notebooks/3.2_arch_designer_agent.py` | Runnable walkthrough: load tools, register, run 3 test queries |
| `docs/PIPELINE_GUIDE.md` | Data pipeline (ingestion → chunking → VS index) that feeds the agent |
