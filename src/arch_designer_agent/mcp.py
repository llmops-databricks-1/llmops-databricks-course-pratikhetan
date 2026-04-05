"""MCP (Model Context Protocol) integration utilities."""

import os
from collections.abc import Callable
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from loguru import logger
from pydantic import BaseModel


class DatabricksOAuth:
    """Production-grade token provider for Databricks agents.

    Auth strategy (checked in order):
    1. SPN (Service Principal) — DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET
       env vars are set.  Uses M2M OAuth via `databricks.sdk.oauth.ClientCredentials`.
       Tokens are cached and auto-refreshed by the SDK — no expiry surprises.
       This is the recommended path for Databricks Jobs and CI/CD pipelines.

    2. PAT fallback — no SPN env vars found (interactive notebook sessions).
       Creates a short-lived PAT via `w.tokens.create()`.  Token lifetime is
       capped at 1 hour; fine for interactive/dev use.

    Usage:
        oauth = DatabricksOAuth(w)
        token = oauth.token()   # safe to call on every request
    """

    def __init__(self, w: WorkspaceClient) -> None:
        self._w = w
        self._spn_provider: Any | None = None

        client_id = os.environ.get("DATABRICKS_CLIENT_ID", "")
        client_secret = os.environ.get("DATABRICKS_CLIENT_SECRET", "")

        if client_id and client_secret:
            try:
                from databricks.sdk.oauth import ClientCredentials  # type: ignore[import]

                self._spn_provider = ClientCredentials(
                    client_id=client_id,
                    client_secret=client_secret,
                    host=w.config.host,
                    scopes=["all-apis"],
                    use_params=False,
                )
                logger.info("Auth: SPN M2M OAuth (auto-refresh enabled)")
            except Exception as exc:
                logger.warning(f"Auth: SPN env vars set but ClientCredentials failed ({exc}); falling back to PAT")
        else:
            logger.info("Auth: no SPN env vars found — using short-lived PAT (notebook/dev mode)")

    def token(self) -> str:
        """Return a valid bearer token.

        SPN path: returns a cached OAuth token; refreshes transparently when expired.
        PAT path: creates a new short-lived PAT good for 1 hour.
        Safe to call on every request.
        """
        if self._spn_provider is not None:
            return self._spn_provider.token().access_token
        return self._w.tokens.create(lifetime_seconds=3600).token_value

    @property
    def is_spn(self) -> bool:
        """True when using SPN M2M OAuth (production mode)."""
        return self._spn_provider is not None


def load_spn_credentials(secret_scope: str) -> None:
    """Load SPN credentials from Databricks Secrets into environment variables.

    Call this ONCE at the top of a Databricks job/notebook before instantiating
    WorkspaceClient or DatabricksExpertAgent.  Mirrors the pattern from
    notebooks/3.5.spn_authentication_in_action.py.

    Requires two secrets in the given scope:
        client_id     — the Service Principal application ID
        client_secret — the Service Principal OAuth secret

    Args:
        secret_scope: Databricks secret scope name (e.g. "llmops_course")

    Example (Databricks notebook):
        load_spn_credentials("llmops_course")
        w = WorkspaceClient()   # WorkspaceClient now picks up SPN credentials
        agent = DatabricksExpertAgent(spark=spark, config=cfg, workspace_client=w)
    """
    try:
        # dbutils is injected by Databricks into notebook/job globals — not a regular import.
        # Reference it directly; NameError is raised cleanly outside Databricks.
        _dbutils = dbutils  # type: ignore[name-defined]  # noqa: F821
        os.environ["DATABRICKS_CLIENT_ID"] = _dbutils.secrets.get(secret_scope, "client_id")
        os.environ["DATABRICKS_CLIENT_SECRET"] = _dbutils.secrets.get(secret_scope, "client_secret")
        logger.info(f"SPN credentials loaded from secret scope '{secret_scope}'")
    except Exception as exc:
        logger.warning(f"load_spn_credentials: could not load from secrets ({exc}). Set env vars manually.")


class ToolInfo(BaseModel):
    """Tool information for agent integration.

    Attributes:
        name: Tool name
        spec: JSON description of the tool (OpenAI function-calling format)
        exec_fn: Function that implements the tool logic
    """

    name: str
    spec: dict
    exec_fn: Callable

    class Config:
        arbitrary_types_allowed = True


class ToolRegistry:
    """Registry for managing agent tools (custom + MCP)."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolInfo] = {}

    def register(self, tool: ToolInfo) -> None:
        """Register a single tool."""
        self._tools[tool.name] = tool

    def register_many(self, tools: list[ToolInfo]) -> None:
        """Register a list of tools at once."""
        for tool in tools:
            self.register(tool)

    def get_tool(self, name: str) -> ToolInfo:
        """Return a tool by name, raising ValueError if not found."""
        if name not in self._tools:
            raise ValueError(f"Tool not found: {name}")
        return self._tools[name]

    def get_all_specs(self) -> list[dict]:
        """Return all tool specs in OpenAI function-calling format."""
        return [tool.spec for tool in self._tools.values()]

    def execute(self, name: str, args: dict[str, Any]) -> Any:
        """Execute a registered tool with keyword arguments."""
        return self.get_tool(name).exec_fn(**args)

    def list_tools(self) -> list[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())


def create_managed_exec_fn(server_url: str, tool_name: str, w: WorkspaceClient) -> Callable:
    """Create an execution function for an MCP tool.

    Args:
        server_url: MCP server URL
        tool_name: Name of the tool
        w: Databricks workspace client

    Returns:
        Callable that executes the tool
    """

    def exec_fn(**kwargs):
        client = DatabricksMCPClient(server_url=server_url, workspace_client=w)
        response = client.call_tool(tool_name, kwargs)
        return "".join([c.text for c in response.content])

    return exec_fn


async def create_mcp_tools(w: WorkspaceClient, url_list: list[str]) -> list[ToolInfo]:
    """Create tools from MCP servers.

    Args:
        w: Databricks workspace client
        url_list: List of MCP server URLs

    Returns:
        List of ToolInfo objects
    """
    tools = []
    for server_url in url_list:
        try:
            mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=w)
            mcp_tools = mcp_client.list_tools()
        except Exception as exc:
            cause = exc
            if hasattr(exc, "exceptions") and exc.exceptions:
                cause = exc.exceptions[0]
            logger.warning(f"  Skipping MCP server {server_url}: {type(cause).__name__}: {cause}")
            continue
        for mcp_tool in mcp_tools:
            input_schema = mcp_tool.inputSchema.copy() if mcp_tool.inputSchema else {}
            tool_spec = {
                "type": "function",
                "function": {
                    "name": mcp_tool.name,
                    "parameters": input_schema,
                    "description": mcp_tool.description or f"Tool: {mcp_tool.name}",
                },
            }
            exec_fn = create_managed_exec_fn(server_url, mcp_tool.name, w)
            tools.append(ToolInfo(name=mcp_tool.name, spec=tool_spec, exec_fn=exec_fn))
    return tools
