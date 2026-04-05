# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.2: Model Context Protocol (MCP) Integration
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What is MCP?
# MAGIC - MCP vs custom functions
# MAGIC - Databricks MCP servers
# MAGIC - Vector Search MCP
# MAGIC - Genie Space MCP
# MAGIC - Creating MCP tools for agents
# COMMAND ----------

import asyncio
import json

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
import nest_asyncio
from loguru import logger
from pyspark.sql import SparkSession

from arxiv_curator.config import load_config, get_env
from arxiv_curator.mcp import create_mcp_tools

# Enable nested event loops (required for Databricks notebooks)
nest_asyncio.apply()

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

# Load configuration
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. What is Model Context Protocol (MCP)?
# MAGIC
# MAGIC **MCP** is a standardized protocol for connecting AI models to external data sources and tools.
# MAGIC
# MAGIC ### Key Concepts:
# MAGIC
# MAGIC - **MCP Server**: Exposes tools and resources
# MAGIC - **MCP Client**: Connects to servers and calls tools
# MAGIC - **Tools**: Functions that can be called
# MAGIC - **Resources**: Data that can be accessed
# MAGIC
# MAGIC ### Why MCP?
# MAGIC
# MAGIC **Standardized**: Common protocol across different systems
# MAGIC **Reusable**: One MCP server, many agents
# MAGIC **Managed**: Databricks manages the infrastructure
# MAGIC **Secure**: Built-in authentication and authorization
# MAGIC **Scalable**: Enterprise-grade performance

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. MCP vs Custom Functions
# MAGIC
# MAGIC | Aspect | Custom Functions | MCP |
# MAGIC |--------|-----------------|-----|
# MAGIC | **Setup** | Write Python code | Use existing MCP servers |
# MAGIC | **Maintenance** | You maintain | Databricks maintains |
# MAGIC | **Reusability** | Per-agent | Across agents |
# MAGIC | **Security** | Manual | Built-in |
# MAGIC | **Scalability** | Manual | Automatic |
# MAGIC | **Best For** | Custom logic | Standard operations |
# MAGIC
# MAGIC **Use MCP when**: You need standard operations (search, query, etc.)
# MAGIC **Use Custom Functions when**: You need custom business logic

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Databricks MCP Servers
# MAGIC
# MAGIC Databricks provides managed MCP servers for:
# MAGIC
# MAGIC 1. **Vector Search MCP**: Search vector indexes
# MAGIC 2. **Genie Space MCP**: Query data using natural language
# MAGIC 3. **Unity Catalog Functions MCP**: Execute UC functions
# MAGIC 4. **SQL Warehouse MCP**: Execute SQL queries

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Vector Search MCP

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vector Search MCP URL Format:
# MAGIC ```
# MAGIC {workspace_host}/api/2.0/mcp/vector-search/{catalog}/{schema}
# MAGIC ```
# MAGIC
# MAGIC **How it works:**
# MAGIC - The MCP server scans all vector search indexes in the specified catalog/schema
# MAGIC - For each index, it creates a tool with name: `catalog__schema__index_name`
# MAGIC - Each tool takes a single parameter: `query` (the search text)
# MAGIC - The tool automatically handles embedding and similarity search

# COMMAND ----------

# Create Vector Search MCP URL (matches O'Reilly source)
host = w.config.host
vector_search_mcp_url = f"{host}/api/2.0/mcp/vector-search/{cfg.catalog}/{cfg.schema}"

logger.info(f"Vector Search MCP URL:")
logger.info(vector_search_mcp_url)

# COMMAND ----------

# MAGIC %md
# MAGIC ### List Available Tools from Vector Search MCP

# COMMAND ----------

# Connect to Vector Search MCP
vs_mcp_client = DatabricksMCPClient(
    server_url=vector_search_mcp_url,
    workspace_client=w
)

# List available tools
vs_tools = vs_mcp_client.list_tools()

logger.info(f"Vector Search MCP Tools ({len(vs_tools)}):")
logger.info("=" * 80)
for tool in vs_tools:
    logger.info(f"Tool: {tool.name}")
    logger.info(f"Description: {tool.description}")
    if tool.inputSchema:
        logger.info(f"Parameters: {list(tool.inputSchema.get('properties', {}).keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Call Vector Search Tool
# MAGIC
# MAGIC **Important**: The MCP tool name uses double underscores:
# MAGIC - Tool name: `workspace__course_data__arxiv_index`
# MAGIC - Parameter: `query` (just the search query text)
# MAGIC - The index is already specified in the tool name itself

# COMMAND ----------

# Search for papers about machine learning
# The tool name is the index name with '__' separators
tool_name = f"{cfg.catalog}__{cfg.schema}__arxiv_index"

search_result = vs_mcp_client.call_tool(
    tool_name,
    {
        "query": "machine learning and neural networks"
    }
)

logger.info("Search Results:")
logger.info("=" * 80)
for content in search_result.content:
    logger.info(content.text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Genie Space MCP

# COMMAND ----------

# MAGIC %md
# MAGIC ### Genie Space MCP URL Format:
# MAGIC ```
# MAGIC {workspace_host}/api/2.0/mcp/genie/{genie_space_id}
# MAGIC ```
# MAGIC
# MAGIC **Genie** allows natural language queries over your data.

# COMMAND ----------

# Check if Genie space is configured
if hasattr(cfg, 'genie_space_id') and cfg.genie_space_id:
    genie_mcp_url = f"{host}/api/2.0/mcp/genie/{cfg.genie_space_id}"
    logger.info(f"Genie MCP URL:")
    logger.info(genie_mcp_url)
    
    # Connect to Genie MCP
    genie_mcp_client = DatabricksMCPClient(
        server_url=genie_mcp_url,
        workspace_client=w
    )
    
    # List available tools
    genie_tools = genie_mcp_client.list_tools()
    
    logger.info(f"Genie MCP Tools ({len(genie_tools)}):")
    logger.info("=" * 80)
    for tool in genie_tools:
        logger.info(f"Tool: {tool.name}")
        logger.info(f"Description: {tool.description}")
else:
    logger.warning("⚠️ Genie space not configured in project_config.yml")
    logger.info("To use Genie MCP, add 'genie_space_id' to your configuration")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Creating MCP Tools for Agents
# MAGIC
# MAGIC **Using `arxiv_curator.mcp` module:**
# MAGIC
# MAGIC We've imported the following from our custom package:
# MAGIC - `ToolInfo`: Pydantic model for tool information (name, spec, exec_fn)
# MAGIC - `create_managed_exec_fn()`: Creates execution functions for MCP tools
# MAGIC - `create_mcp_tools()`: Converts MCP server tools to agent-compatible tools
# MAGIC
# MAGIC These utilities handle:
# MAGIC 1. Connecting to MCP servers
# MAGIC 2. Listing available tools
# MAGIC 3. Creating OpenAI-compatible tool specifications
# MAGIC 4. Creating execution functions that call the MCP tools

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load All MCP Tools

# COMMAND ----------

# Define MCP server URLs
mcp_urls = [
    f"{host}/api/2.0/mcp/vector-search/{cfg.catalog}/{cfg.schema}"
]

# Add Genie if configured
if hasattr(cfg, 'genie_space_id') and cfg.genie_space_id:
    mcp_urls.append(f"{host}/api/2.0/mcp/genie/{cfg.genie_space_id}")

logger.info(f"Loading tools from {len(mcp_urls)} MCP servers...")

# Create tools
mcp_tools = asyncio.run(create_mcp_tools(w, mcp_urls))

logger.info(f"✓ Loaded {len(mcp_tools)} tools from MCP servers")
logger.info("Available Tools:")
for i, tool in enumerate(mcp_tools, 1):
    logger.info(f"{i}. {tool.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Using MCP Tools

# COMMAND ----------

# Create a tools dictionary for easy access
tools_dict = {tool.name: tool for tool in mcp_tools}

# Example: Use vector search tool directly
# The tool name is the index name with '__' separators
vector_search_tool_name = f"{cfg.catalog}__{cfg.schema}__arxiv_index"

if vector_search_tool_name in tools_dict:
    search_tool = tools_dict[vector_search_tool_name]
    
    # Execute the tool - only takes 'query' parameter
    result = search_tool.exec_fn(
        query="deep learning architectures"
    )
    
    logger.info("Search Results:")
    logger.info(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. MCP Tool Specifications

# COMMAND ----------

# View tool specifications (what the LLM sees)
if mcp_tools:
    logger.info("Tool Specifications for LLM:")
    logger.info("=" * 80)
    
    for tool in mcp_tools[:2]:  # Show first 2 tools
        logger.info(f"Tool: {tool.name}")
        logger.info(json.dumps(tool.spec, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Benefits of MCP

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. **No Code Required**
# MAGIC ```python
# MAGIC # Without MCP: Write custom function
# MAGIC def search_papers(query: str):
# MAGIC     # 50+ lines of code
# MAGIC     pass
# MAGIC
# MAGIC # With MCP: Just use it
# MAGIC tools = asyncio.run(create_mcp_tools(w, [mcp_url]))
# MAGIC ```
# MAGIC
# MAGIC ### 2. **Automatic Updates**
# MAGIC - Databricks maintains the MCP servers
# MAGIC - New features added automatically
# MAGIC - Bug fixes without code changes
# MAGIC
# MAGIC ### 3. **Consistent Interface**
# MAGIC - Same pattern for all MCP tools
# MAGIC - Easy to add new MCP servers
# MAGIC - Standardized error handling
# MAGIC
# MAGIC ### 4. **Enterprise Features**
# MAGIC - Built-in authentication
# MAGIC - Audit logging
# MAGIC - High availability

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Troubleshooting MCP

# COMMAND ----------

def test_mcp_connection(mcp_url: str) -> bool:
    """Test if MCP server is accessible.
    
    Args:
        mcp_url: MCP server URL
        
    Returns:
        True if connection successful
    """
    try:
        client = DatabricksMCPClient(
            server_url=mcp_url,
            workspace_client=w
        )
        tools = client.list_tools()
        logger.info(f"✓ Connected to MCP server")
        logger.info(f"  URL: {mcp_url}")
        logger.info(f"  Tools available: {len(tools)}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to connect to MCP server")
        logger.error(f"  URL: {mcp_url}")
        logger.error(f"  Error: {e}")
        return False

# Test Vector Search MCP
logger.info("Testing Vector Search MCP:")
test_mcp_connection(vector_search_mcp_url)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Using MCP Tools with an Agent
# MAGIC
# MAGIC Now let's create a simple agent that can use MCP tools.

# COMMAND ----------

from openai import OpenAI

class SimpleAgent:
    """A simple agent that can call tools in a loop."""
    
    def __init__(self, llm_endpoint: str, system_prompt: str, tools: list):
        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self._tools_dict = {tool.name: tool for tool in tools}
        self._client = OpenAI(
            api_key=w.tokens.create(lifetime_seconds=1200).token_value,
            base_url=f"{w.config.host}/serving-endpoints"
        )
    
    def get_tool_specs(self) -> list[dict]:
        """Get tool specifications for the LLM."""
        return [tool.spec for tool in self._tools_dict.values()]
    
    def execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool by name."""
        if tool_name not in self._tools_dict:
            raise ValueError(f"Unknown tool: {tool_name}")
        return self._tools_dict[tool_name].exec_fn(**args)
    
    def chat(self, user_message: str, max_iterations: int = 10) -> str:
        """Chat with the agent, allowing tool calls."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        for iteration in range(max_iterations):
            response = self._client.chat.completions.create(
                model=self.llm_endpoint,
                messages=messages,
                tools=self.get_tool_specs() if self._tools_dict else None,
            )
            
            assistant_message = response.choices[0].message
            
            if assistant_message.tool_calls:
                # Add assistant message with tool calls (exclude unsupported fields)
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })
                
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"Calling tool: {tool_name}({tool_args})")
                    
                    try:
                        result = self.execute_tool(tool_name, tool_args)
                    except Exception as e:
                        result = f"Error: {str(e)}"
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })
            else:
                return assistant_message.content
        
        return "Max iterations reached."

# COMMAND ----------

# Create agent with MCP tools
agent = SimpleAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt="You are a helpful research assistant. Use the available tools to search for papers and answer questions.",
    tools=mcp_tools
)

logger.info("✓ Agent created with MCP tools:")
for tool_name in agent._tools_dict.keys():
    logger.info(f"  - {tool_name}")

# COMMAND ----------

# Test agent with MCP vector search tool
logger.info("Testing agent with MCP tools:")
logger.info("=" * 80)

response = agent.chat("Find papers about transformer architectures")
logger.info(f"Agent response: {response}")
