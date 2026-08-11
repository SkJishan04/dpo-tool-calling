"""Tool and function schema definitions for DPO training."""

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class ParameterSchema(BaseModel):
    """Parameter definition for a tool."""
    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (string, int, float, bool, array, object)")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(default=True, description="Is parameter required")
    enum: List[str] = Field(default_factory=list, description="Valid enum values")
    default: Any = Field(default=None, description="Default value")


class ToolSchema(BaseModel):
    """Tool/Function schema definition."""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    category: str = Field(..., description="Tool category (search, database, api, calculation)")
    parameters: List[ParameterSchema] = Field(default_factory=list, description="Tool parameters")
    required_params: List[str] = Field(default_factory=list, description="Required parameters")


class ToolCallJson(BaseModel):
    """Expected JSON output for tool calls."""
    should_call_tool: bool = Field(..., description="Whether to call a tool")
    tool_name: str = Field(default=None, description="Name of the tool to call")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    reasoning: str = Field(..., description="Reasoning for the decision")


class DirectAnswerJson(BaseModel):
    """Expected JSON output for direct text answers."""
    should_call_tool: bool = Field(default=False, description="Always False")
    answer: str = Field(..., description="Direct text answer")
    reasoning: str = Field(..., description="Reasoning for not calling a tool")


# Pre-defined Tool Schemas
SEARCH_TOOL = ToolSchema(
    name="search",
    description="Search the internet for information",
    category="search",
    parameters=[
        ParameterSchema(name="query", type="string", description="Search query"),
        ParameterSchema(name="num_results", type="int", description="Number of results", default=10),
    ],
    required_params=["query"]
)

DATABASE_QUERY_TOOL = ToolSchema(
    name="database_query",
    description="Query a database for structured data",
    category="database",
    parameters=[
        ParameterSchema(name="table", type="string", description="Table name"),
        ParameterSchema(name="filters", type="object", description="Query filters"),
    ],
    required_params=["table"]
)

CALCULATOR_TOOL = ToolSchema(
    name="calculator",
    description="Perform mathematical calculations",
    category="calculation",
    parameters=[
        ParameterSchema(name="expression", type="string", description="Mathematical expression"),
    ],
    required_params=["expression"]
)

WEATHER_TOOL = ToolSchema(
    name="weather",
    description="Get weather information for a location",
    category="api",
    parameters=[
        ParameterSchema(name="location", type="string", description="Location name or coordinates"),
        ParameterSchema(name="units", type="string", description="Temperature units", enum=["celsius", "fahrenheit"]),
    ],
    required_params=["location"]
)

# Registry of all tools
TOOL_REGISTRY = {
    "search": SEARCH_TOOL,
    "database_query": DATABASE_QUERY_TOOL,
    "calculator": CALCULATOR_TOOL,
    "weather": WEATHER_TOOL,
}