"""
Snowflake MCP OpenAI Integration

A powerful integration between Snowflake's MCP server and OpenAI's SDK,
enabling intelligent data analysis and reasoning capabilities.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .agents.basic_agent import main as run_basic_agent
from .agents.reasoning_agent import SmartReasoningAgent

__all__ = [
    "run_basic_agent",
    "SmartReasoningAgent",
] 