#!/usr/bin/env python3
"""
Basic Snowflake MCP Agent

A simple Python app that:
 1. Reads OPENAI_API_KEY from the environment
 2. Launches a local Snowflake MCP server via stdio with extended timeout
 3. Instantiates an OpenAI Agent connected to the MCP server
 4. Runs sample queries against Snowflake Cortex AI tools via Runner

Usage:
  $ source .venv/bin/activate
  $ export OPENAI_API_KEY="sk-..."
  $ export SNOWFLAKE_ACCOUNT_IDENTIFIER="your-account"
  $ export SNOWFLAKE_USERNAME="your-username"
  $ export SNOWFLAKE_PAT="your-pat-token"
  $ python src/agents/basic_agent.py
"""
import os
import asyncio
from pathlib import Path
from agents import Agent, Runner
from agents.mcp.server import MCPServerStdio


async def main():
    # Ensure the OpenAI API key is set
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError(
            "Environment variable OPENAI_API_KEY is not set. "
            "Please export your OpenAI secret key before running."
        )

    # Get Snowflake credentials from environment
    snowflake_account = os.getenv("SNOWFLAKE_ACCOUNT_IDENTIFIER")
    snowflake_username = os.getenv("SNOWFLAKE_USERNAME")
    snowflake_pat = os.getenv("SNOWFLAKE_PAT")
    
    if not all([snowflake_account, snowflake_username, snowflake_pat]):
        raise RuntimeError(
            "Missing Snowflake credentials. Please set: "
            "SNOWFLAKE_ACCOUNT_IDENTIFIER, SNOWFLAKE_USERNAME, SNOWFLAKE_PAT"
        )

    # Get the config file path
    config_path = Path(__file__).parent.parent / "config" / "tools_config.yaml"
    
    # Configure and launch the local MCP server via stdio
    mcp_server = MCPServerStdio(
        params={
            "command": "uvx",
            "args": [
                "--from",
                "git+https://github.com/Snowflake-Labs/mcp",
                "mcp-server-snowflake",
                "--service-config-file",
                str(config_path),
                "--account-identifier",
                snowflake_account,
                "--username",
                snowflake_username,
                "--pat",
                snowflake_pat
            ]
        },
        cache_tools_list=True,                 # Cache tool metadata between runs
        name="snowflake-mcp-server",          # Descriptive server name
        client_session_timeout_seconds=60      # Extend init timeout to 60s
    )

    async with mcp_server as server:
        # List available tools
        tools = await server.list_tools()
        print("Available MCP tools:", tools)

        # Create an agent that leverages Snowflake MCP tools
        agent = Agent(
            name="snowflake-agent",
            instructions=(
                "You are an agent that uses Snowflake Cortex AI tools to interact with the user's Snowflake environment. "
                "Use the available tools to answer data queries accurately."
            ),
            mcp_servers=[server]
        )

        # Example user query
        query = "Why are sales in Germany outperforming other regions?"
        print(f"Running query: {query}")

        # Execute the query via Runner
        try:
            run_result = await Runner.run(agent, query)
            print("Agent result:", run_result.final_output)
        except Exception as e:
            print("Error during agent execution:", e)


if __name__ == "__main__":
    asyncio.run(main()) 