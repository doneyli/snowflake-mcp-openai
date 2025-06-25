#!/usr/bin/env python3
"""
Basic Query Examples

This module demonstrates basic usage patterns for the Snowflake MCP OpenAI integration.
Run these examples to test your setup and understand how to use the basic agent.

Usage:
    python src/examples/basic_queries.py
"""
import asyncio
import os
from pathlib import Path
import sys
from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agents.basic_agent import main as run_basic_agent


async def demo_basic_queries():
    """
    Demonstrate basic query patterns
    """
    print("🎯 Basic Snowflake MCP Query Examples")
    print("=" * 50)
    
    # Load environment variables from .env file  
    load_dotenv()
    
    # Check environment setup
    required_vars = [
        "OPENAI_API_KEY", 
        "SNOWFLAKE_ACCOUNT_IDENTIFIER", 
        "SNOWFLAKE_USERNAME", 
        "SNOWFLAKE_PAT"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file")
        return
    
    print("✅ Environment variables configured")
    print("\n🚀 Running basic agent example...")
    
    try:
        # Run the basic agent with example query
        await run_basic_agent()
    except Exception as e:
        print(f"❌ Error running basic agent: {e}")
        return
    
    print("\n✅ Basic example completed successfully!")
    print("\nTry these query patterns:")
    
    example_queries = [
        "What are our total sales for this quarter?",
        "Show me the top 5 customers by revenue",
        "Find all documents mentioning 'pricing'",
        "What are the trends in our monthly sales?",
        "Analyze customer satisfaction scores",
    ]
    
    for i, query in enumerate(example_queries, 1):
        print(f"{i}. {query}")


if __name__ == "__main__":
    asyncio.run(demo_basic_queries()) 