#!/usr/bin/env python3
"""
Enhanced Reasoning Snowflake MCP Agent

An enhanced Python app with intelligent reasoning capabilities that:
 1. Reads OPENAI_API_KEY from the environment
 2. Launches a local Snowflake MCP server via stdio with extended timeout
 3. Instantiates an OpenAI Agent with adaptive reasoning capabilities
 4. Implements smart iteration with early stopping and quality assessment
 5. Uses parallel processing and caching for efficiency

Usage:
  $ source .venv/bin/activate
  $ export OPENAI_API_KEY="sk-..."
  $ export SNOWFLAKE_ACCOUNT_IDENTIFIER="your-account"
  $ export SNOWFLAKE_USERNAME="your-username"
  $ export SNOWFLAKE_PAT="your-pat-token"
  $ python src/agents/reasoning_agent.py
"""
import os
import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from agents import Agent, Runner
from agents.mcp.server import MCPServerStdio


class ReasoningState(Enum):
    INITIAL = "initial"
    ANALYZING = "analyzing" 
    REFINING = "refining"
    COMPLETED = "completed"
    FAILED = "failed"
    EARLY_COMPLETION = "early_completion"


@dataclass
class ReasoningStep:
    step_number: int
    state: ReasoningState
    query: str
    result: Optional[str] = None
    analysis: Optional[str] = None
    next_actions: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    execution_time: float = 0.0
    result_quality: str = "unknown"


class SmartReasoningAgent:
    def __init__(self, base_agent: Agent, max_iterations: int = 5, confidence_threshold: float = 0.8, 
                 early_stop_threshold: float = 0.9, quality_threshold: float = 0.85):
        self.base_agent = base_agent
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.early_stop_threshold = early_stop_threshold  # Higher threshold for early stopping
        self.quality_threshold = quality_threshold
        self.reasoning_history: List[ReasoningStep] = []
        self.query_cache: Dict[str, str] = {}  # Cache similar queries
        self.covered_aspects: Set[str] = set()  # Track what's been covered
        self.suggested_queries: List[str] = []  # Store Cortex suggested queries
        self.suggested_query_index: int = 0  # Track which suggested query to use next
        
    async def reason_and_execute(self, initial_query: str) -> Dict[str, Any]:
        """
        Intelligent reasoning loop leveraging Cortex Analyst for query interpretation and routing
        """
        print(f"\n🧠 Starting Cortex-powered reasoning for: '{initial_query}'")
        print("=" * 80)
        
        current_query = initial_query
        iteration = 0
        start_time = time.time()
        query_interpretation = None
        
        # First, use Cortex Analyst to interpret the query and get initial guidance
        cortex_result = await self._use_cortex_analyst(initial_query)
        query_interpretation = cortex_result.get("interpretation", "")
        suggested_queries = cortex_result.get("alternative_queries", [])
        has_direct_answer = cortex_result.get("has_answer", False)
        
        print(f"🎯 Cortex interpretation: {query_interpretation}")
        if has_direct_answer:
            print("✨ Cortex provided direct answer!")
        elif suggested_queries:
            print(f"🔄 Cortex suggested {len(suggested_queries)} alternative queries")
        
        # Adjust strategy based on Cortex feedback
        if has_direct_answer:
            adjusted_max_iterations = 1  # Direct answer available
        else:
            adjusted_max_iterations = min(self.max_iterations, max(2, len(suggested_queries)))
        
        print(f"📊 Reasoning strategy: {adjusted_max_iterations} max iterations based on Cortex analysis")
        
        while iteration < adjusted_max_iterations:
            iteration += 1
            print(f"\n🔄 Iteration {iteration}/{adjusted_max_iterations}")
            print("-" * 40)
            
            step_start_time = time.time()
            
            # Check cache first
            cached_result = self._check_cache(current_query)
            if cached_result:
                print("💾 Using cached result")
                step_result = cached_result
            else:
                # Execute current query
                print(f"📋 Executing: {current_query}")
                run_result = await Runner.run(self.base_agent, current_query)
                step_result = run_result.final_output
                self._update_cache(current_query, step_result)
            
            execution_time = time.time() - step_start_time
            
            # Create reasoning step
            step = ReasoningStep(
                step_number=iteration,
                state=ReasoningState.ANALYZING,
                query=current_query,
                result=step_result,
                execution_time=execution_time
            )
            
            try:
                # Analyze the result with enhanced assessment
                analysis_result = await self._enhanced_analyze_result(
                    original_query=initial_query,
                    current_query=current_query,
                    result=step.result,
                    iteration=iteration,
                    covered_aspects=self.covered_aspects
                )
                
                step.analysis = analysis_result.get("analysis", "")
                step.confidence_score = analysis_result.get("confidence", 0.0)
                step.next_actions = analysis_result.get("next_actions", [])
                step.result_quality = analysis_result.get("quality", "unknown")
                
                # Update covered aspects
                new_aspects = analysis_result.get("covered_aspects", [])
                self.covered_aspects.update(new_aspects)
                
                print(f"📊 Result: {step.result[:150]}..." if len(step.result) > 150 else f"📊 Result: {step.result}")
                print(f"🔍 Analysis: {step.analysis}")
                print(f"📈 Confidence: {step.confidence_score:.2f} | Quality: {step.result_quality}")
                print(f"⏱️  Execution time: {execution_time:.2f}s")
                
                self.reasoning_history.append(step)
                
                # Enhanced stopping conditions
                should_stop, stop_reason = self._should_stop_reasoning(step, iteration, adjusted_max_iterations)
                
                if should_stop:
                    if step.confidence_score >= self.early_stop_threshold:
                        step.state = ReasoningState.EARLY_COMPLETION
                        print(f"🚀 Early completion: {stop_reason}")
                    else:
                        step.state = ReasoningState.COMPLETED
                        print(f"✅ Reasoning completed: {stop_reason}")
                    break
                
                # Generate smarter follow-up query
                if step.next_actions:
                    step.state = ReasoningState.REFINING
                    current_query = await self._generate_smart_followup_query(
                        original_query=initial_query,
                        current_result=step.result,
                        next_actions=step.next_actions,
                        covered_aspects=self.covered_aspects,
                        iteration=iteration
                    )
                    print(f"🔄 Next query: {current_query}")
                else:
                    print("⚠️  No clear next actions identified")
                    break
                    
            except Exception as e:
                step.state = ReasoningState.FAILED
                step.result = f"Error: {str(e)}"
                self.reasoning_history.append(step)
                print(f"❌ Error in iteration {iteration}: {e}")
                break
        
        total_time = time.time() - start_time
        
        # Compile final result with quality assessment
        final_result = await self._compile_intelligent_final_result(initial_query)
        
        return {
            "original_query": initial_query,
            "cortex_interpretation": query_interpretation,
            "final_result": final_result,
            "reasoning_steps": len(self.reasoning_history),
            "total_iterations": iteration,
            "estimated_iterations": adjusted_max_iterations,
            "total_execution_time": total_time,
            "average_step_time": total_time / iteration if iteration > 0 else 0,
            "final_confidence": self.reasoning_history[-1].confidence_score if self.reasoning_history else 0.0,
            "completion_reason": self.reasoning_history[-1].state.value if self.reasoning_history else "unknown",
            "covered_aspects": list(self.covered_aspects),
            "reasoning_history": self.reasoning_history,
            "used_cortex_direct_answer": has_direct_answer
        }
    
    async def _use_cortex_analyst(self, query: str) -> Dict[str, Any]:
        """
        Use Cortex Analyst tool to interpret query and get guidance or direct answer
        """
        cortex_prompt = f"""
        Please use the cortex_analyst tool to analyze this query: "{query}"
        
        The cortex_analyst tool will provide:
        1. An interpretation of what the user is asking
        2. Either a direct answer (if available) or alternative queries to try
        
        Based on the cortex_analyst response, provide a JSON summary with:
        - "interpretation": The interpretation provided by cortex_analyst
        - "has_answer": Boolean indicating if a direct answer was provided
        - "answer": The direct answer if available, null otherwise
        - "alternative_queries": List of alternative queries suggested by cortex_analyst
        - "next_action": Recommended next action based on cortex analysis
        
        Use the cortex_analyst tool now.
        """
        
        try:
            cortex_result = await Runner.run(self.base_agent, cortex_prompt)
            result_text = cortex_result.final_output.strip()
            
            # Try to extract structured information from cortex response
            # Look for the actual cortex_analyst tool output in the response
            parsed_result = self._parse_cortex_response(result_text)
            
            return parsed_result
            
        except Exception as e:
            print(f"⚠️  Error using Cortex Analyst: {e}")
            return {
                "interpretation": f"Failed to interpret query: {query}",
                "has_answer": False,
                "answer": None,
                "alternative_queries": [],
                "next_action": "proceed_with_standard_analysis"
            }
    
    def _parse_cortex_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the Cortex Analyst response to extract structured information
        """
        # This is a simplified parser - you may need to adjust based on actual cortex_analyst output format
        try:
            # Look for JSON in the response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
                return json.loads(json_text)
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
                return json.loads(json_text)
            
            # Fallback: Parse text-based response
            return self._parse_cortex_text_response(response_text)
            
        except Exception as e:
            print(f"⚠️  Error parsing Cortex response: {e}")
            return self._create_fallback_cortex_result(response_text)
    
    def _parse_cortex_text_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse text-based Cortex Analyst response when JSON isn't available
        """
        lines = response_text.split('\n')
        interpretation = ""
        alternative_queries = []
        has_answer = False
        answer = None
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for interpretation
            if "interpretation" in line.lower() or "understanding" in line.lower():
                current_section = "interpretation"
                interpretation = line.split(":", 1)[-1].strip() if ":" in line else line
            elif "alternative" in line.lower() and "quer" in line.lower():
                current_section = "queries"
            elif "answer" in line.lower() and not "alternative" in line.lower():
                current_section = "answer"
                if ":" in line:
                    answer = line.split(":", 1)[-1].strip()
                    has_answer = bool(answer and answer.lower() not in ["none", "null", "n/a"])
            elif current_section == "interpretation" and not any(keyword in line.lower() for keyword in ["alternative", "answer", "suggest"]):
                interpretation += " " + line
            elif current_section == "queries" and (line.startswith("-") or line.startswith("*") or line.startswith("1.")):
                query = line.lstrip("-*1234567890. ").strip()
                if query:
                    alternative_queries.append(query)
            elif current_section == "answer" and not any(keyword in line.lower() for keyword in ["interpretation", "alternative", "suggest"]):
                if answer:
                    answer += " " + line
                else:
                    answer = line
                    has_answer = bool(answer and answer.lower() not in ["none", "null", "n/a"])
        
        return {
            "interpretation": interpretation.strip(),
            "has_answer": has_answer,
            "answer": answer.strip() if answer else None,
            "alternative_queries": alternative_queries,
            "next_action": "use_direct_answer" if has_answer else "try_alternative_queries"
        }
    
    def _create_fallback_cortex_result(self, response_text: str) -> Dict[str, Any]:
        """
        Create a fallback result when Cortex parsing fails
        """
        return {
            "interpretation": f"Query analysis (parsing failed): {response_text[:200]}...",
            "has_answer": len(response_text) > 100,  # Assume longer responses might have answers
            "answer": response_text if len(response_text) > 100 else None,
            "alternative_queries": [],
            "next_action": "review_response_manually"
        }
    
    def _check_cache(self, query: str) -> Optional[str]:
        """
        Check if we have a cached result for similar query
        """
        # Simple cache check - in production, you might want fuzzy matching
        return self.query_cache.get(query.lower().strip())
    
    def _update_cache(self, query: str, result: str):
        """
        Update query cache with result
        """
        self.query_cache[query.lower().strip()] = result
    
    def _should_stop_reasoning(self, step: ReasoningStep, iteration: int, max_iterations: int) -> tuple[bool, str]:
        """
        Enhanced logic to determine if reasoning should stop
        """
        # Early completion with high confidence and quality
        if step.confidence_score >= self.early_stop_threshold and step.result_quality in ["high", "excellent"]:
            return True, f"High confidence ({step.confidence_score:.2f}) and quality ({step.result_quality})"
        
        # Standard completion threshold
        if step.confidence_score >= self.confidence_threshold:
            return True, f"Confidence threshold met ({step.confidence_score:.2f})"
        
        # No next actions and reasonable confidence
        if not step.next_actions and step.confidence_score >= 0.6:
            return True, f"No more actions needed (confidence: {step.confidence_score:.2f})"
        
        # Max iterations reached
        if iteration >= max_iterations:
            return True, f"Maximum iterations reached ({iteration}/{max_iterations})"
        
        # Quality-based stopping
        if step.result_quality == "excellent" and step.confidence_score >= 0.7:
            return True, f"Excellent quality result with good confidence"
        
        return False, ""
    
    async def _enhanced_analyze_result(self, original_query: str, current_query: str, result: str, 
                                     iteration: int, covered_aspects: Set[str], cortex_interpretation: str = None) -> Dict[str, Any]:
        """
        Enhanced result analysis with quality assessment, aspect tracking, and Cortex context
        """
        covered_list = list(covered_aspects) if covered_aspects else []
        cortex_context = f"\nCORTEX INTERPRETATION: {cortex_interpretation}" if cortex_interpretation else ""
        
        analysis_prompt = f"""
        Analyze this query result with enhanced assessment criteria and Cortex Analyst context.

        ORIGINAL QUERY: {original_query}
        CURRENT QUERY: {current_query}
        ITERATION: {iteration}
        RESULT: {result}
        ALREADY COVERED ASPECTS: {covered_list}{cortex_context}

        Provide a JSON response with:
        1. "analysis": Brief analysis of what was accomplished and gaps
        2. "confidence": Float 0-1 indicating completeness of original query answer
        3. "quality": "poor", "fair", "good", "high", or "excellent" - assess result quality
        4. "next_actions": List of specific follow-up actions (empty if complete)
        5. "covered_aspects": List of new aspects covered in this result
        6. "is_concise": Boolean - does this result concisely answer the original query?
        7. "cortex_alignment": How well does this result align with Cortex's interpretation of the query?

        Quality criteria:
        - Excellent: Complete, detailed, actionable insights with supporting data
        - High: Good coverage with specific details and clear conclusions
        - Good: Adequate information with some specifics
        - Fair: Basic information but lacks depth or specifics
        - Poor: Incomplete, vague, or insufficient information

        Consider the Cortex interpretation when assessing completeness and alignment.

        Response format: {{"analysis": "...", "confidence": 0.X, "quality": "...", "next_actions": [...], "covered_aspects": [...], "is_concise": true/false, "cortex_alignment": "high/medium/low"}}
        """
        
        try:
            analysis_result = await Runner.run(self.base_agent, analysis_prompt)
            result_text = analysis_result.final_output.strip()
            
            # Extract JSON
            if "```json" in result_text:
                json_start = result_text.find("```json") + 7
                json_end = result_text.find("```", json_start)
                result_text = result_text[json_start:json_end].strip()
            elif "{" in result_text and "}" in result_text:
                json_start = result_text.find("{")
                json_end = result_text.rfind("}") + 1
                result_text = result_text[json_start:json_end]
            
            parsed = json.loads(result_text)
            
            # Boost confidence for concise results
            if parsed.get("is_concise", False) and parsed.get("quality") in ["high", "excellent"]:
                parsed["confidence"] = min(parsed.get("confidence", 0) + 0.2, 1.0)
            
            return parsed
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️  Enhanced analysis failed: {e}")
            return {
                "analysis": f"Analysis parsing failed. Result length: {len(result)} chars",
                "confidence": 0.4 if len(result) > 100 else 0.2,
                "quality": "fair" if len(result) > 50 else "poor",
                "next_actions": ["review_result_manually"],
                "covered_aspects": [],
                "is_concise": False
            }
    
    async def _generate_smart_followup_query(self, original_query: str, current_result: str, 
                                           next_actions: List[str], covered_aspects: Set[str], 
                                           iteration: int) -> str:
        """
        Generate smarter follow-up queries that avoid redundancy
        """
        covered_list = list(covered_aspects) if covered_aspects else []
        
        followup_prompt = f"""
        Generate an efficient follow-up query that avoids redundancy and focuses on the most important gap.

        ORIGINAL QUERY: {original_query}
        CURRENT RESULT: {current_result[:300]}...
        NEXT ACTIONS: {', '.join(next_actions)}
        ALREADY COVERED: {covered_list}
        ITERATION: {iteration}

        Generate a specific, non-redundant query that:
        1. Addresses the highest-priority missing piece
        2. Avoids repeating already covered aspects
        3. Is likely to provide actionable insights
        4. Builds logically on previous results

        Prioritize queries that will provide:
        - Specific data/metrics over general information
        - Actionable insights over descriptive data
        - Root causes over symptoms
        - Quantified results over qualitative descriptions

        Return only the optimized query text, nothing else.
        """
        
        try:
            followup_result = await Runner.run(self.base_agent, followup_prompt)
            return followup_result.final_output.strip().strip('"').strip("'")
        except Exception as e:
            print(f"⚠️  Error generating smart follow-up: {e}")
            return f"Provide more specific details about: {next_actions[0] if next_actions else 'the analysis'}"
    
    async def _compile_intelligent_final_result(self, original_query: str) -> str:
        """
        Compile results with intelligent synthesis and quality optimization
        """
        if not self.reasoning_history:
            return "No results generated."
        
        # Prioritize high-quality results
        quality_weighted_results = []
        for step in self.reasoning_history:
            if step.result and not step.result.startswith("Error:"):
                weight = self._get_quality_weight(step.result_quality)
                quality_weighted_results.append({
                    "step": step.step_number,
                    "result": step.result,
                    "weight": weight,
                    "confidence": step.confidence_score
                })
        
        if not quality_weighted_results:
            return "No successful results generated."
        
        # Sort by quality and confidence
        quality_weighted_results.sort(key=lambda x: (x["weight"], x["confidence"]), reverse=True)
        
        # Use top results for compilation
        top_results = quality_weighted_results[:3]  # Use best 3 results
        formatted_results = [f"Step {r['step']}: {r['result']}" for r in top_results]
        
        compile_prompt = f"""
        Synthesize these high-quality results into an exceptional final answer.

        ORIGINAL QUERY: {original_query}
        
        TOP QUALITY RESULTS:
        {chr(10).join(formatted_results)}

        COVERED ASPECTS: {list(self.covered_aspects)}

        Create a concise, well-structured final answer that:
        1. Directly and completely addresses the original query
        2. Synthesizes insights from multiple results
        3. Provides specific, actionable information
        4. Highlights key findings and recommendations
        5. Is professionally formatted and easy to understand

        Focus on quality over quantity. Provide depth and specificity.
        Do not mention the reasoning process - just deliver the final insights.
        """
        
        try:
            final_result = await Runner.run(self.base_agent, compile_prompt)
            return final_result.final_output
        except Exception as e:
            print(f"⚠️  Error compiling intelligent final result: {e}")
            # Return the highest quality single result
            best_result = max(quality_weighted_results, key=lambda x: (x["weight"], x["confidence"]))
            return best_result["result"]
    
    def _get_quality_weight(self, quality: str) -> float:
        """
        Convert quality assessment to numerical weight
        """
        quality_weights = {
            "excellent": 1.0,
            "high": 0.8,
            "good": 0.6,
            "fair": 0.4,
            "poor": 0.2,
            "unknown": 0.3
        }
        return quality_weights.get(quality.lower(), 0.3)


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
        cache_tools_list=True,
        name="snowflake-mcp-server",
        client_session_timeout_seconds=60
    )

    async with mcp_server as server:
        # List available tools
        tools = await server.list_tools()
        print("🔧 Available MCP tools:", [tool.name if hasattr(tool, 'name') else str(tool) for tool in tools])

        # Create base agent with enhanced instructions for o4-mini
        base_agent = Agent(
            name="snowflake-smart-reasoning-agent",
            model="o4-mini",
            instructions=(
                "You are an intelligent data analysis agent with advanced reasoning capabilities. "
                "You excel at breaking down complex queries, identifying data patterns, and providing actionable insights. "
                "If query is vague, make a logical assumption and continue with the query."
                "If the tool gives you a concrete answer, even if it's zero or null, use it and do not ask follow up questions."
                "When analyzing results, be thorough and precise in your assessments. "
                "Focus on quality over quantity - provide specific, data-driven insights. "
                "Always consider multiple perspectives and potential root causes in your analysis. "
                "Be efficient in your reasoning - avoid redundant queries and focus on high-value information gaps. "
                "Always pass the query as is to the tools, do not modify it."
                "When possible, use multiple tools in a single response to gather concise data efficiently."
            ),
            mcp_servers=[server]
        )

        # Create smart reasoning agent with optimized settings
        reasoning_agent = SmartReasoningAgent(
            base_agent=base_agent,
            max_iterations=5,
            confidence_threshold=0.75,  # Lowered slightly for efficiency
            early_stop_threshold=0.9,   # High threshold for early stopping
            quality_threshold=0.85
        )

        # Test queries with varying complexity
        test_queries = [
            "why are sales in germany outperforming other regions?",
        ]

        for query in test_queries:
            print(f"\n{'='*80}")
            print(f"🚀 TESTING QUERY: {query}")
            print(f"{'='*80}")
            
            try:
                result = await reasoning_agent.reason_and_execute(query)
                
                print(f"\n📋 EXECUTION SUMMARY:")
                print(f"   Original Query: {result['original_query']}")
                print(f"   Cortex Interpretation: {result['cortex_interpretation']}")
                print(f"   Used Cortex Direct Answer: {result['used_cortex_direct_answer']}")
                print(f"   Estimated vs Actual Iterations: {result['estimated_iterations']} → {result['total_iterations']}")
                print(f"   Completion Reason: {result['completion_reason']}")
                print(f"   Total Execution Time: {result['total_execution_time']:.2f}s")
                print(f"   Average Step Time: {result['average_step_time']:.2f}s")
                print(f"   Final Confidence: {result['final_confidence']:.2f}")
                print(f"   Covered Aspects: {', '.join(result['covered_aspects'])}")
                
                print(f"\n📄 FINAL RESULT:")
                print(f"{result['final_result']}")
                
                # Show reasoning efficiency
                if result['total_iterations'] > 1:
                    print(f"\n⚡ REASONING EFFICIENCY:")
                    for i, step in enumerate(result['reasoning_history']):
                        efficiency_indicator = "🚀" if step.confidence_score > 0.8 else "⚡" if step.confidence_score > 0.6 else "🔄"
                        print(f"   {efficiency_indicator} Step {i+1}: {step.state.value} | "
                              f"Confidence: {step.confidence_score:.2f} | "
                              f"Quality: {step.result_quality} | "
                              f"Time: {step.execution_time:.2f}s")
                
            except Exception as e:
                print(f"❌ Error processing query '{query}': {e}")
            
            print(f"\n{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main()) 