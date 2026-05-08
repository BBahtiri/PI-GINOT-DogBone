#!/usr/bin/env python3
"""
PI-GINOT Multi-Agent System — Entry Point & Architecture Reference.

This file provides a programmatic entry point for the LangGraph-based
multi-agent architecture that orchestrates the PI-GINOT physics stack.

Architecture:
    USER (natural language)
         │
    SUPERVISOR AGENT (plans & routes)
         │
    ┌────┼────┬────────────┬──────────┬─────────┬──────────┐
    │    │    │            │          │         │          │
  GEOM  PHYS  RELIABILITY  OPTIMIZER  REFINER   REPORTER
  PARSER ANALYST  JUDGE      AGENT    AGENT     AGENT
    │    │    │            │          │         │
    └────┴────┴────────────┴──────────┴─────────┘
         │
    TOOL LAYER (PI-GINOT Agent, InferenceEngine, Refinement)

Installation:
    pip install langgraph langchain-openai langchain-anthropic streamlit pydantic

    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...

Usage:
    # Interactive (Streamlit)
    streamlit run llm_agents/app.py

    # Programmatic
    python write_out.py
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage

from llm_agents.tools import init_pi_agent
from llm_agents.graph import build_graph


def run_query(query: str, checkpoint_path: str = "checkpoints/best.pt",
              device: str = "auto", max_iterations: int = 12) -> dict:
    """Run a single natural-language query through the multi-agent system.

    Args:
        query: Natural language request (e.g., "Analyze a standard dogbone").
        checkpoint_path: Path to trained PI-GINOT checkpoint.
        device: 'cpu', 'cuda', or 'auto'.
        max_iterations: Maximum routing iterations before forced termination.

    Returns:
        Final agent state dict with keys including:
            final_report, prediction_result, analysis_notes, etc.
    """
    # Initialize PI-GINOT model (idempotent — skips if already loaded)
    init_pi_agent(checkpoint_path, device=device)

    # Build the LangGraph
    graph = build_graph()

    # Construct initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "analysis_notes": [],
        "optimization_history": [],
        "rejection_reasons": [],
        "plots_generated": [],
        "errors": [],
        "iterations": 0,
        "max_iterations": max_iterations,
        "geometry_valid": False,
        "optimization_complete": False,
        "refinement_approved": False,
    }

    # Execute
    config = {"configurable": {"thread_id": "cli-session"}}
    final_state = graph.invoke(initial_state, config=config)

    return final_state


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PI-GINOT Multi-Agent CLI")
    parser.add_argument("query", nargs="?", default="Analyze a standard dogbone specimen",
                        help="Natural language query for the agent system")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()

    print(f"Query: {args.query}\n{'='*60}")
    result = run_query(args.query, checkpoint_path=args.checkpoint, device=args.device)

    print("\n" + "="*60)
    print("FINAL REPORT:")
    print("="*60)
    print(result.get("final_report", "No report generated."))

    print("\n" + "-"*60)
    print("AGENT TRACE:")
    print("-"*60)
    for note in result.get("analysis_notes", []):
        print(f"  {note}")
