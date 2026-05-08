#!/usr/bin/env python3
"""
PI-GINOT Agentic Studio — Multi-Agent System.

Refactored architecture with:
- AgentRouter with structured routing
- BasicAgent subgraphs with ReAct tool loops
- Semantic cache with embedding-based similarity
- Long-term memory with user preferences
- Short-term memory with conversation persistence
- Custom tool nodes with rich output types
- Streamlit UI with chat history and debug panels

Quick start:
    from llm_agents.agents.tools import init_pi_agent
    from llm_agents.agents.network_components import AgentRouter, BasicAgent, State
    from llm_agents.agents import agent_predictor, agent_optimizer

    init_pi_agent("checkpoints/best.pt")
"""

from .agents.tools import init_pi_agent, get_pi_agent
from .agents.network_components import (
    State, BasicAgent, AgentRouter, CustomToolNode,
    FieldPlotOutput, TableOutput, GeometryPlotOutput,
    DiagnosticsOutput, OptimizationTraceOutput, ReportOutput,
)

__all__ = [
    "init_pi_agent", "get_pi_agent",
    "State", "BasicAgent", "AgentRouter", "CustomToolNode",
    "FieldPlotOutput", "TableOutput", "GeometryPlotOutput",
    "DiagnosticsOutput", "OptimizationTraceOutput", "ReportOutput",
]
