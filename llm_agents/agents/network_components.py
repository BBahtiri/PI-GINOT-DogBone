#!/usr/bin/env python3
"""
Shared state, custom outputs, agent base classes, and routing infrastructure.

This module provides the core building blocks for the PI-GINOT multi-agent system:
- Typed output payloads rendered by the Streamlit frontend
- Central State TypedDict flowing through the LangGraph
- CustomToolNode that captures rich outputs from tools
- BasicAgent: reusable subgraph with ReAct-style tool loops
- AgentRouter: structured routing with specialist selection
- Message reduction utilities for long conversations
"""

from typing import Annotated, List, Union, Any, Optional, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dataclasses import dataclass
from copy import deepcopy
import json, time, random

from langchain_core.prompts import (
    ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
)
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.store.base import BaseStore


class FieldPlotOutput(BaseModel):
    """Scatter/contour plot of a predicted field."""
    title: str
    points: List[dict]  # [{x, y, value}]
    field_name: str
    unit: str
    confidence_level: str


class TableOutput(BaseModel):
    """Generic data table."""
    name: str
    data: List[dict]


class GeometryPlotOutput(BaseModel):
    """DogBone geometry outline + collocation points."""
    params: dict
    boundary_points: List[dict]
    interior_points: List[dict]


class DiagnosticsOutput(BaseModel):
    """Reliability diagnostics card."""
    confidence_level: str
    metrics: dict
    rejection_reasons: List[str]
    response_behavior: dict


class OptimizationTraceOutput(BaseModel):
    """Geometry optimization convergence plot."""
    iterations: List[dict]
    objective: str
    best_geometry: dict


class ReportOutput(BaseModel):
    """Final markdown report."""
    title: str
    markdown: str
    sections: List[dict]


def custom_outputs_reducer(existing: list | None, new: list) -> list:
    """Accumulate custom outputs unless explicitly reset (empty list)."""
    if len(new) == 0 and existing and len(existing) > 0:
        return []  # explicit reset
    return (existing or []) + new


class State(TypedDict):
    """Central state flowing through the LangGraph."""
    messages: Annotated[list, add_messages]
    summary: str
    current_agent: str
    custom_outputs: Annotated[
        List[Union[FieldPlotOutput, TableOutput, GeometryPlotOutput,
                   DiagnosticsOutput, OptimizationTraceOutput, ReportOutput]],
        custom_outputs_reducer,
    ]
    # PI-GINOT specific
    geometry_params: Optional[dict]
    last_prediction: Optional[dict]
    confidence_level: Optional[str]
    optimization_history: List[dict]
    cache_hit: bool


class CustomToolNode:
    """Tool executor that captures (message, [custom_outputs]) returns from tools."""

    def __init__(self, tools: list):
        self.tools_by_name = {t.name: t for t in tools}

    def __call__(self, inputs: dict):
        messages = inputs.get("messages", [])
        if not messages:
            raise ValueError("No messages in input")
        last = messages[-1]

        new_messages = []
        custom_outputs = []

        for tool_call in last.tool_calls:
            result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            # Tool may return (message_str, [custom_outputs]) or just a str
            if isinstance(result, tuple):
                msg_content, outputs = result
            else:
                msg_content, outputs = result, None

            new_messages.append(ToolMessage(
                content=str(msg_content),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            ))
            if outputs:
                custom_outputs.extend(outputs)

        if custom_outputs:
            return {"messages": new_messages, "custom_outputs": custom_outputs}
        return {"messages": new_messages}


def get_messages_use_flag(messages):
    """Keep messages marked for inclusion; drop trimmed/stale ones."""
    first_idx = 0
    for i, m in enumerate(messages):
        if m.additional_kwargs.get("use_message") is True:
            first_idx = i
            break
    trimmed = messages[first_idx:]
    return [m for m in trimmed
            if m.additional_kwargs.get("use_message", True) is not False], first_idx


def retry_with_exponential_backoff(func, initial_delay=5, base=2, max_retries=5):
    """Retry function on rate limit errors with exponential backoff."""
    delay = initial_delay
    for i in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                delay *= base
                time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"Exceeded {max_retries} retries")


def summarize_custom_outputs(outputs: list) -> str:
    """Build a text summary of custom outputs for agents that need data context.

    Used by the Reporter to access prior prediction/optimization results
    that live in custom_outputs (not visible in message text).
    """
    if not outputs:
        return ""
    lines = []
    for o in outputs:
        if isinstance(o, DiagnosticsOutput):
            lines.append(
                f"- Diagnostics: confidence={o.confidence_level}, "
                f"metrics={list(o.metrics.keys())[:6]}, "
                f"rejections={o.rejection_reasons}"
            )
        elif isinstance(o, OptimizationTraceOutput):
            lines.append(
                f"- Optimization: objective={o.objective}, "
                f"best_geometry={o.best_geometry}, "
                f"n_iterations={len(o.iterations)}"
            )
        elif isinstance(o, FieldPlotOutput):
            lines.append(
                f"- Field plot: {o.field_name} [{o.unit}], "
                f"confidence={o.confidence_level}, "
                f"n_points={len(o.points)}"
            )
        elif isinstance(o, TableOutput):
            lines.append(
                f"- Table '{o.name}': {len(o.data)} rows"
            )
        elif isinstance(o, GeometryPlotOutput):
            lines.append(
                f"- Geometry: params={o.params}"
            )
    return "\n".join(lines)


@dataclass
class SystemPrompt:
    """Structured system prompt with role, task, format, examples, context."""
    role: str
    task: Optional[str] = None
    output_format: Optional[str] = None
    examples: Optional[str] = None
    context: Optional[str] = None

    def build(self) -> str:
        parts = []
        if self.role:
            parts.append(f"## Role\n{self.role}")
        if self.task:
            parts.append(f"## Task\n{self.task}")
        if self.output_format:
            parts.append(f"## Output Format\n{self.output_format}")
        if self.examples:
            parts.append(f"## Examples\n{self.examples}")
        if self.context:
            parts.append(f"## Context\n{self.context}")
        return "\n\n".join(parts)

    def description(self) -> str:
        """Short description for router agent listings."""
        parts = []
        if self.role:
            parts.append(f"## Role\n{self.role}")
        if self.task:
            parts.append(f"## Task\n{self.task[:500]}")
        return "\n".join(parts)


class BasicAgent:
    """A specialist agent implemented as a LangGraph subgraph with ReAct tool loop.

    Each BasicAgent:
    - Has a system prompt defining its expertise
    - Can call tools in a loop until it produces a final answer
    - Tags messages with sender metadata
    - Optionally persists to long-term memory
    - Injects custom_outputs summary for agents that need prior data context
    """

    def __init__(self, agent_name, system_prompt, llm, tools,
                 avatar_path=None, return_all_messages=False, ltm=None):
        self.name = agent_name
        self.system_prompt = (system_prompt if isinstance(system_prompt, SystemPrompt)
                              else SystemPrompt(role=system_prompt))
        self.llm = llm
        self.ltm = ltm
        self.avatar_path = avatar_path
        self.return_all_messages = return_all_messages
        self.llm_with_tools = llm.bind_tools(tools)

        # Build subgraph: agent -> tools -> agent -> ... -> end
        tool_node = CustomToolNode(tools)
        g = StateGraph(State)
        g.add_node(self.name, self._invoke_llm)
        g.add_node("tools", tool_node)
        g.add_edge(START, self.name)
        g.add_edge("tools", self.name)

        if self.ltm is not None:
            g.add_node("update_memories", self._update_memories)
            g.add_edge("update_memories", END)
            g.add_conditional_edges(
                self.name,
                self._tools_condition,
                {"tools": "tools", "__end__": "update_memories"},
            )
        else:
            g.add_conditional_edges(
                self.name,
                self._tools_condition,
                {"tools": "tools", "__end__": END},
            )

        self.agent_graph = g.compile(name=self.name)

    def _invoke_llm(self, state: State, store: BaseStore = None):
        """Call the LLM with system prompt + message history."""
        system_msg = SystemMessage(self.system_prompt.build())
        prompt = ChatPromptTemplate([system_msg, MessagesPlaceholder("messages")])
        chain = prompt | self.llm_with_tools

        messages = list(state["messages"])
        if state.get("summary"):
            messages = [HumanMessage(
                content=f"[Earlier summary]: {state['summary']}"
            )] + messages

        outputs = state.get("custom_outputs", [])
        if outputs and self.name in ("Reporter",):
            output_summary = summarize_custom_outputs(outputs)
            if output_summary:
                messages = [HumanMessage(
                    content=f"[Prior analysis data available in this session]:\n{output_summary}"
                )] + messages

        # Merge consecutive AI messages (avoids protocol issues)
        merged = []
        for m in messages:
            if (merged and isinstance(m, AIMessage) and isinstance(merged[-1], AIMessage)
                    and not m.tool_calls
                    and not getattr(merged[-1], 'tool_calls', None)):
                merged[-1] = AIMessage(
                    content=merged[-1].content + "\n" + m.content,
                    additional_kwargs={
                        **merged[-1].additional_kwargs, **m.additional_kwargs
                    },
                )
            else:
                merged.append(m)

        if merged and isinstance(merged[-1], AIMessage):
            merged.append(HumanMessage(content="Continue."))

        output = retry_with_exponential_backoff(
            lambda: chain.invoke({"messages": merged})
        )
        return {"messages": [output]}

    def _tools_condition(self, state: State):
        """Route to tools if the last message has tool_calls, else end."""
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "__end__"

    def _update_memories(self, state: State, store: BaseStore = None):
        """Hook for persisting prompt/response to long-term memory."""
        if self.ltm is None:
            return {}
        user_prompt = next(
            (m.content for m in reversed(state["messages"])
             if isinstance(m, HumanMessage)), None
        )
        ai_response = next(
            (m.content for m in reversed(state["messages"])
             if isinstance(m, AIMessage) and m.content), None
        )
        if user_prompt:
            try:
                self.ltm.save_prompt_response(
                    user_id=self.ltm.user_id,
                    prompt=user_prompt,
                    response=ai_response,
                    agent_name=self.name,
                )
            except Exception as e:
                print(f"[LTM] save failed: {e}")
        return {}

    def invoke(self, state: State):
        """Run the agent subgraph and return tagged results."""
        sub_state = deepcopy(state)
        reduced, _ = get_messages_use_flag(sub_state["messages"])
        sub_state["messages"] = reduced

        result = self.agent_graph.invoke(sub_state)

        for m in result["messages"]:
            if isinstance(m, (AIMessage, ToolMessage)):
                if "sender" not in (m.additional_kwargs or {}):
                    m.additional_kwargs = (
                        (m.additional_kwargs or {}) | {"sender": self.name}
                    )

        result["messages"][-1].additional_kwargs["show"] = True

        if self.return_all_messages:
            return result
        return {
            "messages": [result["messages"][-1]],
            "custom_outputs": result.get("custom_outputs", []),
        }


class RouterOutput(BaseModel):
    """Structured output schema for routing decisions."""
    agent: Optional[str] = Field(default=None)
    message: Optional[str] = Field(default=None)
    agent_intro: bool = Field(default=False)


class AgentRouter:
    """Routes user messages to the appropriate specialist agent.

    Uses structured output to decide which agent handles the query.
    Can also answer directly for conversational / routing queries.
    """

    def __init__(self, llm, agents: List[BasicAgent]):
        self.llm = llm
        self.agents = agents
        self.avatar_path = None

        desc = "\n".join([
            f"- **{a.name}**: {a.system_prompt.description()}"
            for a in agents
        ])

        system = f"""You are a router for the PI-GINOT multi-agent system — a \
physics-informed neural operator for DogBone tensile specimens.

# Available specialists
{desc}

# Current agent: {{current_agent}}

# Rules
1. If current_agent is set, stay with it unless user asks to switch.
2. If query is conversational / intro / help, answer directly.
3. For DogBone analysis → Predictor agent
4. For geometry optimization → Optimizer agent
5. For reliability concerns / refinement → Diagnostician agent
6. For generating a written report → Reporter agent
7. If unclear, ask for clarification.

Respond with JSON only."""

        self.system_template = SystemMessagePromptTemplate.from_template(system)
        g = StateGraph(State)
        g.add_node("router", self._route)
        g.add_edge(START, "router")
        g.add_edge("router", END)
        self.graph = g.compile(name="Agent Router")

    def invoke(self, state: State, store: BaseStore = None):
        return self.graph.invoke(state, config={"store": store})

    def _route(self, state: State, store: BaseStore = None):
        current = state.get("current_agent", "")
        if current in ("__end__", "None"):
            current = ""

        sys_msg = self.system_template.format_messages(current_agent=current)[0]
        prompt = ChatPromptTemplate([sys_msg, MessagesPlaceholder("messages")])
        router = prompt | self.llm.with_structured_output(RouterOutput)

        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if last_human is None:
            return

        resp = router.invoke({"messages": [last_human]}, config={"callbacks": []})

        if resp.agent and resp.agent.lower() == current.lower():
            return

        if resp.agent:
            for a in self.agents:
                if a.name.lower() == resp.agent.lower():
                    return {
                        "messages": [AIMessage(
                            content=f"Directing to {a.name}...",
                            additional_kwargs={
                                "sender": "Agent Router",
                                "show": True,
                                "use_message": False,
                            },
                        )],
                        "current_agent": a.name,
                    }

        return {
            "messages": [AIMessage(
                content=resp.message or "How can I help?",
                additional_kwargs={
                    "sender": "Agent Router",
                    "show": True,
                },
            )],
            "current_agent": "__end__",
        }

    def get_intro(self) -> str:
        return """👋 Welcome to **PI-GINOT Agentic Studio**.

I coordinate specialists for physics-informed analysis of DogBone tensile specimens:

🔬 **Predictor** — Run reliability-aware predictions on any DogBone geometry
🎯 **Optimizer** — Find geometries that minimize peak stress or maximize correction range
🩺 **Diagnostician** — Diagnose low-confidence predictions and refine the model
📄 **Reporter** — Generate full narrative reports with physics interpretation

Try: *"Analyze a standard dogbone"* or *"Optimize for minimum peak sigma11"*"""


def reduce_messages(state: State, summarization_model, summarize=True,
                    max_tokens=10_000):
    """Trim old messages and optionally summarize them.

    Flags messages with `use_message` for downstream filtering.
    """
    use_msgs, idx = get_messages_use_flag(state["messages"])
    reduced = trim_messages(
        use_msgs,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
        start_on="human",
    )
    removed = [m for m in use_msgs if m not in reduced]

    for i in range(idx, len(state["messages"])):
        m = state["messages"][i]
        use = m in reduced
        m.additional_kwargs = {
            **(m.additional_kwargs or {}), "use_message": use
        }

    if summarize and removed:
        old = state.get("summary", "")
        prompt = (
            f"Extend this summary with the new messages: {old}"
            if old else "Summarize the conversation."
        )
        summary_msg = HumanMessage(content=prompt)
        resp = summarization_model.invoke(
            removed + [summary_msg], config={"callbacks": []}
        )
        state["summary"] = f"Summary: {resp.content}"

    return state
