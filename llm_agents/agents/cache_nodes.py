#!/usr/bin/env python3
"""
Cache nodes — LangGraph nodes for cache check and cache store operations.

These nodes integrate the SemanticCache into the LangGraph workflow:
- CacheCheckNode: runs before routing, short-circuits if a cache hit is found
- CacheStoreNode: runs after agent execution, stores results for future queries

FIX #6: Rehydrates cached custom_outputs back to Pydantic objects on lookup.
FIX #9: Bypasses cache for agents with side effects (Optimizer, Diagnostician).
"""

from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage

from .network_components import (
    State,
    FieldPlotOutput,
    TableOutput,
    GeometryPlotOutput,
    DiagnosticsOutput,
    OptimizationTraceOutput,
    ReportOutput,
)
from .semantic_cache import SemanticCache


# Agents whose results should NOT be cached (have side effects or stateful logic)
SKIP_CACHE_AGENTS = {"Optimizer", "Diagnostician"}

# Keywords that signal the user wants fresh results even if cached
BYPASS_KEYWORDS = {"refresh", "rerun", "new", "fresh", "again", "redo"}

# Type registry for rehydrating serialized custom outputs
_OUTPUT_TYPE_MAP = {
    "FieldPlotOutput": FieldPlotOutput,
    "TableOutput": TableOutput,
    "GeometryPlotOutput": GeometryPlotOutput,
    "DiagnosticsOutput": DiagnosticsOutput,
    "OptimizationTraceOutput": OptimizationTraceOutput,
    "ReportOutput": ReportOutput,
}


def _rehydrate_custom_outputs(serialized: list) -> list:
    """Reconstruct Pydantic objects from serialized cache entries.

    FIX #6: The SemanticCache stores custom_outputs as
    [{"__type__": "FieldPlotOutput", "data": {...}}, ...]
    This function converts them back to typed Pydantic objects.
    """
    outputs = []
    for entry in serialized:
        if not isinstance(entry, dict):
            outputs.append(entry)
            continue
        type_name = entry.get("__type__")
        data = entry.get("data")
        cls = _OUTPUT_TYPE_MAP.get(type_name)
        if cls and isinstance(data, dict):
            try:
                outputs.append(cls(**data))
            except Exception:
                # If rehydration fails, skip this output silently
                pass
        elif type_name == "raw":
            # Raw string outputs are not rehydratable — skip
            pass
        else:
            # Already a Pydantic object (shouldn't happen in normal flow)
            outputs.append(entry)
    return outputs


class CacheCheckNode:
    """LangGraph node that checks the semantic cache before routing.

    If a cache hit is found, injects the cached response and sets
    cache_hit=True so downstream routing can short-circuit.

    FIX #9: Skips cache for agents with side effects and when
    user explicitly requests fresh results.
    """

    def __init__(self, cache: SemanticCache):
        self.cache = cache

    def __call__(self, state: State) -> dict:
        # FIX #9: Skip cache for agents that have side effects
        current_agent = state.get("current_agent", "")
        if current_agent in SKIP_CACHE_AGENTS:
            return {"cache_hit": False}

        # Extract the last human message
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if last_human is None:
            return {"cache_hit": False}

        # FIX #9: Bypass if user explicitly asks for fresh data
        query_lower = last_human.content.lower()
        if any(kw in query_lower for kw in BYPASS_KEYWORDS):
            return {"cache_hit": False}

        # Check cache
        geometry_params = state.get("geometry_params")
        hit, result = self.cache.lookup(last_human.content, geometry_params)

        if hit and result:
            # FIX #6: Rehydrate custom outputs from serialized form
            raw_outputs = result.get("custom_outputs", [])
            custom_outputs = _rehydrate_custom_outputs(raw_outputs)

            cached_msg = AIMessage(
                content=f"{result.get('message', '')}",
                additional_kwargs={
                    "sender": "Cache",
                    "show": True,
                    "cached": True,
                },
            )
            return {
                "messages": [cached_msg],
                "cache_hit": True,
                "custom_outputs": custom_outputs,
            }

        return {"cache_hit": False}


class CacheStoreNode:
    """LangGraph node that stores agent results in the semantic cache.

    FIX #9: Does not cache results from side-effect agents.
    """

    def __init__(self, cache: SemanticCache):
        self.cache = cache

    def __call__(self, state: State) -> dict:
        # Don't re-cache cached results
        if state.get("cache_hit"):
            return {}

        # FIX #9: Don't cache side-effect agents
        current_agent = state.get("current_agent", "")
        if current_agent in SKIP_CACHE_AGENTS:
            return {}

        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        last_ai = next(
            (m for m in reversed(state["messages"])
             if isinstance(m, AIMessage) and m.content),
            None,
        )

        if last_human and last_ai:
            self.cache.store(
                query=last_human.content,
                result={
                    "message": last_ai.content,
                    "custom_outputs": state.get("custom_outputs", []),
                },
                agent_name=current_agent,
                geometry_params=state.get("geometry_params"),
            )

        return {}
