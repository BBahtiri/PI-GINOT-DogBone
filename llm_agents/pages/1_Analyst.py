#!/usr/bin/env python3
"""
PI-GINOT Agentic Studio — Main Analyst Chat UI.

Provides:
- Multi-agent chat with routing between Predictor, Optimizer, Diagnostician, Reporter
- Rich output rendering (field plots, geometry outlines, diagnostics cards, tables)
- LLM model switching (GPT-4o / Claude Sonnet / Databricks Endpoint)
- Semantic cache with embedding-based deduplication
- Long-term memory for user preferences and geometry library
- Short-term memory for session persistence
- Debug panels (graph visualization, state inspection, cache/memory stats)
- Conversation management with thread IDs
"""

import streamlit as st
import sys
import os
import uuid
from functools import partial

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except Exception:
    pass

# Add agents to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "agents"))
sys.path.insert(0, PROJECT_ROOT)

from agents.network_components import (
    State, AgentRouter, reduce_messages,
    FieldPlotOutput, TableOutput, GeometryPlotOutput,
    DiagnosticsOutput, OptimizationTraceOutput, ReportOutput,
)
from agents import tools
from agents import agent_predictor, agent_optimizer, agent_diagnostician, agent_reporter
from agents.callbackhandler import get_streamlit_cb
from agents.semantic_cache import SemanticCache
from agents.cache_nodes import CacheCheckNode, CacheStoreNode
from agents.long_term_memory import LongTermMemory
from agents.short_term_memory import ShortTermMemory

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

try:
    from databricks_langchain import ChatDatabricks
    _HAS_DATABRICKS = True
except ImportError:
    _HAS_DATABRICKS = False


if "pi_agent_initialized" not in st.session_state:
    # Graceful model loading — app still starts if checkpoint is missing
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "checkpoints", "best.pt"
    )
    try:
        with st.spinner("Loading PI-GINOT model..."):
            tools.init_pi_agent(checkpoint_path, device="auto")
        st.session_state.pi_agent_initialized = True
        st.session_state.pi_agent_error = None
    except Exception as e:
        st.session_state.pi_agent_initialized = True
        st.session_state.pi_agent_error = str(e)
        st.warning(f"⚠️ PI-GINOT model not loaded: {e}\n\nThe chat will work but predictions are disabled.")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "stm" not in st.session_state:
    st.session_state.stm = ShortTermMemory()
    st.session_state.ltm = LongTermMemory(user_id="default")
    # Semantic cache — try OpenAI embeddings, fall back to None (exact-hash only)
    try:
        from langchain_openai import OpenAIEmbeddings
        embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
    except Exception:
        embed_model = None
    st.session_state.cache = SemanticCache(
        embedding_model=embed_model,
        similarity_threshold=0.92,
        ttl_seconds=3600 * 4,  # 4 hours
    )
    tools.set_ltm(st.session_state.ltm)

# Track session
if st.session_state.stm.get_session(st.session_state.thread_id) is None:
    st.session_state.stm.create_session(st.session_state.thread_id)
st.session_state.stm.update_activity(st.session_state.thread_id)


with st.sidebar:
    st.subheader("Model")
    llm_options = []
    llm_captions = []
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_ADMIN_KEY"):
        llm_options.append("GPT-4o")
        llm_captions.append("Fast, cheap")
    if os.environ.get("ANTHROPIC_API_KEY"):
        llm_options.append("Claude Sonnet")
        llm_captions.append("Stronger physics reasoning")
    if _HAS_DATABRICKS:
        llm_options.append("Databricks Endpoint")
        llm_captions.append("Via workspace model serving endpoint")
    if llm_options:
        llm_choice = st.radio("LLM", llm_options, captions=llm_captions)
    else:
        llm_choice = None
        st.warning(
            "No LLM provider configured. Add OPENAI_API_KEY or "
            "ANTHROPIC_API_KEY to .env to enable the Analyst chat."
        )

    st.subheader("Debug")
    show_graph = st.checkbox("Show agent graph")
    show_state = st.checkbox("Show state")

    with st.expander("📊 Cache & Memory"):
        cache_stats = st.session_state.cache.stats()
        col1, col2 = st.columns(2)
        col1.metric("Cache entries", cache_stats["entries"])
        col2.metric("Cache hits", cache_stats["total_hits"])

        prefs = st.session_state.ltm.get_all_preferences()
        st.caption(f"User preferences: {len(prefs)}")

        agent_stats = st.session_state.ltm.get_agent_stats()
        if agent_stats:
            st.caption("Agent usage:")
            for name, count in sorted(agent_stats.items(),
                                       key=lambda x: x[1], reverse=True):
                st.caption(f"  {name}: {count}")

        geos = st.session_state.ltm.list_geometries()
        if geos:
            st.caption(f"Saved geometries: {', '.join(geos)}")

    if st.session_state.get("pi_agent_error"):
        st.error(f"Model: {st.session_state.pi_agent_error}")

    if st.button("🗑️ New Conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        if "graph" in st.session_state:
            del st.session_state.graph
        st.rerun()


if llm_choice is None:
    st.title("PI-GINOT Agentic Studio")
    st.info(
        "The app is running. To enable the Analyst chat, create a .env file "
        "with OPENAI_API_KEY or ANTHROPIC_API_KEY and restart Streamlit."
    )
    st.stop()


if "graph" not in st.session_state or st.session_state.get("last_llm") != llm_choice:
    if llm_choice == "GPT-4o":
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        summarizer = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1000)
    elif llm_choice == "Databricks Endpoint":
        endpoint_name = os.environ.get("DATABRICKS_ENDPOINT_NAME", "databricks-meta-llama-3-3-70b-instruct")
        llm = ChatDatabricks(endpoint=endpoint_name)
        summarizer = ChatDatabricks(endpoint=endpoint_name, max_tokens=1000)
    else:
        llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0.1)
        summarizer = ChatAnthropic(model="claude-haiku-4-5", temperature=0, max_tokens=1000)

    ltm = st.session_state.ltm
    agents = [
        agent_predictor.get_agent(llm, ltm=ltm),
        agent_optimizer.get_agent(llm, ltm=ltm),
        agent_diagnostician.get_agent(llm, ltm=ltm),
        agent_reporter.get_agent(llm, ltm=ltm),
    ]
    router = AgentRouter(llm, agents)

    cache_check = CacheCheckNode(st.session_state.cache)
    cache_store = CacheStoreNode(st.session_state.cache)

    g = StateGraph(State)
    g.add_node("ReduceMessages", partial(
        reduce_messages, summarization_model=summarizer,
        summarize=True, max_tokens=20_000
    ))
    g.add_node("CacheCheck", cache_check)
    g.add_node("Router", router.invoke)
    g.add_node("CacheStore", cache_store)
    for a in agents:
        g.add_node(a.name, a.invoke)

    # Flow: START → ReduceMessages → CacheCheck → (hit? END : Router → Agent → CacheStore → END)
    g.add_edge(START, "ReduceMessages")
    g.add_edge("ReduceMessages", "CacheCheck")

    def route_after_cache(state):
        """Cache hit → skip everything; cache miss → normal routing."""
        if state.get("cache_hit"):
            return "__end__"
        return "Router"

    g.add_conditional_edges(
        "CacheCheck",
        route_after_cache,
        {"Router": "Router", "__end__": END},
    )
    g.add_conditional_edges(
        "Router",
        lambda s: s["current_agent"],
        {a.name: a.name for a in agents} | {"__end__": END},
    )
    for a in agents:
        g.add_edge(a.name, "CacheStore")
    g.add_edge("CacheStore", END)

    st.session_state.graph = g.compile(checkpointer=InMemorySaver())
    st.session_state.router = router
    st.session_state.agents = agents
    st.session_state.last_llm = llm_choice

config = {"configurable": {"thread_id": st.session_state.thread_id}}


st.title("🧬 PI-GINOT Agentic Studio")
left, right = st.columns([2, 1] if (show_graph or show_state) else [1, 0.001])

with left:
    chat_container = st.container()
    placeholder = st.empty()
    user_input = st.chat_input("Ask about a DogBone specimen...")

    # Render history
    state_snapshot = st.session_state.graph.get_state(config)
    state_values = state_snapshot[0] if state_snapshot else {}

    if state_values.get("messages"):
        for msg in state_values["messages"]:
            if msg.type == "human":
                chat_container.chat_message("user").write(msg.content)
            elif (msg.type == "ai" and msg.content
                  and msg.additional_kwargs.get("show", False)):
                sender = msg.additional_kwargs.get("sender", "assistant")
                cached = msg.additional_kwargs.get("cached", False)
                with chat_container.chat_message("assistant"):
                    label = f"↳ {sender}"
                    if cached:
                        label += " (cached)"
                    st.caption(label)
                    st.markdown(msg.content)
    else:
        chat_container.chat_message("assistant").markdown(
            st.session_state.router.get_intro()
        )

    # Handle input
    if user_input:
        chat_container.chat_message("user").write(user_input)
        with placeholder.chat_message("assistant"):
            st_cb = get_streamlit_cb(st.container())

            stream_placeholder = st.empty()
            accumulated = ""
            current_sender = None

            with st.spinner("Thinking..."):
                for msg_chunk, metadata in st.session_state.graph.stream(
                    {"messages": [{"role": "user", "content": user_input}],
                     "custom_outputs": []},
                    config={**config, "callbacks": [st_cb]},
                    stream_mode="messages",
                ):
                    sender = msg_chunk.additional_kwargs.get("sender", "")

                    # Show router transitions
                    if sender == "Agent Router" and msg_chunk.content:
                        st.caption(f"↳ Router: {msg_chunk.content}")
                        continue

                    # Stream agent content progressively
                    if isinstance(msg_chunk, (AIMessage, AIMessageChunk)):
                        if msg_chunk.content and not msg_chunk.tool_calls:
                            if current_sender != sender:
                                accumulated = ""
                                current_sender = sender
                                if sender:
                                    st.caption(f"↳ {sender}")
                            accumulated += msg_chunk.content
                            stream_placeholder.markdown(accumulated)

            # Final render (overwrite stream placeholder with clean version)
            final_state = st.session_state.graph.get_state(config)[0]
            last = final_state["messages"][-1]
            if last.content and last.additional_kwargs.get("sender") != "Agent Router":
                stream_placeholder.markdown(last.content)

    # Render custom outputs
    if state_values.get("custom_outputs"):
        st.divider()
        for out in state_values["custom_outputs"]:
            if isinstance(out, FieldPlotOutput):
                import pandas as pd
                import plotly.express as px
                df = pd.DataFrame(out.points)
                fig = px.scatter(df, x="x", y="y", color="value",
                                 title=out.title,
                                 color_continuous_scale="Viridis",
                                 labels={"value": f"{out.field_name} [{out.unit}]"})
                fig.update_layout(yaxis_scaleanchor="x")
                st.plotly_chart(fig, use_container_width=True)

            elif isinstance(out, GeometryPlotOutput):
                import pandas as pd
                import plotly.graph_objects as go
                fig = go.Figure()
                bnd = pd.DataFrame(out.boundary_points)
                intr = pd.DataFrame(out.interior_points)
                fig.add_trace(go.Scatter(
                    x=bnd.x, y=bnd.y, mode="markers",
                    marker=dict(color="black", size=3), name="boundary"
                ))
                fig.add_trace(go.Scatter(
                    x=intr.x, y=intr.y, mode="markers",
                    marker=dict(color="lightblue", size=2), name="interior"
                ))
                fig.update_layout(
                    title=f"DogBone: {out.params}",
                    yaxis_scaleanchor="x", height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            elif isinstance(out, DiagnosticsOutput):
                level_colors = {
                    "high": "🟢", "medium": "🟡", "low": "🟠", "reject": "🔴"
                }
                with st.expander(
                    f"{level_colors.get(out.confidence_level, '⚪')} "
                    f"Reliability: {out.confidence_level.upper()}",
                    expanded=True,
                ):
                    if out.rejection_reasons:
                        st.error("**Rejection reasons:**\n" +
                                 "\n".join(f"- {r}" for r in out.rejection_reasons))
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Eq residual",
                              f"{out.metrics.get('normalized_equilibrium_residual', 0):.2e}")
                    c2.metric("Section CV",
                              f"{100 * out.metrics.get('section_force_cv', 0):.1f}%")
                    c3.metric("Swap sens.",
                              f"{out.metrics.get('latent_swap_sensitivity', 0):.3f}")
                    if st.toggle("Show all metrics", key=f"metrics_{id(out)}"):
                        st.json(out.metrics)

            elif isinstance(out, TableOutput):
                import pandas as pd
                st.caption(out.name)
                st.dataframe(pd.DataFrame(out.data),
                             use_container_width=True, hide_index=True)

            elif isinstance(out, OptimizationTraceOutput):
                import pandas as pd
                import plotly.express as px
                df = pd.DataFrame(out.iterations)
                if not df.empty:
                    fig = px.scatter(
                        df, x="R_fillet", y="objective", color="confidence",
                        title=f"Optimization trace — {out.objective}",
                        hover_data=["L_total", "W_grip", "W_gauge"]
                    )
                    st.plotly_chart(fig, use_container_width=True)

            elif isinstance(out, ReportOutput):
                st.markdown(out.markdown)
                st.download_button(
                    "📥 Download report",
                    data=out.markdown,
                    file_name=f"{out.title.replace(' ', '_')}.md",
                    mime="text/markdown",
                )

with right:
    if show_graph:
        st.subheader("Agent Graph")
        try:
            st.image(st.session_state.graph.get_graph().draw_mermaid_png(),
                     caption="Full graph")
        except Exception:
            st.info("Graph visualization requires pygraphviz or mermaid.")

    if show_state:
        st.subheader("State")
        st.json({
            k: str(v)[:500]
            for k, v in state_values.items()
            if k != "messages"
        })
