#!/usr/bin/env python3
"""
Streamlit-compatible callback handler for LangChain/LangGraph.

Wraps all on_* methods with Streamlit's script run context to ensure
thread-safe UI updates during async agent execution.
"""

from typing import Callable, TypeVar, Any
import inspect

from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.delta_generator import DeltaGenerator
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler


T = TypeVar("T")


def get_streamlit_cb(parent_container: DeltaGenerator) -> StreamlitCallbackHandler:
    """Create a Streamlit callback handler with proper thread context.

    Wraps all `on_*` methods to ensure they run within Streamlit's
    script context, preventing threading errors during LangGraph execution.
    """
    def wrap(fn: Callable[..., T]) -> Callable[..., T]:
        ctx = get_script_run_ctx()

        def wrapper(*args, **kwargs):
            add_script_run_ctx(ctx=ctx)
            return fn(*args, **kwargs)

        return wrapper

    cb = StreamlitCallbackHandler(parent_container)
    for name, method in inspect.getmembers(cb, predicate=inspect.ismethod):
        if name.startswith("on_"):
            setattr(cb, name, wrap(method))
    return cb
