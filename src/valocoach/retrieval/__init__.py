from __future__ import annotations

from valocoach.retrieval.agents import format_agent_context, get_agent, list_agent_names
from valocoach.retrieval.maps import format_map_context, get_map, list_map_names
from valocoach.retrieval.meta import format_meta_context, get_meta

# Vector pipeline — imported lazily in practice to avoid Ollama import at startup.
# Callers that need embeddings should import directly from the submodules.

__all__ = [
    # Static JSON retrieval
    "format_agent_context",
    "format_map_context",
    "format_meta_context",
    "get_agent",
    "get_map",
    "get_meta",
    "list_agent_names",
    "list_map_names",
]
