#!/usr/bin/env python3
"""
Long-term memory — user preferences and interaction history persistence.

Stores user preferences, common geometry configurations, and interaction
patterns to personalize agent responses across sessions.
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict


class LongTermMemory:
    """Persists user preferences and interaction patterns.

    Storage layers:
    - User preferences (e.g., preferred units, confidence thresholds)
    - Geometry library (named geometry configs the user frequently uses)
    - Interaction history (prompt/response pairs for context)
    - Agent statistics (which agents are used most)
    """

    def __init__(self, user_id: str = "default", storage_dir: str = ".memory/long_term"):
        self.user_id = user_id
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self._data = self._load_or_create()

    def _storage_path(self) -> str:
        return os.path.join(self.storage_dir, f"{self.user_id}.json")

    def _load_or_create(self) -> dict:
        path = self._storage_path()
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {
            "user_id": self.user_id,
            "created_at": datetime.now().isoformat(),
            "preferences": {},
            "geometry_library": {},
            "interaction_history": [],
            "agent_stats": defaultdict(int),
        }

    def _save(self):
        path = self._storage_path()
        with open(path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    # ── Preferences ──

    def get_preference(self, key: str, default=None) -> Any:
        """Get a user preference."""
        return self._data["preferences"].get(key, default)

    def set_preference(self, key: str, value: Any):
        """Set a user preference."""
        self._data["preferences"][key] = value
        self._save()

    def get_all_preferences(self) -> dict:
        """Get all user preferences."""
        return dict(self._data["preferences"])

    # ── Geometry Library ──

    def save_geometry(self, name: str, params: dict, notes: str = ""):
        """Save a named geometry to the user's library."""
        self._data["geometry_library"][name] = {
            "params": params,
            "notes": notes,
            "saved_at": datetime.now().isoformat(),
        }
        self._save()

    def get_geometry(self, name: str) -> Optional[dict]:
        """Retrieve a saved geometry by name."""
        return self._data["geometry_library"].get(name)

    def list_geometries(self) -> List[str]:
        """List all saved geometry names."""
        return list(self._data["geometry_library"].keys())

    # ── Interaction History ──

    def save_prompt_response(self, user_id: str, prompt: str,
                             response: Optional[str], agent_name: str):
        """Save a prompt/response pair to interaction history."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "prompt": prompt[:500],  # truncate long prompts
            "response_preview": (response or "")[:200],
        }
        self._data["interaction_history"].append(entry)
        # Keep last 100 interactions
        if len(self._data["interaction_history"]) > 100:
            self._data["interaction_history"] = self._data["interaction_history"][-100:]

        # Update agent stats
        if "agent_stats" not in self._data:
            self._data["agent_stats"] = {}
        self._data["agent_stats"][agent_name] = (
            self._data["agent_stats"].get(agent_name, 0) + 1
        )
        self._save()

    def get_recent_interactions(self, n: int = 10) -> List[dict]:
        """Get the N most recent interactions."""
        return self._data["interaction_history"][-n:]

    def get_agent_stats(self) -> dict:
        """Get usage statistics per agent."""
        return dict(self._data.get("agent_stats", {}))

    # ── Context for Agents ──

    def get_context_for_agent(self, agent_name: str) -> str:
        """Build a context string for an agent from user preferences and history.

        Used by BasicAgent._invoke_llm to prepend user-specific context.
        """
        prefs = self.get_all_preferences()
        geos = self.list_geometries()
        stats = self.get_agent_stats()

        parts = []
        if prefs:
            parts.append(f"User preferences: {json.dumps(prefs)}")
        if geos:
            parts.append(f"Saved geometries: {', '.join(geos)}")
        if stats:
            parts.append(f"Agent usage: {json.dumps(stats)}")

        return "\n".join(parts) if parts else ""

    def clear(self):
        """Clear all stored data for this user."""
        self._data = {
            "user_id": self.user_id,
            "created_at": datetime.now().isoformat(),
            "preferences": {},
            "geometry_library": {},
            "interaction_history": [],
            "agent_stats": {},
        }
        self._save()
