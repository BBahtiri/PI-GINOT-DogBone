#!/usr/bin/env python3
"""
Short-term memory — conversation persistence via LangGraph checkpointer.

Provides session-level memory that persists across page refreshes
within the same Streamlit session. Each thread_id gets its own
conversation history.
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime


class ShortTermMemory:
    """Manages conversation threads and session persistence.

    In production, this would use a database backend. For now, it wraps
    LangGraph's InMemorySaver checkpointer with additional metadata.
    """

    def __init__(self, storage_dir: str = ".memory/short_term"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self._sessions: Dict[str, dict] = {}

    def create_session(self, thread_id: str, user_id: str = "default") -> dict:
        """Create a new conversation session."""
        session = {
            "thread_id": thread_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "metadata": {},
        }
        self._sessions[thread_id] = session
        return session

    def get_session(self, thread_id: str) -> Optional[dict]:
        """Retrieve session metadata."""
        return self._sessions.get(thread_id)

    def update_activity(self, thread_id: str):
        """Update last activity timestamp."""
        if thread_id in self._sessions:
            self._sessions[thread_id]["last_active"] = datetime.now().isoformat()

    def list_sessions(self, user_id: str = "default") -> List[dict]:
        """List all sessions for a user, sorted by last activity."""
        sessions = [
            s for s in self._sessions.values() if s["user_id"] == user_id
        ]
        return sorted(sessions, key=lambda s: s["last_active"], reverse=True)

    def delete_session(self, thread_id: str):
        """Delete a conversation session."""
        self._sessions.pop(thread_id, None)

    def save_to_disk(self, thread_id: str):
        """Persist session metadata to disk."""
        session = self._sessions.get(thread_id)
        if session:
            path = os.path.join(self.storage_dir, f"{thread_id}.json")
            with open(path, "w") as f:
                json.dump(session, f, indent=2)

    def load_from_disk(self, thread_id: str) -> Optional[dict]:
        """Load session metadata from disk."""
        path = os.path.join(self.storage_dir, f"{thread_id}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                session = json.load(f)
                self._sessions[thread_id] = session
                return session
        return None

    def get_context_window(self, thread_id: str) -> dict:
        """Get context window metadata for the session."""
        session = self.get_session(thread_id)
        if not session:
            return {"exists": False}
        return {
            "exists": True,
            "thread_id": thread_id,
            "created_at": session["created_at"],
            "last_active": session["last_active"],
        }
