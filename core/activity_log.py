"""
core/activity_log.py — Structured agent activity logging.

Provides a consistent log format that the frontend can render
as a real-time activity feed showing exactly what Maya is doing.
"""
import time
from typing import Optional, Dict, Any


def make_log_entry(
    agent: str,
    action: str,
    detail: str = "",
    status: str = "running",  # running | success | error | waiting
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a structured log entry for frontend display.
    
    Args:
        agent: Which agent is acting (e.g. "Maya", "Ingestion", "Cleaning")
        action: Short action description (e.g. "Analyzing schema")
        detail: Longer detail or result snippet
        status: Visual status indicator
        metadata: Extra key-value data (metrics, file names, etc.)
    """
    return {
        "ts": time.time(),
        "agent": agent,
        "action": action,
        "detail": detail[:500] if detail else "",
        "status": status,
        "metadata": metadata or {},
    }


def append_log(state: dict, entry: Dict[str, Any]) -> dict:
    """Append a log entry to state's agent_log."""
    logs = list(state.get("agent_log", []))
    logs.append(entry)
    return {"agent_log": logs}
