"""Editable starter Agent.

This replaces the weak stateless BM25 baseline with the full Shopping Copilot
agent (intent routing, slot state machine, hybrid retrieval + local reranker).
See copilot/agent.py for the implementation.
"""
from __future__ import annotations

# Re-export the real agent under the expected name so the official evaluator
# picks it up without modification.
from copilot.agent import Agent  # noqa: F401
