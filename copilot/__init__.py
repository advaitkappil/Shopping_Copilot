"""Shopping Copilot - conversational e-commerce search agent.

Implements intent routing, a slot state machine, hybrid retrieval and a
deterministic local reranker (no paid API). See copilot/agent.py for the
drop-in Agent entry point.
"""
from __future__ import annotations

__version__ = "0.1.0"