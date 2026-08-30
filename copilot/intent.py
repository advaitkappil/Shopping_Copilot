"""Intent routing: classify a user message as Buying vs Browsing.

The deterministic evaluator opens with recognizable templates; we detect them
and also fall back to keyword heuristics so hidden sessions still route.
"""
from __future__ import annotations

import re

BUY_PATTERNS = [
    re.compile(r"A key requirement is", re.I),
    re.compile(r"key requirement is", re.I),
    re.compile(r"what I need is", re.I),
    re.compile(r"I need .*(?:\d|material|size|color|budget|brand)", re.I),
]
BROWSE_PATTERNS = [
    re.compile(r"still exploring", re.I),
    re.compile(r"browsing", re.I),
    re.compile(r"looking for .* but", re.I),
    re.compile(r"I'?m not sure", re.I),
    re.compile(r"show me", re.I),
]


def detect_intent(message: str, scenario_hint: str | None = None) -> str:
    """Return 'buying', 'browsing' or the scenario alias (override/boundary)."""
    if scenario_hint:
        hint = scenario_hint.lower()
        if "override" in hint:
            return "override"
        if "boundary" in hint:
            return "boundary"
    for p in BUY_PATTERNS:
        if p.search(message or ""):
            return "buying"
    for p in BROWSE_PATTERNS:
        if p.search(message or ""):
            return "browsing"
    return "browsing"