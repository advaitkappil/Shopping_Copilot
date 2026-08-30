"""Attribute elicitation strategy.

Chooses which attribute to ask each turn. The customer (simulator) reveals up
to two constraint strings that match the asked attribute, or any remaining
constraints for 'other'. We prefer attributes we haven't exhausted for info
gain, and we stop asking once the candidate pool looks confusable (proactive
guidance / over-generality handling).
"""
from __future__ import annotations

from .config import ASK_ORDER


def choose_attribute(state, pool_size_estimate: int | None = None, turn: int = 1) -> str | None:
    """Pick the attribute most likely to reveal new discriminating info.

    Returns an allowed attribute string or None (stop asking / recommend only).
    """
    # Always prefer a revealed-domain attribute we haven't tried yet.
    for attr in ASK_ORDER:
        if attr in state.exhausted_attrs:
            continue
        if attr in state.asked_attrs:
            continue
        return attr

    # All tried once; revisit attributes that still might hold info (not flagged
    # as no-preference) in priority order.
    for attr in ASK_ORDER:
        if attr in state.exhausted_attrs:
            continue
        return attr

    return None  # nothing left worth asking -> recommend only