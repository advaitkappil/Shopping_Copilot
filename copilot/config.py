"""Central tunable configuration for the Shopping Copilot agent.

All weights and thresholds live here so they can be swept by the self-tuning
script without touching logic code.
"""
from __future__ import annotations

# Attribute elicitation order. The customer reveals up to two constraint strings
# matching the requested attribute (or, for "other", any remaining constraints).
ASK_ORDER = [
    "other",
    "feature",
    "material",
    "color",
    "size",
    "style",
    "use_case",
    "budget",
    "brand",
    "category",
]

# Retrieval candidate budget pulled from the index per turn (larger = better
# recall, slower). Reranking then narrows this to top_k.
POOL_SIZE = 300

# How strongly each catalog field contributes to a candidate's final score.
FIELD_WEIGHTS = {
    "title": 6.0,
    "categories": 4.0,
    "features": 3.0,
    "details": 2.5,
    "description": 2.0,
    "store": 1.0,
}

# Weight applied to prose tokens parsed from free text (weaker than the
# verbatim phrase-containment signal used in ranking).
PROSE_TOKEN_WEIGHT = 1.0

# Boost for tokens that came from the coarse category discovered online.
CATEGORY_TOKEN_WEIGHT = 2.5

# Profile personalization: how much user preference tags tilt the rerank.
PROFILE_BOOST_WEIGHT = 1.5

# Contradiction / override handling.
OVERRIDE_RESET = False          # wipe old slots when an override message arrives
SLOT_DECAY_FACTOR = 1.0        # per-turn multiplicative decay on old slot weight

# Reranking blend: final = BM25_norm * bm25_w + overlap_score * ov_w
BM25_WEIGHT = 1.0
OVERLAP_WEIGHT = 1.0
MRR_TOP = 10                   # always emit top_k recommendations (hit counts early)