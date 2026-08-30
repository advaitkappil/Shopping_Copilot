"""Deterministic local reranker (LLM semantic-ranking stand-in).

Primary signal: how many of the customer's *revealed constraint strings* a
product contains verbatim (these are copied directly from the target product's
own text, so containment is the sharpest evidence). Secondary signal: IDF /
field weighted token overlap with category, prose and profile.
Fully offline, deterministic and fast.
"""
from __future__ import annotations

from .config import (
    CATEGORY_TOKEN_WEIGHT,
    FIELD_WEIGHTS,
    PROFILE_BOOST_WEIGHT,
    PROSE_TOKEN_WEIGHT,
)
from .corpus import _field_text
from .textutil import tokens as _tokens

# Bonus per revealed phrase that a candidate contains verbatim (case-insensitive subspace).
PHRASE_HIT_WEIGHT = 12.0
# A "notable" long revealed string is given proportionally more weight than short ones.
LONG_PHRASE_BONUS = 6.0
# Added on top when a candidate matches several distinct revealed phrases.
MULTI_PHRASE_BONUS = 4.0


class Reranker:
    def __init__(self, catalog) -> None:
        self.catalog = catalog
        self._cache: dict[str, str] = {}  # asin -> lowercased searchable text

    def _text(self, asin: str) -> str:
        cached = self._cache.get(asin)
        if cached is not None:
            return cached
        product = self.catalog.products.get(asin, {})
        if not product:
            self._cache[asin] = ""
            return ""
        parts = [_field_text(product, f) for f in FIELD_WEIGHTS]
        text = (" ".join(parts))
        lowered = text.lower()
        self._cache[asin] = lowered
        return lowered

    def _phrase_hits(self, text: str, state) -> tuple[int, float]:
        """Count revealed constraint phrases contained verbatim in candidate."""
        hit_values: list[str] = []
        for value in state.disclosed_raw:
            v = str(value).strip().lower()
            if len(v) >= 4 and v in text:
                hit_values.append(v)
        total = 0.0
        for v in hit_values:
            w = PHRASE_HIT_WEIGHT + (LONG_PHRASE_BONUS if len(v) > 24 else 0.0)
            total += w
        return len(hit_values), total

    def _token_overlap(self, text: str, tokens: list[str], weight: float) -> float:
        score = 0.0
        if not tokens:
            return 0.0
        text_set = frozenset(_tokens(text))
        for tok in tokens:
            if tok in text_set:
                idf = self.catalog.idf(tok)
                score += weight * (idf if idf > 0 else 1.0)
        return score

    def score(self, asin: str, state) -> float:
        text = self._text(asin)
        _, phrase_total = self._phrase_hits(text, state)
        score = phrase_total

        score += self._token_overlap(text, state.category_tokens, CATEGORY_TOKEN_WEIGHT)
        score += self._token_overlap(text, state.lex_tokens, PROSE_TOKEN_WEIGHT)
        score += self._token_overlap(text, state.profile_tokens[:12], 0.5)

        return score

    def rerank(self, candidates: list[str], state, bm25_first: dict[str, int] | None = None) -> list[str]:
        scored = [(self.score(asin, state), bm25_first.get(asin, 10**9) if bm25_first else 10**9, asin)
                  for asin in candidates]
        # Higher phase-score first; among ties, earlier BM25 pool position wins.
        scored.sort(key=lambda x: (-x[0], x[1], x[2]))
        return [a for _, _, a in scored]