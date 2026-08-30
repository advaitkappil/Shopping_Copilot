"""Multi-route retrieval: keyword BM25 pool + phrase pool + reranking.

The customer reveals LITERAL constraint strings from the target product. Long
strings are near-unique, so we (1) pull a BM25 pool over the query tokens and
(2) additionally phrase-match each revealed constraint string verbatim and
union those ASINs in. A deterministic overlap reranker then picks top_k, with
BM25 pool position used as a tiebreak among equal-scoring candidates.
"""
from __future__ import annotations

from .config import POOL_SIZE


def retrieve(
    catalog,
    reranker,
    state,
    top_k: int = 10,
    bm25_terms: list[str] | None = None,
) -> tuple[list[str], int]:
    terms = list(state.lex_tokens)
    terms += state.category_tokens
    if bm25_terms:
        terms += bm25_terms

    dedup: list[str] = []
    for t in terms:
        if t and t not in dedup:
            dedup.append(t)

    pool: list[str] = []
    seen: set[str] = set()
    bm25_first: dict[str, int] = {}

    # Route 1: BM25 over all query tokens (records pool position as tiebreak).
    for pos, asin in enumerate(catalog.bm25_pool(dedup, POOL_SIZE)):
        if asin not in seen:
            bm25_first[asin] = pos
            seen.add(asin)
            pool.append(asin)

    # Route 2: verbatim phrase-match on each revealed constraint string.
    for value in state.disclosed_raw:
        phrase = _as_phrase(value)
        if not phrase:
            continue
        try:
            for asin in catalog.phrase_pool([phrase], POOL_SIZE):
                if asin not in seen:
                    seen.add(asin)
                    pool.append(asin)
        except Exception:
            continue

    # Route 3: coarse-category phrase route. Ensures products whose categories
    # match the coarse category always enter the pool (fixes boundary/turn-1
    # recall where OR-BM25 drops category members below the pool cap).
    if state.category:
        try:
            for asin in catalog.phrase_pool([state.category], POOL_SIZE):
                if asin not in seen:
                    seen.add(asin)
                    pool.append(asin)
        except Exception:
            pass

    # Route 4: category-token OR fallback if the pool is tiny (early turns).
    if len(pool) < 10:
        for asin in catalog.fuzzy_pool(state.category_tokens, POOL_SIZE):
            if asin not in seen:
                seen.add(asin)
                pool.append(asin)

    ranked = reranker.rerank(pool, state, bm25_first=bm25_first)
    return ranked[:top_k], len(pool)


def _as_phrase(value: str) -> str:
    """Turn a revealed constraint string into a safe FTS5 phrase query."""
    cleaned = " ".join(str(value).split())
    if len(cleaned) < 3:
        return ""
    return cleaned.replace('"', '" "')