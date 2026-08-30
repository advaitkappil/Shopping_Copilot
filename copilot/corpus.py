"""In-memory product catalog index.

Uses SQLite FTS5 for BM25 candidate retrieval plus a deterministic token
overlap reranker. The catalog is frozen and read-only; we never mutate ASINs.
"""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

from .config import FIELD_WEIGHTS
from .textutil import tokens


def _field_text(product: dict, field: str) -> str:
    value = product.get(field)
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{k} {v}" for k, v in value.items() if v not in (None, ""))
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v not in (None, ""))
    return str(value)


class Catalog:
    """Frozen catalog wrapper caching the FTS index, products and IDF stats."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.products: dict[str, dict] = {}
        self.categories: dict[str, list[str]] = {}
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute("pragma synchronous=off")
        self._build_fts()
        self._df: dict[str, int] = {}
        self._n_docs = len(self.products)
        self._build_idf()

    # ------------------------------------------------------------------ build
    def _build_fts(self) -> None:
        self._conn.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, "
            "store, description, tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                asin = str(product["parent_asin"])
                self.products[asin] = product
                self.categories[asin] = [str(v) for v in product.get("categories") or []]
                batch.append((
                    asin,
                    _field_text(product, "title"),
                    _field_text(product, "categories"),
                    _field_text(product, "features"),
                    _field_text(product, "details"),
                    _field_text(product, "store"),
                    _field_text(product, "description"),
                ))
                if len(batch) >= 2000:
                    self._conn.executemany(
                        "INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch
                    )
                    batch.clear()
        if batch:
            self._conn.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
        self._conn.commit()

    def _build_idf(self) -> None:
        """Document frequency for rare-token emphasis in overlap scoring."""
        df: dict[str, int] = {}
        for asin in self.products:
            seen: set[str] = set()
            for field, weight in FIELD_WEIGHTS.items():
                if weight <= 0:
                    continue
                for tok in set(tokens(_field_text(self.products[asin], field))):
                    key = (field, tok)
                    if key not in seen:
                        seen.add(key)
                        df[tok] = df.get(tok, 0) + 1
        self._df = df

    # ---------------------------------------------------------------- query
    def bm25_pool(self, terms: list[str], pool_size: int) -> list[str]:
        """Return up to pool_size parent_asins by BM25 over the given terms."""
        unique = list(dict.fromkeys(t for t in terms if t and len(t) > 1))[:40]
        if not unique:
            return []
        quoted = [f'"{t}"' for t in unique]
        expression = " OR ".join(quoted)
        try:
            rows = self._conn.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, pool_size),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [str(r[0]) for r in rows]

    def phrase_pool(self, phrases: list[str], pool_size: int) -> list[str]:
        """Exact FTS5 phrase match on each quoted phrase, unioned."""
        result: list[str] = []
        seen: set[str] = set()
        for phrase in phrases:
            match = f'"{phrase}"'
            try:
                rows = self._conn.execute(
                    "SELECT parent_asin FROM products WHERE products MATCH ? LIMIT ?",
                    (match, pool_size),
                ).fetchall()
            except sqlite3.OperationalError:
                continue
            for r in rows:
                asin = str(r[0])
                if asin not in seen:
                    seen.add(asin)
                    result.append(asin)
        return result

    def fuzzy_pool(self, terms: list[str], pool_size: int) -> list[str]:
        """OR-based candidate pool with substring ('*') fallback for long terms."""
        unique = list(dict.fromkeys(t for t in terms if t and len(t) > 1))[:40]
        if not unique:
            return []
        parts: list[str] = []
        for t in unique:
            parts.append(f"{t}*" if len(t) >= 4 else f'"{t}"')
        expression = " OR ".join(parts)
        try:
            rows = self._conn.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? LIMIT ?",
                (expression, pool_size),
            ).fetchall()
        except sqlite3.OperationalError:
            return self.bm25_pool(terms, pool_size)
        return [str(r[0]) for r in rows]

    def idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        if df <= 0:
            return 0.0
        return math.log((self._n_docs + 1) / (df + 1)) + 1.0

    def price(self, asin: str) -> float | None:
        raw = self.products.get(asin, {}).get("price")
        if raw in (None, ""):
            return None
        try:
            return float(str(raw).lstrip("$").replace(",", ""))
        except (ValueError, AttributeError):
            return None