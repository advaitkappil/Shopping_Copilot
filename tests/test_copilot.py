"""Unit tests for the Shopping Copilot agent's core logic."""
from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
import json

from copilot.corpus import Catalog
from copilot.rank import Reranker
from copilot.state import SessionState
from copilot.textutil import extract_constraints, extract_category


class _MiniCatalog:
    """Tiny stand-in catalog for unit tests (no FTS/IDF build needed)."""

    def __init__(self, products):
        self.products = products

    def idf(self, term: str) -> float:
        return 2.0 if term else 0.0


def _make_catalog_file(rows):
    tmp = tempfile.mkdtemp()
    p = Path(tmp) / "catalog.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


class ConstraintParsingTest(unittest.TestCase):
    def test_extract_category(self):
        self.assertEqual(
            extract_category("I'm looking for women dresses, but I'm still exploring."),
            "women dresses",
        )

    def test_extract_constraints(self):
        msg = "For that, what matters is: cotton; color: black."
        self.assertEqual(extract_constraints(msg), ["cotton", "color: black"])


class RerankerTest(unittest.TestCase):
    def test_phrase_containment_outranks_token_only(self):
        # Target contains the revealed phrase verbatim; decoy only shares a word.
        catalog = _MiniCatalog({
            "TARGET": {"title": "Cotton dress cotton", "features": ["warm cotton knit fabric"],
                        "categories": [], "details": {}, "description": [], "store": ""},
            "DECOY": {"title": "Warm dress", "features": ["something else"],
                        "categories": [], "details": {}, "description": [], "store": ""},
        })
        reranker = Reranker(catalog)
        state = SessionState("s1", {})
        state.category = "dresses"
        state.category_tokens = ["dresses"]
        state.reveal("feature", ["warm cotton knit fabric"], "")
        ranked = reranker.rerank(["DECOY", "TARGET"], state)
        self.assertEqual(ranked[0], "TARGET")


class IntentOverrideStateTest(unittest.TestCase):
    def test_override_preserves_positive_phrases(self):
        state = SessionState("s1", {})
        state.reveal("material", ["cotton"], "")
        state.maybe_handle_override(
            "Actually, ignore my earlier preference. What I need is: polyester."
        )
        self.assertTrue(state.override_seen)
        # The new value is recorded:
        self.assertIn("polyester", state.disclosed_raw)
        # With OVERRIDE_RESET=False the useful earlier phrase is retained:
        self.assertIn("cotton", state.disclosed_raw)


class SlotsAccumulateTest(unittest.TestCase):
    def test_reveal_appends_and_dedupes(self):
        state = SessionState("s1", {})
        state.reveal("material", ["cotton", "cotton"], "")
        state.reveal("color", ["black"], "")
        self.assertEqual(set(state.slots["material"]), {"cotton"})
        self.assertEqual(state.slots["color"], ["black"])


if __name__ == "__main__":
    unittest.main()