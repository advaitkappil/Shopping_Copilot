"""Text tokenization and constraint extraction helpers.

Everything here is deterministic and mirrors the evaluator's data shaping so
our retrieval tokens line up with the customer's revealed constraint strings.
"""
from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "still", "about", "around", "just", "your", "im", "like", "need", "what",
    "does", "not", "matter", "matters", "have", "there", "they", "them",
}

# Patterns produced by the deterministic evaluator simulator.
LOOKING_FOR_RE = re.compile(r"I'?m looking for (.+?)(?:[,.]|$)", re.I)
LOOKING_TRAILER = re.compile(r"(?:\s*(?:,)?\s*(?:but )?I'?m still exploring|[.\s]+)$", re.I)
KEY_REQ_RE = re.compile(r"(?:A key requirement is|key requirement is):\s*(.+?)[.]?\s*$", re.I)
WHAT_MATTERS_RE = re.compile(r"For that, what matters is:\s*(.+?)\.?\s*$", re.I)
WHAT_I_NEED_RE = re.compile(r"What I need is:\s*(.+?)\.?\s*$", re.I)
NO_PREF_RE = re.compile(r"I don't have an additional preference for ([\w_]+)\.", re.I)
IGNORE_PREV_RE = re.compile(r"ignore my earlier preference", re.I)
USE_JUDGMENT_RE = re.compile(r"please use your judgment", re.I)


def tokens(text: str) -> list[str]:
    """Lower-cased alphanumeric tokens, stopwords removed, len > 1."""
    out: list[str] = []
    for tok in TOKEN_RE.findall(text or ""):
        t = tok.lower()
        if len(t) > 1 and t not in STOPWORDS:
            out.append(t)
    return out


def unique_tokens(text: str) -> list[str]:
    return list(dict.fromkeys(tokens(text)))


def extract_category(message: str) -> str | None:
    m = LOOKING_FOR_RE.search(message or "")
    if m:
        raw = m.group(1).strip().rstrip(",").strip()
        # Drop trailing conversational filler that isn't part of the coarse
        # category (e.g. "but I'm still exploring").
        raw = LOOKING_TRAILER.sub("", raw).strip()
        return raw or None
    return None


def extract_constraints(message: str) -> list[str]:
    """Pull literal constraint chunks the evaluator embeds in a user message."""
    chunks: list[str] = []
    for regex in (KEY_REQ_RE, WHAT_MATTERS_RE, WHAT_I_NEED_RE):
        m = regex.search(message or "")
        if m:
            text = m.group(1).strip()
            # "what matters" chunks are ';'-separated
            for piece in text.split(";"):
                piece = re.sub(r"\s+", " ", piece).strip().rstrip(",").strip()
                if piece:
                    chunks.append(piece)
    return chunks


def is_override(message: str) -> bool:
    return bool(IGNORE_PREV_RE.search(message or "")) or bool(
        (WHAT_I_NEED_RE.search(message or "")) and ("ignore" in (message or "").lower())
    )


def asked_no_preference(message: str) -> bool:
    return NO_PREF_RE.search(message or "") is not None


def extract_no_preference_attr(message: str) -> str | None:
    m = NO_PREF_RE.search(message or "")
    return m.group(1) if m else None