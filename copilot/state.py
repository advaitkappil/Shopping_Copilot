"""Per-session conversational state tracker.

Implements the dialog strategy pillar: dynamic information accumulation
(incremental slots), intent override (slot erasure + reseed) and personalized
context distillation (short-term slots + long-term profile).
"""
from __future__ import annotations

from collections import defaultdict

from .config import OVERRIDE_RESET, SLOT_DECAY_FACTOR
from .textutil import extract_category, extract_no_preference_attr, is_override


class SessionState:
    def __init__(self, session_id: str, user_profile: dict) -> None:
        self.session_id = session_id
        self.category: str | None = None
        self.category_tokens: list[str] = []

        # intent-type_override-aware slots: attribute -> list of revealed values
        self.slots: dict[str, list[str]] = defaultdict(list)
        self.disclosed_raw: list[str] = []          # literal revealed constraint strings
        self.turn_weights: dict[str, float] = {}     # disclosed_raw -> weight (recency)

        # accumulated lexical query boost tokens
        self.lex_tokens: list[str] = []

        self.asked_attrs: list[str] = []
        self.exhausted_attrs: set[str] = set()
        self.no_pref_attrs: set[str] = set()

        self.override_seen = False
        self.override_value: str | None = None

        # personalization (long-term profile)
        self.profile = user_profile or {}
        self.profile_tokens: list[str] = self._profile_tokens()

    # ------------------------------------------------------------- profile
    def _profile_tokens(self) -> list[str]:
        from .textutil import tokens

        text = ""
        profile = self.profile
        if isinstance(profile, dict):
            tags = profile.get("preference_tags") or []
            summary = profile.get("summary") or ""
            if isinstance(tags, list):
                text += " ".join(str(t) for t in tags)
            else:
                text += str(tags)
            text += " " + str(summary)
            freq = profile.get("purchase_frequency") or ""
            rating = profile.get("rating_style") or ""
            text += " " + str(freq) + " " + str(rating)
        return list(dict.fromkeys(tokens(text)))

    @property
    def disclosed_attrs(self) -> list[str]:
        return list(self.slots.keys())

    # -------------------------------------------------------------- reveal
    def reveal(self, attr: str, values: list[str], message: str) -> None:
        """Record new constraint values the customer just revealed."""
        for value in values:
            if not value or value in self.disclosed_raw:
                continue
            self.slots[attr].append(value)
            self.disclosed_raw.append(value)
            self.turn_weights[value] = 1.0
            from .textutil import tokens

            self.lex_tokens += tokens(value)
        # Mark an attribute exhausted once the customer reported no preference
        # for it (info source consumed).
        self._mark_if_no_pref(message)

    def _mark_if_no_pref(self, message: str) -> None:
        attr = extract_no_preference_attr(message)
        if attr:
            self.exhausted_attrs.add(attr)
            self.no_pref_attrs.add(attr)

    def seed_category(self, message: str) -> None:
        from .textutil import tokens

        cat = extract_category(message)
        if cat and not self.category:
            self.category = cat
            self.category_tokens = list(dict.fromkeys(tokens(cat)))

    # ------------------------------------------------------------- override
    def maybe_handle_override(self, message: str) -> bool:
        """If message signals intent override, erase slots & reseed."""
        if not is_override(message):
            return False
        from .textutil import extract_constraints, tokens

        self.override_seen = True
        if OVERRIDE_RESET:
            # Slot erasure + rewriting: drop everything accumulated so far.
            self.slots.clear()
            self.disclosed_raw = []
            self.lex_tokens = []
            self.turn_weights = {}
            self.exhausted_attrs = set()
            self.no_pref_attrs = set()
            self.asked_attrs = []
        else:
            # Keep useful pre-override phrases (same target card) but re-open the
            # ask queue so the high-information wildcard/feature order restarts
            # now that scoring counts - surfaces the distinctive long phrases.
            self.exhausted_attrs = set()
            self.no_pref_attrs = set()
            self.asked_attrs = []

        # Record the fresh override value (still from the same target card).
        for value in extract_constraints(message):
            self.override_value = value
            self.disclosed_raw.append(value)
            self.lex_tokens += tokens(value)
            self.turn_weights[value] = 1.0
        return True

    def decay(self) -> None:
        """Apply temporal slot decay so the most recent info dominates."""
        for key in list(self.turn_weights):
            self.turn_weights[key] *= SLOT_DECAY_FACTOR

    def mark_asked(self, attr: str) -> None:
        self.asked_attrs.append(attr)
        self.exhausted_attrs.discard(attr)

    def mark_exhausted(self, attr: str) -> None:
        self.exhausted_attrs.add(attr)

    def requeue_last_asked(self) -> None:
        """Pop the most recent attribute so it can be asked again.

        Used when the customer's reply was a boundary "use your judgment"
        response, which consumed the ask without revealing any constraint.
        """
        if self.asked_attrs:
            self.asked_attrs.pop()