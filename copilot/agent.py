"""Shopping Copilot Agent - drop-in replacement for the weak BM25 starter.

Pipeline per turn:
  1. State update: extract category + revealed constraints; handle intent
     override pivot and temporal slot decay.
  2. Multi-route retrieval: BM25 pool + verbatim phrase pool -> local reranker.
  3. Always recommend top_k products and, when the pool is still broad, ask a
     proactive clarification question (candidate-pool overload handling).
"""
from __future__ import annotations

from pathlib import Path

from .ask import choose_attribute
from .corpus import Catalog
from .rank import Reranker
from .retrieval import retrieve
from .state import SessionState
from .textutil import extract_constraints, extract_no_preference_attr, USE_JUDGMENT_RE

# Message templates - natural shopping-copilot voice.
_ATTR_QUESTION = {
    "material": "Quick check - do you have a material preference (e.g. cotton, leather, polyester)?",
    "color": "Any color preference? (black, white, blue, pink, ...)",
    "size": "What size or fit are you after?",
    "style": "Is there a particular style or fit you prefer?",
    "feature": "Are there any specific features you need?",
    "use_case": "What will you use it for - outdoor, work, gym, everyday?",
    "budget": "Do you have a budget in mind?",
    "brand": "Any brand you prefer?",
    "category": "Which product category are you most interested in?",
    "other": "Could you tell me a bit more about what you're looking for?",
}


class Agent:
    """Stateless externally, stateful per session (implements the Agent contract)."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl", catalog: "Catalog | None" = None) -> None:
        self.catalog_path = Path(catalog_path)
        self.catalog = catalog if catalog is not None else Catalog(catalog_path)
        self.reranker = Reranker(self.catalog)
        self._sessions: dict[str, SessionState] = {}
        self._profile: dict[str, dict] = {}

    # ------------------------------------------------------------ interface
    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(session_id, user_profile)
        self._profile[session_id] = user_profile or {}

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        state = self._sessions[session_id]

        # --- 1. Dialog state update ---------------------------------------
        state.seed_category(user_message)
        overridden = state.maybe_handle_override(user_message)

        revealed = extract_constraints(user_message)
        if not overridden:
            # The reply names values for the attribute we asked last turn.
            last_attr = state.asked_attrs[-1] if state.asked_attrs else None
            if revealed and last_attr and last_attr not in state.exhausted_attrs:
                state.reveal(last_attr, revealed, user_message)
            elif revealed:
                state.reveal("other", revealed, user_message)
            elif extract_no_preference_attr(user_message):
                state._mark_if_no_pref(user_message)

            # Boundary scenario: the customer had "no preference" for exactly
            # the attribute we asked and told us to use judgment. Re-queue it so
            # the next turns can still surface its real constraint.
            if USE_JUDGMENT_RE.search(user_message or ""):
                state.requeue_last_asked()

        state.decay()

        # --- 2. Retrieval --------------------------------------------------
        ranked, pool_size = retrieve(self.catalog, self.reranker, state, top_k)

        # --- 3. Elicitation + response -------------------------------------
        ask_attr = choose_attribute(state, pool_size, turn)
        if ask_attr:
            state.mark_asked(ask_attr)
            q = _ATTR_QUESTION.get(ask_attr, "Can you share a bit more detail?")
            message = f"Here are a few options based on what you've shared; to narrow it down further, {q}"
        else:
            message = "Here are the strongest matches I found for you."

        return {
            "message": message,
            "ask_attribute": ask_attr,
            "recommendations": [{"parent_asin": a} for a in ranked],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }