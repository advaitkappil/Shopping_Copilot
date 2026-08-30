"""Tuning / debugging harness.

Replays the official public sessions using the exact evaluator simulator but
prints per-session details (scenario, whether the target entered our pool, at
what turn, and the revealed constraint strings) so we can see where retrieval
loses the target instead of only seeing aggregate metrics.

Usage:
    python tools/tune_harness.py [--limit N] [--show-sessions] [--scenario X]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot.agent import Agent  # noqa: E402
from copilot.textutil import extract_constraints  # noqa: E402

MAX_TURNS = 10
TOP_K = 10


def main() -> None:
    app = argparse.ArgumentParser()
    app.add_argument("--catalog", default="data/catalog.jsonl")
    app.add_argument("--dataset", default="data/public_set.jsonl")
    app.add_argument("--limit", type=int, default=0)
    app.add_argument("--show-sessions", action="store_true")
    app.add_argument("--scenario", default=None, help="filter by scenario type")
    args = app.parse_args()

    from evaluator import local_evaluator as ev

    samples = ev.load_jsonl(args.dataset)
    catalog_ids, categories, products = ev.catalog_index(args.catalog)

    agent = Agent(args.catalog)

    sessions = []
    for sample in samples:
        if args.scenario and sample["scenario_type"] != args.scenario:
            continue
        sessions.append(sample)
    if args.limit:
        sessions = sessions[: args.limit]

    replay(agent, sessions, ev, catalog_ids, categories, products, args.show_sessions)


def replay(agent, samples, ev, catalog_ids, categories, products, show_sessions):
    stats = {"hit": 0, "pool_contained_target": 0, "first_turn_in_pool": None}
    first_pool_turns = []

    for idx, sample in enumerate(samples):
        session_id = f"tune_{idx}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        effective_card, effective_behavior = ev.materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": effective_card, "behavior": effective_behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = ev.initial_message(effective_sample, ev.coarse_category(categories.get(target, [])), disclosed)
        hit_turn = None
        best_rank = None
        target_in_pool_turn = None
        detail_lines = []

        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, user_message, turn, TOP_K)
            ranked = ev.normalize_recommendations(response.get("recommendations"), catalog_ids)
            if target in ranked:
                if target_in_pool_turn is None:
                    target_in_pool_turn = turn
                if override_applied and hit_turn is None:
                    hit_turn = turn
                    best_rank = ranked.index(target) + 1
            if override_applied and hit_turn is not None:
                break
            if turn == MAX_TURNS:
                break
            override = effective_sample.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(override.get("message", ""))
            else:
                pre_disclosed = set(disclosed)
                user_message, boundary_used = ev.customer_reply(effective_sample, response.get("ask_attribute"), disclosed, boundary_used)
                new_reveals = [r for r in extract_constraints(user_message)]
                detail_lines.append((turn, response.get("ask_attribute"), new_reveals))

        if hit_turn is not None:
            stats["hit"] += 1
        if target_in_pool_turn is not None:
            stats["pool_contained_target"] += 1
            first_pool_turns.append(target_in_pool_turn)

        if show_sessions:
            _sp = lambda s: str(s)
            print(
                f"[{_sp(sample['sample_id'])}] {_sp(sample['scenario_type']):<14} "
                f"hit={hit_turn} rank={best_rank} in_pool_turn={target_in_pool_turn} "
                f"target={_sp(target)}"
            )
            for turn, attr, reveals in detail_lines:
                print(f"    t{turn} ask={_sp(attr):<10} reveals={reveals}")

    n = len(samples)
    print("\n=== AGGREGATE (tune harness, not official score) ===")
    print(f"sessions: {n}")
    print(f"official-hits (target ranked & scorable): {stats['hit']} ({stats['hit']/n:.1%})")
    print(f"target ever in our returned top-10: {stats['pool_contained_target']} ({stats['pool_contained_target']/n:.1%})")
    if first_pool_turns:
        print(f"mean first turn target appeared in top-10: {sum(first_pool_turns)/len(first_pool_turns):.2f}")


if __name__ == "__main__":
    main()