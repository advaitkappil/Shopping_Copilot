"""Diagnose failures: does the target get into our pool? at what rank?

Reuses the official simulator, then for the turn where the target IS present in
our returned top-10 (or the final turn) reports the pool size and rank.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot.agent import Agent  # noqa: E402
from copilot.corpus import Catalog  # noqa: E402
from copilot.config import POOL_SIZE  # noqa: E402


def main() -> None:
    app = argparse.ArgumentParser()
    app.add_argument("--catalog", default="data/catalog.jsonl")
    app.add_argument("--dataset", default="data/public_set.jsonl")
    args = app.parse_args()

    from evaluator import local_evaluator as ev

    samples = ev.load_jsonl(args.dataset)
    catalog_ids, categories, products = ev.catalog_index(args.catalog)
    catalog = Catalog(args.catalog)
    agent = Agent(args.catalog, catalog=catalog)

    misses = 0
    hit_ranks = []
    for idx, sample in enumerate(samples):
        session_id = f"d_{idx}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        edge, ebehave = ev.materialize_hidden_fields(sample, products)
        es = {**sample, "intent_card": edge, "behavior": ebehave}
        disclosed = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        msg = ev.initial_message(es, ev.coarse_category(categories.get(target, [])), disclosed)
        final_rank = None
        final_contained = False
        final_turn = None
        for turn in range(1, 11):
            resp = agent.respond(session_id, msg, turn, 10)
            ranked = ev.normalize_recommendations(resp.get("recommendations"), catalog_ids)
            if target in ranked:
                final_contained = True
                final_rank = ranked.index(target) + 1
                final_turn = turn
                if override_applied:
                    break
            if turn == 10:
                break
            ovr = es.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(ovr.get("turn", 3)):
                override_applied = True
                nv = str(ovr.get("new_value", ""))
                if nv:
                    disclosed.add(nv)
                msg = str(ovr.get("message", ""))
            else:
                msg, boundary_used = ev.customer_reply(es, resp.get("ask_attribute"), disclosed, boundary_used)
        if final_contained:
            hit_ranks.append(final_rank)
        else:
            misses += 1
            print(f"MISS {sample['sample_id']} {sample['scenario_type']:<14} target={target}")

    print(f"\nmisses: {misses}/200")
    if hit_ranks:
        from collections import Counter

        dist = Counter(hit_ranks)
        print("rank distribution of hit sessions:", dict(sorted(dist.items())))
        print(f"mean rank among hits: {sum(hit_ranks)/len(hit_ranks):.2f}")


if __name__ == "__main__":
    main()