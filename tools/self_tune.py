"""Self-tuning: search a small grid of config knobs against the dev set.

Uses the official evaluator's scoring function so optimization directly tracks
the metric judges will see (TechScore). Because a full rerank over 200 sessions
is expensive, this grid is deliberately small and each candidate reuses a
single prebuilt Catalog.

Usage:
    python tools/self_tune.py
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot.agent import Agent  # noqa: E402
from copilot.corpus import Catalog  # noqa: E402
import copilot.config as cfg  # noqa: E402


def main() -> None:
    app = argparse.ArgumentParser()
    app.add_argument("--catalog", default="data/catalog.jsonl")
    app.add_argument("--dataset", default="data/public_set.jsonl")
    app.add_argument("--limit", type=int, default=0)
    args = app.parse_args()

    from evaluator import local_evaluator as ev

    samples = ev.load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog_ids, categories, products = ev.catalog_index(args.catalog)
    catalog = Catalog(args.catalog)

    grid = {
        "PHRASE_HIT_WEIGHT": [8.0, 12.0, 18.0],
        "PROSE_TOKEN_WEIGHT": [0.5, 1.0],
        "POOL_SIZE": [200, 300, 500],
    }

    keys = list(grid.keys())
    results: list[tuple[float, dict]] = []
    combos = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
    print(f"Grid search over {len(combos)} combinations (limit={args.limit or 'all'})")
    for combo in combos:
        for k, v in combo.items():
            setattr(cfg, k, v)
        agent = Agent(args.catalog, catalog=catalog)
        res = ev.evaluate(agent, samples, catalog_ids, categories, products)
        score = res["recommended_technical_score"]
        results.append((score, dict(combo)))
        print(f"  score={score:.4f}  {combo}")

    best = max(results, key=lambda x: x[0])
    print("\nBEST:", best[1], "-> score", round(best[0], 4))


if __name__ == "__main__":
    main()