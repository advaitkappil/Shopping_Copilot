"""Sweep ASK_ORDER (and other knobs) against the official evaluator.

Reuses a single Catalog to make many agent configs cheap to evaluate.

Usage:
    python tools/sweep_askorder.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot.agent import Agent  # noqa: E402
from copilot.corpus import Catalog  # noqa: E402


def main() -> None:
    app = argparse.ArgumentParser()
    app.add_argument("--catalog", default="data/catalog.jsonl")
    app.add_argument("--dataset", default="data/public_set.jsonl")
    args = app.parse_args()

    from evaluator import local_evaluator as ev

    samples = ev.load_jsonl(args.dataset)
    catalog_ids, categories, products = ev.catalog_index(args.catalog)

    catalog = Catalog(args.catalog)

    variants = {
        "material/color/feature": ["material", "color", "size", "style", "feature", "use_case", "budget", "brand", "category", "other"],
        "other-first": ["other", "feature", "material", "color", "size", "style", "budget", "use_case"],
        "feature-first": ["feature", "material", "color", "size", "style", "use_case", "budget", "other"],
        "color-size-material": ["size", "color", "material", "feature", "style", "budget", "use_case", "other"],
        "budget-early": ["feature", "budget", "color", "material", "size", "style", "use_case", "other"],
        "material-feature-other": ["material", "feature", "other", "color", "size", "style", "budget"],
        "size-feature-other": ["feature", "size", "material", "color", "style", "budget", "other"],
    }

    print(f"{'variant':<28}{'HitRate':>9}{'MRR':>10}{'MTTC':>8}{'Eff':>7}{'Score':>9}")
    for name, order in variants.items():
        import copilot.config as cfg

        cfg.ASK_ORDER = list(order)
        agent = Agent(args.catalog, catalog=catalog)
        res = ev.evaluate(agent, samples, catalog_ids, categories, products)
        print(
            f"{name:<28}{res['hit_rate_at_10']:>9.3f}{res['mrr']:>10.3f}"
            f"{res['mttc']:>8.2f}{res['efficiency']:>7.3f}{res['recommended_technical_score']:>9.3f}"
        )


if __name__ == "__main__":
    main()