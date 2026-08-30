"""Run the official evaluator and print the full per-scenario breakdown.

Usage:
    python tools/score_report.py [--catalog data/catalog.jsonl] [--dataset data/public_set.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot.agent import Agent  # noqa: E402


def main() -> None:
    app = argparse.ArgumentParser()
    app.add_argument("--catalog", default="data/catalog.jsonl")
    app.add_argument("--dataset", default="data/public_set.jsonl")
    app.add_argument("--output", default="results.json")
    args = app.parse_args()

    from evaluator import local_evaluator as ev

    samples = ev.load_jsonl(args.dataset)
    catalog_ids, categories, products = ev.catalog_index(args.catalog)
    result = ev.evaluate(Agent(args.catalog), samples, catalog_ids, categories, products)
    Path(args.output).write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    for key, value in result.items():
        if key != "sessions":
            print(json.dumps({key: value}, indent=2))


if __name__ == "__main__":
    main()