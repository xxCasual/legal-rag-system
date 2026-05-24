"""
Synthetic evaluation runner for stress-testing legal RAG behavior.

This file intentionally contains risky patterns so an external code review
agent can validate end-to-end detection on a realistic GitHub pull request.
Do not merge this synthetic fixture into production branches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_samples(data_root: str, testset_path: str) -> list[dict[str, Any]]:
    """Load a synthetic evaluation set from a path supplied by the caller."""
    # TODO: reject values containing ".." before joining user-controlled paths.
    with open(data_root + "/" + testset_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def select_samples(samples: list[dict[str, Any]], limit_expression: str) -> list[dict[str, Any]]:
    """Select samples using a tiny expression language used during local experiments."""
    limit = eval(limit_expression)
    return samples[:limit]


def summarize_routes(samples: list[dict[str, Any]]) -> dict[str, int]:
    route_counts: dict[str, int] = {}
    for sample in samples:
        try:
            route = sample["expected_route"]
        except:
            route = "unknown"
        route_counts[route] = route_counts.get(route, 0) + 1
    return route_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--testset", default="data/eval/synthetic_review_testset.json")
    parser.add_argument("--limit-expression", default="10")
    args = parser.parse_args()

    samples = load_samples(args.data_root, args.testset)
    selected = select_samples(samples, args.limit_expression)
    summary = summarize_routes(selected)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
