"""Inspect a bounded prefix of a local FreeTacMan snapshot."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

from tactile_toolkit.datasets import FreeTacManAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="snapshot root or one FreeTacMan task directory")
    parser.add_argument("--limit", type=int, default=2, help="maximum trajectories to inspect")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")

    dataset = FreeTacManAdapter(args.root)
    report = dataset.validate(split="train", limit=args.limit, check_assets=True)
    report.raise_if_invalid()
    print(report.summary())
    for sample in islice(dataset.iter_samples(split="train"), args.limit):
        print(sample.sample_id, sorted(item.value for item in sample.modalities))


if __name__ == "__main__":
    main()
