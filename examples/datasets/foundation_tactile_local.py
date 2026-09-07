"""Inspect a bounded prefix of extracted FoTa WebDataset shards."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

from tactile_toolkit.datasets import FoundationTactileAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="extracted FoTa root or constituent directory")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--source", action="append", dest="sources")
    parser.add_argument("--limit", type=int, default=2, help="maximum records to inspect")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")

    dataset = FoundationTactileAdapter(args.root, sources=args.sources)
    report = dataset.validate(split=args.split, limit=args.limit, check_assets=True)
    report.raise_if_invalid()
    print(report.summary())
    for sample in islice(dataset.iter_samples(split=args.split), args.limit):
        reference = sample.observations["touch"].data
        print(sample.sample_id, sample.task, reference)


if __name__ == "__main__":
    main()
