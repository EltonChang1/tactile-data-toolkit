"""Inspect a bounded prefix of a user-obtained Touch100k release."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

from tactile_toolkit.datasets import Touch100kAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="directory containing data_list.json plus touch/ and vision/",
    )
    parser.add_argument("--manifest", default="data_list.json", help="manifest path under root")
    parser.add_argument("--limit", type=int, default=2, help="maximum records to inspect")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")

    dataset = Touch100kAdapter(args.root, manifest=args.manifest)
    report = dataset.validate(split="train", limit=args.limit, check_assets=True)
    report.raise_if_invalid()
    print(report.summary())
    for sample in islice(dataset.iter_samples(split="train"), args.limit):
        print(sample.sample_id, sample.group_id, sorted(item.value for item in sample.modalities))


if __name__ == "__main__":
    main()
