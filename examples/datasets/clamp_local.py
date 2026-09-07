"""Inspect trusted CLAMP filtered NPZ contacts with object-safe grouping."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

from tactile_toolkit.datasets import ClampFilteredAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="filtered NPZ or its containing directory")
    parser.add_argument(
        "--trust-pickle",
        action="store_true",
        help="allow object-array deserialization after verifying the official MD5",
    )
    parser.add_argument("--limit", type=int, default=2, help="maximum contacts to inspect")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if not args.trust_pickle:
        parser.error("--trust-pickle is required after verifying the official file")

    dataset = ClampFilteredAdapter(args.root, trust_pickle=True)
    samples = list(islice(dataset.iter_samples(), args.limit))
    if not samples:
        raise SystemExit("No CLAMP samples found")
    for sample in samples:
        print(sample.sample_id, sample.group_id, sorted(item.value for item in sample.modalities))


if __name__ == "__main__":
    main()
