"""Inspect a bounded prefix of an extracted, user-authorized TacVerse snapshot."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

from tactile_toolkit.datasets import TacVerseAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory containing extracted task archives")
    parser.add_argument(
        "--task",
        action="append",
        choices=("force", "grating", "shape"),
        help="task to inspect; repeat to select more than one",
    )
    parser.add_argument("--limit", type=int, default=2, help="maximum frames to inspect")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")

    dataset = TacVerseAdapter(args.root, tasks=args.task)
    samples = list(islice(dataset.iter_samples(), args.limit))
    if not samples:
        raise SystemExit("No TacVerse samples found")
    for sample in samples:
        print(sample.sample_id, sample.group_id, sorted(item.value for item in sample.modalities))


if __name__ == "__main__":
    main()
