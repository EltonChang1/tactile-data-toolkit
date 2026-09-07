"""Inspect Penn HaTT recorded-data XML from a licensed local copy."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

from tactile_toolkit.datasets import PennHattAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory containing Penn HaTT XML recordings")
    parser.add_argument("--limit", type=int, default=2, help="maximum recordings to inspect")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")

    dataset = PennHattAdapter(args.root)
    samples = list(islice(dataset.iter_samples(), args.limit))
    if not samples:
        raise SystemExit("No Penn HaTT recorded-data XML found")
    for sample in samples:
        shapes = {
            name: observation.data.shape
            for name, observation in sample.observations.items()
            if hasattr(observation.data, "shape")
        }
        print(sample.sample_id, sample.labels["material"], shapes)


if __name__ == "__main__":
    main()
