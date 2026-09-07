"""Inspect trusted local TacBench force/slip or pose pickle data."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

from tactile_toolkit.datasets import TacBenchForceSlipAdapter, TacBenchPoseAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="downloaded dataset root, batch, or pose bag")
    parser.add_argument("--task", choices=("force-slip", "pose"), required=True)
    parser.add_argument("--sensor", choices=("digit", "gelsight_mini"))
    parser.add_argument("--finger", choices=("index", "middle", "ring"), default="index")
    parser.add_argument(
        "--trust-pickle",
        action="store_true",
        help="allow code-capable pickle deserialization after verifying the source",
    )
    parser.add_argument("--limit", type=int, default=2, help="maximum frames to inspect")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if not args.trust_pickle:
        parser.error("--trust-pickle is required after verifying the official files")

    if args.task == "force-slip":
        dataset = TacBenchForceSlipAdapter(
            args.root,
            sensor=args.sensor,
            trust_pickle=True,
        )
    else:
        dataset = TacBenchPoseAdapter(args.root, finger=args.finger, trust_pickle=True)
    samples = list(islice(dataset.iter_samples(), args.limit))
    if not samples:
        raise SystemExit("No TacBench samples found")
    for sample in samples:
        print(sample.sample_id, sample.group_id, sorted(item.value for item in sample.modalities))


if __name__ == "__main__":
    main()
