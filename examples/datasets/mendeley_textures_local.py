"""Inspect trusted pressure/IMU pairs from extracted Mendeley texture data."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

from tactile_toolkit.datasets import MendeleyTexturesAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory containing extracted pickles_* folders")
    parser.add_argument(
        "--trust-pickle",
        action="store_true",
        help="allow pickle deserialization after verifying the official V1 SHA-256",
    )
    parser.add_argument("--limit", type=int, default=2, help="maximum paired trials to inspect")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if not args.trust_pickle:
        parser.error("--trust-pickle is required after verifying the official archive")

    dataset = MendeleyTexturesAdapter(args.root, trust_pickle=True)
    samples = list(islice(dataset.iter_samples(), args.limit))
    if not samples:
        raise SystemExit("No paired full_baro/full_imu trials found")
    for sample in samples:
        print(
            sample.sample_id,
            sample.labels["exploration_speed_mm_s"],
            sample.observations["pressure.barometer"].data.shape,
            sample.observations["imu"].data.shape,
        )


if __name__ == "__main__":
    main()
