"""Standalone conversion throughput / memory benchmark.

Targets (from the project verification checklist):
- more than 120 frames/s on a standard workstation
- peak RSS under 4 GB for a 10,000-frame trajectory
"""

from __future__ import annotations

import argparse
import json
import sys

from tactile_toolkit.bench import run_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=10_000)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--formats", default="zarr")
    parser.add_argument("--chunk-frames", type=int, default=64)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--depth-method", choices=["dst", "cumsum"], default="dst")
    parser.add_argument("--no-markers", action="store_true")
    parser.add_argument("--no-depth-map", action="store_true")
    parser.add_argument("--from-mcap", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = run_benchmark(
        frames=args.frames,
        height=args.height,
        width=args.width,
        formats=[f.strip() for f in args.formats.split(",") if f.strip()],
        chunk_frames=args.chunk_frames,
        track_markers=not args.no_markers,
        depth_method=args.depth_method,
        stride=args.stride,
        store_depth_map=not args.no_depth_map,
        from_mcap=args.from_mcap,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.summary())
    return 0 if all(result.targets.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
