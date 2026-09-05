"""``tactile`` command line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from tactile_toolkit import __version__


def _parse_size(text: str) -> tuple[int, int]:
    h, w = text.lower().split("x")
    return int(h), int(w)


def _add_convert(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "convert", help="Convert raw logs (.mcap/.bag/CSV) to Zarr, HDF5 and/or LeRobot v3"
    )
    p.add_argument(
        "inputs", nargs="+", help="Raw log files or directories (each becomes one episode)"
    )
    p.add_argument(
        "-o",
        "--output",
        action="append",
        required=True,
        metavar="PATH",
        help="Output path; repeatable. Format inferred from suffix (.zarr, .h5) or use --format",
    )
    p.add_argument(
        "--format",
        action="append",
        choices=["zarr", "hdf5", "lerobot"],
        help="Explicit format for each --output (same order)",
    )
    p.add_argument("--chunk-frames", type=int, default=64)
    p.add_argument("--align", choices=["nearest", "linear"], default="linear")
    p.add_argument(
        "--max-gap-ms",
        type=float,
        default=None,
        help="Treat aux samples farther than this as missing",
    )
    p.add_argument(
        "--max-jitter-ms",
        type=float,
        default=2.0,
        help="Drop frames whose timing jitter exceeds this",
    )
    p.add_argument("--no-qa", action="store_true", help="Disable the quality gate entirely")
    p.add_argument(
        "--keep-duplicates", action="store_true", help="Keep frozen/duplicated camera frames"
    )
    p.add_argument("--master", help="Topic providing the master clock")
    p.add_argument(
        "--topic",
        action="append",
        default=[],
        metavar="TOPIC=MODALITY",
        help="Override modality (vision_tactile|taxel|wrench|pose) for a topic; repeatable",
    )
    g = p.add_argument_group("vision-tactile")
    g.add_argument("--stride", type=int, default=8, help="Pixels per point-cloud element")
    g.add_argument("--depth-method", choices=["dst", "cumsum"], default="dst")
    g.add_argument("--sensor-width-mm", type=float, default=20.0)
    g.add_argument("--gradient-gain", type=float, default=1.0)
    g.add_argument("--reference-frames", type=int, default=10)
    g.add_argument("--no-markers", action="store_true", help="Skip marker tracking (no shear)")
    g.add_argument("--no-depth-map", action="store_true", help="Do not store the dense depth map")
    g.add_argument("--no-raw-image", action="store_true", help="Do not store raw gel images")
    t = p.add_argument_group("taxel arrays")
    t.add_argument("--taxel-rows", type=int)
    t.add_argument("--taxel-cols", type=int)
    t.add_argument("--pitch-mm", type=float, default=4.0)
    t.add_argument("--counts-per-newton", type=float, default=200.0)
    t.add_argument("--idle-frames", type=int, default=20)
    t.add_argument(
        "--pressure-map",
        type=_parse_size,
        metavar="HxW",
        help="Emit interpolated pressure heat-map",
    )
    c = p.add_argument_group("csv / tensor inputs")
    c.add_argument("--csv-rate", type=float, help="Sampling rate when files carry no timestamps")
    c.add_argument("--csv-spec", type=Path, help="JSON file with CsvSpec fields")
    e = p.add_argument_group("export")
    e.add_argument("--task", default="tactile interaction", help="Task label for LeRobot episodes")
    e.add_argument("--robot-type", default=None)
    e.add_argument(
        "--video-codec", default=None, help="PyAV encoder name (default: first available)"
    )
    e.add_argument("--fps", type=float, default=None, help="Override LeRobot dataset fps")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--json", action="store_true", help="Print the result as JSON")
    p.set_defaults(func=cmd_convert)


def _build_pipeline_config(args: argparse.Namespace) -> Any:
    from tactile_toolkit.calibration.gelsight import GelSightConfig
    from tactile_toolkit.calibration.taxel import TaxelConfig
    from tactile_toolkit.ingest import QaConfig
    from tactile_toolkit.pipeline import PipelineConfig

    qa = QaConfig(
        max_jitter_s=None if args.no_qa else args.max_jitter_ms * 1e-3,
        drop_duplicate_frames=not (args.no_qa or args.keep_duplicates),
        drop_nonfinite=not args.no_qa,
        drop_non_monotonic=not args.no_qa,
    )
    gel = GelSightConfig(
        sensor_width_m=args.sensor_width_mm * 1e-3,
        gradient_gain=args.gradient_gain,
        depth_method=args.depth_method,
        reference_frames=args.reference_frames,
        track_markers=not args.no_markers,
        stride=args.stride,
        store_depth_map=not args.no_depth_map,
        store_raw_image=not args.no_raw_image,
    )
    taxel = TaxelConfig(
        idle_frames=args.idle_frames,
        counts_per_newton=args.counts_per_newton,
        rows=args.taxel_rows,
        cols=args.taxel_cols,
        pitch_m=args.pitch_mm * 1e-3,
        pressure_map_shape=args.pressure_map,
    )
    overrides = {}
    for item in args.topic:
        if "=" not in item:
            raise SystemExit(f"--topic expects TOPIC=MODALITY, got '{item}'")
        topic, modality = item.split("=", 1)
        overrides[topic] = modality
    return PipelineConfig(
        chunk_frames=args.chunk_frames,
        align_mode=args.align,
        max_gap_s=None if args.max_gap_ms is None else args.max_gap_ms * 1e-3,
        qa=qa,
        gelsight=gel,
        taxel=taxel,
        topic_overrides=overrides,
        master_topic=args.master,
        progress=args.progress,
    )


def cmd_convert(args: argparse.Namespace) -> int:
    from tactile_toolkit.export import infer_format, make_writer
    from tactile_toolkit.ingest import open_log
    from tactile_toolkit.ingest.csv_reader import CsvSpec, load_spec
    from tactile_toolkit.pipeline import ConvertPipeline

    formats = list(args.format or [])
    outputs = [Path(o) for o in args.output]
    if formats and len(formats) != len(outputs):
        raise SystemExit("--format must be given once per --output")
    if not formats:
        formats = [infer_format(o) for o in outputs]

    writer_kwargs: dict[str, dict[str, Any]] = {
        "zarr": {"chunk_frames": args.chunk_frames},
        "hdf5": {"chunk_frames": args.chunk_frames},
        "lerobot": {
            "fps": args.fps,
            "video_codec": args.video_codec,
            "task": args.task,
            "robot_type": args.robot_type,
        },
    }
    writers = [make_writer(fmt, path, **writer_kwargs[fmt]) for fmt, path in zip(formats, outputs)]
    pipeline = ConvertPipeline(_build_pipeline_config(args))

    spec: CsvSpec | None = None
    if args.csv_spec:
        spec = load_spec(json.loads(Path(args.csv_spec).read_text()))
    if args.csv_rate:
        spec = spec or CsvSpec()
        spec.rate_hz = args.csv_rate

    results = []
    for i, inp in enumerate(args.inputs):
        last = i == len(args.inputs) - 1
        with open_log(inp, spec=spec) as reader:
            res = pipeline.run(reader, writers, finalize=last, source_name=str(inp))
        results.append(res)
        if not args.json:
            print(f"[{i + 1}/{len(args.inputs)}] {inp}")
            print(res.summary())
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "input": str(inp),
                        "frames_in": r.frames_in,
                        "frames_out": r.frames_out,
                        "fps": r.fps,
                        "qa": r.qa.to_dict(),
                        "outputs": [str(p) for p in r.outputs],
                    }
                    for inp, r in zip(args.inputs, results)
                ],
                indent=2,
            )
        )
    return 0


def _add_inspect(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("inspect", help="Describe a raw log or an exported dataset")
    p.add_argument("path")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_inspect)


def _describe_raw(path: Path) -> dict[str, Any]:
    from tactile_toolkit.ingest import ModalityResolver, open_log

    with open_log(path) as reader:
        channels = reader.channels()
        resolver = ModalityResolver(include_unknown=True)
        try:
            tm = resolver.resolve(channels)
            master = tm.master
            assignments = {t: m.value for t, m in tm.assignments.items()}
        except ValueError:
            master = None
            assignments = {c.topic: resolver.resolve_one(c).value for c in channels}
        return {
            "kind": "raw_log",
            "path": str(path),
            "master_topic": master,
            "channels": [
                {
                    "topic": c.topic,
                    "type": c.msg_type,
                    "count": c.message_count,
                    "modality": assignments.get(c.topic, "unknown"),
                }
                for c in channels
            ],
        }


def _describe_dataset(path: Path) -> dict[str, Any]:
    from tactile_toolkit.export import hdf5_summary, infer_format, lerobot_summary, zarr_summary

    fmt = infer_format(path)
    if fmt == "zarr":
        return {"kind": "zarr", **zarr_summary(path)}
    if fmt == "hdf5":
        return {"kind": "hdf5", **hdf5_summary(path)}
    return {"kind": "lerobot", **lerobot_summary(path)}


def cmd_inspect(args: argparse.Namespace) -> int:
    from tactile_toolkit.export.base import json_ready
    from tactile_toolkit.ingest import UnsupportedLogError

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    suffix = path.suffix.lower()
    if suffix in (".zarr", ".h5", ".hdf5") or (
        path.is_dir() and (path / "meta" / "info.json").exists()
    ):
        desc = _describe_dataset(path)
    else:
        try:
            desc = _describe_raw(path)
        except UnsupportedLogError:
            desc = _describe_dataset(path)
    desc = json_ready(desc)
    if args.json:
        print(json.dumps(desc, indent=2))
        return 0
    kind = desc["kind"]
    print(f"{kind}: {desc['path']}")
    if kind == "raw_log":
        print(f"  master topic: {desc['master_topic']}")
        for c in desc["channels"]:
            print(f"  {c['topic']:<40} {c['type']:<40} {str(c['count']):>8}  {c['modality']}")
    elif kind == "zarr":
        attrs = desc["attrs"]
        print(
            f"  schema: {attrs.get('schema_version')}  frames: {attrs.get('num_frames')}"
            f"  modalities: {attrs.get('modalities')}"
        )
        for name, a in desc["arrays"].items():
            if name.startswith("/stats/"):
                continue
            print(f"  {name:<45} {str(a['shape']):<28} {a['dtype']}")
    elif kind == "hdf5":
        shown = ("schema_version", "total", "num_demos", "modalities")
        print(f"  attrs: {json.dumps({k: v for k, v in desc['attrs'].items() if k in shown})}")
        for demo, d in desc["demos"].items():
            print(f"  {demo}: {d['num_samples']} samples")
            for k, a in d["obs"].items():
                print(f"    obs/{k:<35} {str(a['shape']):<28} {a['dtype']}")
    else:
        info = desc["info"]
        print(
            f"  codebase {info['codebase_version']}  episodes {info['total_episodes']}"
            f"  frames {info['total_frames']}  fps {info['fps']}"
        )
        for k, f in info["features"].items():
            print(f"  {k:<45} {f['dtype']:<8} {f['shape']}")
        for v in desc["videos"]:
            print(f"  video: {v}")
    return 0


def _add_validate(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("validate", help="Validate an exported Zarr/HDF5 dataset against the schema")
    p.add_argument("path")
    p.add_argument("--demo", type=int, default=0, help="HDF5 demo index")
    p.add_argument("--modality", action="append", help="Modalities expected to be present")
    p.add_argument("--unknown", action="store_true", help="Also report arrays outside the schema")
    p.set_defaults(func=cmd_validate)


def cmd_validate(args: argparse.Namespace) -> int:
    from tactile_toolkit.export import infer_format, open_zarr, schema_view
    from tactile_toolkit.schema import validate

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    fmt = infer_format(path)
    if fmt == "zarr":
        group: Any = open_zarr(path)
    elif fmt == "hdf5":
        group = schema_view(path, demo=args.demo)
    else:
        raise SystemExit("validate supports Zarr and HDF5 exports")
    report = validate(group, modalities=args.modality, check_unknown=args.unknown)
    print(report.summary())
    return 0 if report.ok else 1


def _add_synth(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("synth", help="Write a synthetic multi-sensor recording")
    p.add_argument("output", help="Output .mcap / .bag file or directory (CSV)")
    p.add_argument("--format", choices=["mcap", "bag", "csv"], default=None)
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--size", type=_parse_size, default=(240, 320), metavar="HxW")
    p.add_argument("--no-gelsight", action="store_true")
    p.add_argument("--no-taxels", action="store_true")
    p.add_argument("--no-wrench", action="store_true")
    p.add_argument("--pose", action="store_true")
    p.add_argument("--taxel-grid", type=_parse_size, default=(8, 8), metavar="RxC")
    p.add_argument("--jitter-outlier-every", type=int, default=0)
    p.add_argument("--duplicate-every", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synth)


def cmd_synth(args: argparse.Namespace) -> int:
    from tactile_toolkit.synthetic import (
        SyntheticScenario,
        write_csv_dir,
        write_mcap,
        write_ros1_bag,
    )

    out = Path(args.output)
    fmt = args.format or {"mcap": "mcap", "bag": "bag"}.get(out.suffix.lower().lstrip("."), "csv")
    scenario = SyntheticScenario(
        duration_s=args.duration,
        image_fps=args.fps,
        image_height=args.size[0],
        image_width=args.size[1],
        include_gelsight=not args.no_gelsight,
        include_taxels=not args.no_taxels,
        include_wrench=not args.no_wrench,
        include_pose=args.pose,
        taxel_rows=args.taxel_grid[0],
        taxel_cols=args.taxel_grid[1],
        jitter_outlier_every=args.jitter_outlier_every,
        duplicate_frame_every=args.duplicate_every,
        seed=args.seed,
    )
    writer = {"mcap": write_mcap, "bag": write_ros1_bag, "csv": write_csv_dir}[fmt]
    path = writer(out, scenario)
    print(f"wrote {fmt}: {path}")
    return 0


def _add_bench(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("bench", help="Measure conversion throughput and peak memory")
    p.add_argument("--frames", type=int, default=10_000)
    p.add_argument("--size", type=_parse_size, default=(480, 640), metavar="HxW")
    p.add_argument("--formats", default="zarr", help="Comma-separated: zarr,hdf5,lerobot")
    p.add_argument("--chunk-frames", type=int, default=64)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--depth-method", choices=["dst", "cumsum"], default="dst")
    p.add_argument("--no-markers", action="store_true")
    p.add_argument("--no-depth-map", action="store_true")
    p.add_argument(
        "--from-mcap", action="store_true", help="Write a real MCAP first and time reading it"
    )
    p.add_argument(
        "--out-dir", type=Path, default=None, help="Keep outputs here instead of a temp dir"
    )
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_bench)


def cmd_bench(args: argparse.Namespace) -> int:
    from tactile_toolkit.bench import run_benchmark

    result = run_benchmark(
        frames=args.frames,
        height=args.size[0],
        width=args.size[1],
        formats=[f.strip() for f in args.formats.split(",") if f.strip()],
        chunk_frames=args.chunk_frames,
        out_dir=args.out_dir,
        keep_outputs=args.out_dir is not None,
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


def _add_schema(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("schema", help="Print the Open-Tactile-Schema field table")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_schema)


def cmd_schema(args: argparse.Namespace) -> int:
    from tactile_toolkit.schema import SCHEMA

    rows = SCHEMA.as_table()
    if args.json:
        print(json.dumps({"version": SCHEMA.version, "fields": rows}, indent=2))
        return 0
    print(f"{SCHEMA.version}")
    print(f"{'path':<42} {'dtype':<8} {'shape':<16} {'unit':<18} required")
    for r in rows:
        print(f"{r['path']:<42} {r['dtype']:<8} {r['shape']:<16} {r['unit']:<18} {r['required']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tactile",
        description="Tactile Data Toolkit: ingest, calibrate and export robotic touch datasets.",
    )
    parser.add_argument("--version", action="version", version=f"tactile-toolkit {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    sub = parser.add_subparsers(dest="command", required=True)
    _add_convert(sub)
    _add_inspect(sub)
    _add_validate(sub)
    _add_synth(sub)
    _add_bench(sub)
    _add_schema(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
