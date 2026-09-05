"""Dataset writers: Zarr, HDF5 (robomimic layout), and LeRobot v3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tactile_toolkit.export.base import BaseWriter, DatasetInfo, json_ready
from tactile_toolkit.export.hdf5_writer import Hdf5Writer, hdf5_summary, schema_view
from tactile_toolkit.export.lerobot_writer import LeRobotV3Writer, lerobot_summary
from tactile_toolkit.export.stats import RunningStats, StatsCollector, stats_to_json
from tactile_toolkit.export.zarr_writer import ZarrWriter, open_zarr, zarr_summary

FORMATS = ("zarr", "hdf5", "lerobot")


def make_writer(fmt: str, path: str | Path, **kwargs: Any) -> BaseWriter:
    """Instantiate a writer by format name (``zarr`` | ``hdf5`` | ``lerobot``)."""
    fmt = fmt.lower()
    if fmt == "zarr":
        return ZarrWriter(path, **kwargs)
    if fmt in ("hdf5", "h5"):
        return Hdf5Writer(path, **kwargs)
    if fmt in ("lerobot", "lerobot_v3", "lerobotv3"):
        return LeRobotV3Writer(path, **kwargs)
    raise ValueError(f"Unknown export format '{fmt}'; expected one of {FORMATS}")


def infer_format(path: str | Path) -> str:
    """Guess the export format from an output path."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".zarr":
        return "zarr"
    if suffix in (".h5", ".hdf5"):
        return "hdf5"
    if suffix in ("", ".lerobot") or (p.is_dir() and (p / "meta" / "info.json").exists()):
        return "lerobot"
    raise ValueError(f"Cannot infer export format from '{path}'; pass it explicitly")


__all__ = [
    "FORMATS",
    "BaseWriter",
    "DatasetInfo",
    "Hdf5Writer",
    "LeRobotV3Writer",
    "RunningStats",
    "StatsCollector",
    "ZarrWriter",
    "hdf5_summary",
    "infer_format",
    "json_ready",
    "lerobot_summary",
    "make_writer",
    "open_zarr",
    "schema_view",
    "stats_to_json",
    "zarr_summary",
]
