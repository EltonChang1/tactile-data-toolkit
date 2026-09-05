"""Dispatch a path to the matching reader backend."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from tactile_toolkit.ingest.base import BaseReader, UnsupportedLogError

ReaderFactory = Callable[..., BaseReader]

_REGISTRY: dict[str, ReaderFactory] = {}


def register_reader(suffix: str, factory: ReaderFactory) -> None:
    _REGISTRY[suffix.lower()] = factory


def _mcap(path: Path, **kw: Any) -> BaseReader:
    from tactile_toolkit.ingest.mcap_reader import McapReader

    return McapReader(path)


def _bag(path: Path, **kw: Any) -> BaseReader:
    from tactile_toolkit.ingest.bag_reader import BagReader

    return BagReader(path)


def _csv(path: Path, **kw: Any) -> BaseReader:
    from tactile_toolkit.ingest.csv_reader import CsvReader

    return CsvReader(path, spec=kw.get("spec"))


register_reader(".mcap", _mcap)
register_reader(".bag", _bag)
register_reader(".db3", _bag)
register_reader(".csv", _csv)
register_reader(".npy", _csv)
register_reader(".npz", _csv)


def detect_format(path: str | Path) -> str:
    """Return the registry key (file suffix or ``"rosbag2"`` / ``"csv-dir"``) for ``path``."""
    p = Path(path)
    if p.is_dir():
        if (p / "metadata.yaml").exists():
            return "rosbag2"
        if any(c.suffix.lower() in (".csv", ".npy", ".npz") for c in p.iterdir()):
            return "csv-dir"
        raise UnsupportedLogError(f"Directory {p} is neither a rosbag2 bag nor a CSV/tensor folder")
    suffix = p.suffix.lower()
    if suffix in _REGISTRY:
        return suffix
    raise UnsupportedLogError(f"Unsupported log format: {p}")


def open_log(path: str | Path, **kwargs: Any) -> BaseReader:
    """Open any supported raw log (``.mcap``, ``.bag``, rosbag2 dir, CSV/NPY/NPZ file or dir)."""
    p = Path(path)
    fmt = detect_format(p)
    if fmt == "rosbag2":
        return _bag(p, **kwargs)
    if fmt == "csv-dir":
        return _csv(p, **kwargs)
    return _REGISTRY[fmt](p, **kwargs)
