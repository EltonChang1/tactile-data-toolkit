"""CSV and raw-tensor (``.npy`` / ``.npz``) reader for custom lab rigs.

Each file (or each array inside an ``.npz``) becomes one topic named after the
file stem, e.g. ``taxels.csv`` -> ``/taxels``. Timestamps are taken from a time
column (CSV), a ``timestamps`` array (NPZ), a sidecar ``<stem>_timestamps.npy``
(NPY), or synthesized from ``CsvSpec.rate_hz``.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tactile_toolkit.ingest.base import BaseReader, UnsupportedLogError
from tactile_toolkit.types import ChannelInfo, Message

TIME_COLUMN_NAMES = (
    "t",
    "time",
    "timestamp",
    "stamp",
    "time_s",
    "timestamp_s",
    "time_ns",
    "timestamp_ns",
    "time_us",
    "timestamp_us",
)


@dataclass
class CsvSpec:
    """Optional hints describing how to interpret tabular / tensor files."""

    time_column: str | None = None
    rate_hz: float | None = None
    """Sampling rate used when no timestamps are available."""
    t0: float = 0.0
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    """Per-topic reshape of each row, e.g. ``{"/taxels": (16, 16)}``."""
    topic_names: dict[str, str] = field(default_factory=dict)
    """Override topic name per file stem."""
    delimiter: str = ","


class CsvReader(BaseReader):
    SUPPORTED_SUFFIXES = (".csv", ".npy", ".npz")

    def __init__(self, path: str | Path, spec: CsvSpec | None = None):
        self.path = Path(path)
        self.spec = spec or CsvSpec()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if self.path.is_dir():
            files = sorted(
                p for p in self.path.iterdir() if p.suffix.lower() in self.SUPPORTED_SUFFIXES
            )
        else:
            files = [self.path]
        if not files:
            raise UnsupportedLogError(f"No CSV/NPY/NPZ files found in {self.path}")
        self._sources: dict[str, tuple[Path, str | None]] = {}
        for f in files:
            suffix = f.suffix.lower()
            if suffix == ".npz":
                with np.load(f) as z:
                    for key in z.files:
                        if _is_time_key(key):
                            continue
                        self._sources[self._topic_for(key)] = (f, key)
            elif suffix == ".npy" and f.stem.endswith("_timestamps"):
                continue
            else:
                self._sources[self._topic_for(f.stem)] = (f, None)
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._channels: list[ChannelInfo] | None = None

    def _topic_for(self, stem: str) -> str:
        name = self.spec.topic_names.get(stem, stem)
        return name if name.startswith("/") else f"/{name}"

    # ------------------------------------------------------------------ loading
    def _load(self, topic: str) -> tuple[np.ndarray, np.ndarray]:
        if topic in self._cache:
            return self._cache[topic]
        path, key = self._sources[topic]
        suffix = path.suffix.lower()
        if suffix == ".csv":
            ts, data = self._load_csv(path)
        elif suffix == ".npz":
            with np.load(path) as z:
                data = np.asarray(z[key])
                ts = None
                for cand in (f"{key}_timestamps", f"{key}_t", "timestamps", "t", "time"):
                    if cand in z.files:
                        ts = _to_seconds(np.asarray(z[cand], dtype=np.float64))
                        break
        else:
            data = np.load(path)
            sidecar = path.with_name(f"{path.stem}_timestamps.npy")
            ts = _to_seconds(np.load(sidecar).astype(np.float64)) if sidecar.exists() else None
        data = np.asarray(data)
        if data.ndim == 1:
            data = data[:, None]
        shape = self.spec.shapes.get(topic)
        if shape is not None:
            data = data.reshape((data.shape[0], *shape))
        if ts is None:
            if self.spec.rate_hz is None:
                raise UnsupportedLogError(
                    f"{path}: no timestamps found; provide CsvSpec.rate_hz or a time column"
                )
            ts = self.spec.t0 + np.arange(data.shape[0], dtype=np.float64) / self.spec.rate_hz
        if ts.shape[0] != data.shape[0]:
            raise UnsupportedLogError(f"{path}: {ts.shape[0]} timestamps for {data.shape[0]} rows")
        self._cache[topic] = (ts, data)
        return ts, data

    def _load_csv(self, path: Path) -> tuple[np.ndarray | None, np.ndarray]:
        with open(path, newline="") as fh:
            reader = csv.reader(fh, delimiter=self.spec.delimiter)
            rows = [r for r in reader if r and any(c.strip() for c in r)]
        header = None
        if rows and not all(_is_number(c) for c in rows[0]):
            header = rows[0]
            rows = rows[1:]
        if not rows:
            return None, np.zeros((0, 0))
        arr = np.asarray(rows, dtype=np.float64)
        if header is None:
            header = [f"c{i}" for i in range(arr.shape[1])]
        header = [h.strip() for h in header]
        time_col = self.spec.time_column
        if time_col is None:
            for name in header:
                if name.lower() in TIME_COLUMN_NAMES:
                    time_col = name
                    break
        ts = None
        if time_col is not None and time_col in header:
            idx = header.index(time_col)
            raw = arr[:, idx]
            if time_col.lower().endswith("_ns"):
                ts = raw * 1e-9
            elif time_col.lower().endswith("_us"):
                ts = raw * 1e-6
            else:
                ts = _to_seconds(raw)
            arr = np.delete(arr, idx, axis=1)
        return ts, arr

    # ----------------------------------------------------------------- reader API
    def channels(self) -> list[ChannelInfo]:
        if self._channels is None:
            infos = []
            for topic in sorted(self._sources):
                ts, data = self._load(topic)
                infos.append(
                    ChannelInfo(
                        topic=topic,
                        msg_type=f"tensor/{data.dtype.name}",
                        message_count=int(data.shape[0]),
                        sample_shape=tuple(int(s) for s in data.shape[1:]),
                    )
                )
            self._channels = infos
        return list(self._channels)

    def iter_messages(self, topics: Iterable[str] | None = None) -> Iterator[Message]:
        wanted = [t for t in sorted(self._sources) if topics is None or t in set(topics)]
        loaded = {t: self._load(t) for t in wanted}
        # Merge streams in timestamp order to mimic a recorded log.
        cursors = {t: 0 for t in wanted}
        while True:
            best = None
            for t in wanted:
                ts, _ = loaded[t]
                i = cursors[t]
                if i < ts.shape[0] and (best is None or ts[i] < best[1]):
                    best = (t, ts[i])
            if best is None:
                return
            t, stamp = best
            ts, data = loaded[t]
            i = cursors[t]
            cursors[t] += 1
            yield Message(
                topic=t,
                msg_type=f"tensor/{data.dtype.name}",
                timestamp=float(stamp),
                data=data[i],
                log_time=float(stamp),
            )


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _is_time_key(key: str) -> bool:
    k = key.lower()
    return k in TIME_COLUMN_NAMES or k.endswith("_timestamps") or k.endswith("_t")


def _to_seconds(ts: np.ndarray) -> np.ndarray:
    """Heuristically convert epoch nanoseconds / microseconds / milliseconds to seconds."""
    if ts.size == 0:
        return ts
    mag = float(np.nanmax(np.abs(ts)))
    if mag > 1e17:
        return ts * 1e-9
    if mag > 1e14:
        return ts * 1e-6
    if mag > 1e11:
        return ts * 1e-3
    return ts


def load_spec(mapping: Mapping[str, object]) -> CsvSpec:
    """Build a :class:`CsvSpec` from a plain mapping (e.g. parsed JSON)."""
    shapes = {str(k): tuple(int(x) for x in v) for k, v in dict(mapping.get("shapes", {})).items()}  # type: ignore[union-attr]
    return CsvSpec(
        time_column=mapping.get("time_column"),  # type: ignore[arg-type]
        rate_hz=mapping.get("rate_hz"),  # type: ignore[arg-type]
        t0=float(mapping.get("t0", 0.0)),  # type: ignore[arg-type]
        shapes=shapes,
        topic_names={str(k): str(v) for k, v in dict(mapping.get("topic_names", {})).items()},  # type: ignore[union-attr]
        delimiter=str(mapping.get("delimiter", ",")),
    )
