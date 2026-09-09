"""HDF5 export in the robomimic / ACT trajectory layout.

Layout::

    /data                      attrs: total, schema_version, sensors, ...
    /data/demo_0               attrs: num_samples, schema_paths
    /data/demo_0/obs/<key>     (T, ...) datasets, key = schema path with '/' -> '_'
    /data/demo_0/actions       (T, 7) canonical RT-X action when present
    /data/demo_0/timestamps    (T,) float64
    /stats/<key>/{mean,std,min,max}

Each dataset carries a ``schema_path`` attribute so the Open-Tactile-Schema
validator can be run through :func:`schema_view`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from tactile_toolkit import schema as S
from tactile_toolkit.export.base import BaseWriter, DatasetInfo, json_ready
from tactile_toolkit.types import TactileChunk, normalize_path


def obs_key(schema_path: str) -> str:
    """``/observation/tactile/normal_force`` -> ``tactile_normal_force``."""
    rel = normalize_path(schema_path).lstrip("/")
    if rel.startswith("observation/"):
        rel = rel[len("observation/") :]
    return rel.replace("/", "_")


class Hdf5Writer(BaseWriter):
    """robomimic / ACT style HDF5: ``/data/demo_N/obs/<key>`` plus ``timestamps``.

    Compression defaults to ``lzf``, which is roughly 5x faster than gzip
    inside HDF5's single-threaded filter pipeline and is available in every
    h5py build (which is what robomimic, ACT and LeRobot's HDF5 importers
    use). Pass ``compression="gzip"`` for maximum portability to non-h5py
    readers, or ``None`` to store raw chunks.
    """

    format_name = "hdf5"

    def __init__(
        self,
        path: str | Path,
        chunk_frames: int = 64,
        compression: str | None = "lzf",
        compression_level: int = 3,
        overwrite: bool = True,
    ):
        super().__init__(path)
        self.chunk_frames = max(1, chunk_frames)
        self.compression = compression
        self.compression_level = compression_level
        if self.path.exists() and not overwrite:
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "w")
        self._data = self._file.create_group("data")
        self._demo: h5py.Group | None = None
        self._datasets: dict[str, h5py.Dataset] = {}
        self._episode_index = 0

    def _begin_episode(self) -> None:
        self._demo = self._data.create_group(f"demo_{self._episode_index}")
        self._demo.create_group("obs")
        self._datasets = {}

    def _dataset_for(self, key: str, sample: np.ndarray) -> h5py.Dataset:
        ds = self._datasets.get(key)
        if ds is not None:
            return ds
        assert self._demo is not None
        sample_shape = tuple(int(s) for s in sample.shape[1:])
        if key == S.TIMESTAMPS:
            parent, name = self._demo, "timestamps"
        elif key == S.ACTION:
            parent, name = self._demo, "actions"
        else:
            parent, name = self._demo["obs"], obs_key(key)
        chunk_t = self.chunk_frames
        frame_bytes = max(1, int(np.prod(sample_shape)) * sample.dtype.itemsize)
        chunk_t = max(1, min(chunk_t, (8 * 1024 * 1024) // frame_bytes))
        kwargs: dict[str, Any] = {}
        if self.compression:
            kwargs["compression"] = self.compression
            if self.compression == "gzip":
                kwargs["compression_opts"] = self.compression_level
        ds = parent.create_dataset(
            name,
            shape=(0, *sample_shape),
            maxshape=(None, *sample_shape),
            chunks=(chunk_t, *sample_shape),
            dtype=sample.dtype,
            **kwargs,
        )
        ds.attrs["schema_path"] = key
        if key in S.SCHEMA:
            ds.attrs["unit"] = S.SCHEMA[key].unit
        self._datasets[key] = ds
        return ds

    def _append(self, chunk: TactileChunk) -> None:
        for key, value in chunk.items():
            value = np.asarray(value)
            ds = self._dataset_for(normalize_path(key), value)
            n0 = ds.shape[0]
            ds.resize(n0 + value.shape[0], axis=0)
            ds[n0:] = value

    def _end_episode(self) -> None:
        assert self._demo is not None
        self._demo.attrs["num_samples"] = self.episode_frames
        self._demo.attrs["schema_paths"] = json.dumps(
            {
                ("actions" if k == S.ACTION else obs_key(k)): k
                for k in self._datasets
                if k != S.TIMESTAMPS
            }
        )
        self._episode_index += 1

    def _finalize(self, info: DatasetInfo) -> None:
        self._data.attrs["total"] = self.frames_written
        self._data.attrs["num_demos"] = self._episode_index
        for k, v in info.to_attrs().items():
            self._data.attrs[k] = v if isinstance(v, str | int | float | bool) else json.dumps(v)
        self._data.attrs["env_args"] = json.dumps(
            {"env_name": "tactile", "type": 0, "env_kwargs": {"schema": S.SCHEMA_VERSION}}
        )
        if info.stats:
            sg = self._file.create_group("stats")
            for key, s in info.stats.items():
                g = sg.create_group(obs_key(key))
                g.attrs["schema_path"] = key
                g.attrs["count"] = int(s.get("count", 0))
                for name in ("mean", "std", "min", "max"):
                    if name in s:
                        g.create_dataset(name, data=np.asarray(s[name], dtype=np.float32))
        self._file.close()

    def abort(self) -> None:
        try:
            self._file.close()
        except Exception:  # noqa: BLE001
            pass


class _SchemaView(Mapping[str, Any]):
    """Map schema paths to the datasets of one demo (for validation / loading)."""

    def __init__(self, demo: h5py.Group):
        self._demo = demo
        self._map: dict[str, h5py.Dataset] = {}
        if "timestamps" in demo:
            self._map[S.TIMESTAMPS] = demo["timestamps"]
        if "actions" in demo:
            self._map[S.ACTION] = demo["actions"]
        for name, ds in demo["obs"].items():
            self._map[str(ds.attrs.get("schema_path", f"/observation/{name}"))] = ds
        self.attrs = {"modalities": _load_modalities(demo.file["data"])}

    def __getitem__(self, key: str) -> Any:
        return self._map[normalize_path(key)]

    def __iter__(self):
        return iter(self._map)

    def __len__(self) -> int:
        return len(self._map)


def _load_modalities(data_group: h5py.Group) -> list[str]:
    raw = data_group.attrs.get("modalities")
    if raw is None:
        return []
    if isinstance(raw, bytes):
        raw = raw.decode()
    if isinstance(raw, str):
        try:
            return list(json.loads(raw))
        except json.JSONDecodeError:
            return [raw]
    return [str(x) for x in raw]


def schema_view(file: h5py.File | str | Path, demo: int = 0) -> Mapping[str, Any]:
    """Schema-path keyed view of ``/data/demo_<demo>`` usable with :func:`schema.validate`."""
    if not isinstance(file, h5py.File):
        file = h5py.File(file, "r")
    return _SchemaView(file["data"][f"demo_{demo}"])


def hdf5_summary(path: str | Path) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        demos = {}
        for name, g in f["data"].items():
            demos[name] = {
                "num_samples": int(g.attrs.get("num_samples", 0)),
                "actions": (
                    {"shape": list(g["actions"].shape), "dtype": str(g["actions"].dtype)}
                    if "actions" in g
                    else None
                ),
                "obs": {
                    k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in g["obs"].items()
                },
            }
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f["data"].attrs.items()}
        return {"path": str(path), "attrs": json_ready(attrs), "demos": demos}
