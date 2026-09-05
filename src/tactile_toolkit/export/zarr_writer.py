"""Chunked, compressed Zarr (v3) export mirroring the schema hierarchy."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from tactile_toolkit import schema as S
from tactile_toolkit.export.base import BaseWriter, DatasetInfo, json_ready
from tactile_toolkit.export.stats import stats_to_json
from tactile_toolkit.types import TactileChunk, normalize_path

_TARGET_CHUNK_BYTES = 16 * 1024 * 1024


class ZarrWriter(BaseWriter):
    """Append-only Zarr store with one array per schema field.

    Arrays are created lazily from the first chunk (which fixes dtypes and
    trailing dimensions) and grown along the time axis with each ``append``.
    Chunk sizes along time are capped so that no single compressed chunk
    exceeds roughly 16 MiB, which keeps random access and cloud streaming
    efficient for large image arrays.

    Incoming frames are staged per array until a whole number of storage
    chunks is available, so each write touches only complete chunks and
    never has to read back and re-encode a partially filled one. The
    remainder is flushed on ``close``.
    """

    format_name = "zarr"

    def __init__(
        self,
        path: str | Path,
        chunk_frames: int = 64,
        compression_level: int = 3,
        overwrite: bool = True,
        episode_boundaries: bool = True,
    ):
        super().__init__(path)
        self.chunk_frames = max(1, chunk_frames)
        self.compression_level = compression_level
        self.episode_boundaries = episode_boundaries
        if self.path.exists():
            if not overwrite:
                raise FileExistsError(self.path)
            shutil.rmtree(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._root = zarr.open_group(str(self.path), mode="w")
        self._arrays: dict[str, zarr.Array] = {}
        self._pending: dict[str, list[np.ndarray]] = {}
        self._pending_frames: dict[str, int] = {}
        self._episode_starts: list[int] = []
        self._episode_lengths: list[int] = []

    @property
    def root(self) -> zarr.Group:
        return self._root

    def _chunks_for(self, sample_shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
        frame_bytes = max(1, int(np.prod(sample_shape)) * itemsize)
        t = max(1, min(self.chunk_frames, _TARGET_CHUNK_BYTES // frame_bytes))
        return (t, *sample_shape)

    def _get_array(self, key: str, sample: np.ndarray) -> zarr.Array:
        arr = self._arrays.get(key)
        if arr is not None:
            return arr
        rel = key.lstrip("/")
        parent_path, _, name = rel.rpartition("/")
        parent = self._root.require_group(parent_path) if parent_path else self._root
        sample_shape = tuple(int(s) for s in sample.shape[1:])
        arr = parent.create_array(
            name=name,
            shape=(0, *sample_shape),
            chunks=self._chunks_for(sample_shape, sample.dtype.itemsize),
            dtype=sample.dtype,
            compressors=zarr.codecs.ZstdCodec(level=self.compression_level),
            # Every chunk we write carries data; skip the fill-value scan.
            config={"write_empty_chunks": True},
        )
        spec = S.SCHEMA[key] if key in S.SCHEMA else None
        if spec is not None:
            arr.attrs.update({"unit": spec.unit, "description": spec.description})
        self._arrays[key] = arr
        return arr

    def _begin_episode(self) -> None:
        self._episode_starts.append(self.frames_written)

    def _append(self, chunk: TactileChunk) -> None:
        for key, value in chunk.items():
            value = np.asarray(value)
            key = normalize_path(key)
            arr = self._get_array(key, value)
            if arr.dtype != value.dtype:
                value = value.astype(arr.dtype, copy=False)
            self._pending.setdefault(key, []).append(value)
            self._pending_frames[key] = self._pending_frames.get(key, 0) + int(value.shape[0])
            self._flush(key, final=False)

    def _flush(self, key: str, final: bool) -> None:
        arr = self._arrays[key]
        chunk_t = int(arr.chunks[0])
        pending = self._pending_frames.get(key, 0)
        n_write = pending if final else (pending // chunk_t) * chunk_t
        if n_write <= 0:
            return
        parts = self._pending[key]
        data = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        arr.append(data[:n_write], axis=0)
        rest = data[n_write:]
        # Copy so the remainder does not pin the full concatenated buffer.
        self._pending[key] = [rest.copy()] if rest.shape[0] else []
        self._pending_frames[key] = int(rest.shape[0])

    def _end_episode(self) -> None:
        self._episode_lengths.append(self.episode_frames)

    def _finalize(self, info: DatasetInfo) -> None:
        for key in list(self._arrays):
            self._flush(key, final=True)
        attrs = info.to_attrs()
        attrs["format"] = "zarr"
        attrs["num_frames"] = self.frames_written
        attrs["num_episodes"] = len(self._episode_lengths)
        self._root.attrs.update(attrs)
        if self.episode_boundaries and self._episode_lengths:
            ep = self._root.require_group("episodes")
            starts = np.asarray(self._episode_starts, dtype=np.int64)
            lengths = np.asarray(self._episode_lengths, dtype=np.int64)
            a = ep.create_array(name="start", shape=starts.shape, dtype=starts.dtype)
            a[:] = starts
            b = ep.create_array(name="length", shape=lengths.shape, dtype=lengths.dtype)
            b[:] = lengths
        if info.stats:
            sg = self._root.require_group("stats")
            for key, s in info.stats.items():
                g = sg.require_group(key.strip("/").replace("/", "."))
                for name in ("mean", "std", "min", "max"):
                    if name in s:
                        val = np.asarray(s[name], dtype=np.float32)
                        if val.ndim == 0:
                            val = val.reshape(1)
                        arr = g.create_array(name=name, shape=val.shape, dtype=val.dtype)
                        arr[...] = val
                g.attrs["count"] = int(s.get("count", 0))
                g.attrs["schema_path"] = key
            self._root.attrs["stats_summary"] = json_ready(
                {k: {"count": v.get("count", 0)} for k, v in stats_to_json(info.stats).items()}
            )


def open_zarr(path: str | Path, mode: str = "r") -> zarr.Group:
    """Open an exported Zarr store."""
    return zarr.open_group(str(path), mode=mode)


def zarr_summary(path: str | Path) -> dict[str, Any]:
    """Compact description of an exported store: arrays, shapes, dtypes and attributes."""
    root = open_zarr(path)
    arrays: dict[str, dict[str, Any]] = {}

    def walk(group: zarr.Group, prefix: str) -> None:
        for name, arr in group.arrays():
            arrays[f"{prefix}/{name}"] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "chunks": list(arr.chunks),
            }
        for name, sub in group.groups():
            walk(sub, f"{prefix}/{name}")

    walk(root, "")
    return {"path": str(path), "attrs": dict(root.attrs), "arrays": arrays}
