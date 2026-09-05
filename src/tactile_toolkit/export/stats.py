"""Streaming feature statistics (mean / std / min / max / count) over the time axis."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class RunningStats:
    """Welford / Chan parallel-merge statistics over axis 0 of successive batches."""

    def __init__(self) -> None:
        self.count = 0
        self.mean: np.ndarray | None = None
        self._m2: np.ndarray | None = None
        self.min: np.ndarray | None = None
        self.max: np.ndarray | None = None

    def update(self, batch: np.ndarray) -> None:
        x = np.asarray(batch)
        if x.dtype == bool:
            x = x.astype(np.float32)
        elif x.dtype.kind in "iu":
            x = x.astype(np.float64)
        n = int(x.shape[0])
        if n == 0:
            return
        # Accumulate in float64 without materialising a float64 copy of the batch.
        b_mean = x.mean(axis=0, dtype=np.float64)
        d = x - b_mean
        np.multiply(d, d, out=d)
        b_m2 = d.sum(axis=0)
        b_min = x.min(axis=0).astype(np.float64)
        b_max = x.max(axis=0).astype(np.float64)
        self.merge(n, b_mean, b_m2, b_min, b_max)

    def merge(
        self, n: int, b_mean: np.ndarray, b_m2: np.ndarray, b_min: np.ndarray, b_max: np.ndarray
    ) -> None:
        """Fold pre-computed batch moments (count, mean, M2, min, max) into the running state."""
        if n <= 0:
            return
        b_mean = np.asarray(b_mean, dtype=np.float64)
        b_m2 = np.asarray(b_m2, dtype=np.float64)
        b_min = np.asarray(b_min, dtype=np.float64)
        b_max = np.asarray(b_max, dtype=np.float64)
        if self.mean is None:
            self.count, self.mean, self._m2, self.min, self.max = n, b_mean, b_m2, b_min, b_max
            return
        assert self._m2 is not None and self.min is not None and self.max is not None
        total = self.count + n
        delta = b_mean - self.mean
        self.mean = self.mean + delta * (n / total)
        self._m2 = self._m2 + b_m2 + delta**2 * (self.count * n / total)
        self.min = np.minimum(self.min, b_min)
        self.max = np.maximum(self.max, b_max)
        self.count = total

    def update_uint8_histogram(self, counts: np.ndarray, scale: float = 1.0 / 255.0) -> None:
        """Exact moments of ``uint8`` data given per-feature histograms ``(F, 256)``.

        Values are mapped to ``bin * scale`` (``[0, 1]`` by default), which is
        how image statistics are conventionally reported.
        """
        counts = np.asarray(counts, dtype=np.float64)
        if counts.ndim == 1:
            counts = counts[None]
        levels = np.arange(256, dtype=np.float64) * scale
        n_per = counts.sum(axis=1)
        n = int(n_per[0])
        if n <= 0:
            return
        mean = (counts * levels).sum(axis=1) / n_per
        m2 = (counts * (levels[None] - mean[:, None]) ** 2).sum(axis=1)
        nonzero = counts > 0
        mn = np.array([levels[row].min() for row in nonzero])
        mx = np.array([levels[row].max() for row in nonzero])
        self.merge(n, mean, m2, mn, mx)

    @property
    def var(self) -> np.ndarray:
        if self.mean is None or self._m2 is None or self.count < 1:
            raise RuntimeError("No data")
        return self._m2 / self.count

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    def result(self) -> dict[str, Any]:
        if self.mean is None:
            return {"count": 0}
        return {
            "mean": np.asarray(self.mean, dtype=np.float32),
            "std": np.asarray(self.std, dtype=np.float32),
            "min": np.asarray(self.min, dtype=np.float32),
            "max": np.asarray(self.max, dtype=np.float32),
            "count": int(self.count),
        }


class StatsCollector:
    """Per-key :class:`RunningStats` with image-aware reduction.

    Arrays whose key is registered as an image are reduced to per-channel
    statistics on a ``[0, 1]`` scale (shape ``(C, 1, 1)``), matching the
    convention used by LeRobot datasets. Dense spatial maps (depth, pressure:
    more than ``dense_threshold`` elements per frame) are reduced to scalar
    statistics over all elements. Everything else keeps its feature shape.
    """

    def __init__(
        self,
        image_keys: set[str] | None = None,
        image_subsample: int = 8,
        dense_threshold: int = 65_536,
    ):
        self.image_keys = set(image_keys or ())
        self.image_subsample = max(1, image_subsample)
        self.dense_threshold = int(dense_threshold)
        self._stats: dict[str, RunningStats] = {}
        self._dense: set[str] = set()

    def _is_dense(self, key: str, arr: np.ndarray) -> bool:
        if key in self._dense:
            return True
        per_frame = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
        if arr.ndim >= 3 and per_frame > self.dense_threshold:
            self._dense.add(key)
            return True
        return False

    def update(self, arrays: Mapping[str, np.ndarray]) -> None:
        for key, arr in arrays.items():
            arr = np.asarray(arr)
            if arr.shape[0] == 0:
                continue
            rs = self._stats.setdefault(key, RunningStats())
            if key in self.image_keys and arr.ndim == 4:
                sub = arr[:: self.image_subsample]
                if sub.dtype == np.uint8:
                    # Exact per-channel moments from histograms: no float copy of the pixels.
                    hist = np.stack(
                        [
                            np.bincount(sub[..., c].ravel(), minlength=256)
                            for c in range(sub.shape[-1])
                        ]
                    )
                    rs.update_uint8_histogram(hist)
                else:
                    # Collapse spatial axes: statistics per channel over all pixels.
                    flat = sub.astype(np.float32).reshape(-1, sub.shape[-1])
                    rs.update(flat)
            elif self._is_dense(key, arr):
                sub = arr[:: self.image_subsample]
                rs.update(sub.reshape(-1, 1))
            else:
                rs.update(arr)

    def result(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for key, rs in self._stats.items():
            r = rs.result()
            if key in self.image_keys and r.get("count", 0):
                for k in ("mean", "std", "min", "max"):
                    r[k] = r[k].reshape(-1, 1, 1)
            out[key] = r
        return out


def stats_to_json(stats: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, s in stats.items():
        entry: dict[str, Any] = {}
        for k, v in s.items():
            if isinstance(v, np.ndarray):
                entry[k] = v.tolist()
            elif isinstance(v, np.generic):
                entry[k] = v.item()
            else:
                entry[k] = v
        out[key] = entry
    return out
