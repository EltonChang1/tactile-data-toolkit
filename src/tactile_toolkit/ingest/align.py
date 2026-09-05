"""Resample heterogeneous sensor streams onto a common master clock."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

from tactile_toolkit.types import Stream

AlignMode = Literal["nearest", "linear"]


class TimeAligner:
    """Align source streams to target timestamps.

    ``nearest`` picks the source sample closest in time (suitable for images
    and discrete signals). ``linear`` interpolates between neighbours and holds
    the edge value outside the source range (suitable for high-rate analog
    signals such as 1 kHz force/torque). Targets farther than ``max_gap_s``
    from any source sample are filled with ``fill_value`` so the quality gate
    can drop them.
    """

    def __init__(
        self,
        mode: AlignMode = "nearest",
        max_gap_s: float | None = None,
        fill_value: float = np.nan,
    ):
        if mode not in ("nearest", "linear"):
            raise ValueError(f"Unknown alignment mode '{mode}'")
        self.mode = mode
        self.max_gap_s = max_gap_s
        self.fill_value = fill_value

    # ------------------------------------------------------------------ indices
    @staticmethod
    def nearest_indices(
        source_ts: np.ndarray, target_ts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(indices, gaps)`` of the closest source sample for each target."""
        source_ts = np.asarray(source_ts, dtype=np.float64)
        target_ts = np.asarray(target_ts, dtype=np.float64)
        n = source_ts.shape[0]
        if n == 0:
            raise ValueError("Cannot align against an empty source stream")
        right = np.searchsorted(source_ts, target_ts, side="left")
        right = np.clip(right, 0, n - 1)
        left = np.clip(right - 1, 0, n - 1)
        d_right = np.abs(source_ts[right] - target_ts)
        d_left = np.abs(source_ts[left] - target_ts)
        idx = np.where(d_left <= d_right, left, right)
        gaps = np.minimum(d_left, d_right)
        return idx, gaps

    @staticmethod
    def linear_weights(
        source_ts: np.ndarray, target_ts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(i0, i1, w, gaps)`` such that ``x = (1-w)*src[i0] + w*src[i1]``."""
        source_ts = np.asarray(source_ts, dtype=np.float64)
        target_ts = np.asarray(target_ts, dtype=np.float64)
        n = source_ts.shape[0]
        if n == 0:
            raise ValueError("Cannot align against an empty source stream")
        if n == 1:
            i0 = np.zeros_like(target_ts, dtype=np.int64)
            return i0, i0, np.zeros_like(target_ts), np.abs(source_ts[0] - target_ts)
        i1 = np.searchsorted(source_ts, target_ts, side="right")
        i1 = np.clip(i1, 1, n - 1)
        i0 = i1 - 1
        span = source_ts[i1] - source_ts[i0]
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(span > 0, (target_ts - source_ts[i0]) / span, 0.0)
        w = np.clip(w, 0.0, 1.0)
        gaps = np.minimum(np.abs(source_ts[i0] - target_ts), np.abs(source_ts[i1] - target_ts))
        return i0, i1, w, gaps

    # ---------------------------------------------------------------- alignment
    def align(self, target_ts: np.ndarray, stream: Stream) -> np.ndarray:
        """``stream.data`` resampled at ``target_ts``; shape ``(len(target_ts), ...)``."""
        target_ts = np.asarray(target_ts, dtype=np.float64)
        data = stream.data
        mode = self.mode
        if mode == "linear" and data.dtype.kind not in "fc":
            mode = "nearest"

        if mode == "nearest":
            idx, gaps = self.nearest_indices(stream.timestamps, target_ts)
            out = data[idx]
        else:
            i0, i1, w, gaps = self.linear_weights(stream.timestamps, target_ts)
            w = w.reshape((-1,) + (1,) * (data.ndim - 1))
            out = (1.0 - w) * data[i0] + w * data[i1]
            out = out.astype(np.result_type(data.dtype, np.float32), copy=False)

        if self.max_gap_s is not None:
            bad = gaps > self.max_gap_s
            if np.any(bad):
                if out.dtype.kind not in "fc":
                    out = out.astype(np.float32)
                out = out.copy()
                out[bad] = self.fill_value
        return out

    def align_many(
        self, target_ts: np.ndarray, streams: Mapping[str, Stream]
    ) -> dict[str, np.ndarray]:
        return {name: self.align(target_ts, s) for name, s in streams.items()}

    # --------------------------------------------------------------- torch path
    def align_torch(self, target_ts: Any, source_ts: Any, data: Any) -> Any:
        """Torch implementation for tensors already on an accelerator."""
        import torch

        target_ts = torch.as_tensor(target_ts, dtype=torch.float64)
        source_ts = torch.as_tensor(source_ts, dtype=torch.float64)
        data = torch.as_tensor(data)
        n = source_ts.shape[0]
        if self.mode == "nearest" or not data.is_floating_point():
            right = torch.searchsorted(source_ts, target_ts).clamp(0, n - 1)
            left = (right - 1).clamp(0, n - 1)
            pick_left = (source_ts[left] - target_ts).abs() <= (source_ts[right] - target_ts).abs()
            idx = torch.where(pick_left, left, right)
            return data[idx]
        i1 = torch.searchsorted(source_ts, target_ts, right=True).clamp(1, n - 1)
        i0 = i1 - 1
        span = source_ts[i1] - source_ts[i0]
        w = torch.where(span > 0, (target_ts - source_ts[i0]) / span, torch.zeros_like(span))
        w = w.clamp(0, 1).to(data.dtype).reshape((-1,) + (1,) * (data.dim() - 1))
        return (1 - w) * data[i0] + w * data[i1]
