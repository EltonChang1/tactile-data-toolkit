"""Quality-assurance gatekeeper: drop jittery, duplicated, or corrupt frames."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class QaConfig:
    max_jitter_s: float | None = 0.002
    """Drop a frame when its inter-frame interval deviates from the nominal period by more."""
    nominal_period_s: float | None = None
    """Expected frame period. Estimated from the first chunk when ``None``."""
    drop_duplicate_frames: bool = True
    """Drop frames whose image is identical to the previous kept frame (frozen camera)."""
    drop_nonfinite: bool = True
    """Drop frames where any aligned auxiliary array contains NaN/Inf."""
    drop_non_monotonic: bool = True
    """Drop frames whose timestamp goes backwards."""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QaReport:
    total: int = 0
    kept: int = 0
    dropped_jitter: int = 0
    dropped_duplicate: int = 0
    dropped_nonfinite: int = 0
    dropped_non_monotonic: int = 0
    nominal_period_s: float | None = None
    max_jitter_s: float = 0.0
    sum_jitter_s: float = 0.0
    jitter_samples: int = 0

    @property
    def dropped(self) -> int:
        return self.total - self.kept

    @property
    def mean_jitter_s(self) -> float:
        return self.sum_jitter_s / self.jitter_samples if self.jitter_samples else 0.0

    def merge(self, other: QaReport) -> QaReport:
        return QaReport(
            total=self.total + other.total,
            kept=self.kept + other.kept,
            dropped_jitter=self.dropped_jitter + other.dropped_jitter,
            dropped_duplicate=self.dropped_duplicate + other.dropped_duplicate,
            dropped_nonfinite=self.dropped_nonfinite + other.dropped_nonfinite,
            dropped_non_monotonic=self.dropped_non_monotonic + other.dropped_non_monotonic,
            nominal_period_s=self.nominal_period_s or other.nominal_period_s,
            max_jitter_s=max(self.max_jitter_s, other.max_jitter_s),
            sum_jitter_s=self.sum_jitter_s + other.sum_jitter_s,
            jitter_samples=self.jitter_samples + other.jitter_samples,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dropped"] = self.dropped
        d["mean_jitter_s"] = self.mean_jitter_s
        return d

    def summary(self) -> str:
        period = f"{self.nominal_period_s * 1e3:.3f} ms" if self.nominal_period_s else "n/a"
        return (
            f"frames: {self.total} total, {self.kept} kept, {self.dropped} dropped "
            f"(jitter={self.dropped_jitter}, duplicate={self.dropped_duplicate}, "
            f"nonfinite={self.dropped_nonfinite}, non_monotonic={self.dropped_non_monotonic}); "
            f"nominal period {period}, jitter mean {self.mean_jitter_s * 1e3:.3f} ms, "
            f"max {self.max_jitter_s * 1e3:.3f} ms"
        )


class QualityGate:
    """Stateful frame filter that can be applied chunk by chunk.

    The gate remembers the last kept timestamp and frame so that checks remain
    correct across chunk boundaries. Call :meth:`reset` between trajectories.
    """

    def __init__(self, config: QaConfig | None = None):
        self.config = config or QaConfig()
        self.reset()

    def reset(self) -> None:
        self._last_ts: float | None = None
        self._last_frame: np.ndarray | None = None
        self._nominal: float | None = self.config.nominal_period_s
        self.report = QaReport(nominal_period_s=self._nominal)

    @property
    def nominal_period_s(self) -> float | None:
        return self._nominal

    def _estimate_period(self, timestamps: np.ndarray) -> None:
        if self._nominal is not None or timestamps.shape[0] < 3:
            return
        dt = np.diff(timestamps)
        dt = dt[dt > 0]
        if dt.size:
            self._nominal = float(np.median(dt))
            self.report.nominal_period_s = self._nominal

    def evaluate(
        self,
        timestamps: np.ndarray,
        frames: np.ndarray | None = None,
        aligned: Mapping[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, QaReport]:
        """Return ``(keep_mask, chunk_report)`` for one chunk of the master stream."""
        cfg = self.config
        timestamps = np.asarray(timestamps, dtype=np.float64)
        n = timestamps.shape[0]
        keep = np.ones(n, dtype=bool)
        chunk = QaReport(total=n)
        if n == 0:
            return keep, chunk

        self._estimate_period(timestamps)
        chunk.nominal_period_s = self._nominal
        nominal = self._nominal

        # Vectorised non-finite check on the aligned auxiliary arrays.
        nonfinite = np.zeros(n, dtype=bool)
        if cfg.drop_nonfinite and aligned:
            for arr in aligned.values():
                arr = np.asarray(arr)
                if arr.dtype.kind in "fc":
                    nonfinite |= ~np.isfinite(arr.reshape(n, -1)).all(axis=1)

        check_dups = cfg.drop_duplicate_frames and frames is not None
        if check_dups:
            frames = np.asarray(frames)

        # Sequential pass: every check is relative to the last *kept* frame so a
        # single late frame does not cascade into dropping its on-time successors.
        last_ts = self._last_ts
        last_frame = self._last_frame
        for i in range(n):
            t = float(timestamps[i])
            if nonfinite[i]:
                keep[i] = False
                chunk.dropped_nonfinite += 1
                continue
            if last_ts is not None:
                dt = t - last_ts
                if cfg.drop_non_monotonic and dt < 0:
                    keep[i] = False
                    chunk.dropped_non_monotonic += 1
                    continue
                if nominal is not None and nominal > 0:
                    k = max(1.0, float(np.round(dt / nominal)))
                    jitter = abs(dt - k * nominal)
                    chunk.jitter_samples += 1
                    chunk.sum_jitter_s += jitter
                    chunk.max_jitter_s = max(chunk.max_jitter_s, jitter)
                    if cfg.max_jitter_s is not None and jitter > cfg.max_jitter_s:
                        keep[i] = False
                        chunk.dropped_jitter += 1
                        continue
            if check_dups:
                frame = frames[i]
                if (
                    last_frame is not None
                    and last_frame.shape == frame.shape
                    and np.array_equal(last_frame, frame)
                ):
                    keep[i] = False
                    chunk.dropped_duplicate += 1
                    continue
                last_frame = frame
            last_ts = t

        chunk.kept = int(keep.sum())
        self._last_ts = last_ts
        if check_dups and last_frame is not None:
            self._last_frame = np.array(last_frame, copy=True)
        self.report = self.report.merge(chunk)
        return keep, chunk
