"""Zero-tare baseline estimation and slow drift compensation for taxel arrays."""

from __future__ import annotations

import numpy as np


class ZeroTare:
    """Estimate the no-load baseline from the first idle samples.

    The baseline is the per-taxel median of the first ``idle_frames`` samples;
    the per-taxel noise level (robust std via MAD) is kept to derive a default
    contact threshold.
    """

    def __init__(self, idle_frames: int = 20):
        self.idle_frames = max(1, idle_frames)
        self._buffer: list[np.ndarray] = []
        self.baseline: np.ndarray | None = None
        self.noise_std: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return self.baseline is not None

    def feed(self, samples: np.ndarray) -> None:
        if self.ready:
            return
        for s in np.asarray(samples):
            self._buffer.append(np.asarray(s, dtype=np.float32))
            if len(self._buffer) >= self.idle_frames:
                break
        if len(self._buffer) >= self.idle_frames:
            self._finalize()

    def force_ready(self) -> None:
        if not self.ready:
            if not self._buffer:
                raise RuntimeError("No samples were fed to ZeroTare")
            self._finalize()

    def _finalize(self) -> None:
        block = np.stack(self._buffer, axis=0)
        self.baseline = np.median(block, axis=0).astype(np.float32)
        mad = np.median(np.abs(block - self.baseline[None]), axis=0)
        self.noise_std = (1.4826 * mad).astype(np.float32)
        self._buffer.clear()


class DriftCompensator:
    """Exponential tracking of the baseline on taxels that are not in contact.

    ``alpha`` is the per-sample update rate; with 100 Hz data ``alpha=1e-3``
    follows drifts slower than roughly 10 s while ignoring contact transients.
    """

    def __init__(self, alpha: float = 1e-3):
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must be in [0, 1)")
        self.alpha = alpha

    def update(
        self, baseline: np.ndarray, samples: np.ndarray, in_contact: np.ndarray
    ) -> np.ndarray:
        """Return the updated baseline after seeing ``samples`` ``(T, N[, A])``.

        ``in_contact`` is ``(T, N)`` boolean; taxels in contact are frozen.
        """
        if self.alpha == 0.0:
            return baseline
        base = baseline.astype(np.float32, copy=True)
        samples = np.asarray(samples, dtype=np.float32)
        free = ~np.asarray(in_contact, dtype=bool)
        if samples.ndim == 3:
            free = free[..., None]
        for t in range(samples.shape[0]):
            delta = samples[t] - base
            base = base + self.alpha * np.where(free[t], delta, 0.0)
        return base
