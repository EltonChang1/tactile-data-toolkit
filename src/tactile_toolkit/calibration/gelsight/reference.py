"""Reference (no-contact) frame estimation and marker masking for gel sensors."""

from __future__ import annotations

import cv2
import numpy as np


def estimate_reference(frames: np.ndarray, n_frames: int = 10) -> np.ndarray:
    """Median of the first ``n_frames`` frames, robust to a stray early contact."""
    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError("frames must have shape (T, H, W, 3)")
    if frames.shape[0] == 0:
        raise ValueError("Cannot estimate a reference from zero frames")
    sample = frames[: max(1, min(n_frames, frames.shape[0]))]
    return np.median(sample, axis=0).astype(np.uint8)


def marker_mask(
    reference: np.ndarray, threshold: int | None = None, dilate_px: int = 2
) -> np.ndarray:
    """Boolean ``(H, W)`` mask covering dark marker dots in the reference image."""
    gray = cv2.cvtColor(np.ascontiguousarray(reference), cv2.COLOR_RGB2GRAY)
    if threshold is None:
        thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Otsu splits marker vs gel; guard against near-uniform images.
        if thr <= 0 or thr >= 250:
            thr = float(np.percentile(gray, 5))
        threshold = int(thr)
    mask = (gray <= threshold).astype(np.uint8)
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    return mask.astype(bool)


class ReferenceEstimator:
    """Accumulate leading frames until a reference can be produced."""

    def __init__(self, n_frames: int = 10):
        self.n_frames = max(1, n_frames)
        self._buffer: list[np.ndarray] = []
        self._reference: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return self._reference is not None

    @property
    def reference(self) -> np.ndarray:
        if self._reference is None:
            raise RuntimeError("Reference not ready")
        return self._reference

    def feed(self, frames: np.ndarray) -> np.ndarray:
        """Consume frames; returns the reference once enough frames were seen."""
        if self._reference is not None:
            return self._reference
        for f in np.asarray(frames):
            self._buffer.append(f)
            if len(self._buffer) >= self.n_frames:
                break
        if len(self._buffer) >= self.n_frames:
            self._reference = estimate_reference(np.stack(self._buffer), self.n_frames)
            self._buffer.clear()
        return self._reference if self._reference is not None else np.stack(self._buffer)

    def force_ready(self) -> np.ndarray:
        """Finalize with whatever frames were collected (short trajectories)."""
        if self._reference is None:
            if not self._buffer:
                raise RuntimeError("No frames were fed to the reference estimator")
            self._reference = estimate_reference(np.stack(self._buffer), len(self._buffer))
            self._buffer.clear()
        return self._reference
