"""Marker dot detection and pyramidal Lucas-Kanade tracking on the gel surface."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from tactile_toolkit.calibration.gelsight.poisson import resolve_workers


def detect_markers(
    reference: np.ndarray,
    threshold: int | None = None,
    min_area: int = 3,
    max_area: int = 600,
) -> np.ndarray:
    """Locate dark marker dots in the reference frame. Returns ``(M, 2)`` ``(x, y)`` pixels."""
    gray = cv2.cvtColor(np.ascontiguousarray(reference), cv2.COLOR_RGB2GRAY)
    if threshold is None:
        thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if thr <= 0 or thr >= 250:
            thr = float(np.percentile(gray, 5))
        threshold = int(thr)
    binary = (gray <= threshold).astype(np.uint8)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    pts = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            pts.append(centroids[i])
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    pts_arr = np.asarray(pts, dtype=np.float32)
    order = np.lexsort((pts_arr[:, 0], np.round(pts_arr[:, 1] / 4.0)))
    return pts_arr[order]


class MarkerTracker:
    """Track marker displacements relative to the reference frame.

    Tracking is performed from the reference to each frame directly (rather
    than frame-to-frame), so displacements do not accumulate drift.
    """

    def __init__(
        self,
        reference: np.ndarray,
        markers: np.ndarray | None = None,
        win_size: int = 21,
        max_level: int = 3,
        max_error: float = 30.0,
    ):
        self.reference = np.ascontiguousarray(reference)
        self.ref_gray = cv2.cvtColor(self.reference, cv2.COLOR_RGB2GRAY)
        self.markers = (
            np.asarray(markers, dtype=np.float32)
            if markers is not None
            else detect_markers(reference)
        )
        self.win_size = (win_size, win_size)
        self.max_level = max_level
        self.max_error = max_error
        self._criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        self._tri: Delaunay | None = None
        if self.num_markers >= 3:
            try:
                self._tri = Delaunay(self.markers.astype(np.float64))
            except Exception:  # noqa: BLE001 - degenerate layouts fall back to nearest
                self._tri = None

    @property
    def num_markers(self) -> int:
        return int(self.markers.shape[0])

    def track(self, frames: np.ndarray, workers: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(displacement_px (T, M, 2), valid (T, M))``.

        ``workers != 1`` splits the batch across threads (OpenCV releases the
        GIL); results do not depend on the split.
        """
        frames = np.asarray(frames)
        if frames.ndim == 3:
            frames = frames[None]
        T = frames.shape[0]
        M = self.num_markers
        disp = np.zeros((T, M, 2), dtype=np.float32)
        valid = np.zeros((T, M), dtype=bool)
        if M == 0:
            return disp, valid
        n_threads = resolve_workers(workers, max(1, T // 4))
        if n_threads == 1 or T < 8:
            self._track_range(frames, 0, T, disp, valid)
            return disp, valid
        bounds = np.linspace(0, T, n_threads + 1).astype(int)
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            list(
                ex.map(
                    lambda ab: self._track_range(frames, int(ab[0]), int(ab[1]), disp, valid),
                    zip(bounds[:-1], bounds[1:], strict=True),
                )
            )
        return disp, valid

    def _track_range(
        self, frames: np.ndarray, start: int, stop: int, disp: np.ndarray, valid: np.ndarray
    ) -> None:
        ref_pts = self.markers.reshape(-1, 1, 2)
        for t in range(start, stop):
            gray = cv2.cvtColor(np.ascontiguousarray(frames[t]), cv2.COLOR_RGB2GRAY)
            pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.ref_gray,
                gray,
                ref_pts,
                None,
                winSize=self.win_size,
                maxLevel=self.max_level,
                criteria=self._criteria,
            )
            ok = (status.ravel() == 1) & (err.ravel() < self.max_error)
            d = pts.reshape(-1, 2) - self.markers
            d[~ok] = 0.0
            disp[t] = d
            valid[t] = ok

    def _interpolate(
        self, displacement: np.ndarray, valid: np.ndarray | None, query: np.ndarray
    ) -> np.ndarray:
        """Evaluate the piecewise-linear displacement field at ``(Q, 2)`` pixel positions.

        Returns ``(Q, T, 2)``; all frames and both components are interpolated
        in a single call over the reference triangulation.
        """
        displacement = np.asarray(displacement, dtype=np.float32)
        T, M, _ = displacement.shape
        values = displacement.copy()
        if valid is not None:
            values[~np.asarray(valid, dtype=bool)] = 0.0
        interp = LinearNDInterpolator(
            self._tri, values.transpose(1, 0, 2).reshape(M, T * 2), fill_value=0.0
        )
        return interp(query).reshape(query.shape[0], T, 2).astype(np.float32, copy=False)

    def grid_field(
        self,
        displacement: np.ndarray,
        valid: np.ndarray | None,
        shape: tuple[int, int],
        stride: int,
    ) -> np.ndarray:
        """Displacement sampled at the centre of every ``stride x stride`` block.

        Returns ``(T, H // stride, W // stride, 2)`` and matches the pooling
        used by :func:`~tactile_toolkit.calibration.gelsight.forces.force_field`,
        without ever materialising a full-resolution field.
        """
        H, W = shape
        T = int(np.asarray(displacement).shape[0])
        h, w = max(1, H // stride), max(1, W // stride)
        if self.num_markers == 0 or self._tri is None:
            return np.zeros((T, h, w, 2), dtype=np.float32)
        ys = (np.arange(h) + 0.5) * stride - 0.5
        xs = (np.arange(w) + 0.5) * stride - 0.5
        qx, qy = np.meshgrid(xs, ys)
        query = np.stack([qx.ravel(), qy.ravel()], axis=1)
        vals = self._interpolate(displacement, valid, query)  # (h*w, T, 2)
        return np.ascontiguousarray(vals.reshape(h, w, T, 2).transpose(2, 0, 1, 3))

    def dense_field(
        self,
        displacement: np.ndarray,
        valid: np.ndarray | None,
        shape: tuple[int, int],
        coarse: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Interpolate sparse marker displacements to a dense ``(T, H, W, 2)`` field.

        Interpolation happens on a coarse grid (default ~1/8 resolution) and is
        upsampled bilinearly, which is both fast and appropriately smooth.
        """
        H, W = shape
        displacement = np.asarray(displacement, dtype=np.float32)
        T, M, _ = displacement.shape
        out = np.zeros((T, H, W, 2), dtype=np.float32)
        if M == 0 or self._tri is None:
            return out
        if coarse is None:
            coarse = (max(8, H // 8), max(8, W // 8))
        ch, cw = coarse
        ys = np.linspace(0, H - 1, ch)
        xs = np.linspace(0, W - 1, cw)
        qx, qy = np.meshgrid(xs, ys)
        query = np.stack([qx.ravel(), qy.ravel()], axis=1)
        coarse_vals = self._interpolate(displacement, valid, query).reshape(ch, cw, T, 2)
        for t in range(T):
            field = np.ascontiguousarray(coarse_vals[:, :, t, :])
            out[t] = cv2.resize(field, (W, H), interpolation=cv2.INTER_LINEAR)
        return out
