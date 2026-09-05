"""Spatial interpolation of sparse taxel readings onto a regular grid."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay


class GridInterpolator:
    """Resample ``(T, N)`` taxel values to a ``(T, rows, cols)`` heat-map.

    The Delaunay triangulation of taxel positions is computed once; each call
    interpolates all frames in a single vectorised evaluation.
    """

    def __init__(
        self,
        positions_xy: np.ndarray,
        grid_shape: tuple[int, int],
        bounds: tuple[float, float, float, float] | None = None,
        method: str = "linear",
    ):
        pos = np.asarray(positions_xy, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError("positions_xy must be (N, 2)")
        self.positions = pos
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        if bounds is None:
            xmin, ymin = pos.min(axis=0)
            xmax, ymax = pos.max(axis=0)
        else:
            xmin, xmax, ymin, ymax = bounds
        rows, cols = self.grid_shape
        xs = np.linspace(xmin, xmax, cols)
        ys = np.linspace(ymin, ymax, rows)
        qx, qy = np.meshgrid(xs, ys)
        self.query = np.stack([qx.ravel(), qy.ravel()], axis=1)
        self.method = method
        self._tri: Delaunay | None = None
        if method == "linear" and pos.shape[0] >= 3:
            try:
                self._tri = Delaunay(pos)
            except Exception:  # noqa: BLE001 - collinear layouts
                self._tri = None

    def __call__(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        squeeze = values.ndim == 1
        if squeeze:
            values = values[None]
        T, N = values.shape
        if N != self.positions.shape[0]:
            raise ValueError(f"expected {self.positions.shape[0]} taxels, got {N}")
        rows, cols = self.grid_shape
        if self._tri is not None:
            interp = LinearNDInterpolator(self._tri, values.T, fill_value=np.nan)
            grid = interp(self.query)  # (Q, T)
            if np.isnan(grid).any():
                near = NearestNDInterpolator(self.positions, values.T)
                fill = near(self.query)
                grid = np.where(np.isnan(grid), fill, grid)
        else:
            near = NearestNDInterpolator(self.positions, values.T)
            grid = near(self.query)
        out = grid.T.reshape(T, rows, cols).astype(np.float32)
        return out[0] if squeeze else out
