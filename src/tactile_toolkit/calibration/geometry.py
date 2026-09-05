"""Sensor surface geometry: where each taxel / sampled pixel sits in metres."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SensorGeometry:
    """Positions of the ``N`` surface elements in the sensor frame.

    Convention: ``x`` right, ``y`` down (image convention), ``z`` pointing out
    of the sensing surface towards the environment. Indentation therefore
    moves surface points towards negative ``z``.
    """

    positions_m: np.ndarray
    """``(N, 3)`` float32."""
    layout: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.positions_m = np.asarray(self.positions_m, dtype=np.float32)
        if self.positions_m.ndim != 2 or self.positions_m.shape[1] != 3:
            raise ValueError("positions_m must have shape (N, 3)")

    @property
    def num_elements(self) -> int:
        return int(self.positions_m.shape[0])

    @property
    def xy(self) -> np.ndarray:
        return self.positions_m[:, :2]

    @property
    def element_area_m2(self) -> float:
        area = self.layout.get("element_area_m2")
        if area is not None:
            return float(area)
        pitch = self.layout.get("pitch_m")
        if pitch is not None:
            return float(pitch) ** 2
        if self.num_elements < 2:
            return 1.0
        # Fallback: nearest-neighbour spacing squared.
        from scipy.spatial import cKDTree

        d, _ = cKDTree(self.xy).query(self.xy, k=2)
        return float(np.median(d[:, 1]) ** 2)

    # ------------------------------------------------------------ constructors
    @classmethod
    def grid(
        cls,
        rows: int,
        cols: int,
        pitch_m: float,
        pitch_y_m: float | None = None,
        centered: bool = True,
        z_m: float = 0.0,
    ) -> SensorGeometry:
        """Regular planar grid in row-major order."""
        pitch_y_m = pitch_m if pitch_y_m is None else pitch_y_m
        ys, xs = np.mgrid[0:rows, 0:cols]
        x = xs.astype(np.float64) * pitch_m
        y = ys.astype(np.float64) * pitch_y_m
        if centered:
            x -= (cols - 1) / 2 * pitch_m
            y -= (rows - 1) / 2 * pitch_y_m
        pos = np.stack([x.ravel(), y.ravel(), np.full(rows * cols, z_m)], axis=1)
        return cls(
            pos,
            layout={
                "type": "grid",
                "rows": rows,
                "cols": cols,
                "pitch_m": pitch_m,
                "pitch_y_m": pitch_y_m,
                "element_area_m2": pitch_m * pitch_y_m,
            },
        )

    @classmethod
    def from_image(
        cls, height: int, width: int, pixel_size_m: float, stride: int = 1
    ) -> SensorGeometry:
        """Centres of ``stride x stride`` pixel blocks of a vision-tactile image."""
        rows = height // stride
        cols = width // stride
        geo = cls.grid(rows, cols, pixel_size_m * stride)
        geo.layout.update(
            {
                "type": "image_grid",
                "image_height": height,
                "image_width": width,
                "pixel_size_m": pixel_size_m,
                "stride": stride,
            }
        )
        return geo

    @classmethod
    def from_positions(cls, xy_m: np.ndarray, z_m: float = 0.0, **layout: Any) -> SensorGeometry:
        xy_m = np.asarray(xy_m, dtype=np.float64)
        if xy_m.ndim != 2 or xy_m.shape[1] not in (2, 3):
            raise ValueError("xy_m must have shape (N, 2) or (N, 3)")
        if xy_m.shape[1] == 2:
            xy_m = np.concatenate([xy_m, np.full((xy_m.shape[0], 1), z_m)], axis=1)
        return cls(xy_m, layout={"type": "custom", **layout})

    def to_dict(self) -> dict[str, Any]:
        return {"layout": dict(self.layout), "num_elements": self.num_elements}
