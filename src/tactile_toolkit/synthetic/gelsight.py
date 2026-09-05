"""Synthetic vision-tactile (gel finger) sensor with a known forward model.

The simulator renders a reference gel image with a dark marker dot grid, and
for each frame encodes the surface gradient of a scripted indentation into the
red/green channels while displacing markers according to a shear field. The
forward model is the exact inverse of
:func:`tactile_toolkit.calibration.gelsight.poisson.rgb_to_gradients`, so
reconstruction quality can be measured against ground truth.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class GelSightSim:
    height: int = 240
    width: int = 320
    marker_rows: int = 8
    marker_cols: int = 10
    marker_radius_px: int = 3
    marker_margin_px: int = 16
    gradient_gain: float = 1.0
    """Colour delta (fraction of 255) per unit surface slope."""
    sensor_width_m: float = 0.020
    base_color: tuple[int, int, int] = (128, 128, 128)
    marker_color: int = 30
    noise_std: float = 1.0
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)
    _reference: np.ndarray = field(init=False, repr=False)
    _marker_positions: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        ys = np.linspace(
            self.marker_margin_px, self.height - 1 - self.marker_margin_px, self.marker_rows
        )
        xs = np.linspace(
            self.marker_margin_px, self.width - 1 - self.marker_margin_px, self.marker_cols
        )
        gx, gy = np.meshgrid(xs, ys)
        self._marker_positions = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
        base = np.empty((self.height, self.width, 3), dtype=np.float32)
        base[..., 0], base[..., 1], base[..., 2] = self.base_color
        self._reference = self._draw_markers(base, self._marker_positions).astype(np.uint8)

    # ----------------------------------------------------------------- geometry
    @property
    def pixel_size_m(self) -> float:
        return self.sensor_width_m / self.width

    @property
    def marker_positions(self) -> np.ndarray:
        """Reference marker centres ``(M, 2)`` as ``(x, y)`` pixels."""
        return self._marker_positions.copy()

    @property
    def reference(self) -> np.ndarray:
        return self._reference.copy()

    def _grid(self) -> tuple[np.ndarray, np.ndarray]:
        y, x = np.mgrid[0 : self.height, 0 : self.width]
        return x.astype(np.float32), y.astype(np.float32)

    # ------------------------------------------------------------------ physics
    def height_map(
        self, center_px: tuple[float, float], radius_px: float, depth_m: float
    ) -> np.ndarray:
        """Gaussian indentation of peak ``depth_m`` (metres) around ``center_px``."""
        x, y = self._grid()
        sigma = max(radius_px / 2.0, 1.0)
        r2 = (x - center_px[0]) ** 2 + (y - center_px[1]) ** 2
        return (depth_m * np.exp(-r2 / (2 * sigma**2))).astype(np.float32)

    def gradients(self, z_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Dimensionless surface slopes ``dz/dx``, ``dz/dy``."""
        gy, gx = np.gradient(z_m.astype(np.float64), self.pixel_size_m)
        return gx.astype(np.float32), gy.astype(np.float32)

    def shear_field(
        self, center_px: tuple[float, float], radius_px: float, shear_px: tuple[float, float]
    ) -> np.ndarray:
        x, y = self._grid()
        sigma = max(radius_px / 2.0, 1.0)
        r2 = (x - center_px[0]) ** 2 + (y - center_px[1]) ** 2
        w = np.exp(-r2 / (2 * sigma**2))
        return np.stack([shear_px[0] * w, shear_px[1] * w], axis=-1).astype(np.float32)

    def _draw_markers(self, img: np.ndarray, positions: np.ndarray) -> np.ndarray:
        out = img.copy()
        for x, y in positions:
            cv2.circle(
                out,
                (int(round(float(x))), int(round(float(y)))),
                self.marker_radius_px,
                (self.marker_color,) * 3,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
        return out

    def render(
        self,
        center_px: tuple[float, float],
        radius_px: float,
        depth_m: float,
        shear_px: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Render one frame. Returns ``(rgb_uint8, ground_truth)``."""
        z = (
            self.height_map(center_px, radius_px, depth_m)
            if depth_m > 0
            else np.zeros((self.height, self.width), dtype=np.float32)
        )
        gx, gy = self.gradients(z)
        frame = np.empty((self.height, self.width, 3), dtype=np.float32)
        frame[..., 0] = self.base_color[0] + self.gradient_gain * gx * 255.0
        frame[..., 1] = self.base_color[1] + self.gradient_gain * gy * 255.0
        frame[..., 2] = self.base_color[2]
        shear = (
            self.shear_field(center_px, radius_px, shear_px)
            if depth_m > 0
            else np.zeros((self.height, self.width, 2), dtype=np.float32)
        )
        xi = np.clip(self._marker_positions[:, 0].round().astype(int), 0, self.width - 1)
        yi = np.clip(self._marker_positions[:, 1].round().astype(int), 0, self.height - 1)
        marker_disp = shear[yi, xi]
        frame = self._draw_markers(frame, self._marker_positions + marker_disp)
        if self.noise_std > 0:
            frame += self._rng.normal(0.0, self.noise_std, frame.shape).astype(np.float32)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        gt = {
            "depth_m": z,
            "shear_px": shear,
            "marker_displacement_px": marker_disp,
            "gx": gx,
            "gy": gy,
        }
        return frame, gt

    # ---------------------------------------------------------------- scripting
    def scripted_contact(
        self, t: float
    ) -> tuple[tuple[float, float], float, float, tuple[float, float]]:
        """Deterministic contact trajectory: ``(center_px, radius_px, depth_m, shear_px)``."""
        cx = self.width / 2 + 0.25 * self.width * np.sin(2 * np.pi * t / 7.0)
        cy = self.height / 2 + 0.20 * self.height * np.cos(2 * np.pi * t / 5.0)
        press = np.sin(2 * np.pi * t / 3.0)
        depth = 1.0e-3 * max(press, 0.0)
        radius = 0.15 * self.width
        shear = (2.5 * np.sin(2 * np.pi * t / 4.0), 2.0 * np.cos(2 * np.pi * t / 6.0))
        return (
            (float(cx), float(cy)),
            float(radius),
            float(depth),
            (float(shear[0]), float(shear[1])),
        )

    def iter_frames(
        self, n_frames: int, fps: float = 30.0, t0: float = 0.0, with_ground_truth: bool = False
    ) -> Iterator[tuple[float, np.ndarray, dict[str, np.ndarray] | None]]:
        """Yield ``(timestamp, frame, ground_truth_or_None)`` following the scripted contact."""
        for i in range(n_frames):
            t = t0 + i / fps
            center, radius, depth, shear = self.scripted_contact(t - t0)
            frame, gt = self.render(center, radius, depth, shear)
            yield t, frame, (gt if with_ground_truth else None)
