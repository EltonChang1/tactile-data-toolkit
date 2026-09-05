"""Chunk-wise taxel / digital-skin calibration: raw counts -> forces, contact, point cloud."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tactile_toolkit import schema as S
from tactile_toolkit.calibration.geometry import SensorGeometry
from tactile_toolkit.calibration.pointcloud import build_point_cloud
from tactile_toolkit.calibration.taxel.baseline import DriftCompensator, ZeroTare
from tactile_toolkit.calibration.taxel.interpolate import GridInterpolator
from tactile_toolkit.types import Modality, SensorMetadata, TactileChunk


@dataclass
class TaxelConfig:
    idle_frames: int = 20
    """Leading samples assumed load-free for zero-tare."""
    counts_per_newton: float = 200.0
    """Sensor gain; forces are ``(counts - baseline) / counts_per_newton``."""
    shear_counts_per_newton: float | None = None
    """Gain for tangential axes of tri-axial skins (defaults to ``counts_per_newton``)."""
    drift_alpha: float = 1e-3
    contact_threshold_n: float | None = None
    """Contact when normal force exceeds this; default is 6 sigma of the idle noise."""
    contact_noise_sigmas: float = 6.0
    rows: int | None = None
    cols: int | None = None
    pitch_m: float = 4.0e-3
    positions_xy_m: list[list[float]] | None = None
    """Explicit taxel positions ``(N, 2)`` in metres for non-grid layouts."""
    pressure_map_shape: tuple[int, int] | None = None
    """Emit an interpolated ``(T, rows, cols)`` pressure heat-map at this resolution."""
    normal_axis: int = 2
    """For tri-axial skins: index of the normal component in the last axis."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _infer_grid(n: int) -> tuple[int, int]:
    r = int(math.sqrt(n))
    while r > 1 and n % r:
        r -= 1
    return r, n // r


class TaxelCalibrator:
    """Stateful per-trajectory calibrator for pressure or tri-axial taxel arrays.

    Accepts raw samples shaped ``(T, N)``, ``(T, rows, cols)``, ``(T, N, 3)`` or
    ``(T, rows, cols, 3)``. Tri-axial input yields shear forces; single-axis
    input yields zero shear.
    """

    def __init__(self, config: TaxelConfig | None = None, sensor_name: str = "taxel_array"):
        self.config = config or TaxelConfig()
        self.sensor_name = sensor_name
        self._tare = ZeroTare(self.config.idle_frames)
        self._drift = DriftCompensator(self.config.drift_alpha)
        self._geometry: SensorGeometry | None = None
        self._interp: GridInterpolator | None = None
        self._triaxial = False
        self._n: int | None = None
        self._threshold_n: float | None = self.config.contact_threshold_n

    @property
    def modality(self) -> Modality:
        return Modality.TAXEL

    @property
    def geometry(self) -> SensorGeometry:
        if self._geometry is None:
            raise RuntimeError("Calibrator has not seen any samples yet")
        return self._geometry

    @property
    def baseline(self) -> np.ndarray:
        if self._tare.baseline is None:
            raise RuntimeError("Baseline not established yet")
        return self._tare.baseline

    def metadata(self) -> SensorMetadata:
        geo = self.geometry
        return SensorMetadata(
            sensor_name=self.sensor_name,
            modality=self.modality,
            num_elements=geo.num_elements,
            layout=dict(geo.layout),
            units={"normal_force": "N", "shear_force": "N", "point_cloud": "m, N"},
            extra={
                "triaxial": self._triaxial,
                "contact_threshold_n": self._threshold_n,
                "config": self.config.to_dict(),
            },
        )

    # ----------------------------------------------------------------- setup
    def _normalize(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim < 2:
            raise ValueError("raw samples must have a leading time axis")
        T = raw.shape[0]
        spatial = raw.shape[1:]
        triaxial = len(spatial) >= 2 and spatial[-1] == 3
        if triaxial:
            n = int(np.prod(spatial[:-1]))
            out = raw.reshape(T, n, 3)
        else:
            n = int(np.prod(spatial))
            out = raw.reshape(T, n)
        if self._n is None:
            self._n = n
            self._triaxial = triaxial
            self._build_geometry(spatial[:-1] if triaxial else spatial)
        elif n != self._n or triaxial != self._triaxial:
            raise ValueError("Taxel sample shape changed within a trajectory")
        return out

    def _build_geometry(self, spatial: tuple[int, ...]) -> None:
        cfg = self.config
        n = self._n or 0
        if cfg.positions_xy_m is not None:
            pos = np.asarray(cfg.positions_xy_m, dtype=np.float64)
            if pos.shape[0] != n:
                raise ValueError(f"positions_xy_m has {pos.shape[0]} rows but data has {n} taxels")
            self._geometry = SensorGeometry.from_positions(pos, pitch_m=cfg.pitch_m)
        else:
            if cfg.rows and cfg.cols:
                rows, cols = cfg.rows, cfg.cols
            elif len(spatial) == 2:
                rows, cols = int(spatial[0]), int(spatial[1])
            else:
                rows, cols = _infer_grid(n)
            if rows * cols != n:
                raise ValueError(f"grid {rows}x{cols} does not match {n} taxels")
            self._geometry = SensorGeometry.grid(rows, cols, cfg.pitch_m)
        if cfg.pressure_map_shape is not None:
            self._interp = GridInterpolator(self._geometry.xy, cfg.pressure_map_shape)

    # --------------------------------------------------------------- process
    def process(self, raw: np.ndarray) -> TactileChunk:
        cfg = self.config
        x = self._normalize(raw)
        T = x.shape[0]

        if not self._tare.ready:
            self._tare.feed(x)
            if not self._tare.ready:
                self._tare.force_ready()
        base = self._tare.baseline
        assert base is not None

        delta = x - base[None]
        if self._triaxial:
            normal_counts = delta[..., cfg.normal_axis]
            tangential = np.delete(delta, cfg.normal_axis, axis=-1)
            shear_gain = cfg.shear_counts_per_newton or cfg.counts_per_newton
            shear = (tangential / np.float32(shear_gain)).astype(np.float32)
        else:
            normal_counts = delta
            shear = np.zeros((T, self._n or 0, 2), dtype=np.float32)
        normal = np.maximum(normal_counts / np.float32(cfg.counts_per_newton), 0.0).astype(
            np.float32
        )

        if self._threshold_n is None:
            noise = self._tare.noise_std
            assert noise is not None
            sigma = noise[..., cfg.normal_axis] if self._triaxial else noise
            self._threshold_n = float(
                max(cfg.contact_noise_sigmas * np.median(sigma) / cfg.counts_per_newton, 1e-6)
            )
        contact = normal > self._threshold_n

        # Slow baseline tracking on free taxels; update for the next chunk.
        new_base = self._drift.update(base, x, contact)
        self._tare.baseline = new_base

        out = TactileChunk()
        out[S.CONTACT_MASK] = contact
        out[S.NORMAL_FORCE] = normal
        out[S.SHEAR_FORCE] = shear
        out[S.POINT_CLOUD] = build_point_cloud(self.geometry, normal, shear)
        if self._interp is not None:
            out[S.PRESSURE_MAP] = self._interp(normal)
        return out
