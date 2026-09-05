"""Assemble the dense ``(T, N, 6)`` contact point cloud ``(x, y, z, fx, fy, fz)``."""

from __future__ import annotations

import numpy as np

from tactile_toolkit.calibration.geometry import SensorGeometry


def build_point_cloud(
    geometry: SensorGeometry,
    normal_force: np.ndarray,
    shear_force: np.ndarray,
    indentation_m: np.ndarray | None = None,
) -> np.ndarray:
    """Combine geometry and per-element forces into a point cloud.

    Parameters
    ----------
    geometry:
        Rest positions of the ``N`` elements.
    normal_force:
        ``(T, N)`` Newtons along the surface normal (positive into the sensor).
    shear_force:
        ``(T, N, 2)`` Newtons in the surface plane.
    indentation_m:
        Optional ``(T, N)`` indentation depth; displaces points along ``-z``.
    """
    normal_force = np.asarray(normal_force, dtype=np.float32)
    shear_force = np.asarray(shear_force, dtype=np.float32)
    T, N = normal_force.shape
    if shear_force.shape != (T, N, 2):
        raise ValueError(f"shear_force must be (T, N, 2); got {shear_force.shape}")
    if geometry.num_elements != N:
        raise ValueError(f"geometry has {geometry.num_elements} elements but forces have {N}")

    cloud = np.empty((T, N, 6), dtype=np.float32)
    cloud[:, :, 0:3] = geometry.positions_m[None, :, :]
    if indentation_m is not None:
        indentation_m = np.asarray(indentation_m, dtype=np.float32)
        if indentation_m.shape != (T, N):
            raise ValueError(f"indentation_m must be (T, N); got {indentation_m.shape}")
        cloud[:, :, 2] -= indentation_m
    cloud[:, :, 3:5] = shear_force
    cloud[:, :, 5] = normal_force
    return cloud
