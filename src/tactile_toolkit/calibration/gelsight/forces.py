"""Map gel deformation (depth + marker shear) to normal and shear force fields."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class ElastomerModel:
    """Linear-elastic gel model converting deformation into force per surface element.

    ``pressure = E * depth / thickness`` and ``shear_stress = G * shear_disp / thickness``;
    multiplying stress by the element area yields force. Defaults approximate a
    soft silicone gel (E ~ 0.3 MPa, G ~ 0.1 MPa, 3 mm thick).
    """

    youngs_modulus_pa: float = 3.0e5
    shear_modulus_pa: float = 1.0e5
    gel_thickness_m: float = 3.0e-3
    contact_depth_threshold_m: float = 5.0e-5

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def normal_force(self, depth_m: np.ndarray, element_area_m2: float) -> np.ndarray:
        k = np.float32(self.youngs_modulus_pa / self.gel_thickness_m * element_area_m2)
        return np.maximum(np.asarray(depth_m, dtype=np.float32), 0.0) * k

    def shear_force(self, shear_disp_m: np.ndarray, element_area_m2: float) -> np.ndarray:
        k = np.float32(self.shear_modulus_pa / self.gel_thickness_m * element_area_m2)
        return np.asarray(shear_disp_m, dtype=np.float32) * k

    def contact_mask(self, depth_m: np.ndarray) -> np.ndarray:
        return np.asarray(depth_m) > self.contact_depth_threshold_m


def pool_mean(x: np.ndarray, stride: int, channels: bool = False) -> np.ndarray:
    """Average-pool the spatial axes of ``(..., H, W)`` or ``(..., H, W, C)`` by ``stride``."""
    if stride <= 1:
        return x
    x = np.asarray(x)
    if channels:
        *lead, H, W, C = x.shape
        h, w = H // stride, W // stride
        v = x[..., : h * stride, : w * stride, :]
        v = v.reshape(*lead, h, stride, w, stride, C)
        return v.mean(axis=(-4, -2), dtype=np.float32)
    *lead, H, W = x.shape
    h, w = H // stride, W // stride
    v = x[..., : h * stride, : w * stride]
    v = v.reshape(*lead, h, stride, w, stride)
    return v.mean(axis=(-3, -1), dtype=np.float32)


def force_field(
    depth_m: np.ndarray,
    shear_disp_px: np.ndarray | None,
    pixel_size_m: float,
    model: ElastomerModel,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-element ``(normal_force (T,N), shear_force (T,N,2), indentation (T,N))``.

    Inputs are full-resolution ``(T, H, W)`` depth and ``(T, H, W, 2)`` marker
    displacement (pixels); outputs are pooled to ``stride`` and flattened in
    row-major order to match :meth:`SensorGeometry.from_image`. The shear
    displacement may also be supplied already pooled, i.e. with spatial shape
    ``(H // stride, W // stride)``, in which case it is used as is.
    """
    depth_m = np.asarray(depth_m, dtype=np.float32)
    T = depth_m.shape[0]
    pooled_depth = pool_mean(depth_m, stride)
    area = (pixel_size_m * stride) ** 2
    normal = model.normal_force(pooled_depth, area).reshape(T, -1)
    if shear_disp_px is None:
        shear = np.zeros((T, normal.shape[1], 2), dtype=np.float32)
    else:
        disp = np.asarray(shear_disp_px, dtype=np.float32)
        if disp.shape[1:3] == pooled_depth.shape[1:3]:
            pooled_disp = disp
        else:
            pooled_disp = pool_mean(disp, stride, channels=True)
        shear = model.shear_force(pooled_disp * np.float32(pixel_size_m), area).reshape(T, -1, 2)
    return normal, shear, pooled_depth.reshape(T, -1)


def dense_force_map(
    depth_m: np.ndarray,
    shear_disp_px: np.ndarray | None,
    pixel_size_m: float,
    model: ElastomerModel,
) -> np.ndarray:
    """Full-resolution ``(T, H, W, 3)`` field ``(fx, fy, fz)`` in Newtons per pixel."""
    normal, shear, _ = force_field(depth_m, shear_disp_px, pixel_size_m, model, stride=1)
    T, H, W = np.asarray(depth_m).shape
    out = np.empty((T, H, W, 3), dtype=np.float32)
    out[..., 0:2] = shear.reshape(T, H, W, 2)
    out[..., 2] = normal.reshape(T, H, W)
    return out
