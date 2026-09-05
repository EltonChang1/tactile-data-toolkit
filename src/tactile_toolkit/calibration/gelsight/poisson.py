"""Photometric gradient decoding and fast Poisson height reconstruction.

The gel camera observes a colour change proportional to the local surface
slope. Gradients are integrated into a height map by solving the Poisson
equation ``laplacian(z) = div(g)`` with zero (Dirichlet) boundary using a
discrete sine transform, which is exact for the 5-point Laplacian and runs in
``O(HW log HW)`` per frame. A cumulative-sum integrator is provided as a fast
but drift-prone alternative.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
from scipy import fft as sfft

DepthMethod = Literal["dst", "cumsum"]


def rgb_to_gradients(
    frames: np.ndarray,
    reference: np.ndarray,
    gain: float = 1.0,
    marker_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode surface slopes ``(gx, gy)`` from colour differences to the reference.

    The forward model is ``R - R_ref = gain * gx * 255`` and
    ``G - G_ref = gain * gy * 255`` with the blue channel as an illumination
    reference; the returned slopes are dimensionless (metres per metre).
    """
    frames = np.asarray(frames)
    reference = np.asarray(reference)
    if frames.ndim == 3:
        frames = frames[None]
    scale = np.float32(1.0 / (255.0 * gain))
    ref = reference.astype(np.float32)
    ref_rb = (ref[..., 0] - ref[..., 2])[None]
    ref_gb = (ref[..., 1] - ref[..., 2])[None]
    blue = frames[..., 2].astype(np.float32)
    gx = frames[..., 0].astype(np.float32)
    gx -= blue
    gx -= ref_rb
    gx *= scale
    gy = frames[..., 1].astype(np.float32)
    gy -= blue
    gy -= ref_gb
    gy *= scale
    if marker_mask is not None:
        # Multiplying by a keep-mask is much cheaper than boolean fancy assignment.
        keep = (~np.asarray(marker_mask, dtype=bool)).astype(np.float32)[None]
        gx *= keep
        gy *= keep
    return gx, gy


def _divergence(
    gx: np.ndarray, gy: np.ndarray, padded: tuple[int, int] | None = None
) -> np.ndarray:
    """Backward-difference divergence, optionally written into a zero-padded buffer."""
    h, w = gx.shape[-2:]
    hp, wp = padded or (h, w)
    full = np.zeros((*gx.shape[:-2], hp, wp), dtype=np.float32)
    f = full[..., :h, :w]
    np.subtract(gx[..., :, 1:], gx[..., :, :-1], out=f[..., :, 1:])
    f[..., :, 0] = gx[..., :, 0]
    f[..., 1:, :] += gy[..., 1:, :]
    f[..., 1:, :] -= gy[..., :-1, :]
    f[..., 0, :] += gy[..., 0, :]
    return full


_DENOM_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _dst_denominator(h: int, w: int) -> np.ndarray:
    key = (h, w)
    d = _DENOM_CACHE.get(key)
    if d is None:
        i = np.arange(1, h + 1, dtype=np.float64)
        j = np.arange(1, w + 1, dtype=np.float64)
        lam_y = 2.0 * np.cos(np.pi * i / (h + 1)) - 2.0
        lam_x = 2.0 * np.cos(np.pi * j / (w + 1)) - 2.0
        d = (lam_y[:, None] + lam_x[None, :]).astype(np.float32)
        _DENOM_CACHE[key] = d
    return d


def _padded_size(n: int) -> int:
    """Smallest ``n' >= n`` such that the DST-I of length ``n'`` (FFT of ``2(n'+1)``) is fast."""
    return sfft.next_fast_len(n + 1, real=True) - 1


def poisson_dst(gx: np.ndarray, gy: np.ndarray, workers: int = -1) -> np.ndarray:
    """Solve ``laplacian(z) = div(g)`` with ``z = 0`` on the boundary. Batched over leading dims.

    The domain is zero-padded to FFT-friendly sizes (moving the Dirichlet
    boundary slightly outwards), which keeps the transforms fast for common
    camera resolutions such as 480x640.
    """
    gx = np.asarray(gx, dtype=np.float32)
    gy = np.asarray(gy, dtype=np.float32)
    h, w = gx.shape[-2:]
    hp, wp = _padded_size(h), _padded_size(w)
    f = _divergence(gx, gy, padded=(hp, wp))
    f_hat = sfft.dstn(f, type=1, axes=(-2, -1), workers=workers, overwrite_x=True)
    f_hat /= _dst_denominator(hp, wp)
    z = sfft.idstn(f_hat, type=1, axes=(-2, -1), workers=workers, overwrite_x=True)
    if (hp, wp) != (h, w):
        z = z[..., :h, :w]
    return np.ascontiguousarray(z, dtype=np.float32)


def poisson_cumsum(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Cumulative integration along rows and columns (fast, accumulates drift)."""
    gx = np.asarray(gx, dtype=np.float32)
    gy = np.asarray(gy, dtype=np.float32)
    z = 0.5 * (np.cumsum(gx, axis=-1) + np.cumsum(gy, axis=-2))
    # Remove the per-frame mean so the no-contact level sits at zero.
    return z - z.mean(axis=(-2, -1), keepdims=True)


def poisson_reconstruct(gx: np.ndarray, gy: np.ndarray, method: DepthMethod = "dst") -> np.ndarray:
    if method == "dst":
        return poisson_dst(gx, gy)
    if method == "cumsum":
        return poisson_cumsum(gx, gy)
    raise ValueError(f"Unknown depth method '{method}'")


def resolve_workers(workers: int, n_items: int, cap: int = 8) -> int:
    """Number of threads to use: ``workers <= 0`` picks ``min(cpu_count, cap)``."""
    if workers <= 0:
        workers = min(os.cpu_count() or 1, cap)
    return max(1, min(workers, n_items))


def _depth_serial(
    frames: np.ndarray,
    reference: np.ndarray,
    pixel_size_m: float,
    gain: float,
    method: DepthMethod,
    marker_mask: np.ndarray | None,
    clip_negative: bool,
) -> np.ndarray:
    gx, gy = rgb_to_gradients(frames, reference, gain=gain, marker_mask=marker_mask)
    if method == "dst":
        z = poisson_dst(gx, gy, workers=1)
    else:
        z = poisson_reconstruct(gx, gy, method=method)
    z *= np.float32(pixel_size_m)
    if clip_negative:
        np.maximum(z, 0.0, out=z)
    return z


def depth_from_frames(
    frames: np.ndarray,
    reference: np.ndarray,
    pixel_size_m: float,
    gain: float = 1.0,
    method: DepthMethod = "dst",
    marker_mask: np.ndarray | None = None,
    clip_negative: bool = True,
    workers: int = 1,
) -> np.ndarray:
    """Indentation depth in metres, shape ``(T, H, W)``.

    With ``workers != 1`` the batch is split along time and processed in
    threads; NumPy and the FFT backend release the GIL on these array sizes,
    so this scales close to linearly on multi-core machines. Results are
    identical to the serial path.
    """
    frames = np.asarray(frames)
    if frames.ndim == 3:
        frames = frames[None]
    T = frames.shape[0]
    n_threads = resolve_workers(workers, max(1, T // 4))
    if n_threads == 1 or T < 8:
        return _depth_serial(
            frames, reference, pixel_size_m, gain, method, marker_mask, clip_negative
        )
    parts = np.array_split(frames, n_threads)
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        outs = list(
            ex.map(
                lambda fr: _depth_serial(
                    fr, reference, pixel_size_m, gain, method, marker_mask, clip_negative
                ),
                parts,
            )
        )
    return np.concatenate(outs, axis=0)
