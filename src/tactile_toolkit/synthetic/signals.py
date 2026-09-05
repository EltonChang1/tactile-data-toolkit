"""Synthetic taxel arrays, wrench, and pose signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TaxelSim:
    """Capacitive/resistive skin patch with per-taxel baseline offsets and slow drift."""

    rows: int = 16
    cols: int = 16
    pitch_m: float = 0.004
    axes: int = 1
    """1 for pressure-only skins, 3 for tri-axial (shear x, shear y, normal) skins."""
    baseline_counts: float = 512.0
    counts_per_newton: float = 200.0
    drift_counts_per_s: float = 0.5
    noise_std_counts: float = 1.5
    seed: int = 1

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._offsets = self.baseline_counts + rng.normal(0, 20, self.num_taxels)
        self._rng = rng

    @property
    def num_taxels(self) -> int:
        return self.rows * self.cols

    def positions(self) -> np.ndarray:
        """Taxel centres ``(N, 2)`` in metres, centred on the patch."""
        ys, xs = np.mgrid[0 : self.rows, 0 : self.cols]
        x = (xs - (self.cols - 1) / 2) * self.pitch_m
        y = (ys - (self.rows - 1) / 2) * self.pitch_m
        return np.stack([x.ravel(), y.ravel()], axis=1)

    def contact_at(self, t: float) -> tuple[np.ndarray, float]:
        """Ground-truth normal force per taxel (Newtons) and total force at time ``t``."""
        pos = self.positions()
        press = np.sin(2 * np.pi * t / 3.0)
        if press <= 0:
            return np.zeros(self.num_taxels), 0.0
        cx = 0.3 * self.cols * self.pitch_m * np.sin(2 * np.pi * t / 7.0)
        cy = 0.3 * self.rows * self.pitch_m * np.cos(2 * np.pi * t / 5.0)
        sigma = 1.5 * self.pitch_m
        r2 = (pos[:, 0] - cx) ** 2 + (pos[:, 1] - cy) ** 2
        force = 0.6 * press * np.exp(-r2 / (2 * sigma**2))
        return force, float(force.sum())

    def sample(self, t: float) -> np.ndarray:
        """Raw counts at time ``t``: ``(N,)`` or ``(N, 3)``."""
        force, _ = self.contact_at(t)
        drift = self.drift_counts_per_s * t
        noise = self._rng.normal(0, self.noise_std_counts, (self.num_taxels, self.axes))
        if self.axes == 1:
            counts = self._offsets + drift + self.counts_per_newton * force + noise[:, 0]
            return counts.astype(np.float32)
        shear_x = 0.15 * force * np.sin(2 * np.pi * t / 4.0)
        shear_y = 0.10 * force * np.cos(2 * np.pi * t / 6.0)
        counts = np.stack(
            [
                self._offsets * 0.0 + self.counts_per_newton * shear_x,
                self._offsets * 0.0 + self.counts_per_newton * shear_y,
                self._offsets + drift + self.counts_per_newton * force,
            ],
            axis=1,
        )
        return (counts + noise).astype(np.float32)

    def generate(
        self, n_samples: int, rate_hz: float, t0: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        ts = t0 + np.arange(n_samples) / rate_hz
        data = np.stack([self.sample(t - t0) for t in ts], axis=0)
        return ts, data


def wrench_at(t: np.ndarray | float, seed: int = 2) -> np.ndarray:
    """Smooth 6-axis wrench ``(..., 6)`` loosely correlated with the synthetic contacts."""
    t = np.asarray(t, dtype=np.float64)
    press = np.clip(np.sin(2 * np.pi * t / 3.0), 0, None)
    fx = 0.3 * np.sin(2 * np.pi * t / 4.0) * press
    fy = 0.2 * np.cos(2 * np.pi * t / 6.0) * press
    fz = -2.5 * press
    tx = 0.02 * np.sin(2 * np.pi * t / 5.0)
    ty = 0.015 * np.cos(2 * np.pi * t / 7.0)
    tz = 0.01 * np.sin(2 * np.pi * t / 9.0)
    w = np.stack([fx, fy, fz, tx, ty, tz], axis=-1)
    rng = np.random.default_rng(seed + int(np.floor(float(np.min(t)) * 1000)) if t.size else seed)
    return (w + rng.normal(0, 0.01, w.shape)).astype(np.float32)


def generate_wrench(
    n_samples: int, rate_hz: float = 1000.0, t0: float = 0.0, seed: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    ts = t0 + np.arange(n_samples) / rate_hz
    return ts, wrench_at(ts - t0, seed=seed)


def pose_at(t: np.ndarray | float) -> np.ndarray:
    """End-effector pose ``(..., 7)`` = (x, y, z, qx, qy, qz, qw) on a slow circle."""
    t = np.asarray(t, dtype=np.float64)
    x = 0.4 + 0.05 * np.sin(2 * np.pi * t / 10.0)
    y = 0.05 * np.cos(2 * np.pi * t / 10.0)
    z = 0.2 - 0.002 * np.clip(np.sin(2 * np.pi * t / 3.0), 0, None)
    yaw = 0.1 * np.sin(2 * np.pi * t / 12.0)
    qz = np.sin(yaw / 2)
    qw = np.cos(yaw / 2)
    zeros = np.zeros_like(x)
    return np.stack([x, y, z, zeros, zeros, qz, qw], axis=-1).astype(np.float32)


def generate_pose(
    n_samples: int, rate_hz: float = 100.0, t0: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    ts = t0 + np.arange(n_samples) / rate_hz
    return ts, pose_at(ts - t0)
