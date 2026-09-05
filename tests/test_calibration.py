"""GelSight and taxel calibration plus point-cloud construction."""

from __future__ import annotations

import numpy as np

from tactile_toolkit import schema as S
from tactile_toolkit.calibration.gelsight import GelSightCalibrator, GelSightConfig, poisson_dst
from tactile_toolkit.calibration.gelsight.poisson import poisson_cumsum, rgb_to_gradients
from tactile_toolkit.calibration.geometry import SensorGeometry
from tactile_toolkit.calibration.pointcloud import build_point_cloud
from tactile_toolkit.calibration.taxel import TaxelCalibrator, TaxelConfig
from tactile_toolkit.synthetic import GelSightSim, TaxelSim


def test_poisson_dst_recovers_smooth_indentation():
    h, w = 48, 64
    y, x = np.mgrid[0:h, 0:w]
    z = np.exp(-((x - w / 2) ** 2 + (y - h / 2) ** 2) / (2 * 8.0**2)).astype(np.float32)
    gx = np.zeros_like(z)
    gy = np.zeros_like(z)
    gx[:, 1:] = z[:, 1:] - z[:, :-1]
    gy[1:, :] = z[1:, :] - z[:-1, :]
    rec = poisson_dst(gx, gy)
    rec = rec - rec.mean()
    z0 = z - z.mean()
    corr = np.corrcoef(z0.ravel(), rec.ravel())[0, 1]
    assert corr > 0.9


def test_gelsight_calibrator_emits_schema_arrays():
    sim = GelSightSim(height=64, width=80, seed=2)
    frames = np.stack([f for _, f, _ in sim.iter_frames(12, fps=30.0)])
    cal = GelSightCalibrator(
        GelSightConfig(stride=8, store_depth_map=True, store_raw_image=True, track_markers=True)
    )
    out = cal.process(frames)
    assert out.num_frames == 12
    assert out[S.RAW_IMAGE].shape == (12, 64, 80, 3)
    assert out[S.DEPTH_MAP].shape == (12, 64, 80)
    n = cal.geometry.num_elements
    assert out[S.NORMAL_FORCE].shape == (12, n)
    assert out[S.SHEAR_FORCE].shape == (12, n, 2)
    assert out[S.POINT_CLOUD].shape == (12, n, 6)
    assert out[S.CONTACT_MASK].dtype == np.bool_
    meta = cal.metadata()
    assert meta.modality.value == "vision_tactile"
    assert meta.num_elements == n


def test_gelsight_cumsum_method_runs():
    sim = GelSightSim(height=32, width=40, seed=0)
    frames = np.stack([f for _, f, _ in sim.iter_frames(4, fps=30.0)])
    gx, gy = rgb_to_gradients(frames, frames[0])
    depth = poisson_cumsum(gx[0], gy[0])
    assert depth.shape == (32, 40)


def test_taxel_calibrator_zero_tare_and_pressure_map():
    sim = TaxelSim(rows=8, cols=8, axes=1, seed=4)
    raw = np.stack([sim.sample(t) for t in np.linspace(0, 1, 30)])
    cal = TaxelCalibrator(
        TaxelConfig(idle_frames=8, rows=8, cols=8, pressure_map_shape=(16, 16), pitch_m=0.004)
    )
    out = cal.process(raw)
    assert out[S.NORMAL_FORCE].shape == (30, 64)
    assert out[S.SHEAR_FORCE].shape == (30, 64, 2)
    assert out[S.PRESSURE_MAP].shape == (30, 16, 16)
    assert np.isfinite(out[S.NORMAL_FORCE]).all()
    # Idle frames should sit near zero after tare.
    assert float(out[S.NORMAL_FORCE][:8].mean()) < 0.05


def test_taxel_triaxial_yields_shear():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(20, 16, 3)).astype(np.float32)
    raw[..., 2] += 40.0
    cal = TaxelCalibrator(TaxelConfig(idle_frames=5, rows=4, cols=4, counts_per_newton=10.0))
    out = cal.process(raw)
    assert out[S.SHEAR_FORCE].shape == (20, 16, 2)
    assert np.any(out[S.SHEAR_FORCE] != 0)


def test_point_cloud_xyz_matches_geometry():
    geo = SensorGeometry.grid(2, 3, pitch_m=0.01)
    normal = np.ones((4, 6), dtype=np.float32)
    shear = np.zeros((4, 6, 2), dtype=np.float32)
    cloud = build_point_cloud(geo, normal, shear)
    np.testing.assert_allclose(cloud[0, :, :2], geo.xy)
    np.testing.assert_allclose(cloud[..., 5], 1.0)
