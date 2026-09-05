"""Zarr, HDF5, and LeRobot writers plus streaming statistics."""

from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from tactile_toolkit import schema as S
from tactile_toolkit.export import (
    DatasetInfo,
    Hdf5Writer,
    LeRobotV3Writer,
    RunningStats,
    ZarrWriter,
    infer_format,
    open_zarr,
    schema_view,
    zarr_summary,
)
from tactile_toolkit.schema import validate
from tactile_toolkit.types import Modality, TactileChunk


def _chunk(t: int = 8, n: int = 6, h: int = 16, w: int = 24) -> TactileChunk:
    return TactileChunk(
        {
            S.TIMESTAMPS: np.linspace(0, 1, t, dtype=np.float64),
            S.CONTACT_MASK: np.zeros((t, n), dtype=bool),
            S.NORMAL_FORCE: np.linspace(0, 1, t * n, dtype=np.float32).reshape(t, n),
            S.SHEAR_FORCE: np.zeros((t, n, 2), dtype=np.float32),
            S.POINT_CLOUD: np.zeros((t, n, 6), dtype=np.float32),
            S.RAW_IMAGE: np.random.default_rng(0).integers(0, 255, (t, h, w, 3), dtype=np.uint8),
            S.WRENCH: np.zeros((t, 6), dtype=np.float32),
        }
    )


def _info() -> DatasetInfo:
    return DatasetInfo(modalities=[Modality.VISION_TACTILE, Modality.WRENCH], fps=30.0)


def test_infer_format_from_suffix(tmp_path):
    assert infer_format(tmp_path / "a.zarr") == "zarr"
    assert infer_format(tmp_path / "a.h5") == "hdf5"
    dest = tmp_path / "episode"
    dest.mkdir()
    (dest / "meta").mkdir()
    (dest / "meta" / "info.json").write_text("{}")
    assert infer_format(dest) == "lerobot"


def test_zarr_writer_roundtrip_and_schema(tmp_out):
    path = tmp_out / "demo.zarr"
    writer = ZarrWriter(path, chunk_frames=4)
    writer.begin_episode()
    writer.append(_chunk(5))
    writer.append(_chunk(5))
    writer.end_episode()
    writer.finalize(_info())
    root = open_zarr(path)
    report = validate(root, modalities=[Modality.VISION_TACTILE, Modality.WRENCH])
    assert report.ok, report.summary()
    assert int(root.attrs["num_frames"]) == 10
    summary = zarr_summary(path)
    assert "/observation/tactile/normal_force" in summary["arrays"]


def test_hdf5_writer_robomimic_layout(tmp_out):
    path = tmp_out / "demo.h5"
    writer = Hdf5Writer(path, chunk_frames=4)
    writer.begin_episode()
    writer.append(_chunk(6))
    writer.end_episode()
    writer.finalize(_info())
    view = schema_view(path, demo=0)
    report = validate(view, modalities=[Modality.VISION_TACTILE, Modality.WRENCH])
    assert report.ok, report.summary()
    assert view[S.NORMAL_FORCE].shape[0] == 6


def test_lerobot_v3_layout(tmp_out):
    path = tmp_out / "lerobot_ds"
    writer = LeRobotV3Writer(path, fps=30.0, task="grasp")
    writer.begin_episode()
    writer.append(_chunk(6, h=16, w=24))
    writer.end_episode()
    writer.finalize(_info())
    info = json.loads((path / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 6
    assert "observation.images.tactile" in info["features"]
    assert (path / "meta" / "stats.json").exists()
    assert (path / "meta" / "tasks.parquet").exists()
    table = pq.read_table(path / "data" / "chunk-000" / "file-000.parquet")
    assert table.num_rows == 6
    assert "observation.tactile.normal_force" in table.column_names
    videos = list((path / "videos").rglob("*.mp4"))
    assert videos


def test_running_stats_match_numpy():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 3)).astype(np.float32)
    rs = RunningStats()
    rs.update(x[:15])
    rs.update(x[15:])
    result = rs.result()
    np.testing.assert_allclose(result["mean"], x.mean(axis=0), atol=1e-5)
    np.testing.assert_allclose(result["std"], x.std(axis=0), atol=1e-5)
    np.testing.assert_allclose(result["min"], x.min(axis=0))
    np.testing.assert_allclose(result["max"], x.max(axis=0))
    assert result["count"] == 40
