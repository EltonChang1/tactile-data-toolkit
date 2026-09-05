"""Ingestion: time alignment, quality gate, modality resolution, CSV/tensor readers."""

from __future__ import annotations

import numpy as np
import pytest

from tactile_toolkit.ingest import (
    CsvReader,
    CsvSpec,
    ModalityResolver,
    QaConfig,
    QualityGate,
    TimeAligner,
    open_log,
)
from tactile_toolkit.synthetic import SyntheticReader, SyntheticScenario, write_csv_dir
from tactile_toolkit.types import ChannelInfo, Modality, Stream


def test_nearest_alignment_picks_closest_sample():
    src = Stream(
        "/wrench",
        "geometry_msgs/msg/WrenchStamped",
        np.array([0.0, 0.010, 0.020]),
        np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
    )
    target = np.array([0.002, 0.014, 0.019])
    out = TimeAligner("nearest").align(target, src)
    np.testing.assert_allclose(out.ravel(), [0.0, 1.0, 2.0])


def test_linear_alignment_interpolates_between_neighbors():
    src = Stream(
        "/wrench",
        "geometry_msgs/msg/WrenchStamped",
        np.array([0.0, 1.0]),
        np.array([[0.0, 10.0], [2.0, 20.0]], dtype=np.float32),
    )
    out = TimeAligner("linear").align(np.array([0.25]), src)
    np.testing.assert_allclose(out[0], [0.5, 12.5], atol=1e-6)


def test_linear_alignment_marks_samples_beyond_max_gap():
    src = Stream(
        "/wrench",
        "geometry_msgs/msg/WrenchStamped",
        np.array([0.0, 1.0]),
        np.array([[1.0], [2.0]], dtype=np.float32),
    )
    out = TimeAligner("linear", max_gap_s=0.1).align(np.array([0.5]), src)
    assert np.isnan(out).all()


def test_quality_gate_drops_jitter_duplicates_and_nonfinite():
    ts = np.array([0.0, 1 / 30, 2 / 30, 3 / 30 + 0.010], dtype=np.float64)
    frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    frames[2] = 2
    frames[3] = 3
    wrench = np.ones((4, 6), dtype=np.float32)
    wrench[2, 0] = np.nan
    keep, report = QualityGate(QaConfig(max_jitter_s=0.002, nominal_period_s=1 / 30)).evaluate(
        ts, frames, {"/wrench": wrench}
    )
    assert report.dropped_duplicate == 1
    assert report.dropped_nonfinite == 1
    assert report.dropped_jitter == 1
    assert int(keep.sum()) == 1


def test_quality_gate_is_stateful_across_chunks():
    gate = QualityGate(QaConfig(max_jitter_s=None, drop_duplicate_frames=True))
    frames_a = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    frames_a[1] = 3
    keep1, _ = gate.evaluate(np.array([0.0, 0.1]), frames_a)
    assert keep1.all()
    frames_b = np.full((1, 4, 4, 3), 3, dtype=np.uint8)
    keep2, report = gate.evaluate(np.array([0.2]), frames_b)
    assert not keep2[0]
    assert report.dropped_duplicate == 1


def test_modality_resolver_uses_message_type_and_overrides():
    channels = [
        ChannelInfo("/cam/image_raw", "sensor_msgs/msg/Image", 10),
        ChannelInfo("/wrist/wrench", "geometry_msgs/msg/WrenchStamped", 100),
        ChannelInfo("/skin/taxels", "std_msgs/msg/Float32MultiArray", 50),
        ChannelInfo("/debug/flags", "std_msgs/msg/String", 4),
    ]
    tm = ModalityResolver().resolve(channels)
    assert tm.master == "/cam/image_raw"
    assert tm.assignments["/wrist/wrench"] is Modality.WRENCH
    assert tm.assignments["/skin/taxels"] is Modality.TAXEL
    assert "/debug/flags" not in tm.assignments

    overridden = ModalityResolver(overrides={"/cam/image_raw": "taxel"}, master="/cam/image_raw")
    tm2 = overridden.resolve(channels)
    assert tm2.master_modality is Modality.TAXEL


def test_csv_reader_roundtrip(tmp_path):
    scenario = SyntheticScenario(
        duration_s=0.4,
        image_fps=20.0,
        image_height=32,
        image_width=40,
        include_gelsight=False,
        include_taxels=True,
        include_wrench=True,
        include_pose=False,
        taxel_rows=4,
        taxel_cols=4,
        seed=3,
    )
    path = write_csv_dir(tmp_path / "csv_log", scenario)
    with open_log(path) as reader:
        channels = {c.topic: c for c in reader.channels()}
        assert any("taxel" in t or "skin" in t for t in channels)
        stream = reader.read_stream(next(iter(channels)))
        assert len(stream) > 0


def test_npy_reader_with_sidecar_timestamps(tmp_path):
    data = np.arange(12, dtype=np.float32).reshape(6, 2)
    stamps = np.linspace(1.0, 2.0, 6)
    npy = tmp_path / "taxels.npy"
    np.save(npy, data)
    np.save(tmp_path / "taxels_timestamps.npy", stamps)
    reader = CsvReader(npy, spec=CsvSpec(rate_hz=30.0))
    stream = reader.read_stream("/taxels")
    assert stream.data.shape == (6, 2)
    np.testing.assert_allclose(stream.timestamps, stamps)


def test_unknown_format_raises(tmp_path):
    from tactile_toolkit.ingest import UnsupportedLogError

    mystery = tmp_path / "notes.txt"
    mystery.write_text("not a log")
    with pytest.raises(UnsupportedLogError):
        open_log(mystery)


def test_synthetic_reader_chunking():
    reader = SyntheticReader(
        SyntheticScenario(
            duration_s=0.5,
            image_fps=20.0,
            image_height=32,
            image_width=40,
            include_taxels=False,
            include_wrench=False,
            seed=1,
        )
    )
    chunks = list(reader.iter_chunks("/gelsight/image_raw", 4))
    total = sum(len(c) for c in chunks)
    assert total == reader.scenario.n_image_frames
    assert all(c.data.ndim == 4 for c in chunks)
