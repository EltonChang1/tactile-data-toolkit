"""FreeTacMan and FoTa adapters on tiny, network-free source-shaped fixtures."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tactile_toolkit.datasets import (
    DATASETS,
    FOUNDATION_TACTILE_REVISION,
    FOUNDATION_TACTILE_SOURCE,
    FREETACMAN_REVISION,
    FREETACMAN_SOURCE,
    DatasetAccessError,
    DatasetValidationError,
    FoundationTactileAdapter,
    FreeTacManAdapter,
    HuggingFaceSnapshot,
    open_dataset,
)
from tactile_toolkit.model import AssetReference, SampleKind
from tactile_toolkit.types import Modality

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "datasets"
FREETACMAN_ROOT = FIXTURE_ROOT / "freetacman"
FOTA_MEMBERS = FIXTURE_ROOT / "foundation_tactile"
REPO_ROOT = Path(__file__).parents[1]


def _write_shard(
    root: Path,
    *,
    source: str = "object_folder",
    split: str = "train",
    separator: str = "-",
    entries: Mapping[str, bytes] | None = None,
) -> Path:
    split_dir = root / source / split
    split_dir.mkdir(parents=True, exist_ok=True)
    shard = split_dir / f"data{separator}000000.tar"
    members = entries or {
        "sample.jpg": (FOTA_MEMBERS / "sample.jpg").read_bytes(),
        "sample.json": (FOTA_MEMBERS / "sample.json").read_bytes(),
    }
    with tarfile.open(shard, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    (split_dir / "count.txt").write_text("1\n", encoding="utf-8")
    return shard


def test_dataset_metadata_and_adapters_are_registered_without_network_access():
    assert DATASETS.metadata("freetacman").revision == FREETACMAN_REVISION
    assert DATASETS.metadata("fota").revision == FOUNDATION_TACTILE_REVISION
    assert isinstance(open_dataset("free-tac-man", root=FREETACMAN_ROOT), FreeTacManAdapter)


def test_freetacman_loads_trajectory_views_pose_and_gripper_lazily():
    samples = list(FreeTacManAdapter(FREETACMAN_ROOT).iter_samples(split="train"))
    assert [sample.sample_id for sample in samples] == [
        "ArrangeFruit/ArrangeFruit_0",
        "ArrangeFruit/ArrangeFruit_1",
    ]
    first, second = samples
    assert first.kind is SampleKind.TRAJECTORY
    assert first.split == "train"
    assert first.group_id == "ArrangeFruit_0"
    assert first.metadata["source_revision"] == FREETACMAN_REVISION
    assert first.metadata["camera_roles"] == {
        "camera1": "vision_tactile",
        "camera2": "vision_tactile",
        "camera3": "vision",
    }
    assert isinstance(first.observations["touch.camera1"].data, AssetReference)
    assert first.observations["pose.tcp"].data.shape == (2, 10)
    np.testing.assert_allclose(first.observations["pose.tcp"].timestamps, [1000.0, 1000.1])
    assert first.observations["gripper.distance"].data.shape == (2, 1)
    assert "vision.camera3" not in second.observations

    report = FreeTacManAdapter(FREETACMAN_ROOT).validate(limit=2, check_assets=True)
    assert report.ok, report.summary()


def test_freetacman_supports_one_task_root_and_explicit_camera_roles():
    adapter = FreeTacManAdapter(
        FREETACMAN_ROOT / "ArrangeFruit",
        camera_modalities={
            1: Modality.VISION,
            2: Modality.VISION_TACTILE,
            3: Modality.VISION_TACTILE,
        },
    )
    sample = next(adapter.iter_samples())
    assert "vision.camera1" in sample.observations
    assert "touch.camera2" in sample.observations
    assert "touch.camera3" in sample.observations


def test_freetacman_reports_layout_split_and_trajectory_errors(tmp_path):
    copied = tmp_path / "freetacman"
    shutil.copytree(FREETACMAN_ROOT, copied)
    (copied / "ArrangeFruit" / "ArrangeFruit_0_camera2.mp4").unlink()
    with pytest.raises(DatasetValidationError, match="missing camera files: camera2"):
        list(FreeTacManAdapter(copied).iter_samples())
    with pytest.raises(DatasetAccessError, match="one train split"):
        list(FreeTacManAdapter(copied).iter_samples(split="test"))

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DatasetAccessError, match=r"No FreeTacMan '\*_traj.csv'"):
        list(FreeTacManAdapter(empty).iter_samples())


def test_freetacman_rejects_non_monotonic_timestamps(tmp_path):
    copied = tmp_path / "freetacman"
    shutil.copytree(FREETACMAN_ROOT, copied)
    trajectory = copied / "ArrangeFruit" / "ArrangeFruit_0_traj.csv"
    text = trajectory.read_text(encoding="utf-8").replace("1000.1", "999.9")
    trajectory.write_text(text, encoding="utf-8")
    with pytest.raises(
        DatasetValidationError, match="timestamps must be finite and non-decreasing"
    ):
        next(FreeTacManAdapter(copied).iter_samples())


def test_fota_streams_paired_members_and_preserves_labels(tmp_path):
    shard = _write_shard(tmp_path)
    sample = next(FoundationTactileAdapter(tmp_path).iter_samples(split="train"))
    assert sample.sample_id == "object_folder/train/data-000000/sample"
    assert sample.labels == {
        "task": "object classification",
        "object_id": 7,
        "object_name": "foam block",
        "sensor": "GelSight Mini",
    }
    assert sample.task == "object classification"
    assert sample.group_id == "7"
    assert sample.observations["touch"].sensor == "GelSight Mini"
    reference = sample.observations["touch"].data
    assert isinstance(reference, AssetReference)
    assert reference.uri == shard.relative_to(tmp_path).as_posix()
    assert reference.member == "sample.jpg"
    assert reference.size_bytes == len((FOTA_MEMBERS / "sample.jpg").read_bytes())
    assert sample.metadata["source_revision"] == FOUNDATION_TACTILE_REVISION

    report = FoundationTactileAdapter(tmp_path).validate(limit=1, check_assets=True)
    assert report.ok, report.summary()


def test_fota_filters_sources_supports_val_and_underscore_shards(tmp_path):
    _write_shard(tmp_path, source="first", split="train")
    _write_shard(tmp_path, source="second", split="val", separator="_")
    adapter = FoundationTactileAdapter(
        tmp_path,
        sources=("second",),
        sensor_by_source={"second": "Publisher sensor domain"},
    )
    sample = next(adapter.iter_samples(split="val"))
    assert sample.split == "val"
    assert sample.metadata["source_dataset"] == "second"
    assert sample.observations["touch"].sensor == "GelSight Mini"
    with pytest.raises(DatasetAccessError, match="FoTa split must be one of"):
        list(adapter.iter_samples(split="validation"))


def test_fota_uses_explicit_source_sensor_mapping_when_json_omits_sensor(tmp_path):
    labels = json.dumps({"task_name": "pose estimation", "trajectory_id": "run-4"}).encode()
    _write_shard(
        tmp_path,
        source="cnc_pose",
        entries={"nested/example.jpg": b"image", "nested/example.json": labels},
    )
    sample = next(
        FoundationTactileAdapter(
            tmp_path, sensor_by_source={"cnc_pose": "GelSight Wedge"}
        ).iter_samples()
    )
    assert sample.observations["touch"].sensor == "GelSight Wedge"
    assert sample.metadata["sensor_identity_source"] == "source mapping"
    assert sample.group_id == "run-4"
    assert sample.observations["touch"].data.member == "nested/example.jpg"


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ({"sample.jpg": b"image"}, "missing its paired JSON"),
        ({"../sample.jpg": b"image", "../sample.json": b"{}"}, "Unsafe FoTa TAR member"),
        ({"sample.jpg": b"image", "sample.json": b"[]"}, "must contain an object"),
    ],
)
def test_fota_rejects_incomplete_unsafe_or_non_object_samples(tmp_path, entries, message):
    _write_shard(tmp_path, entries=entries)
    with pytest.raises(DatasetValidationError, match=message):
        list(FoundationTactileAdapter(tmp_path).iter_samples())


def test_fota_rejects_invalid_count_and_oversized_json(tmp_path):
    _write_shard(tmp_path)
    count = tmp_path / "object_folder" / "train" / "count.txt"
    count.write_text("unknown\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Invalid FoTa count file"):
        list(FoundationTactileAdapter(tmp_path).iter_samples())

    count.write_text("1\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="exceeds max_json_bytes"):
        list(FoundationTactileAdapter(tmp_path, max_json_bytes=4).iter_samples())


def test_hugging_face_snapshots_guard_large_downloads_and_pin_revision(tmp_path, monkeypatch):
    for source in (FREETACMAN_SOURCE, FOUNDATION_TACTILE_SOURCE):
        with pytest.raises(DatasetAccessError, match="confirm_large_download=True"):
            source.download(tmp_path / source.repo_id.split("/")[1])

    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(kwargs["local_dir"])

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    destination = FREETACMAN_SOURCE.download(
        tmp_path / "selected-task",
        allow_patterns="ArrangeFruit/*",
        confirm_large_download=True,
    )
    assert destination == tmp_path / "selected-task"
    assert calls == [
        {
            "repo_id": "OpenDriveLab/FreeTacMan",
            "repo_type": "dataset",
            "revision": FREETACMAN_REVISION,
            "local_dir": tmp_path / "selected-task",
            "allow_patterns": "ArrangeFruit/*",
            "token": None,
        }
    ]


def test_hugging_face_snapshot_validates_identity():
    with pytest.raises(ValueError, match="owner/dataset"):
        HuggingFaceSnapshot("missing-owner", "revision", 1, "https://huggingface.co/datasets/x/y")


def test_local_examples_run_end_to_end_on_bounded_fixtures(tmp_path):
    fota_root = tmp_path / "fota"
    _write_shard(fota_root)
    commands = (
        [
            sys.executable,
            str(REPO_ROOT / "examples/datasets/freetacman_local.py"),
            str(FREETACMAN_ROOT),
            "--limit",
            "1",
        ],
        [
            sys.executable,
            str(REPO_ROOT / "examples/datasets/foundation_tactile_local.py"),
            str(fota_root),
            "--split",
            "train",
            "--limit",
            "1",
        ],
    )
    for command in commands:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert "checked samples: 1" in result.stdout
