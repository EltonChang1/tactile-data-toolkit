"""Step 06 discovery and local adapters on tiny, network-free fixtures."""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tactile_toolkit.datasets import (
    CLAMP_FILTERED_MD5,
    CLAMP_FILTERED_SIZE_BYTES,
    CLAMP_METADATA,
    DATASETS,
    META_TACBENCH_METADATA,
    OAHD_METADATA,
    TACBENCH_DIGIT_FORCE_REVISION,
    TACBENCH_DIGIT_POSE_REVISION,
    TACBENCH_FORCE_SLIP_METADATA,
    TACBENCH_POSE_METADATA,
    TACVERSE_METADATA,
    TACVERSE_REVISION,
    TACVERSE_SOURCE,
    TVL_CORRECTED_REVISION,
    TVL_METADATA,
    ClampFilteredAdapter,
    DatasetAccessError,
    DatasetUnavailableError,
    DatasetValidationError,
    SupportLevel,
    TacBenchForceSlipAdapter,
    TacBenchPoseAdapter,
    TacVerseAdapter,
    open_dataset,
)
from tactile_toolkit.model import AssetReference, SampleKind
from tactile_toolkit.types import Modality

REPO_ROOT = Path(__file__).parents[1]


def _write_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def _image(value: int) -> np.ndarray:
    return np.full((3, 4, 3), value, dtype=np.uint8)


def test_tvl_and_oahd_are_discoverable_without_overclaiming_terms():
    assert TVL_METADATA.revision == TVL_CORRECTED_REVISION
    assert TVL_METADATA.support_level is SupportLevel.DISCOVERABLE
    assert TVL_METADATA.license.allows_redistribution is None
    assert TVL_METADATA.approximate_size_bytes == 75_292_565_517
    assert OAHD_METADATA.support_level is SupportLevel.DISCOVERABLE
    assert OAHD_METADATA.provenance.publisher == "Georgia Tech Healthcare Robotics Lab"
    assert OAHD_METADATA.license.allows_redistribution is None

    with pytest.raises(DatasetUnavailableError, match="has no adapter"):
        open_dataset("touch-vision-language")
    with pytest.raises(DatasetUnavailableError, match="has no adapter"):
        open_dataset("open-access-haptic-database")


def test_step06_metadata_and_pinned_sources_are_registered():
    ids = {item.dataset_id for item in DATASETS.list()}
    assert {
        "tvl",
        "meta-tacbench",
        "tacbench-force-slip",
        "tacbench-pose",
        "tacverse",
        "oahd",
        "clamp",
    } <= ids
    assert META_TACBENCH_METADATA.support_level is SupportLevel.DISCOVERABLE
    assert "bead-maze manipulation" in META_TACBENCH_METADATA.tasks
    assert TACBENCH_FORCE_SLIP_METADATA.support_level is SupportLevel.LOADABLE
    assert TACBENCH_POSE_METADATA.support_level is SupportLevel.LOADABLE
    assert TACVERSE_METADATA.revision == TACVERSE_REVISION
    assert TACVERSE_SOURCE.approximate_size_bytes == 29_404_983_381
    assert CLAMP_METADATA.version == "Harvard Dataverse 1.0"
    assert CLAMP_METADATA.approximate_size_bytes == 6_385_413_968
    assert CLAMP_FILTERED_SIZE_BYTES == 525_694_981
    assert CLAMP_FILTERED_MD5 == "a5de0bbce85b5cf69db73e13029913c7"


def test_tacverse_acquisition_requires_explicit_large_download_confirmation(tmp_path):
    with pytest.raises(DatasetAccessError, match="confirm_large_download=True"):
        TACVERSE_SOURCE.download(tmp_path / "tacverse", token=True)


def test_tacverse_loads_force_shape_and_grating_lazily(tmp_path):
    force = tmp_path / "Force_Regression" / "MagicTac"
    force.mkdir(parents=True)
    (force / "frame_1.jpg").write_bytes(b"fixture")
    (force / "MagicTac.csv").write_text(
        "image_name,fx,fy,fz\nframe_1.jpg,1.0,-2.0,3.5\n", encoding="utf-8"
    )
    shape = tmp_path / "Shape_Classification" / "TacTip" / "sphere"
    shape.mkdir(parents=True)
    (shape / "shape_4.jpg").write_bytes(b"fixture")
    grating = tmp_path / "Grating_Classification" / "ViTacTip" / "Blackdot"
    grating.mkdir(parents=True)
    (grating / "2.0_7_3.jpg").write_bytes(b"fixture")

    samples = list(TacVerseAdapter(tmp_path).iter_samples())
    assert [sample.task for sample in samples] == [
        "force regression",
        "grating classification",
        "shape classification",
    ]
    force_sample, grating_sample, shape_sample = samples
    assert force_sample.labels["force_xyz"] == [1.0, -2.0, 3.5]
    assert force_sample.observations["force.xyz"].modality is Modality.WRENCH
    assert isinstance(force_sample.observations["touch"].data, AssetReference)
    assert grating_sample.labels == {
        "pattern": "Blackdot",
        "pattern_family": "dot",
        "size": 2.0,
        "group": 7,
        "frame": 3,
    }
    assert grating_sample.group_id == "ViTacTip/Blackdot/2.0/7"
    assert shape_sample.labels == {"shape": "sphere"}
    assert shape_sample.observations["touch"].sensor == "TacTip"

    report = TacVerseAdapter(tmp_path).validate(limit=3, check_assets=True)
    assert report.ok, report.summary()
    with pytest.raises(DatasetAccessError, match="does not publish per-sample split"):
        next(TacVerseAdapter(tmp_path).iter_samples(split="train"))


def test_tacverse_rejects_unsafe_and_malformed_native_rows(tmp_path):
    force = tmp_path / "Force_Regression" / "MagicTac"
    force.mkdir(parents=True)
    (force / "MagicTac.csv").write_text(
        "image_name,fx,fy,fz\n../escape.jpg,1,2,3\n", encoding="utf-8"
    )
    with pytest.raises(DatasetValidationError, match="unsafe image_name"):
        list(TacVerseAdapter(tmp_path, tasks=("force",)).iter_samples())

    grating = tmp_path / "Grating_Classification" / "ViTacTip" / "Blackdot"
    grating.mkdir(parents=True)
    (grating / "invalid.jpg").write_bytes(b"fixture")
    with pytest.raises(DatasetValidationError, match="size_group"):
        list(TacVerseAdapter(tmp_path, tasks=("grating",)).iter_samples())

    unknown = tmp_path / "Shape_Classification" / "UnknownSensor" / "sphere"
    unknown.mkdir(parents=True)
    (unknown / "shape.jpg").write_bytes(b"fixture")
    with pytest.raises(DatasetValidationError, match="unknown sensor"):
        list(TacVerseAdapter(tmp_path, tasks=("shape",)).iter_samples())


def test_tacbench_force_slip_requires_trust_and_loads_one_batch(tmp_path):
    batch = tmp_path / "sphere" / "batch_1"
    frames = [_image(10), _image(20), _image(30)]
    _write_pickle(batch / "dataset_digit_00.pkl", frames)
    _write_pickle(
        batch / "dataset_slip_forces.pkl",
        {
            "in_contact": np.array([0, 1, 1]),
            "trajectories": {
                "slide-a": {
                    "indexes": np.array([0, 2]),
                    "forces": np.array([[0.0, 0.0, 1.0], [0.2, -0.3, 2.0]]),
                    "slip_label": np.array([0, 1]),
                }
            },
        },
    )
    with pytest.raises(DatasetAccessError, match="trust_pickle=True"):
        next(TacBenchForceSlipAdapter(tmp_path).iter_samples())

    samples = list(TacBenchForceSlipAdapter(tmp_path, trust_pickle=True).iter_samples())
    assert len(samples) == 2
    assert samples[0].group_id == "sphere/batch_1/slide-a"
    assert samples[1].labels == {"slip": 1, "in_contact": True}
    np.testing.assert_allclose(samples[1].observations["force.xyz"].data, [0.2, -0.3, 2.0])
    assert samples[1].observations["touch"].data[0, 0, 0] == 30
    assert samples[1].metadata["source_revision"] == TACBENCH_DIGIT_FORCE_REVISION
    with pytest.raises(DatasetAccessError, match="does not publish canonical"):
        next(TacBenchForceSlipAdapter(tmp_path, trust_pickle=True).iter_samples(split="test"))


def test_tacbench_force_slip_rejects_out_of_range_frame_indexes(tmp_path):
    batch = tmp_path / "batch_1"
    _write_pickle(batch / "dataset_digit_00.pkl", [_image(1)])
    _write_pickle(
        batch / "dataset_slip_forces.pkl",
        {
            "in_contact": np.array([1]),
            "trajectories": {
                0: {
                    "indexes": np.array([2]),
                    "forces": np.ones((1, 3)),
                    "slip_label": np.zeros(1),
                }
            },
        },
    )
    with pytest.raises(DatasetValidationError, match="out of range"):
        list(TacBenchForceSlipAdapter(tmp_path, trust_pickle=True).iter_samples())


def test_tacbench_force_slip_rejects_nonbinary_labels(tmp_path):
    batch = tmp_path / "batch_1"
    _write_pickle(batch / "dataset_digit_00.pkl", [_image(1)])
    _write_pickle(
        batch / "dataset_slip_forces.pkl",
        {
            "in_contact": np.array([1]),
            "trajectories": {
                0: {
                    "indexes": np.array([0]),
                    "forces": np.ones((1, 3)),
                    "slip_label": np.array([2]),
                }
            },
        },
    )
    with pytest.raises(DatasetValidationError, match="only binary"):
        list(TacBenchForceSlipAdapter(tmp_path, trust_pickle=True).iter_samples())


def test_tacbench_pose_preserves_publisher_split_and_group(tmp_path):
    bag = tmp_path / "train" / "pringles" / "bag_00.pkl"
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 2, axis=0)
    poses[1, 0, 3] = 0.01
    _write_pickle(
        bag,
        {
            "digit_index": [_image(4), _image(5)],
            "object_index_rel_pose_n5": poses,
        },
    )
    with pytest.raises(DatasetAccessError, match="trust_pickle=True"):
        next(TacBenchPoseAdapter(tmp_path).iter_samples())

    samples = list(TacBenchPoseAdapter(tmp_path, trust_pickle=True).iter_samples(split="train"))
    assert len(samples) == 2
    assert samples[0].split == "train"
    assert samples[0].group_id == "train/pringles/bag_00"
    assert samples[0].labels == {"object": "pringles"}
    assert samples[0].observations["pose.relative"].data.shape == (4, 4)
    assert samples[0].metadata["source_revision"] == TACBENCH_DIGIT_POSE_REVISION
    direct = next(TacBenchPoseAdapter(bag, trust_pickle=True).iter_samples(split="train"))
    assert direct.split == "train"
    assert direct.group_id == "train/pringles/bag_00"
    with pytest.raises(DatasetAccessError, match="matched split"):
        next(TacBenchPoseAdapter(tmp_path, trust_pickle=True).iter_samples(split="test"))


def _write_clamp_fixture(path: Path) -> None:
    dataset = np.empty((2, 3, 4), dtype=object)
    dataset[:] = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    lookups = np.empty(1, dtype=object)
    lookups[0] = np.array([0, 1])
    gpt_probs = np.empty(2, dtype=object)
    gpt_probs[:] = [{"paper": 0.1, "porcelain": 0.9}, {"paper": 0.2, "porcelain": 0.8}]
    np.savez(
        path,
        dataset=dataset,
        feature_key=np.array(
            {"Force": 0, "Active thermal": 1, "Contact microphone": 2}, dtype=object
        ),
        filenames=np.array(["device_1/home/cup/trial_1.txt", "device_1/home/cup/trial_2.txt"]),
        material_labels=np.array(["porcelain", "porcelain"], dtype=object),
        gpt_probs=gpt_probs,
        obj_material_labels=np.array(["porcelain"], dtype=object),
        obj_gpt_probs=np.array([[0.15, 0.85]], dtype=object),
        obj_names=np.array(["device_1/home/cup"], dtype=object),
        obj_contact_lookup=lookups,
        img_type=np.array("cropped", dtype=object),
        label_encoding=np.array(["paper", "porcelain"], dtype=object),
    )


def test_clamp_requires_trust_and_yields_grouped_contact_sequences(tmp_path):
    archive = tmp_path / "CLAMP_dataset_filtered.npz"
    _write_clamp_fixture(archive)
    with pytest.raises(DatasetAccessError, match="trust_pickle=True"):
        next(ClampFilteredAdapter(tmp_path).iter_samples())

    samples = list(ClampFilteredAdapter(tmp_path, trust_pickle=True).iter_samples())
    assert len(samples) == 2
    assert samples[0].kind is SampleKind.SEQUENCE
    assert samples[0].group_id == "device_1/home/cup"
    assert samples[0].labels["material"] == "porcelain"
    assert samples[0].labels["vision_material_probabilities"] == pytest.approx(
        {"paper": 0.1, "porcelain": 0.9}
    )
    assert samples[0].modalities == {
        Modality.PRESSURE,
        Modality.TEMPERATURE,
        Modality.VIBRATION,
    }
    assert samples[0].observations["pressure.force"].data.shape == (4, 1)
    assert samples[0].metadata["dataverse_doi"] == "10.7910/DVN/HNS2Z4"
    report = ClampFilteredAdapter(tmp_path, trust_pickle=True).validate(limit=2)
    assert report.ok, report.summary()
    with pytest.raises(DatasetAccessError, match="split by the emitted object"):
        next(ClampFilteredAdapter(tmp_path, trust_pickle=True).iter_samples(split="train"))


def test_clamp_rejects_incomplete_object_grouping(tmp_path):
    archive = tmp_path / "CLAMP_dataset_filtered.npz"
    _write_clamp_fixture(archive)
    with np.load(archive, allow_pickle=True) as original:
        content = {key: original[key] for key in original.files}
    lookup = np.empty(1, dtype=object)
    lookup[0] = np.array([0])
    content["obj_contact_lookup"] = lookup
    np.savez(archive, **content)
    with pytest.raises(DatasetValidationError, match="grouping is incomplete"):
        list(ClampFilteredAdapter(archive, trust_pickle=True).iter_samples())


def test_step06_local_examples_run_on_bounded_fixtures(tmp_path):
    tacverse = tmp_path / "tacverse" / "Shape_Classification" / "TacTip" / "sphere"
    tacverse.mkdir(parents=True)
    (tacverse / "shape_1.jpg").write_bytes(b"fixture")

    pose_root = tmp_path / "pose"
    pose = np.eye(4, dtype=np.float32)[None, ...]
    _write_pickle(
        pose_root / "train" / "sugar" / "bag_00.pkl",
        {"digit_index": [_image(1)], "object_index_rel_pose_n5": pose},
    )

    clamp = tmp_path / "CLAMP_dataset_filtered.npz"
    _write_clamp_fixture(clamp)
    commands = (
        [
            sys.executable,
            str(REPO_ROOT / "examples/datasets/tacverse_local.py"),
            str(tmp_path / "tacverse"),
            "--task",
            "shape",
            "--limit",
            "1",
        ],
        [
            sys.executable,
            str(REPO_ROOT / "examples/datasets/tacbench_local.py"),
            str(pose_root),
            "--task",
            "pose",
            "--trust-pickle",
            "--limit",
            "1",
        ],
        [
            sys.executable,
            str(REPO_ROOT / "examples/datasets/clamp_local.py"),
            str(clamp),
            "--trust-pickle",
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
        assert "contact/00000000" in result.stdout or "vision_tactile" in result.stdout
