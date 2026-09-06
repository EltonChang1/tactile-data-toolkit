"""Normalized model coverage for heterogeneous published tactile datasets."""

from __future__ import annotations

import numpy as np
import pytest

from tactile_toolkit import AssetReference, SampleKind, TactileObservation, TactileSample
from tactile_toolkit.model import TACTILE_MODALITIES
from tactile_toolkit.types import Modality


def test_asset_reference_is_lazy_and_json_compatible():
    checksum = "A" * 64
    asset = AssetReference(
        " hf://datasets/example/revision/data.tar ",
        member="train\\0001.jpg",
        media_type="image/jpeg",
        sha256=checksum,
        size_bytes=128,
    )
    assert asset.uri == "hf://datasets/example/revision/data.tar"
    assert asset.member == "train/0001.jpg"
    assert asset.sha256 == checksum.lower()
    assert asset.to_dict()["size_bytes"] == 128


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"uri": ""}, "uri"),
        ({"uri": "archive.tar", "member": "../secret"}, "safe relative"),
        ({"uri": "archive.tar", "sha256": "bad"}, "64 hexadecimal"),
        ({"uri": "archive.tar", "size_bytes": -1}, "non-negative"),
    ],
)
def test_asset_reference_rejects_invalid_descriptors(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AssetReference(**kwargs)


def test_multimodal_frame_keeps_large_touch_asset_lazy():
    sample = TactileSample(
        sample_id="tvl/hct/000001",
        kind="frame",
        task="touch-language alignment",
        split="train",
        group_id="object-17",
        observations={
            "touch": TactileObservation(
                Modality.VISION_TACTILE,
                AssetReference("dataset/touch/000001.jpg", media_type="image/jpeg"),
                sensor="DIGIT",
                timestamp=1.25,
            ),
            "vision": TactileObservation(
                Modality.VISION,
                np.zeros((16, 24, 3), dtype=np.uint8),
                encoding="rgb8",
                timestamp=1.25,
            ),
            "language": TactileObservation(Modality.LANGUAGE, "rough woven fabric"),
        },
        labels={"contact": True},
    )
    assert sample.kind is SampleKind.FRAME
    assert sample.is_lazy
    assert sample.modalities == {
        Modality.VISION_TACTILE,
        Modality.VISION,
        Modality.LANGUAGE,
    }
    assert list(sample.by_modality("vision_tactile")) == ["touch"]


def test_temporal_observation_validates_its_leading_axis():
    observation = TactileObservation(
        Modality.VIBRATION,
        np.zeros((3, 6), dtype=np.float32),
        timestamps=[0.0, 0.1, 0.2],
        unit="m/s^2",
    )
    assert observation.timestamps is not None
    assert observation.timestamps.dtype == np.float64

    with pytest.raises(ValueError, match="leading dimension"):
        TactileObservation(
            Modality.VIBRATION,
            np.zeros((2, 6), dtype=np.float32),
            timestamps=[0.0, 0.1, 0.2],
        )
    with pytest.raises(ValueError, match="non-decreasing"):
        TactileObservation(
            Modality.PRESSURE,
            np.zeros((2,), dtype=np.float32),
            timestamps=[0.2, 0.1],
        )


def test_observation_distinguishes_frame_and_sequence_time():
    with pytest.raises(ValueError, match="not both"):
        TactileObservation(
            Modality.VISION_TACTILE,
            np.zeros((1, 2, 2, 3), dtype=np.uint8),
            timestamp=0.0,
            timestamps=[0.0],
        )
    with pytest.raises(TypeError, match="only valid for language"):
        TactileObservation(Modality.VISION, "not-an-asset-reference")


def test_sample_requires_a_tactile_or_haptic_observation():
    assert Modality.PRESSURE in TACTILE_MODALITIES
    with pytest.raises(ValueError, match="tactile or haptic"):
        TactileSample(
            "vision-only",
            {"vision": TactileObservation(Modality.VISION, np.zeros((2, 2, 3)))},
        )
