"""Step 07 metadata, acquisition, and local adapters on network-free fixtures."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from tactile_toolkit.datasets import (
    DATASETS,
    LMT_69_METADATA,
    LMT_108_METADATA,
    LMT_184_METADATA,
    LMT_METADATA,
    MENDELEY_NOTEBOOK_FILENAME,
    MENDELEY_NOTEBOOK_SHA256,
    MENDELEY_TEXTURES_FILENAME,
    MENDELEY_TEXTURES_METADATA,
    MENDELEY_TEXTURES_SHA256,
    MENDELEY_TEXTURES_SIZE_BYTES,
    MENDELEY_TEXTURES_URL,
    PENN_HATT_METADATA,
    DatasetAccessError,
    DatasetCache,
    DatasetUnavailableError,
    DatasetValidationError,
    MendeleyTexturesAdapter,
    PennHattAdapter,
    SupportLevel,
    download_mendeley_textures,
    open_dataset,
)
from tactile_toolkit.model import SampleKind
from tactile_toolkit.types import Modality


def _pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def _mendeley_fixture(root: Path, *, value: float = 1.0) -> tuple[Path, Path]:
    texture = root / "multimodal" / "pickles_35" / "texture_02"
    timestamps = np.array([4.0, 4.1, 4.2])
    baro = texture / "full_baro" / "baro0201.pkl"
    imu = texture / "full_imu" / "imu0201.pkl"
    _pickle(
        baro,
        {
            "baro": np.array([value, 2.0, 3.0]),
            "__timestamps_s__": timestamps,
        },
    )
    _pickle(
        imu,
        {
            **{
                name: np.arange(3, dtype=float) + index
                for index, name in enumerate(
                    (
                        "imu_ax",
                        "imu_ay",
                        "imu_az",
                        "imu_gx",
                        "imu_gy",
                        "imu_gz",
                        "imu_mx",
                        "imu_my",
                        "imu_mz",
                    )
                )
            },
            "__timestamps_s__": timestamps,
        },
    )
    return baro, imu


def test_lmt_release_records_are_discoverable_without_a_loader():
    ids = {item.dataset_id for item in DATASETS.list()}
    assert {
        "lmt-textures",
        "lmt-textures-69",
        "lmt-textures-108",
        "lmt-textures-184",
    } <= ids
    assert LMT_METADATA.support_level is SupportLevel.DISCOVERABLE
    assert LMT_METADATA.license.allows_redistribution is None
    assert LMT_69_METADATA.version == "1.4"
    assert LMT_69_METADATA.approximate_size_bytes == 6_934_000_000
    assert LMT_108_METADATA.approximate_size_bytes == 3_190_000_000
    assert LMT_184_METADATA.approximate_size_bytes == 83_484_000_000
    with pytest.raises(DatasetUnavailableError, match="has no adapter"):
        open_dataset("lmt-184")


def test_penn_hatt_reads_documented_xml_channels_and_units(tmp_path):
    recording = tmp_path / "RecordedData" / "Foam" / "trial_01.xml"
    recording.parent.mkdir(parents=True)
    recording.write_text(
        """<?xml version="1.0"?>
<Recording>
  <SampleRate>10000</SampleRate>
  <AccelUnits>m/s^2</AccelUnits>
  <ForceUnits>N</ForceUnits>
  <PositionUnits>mm</PositionUnits>
  <SpeedUnits>mm/s</SpeedUnits>
  <Accel><x>1 2 3</x><y>4 5 6</y><z>7 8 9</z></Accel>
  <Force><x>1,2,3</x><y>4,5,6</y><z>7,8,9</z></Force>
  <ForceNormal>10; 11; 12</ForceNormal>
  <ForceTangential>0.1; 0.2; 0.3</ForceTangential>
  <Position><x>0 1 2</x><y>3 4 5</y><z>6 7 8</z></Position>
  <Speed>30 30 30</Speed>
</Recording>
""",
        encoding="utf-8",
    )

    sample = next(PennHattAdapter(tmp_path).iter_samples())
    assert sample.kind is SampleKind.SEQUENCE
    assert sample.labels == {"material": "Foam"}
    assert sample.group_id == "RecordedData/Foam/trial_01"
    assert set(sample.observations) == {
        "acceleration",
        "force",
        "force.normal_tangential",
        "position",
        "speed",
    }
    assert sample.observations["acceleration"].modality is Modality.VIBRATION
    assert sample.observations["force"].unit == "N"
    np.testing.assert_allclose(sample.observations["speed"].timestamps, [0, 0.0001, 0.0002])
    assert PENN_HATT_METADATA.license.allows_commercial_use is False
    assert PennHattAdapter(tmp_path).validate(limit=1).ok


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("<!DOCTYPE foo><Recording/>", "DTD/entities"),
        ("<Recording><Accel><x>1</x></Accel></Recording>", "x, y, and z"),
        (
            "<Recording><AccelX>1</AccelX><AccelY>2</AccelY><AccelZ>3</AccelZ></Recording>",
            "SampleRate",
        ),
    ],
)
def test_penn_hatt_rejects_unsafe_or_malformed_recordings(tmp_path, payload, message):
    path = tmp_path / "material" / "recording.xml"
    path.parent.mkdir()
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(DatasetValidationError, match=message):
        list(PennHattAdapter(tmp_path).iter_samples())


def test_penn_hatt_rejects_splits_and_unrecognized_roots(tmp_path):
    with pytest.raises(DatasetAccessError, match="canonical"):
        next(PennHattAdapter(tmp_path).iter_samples(split="train"))
    with pytest.raises(DatasetAccessError, match="No documented"):
        list(PennHattAdapter(tmp_path).iter_samples())


def test_mendeley_metadata_uses_verified_artifact_facts():
    assert MENDELEY_TEXTURES_METADATA.support_level is SupportLevel.LOADABLE
    assert MENDELEY_TEXTURES_METADATA.license.spdx_id == "CC-BY-4.0"
    assert MENDELEY_TEXTURES_METADATA.checksums[MENDELEY_TEXTURES_FILENAME] == (
        MENDELEY_TEXTURES_SHA256
    )
    assert MENDELEY_TEXTURES_METADATA.checksums[MENDELEY_NOTEBOOK_FILENAME] == (
        MENDELEY_NOTEBOOK_SHA256
    )
    assert MENDELEY_TEXTURES_SIZE_BYTES == 3_831_798_836
    assert MENDELEY_TEXTURES_METADATA.approximate_size_bytes == 3_831_828_701
    assert "30, 35, and 40 mm/s" in MENDELEY_TEXTURES_METADATA.provenance.notes


def test_mendeley_download_requires_confirmation_and_passes_integrity(monkeypatch, tmp_path):
    with pytest.raises(DatasetAccessError, match="confirm_large_download=True"):
        download_mendeley_textures(DatasetCache(tmp_path))

    expected = tmp_path / MENDELEY_TEXTURES_FILENAME
    captured: dict[str, object] = {}

    def fake_download(self, url, dataset_id, **kwargs):
        captured.update(url=url, dataset_id=dataset_id, **kwargs)
        return expected

    monkeypatch.setattr(DatasetCache, "download", fake_download)
    result = download_mendeley_textures(
        DatasetCache(tmp_path), confirm_large_download=True, timeout_s=45
    )
    assert result == expected
    assert captured == {
        "url": MENDELEY_TEXTURES_URL,
        "dataset_id": "mendeley-tactile-textures",
        "filename": MENDELEY_TEXTURES_FILENAME,
        "sha256": MENDELEY_TEXTURES_SHA256,
        "size_bytes": MENDELEY_TEXTURES_SIZE_BYTES,
        "timeout_s": 45,
    }


def test_mendeley_adapter_requires_trust_and_loads_primary_pairs(tmp_path):
    _mendeley_fixture(tmp_path)
    with pytest.raises(DatasetAccessError, match="trust_pickle=True"):
        next(MendeleyTexturesAdapter(tmp_path).iter_samples())

    sample = next(MendeleyTexturesAdapter(tmp_path, trust_pickle=True).iter_samples())
    assert sample.sample_id == "pickles_35/texture_02/0201"
    assert sample.labels == {
        "texture": "texture_02",
        "texture_index": 2,
        "exploration_speed_mm_s": 35,
        "trial": "0201",
    }
    assert sample.group_id == sample.sample_id
    assert sample.observations["pressure.barometer"].data.shape == (3, 1)
    assert sample.observations["imu"].data.shape == (3, 9)
    np.testing.assert_allclose(sample.observations["imu"].timestamps, [0, 0.1, 0.2])
    assert MendeleyTexturesAdapter(tmp_path, trust_pickle=True).validate(limit=1).ok


def test_mendeley_adapter_rejects_missing_pair_and_nonfinite_data(tmp_path):
    _, imu = _mendeley_fixture(tmp_path)
    imu.unlink()
    with pytest.raises(DatasetAccessError, match="paired pickle is missing"):
        list(MendeleyTexturesAdapter(tmp_path, trust_pickle=True).iter_samples())

    tmp_path_2 = tmp_path / "second"
    _mendeley_fixture(tmp_path_2, value=float("nan"))
    with pytest.raises(DatasetValidationError, match="non-finite"):
        list(MendeleyTexturesAdapter(tmp_path_2, trust_pickle=True).iter_samples())


def test_mendeley_adapter_rejects_splits(tmp_path):
    with pytest.raises(DatasetAccessError, match="canonical"):
        next(MendeleyTexturesAdapter(tmp_path, trust_pickle=True).iter_samples(split="test"))
