"""Touch100k loading and honest Tac2Pose/MIT GelSight discovery records."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tactile_toolkit.datasets import (
    DATASETS,
    MIT_GELSIGHT_DATASETS_METADATA,
    MIT_GELSIGHT_FORCE_SHEAR_SLIP_METADATA,
    MIT_GELSIGHT_HARDNESS_METADATA,
    MIT_GELSIGHT_NEURAL_SLIP_METADATA,
    TAC2POSE_METADATA,
    TOUCH100K_SCHEMA_REVISION,
    DatasetAccessError,
    DatasetUnavailableError,
    DatasetValidationError,
    SupportLevel,
    Touch100kAdapter,
    open_dataset,
)
from tactile_toolkit.model import AssetReference, SampleKind
from tactile_toolkit.types import Modality

REPO_ROOT = Path(__file__).parents[1]
TOUCH100K_ROOT = Path(__file__).parent / "fixtures" / "datasets" / "touch100k"


def test_tac2pose_and_mit_gelsight_studies_are_discoverable_not_loadable():
    records = (
        TAC2POSE_METADATA,
        MIT_GELSIGHT_DATASETS_METADATA,
        MIT_GELSIGHT_HARDNESS_METADATA,
        MIT_GELSIGHT_FORCE_SHEAR_SLIP_METADATA,
        MIT_GELSIGHT_NEURAL_SLIP_METADATA,
    )
    assert {item.dataset_id for item in records} <= {
        item.dataset_id for item in DATASETS.list(minimum_support="discoverable")
    }
    assert all(item.support_level is SupportLevel.DISCOVERABLE for item in records)
    assert all(item.license.allows_redistribution is None for item in records)
    assert MIT_GELSIGHT_HARDNESS_METADATA.formats[0] == "avi"

    with pytest.raises(DatasetUnavailableError, match="has no adapter") as error:
        open_dataset("tac2pose")
    assert "mcube.mit.edu" in str(error.value)


def test_touch100k_loads_images_and_multigranularity_language_lazily():
    samples = list(Touch100kAdapter(TOUCH100K_ROOT).iter_samples(split="train"))
    assert [sample.sample_id for sample in samples] == [
        "20220601_175211__0000019654",
        "2_rec_02245_frame0214",
    ]
    first = samples[0]
    assert first.kind is SampleKind.FRAME
    assert first.split == "train"
    assert first.group_id == "20220601_175211"
    assert first.modalities == {
        Modality.VISION_TACTILE,
        Modality.VISION,
        Modality.LANGUAGE,
    }
    assert first.observations["language.sentence"].data.startswith("The fingers contact")
    assert first.observations["language.phrase"].data.startswith("fine and compact")
    assert isinstance(first.observations["touch"].data, AssetReference)
    assert first.observations["touch"].data.uri.startswith("touch/")
    assert first.observations["vision"].data.uri.startswith("vision/")
    assert first.labels["manifest_extra"] == {"fixture_note": "synthetic paraphrase"}
    assert first.metadata["loader_schema_revision"] == TOUCH100K_SCHEMA_REVISION

    report = Touch100kAdapter(TOUCH100K_ROOT).validate(split="train", limit=2, check_assets=True)
    assert report.ok, report.summary()
    assert report.checked_samples == 2


def test_touch100k_is_registered_and_has_no_implicit_acquisition():
    adapter = open_dataset("touch-100k", root=TOUCH100K_ROOT)
    assert isinstance(adapter, Touch100kAdapter)
    assert adapter.metadata.license.spdx_id == "CC-BY-NC-4.0"
    assert adapter.metadata.approximate_size_bytes is None
    assert adapter.metadata.revision is None
    assert adapter.metadata.access[0].kind.value == "google_drive"

    with pytest.raises(DatasetAccessError, match="requires a local root"):
        next(Touch100kAdapter().iter_samples())
    with pytest.raises(DatasetAccessError, match="without canonical validation/test splits"):
        next(Touch100kAdapter(TOUCH100K_ROOT).iter_samples(split="validation"))


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (
            '{"img": "../escape.jpg", "sentence_desc": "sentence", "phrase_desc": "phrase"}\n',
            "safe POSIX relative path",
        ),
        (
            '{"img": "sample.png", "sentence_desc": "sentence", "phrase_desc": "phrase"}\n',
            "unsupported image format",
        ),
        (
            '{"img": "sample.jpg", "sentence_desc": "", "phrase_desc": "phrase"}\n',
            "must be non-empty text",
        ),
        ('{"img": "sample.jpg", "sentence_desc": "sentence"}\n', "must be non-empty text"),
        ("not-json\n", "Invalid Touch100k JSON"),
        ("[]\n", "must contain an object"),
    ],
)
def test_touch100k_rejects_unsafe_or_malformed_manifest_rows(tmp_path, manifest, message):
    (tmp_path / "data_list.json").write_text(manifest, encoding="utf-8")
    with pytest.raises(DatasetValidationError, match=message):
        list(Touch100kAdapter(tmp_path).iter_samples())


def test_touch100k_rejects_missing_pairs_duplicates_and_oversized_rows(tmp_path):
    shutil.copytree(TOUCH100K_ROOT, tmp_path, dirs_exist_ok=True)
    missing = tmp_path / "vision" / "20220601_175211__0000019654.jpg"
    missing.unlink()
    with pytest.raises(DatasetValidationError, match="vision image is missing"):
        list(Touch100kAdapter(tmp_path).iter_samples())

    shutil.copy2(TOUCH100K_ROOT / "vision" / missing.name, missing)
    manifest = tmp_path / "data_list.json"
    first = manifest.read_text(encoding="utf-8").splitlines()[0]
    manifest.write_text(f"{first}\n{first}\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Duplicate Touch100k image identifier"):
        list(Touch100kAdapter(tmp_path).iter_samples())
    with pytest.raises(DatasetValidationError, match="exceeds max_line_bytes"):
        list(Touch100kAdapter(tmp_path, max_line_bytes=20).iter_samples())


def test_touch100k_rejects_missing_or_empty_manifest(tmp_path):
    with pytest.raises(DatasetAccessError, match="manifest is missing"):
        list(Touch100kAdapter(tmp_path).iter_samples())
    (tmp_path / "data_list.json").write_text("\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="has no data rows"):
        list(Touch100kAdapter(tmp_path).iter_samples())


def test_touch100k_example_runs_on_network_free_fixture():
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "examples/datasets/touch100k_local.py"),
            str(TOUCH100K_ROOT),
            "--limit",
            "1",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert "checked samples: 1" in result.stdout
    assert "vision_tactile" in result.stdout
