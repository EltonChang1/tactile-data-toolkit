"""Step 08 reference semantics and community first-sample workflow."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tactile_toolkit.datasets import (
    AWESOME_TOUCH_DATASET_SECTION_URL,
    AWESOME_TOUCH_METADATA,
    AWESOME_TOUCH_REVISION,
    DatasetMetadata,
    DatasetMetadataError,
    DatasetUnavailableError,
    ResourceKind,
    SupportLevel,
    list_datasets,
    open_dataset,
)

REPO_ROOT = Path(__file__).parents[1]
FIRST_SAMPLE = REPO_ROOT / "examples" / "datasets" / "first_sample.py"
TOUCH100K_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "datasets" / "touch100k"

README_ENTRIES = {
    "FreeTacMan (OpenDriveLab)",
    "FoTa / FoundationTactile",
    "Tac2Pose",
    "MIT GelSight research datasets",
    "Touch100k",
    "Touch-Vision-Language / TVL",
    "Meta DIGIT benchmarks / Sparsh TacBench",
    "TacVerse",
    "Open Access Haptic Database / OAHD",
    "CLAMP",
    "LMT Haptic Texture Database",
    "Penn Haptic Texture Toolkit / HaTT",
    "Multimodal Tactile Texture Dataset (Mendeley Data)",
    "Awesome-Touch",
}
MATRIX_SOURCES = {
    "FreeTacMan",
    "FoTa / FoundationTactile",
    "Tac2Pose",
    "MIT GelSight study datasets",
    "Touch100k",
    "TVL",
    "Meta DIGIT / TacBench",
    "TacVerse",
    "OAHD",
    "CLAMP",
    "LMT Haptic Texture Database",
    "Penn HaTT",
    "Mendeley Multimodal Tactile Texture",
    "Awesome-Touch",
}


def test_awesome_touch_is_a_pinned_reference_not_a_dataset():
    assert AWESOME_TOUCH_METADATA.resource_kind is ResourceKind.REFERENCE_INDEX
    assert AWESOME_TOUCH_METADATA.support_level is SupportLevel.DISCOVERABLE
    assert AWESOME_TOUCH_METADATA.modalities == ()
    assert AWESOME_TOUCH_METADATA.sensors == ()
    assert AWESOME_TOUCH_METADATA.revision == AWESOME_TOUCH_REVISION
    assert AWESOME_TOUCH_REVISION in AWESOME_TOUCH_DATASET_SECTION_URL
    assert AWESOME_TOUCH_METADATA.license.spdx_id == "MIT"
    assert "index repository only" in AWESOME_TOUCH_METADATA.license.notes
    with pytest.raises(DatasetUnavailableError, match="has no adapter"):
        open_dataset("awesome-touch")


def test_resource_kind_roundtrips_rejects_unknown_values_and_filters():
    restored = DatasetMetadata.from_dict(AWESOME_TOUCH_METADATA.to_dict())
    assert restored == AWESOME_TOUCH_METADATA

    invalid = AWESOME_TOUCH_METADATA.to_dict()
    invalid["resource_kind"] = "link-list-ish"
    with pytest.raises(DatasetMetadataError, match="Unknown resource kind"):
        DatasetMetadata.from_dict(invalid)

    references = list_datasets(resource_kind="reference_index")
    assert [item.dataset_id for item in references] == ["awesome-touch"]
    collections = {item.dataset_id for item in list_datasets(resource_kind="collection")}
    assert collections == {
        "lmt-textures",
        "meta-tacbench",
        "mit-gelsight-datasets",
        "oahd",
    }
    assert all(
        item.resource_kind is ResourceKind.DATASET
        for item in list_datasets(resource_kind="dataset")
    )


def test_readme_has_every_required_entry_with_exactly_two_sentences():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    section = text.split("### Published datasets and reference sources\n", 1)[1].split(
        "### Dataset guides\n", 1
    )[0]
    entries = re.findall(r"^#### (.+?)\n\n(.+?)(?=\n\n#### |\Z)", section, re.MULTILINE | re.DOTALL)
    assert {heading for heading, _ in entries} == README_ENTRIES
    for heading, body in entries:
        visible = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", body.strip())
        boundaries = re.findall(r"[.!?](?=\s|$)", visible)
        assert len(boundaries) == 2, f"{heading} has {len(boundaries)} sentences: {visible}"
        sentences = re.split(r"(?<=[.!?])\s+", visible)
        assert sentences[1].startswith(("It contains ", "They contain ")), heading


def test_required_source_matrix_is_complete_and_well_formed():
    text = (REPO_ROOT / "docs" / "dataset-coverage.md").read_text(encoding="utf-8")
    section = text.split("## Required-source compatibility matrix\n", 1)[1].split("\n## ", 1)[0]
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in section.splitlines()
        if line.startswith("|") and not line.startswith("| ---")
    ]
    header, *records = rows
    assert header == [
        "Source",
        "Kind",
        "Native unit and packaging",
        "Normalized sample mapping",
        "Access constraint",
        "Current",
        "Implemented boundary or next qualification",
    ]
    assert all(len(row) == len(header) for row in records)
    assert {row[0] for row in records} == MATRIX_SOURCES


def test_first_sample_script_lists_kind_support_and_adapter():
    result = subprocess.run(
        [sys.executable, str(FIRST_SAMPLE), "--list"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "dataset_id\tresource_kind\tsupport\tsample_adapter" in result.stdout
    assert "awesome-touch\treference_index\tdiscoverable\tno" in result.stdout
    assert "touch100k\tdataset\tloadable\tyes" in result.stdout


def test_first_sample_script_runs_fixture_end_to_end():
    result = subprocess.run(
        [
            sys.executable,
            str(FIRST_SAMPLE),
            "touch100k",
            str(TOUCH100K_FIXTURE),
            "--split",
            "train",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["dataset"] == {
        "dataset_id": "touch100k",
        "license": "Creative Commons Attribution-NonCommercial 4.0 International",
        "name": "Touch100k",
        "resource_kind": "dataset",
        "source": "https://cocacola-lab.github.io/Touch100k/",
        "support": "loadable",
    }
    assert payload["sample"]["sample_id"] == "20220601_175211__0000019654"
    assert payload["sample"]["observations"]["touch"]["storage"] == "lazy_asset"
    assert payload["sample"]["observations"]["language.sentence"]["storage"] == "text"
