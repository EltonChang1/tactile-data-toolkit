"""Lazy local adapter for the official Touch100k release layout."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

from tactile_toolkit.datasets.base import DatasetAdapter
from tactile_toolkit.datasets.errors import DatasetAccessError, DatasetValidationError
from tactile_toolkit.datasets.metadata import (
    AccessKind,
    AccessSource,
    Citation,
    DatasetMetadata,
    LicenseInfo,
    ProvenanceInfo,
    SupportLevel,
)
from tactile_toolkit.datasets.registry import dataset_adapter
from tactile_toolkit.model import AssetReference, TactileObservation, TactileSample
from tactile_toolkit.types import Modality

TOUCH100K_SCHEMA_REVISION = "839dec7b1e1745c9ced0436d48c8d14cc4df8ec2"
TOUCH100K_DATA_URL = (
    "https://drive.google.com/drive/folders/1QOvbkIZtpJpz4Ry_Zg3ouXX-zLJNqu9m?usp=sharing"
)

TOUCH100K_METADATA = DatasetMetadata(
    dataset_id="touch100k",
    name="Touch100k",
    description=(
        "A touch-language-vision corpus for learning GelSight representations aligned with rich "
        "sentence-level and compact phrase-level tactile descriptions."
    ),
    homepage="https://cocacola-lab.github.io/Touch100k/",
    version="unversioned Google Drive release",
    provenance=ProvenanceInfo(
        publisher="CoLa Lab, Beijing Jiaotong University",
        source_url="https://cocacola-lab.github.io/Touch100k/",
        creators=(
            "Ning Cheng",
            "Changhao Guan",
            "Jing Gao",
            "Weihao Wang",
            "You Li",
            "Fandong Meng",
            "Jie Zhou",
            "Bin Fang",
            "Jinan Xu",
            "Wenjuan Han",
        ),
        institutions=(
            "Beijing Jiaotong University",
            "WeChat AI, Tencent Inc.",
            "Beijing University of Posts and Telecommunications",
        ),
        derived_from=("Touch and Go (TAG)", "VisGel"),
        notes=(
            "The paper reports 91,982 TAG observations and 10,000 VisGel observations before "
            "language generation and filtering."
        ),
    ),
    license=LicenseInfo(
        name="Creative Commons Attribution-NonCommercial 4.0 International",
        spdx_id="CC-BY-NC-4.0",
        url=(
            f"https://github.com/cocacola-lab/TLV-Link/blob/{TOUCH100K_SCHEMA_REVISION}/README.md"
        ),
        allows_redistribution=True,
        allows_commercial_use=False,
        requires_attribution=True,
        notes=(
            "The official repository declares CC BY-NC 4.0 for data, MIT for code, and "
            "research-only use for models trained on the dataset."
        ),
    ),
    citations=(
        Citation(
            title=(
                "Touch100k: A Large-Scale Touch-Language-Vision Dataset for Touch-Centric "
                "Multimodal Representation"
            ),
            url="https://arxiv.org/abs/2406.03813",
            doi="10.48550/arXiv.2406.03813",
            authors=(
                "Ning Cheng",
                "Changhao Guan",
                "Jing Gao",
                "Weihao Wang",
                "You Li",
                "Fandong Meng",
                "Jie Zhou",
                "Bin Fang",
                "Jinan Xu",
                "Wenjuan Han",
            ),
            year=2024,
        ),
    ),
    access=(
        AccessSource(
            AccessKind.GOOGLE_DRIVE,
            TOUCH100K_DATA_URL,
            notes=(
                "Manual publisher-controlled release; on 2026-09-06 the folder described itself "
                "as partially uploaded and exposed 30 manifest rows."
            ),
        ),
    ),
    modalities=(Modality.VISION_TACTILE, Modality.VISION, Modality.LANGUAGE),
    sensors=("GelSight (TAG and VisGel source datasets)", "conventional RGB camera"),
    tasks=(
        "touch-centric multimodal representation learning",
        "material property identification",
        "grasp stability prediction",
    ),
    formats=("json lines", "jpeg"),
    support_level=SupportLevel.LOADABLE,
    limitations=(
        "The paper reports 100,147 final samples, but the publisher's current Google Drive folder "
        "is explicitly partial; the adapter reports only records actually present locally.",
        "The data release has no immutable revision, total byte count, or published checksums; "
        "the pinned revision identifies the official loader schema, not the data bytes.",
        "Descriptions were generated with GPT-4V and then filtered and manually corrected; they "
        "remain generated language rather than direct human tactile reports.",
        "Per-row source-dataset and object labels are absent from the official manifest schema.",
    ),
    aliases=("touch-100k", "tlv-link-touch100k"),
)

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg"})
_MANIFEST_FIELDS = frozenset({"img", "sentence_desc", "phrase_desc"})


def _relative_path(value: str | Path, field_name: str) -> PurePosixPath:
    raw = str(value).strip()
    if not raw or "\\" in raw or "\x00" in raw:
        raise DatasetValidationError(f"Touch100k {field_name} must be a safe POSIX relative path")
    path = PurePosixPath(raw)
    if path.is_absolute() or path == PurePosixPath(".") or ".." in path.parts:
        raise DatasetValidationError(f"Touch100k {field_name} must be a safe POSIX relative path")
    return path


def _required_text(row: dict[str, Any], field_name: str, line_number: int) -> str:
    value = row.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(
            f"Touch100k manifest line {line_number} field {field_name!r} must be non-empty text"
        )
    return value.strip()


def _asset(root: Path, relative: PurePosixPath, *, role: str) -> AssetReference:
    candidate = root.joinpath(*relative.parts)
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise DatasetValidationError(f"Touch100k {role} asset escapes the dataset root: {relative}")
    if not candidate.is_file():
        raise DatasetValidationError(f"Touch100k {role} image is missing: {candidate}")
    return AssetReference(
        relative.as_posix(),
        media_type="image/jpeg",
        size_bytes=candidate.stat().st_size,
    )


@dataset_adapter(TOUCH100K_METADATA)
class Touch100kAdapter(DatasetAdapter):
    """Yield paired images and two language granularities from ``data_list.json``."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        manifest: str | Path = "data_list.json",
        touch_directory: str | Path = "touch",
        vision_directory: str | Path = "vision",
        max_line_bytes: int = 1_000_000,
    ):
        super().__init__(root)
        self.manifest = _relative_path(manifest, "manifest")
        self.touch_directory = _relative_path(touch_directory, "touch_directory")
        self.vision_directory = _relative_path(vision_directory, "vision_directory")
        if max_line_bytes <= 0:
            raise ValueError("max_line_bytes must be positive")
        self.max_line_bytes = max_line_bytes

    @staticmethod
    def _group_id(image: PurePosixPath) -> str:
        stem = image.stem
        return stem.split("__", 1)[0] if "__" in stem else stem

    def iter_samples(self, *, split: str | None = None) -> Iterator[TactileSample]:
        if split not in {None, "train"}:
            raise DatasetAccessError(
                "Touch100k publishes one training corpus without canonical validation/test "
                f"splits; use split='train' or None, not {split!r}"
            )
        root = self.require_root()
        if not root.is_dir():
            raise DatasetAccessError(f"Touch100k root must be a directory: {root}")
        manifest_path = root.joinpath(*self.manifest.parts)
        if not manifest_path.resolve().is_relative_to(root.resolve()):
            raise DatasetValidationError(
                f"Touch100k manifest escapes the dataset root: {manifest_path}"
            )
        if not manifest_path.is_file():
            raise DatasetAccessError(
                f"Touch100k manifest is missing: {manifest_path}; expected data_list.json beside "
                "the touch/ and vision/ directories"
            )

        seen: set[str] = set()
        emitted = 0
        try:
            handle = manifest_path.open("rb")
        except OSError as exc:
            raise DatasetAccessError(
                f"Could not open Touch100k manifest {manifest_path}: {exc}"
            ) from exc
        with handle:
            line_number = 0
            while True:
                raw_line = handle.readline(self.max_line_bytes + 1)
                if not raw_line:
                    break
                line_number += 1
                if len(raw_line) > self.max_line_bytes:
                    raise DatasetValidationError(
                        f"Touch100k manifest line {line_number} exceeds "
                        f"max_line_bytes={self.max_line_bytes}"
                    )
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise DatasetValidationError(
                        f"Invalid Touch100k JSON on manifest line {line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise DatasetValidationError(
                        f"Touch100k manifest line {line_number} must contain an object"
                    )

                image = _relative_path(
                    _required_text(row, "img", line_number),
                    f"manifest line {line_number} image",
                )
                if image.suffix.lower() not in _IMAGE_SUFFIXES:
                    raise DatasetValidationError(
                        f"Touch100k manifest line {line_number} references unsupported image "
                        f"format {image.suffix!r}; expected JPEG"
                    )
                image_id = image.as_posix()
                if image_id in seen:
                    raise DatasetValidationError(
                        f"Duplicate Touch100k image identifier on manifest line {line_number}: "
                        f"{image_id}"
                    )
                seen.add(image_id)

                sentence = _required_text(row, "sentence_desc", line_number)
                phrase = _required_text(row, "phrase_desc", line_number)
                touch_relative = self.touch_directory / image
                vision_relative = self.vision_directory / image
                extras = {key: value for key, value in row.items() if key not in _MANIFEST_FIELDS}
                emitted += 1
                yield TactileSample(
                    sample_id=image.with_suffix("").as_posix(),
                    observations={
                        "touch": TactileObservation(
                            Modality.VISION_TACTILE,
                            _asset(root, touch_relative, role="touch"),
                            sensor="GelSight (TAG or VisGel; source not declared per row)",
                            encoding="JPEG",
                        ),
                        "vision": TactileObservation(
                            Modality.VISION,
                            _asset(root, vision_relative, role="vision"),
                            encoding="JPEG",
                        ),
                        "language.sentence": TactileObservation(
                            Modality.LANGUAGE,
                            sentence,
                            encoding="publisher sentence-level description",
                        ),
                        "language.phrase": TactileObservation(
                            Modality.LANGUAGE,
                            phrase,
                            encoding="publisher phrase-level description",
                        ),
                    },
                    labels={"image_id": image_id, "manifest_extra": extras},
                    task="touch-centric multimodal representation learning",
                    split="train",
                    group_id=self._group_id(image),
                    metadata={
                        "manifest": self.manifest.as_posix(),
                        "manifest_line": line_number,
                        "loader_schema_revision": TOUCH100K_SCHEMA_REVISION,
                        "grouping_strategy": "recording prefix before '__', else image stem",
                    },
                )

        if emitted == 0:
            raise DatasetValidationError(f"Touch100k manifest has no data rows: {manifest_path}")
