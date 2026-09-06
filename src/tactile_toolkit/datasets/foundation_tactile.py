"""Streaming local WebDataset adapter for FoTa / FoundationTactile."""

from __future__ import annotations

import json
import tarfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from tactile_toolkit.datasets.base import DatasetAdapter
from tactile_toolkit.datasets.errors import DatasetAccessError, DatasetValidationError
from tactile_toolkit.datasets.huggingface import HuggingFaceSnapshot
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

FOUNDATION_TACTILE_REVISION = "e1a16123575eb26e789cf0129ced6f3ba081f2ed"
FOUNDATION_TACTILE_SOURCE = HuggingFaceSnapshot(
    repo_id="alanz-mit/FoundationTactile",
    revision=FOUNDATION_TACTILE_REVISION,
    approximate_size_bytes=397_000_000_000,
    browser_url="https://huggingface.co/datasets/alanz-mit/FoundationTactile",
)

FOUNDATION_TACTILE_METADATA = DatasetMetadata(
    dataset_id="foundation-tactile",
    name="FoTa / FoundationTactile",
    description=(
        "A unified representation-learning corpus aggregating heterogeneous camera-based "
        "tactile images and task labels."
    ),
    homepage=FOUNDATION_TACTILE_SOURCE.browser_url,
    version="FoTa release",
    revision=FOUNDATION_TACTILE_REVISION,
    provenance=ProvenanceInfo(
        publisher="MIT CSAIL",
        source_url=FOUNDATION_TACTILE_SOURCE.browser_url,
        creators=("Jialiang Zhao", "Yuxiang Ma", "Lirui Wang", "Edward H. Adelson"),
        institutions=("MIT CSAIL",),
        derived_from=(
            "VisGel",
            "Touch-Vision-Language",
            "Touch and Go",
            "Calandra et al. 2017",
            "Yuan et al. 2018",
            "YCB-Sight",
            "ObjectFolder-Real",
            "FoTa in-house collections",
        ),
        notes="Sensor and constituent-source identity must remain attached to every sample.",
    ),
    license=LicenseInfo(
        name="MIT License",
        spdx_id="MIT",
        url=f"{FOUNDATION_TACTILE_SOURCE.browser_url}/blob/{FOUNDATION_TACTILE_REVISION}/README.md",
        allows_redistribution=True,
        allows_commercial_use=True,
        requires_attribution=True,
        notes=(
            "The aggregate dataset card declares MIT; constituent-source provenance remains "
            "necessary when reusing individual samples."
        ),
    ),
    citations=(
        Citation(
            title=(
                "Transferable Tactile Transformers for Representation Learning Across Diverse "
                "Sensors and Tasks"
            ),
            url="https://arxiv.org/abs/2406.13640",
            doi="10.48550/arXiv.2406.13640",
            authors=("Jialiang Zhao", "Yuxiang Ma", "Lirui Wang", "Edward H. Adelson"),
            year=2024,
        ),
    ),
    access=(AccessSource(AccessKind.HUGGING_FACE, FOUNDATION_TACTILE_SOURCE.browser_url),),
    modalities=(Modality.VISION_TACTILE,),
    sensors=("13 publisher-defined camera-based tactile sensor domains",),
    tasks=(
        "masked image reconstruction",
        "object and material classification",
        "textile property classification",
        "pose estimation",
        "grasp outcome prediction",
        "semantic description",
    ),
    formats=("split zip", "webdataset tar", "jpeg", "json", "txt"),
    support_level=SupportLevel.LOADABLE,
    approximate_size_bytes=FOUNDATION_TACTILE_SOURCE.approximate_size_bytes,
    limitations=(
        "The hosted data are a multi-volume ZIP that must be fully downloaded and reassembled "
        "before shard-level access.",
        "JSON label schemas vary by constituent dataset and are preserved without coercion.",
        "FoTa is imbalanced and its two largest sensor domains contribute over half the images.",
    ),
    aliases=("fota", "foundationtactile", "foundation-tactile-fota"),
)

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg"})
_SPLITS = frozenset({"train", "val"})


def _safe_member_name(name: str, shard: Path) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise DatasetValidationError(f"Unsafe FoTa TAR member {name!r} in {shard}")
    return path


def _label_text(labels: Mapping[str, Any], names: Sequence[str]) -> str | None:
    for name in names:
        value = labels.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _group_text(labels: Mapping[str, Any]) -> str | None:
    for name in ("group_id", "trajectory_id", "episode_id", "object_id"):
        value = labels.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
    return None


@dataset_adapter(FOUNDATION_TACTILE_METADATA)
class FoundationTactileAdapter(DatasetAdapter):
    """Yield paired JPEG/JSON records from extracted FoTa WebDataset shards."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        sources: Sequence[str] | None = None,
        sensor_by_source: Mapping[str, str] | None = None,
        max_json_bytes: int = 1_000_000,
    ):
        super().__init__(root)
        if isinstance(sources, str):
            sources = (sources,)
        self.sources = frozenset(value.strip() for value in sources or () if value.strip())
        self.sensor_by_source = dict(sensor_by_source or {})
        if max_json_bytes <= 0:
            raise ValueError("max_json_bytes must be positive")
        self.max_json_bytes = max_json_bytes

    @staticmethod
    def _shards(split_dir: Path) -> list[Path]:
        return sorted(
            path
            for path in split_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".tar"
            and (path.name.startswith("data-") or path.name.startswith("data_"))
        )

    def _split_directories(self, root: Path, split: str | None) -> list[Path]:
        requested = {split} if split is not None else set(_SPLITS)
        if not requested <= _SPLITS:
            choices = ", ".join(sorted(_SPLITS))
            raise DatasetAccessError(f"FoTa split must be one of {choices}; received {split!r}")
        candidates: list[Path] = []
        if root.name in requested and self._shards(root):
            candidates.append(root)
        for path in root.rglob("*"):
            if path.is_dir() and path.name in requested and self._shards(path):
                candidates.append(path)
        selected = []
        for path in sorted(set(candidates)):
            source = path.parent.name
            if not self.sources or source in self.sources:
                selected.append(path)
        if not selected:
            qualifier = f" for sources {sorted(self.sources)}" if self.sources else ""
            raise DatasetAccessError(
                f"No extracted FoTa WebDataset shards were found under {root}{qualifier}; "
                "expected <source>/<train|val>/data-*.tar"
            )
        return selected

    @staticmethod
    def _declared_count(split_dir: Path) -> int | None:
        count_path = split_dir / "count.txt"
        if not count_path.exists():
            return None
        try:
            value = int(count_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            raise DatasetValidationError(f"Invalid FoTa count file {count_path}: {exc}") from exc
        if value < 0:
            raise DatasetValidationError(f"FoTa count must be non-negative: {count_path}")
        return value

    def _read_labels(
        self, archive: tarfile.TarFile, member: tarfile.TarInfo, shard: Path
    ) -> dict[str, Any]:
        if member.size > self.max_json_bytes:
            raise DatasetValidationError(
                f"FoTa JSON member {member.name!r} in {shard} exceeds "
                f"max_json_bytes={self.max_json_bytes}"
            )
        handle = archive.extractfile(member)
        if handle is None:
            raise DatasetValidationError(
                f"Could not read FoTa JSON member {member.name!r} in {shard}"
            )
        try:
            value = json.loads(handle.read(self.max_json_bytes + 1).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatasetValidationError(
                f"Invalid FoTa JSON member {member.name!r} in {shard}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise DatasetValidationError(
                f"FoTa JSON member {member.name!r} in {shard} must contain an object"
            )
        return value

    def _sample(
        self,
        *,
        root: Path,
        split_dir: Path,
        shard: Path,
        key: str,
        image: tarfile.TarInfo | None,
        labels: dict[str, Any] | None,
        declared_count: int | None,
    ) -> TactileSample:
        if image is None or labels is None:
            missing = "JPEG" if image is None else "JSON"
            raise DatasetValidationError(
                f"FoTa sample {key!r} in {shard} is missing its paired {missing} member"
            )
        source = split_dir.parent.name
        sensor = _label_text(labels, ("sensor", "sensor_name", "sensor_type"))
        sensor_origin = "json"
        if sensor is None:
            sensor = self.sensor_by_source.get(source, f"FoTa source: {source}")
            sensor_origin = (
                "source mapping" if source in self.sensor_by_source else "source directory"
            )
        task = _label_text(labels, ("task", "task_name", "label_task"))
        split = split_dir.name
        shard_relative = shard.relative_to(root).as_posix()
        sample_id = f"{source}/{split}/{shard.stem}/{key}"
        return TactileSample(
            sample_id=sample_id,
            observations={
                "touch": TactileObservation(
                    Modality.VISION_TACTILE,
                    AssetReference(
                        shard_relative,
                        member=image.name,
                        media_type="image/jpeg",
                        size_bytes=image.size,
                    ),
                    sensor=sensor,
                    encoding="JPEG in WebDataset TAR",
                )
            },
            labels=labels,
            task=task,
            split=split,
            group_id=_group_text(labels),
            metadata={
                "source_revision": FOUNDATION_TACTILE_REVISION,
                "source_dataset": source,
                "sensor_identity_source": sensor_origin,
                "shard": shard_relative,
                "json_member": f"{key}.json",
                "declared_split_count": declared_count,
            },
        )

    def _iter_shard(
        self, root: Path, split_dir: Path, shard: Path, declared_count: int | None
    ) -> Iterator[TactileSample]:
        current_key: str | None = None
        image: tarfile.TarInfo | None = None
        labels: dict[str, Any] | None = None
        try:
            with tarfile.open(shard, mode="r|*") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    path = _safe_member_name(member.name, shard)
                    suffix = path.suffix.lower()
                    if suffix not in _IMAGE_SUFFIXES and suffix != ".json":
                        continue
                    key = str(path.with_suffix(""))
                    if current_key is not None and key != current_key:
                        yield self._sample(
                            root=root,
                            split_dir=split_dir,
                            shard=shard,
                            key=current_key,
                            image=image,
                            labels=labels,
                            declared_count=declared_count,
                        )
                        image = None
                        labels = None
                    current_key = key
                    if suffix in _IMAGE_SUFFIXES:
                        if image is not None:
                            raise DatasetValidationError(
                                f"Duplicate FoTa JPEG member for {key!r} in {shard}"
                            )
                        image = member
                    else:
                        if labels is not None:
                            raise DatasetValidationError(
                                f"Duplicate FoTa JSON member for {key!r} in {shard}"
                            )
                        labels = self._read_labels(archive, member, shard)
                if current_key is not None:
                    yield self._sample(
                        root=root,
                        split_dir=split_dir,
                        shard=shard,
                        key=current_key,
                        image=image,
                        labels=labels,
                        declared_count=declared_count,
                    )
        except DatasetValidationError:
            raise
        except (OSError, tarfile.TarError) as exc:
            raise DatasetValidationError(f"Could not stream FoTa shard {shard}: {exc}") from exc

    def iter_samples(self, *, split: str | None = None) -> Iterator[TactileSample]:
        root = self.require_root()
        for split_dir in self._split_directories(root, split):
            declared_count = self._declared_count(split_dir)
            for shard in self._shards(split_dir):
                yield from self._iter_shard(root, split_dir, shard, declared_count)
