"""Lazy adapter for the official TacVerse release layout."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

import numpy as np

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

TACVERSE_REVISION = "0bc27afe0d8f6c878b79e1eb0825255541ccaceb"
TACVERSE_CODE_REVISION = "abe33c1e896d9529f0798cc75e36a63da9753e79"
TACVERSE_SOURCE = HuggingFaceSnapshot(
    repo_id="Lan-2025/Tactile",
    revision=TACVERSE_REVISION,
    approximate_size_bytes=29_404_983_381,
    browser_url="https://huggingface.co/datasets/Lan-2025/Tactile",
)

TACVERSE_METADATA = DatasetMetadata(
    dataset_id="tacverse",
    name="TacVerse",
    description=(
        "A seven-sensor vision-based tactile benchmark for cross-sensor shape, grating, and "
        "force representation learning."
    ),
    homepage="https://lannwei.github.io/Tactile_Database/",
    version="2026 Hugging Face release",
    revision=TACVERSE_REVISION,
    provenance=ProvenanceInfo(
        publisher="TacVerse project authors",
        source_url="https://lannwei.github.io/Tactile_Database/",
        creators=(
            "Lan Wei",
            "Gurmeher Khurana",
            "Sirine Bhouri",
            "Wenhao Hong",
            "Zeyuan Xin",
            "Qingzheng Cong",
            "Wen Fan",
            "Yanzheng Xiang",
            "Dandan Zhang",
        ),
        institutions=(
            "Imperial College London",
            "Queen Mary University of London",
            "King's College London",
        ),
        notes=(
            "The adapter schema follows the official benchmark code at commit "
            f"{TACVERSE_CODE_REVISION}."
        ),
    ),
    license=LicenseInfo(
        name="Creative Commons Attribution 4.0 International",
        spdx_id="CC-BY-4.0",
        url="https://huggingface.co/datasets/Lan-2025/Tactile",
        allows_redistribution=True,
        allows_commercial_use=True,
        requires_attribution=True,
        notes="The data card declares CC BY 4.0; the separate code repository has no license.",
    ),
    citations=(
        Citation(
            title=(
                "TacVerse: A Multi-Sensor Dataset and Benchmark for Cross-Sensor "
                "Vision-Based Tactile Perception"
            ),
            url="https://arxiv.org/abs/2606.25877",
            doi="10.48550/arXiv.2606.25877",
            year=2026,
        ),
    ),
    access=(
        AccessSource(
            AccessKind.HUGGING_FACE,
            "https://huggingface.co/datasets/Lan-2025/Tactile",
            requires_authentication=True,
            requires_acceptance=True,
            notes="Gated contact-sharing flow; the repository contains four large ZIP archives.",
        ),
    ),
    modalities=(Modality.VISION_TACTILE, Modality.WRENCH),
    sensors=(
        "GelSightNoMarker",
        "GelSightMarker",
        "MagicGripper",
        "MagicTac",
        "TacTip",
        "ViTac",
        "ViTacTip",
    ),
    tasks=("shape classification", "grating classification", "force regression"),
    formats=("zip", "jpeg", "csv"),
    support_level=SupportLevel.LOADABLE,
    approximate_size_bytes=TACVERSE_SOURCE.approximate_size_bytes,
    limitations=(
        "The gated snapshot must be accepted and downloaded through the publisher's Hugging "
        "Face flow, then its task archives must be extracted before local loading.",
        "The official benchmark code derives train/validation/test partitions at runtime rather "
        "than shipping per-sample split labels; this adapter preserves samples without inventing "
        "a canonical split.",
        "The recent code repository has no declared code license and is used only as a documented "
        "schema reference.",
    ),
    aliases=("tac-verse", "lan-2025-tactile"),
)

_TASK_DIRECTORIES = {
    "force": "Force_Regression",
    "grating": "Grating_Classification",
    "shape": "Shape_Classification",
}
_SENSORS = frozenset(TACVERSE_METADATA.sensors)


def _safe_relative(root: Path, path: Path, role: str) -> str:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise DatasetValidationError(f"TacVerse {role} escapes the dataset root: {path}")
    return path.relative_to(root).as_posix()


def _image_reference(root: Path, path: Path) -> AssetReference:
    relative = _safe_relative(root, path, "image")
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        raise DatasetValidationError(f"TacVerse image must be JPEG: {path}")
    if not path.is_file():
        raise DatasetValidationError(f"TacVerse image is missing: {path}")
    return AssetReference(relative, media_type="image/jpeg", size_bytes=path.stat().st_size)


@dataset_adapter(TACVERSE_METADATA)
class TacVerseAdapter(DatasetAdapter):
    """Yield lazy frames from extracted force, shape, and grating task directories."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        tasks: tuple[str, ...] | list[str] | set[str] | None = None,
        max_csv_line_bytes: int = 1_000_000,
    ):
        super().__init__(root)
        requested = tuple(tasks) if tasks is not None else tuple(_TASK_DIRECTORIES)
        invalid = sorted(set(requested) - _TASK_DIRECTORIES.keys())
        if invalid:
            raise ValueError(
                f"Unknown TacVerse tasks {invalid}; expected force, grating, and/or shape"
            )
        self.tasks = tuple(dict.fromkeys(requested))
        if not self.tasks:
            raise ValueError("TacVerse tasks must not be empty")
        if max_csv_line_bytes <= 0:
            raise ValueError("max_csv_line_bytes must be positive")
        self.max_csv_line_bytes = max_csv_line_bytes

    @staticmethod
    def _task_root(root: Path, task: str) -> Path | None:
        directory = _TASK_DIRECTORIES[task]
        if root.name == directory:
            return root
        candidate = root / directory
        return candidate if candidate.is_dir() else None

    def _bounded_lines(self, path: Path) -> Iterator[str]:
        try:
            handle = path.open("r", encoding="utf-8-sig", newline="")
        except OSError as exc:
            raise DatasetAccessError(f"Could not open TacVerse CSV {path}: {exc}") from exc
        with handle:
            for line_number, line in enumerate(handle, start=1):
                if len(line.encode("utf-8")) > self.max_csv_line_bytes:
                    raise DatasetValidationError(
                        f"TacVerse CSV {path} line {line_number} exceeds "
                        f"max_csv_line_bytes={self.max_csv_line_bytes}"
                    )
                yield line

    @staticmethod
    def _sensor_directories(task_root: Path) -> Iterator[Path]:
        for sensor_dir in sorted(path for path in task_root.iterdir() if path.is_dir()):
            if sensor_dir.name not in _SENSORS:
                raise DatasetValidationError(
                    f"TacVerse task directory contains unknown sensor {sensor_dir.name!r}; "
                    f"expected one of {sorted(_SENSORS)}"
                )
            yield sensor_dir

    def _force_samples(self, root: Path, task_root: Path) -> Iterator[TactileSample]:
        found_csv = False
        for sensor_dir in self._sensor_directories(task_root):
            csv_path = sensor_dir / f"{sensor_dir.name}.csv"
            if not csv_path.is_file():
                continue
            found_csv = True
            reader = csv.DictReader(self._bounded_lines(csv_path))
            required = {"image_name", "fx", "fy", "fz"}
            raw_fields = reader.fieldnames or ()
            if not all(isinstance(field, str) for field in raw_fields):
                raise DatasetValidationError(f"TacVerse force CSV {csv_path} has an invalid header")
            reader.fieldnames = [field.strip() for field in raw_fields]
            fields = set(reader.fieldnames)
            if not required <= fields:
                raise DatasetValidationError(
                    f"TacVerse force CSV {csv_path} requires columns {sorted(required)}; "
                    f"found {sorted(fields)}"
                )
            seen: set[str] = set()
            for row_number, row in enumerate(reader, start=2):
                image_name = (row.get("image_name") or "").strip()
                posix = PurePosixPath(image_name.replace("\\", "/"))
                if (
                    not image_name
                    or posix.is_absolute()
                    or ".." in posix.parts
                    or len(posix.parts) != 1
                ):
                    raise DatasetValidationError(
                        f"TacVerse force CSV {csv_path} row {row_number} has an unsafe image_name"
                    )
                if image_name in seen:
                    raise DatasetValidationError(
                        f"TacVerse force CSV {csv_path} repeats image_name {image_name!r}"
                    )
                seen.add(image_name)
                try:
                    force = np.asarray(
                        [float(row[axis]) for axis in ("fx", "fy", "fz")], dtype=np.float32
                    )
                except (TypeError, ValueError) as exc:
                    raise DatasetValidationError(
                        f"TacVerse force CSV {csv_path} row {row_number} has invalid force values"
                    ) from exc
                if not np.isfinite(force).all():
                    raise DatasetValidationError(
                        f"TacVerse force CSV {csv_path} row {row_number} has non-finite force"
                    )
                image = sensor_dir / image_name
                relative = _safe_relative(root, image, "force image")
                yield TactileSample(
                    sample_id=f"force/{sensor_dir.name}/{Path(image_name).stem}",
                    observations={
                        "touch": TactileObservation(
                            Modality.VISION_TACTILE,
                            _image_reference(root, image),
                            sensor=sensor_dir.name,
                        ),
                        "force.xyz": TactileObservation(
                            Modality.WRENCH,
                            force,
                            metadata={
                                "axes": ["fx", "fy", "fz"],
                                "unit_status": "not stated in the release manifest",
                            },
                        ),
                    },
                    labels={"force_xyz": force.tolist()},
                    task="force regression",
                    group_id=f"{sensor_dir.name}/{Path(image_name).stem}",
                    metadata={
                        "source_revision": TACVERSE_REVISION,
                        "asset": relative,
                        "split_policy": "not published; use official benchmark code",
                    },
                )
        if not found_csv:
            raise DatasetAccessError(
                f"No TacVerse <sensor>/<sensor>.csv force manifests were found under {task_root}"
            )

    def _shape_samples(self, root: Path, task_root: Path) -> Iterator[TactileSample]:
        found = False
        for sensor_dir in self._sensor_directories(task_root):
            for class_dir in sorted(path for path in sensor_dir.iterdir() if path.is_dir()):
                for image in sorted(class_dir.glob("*.jpg")):
                    found = True
                    yield TactileSample(
                        sample_id=f"shape/{sensor_dir.name}/{class_dir.name}/{image.stem}",
                        observations={
                            "touch": TactileObservation(
                                Modality.VISION_TACTILE,
                                _image_reference(root, image),
                                sensor=sensor_dir.name,
                            )
                        },
                        labels={"shape": class_dir.name},
                        task="shape classification",
                        group_id=f"{sensor_dir.name}/{class_dir.name}/{image.stem}",
                        metadata={
                            "source_revision": TACVERSE_REVISION,
                            "split_policy": "ordered 60/20/20 split is derived by official code",
                        },
                    )
        if not found:
            raise DatasetAccessError(
                f"No TacVerse <sensor>/<shape>/*.jpg samples were found under {task_root}"
            )

    def _grating_samples(self, root: Path, task_root: Path) -> Iterator[TactileSample]:
        found = False
        for sensor_dir in self._sensor_directories(task_root):
            for pattern_dir in sorted(path for path in sensor_dir.iterdir() if path.is_dir()):
                for image in sorted(pattern_dir.glob("*.jpg")):
                    parts = image.stem.split("_")
                    if len(parts) not in {2, 3}:
                        raise DatasetValidationError(
                            f"TacVerse grating filename must be size_group[_frame].jpg: {image}"
                        )
                    try:
                        size = float(parts[0])
                        group = int(parts[1])
                        frame = int(parts[2]) if len(parts) == 3 else 0
                    except ValueError as exc:
                        raise DatasetValidationError(
                            f"TacVerse grating filename has invalid numeric fields: {image}"
                        ) from exc
                    if not math.isfinite(size) or group < 0 or frame < 0:
                        raise DatasetValidationError(
                            f"TacVerse grating filename has invalid values: {image}"
                        )
                    found = True
                    pattern = pattern_dir.name
                    pattern_family = "dot" if "dot" in pattern.lower() else "line"
                    yield TactileSample(
                        sample_id=(f"grating/{sensor_dir.name}/{pattern}/{image.stem}"),
                        observations={
                            "touch": TactileObservation(
                                Modality.VISION_TACTILE,
                                _image_reference(root, image),
                                sensor=sensor_dir.name,
                            )
                        },
                        labels={
                            "pattern": pattern,
                            "pattern_family": pattern_family,
                            "size": size,
                            "group": group,
                            "frame": frame,
                        },
                        task="grating classification",
                        group_id=f"{sensor_dir.name}/{pattern}/{parts[0]}/{group}",
                        metadata={
                            "source_revision": TACVERSE_REVISION,
                            "split_policy": "protocol subsets are derived by official code",
                        },
                    )
        if not found:
            raise DatasetAccessError(
                f"No TacVerse <sensor>/<pattern>/*.jpg samples were found under {task_root}"
            )

    def iter_samples(self, *, split: str | None = None) -> Iterator[TactileSample]:
        if split is not None:
            raise DatasetAccessError(
                "TacVerse does not publish per-sample split labels; pass split=None and reproduce "
                "a named protocol from the pinned official benchmark code"
            )
        root = self.require_root()
        if not root.is_dir():
            raise DatasetAccessError(f"TacVerse root must be an extracted directory: {root}")
        found_task = False
        for task in self.tasks:
            task_root = self._task_root(root, task)
            if task_root is None:
                continue
            found_task = True
            if task == "force":
                yield from self._force_samples(root, task_root)
            elif task == "shape":
                yield from self._shape_samples(root, task_root)
            else:
                yield from self._grating_samples(root, task_root)
        if not found_task:
            expected = ", ".join(_TASK_DIRECTORIES[task] for task in self.tasks)
            raise DatasetAccessError(
                f"No requested TacVerse task directories were found under {root}; "
                f"expected {expected}"
            )
