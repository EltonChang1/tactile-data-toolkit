"""Lazy local adapter for the official FreeTacMan release."""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Iterator, Mapping
from pathlib import Path

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
from tactile_toolkit.model import AssetReference, SampleKind, TactileObservation, TactileSample
from tactile_toolkit.types import Modality

FREETACMAN_REVISION = "030316fb41d6fa1e58cccb4bbe0f6fddbb932671"
FREETACMAN_SOURCE = HuggingFaceSnapshot(
    repo_id="OpenDriveLab/FreeTacMan",
    revision=FREETACMAN_REVISION,
    approximate_size_bytes=50_300_000_000,
    browser_url="https://huggingface.co/datasets/OpenDriveLab/FreeTacMan",
)

FREETACMAN_METADATA = DatasetMetadata(
    dataset_id="freetacman",
    name="FreeTacMan",
    description=(
        "OpenDriveLab's robot-free visuo-tactile manipulation trajectories with synchronized "
        "camera views, tool-center-point tracking, and gripper state."
    ),
    homepage="https://opendrivelab.com/FreeTacMan",
    version="ICRA 2026 release",
    revision=FREETACMAN_REVISION,
    provenance=ProvenanceInfo(
        publisher="OpenDriveLab",
        source_url=FREETACMAN_SOURCE.browser_url,
        creators=(
            "Longyan Wu",
            "Checheng Yu",
            "Jieji Ren",
            "Li Chen",
            "Yufei Jiang",
            "Ran Huang",
            "Guoying Gu",
            "Hongyang Li",
        ),
        institutions=(
            "Shanghai Innovation Institute",
            "The University of Hong Kong",
            "Shanghai Jiao Tong University",
        ),
    ),
    license=LicenseInfo(
        name="MIT License",
        spdx_id="MIT",
        url=FREETACMAN_SOURCE.browser_url,
        allows_redistribution=True,
        allows_commercial_use=True,
        requires_attribution=True,
        notes="Dataset license declared by the official Hugging Face dataset card.",
    ),
    citations=(
        Citation(
            title=(
                "FreeTacMan: Robot-free Visuo-Tactile Data Collection System for "
                "Contact-rich Manipulation"
            ),
            url="https://arxiv.org/abs/2506.01941",
            doi="10.48550/arXiv.2506.01941",
            authors=(
                "Longyan Wu",
                "Checheng Yu",
                "Jieji Ren",
                "Li Chen",
                "Yufei Jiang",
                "Ran Huang",
                "Guoying Gu",
                "Hongyang Li",
            ),
            year=2026,
        ),
    ),
    access=(AccessSource(AccessKind.HUGGING_FACE, FREETACMAN_SOURCE.browser_url),),
    modalities=(
        Modality.VISION_TACTILE,
        Modality.VISION,
        Modality.POSE,
        Modality.JOINT_STATE,
    ),
    sensors=(
        "FreeTacMan camera-based tactile sensors",
        "wrist-mounted fisheye camera",
        "OptiTrack motion-capture system",
    ),
    tasks=(
        "contact-rich manipulation",
        "visuo-tactile representation learning",
        "imitation learning",
    ),
    formats=("mp4", "csv"),
    support_level=SupportLevel.LOADABLE,
    approximate_size_bytes=FREETACMAN_SOURCE.approximate_size_bytes,
    limitations=(
        "The pinned hosted snapshot has 46 task directories although the paper and dataset card "
        "describe 50 tasks.",
        "The adapter exposes MP4 files lazily and does not invent force, contact, or calibration.",
        "Camera 3 is absent from some trajectories and camera filename capitalization varies.",
    ),
    aliases=("free-tac-man", "opendrivelab-freetacman"),
)

_VIDEO_NAME = re.compile(r"^(?P<trajectory>.+)_camera(?P<camera>[123])\.mp4$", re.IGNORECASE)
_TRAJECTORY_NAME = re.compile(r"^(?P<trajectory>.+)_traj\.csv$", re.IGNORECASE)
_CSV_COLUMNS = (
    "timestamp",
    "TCP_pos_x",
    "TCP_pos_y",
    "TCP_pos_z",
    "TCP_euler_x",
    "TCP_euler_y",
    "TCP_euler_z",
    "gripper_distance",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
)
_POSE_COLUMNS = (
    "TCP_pos_x",
    "TCP_pos_y",
    "TCP_pos_z",
    "TCP_euler_x",
    "TCP_euler_y",
    "TCP_euler_z",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
)
_DEFAULT_CAMERA_MODALITIES = {
    1: Modality.VISION_TACTILE,
    2: Modality.VISION_TACTILE,
    3: Modality.VISION,
}


def _relative_asset(path: Path, root: Path) -> AssetReference:
    return AssetReference(
        path.relative_to(root).as_posix(),
        media_type="video/mp4",
        size_bytes=path.stat().st_size,
    )


def _read_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = [column for column in _CSV_COLUMNS if column not in (reader.fieldnames or ())]
            if missing:
                raise DatasetValidationError(
                    f"FreeTacMan trajectory {path} is missing columns: {', '.join(missing)}"
                )
            rows = [
                [float(row[column]) for column in _CSV_COLUMNS]
                for row in reader
                if any(value and value.strip() for value in row.values())
            ]
    except DatasetValidationError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise DatasetValidationError(
            f"Could not parse FreeTacMan trajectory {path}: {exc}"
        ) from exc

    if not rows:
        raise DatasetValidationError(f"FreeTacMan trajectory has no data rows: {path}")
    values = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(values).all():
        raise DatasetValidationError(f"FreeTacMan trajectory contains non-finite values: {path}")
    timestamps = values[:, 0]
    if any(not math.isfinite(value) for value in timestamps) or np.any(np.diff(timestamps) < 0):
        raise DatasetValidationError(
            f"FreeTacMan timestamps must be finite and non-decreasing: {path}"
        )
    index = {name: position for position, name in enumerate(_CSV_COLUMNS)}
    pose = values[:, [index[name] for name in _POSE_COLUMNS]]
    gripper = values[:, [index["gripper_distance"]]]
    return timestamps, pose, gripper


@dataset_adapter(FREETACMAN_METADATA)
class FreeTacManAdapter(DatasetAdapter):
    """Yield one lazy multimodal trajectory per FreeTacMan demonstration."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        camera_modalities: Mapping[int, Modality | str] | None = None,
    ):
        super().__init__(root)
        configured = camera_modalities or _DEFAULT_CAMERA_MODALITIES
        self.camera_modalities = {
            number: Modality.parse(value) for number, value in configured.items()
        }
        if not self.camera_modalities or any(
            number not in {1, 2, 3} for number in self.camera_modalities
        ):
            raise ValueError("camera_modalities keys must be camera numbers 1, 2, or 3")
        if any(
            value not in {Modality.VISION, Modality.VISION_TACTILE}
            for value in self.camera_modalities.values()
        ):
            raise ValueError("camera_modalities values must be vision or vision_tactile")
        if not any(value is Modality.VISION_TACTILE for value in self.camera_modalities.values()):
            raise ValueError("camera_modalities must identify at least one vision-tactile camera")

    def _task_directories(self, root: Path) -> list[Path]:
        if any(_TRAJECTORY_NAME.fullmatch(path.name) for path in root.iterdir() if path.is_file()):
            return [root]
        return sorted(
            path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
        )

    def iter_samples(self, *, split: str | None = None) -> Iterator[TactileSample]:
        if split not in {None, "train"}:
            raise DatasetAccessError(
                "FreeTacMan publishes one train split; use split='train' or None"
            )
        root = self.require_root()
        found = False
        for task_dir in self._task_directories(root):
            trajectories: dict[str, Path] = {}
            videos: dict[str, dict[int, Path]] = {}
            for path in task_dir.iterdir():
                if not path.is_file():
                    continue
                trajectory_match = _TRAJECTORY_NAME.fullmatch(path.name)
                if trajectory_match:
                    trajectories[trajectory_match.group("trajectory")] = path
                    continue
                video_match = _VIDEO_NAME.fullmatch(path.name)
                if video_match:
                    trajectory_id = video_match.group("trajectory")
                    camera = int(video_match.group("camera"))
                    existing = videos.setdefault(trajectory_id, {}).get(camera)
                    if existing is not None:
                        raise DatasetValidationError(
                            f"FreeTacMan trajectory {trajectory_id!r} has duplicate camera{camera} "
                            f"files: {existing.name}, {path.name}"
                        )
                    videos[trajectory_id][camera] = path

            for trajectory_id in sorted(trajectories):
                found = True
                camera_files = videos.get(trajectory_id, {})
                missing = sorted({1, 2} - camera_files.keys())
                if missing:
                    raise DatasetValidationError(
                        f"FreeTacMan trajectory {trajectory_id!r} is missing camera files: "
                        + ", ".join(f"camera{number}" for number in missing)
                    )
                timestamps, pose, gripper = _read_trajectory(trajectories[trajectory_id])
                observations: dict[str, TactileObservation] = {}
                roles: dict[str, str] = {}
                for camera, video in sorted(camera_files.items()):
                    modality = self.camera_modalities.get(camera)
                    if modality is None:
                        continue
                    name = (
                        f"touch.camera{camera}"
                        if modality is Modality.VISION_TACTILE
                        else f"vision.camera{camera}"
                    )
                    roles[f"camera{camera}"] = modality.value
                    observations[name] = TactileObservation(
                        modality,
                        _relative_asset(video, root),
                        sensor=f"FreeTacMan camera {camera}",
                        encoding="MP4",
                    )
                if not any(
                    observation.modality is Modality.VISION_TACTILE
                    for observation in observations.values()
                ):
                    raise DatasetValidationError(
                        f"FreeTacMan trajectory {trajectory_id!r} has no configured tactile view"
                    )
                observations["pose.tcp"] = TactileObservation(
                    Modality.POSE,
                    pose,
                    sensor="OptiTrack",
                    timestamps=timestamps,
                    encoding="xyz_mm,euler_deg,quaternion_wxyz",
                    metadata={"columns": list(_POSE_COLUMNS)},
                )
                observations["gripper.distance"] = TactileObservation(
                    Modality.JOINT_STATE,
                    gripper,
                    sensor="FreeTacMan gripper",
                    timestamps=timestamps,
                    unit="mm",
                    encoding="gripper_distance",
                )
                task = task_dir.name
                yield TactileSample(
                    sample_id=f"{task}/{trajectory_id}",
                    observations=observations,
                    kind=SampleKind.TRAJECTORY,
                    labels={"task": task},
                    task="contact-rich manipulation",
                    split="train",
                    group_id=trajectory_id,
                    metadata={
                        "source_revision": FREETACMAN_REVISION,
                        "source_task": task,
                        "trajectory_file": trajectories[trajectory_id].relative_to(root).as_posix(),
                        "trajectory_rows": len(timestamps),
                        "camera_roles": roles,
                    },
                )
        if not found:
            raise DatasetAccessError(
                f"No FreeTacMan '*_traj.csv' files were found under {root}; pass the extracted "
                "Hugging Face snapshot root or one task directory"
            )
