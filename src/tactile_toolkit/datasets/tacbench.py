"""Guarded adapters for the in-house Meta Sparsh TacBench releases."""

from __future__ import annotations

import pickle
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
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
from tactile_toolkit.datasets.registry import dataset_adapter, register_dataset
from tactile_toolkit.model import SampleKind, TactileObservation, TactileSample
from tactile_toolkit.types import Modality

TACBENCH_CODE_REVISION = "fee6a05a97330eca014b59c71ac41596fe3974a7"
TACBENCH_DIGIT_FORCE_REVISION = "408e5b82e01b1cc471c8e747b686303afbababdc"
TACBENCH_GELSIGHT_FORCE_REVISION = "136db55485b6501605b1d2648ce0fe44740e302e"
TACBENCH_DIGIT_POSE_REVISION = "91faf6c077dbcd8dd8a5c03eb309c32cc4a63c66"

TACBENCH_DIGIT_FORCE_SOURCE = HuggingFaceSnapshot(
    "facebook/digit-force-estimation",
    TACBENCH_DIGIT_FORCE_REVISION,
    10_803_966_994,
    "https://huggingface.co/datasets/facebook/digit-force-estimation",
)
TACBENCH_GELSIGHT_FORCE_SOURCE = HuggingFaceSnapshot(
    "facebook/gelsight-force-estimation",
    TACBENCH_GELSIGHT_FORCE_REVISION,
    52_808_308_327,
    "https://huggingface.co/datasets/facebook/gelsight-force-estimation",
)
TACBENCH_DIGIT_POSE_SOURCE = HuggingFaceSnapshot(
    "facebook/digit-pose-estimation",
    TACBENCH_DIGIT_POSE_REVISION,
    18_704_238_281,
    "https://huggingface.co/datasets/facebook/digit-pose-estimation",
)

_LICENSE = LicenseInfo(
    name="Creative Commons Attribution-NonCommercial 4.0 International",
    spdx_id="CC-BY-NC-4.0",
    url=(f"https://github.com/facebookresearch/sparsh/blob/{TACBENCH_CODE_REVISION}/LICENSE.md"),
    allows_redistribution=True,
    allows_commercial_use=False,
    requires_attribution=True,
    notes=(
        "The in-house TacBench dataset cards declare CC BY-NC 4.0; externally sourced TacBench "
        "tasks retain their own upstream terms and are not loaded by these adapters."
    ),
)
_CITATION = Citation(
    title="Sparsh: Self-supervised Touch Representations for Vision-based Tactile Sensing",
    url="https://arxiv.org/abs/2410.24090",
    doi="10.48550/arXiv.2410.24090",
    year=2024,
)
_PROVENANCE = ProvenanceInfo(
    publisher="Meta AI Research",
    source_url="https://ai.meta.com/research/publications/sparsh-self-supervised-touch-representations-for-vision-based-tactile-sensing/",
    creators=(
        "Carolina Higuera",
        "Akash Sharma",
        "Chaithanya Krishna Bodduluri",
        "Taosha Fan",
        "Patrick Lancaster",
        "Mrinal Kalakrishnan",
        "Michael Kaess",
        "Byron Boots",
        "Mike Lambeta",
        "Tingfan Wu",
        "Mustafa Mukadam",
    ),
    institutions=("Meta AI", "Carnegie Mellon University", "University of Washington"),
    notes=f"Native layout follows the official Sparsh code at {TACBENCH_CODE_REVISION}.",
)

META_TACBENCH_METADATA = DatasetMetadata(
    dataset_id="meta-tacbench",
    name="Meta DIGIT benchmarks / Sparsh TacBench",
    description=(
        "The six-task evaluation suite for Sparsh tactile representations, with only Meta's "
        "in-house force, slip, and pose subsets adapted here."
    ),
    homepage="https://github.com/facebookresearch/sparsh",
    version="CoRL 2024 release",
    revision=TACBENCH_CODE_REVISION,
    provenance=_PROVENANCE,
    license=_LICENSE,
    citations=(_CITATION,),
    access=(
        AccessSource(
            AccessKind.REFERENCE,
            "https://huggingface.co/collections/facebook/sparsh",
            notes="Official collection of models and task datasets.",
        ),
    ),
    modalities=(Modality.VISION_TACTILE, Modality.WRENCH, Modality.POSE),
    sensors=("DIGIT", "GelSight Mini", "GelSight 2017"),
    tasks=(
        "force estimation",
        "slip detection",
        "pose estimation",
        "grasp stability",
        "textile recognition",
        "bead-maze manipulation",
    ),
    formats=("pickle", "jpeg", "configuration files"),
    support_level=SupportLevel.DISCOVERABLE,
    limitations=(
        "The umbrella includes third-party benchmark components whose upstream terms and layouts "
        "are not superseded by the Sparsh repository license.",
        "The official source repository was archived read-only on 2026-04-01.",
        "Use the loadable child records for the three in-house releases.",
    ),
    aliases=("meta-digit-benchmarks", "sparsh-tacbench", "tacbench"),
)

TACBENCH_FORCE_SLIP_METADATA = DatasetMetadata(
    dataset_id="tacbench-force-slip",
    name="TacBench DIGIT and GelSight Mini force/slip",
    description="Meta's paired tactile, three-axis force, contact, and slip trajectories.",
    homepage="https://github.com/facebookresearch/sparsh",
    version="CoRL 2024 release",
    revision=(f"digit:{TACBENCH_DIGIT_FORCE_REVISION};gelsight:{TACBENCH_GELSIGHT_FORCE_REVISION}"),
    provenance=_PROVENANCE,
    license=_LICENSE,
    citations=(_CITATION,),
    access=(
        AccessSource(
            AccessKind.HUGGING_FACE,
            TACBENCH_DIGIT_FORCE_SOURCE.browser_url,
            notes=f"Pinned revision {TACBENCH_DIGIT_FORCE_REVISION} (10.8 GB).",
        ),
        AccessSource(
            AccessKind.HUGGING_FACE,
            TACBENCH_GELSIGHT_FORCE_SOURCE.browser_url,
            notes=f"Pinned revision {TACBENCH_GELSIGHT_FORCE_REVISION} (52.8 GB).",
        ),
    ),
    modalities=(Modality.VISION_TACTILE, Modality.WRENCH),
    sensors=("DIGIT", "GelSight Mini", "ATI Nano17 force/torque sensor"),
    tasks=("force estimation", "slip detection"),
    formats=("pickle", "encoded tactile images"),
    support_level=SupportLevel.LOADABLE,
    approximate_size_bytes=(
        TACBENCH_DIGIT_FORCE_SOURCE.approximate_size_bytes
        + TACBENCH_GELSIGHT_FORCE_SOURCE.approximate_size_bytes
    ),
    limitations=(
        "Native Python pickle can execute arbitrary code and is rejected unless trust_pickle=True.",
        "Iteration materializes one batch's encoded frame list, then decodes one frame at a time.",
        "The adapter exposes the publisher's three force axes as a wrench modality without "
        "inventing torque axes.",
    ),
    aliases=("sparsh-force-slip", "digit-force-estimation", "gelsight-force-estimation"),
)

TACBENCH_POSE_METADATA = DatasetMetadata(
    dataset_id="tacbench-pose",
    name="TacBench DIGIT pose estimation",
    description="Meta's three-finger DIGIT sequences paired with relative SE(3) object motion.",
    homepage="https://github.com/facebookresearch/sparsh",
    version="CoRL 2024 release",
    revision=TACBENCH_DIGIT_POSE_REVISION,
    provenance=_PROVENANCE,
    license=_LICENSE,
    citations=(_CITATION,),
    access=(
        AccessSource(
            AccessKind.HUGGING_FACE,
            TACBENCH_DIGIT_POSE_SOURCE.browser_url,
            notes=f"Pinned revision {TACBENCH_DIGIT_POSE_REVISION} (18.7 GB).",
        ),
    ),
    modalities=(Modality.VISION_TACTILE, Modality.POSE),
    sensors=("DIGIT index finger", "DIGIT middle finger", "DIGIT ring finger"),
    tasks=("pose estimation",),
    formats=("pickle", "encoded tactile images", "4x4 transformation matrices"),
    support_level=SupportLevel.LOADABLE,
    approximate_size_bytes=TACBENCH_DIGIT_POSE_SOURCE.approximate_size_bytes,
    limitations=(
        "Native Python pickle can execute arbitrary code and is rejected unless trust_pickle=True.",
        "One bag is materialized at a time; reference backgrounds and non-selected finger streams "
        "are not applied automatically.",
    ),
    aliases=("sparsh-pose", "digit-pose-estimation"),
)


def _load_trusted_pickle(path: Path, *, trust_pickle: bool, max_pickle_bytes: int) -> Any:
    if not trust_pickle:
        raise DatasetAccessError(
            "TacBench uses Python pickle, which can execute arbitrary code; inspect the files and "
            "pass trust_pickle=True only for a trusted official copy"
        )
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise DatasetAccessError(f"Could not inspect TacBench pickle {path}: {exc}") from exc
    if size > max_pickle_bytes:
        raise DatasetAccessError(
            f"TacBench pickle {path} is {size} bytes, above max_pickle_bytes={max_pickle_bytes}"
        )
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)  # noqa: S301 - guarded by explicit trust_pickle opt-in
    except (
        OSError,
        pickle.PickleError,
        EOFError,
        AttributeError,
        ImportError,
        TypeError,
        ValueError,
    ) as exc:
        raise DatasetValidationError(
            f"Could not deserialize TacBench pickle {path}: {exc}"
        ) from exc


def _decode_image(value: Any, context: str) -> np.ndarray:
    if isinstance(value, (bytes, bytearray, memoryview)):
        encoded = np.frombuffer(value, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise DatasetValidationError(f"Could not decode TacBench tactile image in {context}")
        value = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise DatasetValidationError(
            f"TacBench tactile image in {context} must have shape (H, W, 3)"
        )
    if image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
            raise DatasetValidationError(f"TacBench tactile image in {context} is invalid")
        if image.min() < 0 or image.max() > 255:
            raise DatasetValidationError(
                f"TacBench tactile image in {context} falls outside uint8 range"
            )
        image = image.astype(np.uint8)
    return image


def _mapping(value: Any, context: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise DatasetValidationError(f"TacBench {context} must be a mapping")
    return value


def _binary_vector(value: Any, expected_length: int, context: str) -> np.ndarray:
    vector = np.asarray(value)
    if vector.ndim != 1 or len(vector) != expected_length:
        raise DatasetValidationError(
            f"TacBench {context} must contain {expected_length} binary values"
        )
    if not np.isin(vector, (0, 1, False, True)).all():
        raise DatasetValidationError(f"TacBench {context} must contain only binary values")
    return vector.astype(np.uint8, copy=False)


@dataset_adapter(TACBENCH_FORCE_SLIP_METADATA)
class TacBenchForceSlipAdapter(DatasetAdapter):
    """Load one trusted force/slip batch at a time and yield decoded tactile frames."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        sensor: str | None = None,
        trust_pickle: bool = False,
        max_pickle_bytes: int = 8_000_000_000,
    ):
        super().__init__(root)
        normalized = sensor.strip().lower() if sensor is not None else None
        aliases = {"digit": "digit", "gelsight": "gelsight", "gelsight_mini": "gelsight"}
        if normalized is not None and normalized not in aliases:
            raise ValueError("TacBench sensor must be 'digit', 'gelsight', or 'gelsight_mini'")
        self.sensor = aliases.get(normalized) if normalized is not None else None
        self.trust_pickle = trust_pickle
        if max_pickle_bytes <= 0:
            raise ValueError("max_pickle_bytes must be positive")
        self.max_pickle_bytes = max_pickle_bytes

    def _batch_directories(self, root: Path) -> list[Path]:
        if (root / "dataset_slip_forces.pkl").is_file():
            return [root]
        return sorted({path.parent for path in root.rglob("dataset_slip_forces.pkl")})

    def _sensor_for_batch(self, batch: Path) -> str:
        available = [
            sensor for sensor in ("digit", "gelsight") if any(batch.glob(f"dataset_{sensor}_*.pkl"))
        ]
        if self.sensor is not None:
            if self.sensor not in available:
                raise DatasetAccessError(
                    f"TacBench batch {batch} has no dataset_{self.sensor}_*.pkl frame shards"
                )
            return self.sensor
        if len(available) != 1:
            raise DatasetAccessError(
                f"Could not infer one TacBench sensor in {batch}; pass sensor='digit' or "
                "sensor='gelsight_mini'"
            )
        return available[0]

    def _iter_batch(self, root: Path, batch: Path) -> Iterator[TactileSample]:
        sensor_key = self._sensor_for_batch(batch)
        sensor_name = "DIGIT" if sensor_key == "digit" else "GelSight Mini"
        frames: list[Any] = []
        for shard in sorted(batch.glob(f"dataset_{sensor_key}_*.pkl")):
            value = _load_trusted_pickle(
                shard,
                trust_pickle=self.trust_pickle,
                max_pickle_bytes=self.max_pickle_bytes,
            )
            if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
                raise DatasetValidationError(
                    f"TacBench frame shard {shard} must contain a sequence"
                )
            frames.extend(value)
        labels_path = batch / "dataset_slip_forces.pkl"
        payload = _mapping(
            _load_trusted_pickle(
                labels_path,
                trust_pickle=self.trust_pickle,
                max_pickle_bytes=self.max_pickle_bytes,
            ),
            f"label file {labels_path}",
        )
        trajectories = _mapping(payload.get("trajectories"), "trajectories")
        if not frames:
            raise DatasetValidationError(f"TacBench batch {batch} contains no tactile frames")
        in_contact = _binary_vector(
            payload.get("in_contact"), len(frames), f"in_contact in {batch}"
        )
        if not trajectories:
            raise DatasetValidationError(f"TacBench batch {batch} contains no trajectories")
        batch_id = batch.relative_to(root).as_posix() if batch != root else batch.name
        for trajectory_id, raw_trajectory in trajectories.items():
            trajectory = _mapping(raw_trajectory, f"trajectory {trajectory_id!r}")
            indexes = np.asarray(trajectory.get("indexes"))
            forces = np.asarray(trajectory.get("forces"), dtype=np.float32)
            if indexes.ndim != 1 or not np.issubdtype(indexes.dtype, np.integer):
                raise DatasetValidationError(
                    f"TacBench trajectory {trajectory_id!r} indexes must be a 1D integer array"
                )
            count = len(indexes)
            slips = _binary_vector(
                trajectory.get("slip_label"),
                count,
                f"trajectory {trajectory_id!r} slip_label",
            )
            if forces.ndim != 2 or forces.shape != (count, 3):
                raise DatasetValidationError(
                    f"TacBench trajectory {trajectory_id!r} forces must have shape ({count}, 3)"
                )
            if not np.isfinite(forces).all():
                raise DatasetValidationError(
                    f"TacBench trajectory {trajectory_id!r} contains non-finite force"
                )
            for offset, frame_index_value in enumerate(indexes):
                frame_index = int(frame_index_value)
                if not 0 <= frame_index < len(frames):
                    raise DatasetValidationError(
                        f"TacBench trajectory {trajectory_id!r} frame index {frame_index} is out "
                        f"of range for {len(frames)} frames"
                    )
                context = f"{batch}/trajectory {trajectory_id!r}/frame {frame_index}"
                yield TactileSample(
                    sample_id=f"{batch_id}/{trajectory_id}/{offset:06d}",
                    observations={
                        "touch": TactileObservation(
                            Modality.VISION_TACTILE,
                            _decode_image(frames[frame_index], context),
                            sensor=sensor_name,
                        ),
                        "force.xyz": TactileObservation(
                            Modality.WRENCH,
                            forces[offset],
                            sensor="ATI Nano17",
                            unit="N",
                            metadata={"axes": ["fx", "fy", "fz"]},
                        ),
                    },
                    labels={
                        "slip": int(slips[offset]),
                        "in_contact": bool(in_contact[frame_index]),
                    },
                    task="force estimation and slip detection",
                    group_id=f"{batch_id}/{trajectory_id}",
                    metadata={
                        "source_revision": (
                            TACBENCH_DIGIT_FORCE_REVISION
                            if sensor_key == "digit"
                            else TACBENCH_GELSIGHT_FORCE_REVISION
                        ),
                        "native_frame_index": frame_index,
                        "trusted_pickle": True,
                    },
                )

    def iter_samples(self, *, split: str | None = None) -> Iterator[TactileSample]:
        if split is not None:
            raise DatasetAccessError(
                "TacBench force/slip does not publish canonical per-sample splits; use split=None"
            )
        root = self.require_root()
        if not root.is_dir():
            raise DatasetAccessError(f"TacBench force/slip root must be a directory: {root}")
        batches = self._batch_directories(root)
        if not batches:
            raise DatasetAccessError(
                f"No TacBench dataset_slip_forces.pkl batch labels were found under {root}"
            )
        for batch in batches:
            yield from self._iter_batch(root, batch)


@dataset_adapter(TACBENCH_POSE_METADATA)
class TacBenchPoseAdapter(DatasetAdapter):
    """Yield one selected-finger frame and relative pose from each trusted pose bag."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        finger: str = "index",
        stride: int = 5,
        trust_pickle: bool = False,
        max_pickle_bytes: int = 8_000_000_000,
    ):
        super().__init__(root)
        self.finger = finger.strip().lower()
        if self.finger not in {"index", "middle", "ring"}:
            raise ValueError("TacBench pose finger must be index, middle, or ring")
        if stride <= 0:
            raise ValueError("TacBench pose stride must be positive")
        self.stride = stride
        self.trust_pickle = trust_pickle
        if max_pickle_bytes <= 0:
            raise ValueError("max_pickle_bytes must be positive")
        self.max_pickle_bytes = max_pickle_bytes

    @staticmethod
    def _split(path: Path, root: Path) -> str | None:
        parts = path.relative_to(root).parts
        return next((part for part in parts if part in {"train", "test"}), None)

    def iter_samples(self, *, split: str | None = None) -> Iterator[TactileSample]:
        if split not in {None, "train", "test"}:
            raise DatasetAccessError("TacBench pose split must be 'train', 'test', or None")
        root = self.require_root()
        if root.is_file():
            bags = [root]
            search_root = (
                root.parent.parent.parent
                if root.parent.parent.name in {"train", "test"}
                else root.parent
            )
        elif root.is_dir():
            bags = sorted(root.rglob("bag_*.pkl"))
            search_root = root
        else:
            raise DatasetAccessError(f"TacBench pose root is not a file or directory: {root}")
        if not bags:
            raise DatasetAccessError(f"No TacBench pose bag_*.pkl files were found under {root}")
        touch_key = f"digit_{self.finger}"
        pose_key = f"object_{self.finger}_rel_pose_n{self.stride}"
        matched = False
        for bag in bags:
            bag_split = self._split(bag, search_root)
            if split is not None and bag_split != split:
                continue
            matched = True
            payload = _mapping(
                _load_trusted_pickle(
                    bag,
                    trust_pickle=self.trust_pickle,
                    max_pickle_bytes=self.max_pickle_bytes,
                ),
                f"pose bag {bag}",
            )
            if touch_key not in payload or pose_key not in payload:
                raise DatasetValidationError(
                    f"TacBench pose bag {bag} requires {touch_key!r} and {pose_key!r}"
                )
            touches = payload[touch_key]
            poses = np.asarray(payload[pose_key], dtype=np.float32)
            if isinstance(touches, (str, bytes, bytearray)) or not isinstance(touches, Sequence):
                raise DatasetValidationError(f"TacBench {touch_key} in {bag} must be a sequence")
            count = min(len(touches), len(poses))
            if count == 0 or poses.ndim != 3 or poses.shape[1:] != (4, 4):
                raise DatasetValidationError(
                    f"TacBench {pose_key} in {bag} must contain 4x4 matrices paired with images"
                )
            if not np.isfinite(poses[:count]).all():
                raise DatasetValidationError(f"TacBench pose bag {bag} has non-finite transforms")
            relative = bag.relative_to(search_root).as_posix()
            object_name = bag.parent.name if bag.parent.name not in {"train", "test"} else "unknown"
            group = relative.removesuffix(bag.suffix)
            for index in range(count):
                yield TactileSample(
                    sample_id=f"{group}/{self.finger}/{index:06d}",
                    kind=SampleKind.FRAME,
                    observations={
                        "touch": TactileObservation(
                            Modality.VISION_TACTILE,
                            _decode_image(touches[index], f"{bag}/{touch_key}/{index}"),
                            sensor=f"DIGIT {self.finger} finger",
                        ),
                        "pose.relative": TactileObservation(
                            Modality.POSE,
                            poses[index],
                            frame=f"object relative to {self.finger} finger",
                            metadata={"stride": self.stride, "representation": "SE(3) matrix"},
                        ),
                    },
                    labels={"object": object_name},
                    task="pose estimation",
                    split=bag_split,
                    group_id=group,
                    metadata={
                        "source_revision": TACBENCH_DIGIT_POSE_REVISION,
                        "native_bag": relative,
                        "trusted_pickle": True,
                    },
                )
        if not matched:
            raise DatasetAccessError(f"No TacBench pose bags matched split={split!r} under {root}")


register_dataset(META_TACBENCH_METADATA)
