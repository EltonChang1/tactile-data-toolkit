"""Guarded local adapter for the official filtered CLAMP NPZ release."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

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
from tactile_toolkit.model import SampleKind, TactileObservation, TactileSample
from tactile_toolkit.types import Modality

CLAMP_DATAVERSE_DOI = "10.7910/DVN/HNS2Z4"
CLAMP_FILTERED_FILE_ID = 13_302_010
CLAMP_FILTERED_SIZE_BYTES = 525_694_981
CLAMP_FILTERED_MD5 = "a5de0bbce85b5cf69db73e13029913c7"
CLAMP_FILTERED_URL = f"https://dataverse.harvard.edu/api/access/datafile/{CLAMP_FILTERED_FILE_ID}"
CLAMP_CODE_REVISION = "9b23a9ffddaa628ee1bb99cefdbf728bf2d016bf"

CLAMP_METADATA = DatasetMetadata(
    dataset_id="clamp",
    name="CLAMP",
    description=(
        "A crowdsourced in-the-wild multimodal haptic corpus for material and compliance "
        "recognition across household objects and devices."
    ),
    homepage="https://emprise.cs.cornell.edu/clamp/",
    version="Harvard Dataverse 1.0",
    revision="1.0 (released 2026-01-04)",
    provenance=ProvenanceInfo(
        publisher="Cornell EmPRISE Lab",
        source_url="https://doi.org/10.7910/DVN/HNS2Z4",
        creators=(
            "Pranav N. Thakkar",
            "Shubhangi Sinha",
            "Karan Baijal",
            "Yuhan Bian",
            "Leah Lackey",
            "Ben Dodson",
            "Heisen Kong",
            "Jueun Kwon",
            "Amber Li",
            "Yifei Hu",
            "Alexios Rekoutis",
            "Tom Silver",
            "Tapomayukh Bhattacharjee",
        ),
        institutions=("Cornell University", "Georgia Institute of Technology"),
        notes=(
            "The adapter schema follows the official CLAMP code at commit "
            f"{CLAMP_CODE_REVISION}; Harvard Dataverse is the authoritative data release."
        ),
    ),
    license=LicenseInfo(
        name="Creative Commons Attribution 4.0 International",
        spdx_id="CC-BY-4.0",
        url="https://creativecommons.org/licenses/by/4.0/",
        allows_redistribution=True,
        allows_commercial_use=True,
        requires_attribution=True,
        notes="Harvard Dataverse release 1.0 declares CC BY 4.0; the code is BSD-3-Clause.",
    ),
    citations=(
        Citation(
            title=(
                "CLAMP: Crowdsourcing a LArge-scale in-the-wild haptic dataset with an "
                "open-source device for Multimodal robot Perception"
            ),
            url="https://arxiv.org/abs/2505.21495",
            doi="10.48550/arXiv.2505.21495",
            year=2025,
        ),
        Citation(
            title="CLAMP dataset, Harvard Dataverse version 1.0",
            url="https://doi.org/10.7910/DVN/HNS2Z4",
            doi="10.7910/DVN/HNS2Z4",
            year=2026,
        ),
    ),
    access=(
        AccessSource(
            AccessKind.DATAVERSE,
            "https://doi.org/10.7910/DVN/HNS2Z4",
            notes=(
                "Version 1.0 contains 30 files totaling 6,385,413,968 bytes; the recommended "
                f"filtered NPZ is {CLAMP_FILTERED_SIZE_BYTES} bytes with publisher MD5 "
                f"{CLAMP_FILTERED_MD5}."
            ),
        ),
        AccessSource(
            AccessKind.HTTP,
            CLAMP_FILTERED_URL,
            notes="Stable Dataverse datafile API route for CLAMP_dataset_filtered.npz.",
        ),
    ),
    modalities=(
        Modality.PRESSURE,
        Modality.TEMPERATURE,
        Modality.VIBRATION,
        Modality.JOINT_STATE,
        Modality.VISION,
        Modality.LANGUAGE,
    ),
    sensors=(
        "CLAMP open-source device",
        "fabric force sensor",
        "active/passive thermistors",
        "contact microphone",
        "IMUs",
        "RGB camera",
    ),
    tasks=("material identification", "compliance recognition"),
    formats=("npz with object arrays", "zip", "text", "jpeg", "wav"),
    support_level=SupportLevel.LOADABLE,
    approximate_size_bytes=6_385_413_968,
    limitations=(
        "The official filtered NPZ contains NumPy object arrays and is rejected unless "
        "trust_pickle=True because loading it can execute arbitrary code.",
        "Loading the 525.7 MB filtered archive materializes its object arrays; the adapter yields "
        "one normalized contact sequence at a time after that trust boundary.",
        "The release publishes MD5 rather than SHA-256 checksums; callers should verify the "
        "publisher MD5 and record a local SHA-256 before deserialization.",
        "Official train/validation/test indices are generated from a seed at training time, so "
        "the adapter preserves object groups but does not invent a split.",
    ),
    aliases=("clamp-haptics", "clamp-dataset"),
)

_FEATURES: dict[str, tuple[str, Modality, str | None]] = {
    "Force": ("pressure.force", Modality.PRESSURE, None),
    "Active thermal": ("temperature.active", Modality.TEMPERATURE, None),
    "Passive thermal": ("temperature.passive", Modality.TEMPERATURE, None),
    "Active thermal diff": (
        "temperature.active_diff",
        Modality.TEMPERATURE,
        None,
    ),
    "Passive thermal diff": (
        "temperature.passive_diff",
        Modality.TEMPERATURE,
        None,
    ),
    "Contact microphone": ("vibration.contact_microphone", Modality.VIBRATION, None),
    "Force diff": ("pressure.force_diff", Modality.PRESSURE, None),
    "Proprioception": ("joint.proprioception", Modality.JOINT_STATE, None),
    "Impedance": ("pressure.impedance", Modality.PRESSURE, None),
}
_REQUIRED_KEYS = frozenset(
    {
        "dataset",
        "feature_key",
        "filenames",
        "material_labels",
        "obj_material_labels",
        "obj_names",
        "obj_contact_lookup",
        "img_type",
        "label_encoding",
    }
)


def _clean_text(value: Any, context: str) -> str:
    text = str(value).strip()
    if not text:
        raise DatasetValidationError(f"CLAMP {context} must not be empty")
    return text


def _safe_native_filename(value: Any, context: str) -> str:
    text = _clean_text(value, context).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "\x00" in text:
        raise DatasetValidationError(f"CLAMP {context} must be a safe relative provenance path")
    return path.as_posix()


@dataset_adapter(CLAMP_METADATA)
class ClampFilteredAdapter(DatasetAdapter):
    """Expose filtered CLAMP contacts after an explicit object-array trust decision."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        filename: str = "CLAMP_dataset_filtered.npz",
        trust_pickle: bool = False,
        max_archive_bytes: int = 2_000_000_000,
    ):
        super().__init__(root)
        candidate = PurePosixPath(filename.replace("\\", "/"))
        if candidate.is_absolute() or ".." in candidate.parts or len(candidate.parts) != 1:
            raise ValueError("CLAMP filename must be one safe filename")
        self.filename = candidate.name
        self.trust_pickle = trust_pickle
        if max_archive_bytes <= 0:
            raise ValueError("max_archive_bytes must be positive")
        self.max_archive_bytes = max_archive_bytes

    def _archive_path(self) -> Path:
        root = self.require_root()
        path = root if root.is_file() else root / self.filename
        if not path.is_file():
            raise DatasetAccessError(
                f"CLAMP filtered archive is missing: {path}; download "
                "CLAMP_dataset_filtered.npz from the official Dataverse release"
            )
        if path.suffix.lower() != ".npz":
            raise DatasetAccessError(f"CLAMP filtered archive must be an .npz file: {path}")
        size = path.stat().st_size
        if size > self.max_archive_bytes:
            raise DatasetAccessError(
                f"CLAMP archive is {size} bytes, above max_archive_bytes={self.max_archive_bytes}"
            )
        if not self.trust_pickle:
            raise DatasetAccessError(
                "CLAMP filtered NPZ contains pickle-backed object arrays that can execute "
                "arbitrary code; verify the official MD5 and pass trust_pickle=True only for a "
                "trusted copy"
            )
        return path

    @staticmethod
    def _feature_map(value: Any, feature_count: int) -> dict[str, int]:
        if not isinstance(value, Mapping):
            raise DatasetValidationError("CLAMP feature_key must contain a mapping")
        result: dict[str, int] = {}
        for raw_name, raw_index in value.items():
            name = _clean_text(raw_name, "feature name")
            if name not in _FEATURES:
                continue
            if not isinstance(raw_index, (int, np.integer)):
                raise DatasetValidationError(f"CLAMP feature index for {name!r} must be integer")
            index = int(raw_index)
            if not 0 <= index < feature_count:
                raise DatasetValidationError(
                    f"CLAMP feature index {index} for {name!r} is outside {feature_count} features"
                )
            result[name] = index
        if not result:
            raise DatasetValidationError("CLAMP feature_key has no supported haptic features")
        return result

    @staticmethod
    def _contact_groups(
        object_names: np.ndarray, lookups: np.ndarray, count: int
    ) -> dict[int, str]:
        if len(object_names) != len(lookups):
            raise DatasetValidationError(
                "CLAMP obj_names and obj_contact_lookup must have the same length"
            )
        groups: dict[int, str] = {}
        for object_index, (raw_name, raw_indexes) in enumerate(zip(object_names, lookups)):
            name = _safe_native_filename(raw_name, f"object name {object_index}")
            indexes = np.asarray(raw_indexes)
            if indexes.ndim != 1 or not all(
                isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                for value in indexes
            ):
                raise DatasetValidationError(
                    f"CLAMP object contact lookup {object_index} must be a 1D integer array"
                )
            for raw_index in indexes:
                index = int(raw_index)
                if not 0 <= index < count:
                    raise DatasetValidationError(
                        f"CLAMP object contact index {index} is outside {count} contacts"
                    )
                if index in groups:
                    raise DatasetValidationError(
                        f"CLAMP contact {index} belongs to more than one object group"
                    )
                groups[index] = name
        if len(groups) != count:
            raise DatasetValidationError(
                f"CLAMP object lookup covers {len(groups)} of {count} contacts; "
                "leakage-safe grouping is incomplete"
            )
        return groups

    def iter_samples(self, *, split: str | None = None) -> Iterator[TactileSample]:
        if split is not None:
            raise DatasetAccessError(
                "CLAMP release 1.0 does not publish fixed per-contact splits; pass split=None and "
                "split by the emitted object group_id"
            )
        path = self._archive_path()
        try:
            archive = np.load(path, allow_pickle=True)  # noqa: S301 - explicit trust_pickle gate
        except (OSError, ValueError) as exc:
            raise DatasetValidationError(f"Could not open trusted CLAMP NPZ {path}: {exc}") from exc
        with archive:
            missing = sorted(_REQUIRED_KEYS - set(archive.files))
            if missing:
                raise DatasetValidationError(f"CLAMP filtered NPZ is missing keys: {missing}")
            try:
                dataset = np.asarray(archive["dataset"])
                feature_key = archive["feature_key"].item()
                filenames = np.asarray(archive["filenames"])
                materials = np.asarray(archive["material_labels"])
                object_names = np.asarray(archive["obj_names"])
                object_lookups = np.asarray(archive["obj_contact_lookup"], dtype=object)
                image_type = _clean_text(archive["img_type"].item(), "img_type")
                label_encoding = [str(value) for value in archive["label_encoding"].tolist()]
                gpt_probs = (
                    np.asarray(archive["gpt_probs"]) if "gpt_probs" in archive.files else None
                )
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                raise DatasetValidationError(
                    f"Invalid CLAMP filtered NPZ structure: {exc}"
                ) from exc
            if dataset.ndim != 3:
                raise DatasetValidationError(
                    f"CLAMP dataset must have shape (contacts, features, time), got {dataset.shape}"
                )
            count, feature_count, _ = dataset.shape
            if count == 0:
                raise DatasetValidationError("CLAMP filtered NPZ contains no contacts")
            if len(filenames) != count or len(materials) != count:
                raise DatasetValidationError(
                    "CLAMP filenames and material_labels must match the contact count"
                )
            if gpt_probs is not None and len(gpt_probs) != count:
                raise DatasetValidationError("CLAMP gpt_probs must match the contact count")
            features = self._feature_map(feature_key, feature_count)
            groups = self._contact_groups(object_names, object_lookups, count)
            for contact_index in range(count):
                observations: dict[str, TactileObservation] = {}
                for feature_name, feature_index in features.items():
                    name, modality, unit = _FEATURES[feature_name]
                    values = np.asarray(dataset[contact_index, feature_index], dtype=np.float32)
                    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
                        raise DatasetValidationError(
                            f"CLAMP contact {contact_index} feature {feature_name!r} must be a "
                            "non-empty finite 1D signal"
                        )
                    observations[name] = TactileObservation(
                        modality,
                        values[:, None],
                        unit=unit,
                        encoding="official filtered feature",
                        metadata={
                            "native_feature": feature_name,
                            "unit_status": "not declared in the filtered NPZ",
                        },
                    )
                labels: dict[str, Any] = {
                    "material": _clean_text(materials[contact_index], "material label"),
                    "object": groups[contact_index],
                    "native_filename": _safe_native_filename(
                        filenames[contact_index], f"filename {contact_index}"
                    ),
                    "image_type": image_type,
                    "label_encoding": label_encoding,
                }
                if gpt_probs is not None:
                    raw_probabilities = gpt_probs[contact_index]
                    if isinstance(raw_probabilities, np.ndarray) and raw_probabilities.ndim == 0:
                        raw_probabilities = raw_probabilities.item()
                    if isinstance(raw_probabilities, Mapping):
                        probabilities_by_material: dict[str, float] = {}
                        for raw_label, raw_probability in raw_probabilities.items():
                            label = _clean_text(raw_label, "vision material label")
                            try:
                                probability = float(raw_probability)
                            except (TypeError, ValueError) as exc:
                                raise DatasetValidationError(
                                    f"CLAMP gpt_probs for contact {contact_index} contains a "
                                    f"non-numeric value for {label!r}"
                                ) from exc
                            if not np.isfinite(probability):
                                raise DatasetValidationError(
                                    f"CLAMP gpt_probs for contact {contact_index} contains a "
                                    f"non-finite value for {label!r}"
                                )
                            probabilities_by_material[label] = probability
                        labels["vision_material_probabilities"] = probabilities_by_material
                    else:
                        probabilities = np.asarray(raw_probabilities, dtype=np.float32)
                        if probabilities.ndim != 1 or not np.isfinite(probabilities).all():
                            raise DatasetValidationError(
                                f"CLAMP gpt_probs for contact {contact_index} must be a finite "
                                "vector or material-to-probability mapping"
                            )
                        labels["vision_material_probabilities"] = probabilities.tolist()
                yield TactileSample(
                    sample_id=f"contact/{contact_index:08d}",
                    kind=SampleKind.SEQUENCE,
                    observations=observations,
                    labels=labels,
                    task="material identification",
                    group_id=groups[contact_index],
                    metadata={
                        "dataverse_doi": CLAMP_DATAVERSE_DOI,
                        "dataverse_version": "1.0",
                        "trusted_pickle": True,
                    },
                )
