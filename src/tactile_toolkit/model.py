"""Normalized source-data model used between dataset adapters and conversion.

The model deliberately sits before :class:`~tactile_toolkit.types.TactileChunk`.
Published datasets can therefore expose raw, lazy, multimodal samples without
inventing calibrated force fields that their source data does not contain.
"""

from __future__ import annotations

import math
import string
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, TypeAlias

import numpy as np

from tactile_toolkit.types import Modality


class SampleKind(StrEnum):
    """Temporal granularity of one adapter output item."""

    FRAME = "frame"
    SEQUENCE = "sequence"
    TRAJECTORY = "trajectory"

    @classmethod
    def parse(cls, value: str | SampleKind) -> SampleKind:
        if isinstance(value, SampleKind):
            return value
        try:
            return cls(value)
        except ValueError:
            return cls[value.upper()]


@dataclass(frozen=True)
class AssetReference:
    """Lazy reference to an official or user-provided data asset.

    ``uri`` may be a local path or a remote URI. ``member`` identifies one
    relative member inside a container such as TAR, ZIP, HDF5, or NPZ without
    requiring the adapter to extract or deserialize that container eagerly.
    """

    uri: str
    member: str | None = None
    media_type: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        uri = self.uri.strip()
        if not uri:
            raise ValueError("AssetReference.uri must not be empty")
        object.__setattr__(self, "uri", uri)

        if self.member is not None:
            raw_member = self.member.strip().replace("\\", "/")
            member_path = PurePosixPath(raw_member)
            if not raw_member or member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError("AssetReference.member must be a safe relative path")
            object.__setattr__(self, "member", str(member_path))

        if self.media_type is not None:
            media_type = self.media_type.strip()
            if not media_type:
                raise ValueError("AssetReference.media_type must not be empty")
            object.__setattr__(self, "media_type", media_type)

        if self.sha256 is not None:
            checksum = self.sha256.lower()
            if len(checksum) != 64 or any(c not in string.hexdigits for c in checksum):
                raise ValueError("AssetReference.sha256 must contain 64 hexadecimal characters")
            object.__setattr__(self, "sha256", checksum)

        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("AssetReference.size_bytes must be non-negative")

    def to_dict(self) -> dict[str, str | int | None]:
        """Return a JSON-compatible asset descriptor without reading the asset."""
        return {
            "uri": self.uri,
            "member": self.member,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


ObservationData: TypeAlias = np.ndarray | str | AssetReference


@dataclass
class TactileObservation:
    """One named modality in a normalized tactile sample.

    NumPy arrays hold materialized data, text is reserved for language, and an
    :class:`AssetReference` keeps large media lazy. ``timestamp`` is for a
    single observation; ``timestamps`` identifies the leading time axis of a
    sequence. Only one form may be present.
    """

    modality: Modality
    data: ObservationData = field(repr=False)
    sensor: str | None = None
    timestamp: float | None = None
    timestamps: np.ndarray | None = field(default=None, repr=False)
    unit: str | None = None
    frame: str | None = None
    encoding: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.modality = Modality.parse(self.modality)
        if isinstance(self.data, str):
            if self.modality is not Modality.LANGUAGE:
                raise TypeError("String observation data are only valid for language")
            self.data = self.data.strip()
            if not self.data:
                raise ValueError("Language observation data must not be empty")
        elif isinstance(self.data, np.ndarray):
            self.data = np.asarray(self.data)
        elif not isinstance(self.data, AssetReference):
            raise TypeError("Observation data must be a NumPy array, text, or AssetReference")

        if self.timestamp is not None and self.timestamps is not None:
            raise ValueError("Use timestamp for a frame or timestamps for a sequence, not both")
        if self.timestamp is not None:
            self.timestamp = float(self.timestamp)
            if not math.isfinite(self.timestamp):
                raise ValueError("TactileObservation.timestamp must be finite")
        if self.timestamps is not None:
            timestamps = np.asarray(self.timestamps, dtype=np.float64)
            if timestamps.ndim != 1:
                raise ValueError("TactileObservation.timestamps must be one-dimensional")
            if timestamps.size == 0 or not np.isfinite(timestamps).all():
                raise ValueError("TactileObservation.timestamps must be non-empty and finite")
            if np.any(np.diff(timestamps) < 0):
                raise ValueError("TactileObservation.timestamps must be non-decreasing")
            if isinstance(self.data, np.ndarray):
                if self.data.ndim == 0 or self.data.shape[0] != timestamps.shape[0]:
                    raise ValueError(
                        "Materialized sequence data must share the timestamps leading dimension"
                    )
            self.timestamps = timestamps

        for attribute in ("sensor", "unit", "frame", "encoding"):
            value = getattr(self, attribute)
            if value is not None:
                value = value.strip()
                if not value:
                    raise ValueError(f"TactileObservation.{attribute} must not be empty")
                setattr(self, attribute, value)
        self.metadata = dict(self.metadata)

    @property
    def is_lazy(self) -> bool:
        return isinstance(self.data, AssetReference)


TACTILE_MODALITIES = frozenset(
    {
        Modality.VISION_TACTILE,
        Modality.TAXEL,
        Modality.WRENCH,
        Modality.VIBRATION,
        Modality.TEMPERATURE,
        Modality.PRESSURE,
    }
)


@dataclass
class TactileSample:
    """A normalized frame, sequence, or trajectory from a published dataset.

    Observation names are adapter-defined stable keys such as ``touch.left`` or
    ``vision.scene``. Labels remain task-specific, while ``group_id`` captures
    an object, participant, trial, or other unit that must stay intact when
    producing leakage-safe splits.
    """

    sample_id: str
    observations: Mapping[str, TactileObservation]
    kind: SampleKind = SampleKind.FRAME
    labels: dict[str, Any] = field(default_factory=dict)
    task: str | None = None
    split: str | None = None
    group_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sample_id = self.sample_id.strip()
        if not self.sample_id:
            raise ValueError("TactileSample.sample_id must not be empty")
        self.kind = SampleKind.parse(self.kind)

        normalized: dict[str, TactileObservation] = {}
        for raw_name, observation in self.observations.items():
            name = raw_name.strip()
            if not name:
                raise ValueError("TactileSample observation names must not be empty")
            if name in normalized:
                raise ValueError(f"Duplicate observation name after normalization: {name}")
            if not isinstance(observation, TactileObservation):
                raise TypeError(f"Observation '{name}' must be a TactileObservation")
            normalized[name] = observation
        if not normalized:
            raise ValueError("TactileSample must contain at least one observation")
        if not any(obs.modality in TACTILE_MODALITIES for obs in normalized.values()):
            raise ValueError("TactileSample must contain at least one tactile or haptic modality")
        self.observations = normalized

        for attribute in ("task", "split", "group_id"):
            value = getattr(self, attribute)
            if value is not None:
                value = value.strip()
                if not value:
                    raise ValueError(f"TactileSample.{attribute} must not be empty")
                setattr(self, attribute, value)
        self.labels = dict(self.labels)
        self.metadata = dict(self.metadata)

    @property
    def modalities(self) -> frozenset[Modality]:
        """Modalities present in the sample."""
        return frozenset(observation.modality for observation in self.observations.values())

    @property
    def is_lazy(self) -> bool:
        """Whether at least one observation still points to an external asset."""
        return any(observation.is_lazy for observation in self.observations.values())

    def by_modality(self, modality: Modality | str) -> dict[str, TactileObservation]:
        """Return observations matching ``modality`` while preserving their names."""
        parsed = Modality.parse(modality)
        return {
            name: observation
            for name, observation in self.observations.items()
            if observation.modality is parsed
        }
