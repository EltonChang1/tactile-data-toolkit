"""Core data types shared across ingestion, calibration, and export."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


class Modality(StrEnum):
    """Sensor modality of a data stream."""

    VISION_TACTILE = "vision_tactile"
    TAXEL = "taxel"
    WRENCH = "wrench"
    POSE = "pose"
    JOINT_STATE = "joint_state"
    VISION = "vision"
    LANGUAGE = "language"
    AUDIO = "audio"
    VIBRATION = "vibration"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    IMU = "imu"
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, value: str | Modality) -> Modality:
        if isinstance(value, Modality):
            return value
        try:
            return cls(value)
        except ValueError:
            return cls[value.upper()]


@dataclass
class Message:
    """One decoded message from a raw log."""

    topic: str
    msg_type: str
    timestamp: float
    """Sensor timestamp in seconds (header stamp when available, log time otherwise)."""
    data: np.ndarray
    log_time: float | None = None


@dataclass
class Stream:
    """A time series of samples for a single topic."""

    topic: str
    msg_type: str
    timestamps: np.ndarray
    """Shape ``(T,)`` float64 seconds, monotonically non-decreasing."""
    data: np.ndarray
    """Shape ``(T, ...)``."""
    modality: Modality = Modality.UNKNOWN

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=np.float64)
        self.data = np.asarray(self.data)
        if self.timestamps.ndim != 1:
            raise ValueError("Stream.timestamps must be one-dimensional")
        if self.data.shape[0] != self.timestamps.shape[0]:
            raise ValueError(
                f"Stream '{self.topic}': data has {self.data.shape[0]} samples but "
                f"{self.timestamps.shape[0]} timestamps"
            )

    def __len__(self) -> int:
        return int(self.timestamps.shape[0])

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape[1:])

    @property
    def nominal_rate_hz(self) -> float:
        if len(self) < 2:
            return float("nan")
        dt = np.median(np.diff(self.timestamps))
        return float(1.0 / dt) if dt > 0 else float("inf")

    def slice(self, start: int, stop: int) -> Stream:
        return Stream(
            topic=self.topic,
            msg_type=self.msg_type,
            timestamps=self.timestamps[start:stop],
            data=self.data[start:stop],
            modality=self.modality,
        )


@dataclass
class ChannelInfo:
    """Static description of a topic in a raw log."""

    topic: str
    msg_type: str
    message_count: int | None = None
    modality: Modality = Modality.UNKNOWN
    sample_shape: tuple[int, ...] | None = None
    """Shape of one decoded sample when known up front (tensor / synthetic sources)."""


@dataclass
class SensorMetadata:
    """Describes the physical sensor that produced a trajectory."""

    sensor_name: str
    modality: Modality
    source_topic: str | None = None
    num_elements: int = 0
    """``N`` in the schema: number of taxels or sampled surface points."""
    layout: dict[str, Any] = field(default_factory=dict)
    """Geometry description (``rows``, ``cols``, ``pitch_m``, ``pixel_size_m``, ``stride``...)."""
    units: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor_name": self.sensor_name,
            "modality": self.modality.value,
            "source_topic": self.source_topic,
            "num_elements": int(self.num_elements),
            "layout": dict(self.layout),
            "units": dict(self.units),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SensorMetadata:
        return cls(
            sensor_name=d["sensor_name"],
            modality=Modality.parse(d["modality"]),
            source_topic=d.get("source_topic"),
            num_elements=int(d.get("num_elements", 0)),
            layout=dict(d.get("layout", {})),
            units=dict(d.get("units", {})),
            extra=dict(d.get("extra", {})),
        )


class TactileChunk(Mapping[str, np.ndarray]):
    """A time window of schema-conformant arrays keyed by schema path.

    All arrays share the same leading dimension ``T``. Chunks are the unit of
    work that flows through the pipeline, so that trajectories of arbitrary
    length can be converted with bounded memory.
    """

    def __init__(self, arrays: Mapping[str, np.ndarray] | None = None):
        self._arrays: dict[str, np.ndarray] = {}
        if arrays:
            for key, value in arrays.items():
                self[key] = value

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        key = normalize_path(key)
        value = np.asarray(value)
        if self._arrays:
            expected = self.num_frames
            if value.shape[0] != expected:
                raise ValueError(f"Array '{key}' has {value.shape[0]} frames; chunk has {expected}")
        self._arrays[key] = value

    def __getitem__(self, key: str) -> np.ndarray:
        return self._arrays[normalize_path(key)]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and normalize_path(key) in self._arrays

    def __iter__(self) -> Iterator[str]:
        return iter(self._arrays)

    def __len__(self) -> int:
        return len(self._arrays)

    @property
    def num_frames(self) -> int:
        if not self._arrays:
            return 0
        return int(next(iter(self._arrays.values())).shape[0])

    def select(self, mask: np.ndarray) -> TactileChunk:
        """Return a new chunk keeping only frames where ``mask`` is True."""
        mask = np.asarray(mask, dtype=bool)
        return TactileChunk({k: v[mask] for k, v in self._arrays.items()})

    def concat(self, other: TactileChunk) -> TactileChunk:
        if set(self._arrays) != set(other._arrays):
            raise ValueError("Cannot concatenate chunks with different keys")
        return TactileChunk(
            {k: np.concatenate([self._arrays[k], other._arrays[k]], axis=0) for k in self._arrays}
        )

    def to_dict(self) -> dict[str, np.ndarray]:
        return dict(self._arrays)


def normalize_path(path: str) -> str:
    """Normalize a schema key path to the canonical ``/a/b/c`` form."""
    path = path.strip().replace("\\", "/")
    while "//" in path:
        path = path.replace("//", "/")
    if not path.startswith("/"):
        path = "/" + path
    return path.rstrip("/") or "/"
