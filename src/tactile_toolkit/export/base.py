"""Writer interface and the dataset-level metadata bundle passed at finalization."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tactile_toolkit.schema import SCHEMA_VERSION
from tactile_toolkit.types import Modality, SensorMetadata, TactileChunk


@dataclass
class DatasetInfo:
    """Everything a writer needs to describe the exported dataset."""

    modalities: list[Modality] = field(default_factory=list)
    sensors: list[SensorMetadata] = field(default_factory=list)
    fps: float = 30.0
    source: dict[str, Any] = field(default_factory=dict)
    qa: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    pipeline: dict[str, Any] = field(default_factory=dict)
    task: str = "tactile interaction"
    robot_type: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_attrs(self) -> dict[str, Any]:
        """JSON-compatible attribute dictionary (without bulky statistics)."""
        return json_ready(
            {
                "schema_version": SCHEMA_VERSION,
                "modalities": [m.value for m in self.modalities],
                "sensors": [s.to_dict() for s in self.sensors],
                "fps": self.fps,
                "source": self.source,
                "qa": self.qa,
                "pipeline": self.pipeline,
                "task": self.task,
                "robot_type": self.robot_type,
                **self.extra,
            }
        )


def json_ready(obj: Any) -> Any:
    """Recursively convert NumPy scalars/arrays, paths and enums to JSON-native types."""
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Modality):
        return obj.value
    if hasattr(obj, "to_dict"):
        return json_ready(obj.to_dict())
    return obj


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(json_ready(obj), fh, indent=2)


class BaseWriter(ABC):
    """Streaming dataset writer.

    Lifecycle: ``begin_episode()`` (optional, implicit on first ``append``),
    ``append(chunk)`` repeatedly, ``end_episode()``, then ``finalize(info)``
    exactly once. Writers may be reused across several episodes.
    """

    format_name: str = "base"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.frames_written = 0
        self.episode_frames = 0
        self._in_episode = False
        self._finalized = False

    def __enter__(self) -> BaseWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None and not self._finalized:
            self.finalize(DatasetInfo())
        elif exc_type is not None:
            self.abort()

    def begin_episode(self) -> None:
        if self._in_episode:
            raise RuntimeError("Episode already in progress")
        self._in_episode = True
        self.episode_frames = 0
        self._begin_episode()

    def append(self, chunk: TactileChunk) -> None:
        if self._finalized:
            raise RuntimeError("Writer already finalized")
        if not self._in_episode:
            self.begin_episode()
        n = chunk.num_frames
        if n == 0:
            return
        self._append(chunk)
        self.frames_written += n
        self.episode_frames += n

    def end_episode(self) -> None:
        if not self._in_episode:
            return
        self._end_episode()
        self._in_episode = False

    def finalize(self, info: DatasetInfo) -> Path:
        if self._finalized:
            return self.path
        if self._in_episode:
            self.end_episode()
        self._finalize(info)
        self._finalized = True
        return self.path

    def abort(self) -> None:  # noqa: B027 - optional override
        """Release resources after a failure without writing metadata."""

    # ------------------------------------------------------------ overrides
    def _begin_episode(self) -> None:  # noqa: B027
        pass

    @abstractmethod
    def _append(self, chunk: TactileChunk) -> None: ...

    def _end_episode(self) -> None:  # noqa: B027
        pass

    @abstractmethod
    def _finalize(self, info: DatasetInfo) -> None: ...
