"""Reader interface shared by all raw-log backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from tactile_toolkit.types import ChannelInfo, Message, Stream


class UnsupportedLogError(ValueError):
    """Raised when a file cannot be handled by any registered reader."""


class BaseReader(ABC):
    """Lazily iterate decoded messages from a raw sensor log.

    Implementations must yield messages in non-decreasing timestamp order per
    topic. Messages are decoded to NumPy arrays by :mod:`tactile_toolkit.ingest.decode`.
    """

    def __enter__(self) -> BaseReader:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:  # noqa: B027 - optional override
        """Release file handles."""

    @abstractmethod
    def channels(self) -> list[ChannelInfo]:
        """Describe every topic available in the log."""

    @abstractmethod
    def iter_messages(self, topics: Iterable[str] | None = None) -> Iterator[Message]:
        """Yield decoded messages, optionally restricted to ``topics``."""

    # ------------------------------------------------------------------ helpers
    def topic_names(self) -> list[str]:
        return [c.topic for c in self.channels()]

    def read_stream(self, topic: str) -> Stream:
        """Load an entire topic into memory as a :class:`Stream`."""
        timestamps: list[float] = []
        samples: list[np.ndarray] = []
        msg_type = ""
        for msg in self.iter_messages([topic]):
            msg_type = msg.msg_type
            timestamps.append(msg.timestamp)
            samples.append(msg.data)
        if not samples:
            for c in self.channels():
                if c.topic == topic:
                    msg_type = c.msg_type
            return Stream(topic, msg_type, np.zeros((0,), dtype=np.float64), np.zeros((0,)))
        data = np.stack(samples, axis=0)
        return _sorted_stream(Stream(topic, msg_type, np.asarray(timestamps), data))

    def read_streams(self, topics: Iterable[str]) -> dict[str, Stream]:
        """Load several small topics in a single pass over the log."""
        topics = list(topics)
        if not topics:
            return {}
        buffers: dict[str, tuple[list[float], list[np.ndarray], list[str]]] = {
            t: ([], [], []) for t in topics
        }
        for msg in self.iter_messages(topics):
            ts, data, types = buffers[msg.topic]
            ts.append(msg.timestamp)
            data.append(msg.data)
            if not types:
                types.append(msg.msg_type)
        out: dict[str, Stream] = {}
        type_by_topic = {c.topic: c.msg_type for c in self.channels()}
        for topic, (ts, data, types) in buffers.items():
            msg_type = types[0] if types else type_by_topic.get(topic, "")
            if data:
                out[topic] = _sorted_stream(
                    Stream(topic, msg_type, np.asarray(ts), np.stack(data, axis=0))
                )
            else:
                out[topic] = Stream(
                    topic, msg_type, np.zeros((0,), dtype=np.float64), np.zeros((0,))
                )
        return out

    def iter_chunks(self, topic: str, chunk_size: int) -> Iterator[Stream]:
        """Yield a topic in windows of ``chunk_size`` samples without loading it fully."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        timestamps: list[float] = []
        samples: list[np.ndarray] = []
        msg_type = ""
        for msg in self.iter_messages([topic]):
            msg_type = msg.msg_type
            timestamps.append(msg.timestamp)
            samples.append(msg.data)
            if len(samples) >= chunk_size:
                yield Stream(topic, msg_type, np.asarray(timestamps), np.stack(samples, axis=0))
                timestamps, samples = [], []
        if samples:
            yield Stream(topic, msg_type, np.asarray(timestamps), np.stack(samples, axis=0))


def _sorted_stream(stream: Stream) -> Stream:
    ts = stream.timestamps
    if ts.shape[0] > 1 and np.any(np.diff(ts) < 0):
        order = np.argsort(ts, kind="stable")
        return Stream(stream.topic, stream.msg_type, ts[order], stream.data[order], stream.modality)
    return stream
