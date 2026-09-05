"""ROS 1 ``.bag`` and ROS 2 rosbag2 directory reader built on ``rosbags``."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

from tactile_toolkit.ingest import decode as _decode
from tactile_toolkit.ingest.base import BaseReader
from tactile_toolkit.types import ChannelInfo, Message

log = logging.getLogger(__name__)


class BagReader(BaseReader):
    """Read ROS 1 bags (``.bag``) and ROS 2 bags (directory with ``metadata.yaml``).

    ``rosbags`` is a pure-Python implementation and does not require a ROS
    installation. Custom message types embedded in ROS 1 bags are registered
    automatically from their connection metadata.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        from rosbags.highlevel import AnyReader

        self._reader = AnyReader([self.path])
        self._reader.open()
        self._channels: list[ChannelInfo] | None = None

    def close(self) -> None:
        try:
            self._reader.close()
        except Exception:  # noqa: BLE001 - already closed
            pass

    def channels(self) -> list[ChannelInfo]:
        if self._channels is None:
            infos = []
            for topic, info in self._reader.topics.items():
                infos.append(
                    ChannelInfo(
                        topic=topic,
                        msg_type=str(info.msgtype),
                        message_count=int(info.msgcount) if info.msgcount is not None else None,
                    )
                )
            self._channels = sorted(infos, key=lambda c: c.topic)
        return list(self._channels)

    def iter_messages(self, topics: Iterable[str] | None = None) -> Iterator[Message]:
        wanted = set(topics) if topics is not None else None
        connections = [c for c in self._reader.connections if wanted is None or c.topic in wanted]
        if not connections:
            return
        unsupported: set[str] = set()
        for conn, ts_ns, raw in self._reader.messages(connections=connections):
            if conn.topic in unsupported:
                continue
            try:
                ros_msg = self._reader.deserialize(raw, conn.msgtype)
                data = _decode.decode(conn.msgtype, ros_msg)
            except _decode.UnsupportedMessageType:
                unsupported.add(conn.topic)
                log.warning("Skipping topic %s: unsupported type %s", conn.topic, conn.msgtype)
                continue
            log_time = ts_ns * 1e-9
            stamp = _decode.header_stamp(ros_msg)
            yield Message(
                topic=conn.topic,
                msg_type=conn.msgtype,
                timestamp=stamp if stamp is not None and stamp > 0 else log_time,
                data=data,
                log_time=log_time,
            )
