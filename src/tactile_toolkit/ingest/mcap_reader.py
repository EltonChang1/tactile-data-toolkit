"""ROS 2 MCAP reader."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path

from tactile_toolkit.ingest import decode as _decode
from tactile_toolkit.ingest.base import BaseReader
from tactile_toolkit.types import ChannelInfo, Message

log = logging.getLogger(__name__)


class McapReader(BaseReader):
    """Read ROS 2 messages from an ``.mcap`` file.

    Only channels whose schema can be decoded (ROS 2 ``cdr`` encoding) and
    whose message type has a registered decoder are exposed.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._file = open(self.path, "rb")  # noqa: SIM115 - closed in close()
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory

        self._reader = make_reader(self._file, decoder_factories=[DecoderFactory()])
        self._channels: list[ChannelInfo] | None = None

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def channels(self) -> list[ChannelInfo]:
        if self._channels is None:
            summary = self._reader.get_summary()
            infos: list[ChannelInfo] = []
            if summary is not None:
                counts = {}
                if summary.statistics is not None:
                    counts = dict(summary.statistics.channel_message_counts)
                for cid, ch in summary.channels.items():
                    schema = summary.schemas.get(ch.schema_id)
                    msg_type = schema.name if schema is not None else ""
                    infos.append(
                        ChannelInfo(
                            topic=ch.topic,
                            msg_type=msg_type,
                            message_count=counts.get(cid),
                        )
                    )
            self._channels = sorted(infos, key=lambda c: c.topic)
        return list(self._channels)

    def iter_messages(self, topics: Iterable[str] | None = None) -> Iterator[Message]:
        topic_list = list(topics) if topics is not None else None
        unsupported: set[str] = set()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for schema, channel, message, ros_msg in self._reader.iter_decoded_messages(
                topics=topic_list, log_time_order=True
            ):
                msg_type = schema.name if schema is not None else ""
                if channel.topic in unsupported:
                    continue
                try:
                    data = _decode.decode(msg_type, ros_msg)
                except _decode.UnsupportedMessageType:
                    unsupported.add(channel.topic)
                    log.warning("Skipping topic %s: unsupported type %s", channel.topic, msg_type)
                    continue
                log_time = message.log_time * 1e-9
                stamp = _decode.header_stamp(ros_msg)
                yield Message(
                    topic=channel.topic,
                    msg_type=msg_type,
                    timestamp=stamp if stamp is not None and stamp > 0 else log_time,
                    data=data,
                    log_time=log_time,
                )
