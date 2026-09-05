"""Write synthetic scenarios to real on-disk log formats (MCAP, ROS 1 bag, CSV/NPY)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from tactile_toolkit.synthetic.scenario import (
    IMAGE_TYPE,
    POSE_TYPE,
    TAXEL_TYPE,
    WRENCH_TYPE,
    SyntheticReader,
    SyntheticScenario,
)
from tactile_toolkit.types import Message


def _split_time(t: float) -> tuple[int, int]:
    sec = int(np.floor(t))
    nanosec = int(round((t - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return sec, nanosec


class _RosMessageFactory:
    """Build rosbags message objects for the scenario topics."""

    def __init__(self, typestore: Any, ros1: bool):
        self.ts = typestore
        self.ros1 = ros1
        T = typestore.types
        self.Time = T["builtin_interfaces/msg/Time"]
        self.Header = T["std_msgs/msg/Header"]
        self.Image = T[IMAGE_TYPE]
        self.WrenchStamped = T[WRENCH_TYPE]
        self.Wrench = T["geometry_msgs/msg/Wrench"]
        self.Vector3 = T["geometry_msgs/msg/Vector3"]
        self.MultiArray = T[TAXEL_TYPE]
        self.Layout = T["std_msgs/msg/MultiArrayLayout"]
        self.Dim = T["std_msgs/msg/MultiArrayDimension"]
        self.PoseStamped = T[POSE_TYPE]
        self.Pose = T["geometry_msgs/msg/Pose"]
        self.Point = T["geometry_msgs/msg/Point"]
        self.Quaternion = T["geometry_msgs/msg/Quaternion"]

    def header(self, t: float, frame_id: str) -> Any:
        sec, nanosec = _split_time(t)
        stamp = self.Time(sec=sec, nanosec=nanosec)
        if self.ros1:
            return self.Header(seq=0, stamp=stamp, frame_id=frame_id)
        return self.Header(stamp=stamp, frame_id=frame_id)

    def build(self, msg: Message, scenario: SyntheticScenario) -> Any:
        if msg.msg_type == IMAGE_TYPE:
            frame = np.ascontiguousarray(msg.data, dtype=np.uint8)
            h, w = frame.shape[:2]
            return self.Image(
                header=self.header(msg.timestamp, "gel"),
                height=h,
                width=w,
                encoding="rgb8",
                is_bigendian=0,
                step=w * 3,
                data=frame.reshape(-1),
            )
        if msg.msg_type == WRENCH_TYPE:
            d = [float(x) for x in msg.data]
            return self.WrenchStamped(
                header=self.header(msg.timestamp, "wrist"),
                wrench=self.Wrench(
                    force=self.Vector3(x=d[0], y=d[1], z=d[2]),
                    torque=self.Vector3(x=d[3], y=d[4], z=d[5]),
                ),
            )
        if msg.msg_type == TAXEL_TYPE:
            data = np.asarray(msg.data, dtype=np.float32)
            dims = []
            stride = int(data.size)
            for label, size in zip(("rows", "cols", "axes"), data.shape):
                dims.append(self.Dim(label=label, size=int(size), stride=int(stride)))
                stride //= int(size)
            return self.MultiArray(
                layout=self.Layout(dim=dims, data_offset=0), data=data.reshape(-1)
            )
        if msg.msg_type == POSE_TYPE:
            d = [float(x) for x in msg.data]
            return self.PoseStamped(
                header=self.header(msg.timestamp, "base"),
                pose=self.Pose(
                    position=self.Point(x=d[0], y=d[1], z=d[2]),
                    orientation=self.Quaternion(x=d[3], y=d[4], z=d[5], w=d[6]),
                ),
            )
        raise ValueError(f"Unsupported synthetic message type {msg.msg_type}")


def write_mcap(path: str | Path, scenario: SyntheticScenario | None = None) -> Path:
    """Write the scenario as a ROS 2 MCAP file (CDR encoding, zstd chunks)."""
    from mcap.well_known import SchemaEncoding
    from mcap.writer import Writer
    from rosbags.typesys import Stores, get_typestore

    scenario = scenario or SyntheticScenario()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    factory = _RosMessageFactory(typestore, ros1=False)
    reader = SyntheticReader(scenario)

    with open(path, "wb") as fh:
        writer = Writer(fh)
        writer.start(profile="ros2", library="tactile-toolkit synthetic")
        schema_ids: dict[str, int] = {}
        channel_ids: dict[str, int] = {}
        for ch in reader.channels():
            if ch.msg_type not in schema_ids:
                msgdef, _ = typestore.generate_msgdef(ch.msg_type, ros_version=2)
                schema_ids[ch.msg_type] = writer.register_schema(
                    ch.msg_type, SchemaEncoding.ROS2, msgdef.encode()
                )
            channel_ids[ch.topic] = writer.register_channel(
                ch.topic, "cdr", schema_ids[ch.msg_type]
            )
        seq = 0
        for msg in reader.iter_messages():
            ros_msg = factory.build(msg, scenario)
            raw = typestore.serialize_cdr(ros_msg, msg.msg_type)
            t_ns = int(round(msg.timestamp * 1e9))
            writer.add_message(
                channel_id=channel_ids[msg.topic],
                log_time=t_ns,
                publish_time=t_ns,
                sequence=seq,
                data=bytes(raw),
            )
            seq += 1
        writer.finish()
    return path


def write_ros1_bag(path: str | Path, scenario: SyntheticScenario | None = None) -> Path:
    """Write the scenario as a ROS 1 ``.bag`` file."""
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore

    scenario = scenario or SyntheticScenario()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    typestore = get_typestore(Stores.ROS1_NOETIC)
    factory = _RosMessageFactory(typestore, ros1=True)
    reader = SyntheticReader(scenario)

    with Writer(path) as writer:
        conns = {
            ch.topic: writer.add_connection(ch.topic, ch.msg_type, typestore=typestore)
            for ch in reader.channels()
        }
        for msg in reader.iter_messages():
            ros_msg = factory.build(msg, scenario)
            raw = typestore.serialize_ros1(ros_msg, msg.msg_type)
            writer.write(conns[msg.topic], int(round(msg.timestamp * 1e9)), raw)
    return path


def write_csv_dir(directory: str | Path, scenario: SyntheticScenario | None = None) -> Path:
    """Write the scenario as a folder of CSV tables plus an NPY image tensor."""
    scenario = scenario or SyntheticScenario()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    reader = SyntheticReader(scenario)

    def dump_csv(name: str, topic: str) -> None:
        stream = reader.read_stream(topic)
        flat = stream.data.reshape(stream.data.shape[0], -1)
        with open(directory / f"{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["timestamp", *[f"c{i}" for i in range(flat.shape[1])]])
            for t, row in zip(stream.timestamps, flat):
                w.writerow([f"{t:.9f}", *[f"{v:.6g}" for v in row]])

    if scenario.include_taxels:
        dump_csv("taxels", scenario.taxel_topic)
    if scenario.include_wrench:
        dump_csv("wrench", scenario.wrench_topic)
    if scenario.include_pose:
        dump_csv("pose", scenario.pose_topic)
    if scenario.include_gelsight:
        stream = reader.read_stream(scenario.gelsight_topic)
        np.save(directory / "gelsight.npy", stream.data)
        np.save(directory / "gelsight_timestamps.npy", stream.timestamps)
    return directory
