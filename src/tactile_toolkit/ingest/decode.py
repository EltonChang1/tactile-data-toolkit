"""Convert decoded ROS 1 / ROS 2 message objects into NumPy arrays.

Both ``mcap_ros2`` and ``rosbags`` expose messages as attribute objects with
identical field names, so a single duck-typed decoder serves both backends.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class UnsupportedMessageType(ValueError):
    pass


def canonical_type(msg_type: str) -> str:
    """Map ``pkg/msg/Name`` and ``pkg/Name`` to ``pkg/Name``."""
    parts = msg_type.split("/")
    if len(parts) == 3 and parts[1] == "msg":
        return f"{parts[0]}/{parts[2]}"
    return msg_type


def header_stamp(msg: Any) -> float | None:
    """Return the header stamp in seconds, or ``None`` when the message has none."""
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None) if header is not None else getattr(msg, "stamp", None)
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", None)
    if sec is None:
        sec = getattr(stamp, "secs", None)
    nsec = getattr(stamp, "nanosec", None)
    if nsec is None:
        nsec = getattr(stamp, "nsecs", 0)
    if sec is None:
        return None
    return float(sec) + float(nsec) * 1e-9


_CHANNELS = {
    "rgb8": 3,
    "bgr8": 3,
    "rgba8": 4,
    "bgra8": 4,
    "mono8": 1,
    "8UC1": 1,
    "8UC3": 3,
    "8UC4": 4,
    "mono16": 1,
    "16UC1": 1,
}


def decode_image(msg: Any) -> np.ndarray:
    """Decode ``sensor_msgs/Image`` into an ``(H, W, 3)`` uint8 RGB array."""
    h, w = int(msg.height), int(msg.width)
    encoding = (
        str(getattr(msg, "encoding", "rgb8")).lower() if getattr(msg, "encoding", None) else ""
    )
    data = msg.data
    buf = (
        np.frombuffer(bytes(data), dtype=np.uint8)
        if isinstance(data, bytes | bytearray | memoryview)
        else np.asarray(data)
    )
    if buf.dtype != np.uint8:
        buf = buf.astype(np.uint8, copy=False).ravel()
    buf = buf.ravel()

    if encoding in ("mono16", "16uc1"):
        arr16 = buf.view(np.uint16).reshape(h, w)
        arr = (arr16 >> 8).astype(np.uint8)
        return np.repeat(arr[..., None], 3, axis=-1)

    channels = _CHANNELS.get(encoding)
    if channels is None:
        if h * w == 0:
            raise UnsupportedMessageType(f"Empty image with encoding '{encoding}'")
        channels = buf.size // (h * w)
        if channels not in (1, 3, 4):
            raise UnsupportedMessageType(f"Unsupported image encoding '{encoding}'")
    step = int(getattr(msg, "step", 0) or 0)
    if step and step != w * channels:
        rows = buf[: h * step].reshape(h, step)[:, : w * channels]
        arr = rows.reshape(h, w, channels)
    else:
        arr = buf[: h * w * channels].reshape(h, w, channels)

    if encoding.startswith("bgr"):
        arr = arr[..., [2, 1, 0]]
    elif channels == 4:
        arr = arr[..., :3]
    elif channels == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return np.ascontiguousarray(arr[..., :3])


def decode_compressed_image(msg: Any) -> np.ndarray:
    import cv2

    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise UnsupportedMessageType("Failed to decode compressed image")
    return np.ascontiguousarray(img[..., ::-1])


def _vec3(v: Any) -> list[float]:
    return [float(v.x), float(v.y), float(v.z)]


def decode_wrench(msg: Any) -> np.ndarray:
    w = getattr(msg, "wrench", msg)
    return np.asarray(_vec3(w.force) + _vec3(w.torque), dtype=np.float64)


def decode_pose(msg: Any) -> np.ndarray:
    p = getattr(msg, "pose", msg)
    # PoseWithCovarianceStamped nests pose.pose
    if hasattr(p, "pose") and not hasattr(p, "position"):
        p = p.pose
    q = p.orientation
    return np.asarray(
        _vec3(p.position) + [float(q.x), float(q.y), float(q.z), float(q.w)], dtype=np.float64
    )


def decode_transform(msg: Any) -> np.ndarray:
    t = getattr(msg, "transform", msg)
    q = t.rotation
    return np.asarray(
        _vec3(t.translation) + [float(q.x), float(q.y), float(q.z), float(q.w)], dtype=np.float64
    )


def decode_joint_state(msg: Any) -> np.ndarray:
    return np.asarray(list(msg.position), dtype=np.float64)


def decode_multiarray(msg: Any) -> np.ndarray:
    data = np.asarray(list(msg.data) if not hasattr(msg.data, "shape") else msg.data)
    layout = getattr(msg, "layout", None)
    dims = list(getattr(layout, "dim", []) or []) if layout is not None else []
    if dims:
        shape = [int(d.size) for d in dims if int(d.size) > 0]
        if shape and int(np.prod(shape)) == data.size:
            data = data.reshape(shape)
    if data.dtype.kind in "iu" and data.dtype.itemsize < 4:
        return data.astype(np.int32)
    return data


def decode_scalar(msg: Any) -> np.ndarray:
    return np.asarray([msg.data], dtype=np.float64)


_DECODERS: dict[str, Callable[[Any], np.ndarray]] = {
    "sensor_msgs/Image": decode_image,
    "sensor_msgs/CompressedImage": decode_compressed_image,
    "geometry_msgs/WrenchStamped": decode_wrench,
    "geometry_msgs/Wrench": decode_wrench,
    "geometry_msgs/PoseStamped": decode_pose,
    "geometry_msgs/Pose": decode_pose,
    "geometry_msgs/PoseWithCovarianceStamped": decode_pose,
    "geometry_msgs/TransformStamped": decode_transform,
    "geometry_msgs/Transform": decode_transform,
    "sensor_msgs/JointState": decode_joint_state,
    "std_msgs/Float32MultiArray": decode_multiarray,
    "std_msgs/Float64MultiArray": decode_multiarray,
    "std_msgs/Int8MultiArray": decode_multiarray,
    "std_msgs/UInt8MultiArray": decode_multiarray,
    "std_msgs/Int16MultiArray": decode_multiarray,
    "std_msgs/UInt16MultiArray": decode_multiarray,
    "std_msgs/Int32MultiArray": decode_multiarray,
    "std_msgs/UInt32MultiArray": decode_multiarray,
    "std_msgs/Float32": decode_scalar,
    "std_msgs/Float64": decode_scalar,
    "std_msgs/Int32": decode_scalar,
}


def register_decoder(msg_type: str, fn: Callable[[Any], np.ndarray]) -> None:
    """Register a decoder for a custom message type (``pkg/Name`` or ``pkg/msg/Name``)."""
    _DECODERS[canonical_type(msg_type)] = fn


def supports(msg_type: str) -> bool:
    return canonical_type(msg_type) in _DECODERS


def decode(msg_type: str, msg: Any) -> np.ndarray:
    """Decode ``msg`` of ROS type ``msg_type`` into an array."""
    fn = _DECODERS.get(canonical_type(msg_type))
    if fn is None:
        data = getattr(msg, "data", None)
        if data is not None and not isinstance(data, str | bytes):
            try:
                return np.asarray(list(data), dtype=np.float64)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise UnsupportedMessageType(msg_type) from exc
        raise UnsupportedMessageType(msg_type)
    return fn(msg)
