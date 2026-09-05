"""Assign a sensor modality to each topic and pick the master clock."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from tactile_toolkit.ingest.decode import canonical_type
from tactile_toolkit.types import ChannelInfo, Modality

_TYPE_RULES: dict[str, Modality] = {
    "sensor_msgs/Image": Modality.VISION_TACTILE,
    "sensor_msgs/CompressedImage": Modality.VISION_TACTILE,
    "geometry_msgs/WrenchStamped": Modality.WRENCH,
    "geometry_msgs/Wrench": Modality.WRENCH,
    "geometry_msgs/PoseStamped": Modality.POSE,
    "geometry_msgs/Pose": Modality.POSE,
    "geometry_msgs/PoseWithCovarianceStamped": Modality.POSE,
    "geometry_msgs/TransformStamped": Modality.POSE,
    "geometry_msgs/Transform": Modality.POSE,
    "sensor_msgs/JointState": Modality.JOINT_STATE,
}

_TOPIC_HINTS: tuple[tuple[re.Pattern[str], Modality], ...] = (
    (re.compile(r"(gelsight|digit|gel|image|camera|rgb)", re.I), Modality.VISION_TACTILE),
    (re.compile(r"(wrench|force_torque|ft_sensor|/ft\b|netft|ati)", re.I), Modality.WRENCH),
    (re.compile(r"(taxel|skin|xela|uskin|pressure|tactile|contact)", re.I), Modality.TAXEL),
    (re.compile(r"(pose|tcp|ee_|end_effector|tf\b)", re.I), Modality.POSE),
    (re.compile(r"(joint)", re.I), Modality.JOINT_STATE),
)


@dataclass
class TopicMap:
    """Resolved mapping of topics to modalities plus the master-clock topic."""

    master: str
    assignments: dict[str, Modality] = field(default_factory=dict)

    @property
    def master_modality(self) -> Modality:
        return self.assignments[self.master]

    def topics_for(self, modality: Modality) -> list[str]:
        return [t for t, m in self.assignments.items() if m == modality]

    def auxiliary_topics(self) -> list[str]:
        return [
            t
            for t, m in self.assignments.items()
            if t != self.master and m not in (Modality.UNKNOWN,)
        ]

    @property
    def modalities(self) -> list[Modality]:
        seen: list[Modality] = []
        for m in self.assignments.values():
            if m not in seen and m is not Modality.UNKNOWN:
                seen.append(m)
        return seen


class ModalityResolver:
    """Infer modalities from message types, topic names, and sample shapes.

    Parameters
    ----------
    overrides:
        Explicit ``{topic: modality}`` assignments that win over heuristics.
    master:
        Topic to use as the master clock. Defaults to the first vision-tactile
        topic, then the first taxel topic.
    include_unknown:
        Keep topics whose modality could not be resolved (as ``UNKNOWN``).
    """

    def __init__(
        self,
        overrides: Mapping[str, Modality | str] | None = None,
        master: str | None = None,
        include_unknown: bool = False,
    ):
        self.overrides = {k: Modality.parse(v) for k, v in (overrides or {}).items()}
        self.master = master
        self.include_unknown = include_unknown

    def resolve_one(
        self,
        channel: ChannelInfo,
        sample_shape: tuple[int, ...] | None = None,
        sample_dtype: str | None = None,
    ) -> Modality:
        if channel.topic in self.overrides:
            return self.overrides[channel.topic]
        ctype = canonical_type(channel.msg_type)
        if ctype in _TYPE_RULES:
            return _TYPE_RULES[ctype]
        topic_hint = self._topic_hint(channel.topic)
        if ctype.endswith("MultiArray") or ctype.startswith("tensor/"):
            if sample_shape is not None:
                by_shape = self._shape_hint(sample_shape, sample_dtype)
                if by_shape is not None and (
                    topic_hint is None or by_shape is Modality.VISION_TACTILE
                ):
                    return by_shape
            if topic_hint is not None:
                return topic_hint
            if sample_shape is not None:
                return Modality.TAXEL
        return topic_hint or Modality.UNKNOWN

    @staticmethod
    def _topic_hint(topic: str) -> Modality | None:
        for pattern, modality in _TOPIC_HINTS:
            if pattern.search(topic):
                return modality
        return None

    @staticmethod
    def _shape_hint(shape: tuple[int, ...], dtype: str | None) -> Modality | None:
        if len(shape) == 3 and shape[-1] in (1, 3, 4) and (dtype or "").startswith("uint8"):
            return Modality.VISION_TACTILE
        if len(shape) == 1 and shape[0] == 6:
            return Modality.WRENCH
        if len(shape) == 1 and shape[0] == 7:
            return Modality.POSE
        if len(shape) >= 1:
            return Modality.TAXEL
        return None

    def resolve(
        self,
        channels: Iterable[ChannelInfo],
        sample_shapes: Mapping[str, tuple[int, ...]] | None = None,
        sample_dtypes: Mapping[str, str] | None = None,
    ) -> TopicMap:
        assignments: dict[str, Modality] = {}
        for ch in channels:
            shape = (sample_shapes or {}).get(ch.topic, ch.sample_shape)
            dtype = (sample_dtypes or {}).get(ch.topic)
            if dtype is None and ch.msg_type.startswith("tensor/"):
                dtype = ch.msg_type.split("/", 1)[1]
            modality = self.resolve_one(ch, shape, dtype)
            if modality is Modality.UNKNOWN and not self.include_unknown:
                continue
            assignments[ch.topic] = modality

        master = self.master
        if master is None:
            for wanted in (Modality.VISION_TACTILE, Modality.TAXEL):
                cands = [t for t, m in assignments.items() if m == wanted]
                if cands:
                    master = cands[0]
                    break
        if master is None:
            raise ValueError(
                "No tactile topic (vision-tactile image or taxel array) found; "
                f"resolved modalities: {assignments or 'none'}"
            )
        if master not in assignments:
            raise ValueError(f"Master topic '{master}' is not present in the log")
        if assignments[master] not in (Modality.VISION_TACTILE, Modality.TAXEL):
            raise ValueError(
                f"Master topic '{master}' has modality {assignments[master].value}; "
                "the master clock must be a tactile stream"
            )
        return TopicMap(master=master, assignments=assignments)
