"""Scripted multi-sensor recording scenario and an in-memory reader for it."""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from tactile_toolkit.ingest.base import BaseReader
from tactile_toolkit.synthetic.gelsight import GelSightSim
from tactile_toolkit.synthetic.signals import TaxelSim, pose_at, wrench_at
from tactile_toolkit.types import ChannelInfo, Message, Modality

IMAGE_TYPE = "sensor_msgs/msg/Image"
WRENCH_TYPE = "geometry_msgs/msg/WrenchStamped"
TAXEL_TYPE = "std_msgs/msg/Float32MultiArray"
POSE_TYPE = "geometry_msgs/msg/PoseStamped"


@dataclass
class SyntheticScenario:
    duration_s: float = 4.0
    t0: float = 1_700_000_000.0
    image_fps: float = 30.0
    image_height: int = 240
    image_width: int = 320
    wrench_rate_hz: float = 1000.0
    taxel_rate_hz: float = 100.0
    pose_rate_hz: float = 100.0
    include_gelsight: bool = True
    include_taxels: bool = True
    include_wrench: bool = True
    include_pose: bool = False
    taxel_rows: int = 8
    taxel_cols: int = 8
    taxel_axes: int = 1
    timestamp_jitter_s: float = 0.0
    """Gaussian jitter added to image timestamps."""
    jitter_outlier_every: int = 0
    """Every k-th image frame is delayed by 10 ms (exercises the quality gate)."""
    duplicate_frame_every: int = 0
    """Every k-th image frame repeats the previous frame (frozen camera)."""
    seed: int = 0
    gelsight_topic: str = "/gelsight/image_raw"
    wrench_topic: str = "/wrist/wrench"
    taxel_topic: str = "/skin/taxels"
    pose_topic: str = "/ee/pose"
    n_image_frames_override: int | None = field(default=None, repr=False)

    @property
    def n_image_frames(self) -> int:
        if self.n_image_frames_override is not None:
            return self.n_image_frames_override
        return int(round(self.duration_s * self.image_fps))

    @property
    def n_wrench_samples(self) -> int:
        return int(round(self.duration_s * self.wrench_rate_hz))

    @property
    def n_taxel_samples(self) -> int:
        return int(round(self.duration_s * self.taxel_rate_hz))

    @property
    def n_pose_samples(self) -> int:
        return int(round(self.duration_s * self.pose_rate_hz))

    def gelsight_sim(self) -> GelSightSim:
        return GelSightSim(height=self.image_height, width=self.image_width, seed=self.seed)

    def taxel_sim(self) -> TaxelSim:
        return TaxelSim(
            rows=self.taxel_rows, cols=self.taxel_cols, axes=self.taxel_axes, seed=self.seed + 1
        )

    @classmethod
    def for_frames(cls, n_frames: int, **kwargs) -> SyntheticScenario:
        fps = float(kwargs.pop("image_fps", 30.0))
        sc = cls(duration_s=n_frames / fps, image_fps=fps, **kwargs)
        sc.n_image_frames_override = n_frames
        return sc


class SyntheticReader(BaseReader):
    """In-memory :class:`BaseReader` producing a scenario lazily in time order."""

    def __init__(self, scenario: SyntheticScenario | None = None):
        self.scenario = scenario or SyntheticScenario()
        self._gel = self.scenario.gelsight_sim() if self.scenario.include_gelsight else None
        self._tax = self.scenario.taxel_sim() if self.scenario.include_taxels else None

    def channels(self) -> list[ChannelInfo]:
        sc = self.scenario
        out = []
        if sc.include_gelsight:
            out.append(
                ChannelInfo(
                    sc.gelsight_topic, IMAGE_TYPE, sc.n_image_frames, Modality.VISION_TACTILE
                )
            )
        if sc.include_taxels:
            out.append(ChannelInfo(sc.taxel_topic, TAXEL_TYPE, sc.n_taxel_samples, Modality.TAXEL))
        if sc.include_wrench:
            out.append(
                ChannelInfo(sc.wrench_topic, WRENCH_TYPE, sc.n_wrench_samples, Modality.WRENCH)
            )
        if sc.include_pose:
            out.append(ChannelInfo(sc.pose_topic, POSE_TYPE, sc.n_pose_samples, Modality.POSE))
        return out

    # ------------------------------------------------------------- generators
    def _images(self) -> Iterator[Message]:
        sc = self.scenario
        assert self._gel is not None
        rng = np.random.default_rng(sc.seed + 100)
        prev: np.ndarray | None = None
        for i, (t, frame, _) in enumerate(
            self._gel.iter_frames(sc.n_image_frames, sc.image_fps, sc.t0)
        ):
            ts = t
            if sc.timestamp_jitter_s > 0:
                ts += float(rng.normal(0, sc.timestamp_jitter_s))
            if sc.jitter_outlier_every and i > 0 and i % sc.jitter_outlier_every == 0:
                ts += 0.010
            if (
                sc.duplicate_frame_every
                and i > 0
                and i % sc.duplicate_frame_every == 0
                and prev is not None
            ):
                frame = prev
            prev = frame
            yield Message(sc.gelsight_topic, IMAGE_TYPE, float(ts), frame, float(ts))

    def _taxels(self) -> Iterator[Message]:
        sc = self.scenario
        assert self._tax is not None
        for i in range(sc.n_taxel_samples):
            rel = i / sc.taxel_rate_hz
            data = self._tax.sample(rel)
            yield Message(sc.taxel_topic, TAXEL_TYPE, sc.t0 + rel, data, sc.t0 + rel)

    def _wrench(self) -> Iterator[Message]:
        sc = self.scenario
        block = 1000
        for start in range(0, sc.n_wrench_samples, block):
            n = min(block, sc.n_wrench_samples - start)
            rel = (start + np.arange(n)) / sc.wrench_rate_hz
            data = wrench_at(rel, seed=sc.seed + 2)
            for j in range(n):
                yield Message(
                    sc.wrench_topic,
                    WRENCH_TYPE,
                    sc.t0 + float(rel[j]),
                    data[j],
                    sc.t0 + float(rel[j]),
                )

    def _pose(self) -> Iterator[Message]:
        sc = self.scenario
        rel = np.arange(sc.n_pose_samples) / sc.pose_rate_hz
        data = pose_at(rel)
        for j in range(sc.n_pose_samples):
            yield Message(
                sc.pose_topic, POSE_TYPE, sc.t0 + float(rel[j]), data[j], sc.t0 + float(rel[j])
            )

    def iter_messages(self, topics: Iterable[str] | None = None) -> Iterator[Message]:
        sc = self.scenario
        wanted = set(topics) if topics is not None else None
        gens: list[Iterator[Message]] = []
        if sc.include_gelsight and (wanted is None or sc.gelsight_topic in wanted):
            gens.append(self._images())
        if sc.include_taxels and (wanted is None or sc.taxel_topic in wanted):
            gens.append(self._taxels())
        if sc.include_wrench and (wanted is None or sc.wrench_topic in wanted):
            gens.append(self._wrench())
        if sc.include_pose and (wanted is None or sc.pose_topic in wanted):
            gens.append(self._pose())
        if len(gens) == 1:
            yield from gens[0]
            return
        heap: list[tuple[float, int, int, Message]] = []
        counter = 0
        for gi, g in enumerate(gens):
            try:
                m = next(g)
            except StopIteration:
                continue
            heap.append((m.timestamp, gi, counter, m))
            counter += 1
        heapq.heapify(heap)
        while heap:
            _, gi, _, m = heapq.heappop(heap)
            yield m
            try:
                nxt = next(gens[gi])
            except StopIteration:
                continue
            heapq.heappush(heap, (nxt.timestamp, gi, counter, nxt))
            counter += 1
