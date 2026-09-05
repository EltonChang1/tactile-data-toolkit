"""End-to-end streaming conversion: raw log -> aligned, calibrated, schema-conformant dataset."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tactile_toolkit import schema as S
from tactile_toolkit.calibration.gelsight import GelSightCalibrator, GelSightConfig
from tactile_toolkit.calibration.taxel import TaxelCalibrator, TaxelConfig
from tactile_toolkit.export import BaseWriter, DatasetInfo, StatsCollector, make_writer
from tactile_toolkit.ingest import (
    BaseReader,
    ModalityResolver,
    QaConfig,
    QaReport,
    QualityGate,
    TimeAligner,
    TopicMap,
    open_log,
)
from tactile_toolkit.ingest.align import AlignMode
from tactile_toolkit.types import Modality, Stream

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    chunk_frames: int = 64
    """Master-stream frames processed per step; bounds peak memory."""
    align_mode: AlignMode = "linear"
    max_gap_s: float | None = None
    """Auxiliary samples farther than this from a master frame are treated as missing."""
    qa: QaConfig = field(default_factory=QaConfig)
    gelsight: GelSightConfig = field(default_factory=GelSightConfig)
    taxel: TaxelConfig = field(default_factory=TaxelConfig)
    topic_overrides: dict[str, str] = field(default_factory=dict)
    master_topic: str | None = None
    include_wrench: bool = True
    include_pose: bool = True
    progress: bool = False
    overlap_export: bool = True
    """Run statistics and writer appends on a background thread, one chunk behind calibration."""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gelsight"] = self.gelsight.to_dict()
        d["taxel"] = self.taxel.to_dict()
        d["qa"] = self.qa.to_dict()
        return d


@dataclass
class ConvertResult:
    frames_in: int
    frames_out: int
    qa: QaReport
    topic_map: TopicMap
    outputs: list[Path]
    elapsed_s: float
    info: DatasetInfo

    @property
    def fps(self) -> float:
        return self.frames_in / self.elapsed_s if self.elapsed_s > 0 else float("inf")

    def summary(self) -> str:
        lines = [
            f"master topic   : {self.topic_map.master} ({self.topic_map.master_modality.value})",
            f"frames         : {self.frames_in} read, {self.frames_out} written",
            f"throughput     : {self.fps:.1f} frames/s ({self.elapsed_s:.2f} s)",
            f"quality gate   : {self.qa.summary()}",
        ]
        for p in self.outputs:
            lines.append(f"output         : {p}")
        return "\n".join(lines)


class ConvertPipeline:
    """Wire reader -> modality resolver -> aligner -> quality gate -> calibrator -> writers."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    # ------------------------------------------------------------- utilities
    def resolve(self, reader: BaseReader) -> TopicMap:
        resolver = ModalityResolver(
            overrides=self.config.topic_overrides, master=self.config.master_topic
        )
        return resolver.resolve(reader.channels())

    def _make_calibrator(
        self, modality: Modality, topic: str
    ) -> GelSightCalibrator | TaxelCalibrator:
        name = topic.strip("/").replace("/", "_") or modality.value
        if modality is Modality.VISION_TACTILE:
            return GelSightCalibrator(self.config.gelsight, sensor_name=name)
        if modality is Modality.TAXEL:
            return TaxelCalibrator(self.config.taxel, sensor_name=name)
        raise ValueError(f"No calibrator for master modality {modality.value}")

    # ------------------------------------------------------------------- run
    def run(
        self,
        source: BaseReader | str | Path,
        writers: Sequence[BaseWriter],
        finalize: bool = True,
        source_name: str | None = None,
        extra_info: Mapping[str, Any] | None = None,
    ) -> ConvertResult:
        """Convert one trajectory (one episode) and stream it into ``writers``."""
        cfg = self.config
        t_start = time.perf_counter()
        owned_reader = not isinstance(source, BaseReader)
        reader = open_log(source) if owned_reader else source
        try:
            topic_map = self.resolve(reader)
            master = topic_map.master
            master_modality = topic_map.master_modality
            log.info("Master stream %s (%s)", master, master_modality.value)

            wrench_topic = (
                next(iter(topic_map.topics_for(Modality.WRENCH)), None)
                if cfg.include_wrench
                else None
            )
            pose_topic = (
                next(iter(topic_map.topics_for(Modality.POSE)), None) if cfg.include_pose else None
            )
            aux_topics = [t for t in (wrench_topic, pose_topic) if t is not None]
            ignored = [t for t in topic_map.auxiliary_topics() if t not in aux_topics]
            if ignored:
                log.info("Ignoring topics without a schema slot: %s", ignored)
            aux: dict[str, Stream] = reader.read_streams(aux_topics) if aux_topics else {}
            aux = {k: v for k, v in aux.items() if len(v) > 0}
            aligner = TimeAligner(cfg.align_mode, max_gap_s=cfg.max_gap_s)

            calibrator = self._make_calibrator(master_modality, master)
            gate = QualityGate(cfg.qa)
            stats = StatsCollector(image_keys={S.RAW_IMAGE})
            modalities = [master_modality]
            if wrench_topic in aux:
                modalities.append(Modality.WRENCH)
            if pose_topic in aux:
                modalities.append(Modality.POSE)

            for w in writers:
                w.begin_episode()

            frames_in = 0
            frames_out = 0
            progress = None
            if cfg.progress:
                try:
                    from tqdm import tqdm

                    total = next(
                        (c.message_count for c in reader.channels() if c.topic == master), None
                    )
                    progress = tqdm(total=total, unit="frame", desc="convert")
                except ImportError:  # pragma: no cover
                    progress = None

            writer_pool = (
                ThreadPoolExecutor(max_workers=len(writers))
                if cfg.overlap_export and len(writers) > 1
                else None
            )

            def sink(out: Any) -> None:
                stats.update(out)
                if writer_pool is None:
                    for w in writers:
                        w.append(out)
                else:
                    # Each writer owns its own files; encoding in every format
                    # at once keeps compressors and video encoders busy together.
                    list(writer_pool.map(lambda w: w.append(out), writers))

            # Statistics and compression release the GIL, so exporting the
            # previous chunk while calibrating the next one overlaps cleanly.
            # At most one chunk is in flight, which keeps memory bounded.
            executor = ThreadPoolExecutor(max_workers=1) if cfg.overlap_export else None
            pending: Future[None] | None = None
            try:
                for chunk in reader.iter_chunks(master, cfg.chunk_frames):
                    n = len(chunk)
                    frames_in += n
                    aligned = {t: aligner.align(chunk.timestamps, s) for t, s in aux.items()}
                    frames = chunk.data if master_modality is Modality.VISION_TACTILE else None
                    keep, _ = gate.evaluate(chunk.timestamps, frames, aligned)
                    if progress is not None:
                        progress.update(n)
                    if not keep.any():
                        continue
                    kept_data = chunk.data[keep]
                    out = calibrator.process(kept_data)
                    out[S.TIMESTAMPS] = chunk.timestamps[keep].astype(np.float64)
                    if wrench_topic in aligned:
                        out[S.WRENCH] = (
                            aligned[wrench_topic][keep].astype(np.float32).reshape(-1, 6)
                        )
                    if pose_topic in aligned:
                        out[S.POSE] = aligned[pose_topic][keep].astype(np.float32).reshape(-1, 7)
                    frames_out += out.num_frames
                    if executor is None:
                        sink(out)
                    else:
                        if pending is not None:
                            pending.result()
                        pending = executor.submit(sink, out)
                if pending is not None:
                    pending.result()
            finally:
                if executor is not None:
                    executor.shutdown(wait=True)
                if writer_pool is not None:
                    writer_pool.shutdown(wait=True)

            if progress is not None:
                progress.close()
            for w in writers:
                w.end_episode()

            fps = 1.0 / gate.nominal_period_s if gate.nominal_period_s else 30.0
            info = DatasetInfo(
                modalities=modalities,
                sensors=[calibrator.metadata()] if frames_out else [],
                fps=fps,
                source={
                    "path": str(source_name or (source if owned_reader else type(reader).__name__)),
                    "topics": {t: m.value for t, m in topic_map.assignments.items()},
                    "master_topic": master,
                },
                qa=gate.report.to_dict(),
                stats=stats.result(),
                pipeline=cfg.to_dict(),
                extra=dict(extra_info or {}),
            )
            outputs: list[Path] = []
            if finalize:
                for w in writers:
                    outputs.append(w.finalize(info))
            else:
                outputs = [w.path for w in writers]
            elapsed = time.perf_counter() - t_start
            return ConvertResult(
                frames_in, frames_out, gate.report, topic_map, outputs, elapsed, info
            )
        except Exception:
            for w in writers:
                w.abort()
            raise
        finally:
            if owned_reader:
                reader.close()

    def convert(
        self,
        source: BaseReader | str | Path,
        outputs: Iterable[tuple[str, str | Path] | str | Path],
        writer_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
        **run_kwargs: Any,
    ) -> ConvertResult:
        """Convenience wrapper: build writers from ``(format, path)`` pairs or paths and run."""
        from tactile_toolkit.export import infer_format

        writers: list[BaseWriter] = []
        for item in outputs:
            if isinstance(item, tuple):
                fmt, path = item
            else:
                fmt, path = infer_format(item), item
            kwargs = dict((writer_kwargs or {}).get(fmt, {}))
            writers.append(make_writer(fmt, path, **kwargs))
        return self.run(source, writers, **run_kwargs)
