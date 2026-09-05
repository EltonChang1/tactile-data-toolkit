"""Conversion throughput and peak-memory benchmark on synthetic trajectories."""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tactile_toolkit.calibration.gelsight import GelSightConfig
from tactile_toolkit.export import make_writer
from tactile_toolkit.ingest.base import BaseReader
from tactile_toolkit.pipeline import ConvertPipeline, PipelineConfig
from tactile_toolkit.synthetic import GelSightSim, SyntheticScenario, wrench_at
from tactile_toolkit.synthetic.scenario import IMAGE_TYPE, WRENCH_TYPE
from tactile_toolkit.types import ChannelInfo, Message, Modality


class _PeakMemoryMonitor:
    """Sample the process RSS on a background thread and keep the maximum."""

    def __init__(self, interval_s: float = 0.02):
        self.interval_s = interval_s
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import psutil

            self._proc: Any = psutil.Process()
        except ImportError:  # pragma: no cover
            self._proc = None

    def _rss(self) -> int:
        if self._proc is not None:
            return int(self._proc.memory_info().rss)
        try:
            import resource

            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return int(ru * 1024)
        except Exception:  # noqa: BLE001 - Windows without psutil
            return 0

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self._rss())
            self._stop.wait(self.interval_s)

    def __enter__(self) -> _PeakMemoryMonitor:
        self.peak_bytes = self._rss()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self._rss())

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024**2)


class FrameBankReader(BaseReader):
    """Replay a bank of pre-rendered frames as a long trajectory.

    Rendering synthetic frames costs as much as converting them, so the
    benchmark renders ``bank_size`` distinct frames once and cycles through
    them with fresh timestamps. Consecutive frames are always different, which
    keeps the duplicate-frame check honest.
    """

    def __init__(
        self,
        n_frames: int,
        height: int = 480,
        width: int = 640,
        fps: float = 30.0,
        bank_size: int = 128,
        wrench_rate_hz: float = 1000.0,
        seed: int = 0,
    ):
        self.n_frames = int(n_frames)
        self.fps = float(fps)
        self.t0 = 1_700_000_000.0
        self.image_topic = "/gelsight/image_raw"
        self.wrench_topic = "/wrist/wrench"
        self.wrench_rate_hz = wrench_rate_hz
        sim = GelSightSim(height=height, width=width, seed=seed)
        bank_size = max(2, min(bank_size, self.n_frames))
        self.bank = np.stack([f for _, f, _ in sim.iter_frames(bank_size, fps)])
        self.height, self.width = height, width

    def channels(self) -> list[ChannelInfo]:
        n_wrench = int(self.n_frames / self.fps * self.wrench_rate_hz)
        return [
            ChannelInfo(
                self.image_topic,
                IMAGE_TYPE,
                self.n_frames,
                Modality.VISION_TACTILE,
                (self.height, self.width, 3),
            ),
            ChannelInfo(self.wrench_topic, WRENCH_TYPE, n_wrench, Modality.WRENCH, (6,)),
        ]

    def iter_messages(self, topics: Iterable[str] | None = None) -> Iterator[Message]:
        wanted = set(topics) if topics is not None else {self.image_topic, self.wrench_topic}
        if self.wrench_topic in wanted:
            n = int(self.n_frames / self.fps * self.wrench_rate_hz)
            rel = np.arange(n) / self.wrench_rate_hz
            data = wrench_at(rel)
            for i in range(n):
                t = self.t0 + float(rel[i])
                yield Message(self.wrench_topic, WRENCH_TYPE, t, data[i], t)
        if self.image_topic in wanted:
            k = self.bank.shape[0]
            for i in range(self.n_frames):
                t = self.t0 + i / self.fps
                yield Message(self.image_topic, IMAGE_TYPE, t, self.bank[i % k], t)


@dataclass
class BenchResult:
    frames: int
    frames_out: int
    elapsed_s: float
    fps: float
    peak_rss_mb: float
    image_size: tuple[int, int]
    formats: list[str]
    chunk_frames: int
    source: str
    outputs: list[str] = field(default_factory=list)
    targets: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        fps_ok = "PASS" if self.targets.get("fps", False) else "FAIL"
        mem_ok = "PASS" if self.targets.get("memory", False) else "FAIL"
        h, w = self.image_size
        lines = [
            f"source         : {self.source}",
            f"frames         : {self.frames} in / {self.frames_out} out @ {h}x{w}",
            f"formats        : {', '.join(self.formats)} (chunk_frames={self.chunk_frames})",
            f"elapsed        : {self.elapsed_s:.2f} s",
            f"throughput     : {self.fps:.1f} frames/s   [target > 120: {fps_ok}]",
            f"peak RSS       : {self.peak_rss_mb:.0f} MB    [target < 4096: {mem_ok}]",
        ]
        return "\n".join(lines)


def run_benchmark(
    frames: int = 10_000,
    height: int = 480,
    width: int = 640,
    formats: Iterable[str] = ("zarr",),
    chunk_frames: int = 64,
    out_dir: str | Path | None = None,
    keep_outputs: bool = False,
    track_markers: bool = True,
    depth_method: str = "dst",
    stride: int = 8,
    store_depth_map: bool = True,
    from_mcap: bool = False,
    fps_target: float = 120.0,
    memory_target_mb: float = 4096.0,
) -> BenchResult:
    """Convert a synthetic ``frames``-long trajectory and report throughput / peak memory."""
    formats = list(formats)
    tmp_root = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="tactile_bench_"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    reader: BaseReader | Path
    if from_mcap:
        from tactile_toolkit.synthetic import write_mcap

        scenario = SyntheticScenario.for_frames(
            frames, image_height=height, image_width=width, include_taxels=False
        )
        reader = write_mcap(tmp_root / "bench_input.mcap", scenario)
        source = f"mcap ({reader.stat().st_size / 1e6:.1f} MB)"
    else:
        reader = FrameBankReader(frames, height=height, width=width)
        source = "in-memory frame bank"

    gel = GelSightConfig(
        track_markers=track_markers,
        depth_method=depth_method,  # type: ignore[arg-type]
        stride=stride,
        store_depth_map=store_depth_map,
    )
    config = PipelineConfig(chunk_frames=chunk_frames, gelsight=gel)
    pipeline = ConvertPipeline(config)
    writers = []
    for fmt in formats:
        ext = {"zarr": ".zarr", "hdf5": ".h5", "lerobot": ""}[fmt]
        writers.append(make_writer(fmt, tmp_root / f"bench_output_{fmt}{ext}"))

    with _PeakMemoryMonitor() as mem:
        t0 = time.perf_counter()
        result = pipeline.run(reader, writers, source_name="benchmark")
        elapsed = time.perf_counter() - t0

    throughput = frames / elapsed if elapsed > 0 else float("inf")
    bench = BenchResult(
        frames=frames,
        frames_out=result.frames_out,
        elapsed_s=elapsed,
        fps=throughput,
        peak_rss_mb=mem.peak_mb,
        image_size=(height, width),
        formats=formats,
        chunk_frames=chunk_frames,
        source=source,
        outputs=[str(p) for p in result.outputs],
        targets={"fps": throughput > fps_target, "memory": mem.peak_mb < memory_target_mb},
    )
    if not keep_outputs and out_dir is None:
        shutil.rmtree(tmp_root, ignore_errors=True)
        bench.outputs = []
    return bench
