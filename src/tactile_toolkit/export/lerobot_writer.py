"""LeRobot dataset v3.0 export (Parquet tables + MP4 videos + JSON/Parquet metadata).

Layout produced::

    meta/info.json
    meta/stats.json
    meta/tasks.parquet
    meta/episodes/chunk-000/file-000.parquet
    data/chunk-000/file-000.parquet
    videos/<video_key>/chunk-000/file-000.mp4

Every converted trajectory becomes one episode. Scalar bookkeeping columns
(``timestamp``, ``frame_index``, ``episode_index``, ``index``, ``task_index``)
follow the LeRobot conventions; ``timestamp`` is ``frame_index / fps`` while the
original sensor clock is preserved in ``observation.sensor_timestamp``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tactile_toolkit import schema as S
from tactile_toolkit.export.base import BaseWriter, DatasetInfo, dump_json
from tactile_toolkit.export.stats import StatsCollector, stats_to_json
from tactile_toolkit.export.video import VideoEncoder
from tactile_toolkit.types import TactileChunk, normalize_path

CODEBASE_VERSION = "v3.0"
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
TASKS_PATH = "meta/tasks.parquet"
INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"

SENSOR_TIMESTAMP_KEY = "observation.sensor_timestamp"
DENSE_MAP_KEYS = (S.DEPTH_MAP, S.PRESSURE_MAP)


def feature_key(schema_path: str) -> str:
    """``/observation/tactile/normal_force`` -> ``observation.tactile.normal_force``."""
    p = normalize_path(schema_path)
    if p == S.RAW_IMAGE:
        return "observation.images.tactile"
    return p.strip("/").replace("/", ".")


def _pa_dtype(dtype: np.dtype) -> pa.DataType:
    return pa.from_numpy_dtype(np.dtype(dtype))


def _hf_dtype(dtype: np.dtype) -> str:
    return np.dtype(dtype).name


class LeRobotV3Writer(BaseWriter):
    format_name = "lerobot"

    def __init__(
        self,
        path: str | Path,
        fps: float | None = None,
        video_codec: str | None = None,
        video_crf: int = 23,
        include_dense_maps: bool = False,
        task: str = "tactile interaction",
        robot_type: str | None = None,
        overwrite: bool = True,
    ):
        super().__init__(path)
        if self.path.exists():
            if not overwrite:
                raise FileExistsError(self.path)
            shutil.rmtree(self.path)
        for sub in ("meta", "data", "videos"):
            (self.path / sub).mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.video_codec = video_codec
        self.video_crf = video_crf
        self.include_dense_maps = include_dense_maps
        self.default_task = task
        self.robot_type = robot_type

        self._schema: pa.Schema | None = None
        self._pq: pq.ParquetWriter | None = None
        self._features: dict[str, dict[str, Any]] = {}
        self._videos: dict[str, VideoEncoder] = {}
        self._video_keys: dict[str, str] = {}  # schema path -> feature key
        self._episodes: list[dict[str, Any]] = []
        self._episode_index = 0
        self._episode_start_frame = 0
        self._episode_first_ts: float | None = None
        self._episode_stats: StatsCollector | None = None
        self._global_stats = StatsCollector(image_keys={"observation.images.tactile"})
        self._tasks: dict[str, int] = {}
        self._video_start_s: dict[str, float] = {}

    # ---------------------------------------------------------------- helpers
    def _task_index(self, task: str) -> int:
        if task not in self._tasks:
            self._tasks[task] = len(self._tasks)
        return self._tasks[task]

    def _ensure_fps(self, chunk: TactileChunk) -> None:
        if self.fps is not None:
            return
        ts = np.asarray(chunk.get(S.TIMESTAMPS, np.zeros(0)))
        if ts.shape[0] >= 2:
            dt = np.median(np.diff(ts))
            self.fps = float(1.0 / dt) if dt > 0 else 30.0
        else:
            self.fps = 30.0

    def _build_schema(self, chunk: TactileChunk) -> None:
        fields: list[pa.Field] = [
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
            pa.field(SENSOR_TIMESTAMP_KEY, pa.float64()),
        ]
        hf: dict[str, Any] = {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            SENSOR_TIMESTAMP_KEY: {"dtype": "float64", "shape": [1], "names": None},
        }
        for key, arr in chunk.items():
            key = normalize_path(key)
            if key == S.TIMESTAMPS:
                continue
            arr = np.asarray(arr)
            fkey = feature_key(key)
            if key == S.RAW_IMAGE:
                self._video_keys[key] = fkey
                H, W = arr.shape[1:3]
                hf[fkey] = {
                    "dtype": "video",
                    "shape": [int(H), int(W), 3],
                    "names": ["height", "width", "channels"],
                    "info": None,
                }
                continue
            if key in DENSE_MAP_KEYS and not self.include_dense_maps:
                continue
            sample_shape = [int(s) for s in arr.shape[1:]]
            base = _pa_dtype(arr.dtype)
            if len(sample_shape) == 0:
                fields.append(pa.field(fkey, base))
                hf[fkey] = {"dtype": _hf_dtype(arr.dtype), "shape": [1], "names": None}
            elif len(sample_shape) == 1:
                fields.append(pa.field(fkey, pa.list_(base, sample_shape[0])))
                hf[fkey] = {"dtype": _hf_dtype(arr.dtype), "shape": sample_shape, "names": None}
            else:
                t: pa.DataType = base
                for _ in sample_shape:
                    t = pa.list_(t)
                fields.append(pa.field(fkey, t))
                hf[fkey] = {"dtype": _hf_dtype(arr.dtype), "shape": sample_shape, "names": None}
        self._schema = pa.schema(fields)
        self._features = hf
        data_file = self.path / DATA_PATH.format(chunk_index=0, file_index=0)
        data_file.parent.mkdir(parents=True, exist_ok=True)
        self._pq = pq.ParquetWriter(str(data_file), self._schema, compression="snappy")

    @staticmethod
    def _to_column(arr: np.ndarray, field_type: pa.DataType) -> pa.Array:
        if arr.ndim == 1:
            return pa.array(arr, type=field_type)
        if arr.ndim == 2 and pa.types.is_fixed_size_list(field_type):
            flat = pa.array(np.ascontiguousarray(arr).reshape(-1), type=field_type.value_type)
            return pa.FixedSizeListArray.from_arrays(flat, arr.shape[1])
        # Nested variable-size lists for rank >= 2.
        return pa.array(arr.tolist(), type=field_type)

    # -------------------------------------------------------------- lifecycle
    def _begin_episode(self) -> None:
        self._episode_start_frame = self.frames_written
        self._episode_first_ts = None
        self._episode_stats = StatsCollector(image_keys={"observation.images.tactile"})
        for fkey in self._video_keys.values():
            enc = self._videos.get(fkey)
            self._video_start_s[fkey] = enc.duration_s if enc is not None else 0.0

    def _append(self, chunk: TactileChunk) -> None:
        self._ensure_fps(chunk)
        if self._schema is None:
            self._build_schema(chunk)
            for fkey in self._video_keys.values():
                self._video_start_s.setdefault(fkey, 0.0)
        assert self._schema is not None and self._pq is not None and self.fps is not None
        n = chunk.num_frames
        ts = (
            np.asarray(chunk[S.TIMESTAMPS], dtype=np.float64)
            if S.TIMESTAMPS in chunk
            else (self.frames_written + np.arange(n)) / self.fps
        )
        if self._episode_first_ts is None:
            self._episode_first_ts = float(ts[0]) if n else 0.0
        frame_index = self.episode_frames + np.arange(n, dtype=np.int64)
        task_idx = self._task_index(self.default_task)

        columns: dict[str, pa.Array] = {
            "timestamp": pa.array((frame_index / self.fps).astype(np.float32)),
            "frame_index": pa.array(frame_index),
            "episode_index": pa.array(np.full(n, self._episode_index, dtype=np.int64)),
            "index": pa.array(self.frames_written + np.arange(n, dtype=np.int64)),
            "task_index": pa.array(np.full(n, task_idx, dtype=np.int64)),
            SENSOR_TIMESTAMP_KEY: pa.array(ts, type=pa.float64()),
        }
        stats_batch: dict[str, np.ndarray] = {}
        for key, arr in chunk.items():
            key = normalize_path(key)
            if key == S.TIMESTAMPS:
                continue
            arr = np.asarray(arr)
            fkey = feature_key(key)
            if key == S.RAW_IMAGE:
                enc = self._videos.get(fkey)
                if enc is None:
                    enc = VideoEncoder(
                        self.path / VIDEO_PATH.format(video_key=fkey, chunk_index=0, file_index=0),
                        fps=self.fps,
                        codec=self.video_codec,
                        crf=self.video_crf,
                    )
                    self._videos[fkey] = enc
                enc.write(arr)
                stats_batch[fkey] = arr
                continue
            if fkey not in self._schema.names:
                continue
            columns[fkey] = self._to_column(arr, self._schema.field(fkey).type)
            stats_batch[fkey] = arr
        for name in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            stats_batch[name] = columns[name].to_numpy(zero_copy_only=False)
        table = pa.table({name: columns[name] for name in self._schema.names}, schema=self._schema)
        self._pq.write_table(table)
        assert self._episode_stats is not None
        self._episode_stats.update(stats_batch)
        self._global_stats.update(stats_batch)

    def _end_episode(self) -> None:
        assert self.fps is not None
        n = self.episode_frames
        ep: dict[str, Any] = {
            "episode_index": self._episode_index,
            "tasks": [self.default_task],
            "length": n,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": self._episode_start_frame,
            "dataset_to_index": self._episode_start_frame + n,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for fkey in self._video_keys.values():
            start = self._video_start_s.get(fkey, 0.0)
            ep[f"videos/{fkey}/chunk_index"] = 0
            ep[f"videos/{fkey}/file_index"] = 0
            ep[f"videos/{fkey}/from_timestamp"] = float(start)
            ep[f"videos/{fkey}/to_timestamp"] = float(start + n / self.fps)
        if self._episode_stats is not None:
            for fkey, s in stats_to_json(self._episode_stats.result()).items():
                for stat_name, value in s.items():
                    ep[f"stats/{fkey}/{stat_name}"] = value
        self._episodes.append(ep)
        self._episode_index += 1

    def _finalize(self, info: DatasetInfo) -> None:
        if self._pq is not None:
            self._pq.close()
        video_infos = {fkey: enc.close() for fkey, enc in self._videos.items()}
        for fkey, vinfo in video_infos.items():
            if fkey in self._features:
                self._features[fkey]["info"] = vinfo
        fps = self.fps or info.fps or 30.0
        self._write_tasks()
        self._write_episodes()
        stats = stats_to_json(self._global_stats.result())
        dump_json(self.path / STATS_PATH, stats)
        meta = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.robot_type or info.robot_type,
            "total_episodes": len(self._episodes),
            "total_frames": self.frames_written,
            "total_tasks": len(self._tasks),
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "fps": int(round(fps)),
            "splits": {"train": f"0:{len(self._episodes)}"},
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH if self._video_keys else None,
            "features": self._features,
        }
        dump_json(self.path / INFO_PATH, meta)
        extra = info.to_attrs()
        extra["fps_measured"] = fps
        dump_json(self.path / "meta" / "tactile_info.json", extra)

    def abort(self) -> None:
        if self._pq is not None:
            try:
                self._pq.close()
            except Exception:  # noqa: BLE001
                pass
        for enc in self._videos.values():
            try:
                enc.close()
            except Exception:  # noqa: BLE001
                pass

    # ---------------------------------------------------------------- metadata
    def _write_tasks(self) -> None:
        tasks = list(self._tasks.items()) or [(self.default_task, 0)]
        names = [t for t, _ in tasks]
        idx = [i for _, i in tasks]
        table = pa.table(
            {"task_index": pa.array(idx, pa.int64()), "task": pa.array(names, pa.string())}
        )
        # pandas-compatible metadata so readers see ``task`` as the index.
        pandas_md = {
            "index_columns": ["task"],
            "column_indexes": [
                {
                    "name": None,
                    "field_name": None,
                    "pandas_type": "unicode",
                    "numpy_type": "object",
                    "metadata": {"encoding": "UTF-8"},
                }
            ],
            "columns": [
                {
                    "name": "task_index",
                    "field_name": "task_index",
                    "pandas_type": "int64",
                    "numpy_type": "int64",
                    "metadata": None,
                },
                {
                    "name": "task",
                    "field_name": "task",
                    "pandas_type": "unicode",
                    "numpy_type": "object",
                    "metadata": None,
                },
            ],
            "creator": {"library": "pyarrow", "version": pa.__version__},
            "pandas_version": "2.2.0",
        }
        table = table.replace_schema_metadata({b"pandas": json.dumps(pandas_md).encode()})
        pq.write_table(table, str(self.path / TASKS_PATH))

    def _write_episodes(self) -> None:
        if not self._episodes:
            return
        keys: list[str] = []
        for ep in self._episodes:
            for k in ep:
                if k not in keys:
                    keys.append(k)
        columns = {k: [ep.get(k) for ep in self._episodes] for k in keys}
        table = pa.table(columns)
        out = self.path / EPISODES_PATH.format(chunk_index=0, file_index=0)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, str(out))


def lerobot_summary(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path / INFO_PATH, encoding="utf-8") as fh:
        info = json.load(fh)
    data_files = sorted(str(p.relative_to(path)) for p in (path / "data").rglob("*.parquet"))
    videos = sorted(str(p.relative_to(path)) for p in (path / "videos").rglob("*.mp4"))
    return {"path": str(path), "info": info, "data_files": data_files, "videos": videos}
