# Architecture

The toolkit is a streaming conversion engine: raw logs enter as typed message streams, a shared tactile schema leaves as chunked arrays. Nothing between those two ends holds a full trajectory in memory unless the caller asks for it.

## Stages

1. **Read.** `tactile_toolkit.ingest.open_log` dispatches on path:
   - `.mcap` — ROS 2 CDR via `mcap` + `mcap-ros2-support`
   - `.bag` / rosbag2 directories — ROS 1 and ROS 2 via `rosbags`
   - `.csv` / `.npy` / `.npz` (file or folder) — column-spec driven lab dumps
2. **Resolve.** `ModalityResolver` maps topics to `vision_tactile | taxel | wrench | pose` from message type, topic name, and sample shape. Callers can override any topic and pick the master clock.
3. **Align.** `TimeAligner` resamples auxiliary streams onto the master timestamps. `nearest` is used for discrete signals; `linear` for analog wrench. Samples farther than `max_gap_s` become NaN so the quality gate can drop them.
4. **Quality gate.** `QualityGate` is stateful across chunks. It drops frames whose inter-frame interval deviates from the nominal period by more than 2 ms (configurable), frozen/duplicated images, non-monotonic timestamps, and non-finite auxiliary rows.
5. **Calibrate.** The master modality selects a calibrator:
   - **GelSight.** Median of the first idle frames is the photometric reference. RGB differences become surface slopes; a DST Poisson solver (or the faster `cumsum` integrator) reconstructs depth. Marker dots are tracked with pyramidal Lucas–Kanade and converted to shear. Depth and shear become per-element forces through a linear elastomer model.
   - **Taxel.** The first idle frames set a zero-tare baseline. Slow EMA drift compensation updates free taxels. Optional `griddata` interpolation emits a pressure heat-map.
6. **Point cloud.** `SensorGeometry` places each taxel or pooled pixel in metres. `build_point_cloud` writes `(T, N, 6)` as `(x, y, z, fx, fy, fz)`.
7. **Export.** One or more writers consume `TactileChunk` objects:
   - **Zarr v3** — schema paths as groups, zstd chunks, episode index arrays, `/stats`
   - **HDF5** — robomimic `/data/demo_N/obs/<key>` with `schema_path` attributes
   - **LeRobot v3** — `meta/info.json`, `meta/stats.json`, Parquet tables, MP4 gel video via PyAV

`ConvertPipeline` overlaps calibration of chunk *k+1* with compression of chunk *k* on a background thread so CPU-bound reconstruction and I/O-bound encoding run together.

## Memory bound

Writers grow arrays along time only. Image arrays are chunked so no compressed block exceeds roughly 16 MiB. The quality gate and calibrators keep only a reference frame, a marker grid, and the previous timestamp. That is what keeps a 10,000-frame conversion under 4 GB RSS.

## Training loaders

`TactileZarrDataset` opens the store lazily inside each worker process (`__getstate__` drops file handles). `make_dataloader` turns that into a `torch.utils.data.DataLoader` that works under both fork and spawn.
