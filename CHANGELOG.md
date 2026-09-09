# Changelog

## 0.2.0

- Add Open X-Embodiment RLDS episode adaptation with explicit dataset-specific transforms
- Add canonical 7-DoF RT-X observation/action schema fields and episode markers
- Add exact NumPy RT-1-X 11-token action codec and 15-step model-window preparation
- Export workspace RGB as LeRobot video and canonical actions in native HDF5 layout

## 0.1.0

Initial release.

- Unified readers for MCAP, ROS 1 bags, rosbag2, CSV, NPY, and NPZ
- Time alignment (nearest / linear) and a quality gate for jitter, frozen frames, and non-finite samples
- GelSight calibration: reference estimation, DST Poisson depth, Lucas–Kanade markers, force field
- Taxel calibration: zero-tare, drift compensation, optional pressure-map interpolation
- Open-Tactile-Schema validator
- Streaming writers for Zarr v3, robomimic-style HDF5, and LeRobot v3
- PyTorch `TactileZarrDataset` with multi-worker `DataLoader` support
- `tactile` CLI (`convert`, `inspect`, `validate`, `synth`, `schema`, `bench`)
- Synthetic multi-sensor recordings for tests and the quickstart
