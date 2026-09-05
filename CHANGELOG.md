# Changelog

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
