# Tactile Data Toolkit

Universal ingestion, calibration, and standardized dataset engine for robotic tactile sensing.

Optical gel fingers, capacitive skins, and wrist force/torque sensors all record contact, but they do it in incompatible file formats and coordinate frames. This package reads those raw logs, time-aligns them, applies sensor-specific calibration, and writes a single **Open-Tactile-Schema** trajectory that loads in Zarr, HDF5 (robomimic / ACT), LeRobot v3, and PyTorch.

| Input | Modalities | Output |
| --- | --- | --- |
| ROS 2 / MCAP, ROS 1 `.bag`, CSV / NPY / NPZ | Vision-tactile (GelSight-class), taxel arrays, 6-axis wrench | Zarr, HDF5, LeRobot v3 (Parquet + MP4) |

## Install

Python 3.11 or newer. From a clone of this repository:

```bash
uv sync --extra dev
```

Or with pip:

```bash
pip install -e ".[dev]"
```

PyTorch is optional. Install the `torch` extra (or `dev`) when you want `TactileZarrDataset` and `DataLoader` streaming.

## Quickstart

Generate a synthetic multi-sensor recording and convert it. No hardware required.

```bash
tactile synth scene.mcap --duration 4 --fps 30 --size 240x320
tactile inspect scene.mcap
tactile convert scene.mcap -o dataset.zarr -o dataset.h5 -o lerobot_ds --format lerobot
tactile validate dataset.zarr
```

The same path in Python:

```python
from tactile_toolkit.pipeline import ConvertPipeline
from tactile_toolkit.synthetic import SyntheticScenario, write_mcap

write_mcap("scene.mcap", SyntheticScenario(duration_s=4.0))
ConvertPipeline().convert("scene.mcap", [("zarr", "dataset.zarr")])
```

Load the Zarr store in a training loop:

```python
from tactile_toolkit.dataset import TactileZarrDataset, make_dataloader

dataset = TactileZarrDataset("dataset.zarr")
loader = make_dataloader(dataset, batch_size=16, num_workers=2)
batch = next(iter(loader))
```

## Open-Tactile-Schema

Every export uses the same hierarchical keys, regardless of source hardware. See [docs/schema.md](docs/schema.md) for units, optional fields, and the validator.

| Path | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `/timestamps` | float64 | `(T,)` | Master-clock time in seconds |
| `/observation/tactile/contact_mask` | bool | `(T, N)` | Per-element contact |
| `/observation/tactile/normal_force` | float32 | `(T, N)` | Normal force (N) |
| `/observation/tactile/shear_force` | float32 | `(T, N, 2)` | In-plane shear (N) |
| `/observation/tactile/point_cloud` | float32 | `(T, N, 6)` | `(x, y, z, fx, fy, fz)` |
| `/observation/tactile/raw_image` | uint8 | `(T, H, W, 3)` | Gel camera (vision-tactile) |
| `/observation/wrench` | float32 | `(T, 6)` | Wrist wrench when present |

## Pipeline

```text
MCAP / bag / CSV  ->  modality resolve  ->  time align  ->  quality gate
        ->  GelSight or taxel calibration  ->  3D contact cloud
        ->  Zarr / HDF5 / LeRobot v3
```

- **Alignment.** High-rate wrench streams (typically 1 kHz) are resampled onto the tactile master clock with nearest-neighbor or linear interpolation.
- **Quality gate.** Frames with timing jitter above 2 ms, frozen camera images, or non-finite auxiliary samples are dropped.
- **Vision-tactile.** Median idle reference, DST Poisson depth (optional `cumsum` integrator), Lucas–Kanade marker tracking, linear elastomer force model.
- **Taxel arrays.** Zero-tare baseline, slow drift compensation, optional grid interpolation for a pressure heat-map.
- **Streaming.** Conversion walks the log in chunks (default 64 frames) so a 10,000-frame trajectory stays under 4 GB RSS.

Architecture notes live in [docs/architecture.md](docs/architecture.md).

## Dataset ecosystem

The evidence-checked [dataset source catalog](docs/dataset-catalog.md) records canonical sources, licenses, formats, access constraints, and proposed support levels for the tactile libraries covered by this project. The [compatibility matrix](docs/dataset-coverage.md) distinguishes catalog, loading, conversion, and benchmark support while documenting how heterogeneous source data map into the toolkit.

Contributors can use the [published-dataset extension guide](docs/adding-datasets.md) to add typed metadata, lazy adapters, verified acquisition, and network-free fixtures without redistributing third-party corpora.

## Command line

```text
tactile convert INPUT... -o OUT.zarr [-o OUT.h5] [--format lerobot]
tactile inspect PATH
tactile validate DATASET.zarr
tactile synth scene.mcap --duration 4
tactile schema
tactile bench --frames 10000
```

`tactile convert` accepts `--topic /name=modality` overrides, `--master` to pick the clock topic, `--align nearest|linear`, and GelSight / taxel calibration flags (`tactile convert -h`).

## Verification

```bash
uv run pytest
uv run python benchmarks/bench_convert.py --frames 10000
```

The benchmark reports frames/s and peak RSS. The project targets are **> 120 FPS** and **< 4 GB** for a 10,000-frame 480×640 conversion.

## License

Apache License 2.0. See [LICENSE](LICENSE).
