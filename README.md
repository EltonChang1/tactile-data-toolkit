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

The evidence-checked [dataset source catalog](docs/dataset-catalog.md) records canonical sources, licenses, formats, access constraints, and support levels for the tactile libraries covered by this project. The [compatibility matrix](docs/dataset-coverage.md) distinguishes catalog, loading, conversion, and benchmark support while documenting how heterogeneous source data map into the toolkit.

Contributors can use the [published-dataset extension guide](docs/adding-datasets.md) to add typed metadata, lazy adapters, verified acquisition, and network-free fixtures without redistributing third-party corpora.

### Supported published datasets

#### FreeTacMan (OpenDriveLab)

[FreeTacMan](https://huggingface.co/datasets/OpenDriveLab/FreeTacMan) is a large robot-free visuo-tactile manipulation corpus whose synchronized demonstrations support representation and policy learning for contact-rich tasks. It contains MP4 camera streams with timestamped tool-center-point and gripper CSV trajectories, which the toolkit exposes through a revision-pinned local lazy adapter with explicit camera-role and integrity validation.

#### FoTa / FoundationTactile

[FoTa / FoundationTactile](https://huggingface.co/datasets/alanz-mit/FoundationTactile) is a multi-sensor, multi-task tactile image corpus designed to support transferable representation learning across heterogeneous optical sensors. It contains more than three million paired JPEG and task-specific JSON records in WebDataset TAR shards, which the toolkit streams from an extracted revision-pinned snapshot while preserving source, sensor, split, labels, and member provenance.

#### Tac2Pose

[Tac2Pose](https://arxiv.org/abs/2204.11701) is an MIT tactile pose-estimation study using GelSlim 3.0 observations and known object geometry to localize 20 real objects from first contact. It describes paired tactile images, calibrated poses, contact renderings, meshes, and calibration touches, which the toolkit exposes as an evidence-backed C2 metadata record because no active official manifest or reusable data terms could be verified.

#### MIT GelSight research datasets

[MIT GelSight research datasets](https://people.csail.mit.edu/yuan_wz/hardness-estimation.htm) are separate study resources for hardness, marker-based force and shear, and slip detection rather than one uniformly packaged corpus. They contain study-specific GelSight image sequences and experimental labels where released, which the toolkit exposes through an umbrella plus three C2 child records while withholding loaders where files or data terms are missing.

#### Touch100k

[Touch100k](https://cocacola-lab.github.io/Touch100k/) is a touch-language-vision corpus designed to align GelSight representations with generated and human-corrected tactile descriptions at two levels of detail. It contains paired touch and scene JPEGs with `img`, `sentence_desc`, and `phrase_desc` JSON-lines records, which the toolkit loads lazily from a user-obtained local copy under CC BY-NC 4.0 while flagging that the current official Drive release is partial.

#### Touch-Vision-Language / TVL

[Touch-Vision-Language / TVL](https://tactile-vlm.github.io/) is a 43,741-pair DIGIT, scene-vision, and open-vocabulary language corpus that matters for multimodal tactile alignment. It contains SSVTP and HCT images with human and generated descriptions, which the toolkit exposes as pinned C2 discovery metadata without downloading because the data cards do not state reuse terms and the original layout has a documented touch/image swap.

#### Meta DIGIT benchmarks / Sparsh TacBench

[Meta DIGIT benchmarks / Sparsh TacBench](https://github.com/facebookresearch/sparsh) is a six-task suite for evaluating transferable representations across DIGIT, GelSight Mini, and linked tactile sources. It contains released force/slip trajectories and relative-pose sequences in native pickle files, which the toolkit exposes through pinned C3 local adapters that require explicit trust while keeping the broader, externally sourced suite at C2.

#### TacVerse

[TacVerse](https://lannwei.github.io/Tactile_Database/) is a 106,800-image benchmark spanning seven vision-based tactile sensors that supports cross-sensor shape, grating, and force research. It contains gated CC BY 4.0 JPEG/CSV task archives, which the toolkit downloads only through an explicitly confirmed pinned Hugging Face source and loads lazily at C3 without fabricating split labels that the official code derives at runtime.

#### Open Access Haptic Database / OAHD

[Open Access Haptic Database / OAHD](https://www.oahd.gatech.edu/) is Georgia Tech Healthcare Robotics Lab's study index for multimodal robot haptics and reusable sensor designs. It contains separate force, thermal, vibration, position, script, and hardware releases, which the toolkit exposes as C2 collection metadata while requiring study-level terms and citations before any child adapter is added.

#### CLAMP

[CLAMP](https://emprise.cs.cornell.edu/clamp/) is a crowdsourced multimodal haptic corpus covering millions of measurements from household objects, devices, and participants for material and compliance recognition. It contains raw device archives and a filtered CC BY 4.0 object-array NPZ with force, thermal, vibration, proprioception, labels, and vision predictions, which the toolkit exposes through a C3 contact-sequence adapter with explicit trust and object-safe grouping.

#### LMT Haptic Texture Database

[LMT Haptic Texture Database](https://www.ce.cit.tum.de/en/lmt/forschung/datensaetze/texture-database/) is a Technical University of Munich collection of controlled and freehand surface explorations whose distinct 69-, 108-, and 184-material releases support texture recognition research. It contains release-dependent acceleration, force or motion settings, images, and perceptual annotations, which the toolkit exposes as separate C2 discovery records without downloading because the publisher provides no substantive reuse license or verified manifest.

#### Penn Haptic Texture Toolkit / HaTT

[Penn Haptic Texture Toolkit / HaTT](https://repository.upenn.edu/bitstreams/960863b6-df14-4770-9068-b1b2bf6a50f1/download) is a 100-surface collection of measured explorations and data-driven haptic models for recognition and virtual-texture rendering under attributed noncommercial research terms. It contains 10 kHz XML acceleration, force, position, and speed recordings plus surface images, models, and code, which the toolkit exposes through a bounded C3 local XML adapter while leaving unverified rendering packages untouched.

#### Multimodal Tactile Texture Dataset (Mendeley Data)

[Multimodal Tactile Texture Dataset](https://data.mendeley.com/datasets/n666tk4mw9/1) is a CC BY 4.0 pressure-and-IMU corpus for classifying 12 surfaces explored at the artifact-verified speeds of 30, 35, and 40 mm/s. It contains paired barometer and nine-axis IMU pickle trials, which the toolkit acquires only after explicit 3.83 GB confirmation and publisher SHA-256 verification and loads at C3 from an extracted trusted copy without duplicating derived axis files.

### Dataset guides

See the [FreeTacMan and FoTa guide](docs/datasets/freetacman-and-fota.md) for acquisition, validation, and runnable local examples.

See the [Tac2Pose, MIT GelSight, and Touch100k guide](docs/datasets/tac2pose-gelsight-touch100k.md) for honest discovery records, noncommercial terms, local layout, validation, and a runnable Touch100k example.

See the [TVL, TacBench, TacVerse, OAHD, and CLAMP guide](docs/datasets/tvl-tacbench-tacverse-oahd-clamp.md) for access boundaries, pinned releases, guarded serialization, local layouts, and runnable examples.

See the [LMT, Penn HaTT, and Mendeley tactile textures guide](docs/datasets/lmt-hatt-mendeley.md) for release boundaries, licensing, guarded local loading, verified acquisition, layouts, and runnable examples.

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
