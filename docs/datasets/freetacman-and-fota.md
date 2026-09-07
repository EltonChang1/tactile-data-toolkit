# FreeTacMan and FoTa

These integrations are C3 loadable adapters: they discover official local layouts and yield normalized, lazy `TactileSample` records. They do not decode every image or video while enumerating, silently acquire a corpus, or claim C4 conversion where calibration and heterogeneous labels cannot map faithfully to Open-Tactile-Schema 0.1.

## Installation

The local adapters need only the base package. Install the official Hugging Face client when you want the optional, revision-pinned acquisition helper:

```bash
pip install -e .
pip install 'huggingface-hub>=0.24'
```

Both snapshots exceed the helper's 10 GB safety threshold. A download therefore raises `DatasetAccessError` unless the caller passes `confirm_large_download=True` after reviewing the publisher page, destination, available storage, and dataset license.

## FreeTacMan

Primary references are the [OpenDriveLab project](https://opendrivelab.com/FreeTacMan), [official dataset](https://huggingface.co/datasets/OpenDriveLab/FreeTacMan), [official code](https://github.com/OpenDriveLab/FreeTacMan), and [paper](https://arxiv.org/abs/2506.01941). The adapter pins dataset revision `030316fb41d6fa1e58cccb4bbe0f6fddbb932671` and accepts either the snapshot root or one task directory.

Download only selected task directories when possible:

```python
from tactile_toolkit.datasets import FREETACMAN_SOURCE

root = FREETACMAN_SOURCE.download(
    "datasets/freetacman",
    allow_patterns=["ArrangeFruit/*", "README.md"],
    confirm_large_download=True,
)
```

Open and validate a bounded prefix without decoding MP4 content:

```python
from tactile_toolkit.datasets import open_dataset

dataset = open_dataset("freetacman", root=root)
report = dataset.validate(split="train", limit=4, check_assets=True)
report.raise_if_invalid()
sample = next(dataset.iter_samples(split="train"))
print(sample.sample_id, sample.modalities, sample.metadata["camera_roles"])
```

The current official snapshot uses both `camera1` and `Camera1` spellings and omits camera 3 for some demonstrations, so discovery is case-insensitive and the third view is optional. The default mapping follows the published three-view release convention—cameras 1 and 2 are tactile and camera 3 is wrist vision—but a differently organized copy can pass `camera_modalities={1: "vision", 2: "vision_tactile"}` explicitly.

Each sample retains lazy MP4 `AssetReference` objects plus materialized `float64` arrays for all published TCP position, Euler, quaternion, and gripper columns. CSV headers, numeric values, finite/non-decreasing timestamps, the two required tactile views, file containment, and stable task/trajectory grouping are validated; no force, contact geometry, or calibration is inferred.

Run the repository example:

```bash
python examples/datasets/freetacman_local.py datasets/freetacman --limit 2
```

## FoTa / FoundationTactile

Primary references are the [official FoundationTactile dataset](https://huggingface.co/datasets/alanz-mit/FoundationTactile), [T3 code](https://github.com/alanzjl/t3), and [paper](https://arxiv.org/abs/2406.13640). The adapter pins revision `e1a16123575eb26e789cf0129ced6f3ba081f2ed`, whose data are hosted as 24 split ZIP volumes and a final ZIP segment rather than individually selectable TAR shards.

Downloading FoTa requires the complete data-volume set, about 397 GB before accounting for working space and extraction:

```python
from tactile_toolkit.datasets import FOUNDATION_TACTILE_SOURCE

snapshot = FOUNDATION_TACTILE_SOURCE.download(
    "datasets/foundation-tactile-download",
    allow_patterns=["dataset/FoTa_dataset.z*", "dataset/FoTa_dataset.zip", "README.md"],
    confirm_large_download=True,
)
```

Follow the official dataset card to reassemble and unpack the multi-volume ZIP, ensure enough space for both archives and extracted files, and pass the extracted FoTa root to the adapter. The expected native structure is `<source>/<train|val>/count.txt` plus `data-*.tar`; the adapter also recognizes `data_*.tar` because that spelling appears in the official card.

```python
from tactile_toolkit.datasets import open_dataset

dataset = open_dataset(
    "fota",
    root="datasets/foundation-tactile-extracted",
    sources=["object_folder"],
    sensor_by_source={"object_folder": "GelSight variant No. 2"},
)
report = dataset.validate(split="val", limit=8, check_assets=True)
report.raise_if_invalid()
sample = next(dataset.iter_samples(split="val"))
print(sample.sample_id, sample.labels, sample.observations["touch"].data.member)
```

The reader streams TAR entries sequentially, requires every WebDataset key to have exactly one JPEG and one JSON object, bounds sidecar size, rejects unsafe member paths, and never extracts archive members. Labels remain uncoerced because constituent schemas differ; sensor identity comes from the JSON when present, then an explicit `sensor_by_source` mapping, and finally a clearly marked source-directory fallback.

Run the repository example:

```bash
python examples/datasets/foundation_tactile_local.py datasets/foundation-tactile-extracted --split val --limit 2
```

## Support boundaries

- The Apache-2.0 toolkit does not relicense either dataset; every sample retains the official revision and provenance.
- Hub Xet object identifiers are not represented as SHA-256 checksums because the publisher does not label them as such.
- FreeTacMan videos stay lazy and FoTa JPEGs stay as TAR-member references, which keeps iteration cheap and makes downstream decoding an explicit choice.
- FoTa's arbitrary task JSON is preserved rather than forced into one label ontology; task-specific converters can be added only when their semantics and units are verified.
