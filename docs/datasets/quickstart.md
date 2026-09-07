# Published-data quickstart

This guide takes a fresh development install through resource discovery and one normalized sample without contacting a public dataset host. It also shows where the toolkit stops when a source is reference-only, gated, noncommercial, very large, or serialized with pickle.

## Install and list resources

From a clone of this repository:

```bash
uv sync --extra dev
uv run python examples/datasets/first_sample.py --list
```

The listing separates four independent facts: stable ID, `resource_kind`, support level, and whether a sample adapter exists. A C2 `reference_index` such as Awesome-Touch is useful for discovery but correctly prints `sample_adapter=no`, while a C3 dataset has a tested local adapter.

Python callers can apply the same typed filter:

```python
from tactile_toolkit.datasets import list_datasets

loadable_datasets = list_datasets(
    resource_kind="dataset",
    minimum_support="loadable",
)
for metadata in loadable_datasets:
    print(metadata.dataset_id, metadata.license.name)
```

An unfiltered `list_datasets()` retains backward-compatible behavior and includes datasets, collections, and reference indexes.

## Inspect the included first sample

The repository includes a two-record Touch100k-compatible fixture with placeholder JPEG assets and synthetic paraphrased descriptions. Use it to exercise the public registry, local adapter, lazy assets, labels, grouping, and JSON inspection path without a network request:

```bash
uv run python examples/datasets/first_sample.py \
  touch100k tests/fixtures/datasets/touch100k --split train
```

The command reads only the first manifest row and prints the dataset ID, resource kind, support, license, source URL, sample ID, task, split, group, labels, and per-observation storage/modality details. Image values remain lazy `AssetReference` objects, so the smoke test does not decode or materialize them.

The equivalent minimal Python path is:

```python
from tactile_toolkit.datasets import open_dataset

dataset = open_dataset(
    "touch100k",
    root="tests/fixtures/datasets/touch100k",
)
sample = next(dataset.iter_samples(split="train"))
print(sample.sample_id, sample.labels, sample.modalities)
```

For a bounded structural and local-asset check, run `dataset.validate(split="train", limit=2, check_assets=True)` and call `raise_if_invalid()` on its report.

## Move from fixtures to publisher data

Read the source's dataset guide and metadata before acquisition. A loader never grants data rights: use the publisher's exact license, citation, access flow, and revision, and keep dataset licenses separate from this repository's Apache-2.0 code license.

| Access boundary | Examples | Required action |
| --- | --- | --- |
| Revision-pinned public data | FreeTacMan, FoTa, Mendeley tactile textures | Review size and terms, explicitly acquire through the documented helper, and verify available integrity metadata |
| User-provided local data | Touch100k, Penn HaTT | Obtain from the publisher under its terms and pass the extracted local root |
| Gated data | TacVerse | Accept publisher conditions, authenticate, and explicitly confirm the large transfer |
| Code-capable serialization | TacBench, CLAMP, Mendeley tactile textures | Verify provenance and integrity, inspect in isolation, then opt in with `--trust-pickle` only if trusted |
| Discovery only | Tac2Pose, MIT GelSight studies, TVL, OAHD, LMT, Awesome-Touch | Follow primary links and limitations; do not expect a sample adapter |

Never point `--trust-pickle` at an unknown or modified file because Python pickle can execute arbitrary code. The common first-sample command forwards that flag only when explicitly supplied and otherwise preserves every adapter's safe default.

## Interpret support honestly

C3 means the toolkit can load and normalize a verified subset of native source fields, not that it can represent every modality in Open-Tactile-Schema or reproduce a published benchmark. C4 additionally requires an honest conversion path, while C5 requires tested canonical splits, metrics, and leakage controls; the [compatibility matrix](../dataset-coverage.md) records each boundary.
