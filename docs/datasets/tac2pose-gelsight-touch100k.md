# Tac2Pose, MIT GelSight studies, and Touch100k

Step 05 separates source discovery from data loading. Tac2Pose and the MIT GelSight studies are C2 discoverable records because an active, licensed source layout is not available for each one, while Touch100k is a C3 local adapter for files users obtain from the publisher under CC BY-NC 4.0.

## Tac2Pose

The primary references are the [paper](https://arxiv.org/abs/2204.11701), [journal article](https://doi.org/10.1177/02783649231196925), and project URL named by the paper. The paper describes GelSlim 3.0 image pairs, corresponding ground-truth object poses, object models and contact renderings for 20 objects, plus 10,000 paired touches used to train its image-to-contact-mask model.

The project URL did not expose an active file manifest during the 2026-09-06 audit, the original implementation is not public, and the dataset's own license is not stated. The journal article's CC BY-NC license covers the publication, so the toolkit does not extend it to files that the publisher has not licensed explicitly.

Use the record to inspect evidence and access notes without triggering network activity:

```python
from tactile_toolkit.datasets import get_dataset_metadata

metadata = get_dataset_metadata("tac2pose")
print(metadata.support_level, metadata.access, metadata.limitations)
```

`open_dataset("tac2pose")` raises `DatasetUnavailableError` with the official references. A local adapter should be added only after a publisher-controlled sample manifest and the data terms are available; until then, guessing the image/pose/mesh schema would make support less reliable, not more.

## MIT GelSight research datasets

“MIT GelSight datasets” is represented as the `mit-gelsight-datasets` umbrella and these study-specific child records:

| Dataset ID | Verified contents and access | Runtime support |
| --- | --- | --- |
| `mit-gelsight-hardness` | Public index listing AVI sequences, calibration assets, about 30.7 GB of named subsets, and filename-encoded shape and Shore 00 hardness; natural-object sequences have no ground-truth hardness | C2 metadata only because the page states no reusable data terms |
| `mit-gelsight-force-shear-slip` | Marker-motion experiments relating deformation to contact-force type, shear, torsion, and incipient slip; the official page links papers and demonstrations | C2 metadata only because no data package or data license is linked |
| `mit-gelsight-neural-slip` | Robot-collected GelSight video sequences described as LSTM input for slip classification | C2 metadata only because no data package, manifest, size, checksums, or terms are linked |

These records prevent the hardness, force/shear, and neural slip studies from being combined under Tac2Pose or presented as one consistent archive. A publication license, a public web page, or downloadable bytes alone is not treated as permission to redistribute or an excuse to invent a shared loader.

## Touch100k

The primary references are the [project page](https://cocacola-lab.github.io/Touch100k/), [paper](https://arxiv.org/abs/2406.03813), [official code](https://github.com/cocacola-lab/TLV-Link), and [publisher's Google Drive](https://drive.google.com/drive/folders/1QOvbkIZtpJpz4Ry_Zg3ouXX-zLJNqu9m?usp=sharing). The paper reports 100,147 final examples sourced from Touch and Go and VisGel, but on 2026-09-06 the Drive folder described itself as partially uploaded and its `data_list.json` exposed 30 records, so no API reports an assumed full-corpus count.

The official loader schema at code revision `839dec7b1e1745c9ced0436d48c8d14cc4df8ec2` expects this local structure:

```text
touch100k/
├── data_list.json
├── touch/
│   └── <img>.jpg
└── vision/
    └── <img>.jpg
```

Despite its `.json` suffix, `data_list.json` contains one JSON object per line. Each row must provide the same `img` filename in both directories plus non-empty `sentence_desc` and `phrase_desc` strings.

Obtain the release manually so the publisher's current notices and CC BY-NC 4.0 terms remain visible, then load it without decoding all images:

```python
from tactile_toolkit.datasets import open_dataset

dataset = open_dataset("touch100k", root="datasets/touch100k")
report = dataset.validate(split="train", limit=32, check_assets=True)
report.raise_if_invalid()
sample = next(dataset.iter_samples(split="train"))
print(sample.sample_id, sample.observations["language.phrase"].data)
```

Enumeration validates bounded UTF-8 JSON lines, safe relative JPEG identifiers, unique image IDs, both paired files, symlink containment, and non-empty descriptions. Touch and scene images remain `AssetReference` values, the two descriptions remain separate language observations, unrecognized manifest fields are preserved in `labels["manifest_extra"]`, and no large data are downloaded implicitly.

The manifest publishes no explicit source-dataset or object ID per row. The adapter therefore retains the filename as the stable sample ID and conservatively groups names at the prefix before `__` when present, recording that fallback policy in sample metadata rather than presenting it as a publisher-defined split.

Run the bounded example:

```bash
python examples/datasets/touch100k_local.py datasets/touch100k --limit 2
```

## Support boundaries

- The toolkit package remains Apache-2.0, while Touch100k data remain CC BY-NC 4.0 and may not be presented as commercially reusable.
- The pinned code commit documents the loader schema only; it is not an immutable revision or checksum for the mutable Google Drive data.
- No checksums, total released bytes, or canonical validation/test splits are published for Touch100k, and the adapter does not fabricate them.
- Generated descriptions can contain model errors despite filtering and manual correction, so applications should retain annotation provenance and audit task-specific bias.
- Tac2Pose and the MIT study records become loadable only when official files, native schemas, and applicable terms are verified.
