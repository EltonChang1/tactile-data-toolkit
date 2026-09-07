# TVL, TacBench, TacVerse, OAHD, and CLAMP

This guide describes the Step 06 support boundary verified on 2026-09-06. Importing or opening any record performs no network access, and native pickle or NumPy object arrays are never deserialized without a per-adapter `trust_pickle=True` decision.

## Capability summary

| Record | Support | What works | Deliberate boundary |
| --- | --- | --- | --- |
| `tvl` | C2 discoverable | Exact original/corrected revisions, sizes, provenance, modalities, links, and known folder swap | No adapter or acquisition until the authors publish dataset reuse terms |
| `meta-tacbench` | C2 discoverable | Honest umbrella for all six tasks and their upstream boundaries | Third-party task sources are not relicensed or silently combined |
| `tacbench-force-slip` | C3 loadable | DIGIT/GelSight Mini batches, decoded touch frames, three force axes, contact/slip labels, trajectory groups | Explicit trusted pickle; no canonical split or invented torque axes |
| `tacbench-pose` | C3 loadable | Index/middle/ring DIGIT frames, relative 4-by-4 transforms, published train/test split, bag groups | Explicit trusted pickle; one selected finger and bag are processed at a time |
| `tacverse` | C3 loadable | Lazy force/shape/grating JPEGs, CSV validation, sensor/task labels, and grating groups | Gated acquisition; official benchmark code derives splits at runtime |
| `oahd` | C2 discoverable | Georgia Tech ownership, collection links, study modalities, sensors, and access limitations | No collection-wide terms, manifest, revision, checksum set, or uniform adapter |
| `clamp` | C3 loadable | Filtered NPZ contacts, typed feature streams, labels, official version/DOI, and object-safe groups | Explicit trusted object-array load; the 525.7 MB file is materialized once opened |

`list_datasets()` and `get_dataset_metadata()` expose all records. `open_dataset()` constructs only the three loadable adapters and never downloads their source data.

## TVL

The original official TVL snapshot is `mlfu7/Touch-Vision-Language-Dataset@3324b2ee956b8d94974bdd8272dce0ade292cb56` (199,349,451,898 hosted bytes). The official code README reports that its image and tactile directories are swapped and links the corrected community snapshot `yoorhim/TVL-revise@aab39e037d8916717dc98add89d58cbd39219770` (75,292,565,517 hosted bytes).

Neither TVL data card nor the project page states a dataset license. The toolkit therefore provides `TVL_METADATA` for evidence-backed discovery but intentionally provides neither `HuggingFaceSnapshot.download()` nor an adapter; the Apache-2.0 code license is not presumed to cover data.

## Meta DIGIT benchmarks / Sparsh TacBench

The official Sparsh collection independently publishes these in-house task sources:

| Source | Pinned Hugging Face revision | Hosted bytes |
| --- | --- | ---: |
| `facebook/digit-force-estimation` | `408e5b82e01b1cc471c8e747b686303afbababdc` | 10,803,966,994 |
| `facebook/gelsight-force-estimation` | `136db55485b6501605b1d2648ce0fe44740e302e` | 52,808,308,327 |
| `facebook/digit-pose-estimation` | `91faf6c077dbcd8dd8a5c03eb309c32cc4a63c66` | 18,704,238,281 |

Each data card declares CC BY-NC 4.0. The broader six-task TacBench also links upstream grasp, textile, and pretraining corpora, so `meta-tacbench` remains an umbrella and the two loadable records cover only Meta's in-house releases.

### Force and slip

The adapter accepts a downloaded dataset root, a probe-shape directory, or one batch directory. It loads a batch's ordered `dataset_digit_*.pkl` or `dataset_gelsight_*.pkl` frame shards plus `dataset_slip_forces.pkl`, validates every trajectory index and label length, decodes one RGB frame at a time, and uses the complete batch/trajectory identity as `group_id`.

```text
sphere/
  batch_1/
    dataset_digit_00.pkl
    dataset_digit_01.pkl
    dataset_slip_forces.pkl
```

```bash
python examples/datasets/tacbench_local.py /data/digit-force-estimation \
  --task force-slip --sensor digit --trust-pickle --limit 2
```

`force.xyz` contains the three publisher force axes in newtons and carries explicit axis metadata. It is a `wrench` modality for normalization but does not claim that absent torque axes exist, and the official release does not publish one canonical per-sample split.

### Pose

The pose adapter finds `bag_*.pkl` below the published train/test object directories. Select `finger=index`, `middle`, or `ring`; the adapter pairs `digit_<finger>` with `object_<finger>_rel_pose_n5`, validates finite 4-by-4 matrices, and keeps the complete bag as the leakage boundary.

```text
train/
  pringles/
    bag_00.pkl
test/
  sugar/
    bag_00.pkl
```

```bash
python examples/datasets/tacbench_local.py /data/digit-pose-estimation \
  --task pose --finger index --trust-pickle --limit 2
```

Python pickle is code-capable, so both adapters fail before opening a file unless `trust_pickle=True`. Inspect provenance, pin the source revision, and verify local hashes in a quarantined environment before granting that trust.

## TacVerse

Accept the publisher's gated Hugging Face access conditions, authenticate with the official client, and explicitly confirm the approximately 29.4 GB snapshot:

```python
from tactile_toolkit.datasets import TACVERSE_SOURCE

TACVERSE_SOURCE.download(
    "/data/tacverse-download",
    token=True,
    confirm_large_download=True,
)
```

Extract only the needed official ZIP archives into this shape:

```text
tacverse/
  Force_Regression/<sensor>/<sensor>.csv
  Force_Regression/<sensor>/*.jpg
  Shape_Classification/<sensor>/<shape>/*.jpg
  Grating_Classification/<sensor>/<pattern>/<size>_<group>[_<frame>].jpg
```

Then inspect any subset without decoding images or loading all records:

```bash
python examples/datasets/tacverse_local.py /data/tacverse --task shape --limit 2
```

The adapter validates local containment, JPEG suffixes, unique CSV image names, finite `(fx, fy, fz)` targets, grating filename fields, and stable task/sensor/group identities. The pinned official code creates ordered 60/20/20 shape and force splits and protocol-specific grating subsets at runtime, so `split=` is rejected rather than presenting those derived experiments as publisher-supplied labels.

## OAHD

OAHD is maintained by the Georgia Tech Healthcare Robotics Lab, while Cornell EmPRISE merely links it as a community resource. Its official pages describe separate experiments involving active/passive thermal signals, fabric force sensing, contact microphones, accelerometers, robot position, and several sensor hardware versions.

The collection provides study-specific downloads, scripts, hardware designs, and associated papers but no uniform version, packaging, checksum inventory, or collection-wide data license. `OAHD_METADATA` therefore makes the official study and sensor pages discoverable while `open_dataset("oahd")` raises an actionable no-adapter error; a future child integration must independently capture its paper and data terms.

## CLAMP

Harvard Dataverse DOI `10.7910/DVN/HNS2Z4` version 1.0 is the authoritative CC BY 4.0 release. Its 30 files total 6,385,413,968 bytes; the recommended `CLAMP_dataset_filtered.npz` is datafile `13302010`, has 525,694,981 bytes, and carries publisher MD5 `a5de0bbce85b5cf69db73e13029913c7`.

Download through Dataverse and verify the MD5 before loading:

```bash
curl -L --continue-at - \
  https://dataverse.harvard.edu/api/access/datafile/13302010 \
  --output CLAMP_dataset_filtered.npz
md5sum CLAMP_dataset_filtered.npz
```

On macOS, use `md5 CLAMP_dataset_filtered.npz` for the second command. For reproducible local processing, also record a SHA-256 because the publisher supplies only MD5.

```bash
python examples/datasets/clamp_local.py /data/CLAMP_dataset_filtered.npz \
  --trust-pickle --limit 2
```

The adapter maps the official filtered feature names to pressure, temperature, vibration, and joint-state observations and preserves material labels, native filenames, image-type provenance, label encoding, and optional vision-derived material probabilities. It validates that every contact belongs to exactly one object lookup and emits that object path as `group_id`, but it does not expose a fixed split because the official training code generates seeded object- or material-level splits.

The NPZ contains object arrays and therefore invokes pickle when opened. `trust_pickle=False` is the safe default, `max_archive_bytes` bounds the accepted container, and opening the official 525.7 MB archive may require substantially more memory after decompression.
