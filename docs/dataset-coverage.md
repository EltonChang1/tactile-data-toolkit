# Dataset architecture and compatibility matrix

This document began as the Step 02 architecture audit and now tracks implemented coverage against the verified [source catalog](dataset-catalog.md). It describes real support rather than treating a documented URL, a compatible file extension, or a manually converted file as an integrated dataset.

## Architecture audit

The existing package is a streaming converter for locally recorded robot logs. Its reader registry recognizes MCAP, ROS bag, CSV, NPY, and NPZ inputs; a modality resolver chooses one vision-tactile or taxel master stream; the pipeline aligns at most one wrench and one pose stream; calibrators produce required force/contact geometry; and Zarr, HDF5, and LeRobot writers serialize `TactileChunk` arrays under Open-Tactile-Schema 0.1.

That path remains appropriate for calibrated trajectories, but the published datasets add a distinct ingestion layer:

```text
official or user-provided dataset
        -> dataset adapter (Step 03+)
        -> TactileSample: raw/lazy observations + labels + grouping
        -> dataset-specific normalization or calibration
        -> TactileChunk: aligned schema arrays
        -> Zarr / HDF5 / LeRobot
```

| Area | Existing capability | Gap exposed by the catalog | Scoped response |
| --- | --- | --- | --- |
| Input dispatch | Local path and file-suffix reader registry | Dataset directories, archives, manifests, gated hubs, and multi-file samples cannot be identified reliably by suffix | Use the separate dataset adapter registry implemented in Step 03; do not overload the log-reader registry |
| Sample model | `Stream` and `TactileChunk` require materialized NumPy arrays with a shared time axis | Static image/label pairs, language, independent sensor clocks, external media, and task-specific labels do not fit | Introduce `TactileSample`, `TactileObservation`, and `AssetReference` before conversion |
| Modalities | Vision-tactile, taxel, wrench, pose, and joint state | RGB vision, language, vibration, thermal, pressure, IMU, and audio are present in required sources | Extend the modality vocabulary without changing existing schema requirements |
| Temporal structure | One tactile master trajectory with aligned auxiliary arrays | Corpora contain independent frames, short sequences, trials, and robot trajectories | Record `frame`, `sequence`, or `trajectory` explicitly and keep per-observation timing |
| Dataset semantics | Free-form `DatasetInfo.source`, `task`, and `extra` dictionaries | No normalized dataset identity, revision, license, citation, split, or leakage group | Use the versioned `DatasetMetadata` schema and sample/task/group fields implemented in Steps 02–03 |
| Acquisition | None | Huge, gated, noncommercial, redirected, and revisioned downloads | Use Step 03's explicit cache/resume/integrity utility only for ordinary HTTP(S); preserve official flows for gated sources |
| Multi-sensor data | One calibrated tactile master per pipeline invocation | FoTa and TacVerse require sensor identity; some trajectories contain multiple fingertips | Preserve a sensor on every observation; design multi-sensor conversion after adapters expose real fixtures |
| Validation | Required calibrated schema fields, shapes, dtypes, and dimensions | It cannot validate native manifests, paired assets, task labels, or license provenance | Keep output-schema validation unchanged and use Step 03's bounded adapter, metadata, split, path, size, and checksum validation |
| Training access | Frame/window access to toolkit-produced Zarr | No direct lazy access to native published datasets | Let adapters yield normalized samples; add training bridges only where formats and terms justify them |

### Deliberate boundaries

- `TactileSample` is an adapter-facing model, not a replacement for `TactileChunk` or Open-Tactile-Schema.
- An observation holds a NumPy array, language text, or a lazy asset reference; merely constructing a sample performs no network or large-file I/O.
- A single timestamp identifies a frame, while a timestamp vector identifies the leading time axis of a materialized sequence.
- `group_id` represents the indivisible split unit—such as object, trajectory, participant, or household—to reduce benchmark leakage.
- Dataset-wide licenses, citations, revisions, and access policy belong to the Step 03 metadata schema rather than being duplicated on every sample.
- Unknown third-party arrays are not forced into `TactileChunk`; conversion may retain raw-only fields in a future schema revision or require a documented calibration stage.

## Support-level definitions

| Grade | User-visible meaning |
| --- | --- |
| C0 — absent | The source is neither documented nor understood by the toolkit. |
| C1 — cataloged | Primary links, terms, format, constraints, and intended support are documented. |
| C2 — discoverable | Runtime metadata can locate the official source and explain lawful acquisition. |
| C3 — loadable | A tested adapter lazily yields normalized samples from an official or user-provided copy. |
| C4 — convertible | Supported source content can be validated and converted to a documented toolkit output without invented fields. |
| C5 — benchmark-ready | Canonical splits/metrics and leakage protections are validated for comparable evaluation. |

FreeTacMan and FoTa reached C3 in Step 04; Step 05 added C2 discovery for Tac2Pose and study-specific MIT GelSight resources plus C3 local loading for Touch100k. Step 06 added C2 discovery for TVL, the complete Meta TacBench umbrella, and OAHD plus C3 guarded/local loading for Meta's force/slip and pose releases, TacVerse, and CLAMP; the generic CSV/NPY/NPZ reader does not raise a grade because it has no knowledge of a published dataset's manifests, semantics, terms, or citations.

## Required-source compatibility matrix

| Source | Native unit and packaging | Normalized sample mapping | Access constraint | Current | Target and scheduled work |
| --- | --- | --- | --- | --- | --- |
| FreeTacMan | Task directories containing wrist/tactile MP4 and timestamped trajectory files | Trajectory; lazy camera assets plus validated pose and gripper arrays, grouped by demonstration | 50.3 GB; pinned snapshot currently has 46 task directories versus 50 described | C3 | Local lazy adapter implemented in Step 04; C4 only with validated calibration or a raw-image output path |
| FoTa / FoundationTactile | Multi-volume ZIP containing WebDataset TAR shards with JPEG/JSON members | Frame; lazy TAR-member image, source/sensor identity, arbitrary task labels, and source group when declared | 397 GB; all ZIP volumes precede shard access | C3 | Streaming local adapter implemented in Step 04; C4 only for compatible task-specific labels |
| Tac2Pose | Paper-described GelSlim images, contact masks, mesh/contact rendering, and pose labels; live manifest unavailable | Intended frame or sequence mapping remains documented but is not fabricated without source files | Data files, native schema, size, checksums, and data terms unverified | C2 | Discoverable metadata implemented in Step 05; C3 deferred pending files and terms |
| MIT GelSight study datasets | Study-specific hardness AVI data and separately described force/shear or neural slip resources | Intended sequence mappings stay study-specific; umbrella and child records preserve the distinction | Hardness download is unlicensed; other studies have no public data package; no unified archive/license | C2 | Umbrella plus three child records implemented in Step 05; loaders deferred pending qualified releases |
| Touch100k | `data_list.json` JSON lines plus paired `touch/` and `vision/` JPEGs | Frame; lazy GelSight touch and RGB vision plus separate phrase/sentence language, filename ID, and conservative recording-prefix group | CC BY-NC 4.0; manual Google Drive; partial mutable release; no data revision/checksums/canonical eval splits | C3 | Local lazy adapter implemented in Step 05; C4 requires a raw multimodal output path |
| TVL | Multipart ZIP images and heterogeneous JSON manifests | Intended frame mapping: DIGIT touch, BRIO vision, language, contact and annotation-source labels | No stated data license; corrected 75.3 GB snapshot; original folder swap | C2 | Pinned discovery metadata; loader deferred until reusable data terms exist |
| Meta DIGIT / TacBench | Pickled image trajectories and labels plus external benchmark sources | Frame; decoded DIGIT/GelSight Mini touch, three-axis force, slip/contact or relative pose, and trajectory/bag grouping | CC BY-NC 4.0; 82.3 GB across three in-house releases; unsafe serialization | C2 umbrella / C3 in-house children | Guarded local force/slip and pose adapters; third-party tasks remain at their upstream boundaries |
| TacVerse | Gated ZIPs containing JPEG directories and per-sensor force CSVs | Frame; lazy sensor-specific touch plus shape/grating labels or three-axis force target and grating group | Gated CC BY 4.0; 29.4 GB; official code derives experimental splits | C3 | Local adapter and confirmed pinned acquisition; benchmark-ready status deferred until protocol reproduction is tested |
| OAHD | Separate study downloads and scripts with inconsistent packaging | Intended sequence mapping: thermal, pressure/force, vibration/audio, IMU/pose according to child study | Collection license unclear; current TLS redirect problem | C2 | Honest umbrella metadata; child adapters deferred until study-specific terms and formats qualify |
| CLAMP | Raw multimodal trials and pickle-backed filtered NPZ | Sequence; pressure/force, thermal, vibration, proprioception, material/vision labels, and exact object groups | CC BY 4.0; 6.39 GB release; 525.7 MB filtered NPZ requires explicit trust | C3 | Guarded filtered-contact adapter with complete object grouping; raw archive and conversion remain deferred |
| LMT Haptic Texture Database | Versioned archives with controlled/freehand signals, images, and ratings | Sequence; vibration, wrench/pose settings, surface vision, material and trial labels | Terms unclear; archive rejects automated requests | C1 | C2 release records in Step 07; loader only after terms/format confirmation |
| Penn HaTT | XML recordings, texture images, models, and rendering code | Sequence; vibration, force, position/speed, surface image, material and trial labels | Noncommercial Penn license; active archive unverified | C1 | C3 local XML adapter in Step 07 if archive can be verified |
| Mendeley Multimodal Tactile Texture | Pickle files by speed/texture plus reader notebook | Sequence; pressure and IMU observations with speed/texture labels | CC BY 4.0; pickle requires trusted conversion | C1 | C3 guarded adapter in Step 07; C4 only after raw pressure/IMU output is defined |
| Awesome-Touch | Markdown bibliography and links | No `TactileSample`; discovery records point to independently verified sources | Linked projects retain their own terms | C1 | C2 reference integration in Step 08 |

## Model-to-schema relationship

The normalized model intentionally carries more modalities than Open-Tactile-Schema 0.1. Existing converters can map timestamps, tactile RGB, calibrated contact/force, wrench, and pose into the schema; scene vision, language, audio/vibration, temperature, IMU, dataset labels, multiple clocks, and external assets require explicit adapter decisions or later schema fields.

This avoids two unsafe shortcuts: synthesizing absent force/contact values merely to satisfy required fields, and placing opaque dataset semantics into unvalidated `extra` dictionaries. Step 03 supplies typed metadata, adapter contracts, a registry, bounded validation, and explicit acquisition around this boundary, while the dataset integrations in Steps 04–07 declare exactly which fields they preserve, derive, or cannot convert.
