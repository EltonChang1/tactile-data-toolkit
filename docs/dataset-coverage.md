# Dataset architecture and compatibility matrix

This document audits the repository at Step 02 of the tactile-library expansion and translates the verified [source catalog](dataset-catalog.md) into an implementation plan. It describes real support rather than treating a documented URL, a compatible file extension, or a manually converted file as an integrated dataset.

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

All named sources are at C1 after this step; the generic CSV/NPY/NPZ reader does not raise that grade because it has no knowledge of their manifests, semantics, terms, or citations. “Target” below is the highest justified goal, not a promise that every native field maps into the current calibrated trajectory schema.

## Required-source compatibility matrix

| Source | Native unit and packaging | Normalized sample mapping | Access constraint | Current | Target and scheduled work |
| --- | --- | --- | --- | --- | --- |
| FreeTacMan | Task archives containing wrist/tactile MP4 and timestamped trajectory files | Trajectory; vision-tactile and scene vision asset references plus pose and gripper sequences, grouped by trajectory | 50.3 GB; avoid corpus download in tests | C1 | C3 in Step 04; C4 only with validated calibration or a raw-image output path |
| FoTa / FoundationTactile | WebDataset TAR shards with JPEG/JSON members | Frame; lazy vision-tactile image, sensor/source identity, task-specific labels, source group | 397 GB; stream selected shards | C1 | C3 in Step 04; C4 only for compatible tasks |
| Tac2Pose | GelSlim images, contact masks, mesh/contact rendering, and pose labels; live manifest unavailable | Frame or sequence; vision-tactile observation, pose, object/group identity, mesh reference | Data files and terms unverified | C1 | C3 local adapter in Step 05 only if files/terms can be confirmed |
| MIT GelSight study datasets | Study-specific hardness, force/shear, or slip resources | Sequence; vision-tactile plus study-specific force, slip, or material labels | No unified package or license | C1 | C2 child records in Step 05; loaders only for qualified releases |
| Touch100k | Touch/vision image directories and JSONL language manifest | Frame; lazy GelSight touch and RGB vision plus phrase/sentence language and object grouping | CC BY-NC 4.0; Google Drive; no stable revision | C1 | C3 local adapter in Step 05 |
| TVL | Git LFS/multipart images and heterogeneous JSON manifests | Frame; DIGIT touch, BRIO vision, language, contact and annotation-source labels | Data license unclear; known folder swap | C1 | C3 local adapter in Step 06 with layout detection |
| Meta DIGIT / TacBench | Pickled image trajectories and labels plus external benchmark sources | Frame or sequence; DIGIT/GelSight touch, force/slip/pose labels, sensor/task/upstream grouping | CC BY-NC 4.0 plus upstream terms; unsafe serialization | C1 | C3 guarded local adapters in Step 06 |
| TacVerse | Gated Hugging Face image folders, annotations, metadata, and fixed splits | Frame; sensor-specific touch plus shape/grating class or force target and protocol split | Gated CC BY 4.0; 24.7 GB | C1 | C5 in Step 06 where canonical protocols are complete |
| OAHD | Separate study downloads and scripts with inconsistent packaging | Sequence; thermal, pressure/force, vibration/audio, IMU/pose according to child study | Collection license unclear; current TLS redirect problem | C1 | C2 child records in Step 06; C3 only for qualified studies |
| CLAMP | Raw multimodal trials and filtered NPZ | Sequence; thermal, force, vibration, IMU, vision, language, object/participant/household groups | Large participant corpus; CC BY 4.0 | C1 | C3 in Step 06 with leakage-safe grouping; C4 only for schema-compatible fields |
| LMT Haptic Texture Database | Versioned archives with controlled/freehand signals, images, and ratings | Sequence; vibration, wrench/pose settings, surface vision, material and trial labels | Terms unclear; archive rejects automated requests | C1 | C2 release records in Step 07; loader only after terms/format confirmation |
| Penn HaTT | XML recordings, texture images, models, and rendering code | Sequence; vibration, force, position/speed, surface image, material and trial labels | Noncommercial Penn license; active archive unverified | C1 | C3 local XML adapter in Step 07 if archive can be verified |
| Mendeley Multimodal Tactile Texture | Pickle files by speed/texture plus reader notebook | Sequence; pressure and IMU observations with speed/texture labels | CC BY 4.0; pickle requires trusted conversion | C1 | C3 guarded adapter in Step 07; C4 only after raw pressure/IMU output is defined |
| Awesome-Touch | Markdown bibliography and links | No `TactileSample`; discovery records point to independently verified sources | Linked projects retain their own terms | C1 | C2 reference integration in Step 08 |

## Model-to-schema relationship

The normalized model intentionally carries more modalities than Open-Tactile-Schema 0.1. Existing converters can map timestamps, tactile RGB, calibrated contact/force, wrench, and pose into the schema; scene vision, language, audio/vibration, temperature, IMU, dataset labels, multiple clocks, and external assets require explicit adapter decisions or later schema fields.

This avoids two unsafe shortcuts: synthesizing absent force/contact values merely to satisfy required fields, and placing opaque dataset semantics into unvalidated `extra` dictionaries. Step 03 supplies typed metadata, adapter contracts, a registry, bounded validation, and explicit acquisition around this boundary, while dataset-specific conversions in Steps 04–07 will declare exactly which fields they preserve, derive, or cannot convert.
