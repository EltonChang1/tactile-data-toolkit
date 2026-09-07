# Tactile dataset source catalog

This catalog is the evidence base for dataset integrations in Tactile Data Toolkit. It was verified against primary sources on 2026-09-06; mutable statistics such as repository stars, downloads, and hosted byte counts are snapshots rather than guarantees.

The terms below distinguish public access from permission to redistribute. A public download without a clear data license is recorded as `terms unclear`, and a code license is never assumed to cover the associated data.

## Support vocabulary

| Level | Meaning |
| --- | --- |
| Planned adapter | The source exposes a stable-enough format and a lawful access route for a typed, lazy loader or converter. |
| Planned local adapter | The format can be supported, but users must obtain the data themselves because access is gated, noncommercial, or otherwise unsuitable for automatic redistribution. |
| Catalog pending terms | The source is useful and publicly described, but access or data-use terms must be clarified before an acquisition integration is responsible. |
| Reference only | The source is an index rather than a dataset and will be exposed only as a discovery resource. |

No planned integration implies that this Apache-2.0 repository relicenses third-party data. Dataset-specific terms, citations, access gates, and provenance must travel with every imported record.

## Coverage summary

| Source | Verified scale and scope | Data rights and access | Toolkit support |
| --- | --- | --- | --- |
| FreeTacMan | More than 3 million paired wrist/tactile frames and more than 10,000 trajectories over 50 described manipulation tasks | MIT-licensed Hugging Face data; 50.3 GB hosted snapshot with 46 task directories | C3 local lazy adapter |
| FoTa / FoundationTactile | 3,083,452 images from 13 camera-based tactile sensors and 11 tasks | MIT-licensed Hugging Face data; 397 GB hosted snapshot | C3 streaming local adapter |
| Tac2Pose | GelSlim 3.0 observations, object meshes, and ground-truth poses for 20 objects | Paper names a project site, but an active manifest and data license could not be confirmed | C2 discoverable metadata; no loader |
| MIT GelSight research datasets | Separate hardness, force/shear, and neural slip resources rather than one unified dataset | Hardness files are public but unlicensed; the other study pages expose no data packages | C2 umbrella and child metadata; no loaders |
| Touch100k | Paper reports 100,147 aligned touch, vision, and multi-granularity language examples | CC BY-NC 4.0 Google Drive release is currently labeled partial and has no stable revision/checksums | C3 local lazy adapter for released records |
| Touch-Vision-Language / TVL | 43,741 in-contact touch/vision pairs with open-vocabulary language labels | Original 199.3 GB and corrected 75.3 GB Hugging Face snapshots; dataset terms are not stated | C2 discoverable metadata; no loader |
| Meta DIGIT benchmarks / TacBench | Six downstream tasks around pretrained tactile representations, with three in-house DIGIT/GelSight Mini releases | In-house data are CC BY-NC 4.0; linked components retain upstream terms; native pickle is unsafe | C2 umbrella plus C3 guarded force/slip and pose adapters |
| TacVerse | 106,800 images from seven vision-based tactile sensors across shape, grating, and force tasks | CC BY 4.0 gated 29.4 GB Hugging Face snapshot | C3 local lazy adapter and pinned acquisition |
| OAHD | An umbrella of household haptic studies with thermal, force, vibration, and robot-state signals | Public study pages and downloads; no collection-wide data license found | C2 discoverable umbrella; no loader |
| CLAMP | 12.3 million multimodal datapoints from 5,357 objects, 25,100 trials, 16 devices, and 41 participants | CC BY 4.0 Harvard Dataverse data; 6.39 GB release | C3 guarded filtered-NPZ adapter |
| LMT Haptic Texture Database | Versioned releases covering 69, 108, or 184 materials with controlled/freehand texture measurements | Public archive index; no substantive reuse terms, checksum set, or verified manifest | C2 umbrella and release-specific discovery records |
| Penn Haptic Texture Toolkit / HaTT | 100 texture models, recorded motion/force/vibration data, and surface images | Attributed noncommercial research terms; user-provided local copy | C3 bounded recorded-data XML adapter |
| Multimodal Tactile Texture Dataset | Pressure/barometer and IMU streams for 12 textures at artifact-verified 30/35/40 mm/s | CC BY 4.0; 3.83 GB Mendeley V1 ZIP with publisher SHA-256; native pickle is unsafe | C3 confirmed acquisition and guarded paired adapter |
| Awesome-Touch | A maintained bibliography and index of tactile sensors, datasets, simulators, and software | MIT-licensed repository; not a data collection | Reference only |

## Verified records

### FreeTacMan (OpenDriveLab)

- **Canonical sources:** [project page](https://opendrivelab.com/FreeTacMan), [dataset repository](https://huggingface.co/datasets/OpenDriveLab/FreeTacMan), [official code](https://github.com/OpenDriveLab/FreeTacMan), and [paper](https://arxiv.org/abs/2506.01941).
- **Citation:** *FreeTacMan: Robot-free Visuo-Tactile Data Collection System for Contact-rich Manipulation* (arXiv:2506.01941; accepted to ICRA 2026 according to the project page).
- **Provenance and contents:** OpenDriveLab reports more than 3 million synchronized wrist-camera/tactile-camera image pairs and more than 10,000 trajectories across 50 contact-rich manipulation tasks. Each task directory contains MP4 video and timestamped trajectories with tool-center-point position, Euler angles, quaternion, and gripper distance.
- **Sensors, modalities, and labels:** The rig records a wrist camera, a vision-based tactile sensor, and robot-independent pose/gripper state. Labels are task identity and time-aligned end-effector state rather than a single classification target.
- **Packaging, revision, and size:** The Hugging Face repository is approximately 50.3 GB and presents task directories with MP4 and trajectory files; the verified snapshot was revision `030316fb41d6fa1e58cccb4bbe0f6fddbb932671`. The pinned snapshot currently exposes 46 task directories even though the paper and card describe 50 tasks, camera filename capitalization varies, and its mutable usage counter showed 9,560 downloads in the preceding month at verification time.
- **Rights and constraints:** The Hugging Face dataset declares MIT; the official code repository declares Apache-2.0. Tests must use synthetic metadata/video fixtures and must not fetch the complete hosted snapshot.
- **Implemented support:** The revision-pinned Hugging Face source requires explicit confirmation before downloading, and a local adapter lazily exposes camera MP4 references while parsing and validating the complete TCP/gripper CSV trajectory. This is C3 loadable support rather than conversion: it preserves missing third views and configurable camera roles without fabricating force, contact, calibration, or a 50-task manifest.

### FoTa / FoundationTactile

- **Canonical sources:** [FoundationTactile on Hugging Face](https://huggingface.co/datasets/alanz-mit/FoundationTactile), [paper](https://arxiv.org/abs/2406.13640), and [official T3 code](https://github.com/alanzjl/t3).
- **Citation:** *T3: Transferable Tactile Transformers* (arXiv:2406.13640).
- **Provenance and contents:** The release aggregates 3,083,452 tactile images from 13 camera-based tactile sensors spanning 11 tasks. It normalizes previously separate sources into a common task-oriented corpus for representation learning rather than claiming all examples were collected under one protocol.
- **Sensors, modalities, and labels:** RGB tactile images come from 13 optical tactile sensor designs; JSON sidecars carry task-specific labels. Sensor and source-dataset identity must be retained because optics, geometry, sampling, and label semantics differ materially.
- **Packaging, revision, and size:** The hosted data payload is reported as approximately 397 GB and contains WebDataset-style split TAR shards after reassembly, each sample pairing a JPEG image with a task-specific JSON object; train/validation count files are also published. The verified Hugging Face revision was `e1a16123575eb26e789cf0129ced6f3ba081f2ed`, but the hosted data are 24 split ZIP volumes plus the final ZIP segment and the viewer cannot render them, so all volumes must be reassembled before shard-level access and adapters cannot depend on viewer-generated Parquet.
- **Rights and constraints:** The dataset card declares MIT. Its usage counter showed 2,124 downloads in the preceding month at verification time, and tests must select tiny explicit shards or local fixtures rather than downloading the corpus.
- **Implemented support:** The local adapter discovers both official `data-*.tar` output names and the dataset-card `data_*.tar` spelling, streams sequential JPEG/JSON pairs without image decoding or extraction, and preserves arbitrary task labels plus source/sensor/split/member provenance. This is C3 loadable support rather than conversion because the aggregate's labels are intentionally heterogeneous and the current schema cannot represent every constituent task without coercion.

### Tac2Pose and MIT GelSight research datasets

- **Canonical sources:** [Tac2Pose paper](https://arxiv.org/abs/2204.11701), [journal DOI](https://doi.org/10.1177/02783649231196925), [MIT project URL named by the paper](http://mcube.mit.edu/research/tac2pose.html), [MIT hardness-estimation resource](https://people.csail.mit.edu/yuan_wz/hardness-estimation.htm), [MIT force/shear/slip resource](https://people.csail.mit.edu/yuan_wz/force-shear-and-slip.htm), and [MIT CSAIL slip project](https://www.csail.mit.edu/research/slip-detection-deep-neural-network-using-gelsight-touch-sensor).
- **Citation:** *Tac2Pose: Tactile Object Pose Estimation from the First Touch* (arXiv:2204.11701; International Journal of Robotics Research, DOI 10.1177/02783649231196925).
- **Provenance and contents:** Tac2Pose collected labeled GelSlim 3.0 observations on 20 objects using an ABB YuMi and two WSG-32 grippers, pairing tactile images with ground-truth object pose and object meshes/contact renderings. The paper also describes 10,000 paired touches used to calibrate a learned image-to-contact-mask model.
- **Sensors, modalities, and labels:** GelSlim 3.0 records compressed 470-by-470 RGB at 90 Hz; the learning pipeline uses resized 160-by-120 tactile input, binary contact masks, object geometry, and SE(3) pose labels. Separate MIT GelSight study pages describe hardness sequences and force, shear, or slip experiments, but they do not form one consistently packaged “GelSight dataset.”
- **Packaging and size:** Tac2Pose's paper says real datasets are available from the project website, but the named project URL did not expose a verifiable file manifest during this audit. The MIT hardness index does expose AVI data subsets totaling approximately 30.7 GB, calibration files, and a filename convention encoding sample shape and Shore 00 hardness, while the force/shear and neural slip pages expose no data package.
- **Rights and constraints:** The journal article is CC BY-NC 4.0, but that publication license is not evidence of the Tac2Pose data license. The MIT study pages state no reusable dataset terms, so public hardness downloads do not justify redistribution or a loader in this toolkit.
- **Implemented support:** Tac2Pose, a non-loadable MIT GelSight umbrella, and three study-specific child records are registered as C2 discoverable metadata with primary references, known formats, access facts, citations, and explicit unknown permissions. A user-supplied Tac2Pose or GelSight study adapter remains deferred until the publisher exposes both a verifiable native layout and applicable data terms.

### Touch100k

- **Canonical sources:** [project page](https://cocacola-lab.github.io/Touch100k/), [paper](https://arxiv.org/abs/2406.03813), [official TLV-Link code](https://github.com/cocacola-lab/TLV-Link), and the [Google Drive release linked by the project](https://drive.google.com/drive/folders/1QOvbkIZtpJpz4Ry_Zg3ouXX-zLJNqu9m?usp=sharing).
- **Citation:** *Touch100k: A Large-Scale Touch-Language-Vision Dataset for Touch-Centric Multimodal Representation* (arXiv:2406.03813).
- **Provenance and contents:** Touch100k aligns roughly 100,000 GelSight touch images with scene/object vision and language. It provides sentence-level descriptions for rich semantics and phrase-level descriptions for concise tactile attributes, supporting tasks such as material-property identification and grasp prediction.
- **Sensors, modalities, and labels:** The tactile modality is GelSight imagery; paired modalities are conventional RGB vision and multi-granularity language. Official loader code references a `dataset/train/touch100k/{touch,vision}` layout and JSONL rows containing image identifiers plus `sentence_desc` and `phrase_desc` fields.
- **Packaging and size:** The official project distributes files through Google Drive rather than a versioned dataset hub, using `data_list.json` beside `touch/` and `vision/` JPEG directories. On 2026-09-06 the Drive folder described itself as partially uploaded and the manifest exposed 30 rows, so the toolkit distinguishes those currently released files from the paper's 100,147 final examples; a complete byte count, checksums, and immutable data revision are not published.
- **Rights and constraints:** The official repository identifies the dataset as CC BY-NC 4.0 and the code as MIT; pretrained models are restricted to research use. The toolkit must not mirror the dataset or make noncommercial data look Apache-licensed.
- **Implemented support:** A C3 local adapter follows the official loader schema pinned at code commit `839dec7b1e1745c9ced0436d48c8d14cc4df8ec2`, bounds JSON-lines parsing, validates safe unique JPEG identifiers and both paired files, and exposes images lazily with separate sentence/phrase language observations. Acquisition remains manual, the pinned code commit is not misrepresented as a data revision, unknown manifest fields are preserved, and the filename-prefix grouping fallback is disclosed rather than presented as a canonical split.

### Touch-Vision-Language / TVL

- **Canonical sources:** [project page](https://tactile-vlm.github.io/), [ICML paper](https://openreview.net/forum?id=tFEOOH9eH0), [official code](https://github.com/Max-Fu/tvl), [original Hugging Face release](https://huggingface.co/datasets/mlfu7/Touch-Vision-Language-Dataset), and [corrected community-hosted revision](https://huggingface.co/datasets/yoorhim/TVL-revise).
- **Citation:** *A Touch, Vision, and Language Dataset for Multimodal Alignment* (ICML 2024, OpenReview tFEOOH9eH0; arXiv:2402.13232).
- **Provenance and contents:** TVL contains 43,741 in-contact image/touch pairs with open-vocabulary language labels, combining 4,587 robot-collected SSVTP pairs and 39,154 human-collected HCT pairs. Its labels include human annotations and vision-language-model-generated descriptions, so annotation provenance is a first-class field.
- **Sensors, modalities, and labels:** Touch comes from Meta DIGIT and vision comes from a Logitech BRIO camera. Samples combine tactile RGB, conventional RGB, contact status, free-form language, collection split, and label-source metadata.
- **Packaging, revision, and size:** Official instructions use Git LFS and multipart ZIP archives, while the original Hugging Face repository also mixes alignment data with instruction-tuning JSON. The verified original revision was `3324b2ee956b8d94974bdd8272dce0ade292cb56` with 199,349,451,898 hosted bytes and the corrected revision was `aab39e037d8916717dc98add89d58cbd39219770` with 75,292,565,517 hosted bytes; the original hosted layout has a reported tactile/image folder swap and its viewer fails on heterogeneous JSON schemas.
- **Rights and constraints:** The official code repository is Apache-2.0, but neither the original dataset card nor project page clearly grants a data license. Until the authors clarify terms, support must require a user-provided copy and preserve source-level provenance.
- **Implemented support:** The C2 runtime record pins both hosted revisions, identifies the corrected community copy linked by the official README, records the folder swap, and exposes official access links without downloading. A loader was deliberately withheld because a public download and an Apache-2.0 code license do not establish permission to process or redistribute the dataset.

### Meta DIGIT benchmarks / Sparsh TacBench

- **Canonical sources:** [Meta research page](https://ai.meta.com/research/publications/sparsh-self-supervised-touch-representations-for-vision-based-tactile-sensing/), [official Sparsh repository](https://github.com/facebookresearch/sparsh), [Hugging Face collection](https://huggingface.co/collections/facebook/sparsh), and [paper](https://arxiv.org/abs/2410.24090).
- **Citation:** *Sparsh: Self-supervised Touch Representations for Vision-based Tactile Sensing* (CoRL 2024; arXiv:2410.24090).
- **Provenance and contents:** Sparsh pretrains representations on more than 460,000 unlabeled tactile images from DIGIT, GelSight 2017, and GelSight Mini, then evaluates them with the six-task TacBench suite. The official release includes in-house force and slip data for DIGIT and GelSight Mini plus DIGIT pose data, while grasp, textile, and pretraining subsets retain links to external upstream datasets.
- **Sensors, modalities, and labels:** Published force/slip records use binarized tactile images and trajectory dictionaries with force or slip labels; pose records use short DIGIT image sequences, robot joint/pose state, relative transforms, and reference-background images. Because TacBench composes several independently collected sources, each sample needs sensor, task, and upstream-dataset identity rather than a generic `DIGIT benchmark` label.
- **Packaging and size:** The in-house releases are pinned separately: DIGIT force/slip is 10,803,966,994 bytes at `408e5b82e01b1cc471c8e747b686303afbababdc`, GelSight Mini force/slip is 52,808,308,327 bytes at `136db55485b6501605b1d2648ce0fe44740e302e`, and DIGIT pose is 18,704,238,281 bytes at `91faf6c077dbcd8dd8a5c03eb309c32cc4a63c66`. The task files are Python pickle objects, and there is no single license or byte count that supersedes linked external components.
- **Rights and constraints:** The Sparsh repository uses CC BY-NC 4.0, and linked upstream datasets may impose additional terms. Untrusted pickle files must never be deserialized without an explicit trust boundary and a safer conversion path.
- **Implemented support:** A C2 umbrella keeps the six-task and upstream-source boundary visible, while C3 child adapters cover the in-house force/slip and pose data. They refuse pickle by default, bound individual file sizes, decode one tactile frame at a time after explicit trust, validate force/slip/pose shapes and indices, preserve trajectory or bag groups, and never bundle external task datasets.

### TacVerse

- **Canonical sources:** [project page](https://lannwei.github.io/Tactile_Database/), [paper](https://arxiv.org/abs/2606.25877), [official code](https://github.com/LannWei/Tactile_Database), and [Hugging Face dataset](https://huggingface.co/datasets/Lan-2025/Tactile).
- **Citation:** *TacVerse: A Multi-Sensor Dataset and Benchmark for Cross-Sensor Vision-Based Tactile Perception* (arXiv:2606.25877, 2026 preprint).
- **Provenance and contents:** TacVerse contains 106,800 tactile images from GelSightNoMarker, GelSightMarker, MagicGripper, MagicTac, TacTip, ViTac, and ViTacTip. It reports 30,094 images across nine shape classes, 40,509 images across 30 grating classes, and 36,197 samples for three-axis force regression, with within-sensor, zero-shot cross-sensor, and few-shot adaptation protocols.
- **Sensors, modalities, and labels:** All seven sources are vision-based tactile sensors, but their optics and morphology differ. Labels cover shape class, grating class, or `(Fx, Fy, Fz)` depending on task; the official benchmark scripts derive ordered and protocol-specific splits from filenames and directory identity rather than reading shipped per-sample split metadata.
- **Packaging, revision, and size:** The Hugging Face repository reports 29,404,983,381 hosted bytes and contains four large ZIP archives, including separate force, grating, and shape archives; the verified revision was `0bc27afe0d8f6c878b79e1eb0825255541ccaceb`. Access is gated behind acceptance of the publisher's conditions and sharing contact information, and its usage counter showed 26 downloads in the preceding month at verification time.
- **Rights and constraints:** The dataset declares CC BY 4.0. The project is a recent preprint and its web citation block was still incomplete during verification, so consumers should pin a dataset revision and cite the arXiv record rather than copying an unfinished BibTeX block.
- **Implemented support:** A revision-pinned Hugging Face source requires explicit large-download confirmation and supports authenticated access, while a C3 local adapter lazily exposes extracted JPEG assets and validates force CSVs, shape directories, grating filenames, sensors, tasks, and grating groups. The official benchmark code derives ordered or protocol-specific splits rather than consuming shipped per-sample split fields, so the adapter rejects `split=` and does not claim C5 benchmark readiness.

### Open Access Haptic Database / OAHD

- **Canonical sources:** [OAHD home](https://www.oahd.gatech.edu/), [multimodal releases](https://www.oahd.gatech.edu/multi-data/), [sensor descriptions](https://www.oahd.gatech.edu/sensors/), [Georgia Tech Healthcare Robotics Lab releases](https://sites.gatech.edu/hrl/releases/), and the [Cornell EmPRISE resource index](https://emprise.cs.cornell.edu/resources/).
- **Citation:** OAHD asks publications to cite the paper associated with each downloaded study rather than one umbrella paper; study-level citations are linked from its release pages.
- **Provenance and contents:** OAHD is a Georgia Tech Healthcare Robotics Lab collection maintained by Tapomayukh Bhattacharjee, not a Cornell EmPRISE dataset. Public study pages include active/passive thermal, force, vibration/contact-microphone or accelerometer signals from handheld and robot experiments, including one real-home pushing release spanning 47 objects and 1,340 episodes.
- **Sensors, modalities, and labels:** Depending on the study, sensors measure active and passive temperature, load-cell force, contact acoustics/vibration, accelerometry, robot position, or kinematics. Object material, contact condition, household scene, and study protocol vary and must not be collapsed into one uniform label set.
- **Packaging and size:** OAHD links per-study data, load scripts, visualization scripts, and sensor/hardware designs rather than a single versioned archive. Collection-wide size, file format, checksums, and immutable revisions are not published consistently, and the site's redirect to its `www` hostname presented a certificate-name mismatch to automated clients during verification.
- **Rights and constraints:** The pages advertise open access, but this audit found no collection-wide data license or reusable terms that can safely be inferred for every release. Individual files therefore remain catalog-only until their accompanying license is captured.
- **Implemented support:** A C2 umbrella runtime record exposes the Georgia Tech provenance, official study and sensor pages, modality/sensor scope, and terms limitation without pretending OAHD is one dataset. Child records and adapters remain deferred until a study's specific paper, native schema, data license, version, and integrity information are independently qualified.

### CLAMP

- **Canonical sources:** [EmPRISE project page](https://emprise.cs.cornell.edu/clamp/), [Harvard Dataverse release](https://doi.org/10.7910/DVN/HNS2Z4), [official code](https://github.com/empriselab/CLAMP), and [paper](https://arxiv.org/abs/2505.21495).
- **Citation:** *CLAMP: Crowdsourcing a LArge-scale in-the-wild haptic dataset with an open-source device for Multimodal robot Perception* (arXiv:2505.21495, revised 2026).
- **Provenance and contents:** CLAMP reports 12.3 million datapoints over 5,357 household objects, 25,100 object trials, 16 devices, and 41 participants/households. It combines haptics with vision and speech/language annotations for material and compliance recognition and publishes raw plus filtered training-ready releases.
- **Sensors, modalities, and labels:** Modalities include active and passive thermal signals, force, contact-microphone vibration, IMU/proprioception, RGB vision, and spoken or textual descriptors. Device, participant/household, object, trial, material, compliance, and filtering provenance are meaningful grouping variables.
- **Packaging, revision, and size:** Harvard Dataverse release 1.0 was published on 2026-01-04 with 30 files totaling 6,385,413,968 bytes. The recommended `CLAMP_dataset_filtered.npz` is datafile `13302010`, contains 525,694,981 bytes, and has publisher MD5 `a5de0bbce85b5cf69db73e13029913c7`; the official code at `9b23a9ffddaa628ee1bb99cefdbf728bf2d016bf` loads its object arrays with `allow_pickle=True`.
- **Rights and constraints:** Harvard Dataverse metadata declares CC BY 4.0 for the dataset, while the code repository is BSD-3-Clause. Participant-related fields require documentation and thoughtful defaults even when the public release is licensed.
- **Implemented support:** The C3 adapter requires explicit trust before opening the pickle-backed filtered NPZ, bounds archive size, validates all official keys/features/signals, and emits one contact sequence at a time with Dataverse provenance, material/vision labels, and complete object lookup groups. It intentionally does not invent a fixed split because the official training code generates seeded object- or material-level indices, and raw ZIP support remains deferred.

### LMT Haptic Texture Database

- **Canonical sources:** [Technical University of Munich dataset page](https://www.ce.cit.tum.de/en/lmt/forschung/datensaetze/texture-database/), [current official download archive](https://zeus.lkn.ei.tum.de/downloads/texture/), [2014 paper record](https://mediatum.ub.tum.de/node?change_language=en&id=1241270), and [2017 paper record](https://mediatum.ub.tum.de/node?change_language=en&id=1420980).
- **Citation:** Cite *A Haptic Texture Database for Tool-mediated Texture Recognition and Classification* for the original database and the release-specific paper linked by TUM; the 69-, 108-, and 184-material releases are not interchangeable.
- **Provenance and contents:** LMT provides multiple generations rather than one fixed “100+ texture” dataset: the archive lists releases for 69, 108, and 184 materials. TUM describes controlled scans using a Phantom Omni and accelerometer plus uncontrolled freehand explorations for the 69-texture dataset; other visual and perceptual fields are release-dependent rather than collection-wide guarantees.
- **Sensors, modalities, and labels:** Core signals include acceleration plus exploration force/position or motion settings, with texture/material identity and controlled/freehand trial metadata. Available visual and perceptual annotations depend on the selected release.
- **Packaging and size:** The official archive index lists 6,934 decimal MB for the 69-material v1.4 release, 3,190 decimal MB for the 108-material bundle, and 83,484 decimal MB for the 184-material bundle. It publishes no checksums or stable machine-readable schema, and the archive did not return a reliable automated response during Step 07 verification.
- **Rights and constraints:** Files are publicly listed, but the institutional page's `Licence` section exposes no substantive reuse terms. The toolkit does not redistribute, download, or parse the files and does not assume that public availability grants permission.
- **Implemented support:** The C2 `lmt-textures` umbrella and release-specific `lmt-textures-69`, `lmt-textures-108`, and `lmt-textures-184` records capture sources, decimal archive sizes, version boundaries, modalities, citations, and the terms limitation. A loader remains deferred until a selected release has clear data terms, an inspectable native manifest, and representative lawful fixtures.

### Penn Haptic Texture Toolkit / HaTT

- **Canonical sources:** [University of Pennsylvania technical report and attached license](https://repository.upenn.edu/bitstreams/960863b6-df14-4770-9068-b1b2bf6a50f1/download) and the [EmPRISE index entry](https://emprise.cs.cornell.edu/hapticdatasets/).
- **Citation:** *The Penn Haptic Texture Toolkit for Modeling, Rendering, and Evaluating Haptic Virtual Textures* (University of Pennsylvania technical report).
- **Provenance and contents:** HaTT covers 100 approximately homogeneous and isotropic textures across paper, plastic, fabric, tile, carpet, foam, metal, stone, carbon fiber, and wood. It distributes calibrated haptic texture/friction models, recorded exploration data, 1024-by-1024 surface images at approximately 15 pixels/mm, and rendering code.
- **Sensors, modalities, and labels:** Each texture has two recorded-data files, each containing ten seconds sampled at 10 kHz with three-axis acceleration, force, position/speed, and texture identity. The XML recordings and rendered-model parameters support vibration/friction playback as well as recognition research.
- **Packaging and size:** The report describes XML recordings, images, model files, and MATLAB/rendering support; the former project download site was not reliably available during this audit. A current official archive size and checksum manifest could not be verified.
- **Rights and constraints:** The attached University of Pennsylvania license permits attributed, noncommercial research use and retains Penn copyright; it is not an unrestricted open-data license. Users must obtain and use the material under those terms.
- **Implemented support:** The C3 local adapter bounds XML file size, rejects DTD/entity declarations, accepts documented nested or compound axis fields, validates finite vectors, sampling rate, units, and normal/tangential pairing, and emits one grouped sequence per recorded-data XML file. It does not mirror the archive, guess image associations, interpret model/rendering files, or invent splits.

### Multimodal Tactile Texture Dataset (Mendeley Data)

- **Canonical sources:** [Mendeley Data version 1](https://data.mendeley.com/datasets/n666tk4mw9/1) and [versioned DOI](https://doi.org/10.17632/n666tk4mw9.1).
- **Citation:** Bruno Monteiro Rocha Lima, Thiago Eustaquio Alves de Oliveira, and Vinicius Prado da Fonseca, *Multimodal Tactile Texture Dataset*, Mendeley Data, V1 (2023), DOI 10.17632/n666tk4mw9.1.
- **Provenance and contents:** The release records 12 textures explored at 30, 35, and 40 mm/s according to the immutable ZIP README and directory structure; the landing-page description instead says 30, 40, and 45 mm/s. It combines a pressure/barometer tactile channel with IMU acceleration, angular-rate, and magnetic-field columns for speed-aware texture classification.
- **Sensors, modalities, and labels:** Primary data are grouped by speed and texture under paired `full_baro` and `full_imu` folders. Texture identity, artifact-derived exploration velocity, trial identity, the `baro` column, and nine `imu_a*`, `imu_g*`, and `imu_m*` columns form the usable labels and signals, but physical units are not published.
- **Packaging and size:** Version 1 was published on 2023-08-15 as a 3,831,798,836-byte ZIP plus a 29,865-byte reader notebook, for 3,831,828,701 hosted bytes. The official versioned file API publishes ZIP SHA-256 `56468a6bc7191b46e12f09fece605d117bf29e8c678ae458b885e01a41917572`; range-based central-directory inspection found 28,808 members and 21,121,661,772 uncompressed bytes without downloading the full archive.
- **Rights and constraints:** Mendeley Data declares CC BY 4.0. Python pickle is executable serialization, so the toolkit must offer an explicit trusted-import boundary and convert validated content to a safe format before normal use.
- **Implemented support:** Explicit acquisition requires confirmation before the 3.83 GB transfer and uses resumable cache download with publisher size and SHA-256 verification. The C3 local adapter requires explicit pickle trust, bounds each file, reads only paired primary pressure/IMU trials, converts native datetime indices to relative seconds, validates paths/columns/values, and ignores derived single-axis folders to prevent duplicate samples.

### Awesome-Touch

- **Canonical source:** [official GitHub repository](https://github.com/linchangyi1/Awesome-Touch).
- **Citation:** Cite the maintained repository and, when following an entry, cite the original dataset, paper, or tool rather than Awesome-Touch alone.
- **Provenance and contents:** Awesome-Touch is a community-maintained bibliography and discovery index, not a dataset. Its sections cover tactile sensors, data collection, multimodal models, manipulation, representation learning, simulation, libraries, datasets, open-source projects, laboratories, and products.
- **Packaging and popularity:** The source is Markdown and repository metadata; at verification time GitHub reported approximately 685 stars, 50 forks, and 415 commits. These mutable signals support its value as a discovery source but do not validate every linked resource.
- **Rights and constraints:** The repository is MIT-licensed, while each linked item retains independent terms. Link presence is neither license evidence nor a guarantee that data remain available.
- **Proposed support:** Expose it as a reference-only source, maintain a curated subset of canonical links, and use automated link/provenance checks without presenting it as loadable tactile data.

## Corrections to the initial source brief

1. OAHD is maintained by Georgia Tech's Healthcare Robotics Lab; Cornell EmPRISE links to OAHD and separately publishes CLAMP.
2. Tac2Pose is a concrete GelSlim pose dataset, but “MIT GelSight datasets” is not one canonical collection with a uniform combination of meshes, force, and slip labels.
3. TacVerse includes force regression in addition to shape and grating classification, and its Hugging Face release is gated even though it is CC BY 4.0.
4. TVL contains 43,741 in-contact pairs, not merely “over 43,000,” and its original hosted layout has a known image/tactile-folder correction that adapters must detect.
5. LMT has distinct 69-, 108-, and 184-material releases; claims about “100+ surfaces” must name the selected version and available modalities.
6. Penn HaTT and Touch100k are available only under noncommercial terms, while TVL, OAHD, Tac2Pose, and LMT do not expose sufficiently clear collection-level data terms for automatic acquisition.
7. The Mendeley landing page says 30, 40, and 45 mm/s, but its immutable V1 ZIP README and path names say 30, 35, and 40 mm/s; toolkit metadata follows the artifact and preserves the discrepancy.
8. Awesome-Touch is a discovery index, not a tactile dataset, so its integration must remain reference-only.

## Integration acceptance checklist

Before moving any record from this catalog into the runtime registry, an implementation must capture:

- canonical project, paper, repository, and download URLs;
- dataset license separately from code/model licenses and any access acknowledgment;
- a version, revision, DOI, or documented fallback when the publisher provides none;
- modalities, sensor model, task/label semantics, sample grouping, formats, size, and known limitations;
- lazy, resumable, or user-supplied acquisition appropriate to the source, with no large test download;
- safe archive/serialization handling, optional checksums, and clear failures for layouts the adapter does not recognize;
- provenance and citation output that survives conversion into Open-Tactile-Schema.
