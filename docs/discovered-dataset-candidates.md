# Discovered tactile dataset candidates

This Step 09 inventory records additional tactile and haptics datasets found through
credited primary sources; it is research input, not a claim that the toolkit supports or
endorses any candidate. The [Step 10 qualification record](discovered-dataset-shortlist.md)
applies legal, access, technical-value, and maintenance gates to these leads.

Evidence was checked on **2026-09-06 America/Los_Angeles** against official project or
institution pages, original papers, publisher repositories, and project-linked source-code
or dataset hosts. Repository stars, forks, watchers, hub downloads, and institutional download
counters are mutable discovery signals rather than quality scores, and every count below is a
dated snapshot.

## Research method and terms

- A candidate had to expose actual tactile or haptic data through a credited primary source,
  not merely appear in an unsourced list or offer code without data.
- `Declared` means the dataset or a clearly scoped release states reusable terms; `project
  scope` means terms exist but Step 10 must verify that they cover the hosted data artifacts;
  and `missing` means public download access was found without substantive data-reuse terms.
- Publicly downloadable does not mean lawfully redistributable. Missing or ambiguous terms are
  recorded as blockers, and no candidate files were downloaded or added to the package.
- Approximate effort is a discovery estimate only: `S` is a small native loader, `M` needs
  multiple layouts or guarded acquisition, and `L` involves very large data, model execution,
  legacy formats, or unusually high maintenance.
- Revisions pin repository and hub evidence where available. Mutable Google Drive, Baidu, and
  institutional files without publisher revisions or checksums are called out explicitly.

The search used [Awesome-Touch](https://github.com/linchangyi1/Awesome-Touch) and published
benchmark dependency lists as leads, then returned to the linked primary source for every
claim. Community indexes such as Open X-Embodiment Tactile were not entered as datasets because
they are discovery projects rather than one versioned tactile corpus.

## Inventory at a glance

| Candidate | Terms state | Access scale or friction | Evidence-backed adoption signal | Estimated effort |
| --- | --- | --- | --- | --- |
| YCB-Sight | Declared CC BY-SA 4.0 data | About 8.5 GB across simulated and real Google Drive releases | 32 GitHub stars; incorporated by FoTa | M |
| PHAC-2 | Declared CC BY-SA 4.0 | 450.4 MB HDF5 plus labels, direct institutional files with MD5 | Institutional record displayed 22,411 aggregate downloads | M |
| TacQuad | Declared MIT on dataset card | About 69.0 GB on Hugging Face | 733 monthly dataset downloads; AnyTouch repository has 97 stars | M/L |
| Touch in the Wild | Declared MIT on dataset card | UI reports 576 GB; API storage counter is about 1.12 TB | 1,741 monthly dataset downloads; 68 code stars | L |
| VisTouch | Declared CC BY 4.0 data | Size unstated; Baidu Netdisk only | 160 GitHub stars | M |
| Eagle Shoal Visual-Tactile Dataset | Declared CDLA-Permissive 1.0 | Repository-hosted legacy text, image, and video tree | 53 GitHub stars and 15 forks | M |
| Tactile MNIST | Project MIT; inspected variant CC BY 2.0; derived terms need review | Hugging Face dataset family; one synthetic variant is 2.84 GB | ICLR 2026 paper; 153 repository commits | M |
| Feeling of Success | Declared CC BY-NC-ND 4.0 data | 100.6 GB HDF5 ZIP; direct institutional record | Reused by Sparsh and AnyTouch | M/L |
| ObjectFolder 2.0 | Declared CC BY 4.0 project data, subject to source-asset terms | Neural object files in 100-object archive chunks | 173 GitHub stars; CVPR benchmark follow-up | L |
| ObjectFolder Real | Project scope unclear for real-data files | Ten-object archive chunks; size and hashes unstated | CVPR 2023 benchmark; reused by Sparsh and AnyTouch | M/L |
| Touch and Go | Project page says CC BY 4.0; artifact scope needs confirmation | Google Drive; size and hashes unstated | Source for Touch100k and reused by FoTa, Sparsh, and AnyTouch | M |
| YCB-Slide | MIT repository; downloaded-data scope unclear | About 4.4 GB across real and simulated Google Drive releases | 34 GitHub stars; reused by AnyTouch | M |
| SSVTP | Missing data terms | Google Drive; size and hashes unstated | RSS 2023 dataset reused by TVL and AnyTouch | S/M after terms resolve |
| VisGel | Missing data terms | About 412 GB through legacy download scripts | 78 GitHub stars; reused by FoTa, Touch100k, and AnyTouch | M/L after terms resolve |
| PhysiCLeAR / Octopi | Missing data terms | Google Drive; multiple mutable release generations | RSS 2024 dataset reused by AnyTouch; 76 code stars | S/M after terms resolve |

The PHAC-2 publisher showed 22,411 at the record-level `Dataset Metrics` counter while its
individual-file counters were much lower, so that number is preserved only as the publisher's
displayed aggregate and not interpreted as unique dataset users. Hugging Face monthly downloads
and storage values are likewise service-reported snapshots rather than lifetime adoption counts.

## Candidate evidence records

### Candidate: YCB-Sight

- **Primary sources and snapshot:** The official
  [YCB-Sight repository](https://github.com/Robo-Touch/YCB-Sight) and the paper
  [Efficient Shape Mapping through Dense Touch and Vision](https://arxiv.org/abs/2109.09884)
  were checked at repository revision `7e688d9d3579d42fc7bb44f320caeeb55e5ef695`;
  GitHub reported 32 stars, 2 forks, and 3 watchers.
- **Contents and schema:** Its simulated release lists 30 YCB objects with GelSight JPEGs,
  contact masks and height maps as NPY, pose text, camera depth arrays, and meshes; its real
  release lists six objects with GelSight and RGB images, TIFF depth, NPY point clouds, robot
  CSV, transforms, and videos.
- **Open-access and license evidence:** The repository explicitly assigns
  [CC BY-SA 4.0 to the dataset and MIT to code](https://github.com/Robo-Touch/YCB-Sight#license),
  so attribution and share-alike obligations would have to remain visible in derivatives.
- **Popularity and adoption:** The repository snapshot is modest but established, and the
  required-source [FoTa dataset](https://huggingface.co/datasets/alanz-mit/FoundationTactile) identifies
  YCB-Sight as an incorporated source dataset.
- **Relevance and overlap:** Contact geometry, calibrated poses, depth, and paired sim/real
  observations make it useful for shape reconstruction; FoTa overlaps its images but does not
  replace the richer native geometric and trajectory records.
- **Access cost and constraints:** Publisher-listed object sizes total roughly 1.9 GB simulated
  and 6.55 GB real, delivered through Google Drive without a versioned manifest or checksums.
- **Likely integration surface and effort:** `M` — two native layouts could share one lazy
  adapter while keeping meshes, arrays, images, videos, and poses external until accessed.
- **Step 10 questions:** Confirm each Drive artifact still matches the documented object lists,
  design share-alike export notices, and decide whether both real and simulated releases merit
  maintained fixtures.

### Candidate: PHAC-2

- **Primary sources and snapshot:** The canonical Max Planck Digital Library record is
  [PHAC-2, version 1.0](https://doi.org/10.17617/3.0C79KW), published in 2023 with its creators
  and the related robotic-learning citation embedded in the record.
- **Contents and schema:** It describes robot haptic exploratory data and images for 60 objects,
  labels for 25 binary haptic adjectives, a 450.4 MB HDF5 train/test file, and a small MATLAB
  adjective-scale file.
- **Open-access and license evidence:** The institutional record declares
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and publishes MD5 values
  `1817bda104b97167b93f80d920f0afa3` for the HDF5 file and
  `8212ded8c50401787c684b46a7b86860` for the MATLAB file.
- **Popularity and adoption:** The institutional page displayed 22,411 aggregate downloads at
  discovery time, subject to the counter qualification above, and PHAC-2 extends a widely used
  haptic-adjective benchmark with a durable DOI.
- **Relevance and overlap:** Its robot exploratory signals, object images, and perceptual labels
  provide a non-optical complement to the image-heavy required catalog; CLAMP is the nearest
  semantic overlap but has different acquisition hardware and material tasks.
- **Access cost and constraints:** Direct institutional files are relatively small and
  checksummed, although the HDF5 group schema needs inspection and the publisher currently uses
  an automated-access challenge on some requests.
- **Likely integration surface and effort:** `M` — a bounded HDF5 adapter plus split, object,
  adjective, and sensor metadata should be practical without redistributing the files.
- **Step 10 questions:** Verify the native HDF5 schema and split leakage units from a lawful
  local copy and determine whether the image component is inside the published HDF5 artifact.

### Candidate: TacQuad

- **Primary sources and snapshot:** TacQuad is introduced by the ICLR 2025
  [AnyTouch repository](https://github.com/GeWu-Lab/AnyTouch) and hosted in the author-linked
  [TacQuad Hugging Face repository](https://huggingface.co/datasets/xxuan01/TacQuad), checked at
  dataset revision `7b46e7836f6688788eb52e6988d563ea334f5dd0` and AnyTouch revision
  `9c43a1a6eb38d904fd767712eb9dcb2d98b8d56b`.
- **Contents and schema:** The release reports 17,524 finely aligned samples over 25 objects and
  55,082 coarsely aligned samples over 99 objects from GelSight Mini, DIGIT, DuraGel, and Tac3D,
  with tactile images, vision, text, CSV manifests, ZIP archives, and CLIP token features.
- **Open-access and license evidence:** The public, ungated Hugging Face data card declares MIT;
  a formerly referenced `ur-whitelab/TacQuad` location was inaccessible, so the author-linked
  `xxuan01` revision is the evidence source rather than silently treating both as equivalent.
- **Popularity and adoption:** Hugging Face reported 733 monthly downloads and 2 likes, while
  AnyTouch reported 97 stars, 9 forks, and 2 watchers at the snapshot.
- **Relevance and overlap:** It is valuable for sensor-invariant representation learning across
  four tactile platforms; it overlaps FoTa's multi-sensor goal but adds aligned vision-language
  supervision and Tac3D measurements.
- **Access cost and constraints:** The hub reported about 69.0 GB of storage, with compressed
  archives and manifests that should be selected lazily rather than cloned wholesale.
- **Likely integration surface and effort:** `M/L` — a manifest-driven multi-sensor adapter is
  tractable, but selective Xet/Hugging Face acquisition and archive validation need care.
- **Step 10 questions:** Confirm that the data-card MIT declaration covers every archive and
  feature file, reconcile the moved repository reference, and identify a minimal lawful fixture.

### Candidate: Touch in the Wild

- **Primary sources and snapshot:** The official
  [Touch in the Wild project](https://binghao-huang.github.io/touch_in_the_wild/),
  [dataset repository](https://huggingface.co/datasets/binghaohuang-robot/touch_in_the_wild-dataset),
  and [code](https://github.com/XinyueZhuXY/touch_in_the_wild) support the NeurIPS 2025 release;
  the dataset was checked at revision `b518ef66eb46c416a0e9970c9f35533bf99503b1`.
- **Contents and schema:** It reports more than 2.6 million vision-touch pairs from 2,700
  demonstrations, 43 tasks, and 12 environments, distributed as UMI-style Zarr/MP4 session data
  with tactile, scene video, and robot or gripper state.
- **Open-access and license evidence:** The public, ungated dataset card states MIT and the code
  repository is MIT, providing unusually clear machine-readable terms for a large manipulation
  release.
- **Popularity and adoption:** Hugging Face reported 1,741 monthly downloads and 3 likes, and
  the code repository reported 68 stars and 4 forks.
- **Relevance and overlap:** Portable in-the-wild demonstrations and UMI-style sessions broaden
  contact-rich policy data beyond controlled labs; the scale and task focus overlap FreeTacMan,
  while collection apparatus and environments are distinct.
- **Access cost and constraints:** The publisher UI reports about 576 GB while its API storage
  counter reports about 1.12 TB, so exact accounting needs reconciliation and default tests or
  examples must never clone it.
- **Likely integration surface and effort:** `L` — a lazy Zarr/session adapter is conceptually
  direct, but selective downloads, video synchronization, storage, and evolving UMI conventions
  create a substantial maintenance surface.
- **Step 10 questions:** Verify artifact-level terms and exact release size against the pinned
  tree, identify stable task/session manifests, and compare its maintenance value with existing
  FreeTacMan support.

### Candidate: VisTouch

- **Primary sources and snapshot:** The author repository
  [VisTouch](https://github.com/liangnjupt/VisTouch), linked to the 2022 IEEE Wireless
  Communications paper in its citation file, was checked at revision
  `a569c7e572a78f232e819b652e0ebb718952e476` with 160 stars, 3 forks, and 1 watcher.
- **Contents and schema:** The release describes 10,498 synchronized clips across eight material
  classes with 640-by-480 video at 30 FPS, 16 kHz audio, and RH56BF3 pressure/force readings,
  stored as AVI, WAV, CSV, and indexed metadata/split files.
- **Open-access and license evidence:** A repository-level
  [DATA_LICENSE](https://github.com/liangnjupt/VisTouch/blob/a569c7e572a78f232e819b652e0ebb718952e476/DATA_LICENSE)
  explicitly applies CC BY 4.0 to dataset files; code is separately MIT licensed.
- **Popularity and adoption:** Its 160-star snapshot is one of the stronger repository signals
  in this inventory, and the release includes predefined splits and baseline scripts.
- **Relevance and overlap:** Synchronized contact force, audio, and scene video offer material
  cues absent from most optical-only corpora; CLAMP and the Mendeley texture set overlap some
  signals but not this clip-level audiovisual combination.
- **Access cost and constraints:** Data are presently linked only through Baidu Netdisk, with no
  stated total size, checksums, or immutable release revision, creating geographic and automation
  friction.
- **Likely integration surface and effort:** `M` — the indexed file formats are accessible, but
  acquisition must remain manual until a stable global endpoint and manifest can be verified.
- **Step 10 questions:** Confirm archive size and hashes, inspect timestamp alignment in a lawful
  copy, and decide whether an adapter can be useful without automating Baidu access.

### Candidate: Eagle Shoal Visual-Tactile Dataset

- **Primary sources and snapshot:** Tsinghua RLL's official
  [Visual-Tactile Dataset repository](https://github.com/tsinghua-rll/Visual-Tactile_Dataset)
  and its [journal paper](https://doi.org/10.1177/1729881418821571) were checked at revision
  `d859ecd8cc8bf7eebd2211ce9c2d52cec56856c9`, with 53 stars, 15 forks, and 2 watchers.
- **Contents and schema:** The Eagle Shoal robot-hand release contains tactile and visual data
  for manipulation/grasp research in a legacy repository tree of text records, JPEGs, and MP4s.
- **Open-access and license evidence:** Its repository explicitly applies the
  [Community Data License Agreement Permissive 1.0](https://github.com/tsinghua-rll/Visual-Tactile_Dataset/blob/d859ecd8cc8bf7eebd2211ce9c2d52cec56856c9/license.txt)
  to the released data.
- **Popularity and adoption:** The 53-star and 15-fork snapshot plus a peer-reviewed primary
  paper show durable, if older, use rather than a newly uploaded unvalidated archive.
- **Relevance and overlap:** It adds robot-hand grasp and manipulation recordings from different
  tactile hardware; Feeling of Success is the closest grasp-success overlap but uses a Sawyer,
  GelSight pair, and a different label objective.
- **Access cost and constraints:** Data are repository-hosted, but the total logical size,
  Git-LFS behavior, split semantics, and absence of publisher checksums need verification.
- **Likely integration surface and effort:** `M` — a read-only legacy-layout adapter would need
  explicit session semantics and bounded media probing.
- **Step 10 questions:** Inventory every tracked/LFS artifact, reconstruct only publisher-defined
  groupings, and confirm that the CDLA notice covers all media directories.

### Candidate: Tactile MNIST

- **Primary sources and snapshot:** The official
  [Tactile MNIST repository](https://github.com/TimSchneider42/tactile-mnist),
  [dataset guide](https://github.com/TimSchneider42/tactile-mnist/blob/9e4e59139e9349ab361a3b9297f4815724ad6387/docs/datasets.md),
  and [ICLR 2026 paper](https://arxiv.org/abs/2506.06361) were checked at revision
  `9e4e59139e9349ab361a3b9297f4815724ad6387`.
- **Contents and schema:** The benchmark reports 13,500 synthetic 3D digits and 153,600 real
  GelSight Mini touches from 600 printed digits, with real sequence/snapshot and synthetic
  variants exposed as Hugging Face datasets in 320-by-240 and 64-by-64 forms.
- **Open-access and license evidence:** The project repository declares MIT and publishes the
  dataset family links, while the inspected synthetic 64-by-64 hub variant declares CC BY 2.0;
  Step 10 must audit every variant and the obligations inherited from MNIST and 3D assets.
- **Popularity and adoption:** The ICLR 2026 publication and active package are the principal
  signals; the repository reported 16 stars, 3 forks, 3 watchers, and 153 commits, while one
  2.84 GB synthetic hub variant reported 38 monthly downloads.
- **Relevance and overlap:** It supplies a compact active-tactile-recognition benchmark with
  real/synthetic comparisons and training environments; Tac2Pose and YCB-Sight overlap shape
  perception but not the digit exploration protocol.
- **Access cost and constraints:** The family spans multiple Hugging Face repositories and a
  Python package, so users should select a named variant rather than receive an implicit bulk
  download.
- **Likely integration surface and effort:** `M` — delegate streaming to Hugging Face or wrap
  local cached variants while normalizing digit, round, trajectory, and real/synthetic grouping.
- **Step 10 questions:** Audit every dataset card and asset license, choose whether the registry
  models a collection with child variants, and verify canonical train/test leakage boundaries.

### Candidate: Feeling of Success

- **Primary sources and snapshot:** The official
  [project page](https://sites.google.com/view/the-feeling-of-success), CoRL 2017
  [paper](https://proceedings.mlr.press/v78/calandra17a.html), and current TU Dresden
  [OPARA record](https://doi.org/10.25532/OPARA-700) replace obsolete download assumptions; the
  accompanying code was checked at revision `7bb895897e369ae9f5fcaeed61d401e019a9cdf1`.
- **Contents and schema:** It contains 9,269 grasp attempts on 106 objects collected with a
  Sawyer arm, WSG-50 gripper, Kinect 2, and two GelSight sensors, distributed primarily as a
  100.6 GB HDF5 ZIP plus a 2.52 GB example shard and notebook.
- **Open-access and license evidence:** The institutional data record declares
  [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/); the no-derivatives
  restriction requires special care around conversion or redistributed fixtures even though
  the separate code is MIT.
- **Popularity and adoption:** The dataset is a documented source in Meta's
  [Sparsh](https://github.com/facebookresearch/sparsh) and
  [AnyTouch](https://github.com/GeWu-Lab/AnyTouch) benchmarks, a stronger downstream signal than
  its code repository's 26-star snapshot alone.
- **Relevance and overlap:** Paired pre/post-contact tactile images, vision, and grasp outcomes
  directly support success prediction; the Eagle Shoal candidate overlaps grasp manipulation but
  differs in apparatus and labeling.
- **Access cost and constraints:** The authoritative OPARA deposit is large and exposes no
  checksums in the inspected metadata; the project notes that HDF5 replaced its original unsafe
  pickle representation.
- **Likely integration surface and effort:** `M/L` — local HDF5 loading is feasible, but bounded
  validation, 100 GB acquisition, and no-derivatives compliance raise the cost.
- **Step 10 questions:** Obtain an artifact manifest/checksum, inspect the HDF5 schema, and get a
  defensible interpretation of whether normalization or converted outputs are permitted.

### Candidate: ObjectFolder 2.0

- **Primary sources and snapshot:** Stanford's official
  [ObjectFolder project](https://objectfolder.stanford.edu/) and
  [repository](https://github.com/rhgao/ObjectFolder) describe the CVPR 2022 release and CVPR
  2023 benchmark; the repository was checked at revision
  `3c6cd8930b2dcbadb6d94dadf2745c956bdcd236` with 173 stars, 13 forks, and 6 watchers.
- **Contents and schema:** ObjectFolder 2.0 represents 1,000 objects as learned neural Object
  Files with VisionNet, AudioNet, and TouchNet checkpoints, NPY query inputs, renderers, and
  downloadable 100-object archive chunks.
- **Open-access and license evidence:** The repository assigns
  [CC BY 4.0 to the project](https://github.com/rhgao/ObjectFolder#license), while noting that
  meshes sourced from ABO, 3D Model Haven, YCB, and Google Scanned Objects retain their original
  licenses and must be audited per asset.
- **Popularity and adoption:** The 173-star snapshot, its CVPR benchmark follow-up, and use by
  AnyTouch provide repository, venue, and downstream-adoption signals.
- **Relevance and overlap:** Generative visual, acoustic, and tactile object representations can
  support synthetic augmentation and cross-modal retrieval; it overlaps simulated tactile
  sources but uniquely exposes queryable multisensory object models.
- **Access cost and constraints:** Stanford hosts archive chunks, but consuming the tactile data
  requires legacy PyTorch model execution and GPU-aware dependencies rather than parsing only
  static samples.
- **Likely integration surface and effort:** `L` — prefer a converter or optional model-backed
  adapter isolated from the core package, with pinned weights and query provenance.
- **Step 10 questions:** Build an object-by-object license map, verify archive sizes and hashes,
  and decide whether executable neural fields are maintainable within the toolkit's scope.

### Candidate: ObjectFolder Real

- **Primary sources and snapshot:** The official
  [ObjectFolder Real download page](https://objectfolder.stanford.edu/objectfolder-real-download)
  and CVPR 2023 [ObjectFolder Benchmark paper](https://openaccess.thecvf.com/content/CVPR2023/html/Gao_The_ObjectFolder_Benchmark_Multisensory_Learning_With_Neural_and_Real_Objects_CVPR_2023_paper.html)
  describe this separate real-object release.
- **Contents and schema:** It covers 100 household objects with GelSight RGB deformation videos
  at 30 to 50 surface points, in-hand and third-view videos, meshes, impact sounds, and force
  profiles, divided into ten-object archive chunks.
- **Open-access and license evidence:** The project calls its resources open source, but the
  inspected real-data download page does not state a separate artifact license; the ObjectFolder
  repository's CC BY 4.0 statement may cover it, but that scope is not assumed here.
- **Popularity and adoption:** It anchors the CVPR 2023 benchmark and is a named source for
  [Sparsh](https://github.com/facebookresearch/sparsh) and
  [AnyTouch](https://github.com/GeWu-Lab/AnyTouch).
- **Relevance and overlap:** Dense object-centric tactile video with vision, sound, geometry,
  and force enables cross-modal tasks not represented by one required dataset; its semantics
  overlap ObjectFolder 2.0 while replacing rendered measurements with physical captures.
- **Access cost and constraints:** Direct archive commands are published, but file sizes,
  immutable revisions, checksums, and a machine-readable manifest are absent from the page.
- **Likely integration surface and effort:** `M/L` after terms resolve — a chunk-selective local
  adapter could normalize object, contact point, view, modality, and trial grouping.
- **Step 10 questions:** Obtain explicit data-artifact terms, record hashes and sizes, and decide
  whether Real and 2.0 are one collection with distinct technical adapters.

### Candidate: Touch and Go

- **Primary sources and snapshot:** The official
  [Touch and Go project](https://touch-and-go.github.io/), NeurIPS 2022
  [Datasets and Benchmarks paper](https://openreview.net/forum?id=ZZ3FeSSPPblo), and
  [code repository](https://github.com/fredfyyang/Touch-and-Go) were checked with code revision
  `d62e871e1d11ed9be1f11409481cc2861ef4e82d` and 41 stars, 8 forks, and 2 watchers.
- **Contents and schema:** The release reports 246,000 paired scene/tactile frames from 13,900
  touches on about 3,971 objects, including raw RGB and GelSight videos, NumPy timestamps, and
  labels spanning 20 material categories.
- **Open-access and license evidence:** The project footer states CC BY 4.0 for Touch and Go, but
  the code repository has no detected license and the Drive artifacts lack per-file notices, so
  Step 10 must confirm that the footer clearly governs the dataset downloads.
- **Popularity and adoption:** It is upstream of Touch100k and appears in FoTa, Sparsh, and
  AnyTouch, giving it unusually strong reuse across the required catalog.
- **Relevance and overlap:** Uncurated indoor/outdoor object contact makes it useful for material
  and touch-vision association; much of its imagery is already represented indirectly through
  FoTa and Touch100k, so native-video value must justify duplication.
- **Access cost and constraints:** Google Drive delivery has no stated total size, version,
  checksums, or stable selective manifest.
- **Likely integration surface and effort:** `M` after scope confirmation — synchronize paired
  videos through published NumPy times and preserve object/touch groups for leakage-safe splits.
- **Step 10 questions:** Confirm artifact license scope, quantify archives, audit canonical
  splits, and measure how much unique native information remains beyond existing adapters.

### Candidate: YCB-Slide

- **Primary sources and snapshot:** Carnegie Mellon R-Pad's official
  [YCB-Slide repository](https://github.com/rpl-cmu/YCB-Slide) and the linked MidasTouch CoRL 2022
  paper were checked at revision `537b225fee3a1040933148a4ea5947fceb598ae1`, with 34 stars,
  4 forks, and 4 watchers.
- **Contents and schema:** It provides real and simulated sliding trajectories for ten YCB
  objects, with 50 trajectories in each domain, DIGIT images, object poses, RGB/webcam views,
  meshes, and simulated masks or height maps; downstream documentation reports about 180,000
  tactile frames.
- **Open-access and license evidence:** The repository is MIT licensed, but its wording covers
  software and associated documentation without clearly assigning that license to the separately
  hosted Google Drive data artifacts.
- **Popularity and adoption:** Beyond its repository snapshot, YCB-Slide is a named downstream
  training dataset in AnyTouch.
- **Relevance and overlap:** Continuous sliding, object pose, and real/simulation pairing support
  tactile localization; YCB-Sight overlaps YCB geometry but emphasizes dense shape mapping rather
  than sliding localization.
- **Access cost and constraints:** Publisher-listed artifacts total about 4.4 GB across real and
  simulated releases; download scripts use mutable Drive IDs, no hashes, and simulated pose data
  include pickle serialization.
- **Likely integration surface and effort:** `M` after terms resolve — lazy image/array loading
  plus an explicit trusted-pickle gate would resemble existing TacBench safeguards.
- **Step 10 questions:** Obtain dataset-level terms, verify Drive hashes and layouts, and assess
  whether official trajectories define an object-safe benchmark split.

### Candidate: SSVTP

- **Primary sources and snapshot:** The official Berkeley
  [SSVTP project](https://sites.google.com/berkeley.edu/ssvtp) and Robotics: Science and Systems
  2023 [paper](https://arxiv.org/abs/2209.13042) are the primary evidence sources.
- **Contents and schema:** The project reports 4,500 spatially aligned scene-vision and DIGIT
  tactile image pairs and evaluates contact/feature localization, anomaly and vision-query tasks,
  plus edge, cable, and seam following.
- **Open-access and license evidence:** A public Google Drive data link exists, but the project
  page and inspected paper materials state no substantive data license; downloadability alone is
  insufficient for toolkit integration.
- **Popularity and adoption:** SSVTP supplies a major portion of TVL and is named by AnyTouch,
  demonstrating downstream benchmark reuse without relying on an unsourced citation count.
- **Relevance and overlap:** Spatial alignment makes it useful for local cross-modal
  correspondence and robotic following; TVL already repackages much of it with language, so
  upstream spatial annotations are the principal unique value.
- **Access cost and constraints:** Size, checksums, immutable version, and file manifest are not
  stated on the official page, and Drive access is mutable.
- **Likely integration surface and effort:** `S/M` after terms resolve — paired-image loading
  should be small, but native spatial correspondences and group structure must be verified.
- **Step 10 questions:** Request or locate explicit reuse terms, compare the archive against TVL,
  and reject integration if license provenance remains absent.

### Candidate: VisGel

- **Primary sources and snapshot:** MIT CSAIL's official
  [VisGel project](https://visgel.csail.mit.edu/), CVPR 2019 paper, and
  [source repository](https://github.com/YunzhuLi/VisGel) were checked at revision
  `b502e37861b722168f0e52f3fb664b10fd869d0a`, with 78 stars, 16 forks, and 2 watchers.
- **Contents and schema:** The project reports 12,000 touches and three million aligned visual
  and GelSight frames across 195 objects, with a 10,000-touch seen-object training partition and
  a 2,000-touch unseen-object test partition.
- **Open-access and license evidence:** Download and training scripts are public, but neither the
  inspected project page nor repository states a license for the dataset or code, so reuse terms
  are unresolved.
- **Popularity and adoption:** VisGel is incorporated or derived into FoTa and Touch100k and is
  named by AnyTouch, in addition to the repository snapshot.
- **Relevance and overlap:** Large aligned scene/touch video supports cross-modal generation and
  representation learning; it substantially overlaps images already flowing through FoTa and
  Touch100k, while its native sequences and original train/test partition may remain unique.
- **Access cost and constraints:** Official scripts list about 328 GB of seen-object and 83.2 GB
  of unseen-object training data, roughly 412 GB total, through legacy endpoints without a modern
  checksummed manifest.
- **Likely integration surface and effort:** `M/L` after terms resolve — video/list parsing is
  feasible, but size, legacy scripts, split validation, and duplicate storage are costly.
- **Step 10 questions:** Obtain explicit data terms, test current endpoints, and reject a native
  adapter if the existing FoTa/Touch100k coverage captures all maintainable use cases.

### Candidate: PhysiCLeAR / Octopi

- **Primary sources and snapshot:** The NUS project's official
  [PhysiCLeAR / Octopi page](https://octopi-tactile-lvlm.github.io/), RSS 2024
  [paper](https://arxiv.org/abs/2405.02794), [Octopi code](https://github.com/clear-nus/octopi),
  and [Octopi 1.5 code](https://github.com/clear-nus/octopi-1.5) were checked at revisions
  `b7f0b0b56a5be950a10c46aa40277dfd51050eab` and
  `63291e15ea1cac26542ecbdfe58192d60f27be00`.
- **Contents and schema:** The original release describes 74 objects and 408 GelSight videos
  collected with pressing and rotating motions, annotated by three people for hardness,
  roughness, and bumpiness and used in five tactile-language tasks; the project also links an
  expanded 1.5 generation.
- **Open-access and license evidence:** Public Google Drive data and model links exist, but no
  license was detected in either code repository or on the inspected project page, so data and
  model reuse terms remain unresolved.
- **Popularity and adoption:** The original code reported 76 stars, 5 forks, and 3 watchers;
  Octopi is also named as a training source by AnyTouch.
- **Relevance and overlap:** Physical-property judgments and tactile question answering add
  language supervision distinct from pure material labels; they overlap TVL and Touch100k enough
  that version and annotation uniqueness must be demonstrated.
- **Access cost and constraints:** Multiple mutable Drive releases have no stated size,
  checksums, immutable revision, or stable machine-readable manifest.
- **Likely integration surface and effort:** `S/M` after terms resolve — video plus label loading
  should be bounded, but the original/1.5 relationship and language-task schemas need modeling.
- **Step 10 questions:** Obtain explicit reuse terms for data, models, and annotations; determine
  whether 1.5 supersedes the original; and compare unique labels with existing language adapters.

## Step 10 qualification inputs

The [Step 10 decision record](discovered-dataset-shortlist.md) applies a documented gate rather
than treating the effort labels above as a ranking.
At minimum, a shortlisted release should have unambiguous artifact-level terms, an authoritative
and reachable acquisition path, enough native schema evidence to avoid fabricated fields, a
clear benefit beyond current toolkit coverage, a leakage-safe grouping strategy, and a realistic
maintenance plan for its size and dependencies.

The unresolved candidates are deliberately retained because their adoption makes a license or
overlap decision worth documenting. If primary-source terms remain missing, access is no longer
functional, or native value is already preserved by FoTa, Touch100k, TVL, or another supported
release, Step 10 should reject or defer the candidate explicitly rather than lowering the gate.
