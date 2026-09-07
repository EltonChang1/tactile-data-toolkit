# Step 10 discovered-dataset qualification and implementation plan

This document applies a maintainability gate to the 15 primary-source candidates in the
[Step 09 inventory](discovered-dataset-candidates.md). It selects four narrowly scoped releases
for Steps 11–12, defers five credible but currently disproportionate integrations, and rejects
six candidates from this program because their artifact terms or practical boundaries are not
clear enough.

These decisions assess toolkit integration, not scientific quality. `REJECT` means “not eligible
in the current 16-step program,” and every deferred or rejected row states what new evidence could
make a later review worthwhile.

Evidence was rechecked on **2026-09-07 America/Los_Angeles**. In addition to the sources recorded
in the inventory, this review queried the publisher APIs for immutable file trees and inspected
the publisher-provided loaders or processing scripts without downloading any large corpus.

## Qualification gates

A release enters the implementation shortlist only when all hard gates pass within the proposed
support boundary:

1. **Artifact terms:** the data artifact—not only adjacent code—has explicit terms that permit
   metadata integration and local loading as proposed.
2. **Authoritative access and integrity:** an official route is reachable and sufficiently
   versioned or bounded; missing checksums must lead to manual/local-only acquisition, never an
   invented integrity claim.
3. **Native-schema evidence:** official documentation, manifests, scripts, or viewer metadata
   describe enough structure to produce typed samples without guessing fields, units, or splits.
4. **Incremental community value:** the native release preserves a useful modality, task,
   grouping, or protocol not already available through a maintained adapter.
5. **Maintenance fit:** network-free fixtures, bounded validation, lazy access, and actionable
   errors are feasible in Steps 11–12 without adding a fragile GPU runtime or downloading a
   multi-hundred-gigabyte corpus in CI.

Terms and provenance are veto gates. Popularity is supporting evidence only: it cannot cure a
missing license, inaccessible artifact, unsafe format, or duplicative maintenance burden.

## Candidate decisions

`S11` and `S12` are qualified commitments for those steps, ordered within each tranche below;
`DEFER` retains a credible future candidate behind a concrete blocker; and `REJECT` removes the
candidate from this program unless materially new primary evidence appears.

| Candidate | Artifact terms | Access and integrity | Native schema | Incremental value / maintenance | Outcome | Decisive reason or reconsideration trigger |
| --- | --- | --- | --- | --- | --- | --- |
| YCB-Sight | Pass: CC BY-SA 4.0 data | Pass, local-only: per-object Drive files; no hashes | Pass: official real/sim trees and scripts | Native depth, pose, masks, and geometry justify moderate two-layout work | `S12` | Qualify local loading; reconsider automated acquisition only if the publisher adds immutable hashes |
| PHAC-2 | Pass: CC BY-SA 4.0 data | Pass: DOI, Dataverse API, sizes, IDs, and MD5 | Pass: publisher MATLAB script names every HDF5 group and signal | Compact non-optical BioTac/adjective data fill a major modality gap | `S11` | Highest-value manageable complement to image-heavy coverage |
| TacQuad | Pass: MIT data card | Pass with explicit size confirmation: pinned hub tree and LFS SHA-256 | Conditional pass: card plus official AnyTouch loader; four-sensor claim must be reconciled with three-sensor loader paths | Aligned multi-sensor and language data are distinct, but the multipart release is about 69 GB | `S12` | Qualify extracted local loading; C3 requires resolving the Tac3D/schema discrepancy without guessing |
| Touch in the Wild | Pass: MIT data card | Weak: UI reports 576 GB while API storage reports about 1.12 TB | Pass: project documents UMI-style Zarr/session data | Strong manipulation value but high overlap with FreeTacMan and extreme test/support cost | `DEFER` | Reconsider after a stable small official subset and exact release accounting exist |
| VisTouch | Pass: CC BY 4.0 data | Weak: Baidu-only archive, unknown size and hashes | Pass: indexed AVI/WAV/CSV layout | Unique audio/force/video signals, but globally reproducible acquisition is not yet maintainable | `DEFER` | Reconsider when an institution-backed global mirror or checksummed manifest appears |
| Eagle Shoal Visual-Tactile Dataset | Pass: CDLA-Permissive 1.0 | Weak: legacy Git repository is about 39.9 GB with no release manifest or checksums | Partial: file types documented, session semantics sparse | Non-GelSight grasp data are useful, but cloning a monolithic legacy repository is disproportionate | `DEFER` | Reconsider after a versioned archive, manifest, and session schema are published |
| Tactile MNIST | Pass for selected child: CC BY 2.0 | Pass: immutable two-Parquet release with LFS SHA-256 | Pass: publisher guide and dataset-viewer feature schema | A 132.7 MB real GelSight Mini benchmark has canonical splits and an active ecosystem | `S11` | Qualify only `touch-real-single-t256-64x64`; the wider family remains separately reviewable |
| Feeling of Success | Pass but restrictive: CC BY-NC-ND 4.0 | Partial: authoritative 100.6 GB HDF5 ZIP, no published digest located | Pass: current HDF5 release and project schema | Grasp outcomes are valuable, but no-derivatives terms make normalized exports and fixtures risky | `DEFER` | Reconsider after publisher clarification that the intended adapter/conversion boundary is permitted |
| ObjectFolder 2.0 | Conditional: CC BY 4.0 project, source assets retain independent terms | Partial: official chunks without a complete checksummed asset/license manifest | Pass for model queries, not ordinary static samples | Distinct simulation value, but GPU/model runtime and per-asset licensing exceed core maintenance scope | `DEFER` | Reconsider with a static tactile export and machine-readable per-object license map |
| ObjectFolder Real | Fail: real-data artifact scope is not explicit | Partial: official chunks lack sizes, revisions, and hashes | Pass: benchmark paper documents modalities | High value and adoption cannot override uncertain artifact terms | `REJECT` | Re-review only after explicit real-data terms and a publisher manifest are posted |
| Touch and Go | Fail: project footer scope over Drive artifacts remains ambiguous | Weak: mutable Drive files lack size, revision, and hashes | Pass: paper and code document paired video/timestamps | Images substantially overlap supported FoTa and Touch100k data | `REJECT` | Re-review if artifact-level terms and a versioned native-video manifest make unique value measurable |
| YCB-Slide | Fail: MIT text is scoped to software/documentation, not Drive data | Weak: mutable Drive files lack hashes; pose data use pickle | Pass: real/sim directory structure is documented | Sliding localization is useful, but terms are a veto and unsafe serialization adds cost | `REJECT` | Re-review after dataset terms and a safe, checksummed pose representation are published |
| SSVTP | Fail: no data-reuse terms found | Weak: mutable Drive link lacks size, version, and hashes | Partial: paired-image count/tasks known, native manifest not verified | Much of the data already flows through supported TVL discovery | `REJECT` | Re-review only with explicit terms and unique upstream spatial annotations in a stable manifest |
| VisGel | Fail: no dataset or code license found | Weak: roughly 412 GB behind legacy scripts without a modern manifest | Pass: original train/test structure is described | Strong adoption, but major overlap with FoTa/Touch100k and very high storage cost | `REJECT` | Re-review only if licensed, checksummed native sequences provide value absent from existing adapters |
| PhysiCLeAR / Octopi | Fail: no data or model terms found | Weak: mutable Drive links and unresolved original/1.5 relationship | Partial: original collection protocol known, release schemas not versioned | Language/property labels overlap TVL and Touch100k | `REJECT` | Re-review after explicit terms and a versioned 1.5 manifest establish non-duplicative content |

## Prioritized implementation packets

The four accepted releases are deliberately bounded. Support begins only when a later step lands
and verifies the corresponding registry metadata and adapter; this Step 10 decision does not raise
any candidate above C0 by itself.

### S11.1 — Tactile MNIST real single-touch 64×64

- **Scope and evidence:** Integrate only
  [`TimSchneider42/tactile-mnist-touch-real-single-t256-64x64`](https://huggingface.co/datasets/TimSchneider42/tactile-mnist-touch-real-single-t256-64x64)
  at revision `9dea65296841c1407fc0f7b26fa4e25449c34396`, not every synthetic, mesh,
  Starstruck, full-resolution, or video-sequence variant.
- **Files and API:** Add an umbrella discovery record for the Tactile MNIST family only if needed
  for honest naming, plus a C3 child record and local PyArrow adapter for
  `data/train-00000-of-00001.parquet` and `data/test-00000-of-00001.parquet`.
- **Acquisition and integrity:** Preserve the publisher's CC BY 2.0 terms, exact total download
  size `132737935`, and LFS SHA-256 values
  `f168f4da532ada9815fe99e26b4f8990290e0ffb6623e31d6f6505897f908545` (train,
  `110578548` bytes) and
  `022f96d37493ee7989e51c1ee829cdd0569263620bc0a76a44e9ef46ac8ac1` (test,
  `22159387` bytes); acquisition must remain explicit and revision-pinned.
- **Normalized mapping:** One Parquet row becomes a 256-touch sequence with GelSight Mini images,
  label, object ID, intended positions, gel pose, gel/run metadata, and publisher timestamps;
  `object_id` is the leakage group and the published 500-train/100-test split is preserved.
- **Tests and examples:** Use a tiny synthetic nested-Parquet fixture to cover both splits,
  batch-wise iteration, image decoding, malformed fixed-length fields, checksum metadata, bounded
  validation, and a first-sample example without network access.
- **License/provenance behavior:** Cite the ICLR 2026 paper and selected hub revision, state that
  other family variants have not been qualified, and add exactly two README sentences for the
  named family while specifying the supported child.
- **Acceptance criteria:** C3 loading must use bounded Parquet batches, emit deterministic unique
  IDs and 256-touch shapes, perform no import-time/network work, and pass full tests and docs.

### S11.2 — PHAC-2 version 1.0

- **Scope and evidence:** Integrate [PHAC-2](https://doi.org/10.17617/3.0C79KW) version 1.0 from
  Edmond/Max Planck's Dataverse record, whose API reports 533 files and `863911842` total bytes:
  530 object JPEGs, one HDF5 file, one MATLAB label-scale file, and the publisher processing script.
- **Files and API:** Add C3 metadata and a local HDF5 adapter for
  `phac_train_test_pos_neg_90_10_1_20.h5`; expose the 530 object images as optional lazy assets
  rather than opening or downloading them during ordinary haptic iteration.
- **Acquisition and integrity:** Use the persistent DOI/Dataverse API and explicit selective
  download; record algorithm-tagged publisher MD5 values without mislabeling them as SHA-256,
  including `1817bda104b97167b93f80d920f0afa3` for the `472270926`-byte HDF5 file.
- **Normalized mapping:** Follow `processPHAC2.m`: parse trial groups ending in the publisher's
  object/trial pattern, preserve adjectives, BioTac electrodes/PAC/PDC/TAC/TDC for both fingers,
  gripper effort/position/velocity, and optional transforms/acceleration, and expose the four
  publisher-defined action masks without inventing timestamps or units.
- **Tests and examples:** Build a synthetic HDF5 fixture with two object groups and all four
  controller states; test optional channels, object-safe grouping, malformed shapes/states,
  selective image references, explicit checksum handling, and a network-free example.
- **License/provenance behavior:** Surface CC BY-SA 4.0, DOI/version, source file IDs, citations,
  checksum algorithm, and share-alike requirements in metadata and documentation.
- **Acceptance criteria:** C3 loading must be lazy per HDF5 group/action, never materialize the
  472 MB file at once, preserve object groups and adjective labels, and fail clearly on unknown
  layouts rather than approximating the PHAC protocol.

### S12.1 — YCB-Sight real and simulated releases

- **Scope and evidence:** Integrate the official
  [YCB-Sight repository](https://github.com/Robo-Touch/YCB-Sight) at revision
  `7e688d9d3579d42fc7bb44f320caeeb55e5ef695`, covering both the documented 30-object simulated
  release and six-object real release.
- **Files and API:** Add C3 local loading with explicit `real`/`sim` mode detection; retain JPEG,
  TIFF, MP4, and mesh-like files as lazy assets and load bounded NPY/text/CSV/JSON metadata only
  when constructing the relevant sample.
- **Acquisition and integrity:** Keep acquisition manual and per-object through official Drive
  links because no publisher checksums or immutable data revision exist; never imply that the
  pinned code revision pins the Drive bytes.
- **Normalized mapping:** Simulated samples preserve GelSight image, contact mask, height map,
  depth and pose; real samples preserve GelSight/RGB/depth/point-cloud associations, robot CSV,
  and transforms. Domain and object are explicit, and object is the leakage group.
- **Tests and examples:** Cover a minimal real object and simulated object, natural filename
  ordering, timestamp association, shape mismatches, missing pairs, unsafe paths, lazy media,
  bounded validation, and a no-network example.
- **License/provenance behavior:** Surface CC BY-SA 4.0 data terms separately from MIT processing
  code and from upstream YCB object terms; exports must retain attribution/share-alike notices.
- **Acceptance criteria:** C3 loading must support each domain independently, remain useful from
  a single publisher object archive, and expose native geometry omitted by FoTa without claiming
  calibrated conversion.

### S12.2 — TacQuad

- **Scope and evidence:** Integrate [TacQuad](https://huggingface.co/datasets/xxuan01/TacQuad) at
  revision `7b46e7836f6688788eb52e6988d563ea334f5dd0`; unrelated text/features for Touch and Go,
  ObjectFolder Real, SSVTP, TVL, and Octopi are outside this integration.
- **Files and API:** Add C3 loading from an explicitly extracted `tacquad.zip` layout driven by
  `contact_indoor.csv` and `contact_outdoor.csv`; touch, scene image, and text remain separate
  observations, and precomputed `.pt` features are not deserialized by default.
- **Acquisition and integrity:** Use the official pinned Hugging Face tree, require
  `confirm_large_download=True`, record all seven multipart sizes and LFS SHA-256 values, and
  explain reassembly/extraction without doing it in tests or ordinary iteration.
- **Normalized mapping:** Preserve fine/coarse alignment, indoor/outdoor domain, object/contact
  groups, sensor identity, frame bounds, scene image, and text. Do not invent canonical splits or
  temporal pairing where the publisher describes only coarse spatial alignment.
- **Tests and examples:** Use a synthetic extracted tree and CSVs to cover GelSight Mini, DIGIT,
  and DuraGel paths, grouping, range validation, missing frames, duplicate rows, language, unsafe
  paths, large-download confirmation, and bounded first-sample inspection.
- **License/provenance behavior:** Surface the hub's MIT declaration, dataset and AnyTouch code
  revisions, archive-member hashes, and derivation boundaries for any linked source text files.
- **Acceptance criteria:** Before C3, reconcile the data card's four-sensor claim with official
  loader paths for only GelSight Mini, DIGIT, and DuraGel; either verify Tac3D's native layout or
  explicitly limit the adapter while retaining honest dataset-level metadata.

## Deferred and rejected follow-up policy

No deferred or rejected release should receive a registry entry merely to increase a coverage
count. A future proposal must link new primary evidence that resolves the row's decisive blocker,
then rerun the same five gates and document why existing adapters do not already preserve the
useful data.

Steps 11 and 12 should not substitute a deferred candidate if one selected packet becomes
blocked. They should complete a coherent subset, document the exact blocker, and leave the
remaining decision unchanged until a later evidence-backed review.
