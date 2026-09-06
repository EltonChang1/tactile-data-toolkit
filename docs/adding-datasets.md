# Adding a published dataset

Dataset integrations live under `tactile_toolkit.datasets` and are deliberately separate from raw-log readers. An adapter describes a published corpus, lazily yields normalized `TactileSample` objects, and never downloads a large or gated dataset merely because it was imported or opened.

## 1. Verify before implementing

Use the original paper, official project/lab page, official repository, and official data host. Record the dataset license separately from code and model licenses; if redistribution or commercial-use permission is not stated, preserve it as `None` rather than guessing.

Before code review, capture:

- canonical homepage, paper or DOI, and official access URL;
- version, immutable revision, published checksums, and approximate bytes when available;
- sensors, modalities, tasks, native formats, split/group semantics, and limitations;
- authentication, terms acceptance, noncommercial restrictions, or manual steps;
- a support target that does not promise conversion for fields the toolkit cannot represent.

Update [the source catalog](dataset-catalog.md) and [compatibility matrix](dataset-coverage.md) when evidence or support changes.

## 2. Declare typed metadata

`DatasetMetadata.to_dict()` is the stable `tactile-dataset-metadata/0.1` interchange shape. Permission booleans are three-valued: `True`, `False`, or `None` when the publisher does not say.

```python
from tactile_toolkit.datasets import (
    AccessKind,
    AccessSource,
    Citation,
    DatasetMetadata,
    LicenseInfo,
    ProvenanceInfo,
    SupportLevel,
)
from tactile_toolkit.types import Modality

METADATA = DatasetMetadata(
    dataset_id="example-touch",
    name="Example Touch",
    description="A concise, evidence-backed description.",
    homepage="https://institution.example/datasets/example-touch",
    version="1.0",
    revision="immutable-release-id",
    provenance=ProvenanceInfo(
        publisher="Publishing institution",
        source_url="https://institution.example/datasets/example-touch",
        creators=("First Author", "Second Author"),
        institutions=("Publishing institution",),
        derived_from=(),
    ),
    license=LicenseInfo(
        name="Creative Commons Attribution 4.0",
        spdx_id="CC-BY-4.0",
        url="https://creativecommons.org/licenses/by/4.0/",
        allows_redistribution=True,
        allows_commercial_use=True,
        requires_attribution=True,
    ),
    citations=(
        Citation(
            title="Example Touch Dataset",
            url="https://doi.org/10.example/touch",
            doi="10.example/touch",
            authors=("First Author", "Second Author"),
            year=2026,
        ),
    ),
    access=(
        AccessSource(
            AccessKind.HTTP,
            "https://institution.example/datasets/example-touch-v1.tar",
        ),
    ),
    modalities=(Modality.VISION_TACTILE,),
    sensors=("Published sensor name",),
    tasks=("material classification",),
    formats=("tar", "jpeg", "json"),
    support_level=SupportLevel.LOADABLE,
    approximate_size_bytes=1_000_000,
    limitations=("Describe coverage and known collection bias.",),
    aliases=("example",),
)
```

JSON metadata can be loaded with `DatasetMetadata.from_dict`; unknown schema versions, invalid URLs, malformed SHA-256 values, and missing required fields raise `DatasetMetadataError` with the offending field.

## 3. Implement a lazy adapter

Use `AssetReference` for images, video, archives, or arrays that should not be opened while enumerating samples. Use NumPy arrays only when the native record is already small and materialized, keep dataset-specific labels in `labels`, and use `group_id` for the indivisible object/trajectory/participant unit used to build leakage-safe splits.

```python
from collections.abc import Iterator

from tactile_toolkit.datasets import DatasetAdapter, dataset_adapter
from tactile_toolkit.model import AssetReference, TactileObservation, TactileSample
from tactile_toolkit.types import Modality


@dataset_adapter(METADATA)
class ExampleAdapter(DatasetAdapter):
    def iter_samples(self, *, split: str | None = None) -> Iterator[TactileSample]:
        root = self.require_root()
        for row in read_official_manifest(root):
            if split is not None and row["split"] != split:
                continue
            yield TactileSample(
                sample_id=row["id"],
                observations={
                    "touch": TactileObservation(
                        Modality.VISION_TACTILE,
                        AssetReference(row["image"], media_type="image/jpeg"),
                        sensor="Published sensor name",
                    )
                },
                labels={"material": row["material"]},
                task="material classification",
                split=row["split"],
                group_id=row["object_id"],
            )
```

Adapter requirements:

- iteration is lazy and deterministic under a pinned revision;
- `split=` filters without relabeling publisher-defined splits;
- sample IDs are stable and unique, and every emitted modality is declared in metadata;
- missing roots, files, fields, or access credentials produce actionable `DatasetError` subclasses;
- unsafe formats such as pickle require explicit user trust and a documented conversion boundary;
- an adapter does not synthesize absent contact, force, timing, or calibration data.

Call `adapter.validate(limit=32, check_assets=True)` to validate a bounded prefix, declared modalities, IDs, splits, local asset containment, size, and checksums. Validation intentionally does not traverse an entire multi-terabyte corpus by default.

## 4. Acquire data explicitly

`DatasetCache` handles ordinary public HTTP(S) assets. It namespaces files by dataset ID, resumes server-supported Range requests from `.part` files, restarts safely when a server ignores Range, verifies expected size and SHA-256, and atomically promotes only verified downloads.

```python
from tactile_toolkit.datasets import DatasetCache

cache = DatasetCache()  # TACTILE_TOOLKIT_CACHE, XDG cache, or platform default
archive = cache.download(
    METADATA.access[0].url,
    METADATA.dataset_id,
    filename="example-touch-v1.tar",
    size_bytes=1_000_000,
    sha256="<publisher-provided 64-character SHA-256>",
)
```

Do not route gated Hugging Face, Google Drive acknowledgment flows, noncommercial datasets, expiring URLs, or manual institutional downloads through this generic helper. Their adapters should show official instructions, accept a user-provided root or authenticated client, and preserve the publisher's terms.

## 5. Test without the corpus

Use `tests/fixtures/datasets/minimal` as the structural example: a versioned metadata JSON document, JSONL manifest, and two tiny synthetic pressure files. Tests should exercise representative layouts, malformed metadata, missing pairs, split behavior, bounded validation, resume behavior, checksum failure, and clear messages without contacting the public internet.

Run:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy --follow-imports=skip src/tactile_toolkit/datasets
```

## Review checklist

- [ ] Primary sources and current license terms are linked.
- [ ] Metadata round-trips through `to_dict()` and `from_dict()`.
- [ ] No import, constructor, test, or ordinary iteration triggers a full download.
- [ ] Assets remain lazy and paths cannot escape the user-provided root during validation.
- [ ] Checksums are verified when publishers provide them; absence is stated, not fabricated.
- [ ] Canonical splits and group boundaries are preserved.
- [ ] Support grades and limitations are updated without overstating conversion coverage.
- [ ] The README entry has exactly two complete sentences in the required format.
