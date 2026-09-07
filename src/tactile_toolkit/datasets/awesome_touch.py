"""Reference-only catalog integration for the community Awesome-Touch index."""

from __future__ import annotations

from tactile_toolkit.datasets.metadata import (
    AccessKind,
    AccessSource,
    Citation,
    DatasetMetadata,
    LicenseInfo,
    ProvenanceInfo,
    ResourceKind,
    SupportLevel,
)
from tactile_toolkit.datasets.registry import register_dataset

AWESOME_TOUCH_REPOSITORY_URL = "https://github.com/linchangyi1/Awesome-Touch"
AWESOME_TOUCH_REVISION = "3d07843d2850eb0111001802d9658c9f91c22aca"
AWESOME_TOUCH_README_URL = f"{AWESOME_TOUCH_REPOSITORY_URL}/blob/{AWESOME_TOUCH_REVISION}/README.md"
AWESOME_TOUCH_DATASET_SECTION_URL = f"{AWESOME_TOUCH_README_URL}#dataset"

AWESOME_TOUCH_METADATA = DatasetMetadata(
    dataset_id="awesome-touch",
    name="Awesome-Touch",
    description=(
        "A community-maintained discovery index for tactile sensing research, software, data, "
        "hardware, products, and laboratories rather than a tactile dataset."
    ),
    homepage=AWESOME_TOUCH_REPOSITORY_URL,
    version="repository snapshot dated 2026-08-21",
    revision=AWESOME_TOUCH_REVISION,
    provenance=ProvenanceInfo(
        publisher="Awesome-Touch maintainers",
        source_url=AWESOME_TOUCH_REPOSITORY_URL,
        creators=("Changyi Lin",),
        notes=(
            "At Step 08 verification the official repository had 768 stars, 56 forks, and 432 "
            "commits; those popularity counts are a mutable 2026-09-06 snapshot."
        ),
    ),
    license=LicenseInfo(
        name="MIT License",
        spdx_id="MIT",
        url=f"{AWESOME_TOUCH_REPOSITORY_URL}/blob/{AWESOME_TOUCH_REVISION}/LICENSE.md",
        allows_redistribution=True,
        allows_commercial_use=True,
        requires_attribution=True,
        notes=(
            "MIT covers the Awesome-Touch index repository only; every linked paper, dataset, "
            "tool, model, and hardware project retains independent terms."
        ),
    ),
    citations=(
        Citation(
            title="Awesome-Touch",
            url=AWESOME_TOUCH_REPOSITORY_URL,
            authors=("Changyi Lin",),
            year=2022,
        ),
    ),
    access=(
        AccessSource(
            AccessKind.REFERENCE,
            AWESOME_TOUCH_REPOSITORY_URL,
            notes="Mutable canonical repository for community discovery and contributions.",
        ),
        AccessSource(
            AccessKind.REFERENCE,
            AWESOME_TOUCH_DATASET_SECTION_URL,
            notes="Revision-pinned dataset-link section used for this catalog snapshot.",
        ),
    ),
    tasks=("tactile and haptics resource discovery",),
    formats=("markdown", "git repository", "external links"),
    resource_kind=ResourceKind.REFERENCE_INDEX,
    support_level=SupportLevel.DISCOVERABLE,
    limitations=(
        "Awesome-Touch is not a dataset and cannot yield TactileSample records.",
        "A link in the index is not evidence of open access, licensing, availability, or quality.",
        "Consumers must verify each linked resource with its primary source before integration.",
        "The index and its popularity signals are mutable; reproducible audits should use the "
        "pinned revision.",
    ),
    aliases=("awesome_touch", "awesome-tactile"),
)

register_dataset(AWESOME_TOUCH_METADATA)
