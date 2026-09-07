"""Evidence-backed discovery records for the versioned LMT texture releases."""

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
from tactile_toolkit.types import Modality

LMT_HOMEPAGE = "https://www.ce.cit.tum.de/en/lmt/forschung/datensaetze/texture-database/"
LMT_ARCHIVE_URL = "https://zeus.lkn.ei.tum.de/downloads/texture/"

_CREATORS = (
    "Matti Strese",
    "Jun-Yong Lee",
    "Clemens Schuwerk",
    "Qingfu Han",
    "Hyoung-Gook Kim",
    "Eckehard Steinbach",
)
_CITATIONS = (
    Citation(
        title="A Haptic Texture Database for Tool-mediated Texture Recognition and Classification",
        url="https://mediatum.ub.tum.de/node?change_language=en&id=1241270",
        authors=_CREATORS,
        year=2014,
    ),
    Citation(
        title="Multimodal Feature-based Surface Material Classification",
        url="https://mediatum.ub.tum.de/node?change_language=en&id=1420980",
        authors=(
            "Matti Strese",
            "Clemens Schuwerk",
            "Albert Iepure",
            "Eckehard Steinbach",
        ),
        year=2017,
        doi="10.1109/TOH.2016.2625787",
    ),
)
_LICENSE = LicenseInfo(
    name="No substantive dataset license published",
    url=LMT_HOMEPAGE,
    allows_redistribution=None,
    allows_commercial_use=None,
    requires_attribution=None,
    notes=(
        "The institutional page exposes a Licence heading but no reusable terms; public "
        "downloadability is not treated as permission to process or redistribute the files."
    ),
)
_ACCESS = (
    AccessSource(
        AccessKind.REFERENCE,
        LMT_HOMEPAGE,
        notes="Institutional description and citations for the LMT texture releases.",
    ),
    AccessSource(
        AccessKind.MANUAL,
        LMT_ARCHIVE_URL,
        notes=(
            "Official archive index; it did not provide a reliable automated response during "
            "verification, so the toolkit does not download from it."
        ),
    ),
)
_LIMITATIONS = (
    "No substantive dataset license, per-file checksum manifest, or stable file schema is "
    "published.",
    "The 69-, 108-, and 184-material archives are distinct releases and must not be combined "
    "as one version.",
    "Support is discovery-only: the toolkit neither downloads nor parses these archives.",
)

LMT_METADATA = DatasetMetadata(
    dataset_id="lmt-textures",
    name="LMT Haptic Texture Database",
    description=(
        "A Technical University of Munich collection of controlled and freehand tool-mediated "
        "texture measurements published as several distinct material releases."
    ),
    homepage=LMT_HOMEPAGE,
    provenance=ProvenanceInfo(
        publisher="Technical University of Munich",
        source_url=LMT_HOMEPAGE,
        creators=_CREATORS,
        institutions=("Technical University of Munich",),
        notes="Umbrella record; select a release-specific child record before citing data.",
    ),
    license=_LICENSE,
    citations=_CITATIONS,
    access=_ACCESS,
    modalities=(Modality.VIBRATION, Modality.WRENCH, Modality.POSE, Modality.VISION),
    sensors=("accelerometer", "Phantom Omni"),
    tasks=("texture recognition", "surface material classification", "haptic similarity"),
    formats=("zip",),
    resource_kind=ResourceKind.COLLECTION,
    support_level=SupportLevel.DISCOVERABLE,
    limitations=_LIMITATIONS,
    aliases=("lmt", "lmt-haptic-texture-database"),
)


def _release(
    dataset_id: str,
    name: str,
    version: str | None,
    materials: int,
    size_bytes: int,
    *,
    modalities: tuple[Modality, ...],
    aliases: tuple[str, ...] = (),
) -> DatasetMetadata:
    return DatasetMetadata(
        dataset_id=dataset_id,
        name=name,
        description=(
            f"A release-specific LMT record for the archive described as containing {materials} "
            "surface materials."
        ),
        homepage=LMT_HOMEPAGE,
        version=version,
        provenance=ProvenanceInfo(
            publisher="Technical University of Munich",
            source_url=LMT_ARCHIVE_URL,
            creators=_CREATORS,
            institutions=("Technical University of Munich",),
            derived_from=("lmt-textures",),
            notes=(
                "Byte size is the decimal-MB listing shown by the official archive index; "
                "archive contents were not downloaded during verification."
            ),
        ),
        license=_LICENSE,
        citations=_CITATIONS,
        access=_ACCESS,
        modalities=modalities,
        sensors=("accelerometer", "Phantom Omni"),
        tasks=("texture recognition", "surface material classification"),
        formats=("zip",),
        support_level=SupportLevel.DISCOVERABLE,
        approximate_size_bytes=size_bytes,
        limitations=_LIMITATIONS,
        aliases=aliases,
    )


LMT_69_METADATA = _release(
    "lmt-textures-69",
    "LMT Haptic Texture Database: 69 materials",
    "1.4",
    69,
    6_934_000_000,
    modalities=(Modality.VIBRATION, Modality.WRENCH, Modality.POSE),
    aliases=("lmt-69",),
)
LMT_108_METADATA = _release(
    "lmt-textures-108",
    "LMT Haptic Texture Database: 108 materials",
    None,
    108,
    3_190_000_000,
    modalities=(Modality.VIBRATION, Modality.WRENCH, Modality.POSE),
    aliases=("lmt-108",),
)
LMT_184_METADATA = _release(
    "lmt-textures-184",
    "LMT Haptic Texture Database: 184 materials",
    None,
    184,
    83_484_000_000,
    modalities=(Modality.VIBRATION, Modality.WRENCH, Modality.POSE),
    aliases=("lmt-184",),
)

register_dataset(LMT_METADATA)
register_dataset(LMT_69_METADATA)
register_dataset(LMT_108_METADATA)
register_dataset(LMT_184_METADATA)
