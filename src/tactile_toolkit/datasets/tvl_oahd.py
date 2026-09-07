"""Honest discovery records for TVL and the OAHD collection."""

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

TVL_ORIGINAL_REVISION = "3324b2ee956b8d94974bdd8272dce0ade292cb56"
TVL_CORRECTED_REVISION = "aab39e037d8916717dc98add89d58cbd39219770"

TVL_METADATA = DatasetMetadata(
    dataset_id="tvl",
    name="Touch-Vision-Language (TVL)",
    description=(
        "An open-vocabulary alignment corpus pairing DIGIT touch, scene vision, and tactile "
        "language, retained as discovery-only until dataset reuse terms are published."
    ),
    homepage="https://tactile-vlm.github.io/",
    version="ICML 2024 release",
    revision=TVL_CORRECTED_REVISION,
    provenance=ProvenanceInfo(
        publisher="Touch-Vision-Language project authors",
        source_url="https://tactile-vlm.github.io/",
        institutions=("Carnegie Mellon University", "Meta AI"),
        derived_from=("SSVTP", "HCT"),
        notes=(
            "The corrected repository is community-hosted but is the revision linked by the "
            "official code README to repair the original image/tactile directory swap."
        ),
    ),
    license=LicenseInfo(
        name="Terms unclear",
        allows_redistribution=None,
        allows_commercial_use=None,
        requires_attribution=None,
        notes=(
            "The official project and both dataset cards state no dataset license; Apache-2.0 "
            "on the code repository is not treated as a license for the data."
        ),
    ),
    citations=(
        Citation(
            title="A Touch, Vision, and Language Dataset for Multimodal Alignment",
            url="https://openreview.net/forum?id=tFEOOH9eH0",
            doi="10.48550/arXiv.2402.13232",
            year=2024,
        ),
    ),
    access=(
        AccessSource(
            AccessKind.HUGGING_FACE,
            "https://huggingface.co/datasets/mlfu7/Touch-Vision-Language-Dataset",
            notes=(
                f"Original {199_349_451_898}-byte snapshot at {TVL_ORIGINAL_REVISION}; its "
                "tactile and image directories are swapped."
            ),
        ),
        AccessSource(
            AccessKind.HUGGING_FACE,
            "https://huggingface.co/datasets/yoorhim/TVL-revise",
            notes=(
                f"Corrected {75_292_565_517}-byte snapshot at {TVL_CORRECTED_REVISION}, linked "
                "from the official code README."
            ),
        ),
    ),
    modalities=(Modality.VISION_TACTILE, Modality.VISION, Modality.LANGUAGE),
    sensors=("Meta DIGIT", "Logitech BRIO"),
    tasks=("touch-vision-language alignment", "open-vocabulary tactile description"),
    formats=("multipart zip", "jpeg", "json"),
    support_level=SupportLevel.DISCOVERABLE,
    approximate_size_bytes=75_292_565_517,
    limitations=(
        "No dataset-level reuse terms were stated on the official project or data cards when "
        "verified on 2026-09-06, so the toolkit does not download or deserialize the corpus.",
        "The original release has a documented tactile/image directory swap and its mixed JSON "
        "files do not form one Hugging Face viewer schema.",
        "Language labels combine human annotations and model-generated descriptions.",
    ),
    aliases=("touch-vision-language", "touch-vision-language-dataset"),
)

OAHD_METADATA = DatasetMetadata(
    dataset_id="oahd",
    name="Open Access Haptic Database (OAHD)",
    description=(
        "A Georgia Tech collection of study-specific multimodal haptic releases and open sensor "
        "designs, represented as an umbrella rather than a uniform dataset."
    ),
    homepage="https://www.oahd.gatech.edu/",
    version="mutable study collection",
    provenance=ProvenanceInfo(
        publisher="Georgia Tech Healthcare Robotics Lab",
        source_url="https://www.oahd.gatech.edu/",
        creators=("Tapomayukh Bhattacharjee",),
        institutions=("Georgia Institute of Technology",),
        notes=(
            "Cornell EmPRISE links to this resource but does not publish OAHD; each OAHD study "
            "has its own paper, files, and experimental protocol."
        ),
    ),
    license=LicenseInfo(
        name="Collection-wide terms unclear",
        allows_redistribution=None,
        allows_commercial_use=None,
        requires_attribution=None,
        notes=(
            "The collection calls its resources open access but does not state one reusable "
            "license covering every linked study."
        ),
    ),
    citations=(
        Citation(
            title="Open Access Haptic Database",
            url="https://www.oahd.gatech.edu/",
        ),
    ),
    access=(
        AccessSource(
            AccessKind.REFERENCE,
            "https://www.oahd.gatech.edu/multi-data/",
            notes="Study pages link their own data, scripts, and associated paper.",
        ),
        AccessSource(
            AccessKind.REFERENCE,
            "https://www.oahd.gatech.edu/sensors/",
            notes="Official sensor descriptions and open hardware design links.",
        ),
    ),
    modalities=(
        Modality.PRESSURE,
        Modality.TEMPERATURE,
        Modality.VIBRATION,
        Modality.AUDIO,
        Modality.IMU,
        Modality.POSE,
    ),
    sensors=(
        "fabric force-sensing skin",
        "fabric force and thermal sensing skin",
        "multimodal tactile sensing module v1/v2",
    ),
    tasks=("material recognition", "object pushing", "contact perception"),
    formats=("study-specific downloads", "Python load and visualization scripts"),
    resource_kind=ResourceKind.COLLECTION,
    support_level=SupportLevel.DISCOVERABLE,
    limitations=(
        "OAHD has no collection-wide immutable revision, size, manifest, checksum set, schema, "
        "or explicit data license.",
        "Users must review the paper and terms attached to a particular child study before use.",
        "The www.oahd.gatech.edu redirect presented a certificate-name mismatch to automated "
        "clients during verification on 2026-09-06.",
    ),
    aliases=("open-access-haptic-database",),
)

register_dataset(TVL_METADATA)
register_dataset(OAHD_METADATA)
