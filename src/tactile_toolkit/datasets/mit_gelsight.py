"""Discoverable metadata for Tac2Pose and study-specific MIT GelSight data."""

from tactile_toolkit.datasets.metadata import (
    AccessKind,
    AccessSource,
    Citation,
    DatasetMetadata,
    LicenseInfo,
    ProvenanceInfo,
    SupportLevel,
)
from tactile_toolkit.datasets.registry import register_dataset
from tactile_toolkit.types import Modality

_UNKNOWN_DATA_TERMS = LicenseInfo(
    name="Data license not stated",
    allows_redistribution=None,
    allows_commercial_use=None,
    requires_attribution=None,
    notes=(
        "The official project or study pages do not state reusable dataset terms; publication "
        "licenses and code licenses do not establish rights for the data."
    ),
)

TAC2POSE_METADATA = DatasetMetadata(
    dataset_id="tac2pose",
    name="Tac2Pose",
    description=(
        "Paper-described GelSlim 3.0 observations, object geometry, and ground-truth poses for "
        "estimating a known object's pose from tactile contact."
    ),
    homepage="https://arxiv.org/abs/2204.11701",
    version="2023 journal article",
    provenance=ProvenanceInfo(
        publisher="MIT MCube Lab",
        source_url="https://arxiv.org/abs/2204.11701",
        creators=("Maria Bauza", "Antonia Bronars", "Alberto Rodriguez"),
        institutions=("Massachusetts Institute of Technology",),
        notes=(
            "The paper names a project URL for data and object models, but no active file "
            "manifest or official implementation could be verified on 2026-09-06."
        ),
    ),
    license=_UNKNOWN_DATA_TERMS,
    citations=(
        Citation(
            title="Tac2Pose: Tactile Object Pose Estimation from the First Touch",
            url="https://doi.org/10.1177/02783649231196925",
            doi="10.1177/02783649231196925",
            authors=("Maria Bauza", "Antonia Bronars", "Alberto Rodriguez"),
            year=2023,
        ),
    ),
    access=(
        AccessSource(
            AccessKind.REFERENCE,
            "http://mcube.mit.edu/research/tac2pose.html",
            notes="Project URL named by the paper; an active data manifest was not available.",
        ),
        AccessSource(
            AccessKind.REFERENCE,
            "https://arxiv.org/abs/2204.11701",
            notes="Author paper describing collection and evaluation data.",
        ),
    ),
    modalities=(Modality.VISION_TACTILE, Modality.POSE),
    sensors=("GelSlim 3.0",),
    tasks=("known-object tactile pose estimation", "contact-mask estimation"),
    formats=("RGB tactile images", "object meshes", "pose labels", "contact renderings"),
    support_level=SupportLevel.DISCOVERABLE,
    limitations=(
        "Official data files, their native schema, total byte size, checksums, and data license "
        "could not be verified.",
        "The CC BY-NC license on the journal article is not treated as a dataset license.",
    ),
    aliases=("tac-2-pose",),
)

MIT_GELSIGHT_DATASETS_METADATA = DatasetMetadata(
    dataset_id="mit-gelsight-datasets",
    name="MIT GelSight research datasets",
    description=(
        "A discoverable umbrella for separate MIT GelSight hardness, force/shear/slip, and neural "
        "slip-detection studies rather than a fictitious unified corpus."
    ),
    homepage="https://people.csail.mit.edu/yuan_wz/hardness-estimation.htm",
    provenance=ProvenanceInfo(
        publisher="MIT CSAIL",
        source_url="https://people.csail.mit.edu/yuan_wz/hardness-estimation.htm",
        creators=("Wenzhen Yuan", "Edward H. Adelson"),
        institutions=("MIT CSAIL",),
        notes=(
            "Use the study-specific child records because sensors, collection protocols, labels, "
            "downloads, and terms differ."
        ),
    ),
    license=_UNKNOWN_DATA_TERMS,
    citations=(
        Citation(
            title=(
                "GelSight: High-Resolution Robot Tactile Sensors for Estimating Geometry and Force"
            ),
            url="https://doi.org/10.3390/s17122762",
            doi="10.3390/s17122762",
            authors=("Wenzhen Yuan", "Siyuan Dong", "Edward H. Adelson"),
            year=2017,
        ),
    ),
    access=(
        AccessSource(
            AccessKind.REFERENCE,
            "https://people.csail.mit.edu/yuan_wz/hardness-estimation.htm",
            notes="Hardness study and dataset index.",
        ),
        AccessSource(
            AccessKind.REFERENCE,
            "https://people.csail.mit.edu/yuan_wz/force-shear-and-slip.htm",
            notes="Force, shear, and slip study page; no data package is linked.",
        ),
        AccessSource(
            AccessKind.REFERENCE,
            "https://www.csail.mit.edu/research/slip-detection-deep-neural-network-using-gelsight-touch-sensor",
            notes="Separate neural slip-detection study; no data package is linked.",
        ),
    ),
    modalities=(Modality.VISION_TACTILE, Modality.WRENCH),
    sensors=("GelSight study-specific prototypes",),
    tasks=("hardness estimation", "force and shear measurement", "slip detection"),
    formats=("study-specific",),
    support_level=SupportLevel.DISCOVERABLE,
    limitations=(
        "This record is an umbrella and must not be opened as if the studies share one schema.",
        "No collection-wide data license, revision, checksum manifest, or total size exists.",
    ),
    aliases=("mit-gelsight", "gelsight-research-datasets"),
)

MIT_GELSIGHT_HARDNESS_METADATA = DatasetMetadata(
    dataset_id="mit-gelsight-hardness",
    name="MIT GelSight hardness-estimation data",
    description=(
        "GelSight pressing sequences on silicone samples of measured Shore 00 hardness and on "
        "natural objects for shape-independent hardness estimation."
    ),
    homepage="https://people.csail.mit.edu/yuan_wz/hardness-estimation.htm",
    version="November 2016 release",
    provenance=ProvenanceInfo(
        publisher="MIT CSAIL",
        source_url="https://people.csail.mit.edu/yuan_wz/hardnessdataset/",
        creators=(
            "Wenzhen Yuan",
            "Chenzhuo Zhu",
            "Andrew Owens",
            "Mandayam A. Srinivasan",
            "Edward H. Adelson",
        ),
        institutions=("MIT CSAIL",),
    ),
    license=_UNKNOWN_DATA_TERMS,
    citations=(
        Citation(
            title=(
                "Shape-independent hardness estimation using deep learning and a GelSight "
                "tactile sensor"
            ),
            url="https://arxiv.org/abs/1704.03955",
            doi="10.1109/ICRA.2017.7989116",
            authors=(
                "Wenzhen Yuan",
                "Chenzhuo Zhu",
                "Andrew Owens",
                "Mandayam A. Srinivasan",
                "Edward H. Adelson",
            ),
            year=2017,
        ),
    ),
    access=(
        AccessSource(
            AccessKind.MANUAL,
            "https://people.csail.mit.edu/yuan_wz/hardnessdataset/",
            notes=(
                "Publisher index for separately hosted data volumes; review and obtain permission "
                "before reuse because data terms are unstated."
            ),
        ),
    ),
    modalities=(Modality.VISION_TACTILE,),
    sensors=("GelSight fingertip prototype",),
    tasks=("object hardness estimation",),
    formats=("avi", "calibration files", "filename-encoded shape and hardness labels"),
    support_level=SupportLevel.DISCOVERABLE,
    approximate_size_bytes=30_700_000_000,
    limitations=(
        "The public index lists about 30.7 GB across separate files but does not document their "
        "relationships or provide checksums.",
        "Natural-object sequences have no ground-truth hardness.",
        "No adapter is provided until the publisher clarifies reusable data terms.",
    ),
    aliases=("gelsight-hardness",),
)

MIT_GELSIGHT_FORCE_SHEAR_SLIP_METADATA = DatasetMetadata(
    dataset_id="mit-gelsight-force-shear-slip",
    name="MIT GelSight force, shear, and slip study",
    description=(
        "GelSight marker-motion experiments relating contact deformation to force type, shear, "
        "torsional load, and incipient slip."
    ),
    homepage="https://people.csail.mit.edu/yuan_wz/force-shear-and-slip.htm",
    version="2015 study",
    provenance=ProvenanceInfo(
        publisher="MIT CSAIL",
        source_url="https://people.csail.mit.edu/yuan_wz/force-shear-and-slip.htm",
        creators=("Wenzhen Yuan", "Rui Li", "Mandayam A. Srinivasan", "Edward H. Adelson"),
        institutions=("MIT CSAIL",),
    ),
    license=_UNKNOWN_DATA_TERMS,
    citations=(
        Citation(
            title="Measurement of Shear and Slip with a GelSight Tactile Sensor",
            url="https://people.csail.mit.edu/yuan_wz/GelSight1/ICRA15_2740_FI.pdf",
            doi="10.1109/ICRA.2015.7139016",
            authors=("Wenzhen Yuan", "Rui Li", "Mandayam A. Srinivasan", "Edward H. Adelson"),
            year=2015,
        ),
    ),
    access=(
        AccessSource(
            AccessKind.REFERENCE,
            "https://people.csail.mit.edu/yuan_wz/force-shear-and-slip.htm",
            notes="Official study page; no downloadable dataset package is linked.",
        ),
    ),
    modalities=(Modality.VISION_TACTILE, Modality.WRENCH),
    sensors=("GelSight marker-based prototype",),
    tasks=("force-type inference", "shear measurement", "incipient slip detection"),
    formats=("no public data package verified",),
    support_level=SupportLevel.DISCOVERABLE,
    limitations=(
        "The study page provides papers and demonstrations but no data files or data license.",
    ),
    aliases=("gelsight-force-shear-slip",),
)

MIT_GELSIGHT_NEURAL_SLIP_METADATA = DatasetMetadata(
    dataset_id="mit-gelsight-neural-slip",
    name="MIT GelSight neural slip-detection study",
    description=(
        "A separate MIT CSAIL project that uses robot-collected GelSight video sequences and an "
        "LSTM to classify whether slip occurs."
    ),
    homepage=(
        "https://www.csail.mit.edu/research/"
        "slip-detection-deep-neural-network-using-gelsight-touch-sensor"
    ),
    version="2017 study page",
    provenance=ProvenanceInfo(
        publisher="MIT CSAIL",
        source_url=(
            "https://www.csail.mit.edu/research/"
            "slip-detection-deep-neural-network-using-gelsight-touch-sensor"
        ),
        creators=("Jianhua Li",),
        institutions=("MIT CSAIL",),
    ),
    license=_UNKNOWN_DATA_TERMS,
    citations=(
        Citation(
            title="Slip Detection with Deep Neural Network by Using GelSight Touch Sensor",
            url=(
                "https://www.csail.mit.edu/research/"
                "slip-detection-deep-neural-network-using-gelsight-touch-sensor"
            ),
            authors=("Jianhua Li",),
            year=2017,
        ),
    ),
    access=(
        AccessSource(
            AccessKind.REFERENCE,
            (
                "https://www.csail.mit.edu/research/"
                "slip-detection-deep-neural-network-using-gelsight-touch-sensor"
            ),
            notes="Official project description; no downloadable dataset package is linked.",
        ),
    ),
    modalities=(Modality.VISION_TACTILE,),
    sensors=("GelSight",),
    tasks=("slip detection",),
    formats=("video sequences described; no public data package verified",),
    support_level=SupportLevel.DISCOVERABLE,
    limitations=("No downloadable files, manifest, size, checksums, or data terms are published.",),
    aliases=("gelsight-neural-slip",),
)

for _metadata in (
    TAC2POSE_METADATA,
    MIT_GELSIGHT_DATASETS_METADATA,
    MIT_GELSIGHT_HARDNESS_METADATA,
    MIT_GELSIGHT_FORCE_SHEAR_SLIP_METADATA,
    MIT_GELSIGHT_NEURAL_SLIP_METADATA,
):
    register_dataset(_metadata)
