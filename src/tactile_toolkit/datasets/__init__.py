"""Published-dataset metadata, adapters, registry, acquisition, and validation."""

from tactile_toolkit.datasets.base import DatasetAdapter
from tactile_toolkit.datasets.download import (
    DatasetCache,
    IntegrityReport,
    default_cache_dir,
    sha256_file,
    verify_file,
)
from tactile_toolkit.datasets.errors import (
    DatasetAccessError,
    DatasetError,
    DatasetIntegrityError,
    DatasetMetadataError,
    DatasetNotFoundError,
    DatasetUnavailableError,
    DatasetValidationError,
)
from tactile_toolkit.datasets.foundation_tactile import (
    FOUNDATION_TACTILE_METADATA,
    FOUNDATION_TACTILE_REVISION,
    FOUNDATION_TACTILE_SOURCE,
    FoundationTactileAdapter,
)
from tactile_toolkit.datasets.freetacman import (
    FREETACMAN_METADATA,
    FREETACMAN_REVISION,
    FREETACMAN_SOURCE,
    FreeTacManAdapter,
)
from tactile_toolkit.datasets.huggingface import HuggingFaceSnapshot
from tactile_toolkit.datasets.metadata import (
    DATASET_METADATA_VERSION,
    AccessKind,
    AccessSource,
    Citation,
    DatasetMetadata,
    LicenseInfo,
    ProvenanceInfo,
    SupportLevel,
    validate_metadata,
)
from tactile_toolkit.datasets.mit_gelsight import (
    MIT_GELSIGHT_DATASETS_METADATA,
    MIT_GELSIGHT_FORCE_SHEAR_SLIP_METADATA,
    MIT_GELSIGHT_HARDNESS_METADATA,
    MIT_GELSIGHT_NEURAL_SLIP_METADATA,
    TAC2POSE_METADATA,
)
from tactile_toolkit.datasets.registry import (
    DATASETS,
    DatasetRegistry,
    RegistryEntry,
    dataset_adapter,
    get_dataset_metadata,
    list_datasets,
    open_dataset,
    register_dataset,
)
from tactile_toolkit.datasets.touch100k import (
    TOUCH100K_DATA_URL,
    TOUCH100K_METADATA,
    TOUCH100K_SCHEMA_REVISION,
    Touch100kAdapter,
)
from tactile_toolkit.datasets.validation import DatasetValidationReport, validate_adapter

__all__ = [
    "DATASETS",
    "DATASET_METADATA_VERSION",
    "FOUNDATION_TACTILE_METADATA",
    "FOUNDATION_TACTILE_REVISION",
    "FOUNDATION_TACTILE_SOURCE",
    "FREETACMAN_METADATA",
    "FREETACMAN_REVISION",
    "FREETACMAN_SOURCE",
    "MIT_GELSIGHT_DATASETS_METADATA",
    "MIT_GELSIGHT_FORCE_SHEAR_SLIP_METADATA",
    "MIT_GELSIGHT_HARDNESS_METADATA",
    "MIT_GELSIGHT_NEURAL_SLIP_METADATA",
    "TAC2POSE_METADATA",
    "TOUCH100K_DATA_URL",
    "TOUCH100K_METADATA",
    "TOUCH100K_SCHEMA_REVISION",
    "AccessKind",
    "AccessSource",
    "Citation",
    "DatasetAccessError",
    "DatasetAdapter",
    "DatasetCache",
    "DatasetError",
    "DatasetIntegrityError",
    "DatasetMetadata",
    "DatasetMetadataError",
    "DatasetNotFoundError",
    "DatasetRegistry",
    "DatasetUnavailableError",
    "DatasetValidationError",
    "DatasetValidationReport",
    "FoundationTactileAdapter",
    "FreeTacManAdapter",
    "HuggingFaceSnapshot",
    "IntegrityReport",
    "LicenseInfo",
    "ProvenanceInfo",
    "RegistryEntry",
    "SupportLevel",
    "Touch100kAdapter",
    "dataset_adapter",
    "default_cache_dir",
    "get_dataset_metadata",
    "list_datasets",
    "open_dataset",
    "register_dataset",
    "sha256_file",
    "validate_adapter",
    "validate_metadata",
    "verify_file",
]
