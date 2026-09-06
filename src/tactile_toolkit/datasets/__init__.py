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
from tactile_toolkit.datasets.validation import DatasetValidationReport, validate_adapter

__all__ = [
    "DATASETS",
    "DATASET_METADATA_VERSION",
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
    "IntegrityReport",
    "LicenseInfo",
    "ProvenanceInfo",
    "RegistryEntry",
    "SupportLevel",
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
