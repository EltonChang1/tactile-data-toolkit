"""Exceptions raised by the published-dataset integration layer."""

from __future__ import annotations


class DatasetError(RuntimeError):
    """Base error for dataset discovery, access, validation, and loading."""


class DatasetMetadataError(DatasetError, ValueError):
    """Dataset metadata does not conform to the supported schema."""


class DatasetNotFoundError(DatasetError, KeyError):
    """No registered dataset matches the requested identifier or alias."""


class DatasetUnavailableError(DatasetError):
    """A dataset is cataloged but has no usable adapter in this installation."""


class DatasetAccessError(DatasetError):
    """A dataset source cannot be accessed through the requested method."""


class DatasetIntegrityError(DatasetError):
    """Downloaded or local data fail an expected size or checksum."""


class DatasetValidationError(DatasetError, ValueError):
    """Metadata or adapter samples fail dataset-level validation."""
