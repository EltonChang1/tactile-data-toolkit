"""Adapter contract for published tactile datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from tactile_toolkit.datasets.errors import DatasetAccessError, DatasetMetadataError
from tactile_toolkit.datasets.metadata import DatasetMetadata
from tactile_toolkit.model import TactileSample

if TYPE_CHECKING:
    from tactile_toolkit.datasets.validation import DatasetValidationReport


class DatasetAdapter(ABC):
    """Lazily expose one published dataset as normalized tactile samples.

    Subclasses declare immutable ``METADATA`` and implement ``iter_samples``.
    Construction and iteration must not download a complete dataset implicitly;
    acquisition is an explicit user action handled separately.
    """

    METADATA: ClassVar[DatasetMetadata]

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root).expanduser() if root is not None else None

    @property
    def metadata(self) -> DatasetMetadata:
        metadata = getattr(type(self), "METADATA", None)
        if not isinstance(metadata, DatasetMetadata):
            raise DatasetMetadataError(
                f"{type(self).__name__} must declare METADATA as DatasetMetadata"
            )
        return metadata

    def require_root(self) -> Path:
        """Return an existing user-provided root or raise an actionable error."""
        if self.root is None:
            raise DatasetAccessError(
                f"{self.metadata.name} requires a local root; obtain the data through an official "
                "access source and pass root=..."
            )
        if not self.root.exists():
            raise DatasetAccessError(
                f"Dataset root does not exist for {self.metadata.dataset_id}: {self.root}"
            )
        return self.root

    @abstractmethod
    def iter_samples(self, *, split: str | None = None) -> Iterator[TactileSample]:
        """Yield normalized samples without materializing the whole dataset."""

    def __iter__(self) -> Iterator[TactileSample]:
        return self.iter_samples()

    def validate(
        self,
        *,
        split: str | None = None,
        limit: int = 32,
        check_assets: bool = False,
    ) -> DatasetValidationReport:
        """Validate a bounded prefix without turning validation into a full download."""
        from tactile_toolkit.datasets.validation import validate_adapter

        return validate_adapter(self, split=split, limit=limit, check_assets=check_assets)
