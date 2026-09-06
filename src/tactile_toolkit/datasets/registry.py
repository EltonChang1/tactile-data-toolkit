"""Thread-safe registry for dataset metadata and optional adapter factories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any, TypeVar

from tactile_toolkit.datasets.base import DatasetAdapter
from tactile_toolkit.datasets.errors import (
    DatasetMetadataError,
    DatasetNotFoundError,
    DatasetUnavailableError,
)
from tactile_toolkit.datasets.metadata import DatasetMetadata, SupportLevel, validate_metadata

AdapterFactory = Callable[..., DatasetAdapter]
AdapterType = TypeVar("AdapterType", bound=type[DatasetAdapter])


def _key(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetNotFoundError("Dataset identifier or alias must not be empty")
    return value.strip().lower()


@dataclass(frozen=True)
class RegistryEntry:
    metadata: DatasetMetadata
    factory: AdapterFactory | None = None


class DatasetRegistry:
    """Resolve stable identifiers and aliases without importing dataset code eagerly."""

    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}
        self._aliases: dict[str, str] = {}
        self._lock = RLock()

    def register(
        self,
        metadata: DatasetMetadata,
        factory: AdapterFactory | None = None,
        *,
        replace: bool = False,
    ) -> DatasetMetadata:
        """Register metadata and, when implemented, a lazy adapter factory."""
        metadata = validate_metadata(metadata)
        if factory is not None and not callable(factory):
            raise TypeError("dataset adapter factory must be callable")
        dataset_id = metadata.dataset_id
        names = (dataset_id, *metadata.aliases)

        with self._lock:
            existing = self._entries.get(dataset_id)
            if existing is not None and not replace:
                raise DatasetMetadataError(f"Dataset {dataset_id!r} is already registered")
            for name in names:
                normalized = _key(name)
                owner = self._aliases.get(normalized)
                if owner is not None and owner != dataset_id:
                    raise DatasetMetadataError(
                        f"Dataset alias {name!r} is already registered to {owner!r}"
                    )

            if existing is not None:
                for alias, owner in list(self._aliases.items()):
                    if owner == dataset_id:
                        del self._aliases[alias]
            self._entries[dataset_id] = RegistryEntry(metadata, factory)
            for name in names:
                self._aliases[_key(name)] = dataset_id
        return metadata

    def entry(self, dataset_id_or_alias: str) -> RegistryEntry:
        normalized = _key(dataset_id_or_alias)
        with self._lock:
            dataset_id = self._aliases.get(normalized)
            if dataset_id is None:
                available = ", ".join(sorted(self._entries)) or "none"
                raise DatasetNotFoundError(
                    f"Unknown dataset {dataset_id_or_alias!r}; registered datasets: {available}"
                )
            return self._entries[dataset_id]

    def metadata(self, dataset_id_or_alias: str) -> DatasetMetadata:
        return self.entry(dataset_id_or_alias).metadata

    def open(self, dataset_id_or_alias: str, **kwargs: Any) -> DatasetAdapter:
        """Construct a registered adapter without performing implicit acquisition."""
        entry = self.entry(dataset_id_or_alias)
        if entry.factory is None:
            sources = ", ".join(source.url for source in entry.metadata.access)
            raise DatasetUnavailableError(
                f"{entry.metadata.name} is {entry.metadata.support_level.value} but has no "
                f"adapter; official access: {sources}"
            )
        adapter = entry.factory(**kwargs)
        if not isinstance(adapter, DatasetAdapter):
            raise TypeError(
                f"Factory for {entry.metadata.dataset_id!r} returned {type(adapter).__name__}, "
                "expected DatasetAdapter"
            )
        if adapter.metadata != entry.metadata:
            raise DatasetMetadataError(
                f"Adapter metadata for {entry.metadata.dataset_id!r} does not match the "
                "registered metadata"
            )
        return adapter

    def list(self, *, minimum_support: SupportLevel | str | None = None) -> list[DatasetMetadata]:
        """List canonical metadata, optionally filtered by capability grade."""
        minimum = SupportLevel.parse(minimum_support) if minimum_support is not None else None
        order = {level: index for index, level in enumerate(SupportLevel)}
        with self._lock:
            values = [entry.metadata for entry in self._entries.values()]
        if minimum is not None:
            values = [item for item in values if order[item.support_level] >= order[minimum]]
        return sorted(values, key=lambda item: item.dataset_id)

    def __contains__(self, dataset_id_or_alias: object) -> bool:
        if not isinstance(dataset_id_or_alias, str) or not dataset_id_or_alias.strip():
            return False
        with self._lock:
            return dataset_id_or_alias.strip().lower() in self._aliases


DATASETS = DatasetRegistry()


def register_dataset(
    metadata: DatasetMetadata,
    factory: AdapterFactory | None = None,
    *,
    replace: bool = False,
) -> DatasetMetadata:
    """Register a dataset in the process-wide default registry."""
    return DATASETS.register(metadata, factory, replace=replace)


def dataset_adapter(
    metadata: DatasetMetadata, *, replace: bool = False
) -> Callable[[AdapterType], AdapterType]:
    """Decorator that binds an adapter class to validated metadata."""

    def decorate(adapter_type: AdapterType) -> AdapterType:
        if not issubclass(adapter_type, DatasetAdapter):
            raise TypeError("@dataset_adapter requires a DatasetAdapter subclass")
        adapter_type.METADATA = validate_metadata(metadata)
        DATASETS.register(adapter_type.METADATA, adapter_type, replace=replace)
        return adapter_type

    return decorate


def get_dataset_metadata(dataset_id_or_alias: str) -> DatasetMetadata:
    return DATASETS.metadata(dataset_id_or_alias)


def open_dataset(dataset_id_or_alias: str, **kwargs: Any) -> DatasetAdapter:
    return DATASETS.open(dataset_id_or_alias, **kwargs)


def list_datasets(*, minimum_support: SupportLevel | str | None = None) -> list[DatasetMetadata]:
    return DATASETS.list(minimum_support=minimum_support)
