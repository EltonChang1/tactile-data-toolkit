"""Versioned metadata schema for published tactile datasets."""

from __future__ import annotations

import re
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any
from urllib.parse import urlparse

from tactile_toolkit.datasets.errors import DatasetMetadataError
from tactile_toolkit.types import Modality

DATASET_METADATA_VERSION = "tactile-dataset-metadata/0.1"
_DATASET_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_REMOTE_SCHEMES = frozenset({"http", "https", "s3", "gs", "hf"})


class SupportLevel(StrEnum):
    """Highest user-visible capability currently implemented for a dataset."""

    CATALOGED = "cataloged"
    DISCOVERABLE = "discoverable"
    LOADABLE = "loadable"
    CONVERTIBLE = "convertible"
    BENCHMARK_READY = "benchmark_ready"

    @classmethod
    def parse(cls, value: str | SupportLevel) -> SupportLevel:
        if isinstance(value, SupportLevel):
            return value
        if not isinstance(value, str):
            raise DatasetMetadataError("support_level must be a string")
        try:
            return cls(value)
        except ValueError:
            try:
                return cls[value.upper()]
            except KeyError as exc:
                choices = ", ".join(item.value for item in cls)
                raise DatasetMetadataError(
                    f"Unknown support level {value!r}; expected one of: {choices}"
                ) from exc


class AccessKind(StrEnum):
    """Mechanism through which the publisher exposes a dataset."""

    HTTP = "http"
    HUGGING_FACE = "hugging_face"
    DATAVERSE = "dataverse"
    GOOGLE_DRIVE = "google_drive"
    MANUAL = "manual"
    REFERENCE = "reference"

    @classmethod
    def parse(cls, value: str | AccessKind) -> AccessKind:
        if isinstance(value, AccessKind):
            return value
        if not isinstance(value, str):
            raise DatasetMetadataError("access.kind must be a string")
        try:
            return cls(value)
        except ValueError:
            try:
                return cls[value.upper()]
            except KeyError as exc:
                choices = ", ".join(item.value for item in cls)
                raise DatasetMetadataError(
                    f"Unknown access kind {value!r}; expected one of: {choices}"
                ) from exc


class ResourceKind(StrEnum):
    """Nature of a catalog record, independent of its implemented support level."""

    DATASET = "dataset"
    COLLECTION = "collection"
    REFERENCE_INDEX = "reference_index"

    @classmethod
    def parse(cls, value: str | ResourceKind) -> ResourceKind:
        if isinstance(value, ResourceKind):
            return value
        if not isinstance(value, str):
            raise DatasetMetadataError("resource_kind must be a string")
        try:
            return cls(value)
        except ValueError:
            try:
                return cls[value.upper()]
            except KeyError as exc:
                choices = ", ".join(item.value for item in cls)
                raise DatasetMetadataError(
                    f"Unknown resource kind {value!r}; expected one of: {choices}"
                ) from exc


def _text(value: Any, field_name: str, *, required: bool = True) -> str | None:
    if value is None:
        if required:
            raise DatasetMetadataError(f"{field_name} is required")
        return None
    if not isinstance(value, str):
        raise DatasetMetadataError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        if required:
            raise DatasetMetadataError(f"{field_name} must not be empty")
        return None
    return cleaned


def _url(value: Any, field_name: str, *, required: bool = True) -> str | None:
    cleaned = _text(value, field_name, required=required)
    if cleaned is None:
        return None
    parsed = urlparse(cleaned)
    if parsed.scheme.lower() not in _REMOTE_SCHEMES or not parsed.netloc:
        raise DatasetMetadataError(f"{field_name} must be an absolute HTTP(S), S3, GS, or HF URL")
    return cleaned


def _strings(values: Sequence[Any] | None, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise DatasetMetadataError(f"{field_name} must be a sequence of strings")
    result: list[str] = []
    for index, value in enumerate(values):
        cleaned = _text(value, f"{field_name}[{index}]")
        assert cleaned is not None
        if cleaned not in result:
            result.append(cleaned)
    return tuple(result)


def _permissions(value: Any, field_name: str) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise DatasetMetadataError(f"{field_name} must be true, false, or null when unknown")


@dataclass(frozen=True)
class LicenseInfo:
    """Dataset license facts, kept separate from code and model licenses."""

    name: str
    spdx_id: str | None = None
    url: str | None = None
    allows_redistribution: bool | None = None
    allows_commercial_use: bool | None = None
    requires_attribution: bool | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "license.name"))
        object.__setattr__(self, "spdx_id", _text(self.spdx_id, "license.spdx_id", required=False))
        object.__setattr__(self, "url", _url(self.url, "license.url", required=False))
        object.__setattr__(self, "notes", _text(self.notes, "license.notes", required=False))
        for name in (
            "allows_redistribution",
            "allows_commercial_use",
            "requires_attribution",
        ):
            object.__setattr__(self, name, _permissions(getattr(self, name), f"license.{name}"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LicenseInfo:
        if not isinstance(value, Mapping):
            raise DatasetMetadataError("license must be an object")
        name = _text(value.get("name"), "license.name")
        assert name is not None
        return cls(
            name=name,
            spdx_id=value.get("spdx_id"),
            url=value.get("url"),
            allows_redistribution=value.get("allows_redistribution"),
            allows_commercial_use=value.get("allows_commercial_use"),
            requires_attribution=value.get("requires_attribution"),
            notes=value.get("notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "spdx_id": self.spdx_id,
            "url": self.url,
            "allows_redistribution": self.allows_redistribution,
            "allows_commercial_use": self.allows_commercial_use,
            "requires_attribution": self.requires_attribution,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class Citation:
    """A primary paper, dataset DOI, or official repository citation."""

    title: str
    url: str | None = None
    authors: tuple[str, ...] = ()
    year: int | None = None
    doi: str | None = None
    bibtex: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "title", _text(self.title, "citation.title"))
        object.__setattr__(self, "url", _url(self.url, "citation.url", required=False))
        object.__setattr__(self, "authors", _strings(self.authors, "citation.authors"))
        object.__setattr__(self, "doi", _text(self.doi, "citation.doi", required=False))
        object.__setattr__(self, "bibtex", _text(self.bibtex, "citation.bibtex", required=False))
        if self.year is not None:
            if not isinstance(self.year, int) or isinstance(self.year, bool):
                raise DatasetMetadataError("citation.year must be an integer")
            if not 1900 <= self.year <= 2200:
                raise DatasetMetadataError("citation.year must be between 1900 and 2200")
        if self.url is None and self.doi is None:
            raise DatasetMetadataError("citation must provide at least one of url or doi")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Citation:
        if not isinstance(value, Mapping):
            raise DatasetMetadataError("each citation must be an object")
        title = _text(value.get("title"), "citation.title")
        assert title is not None
        return cls(
            title=title,
            url=value.get("url"),
            authors=value.get("authors", ()),
            year=value.get("year"),
            doi=value.get("doi"),
            bibtex=value.get("bibtex"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "authors": list(self.authors),
            "year": self.year,
            "doi": self.doi,
            "bibtex": self.bibtex,
        }


@dataclass(frozen=True)
class ProvenanceInfo:
    """Publisher, origin, and derivation facts independent of citation text."""

    publisher: str
    source_url: str
    creators: tuple[str, ...] = ()
    institutions: tuple[str, ...] = ()
    derived_from: tuple[str, ...] = ()
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "publisher", _text(self.publisher, "provenance.publisher"))
        object.__setattr__(self, "source_url", _url(self.source_url, "provenance.source_url"))
        object.__setattr__(self, "creators", _strings(self.creators, "provenance.creators"))
        object.__setattr__(
            self, "institutions", _strings(self.institutions, "provenance.institutions")
        )
        object.__setattr__(
            self, "derived_from", _strings(self.derived_from, "provenance.derived_from")
        )
        object.__setattr__(self, "notes", _text(self.notes, "provenance.notes", required=False))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProvenanceInfo:
        if not isinstance(value, Mapping):
            raise DatasetMetadataError("provenance must be an object")
        publisher = _text(value.get("publisher"), "provenance.publisher")
        source_url = _url(value.get("source_url"), "provenance.source_url")
        assert publisher is not None and source_url is not None
        return cls(
            publisher=publisher,
            source_url=source_url,
            creators=value.get("creators", ()),
            institutions=value.get("institutions", ()),
            derived_from=value.get("derived_from", ()),
            notes=value.get("notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "publisher": self.publisher,
            "source_url": self.source_url,
            "creators": list(self.creators),
            "institutions": list(self.institutions),
            "derived_from": list(self.derived_from),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class AccessSource:
    """One publisher-controlled route to data or access instructions."""

    kind: AccessKind
    url: str
    requires_authentication: bool = False
    requires_acceptance: bool = False
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", AccessKind.parse(self.kind))
        object.__setattr__(self, "url", _url(self.url, "access.url"))
        object.__setattr__(self, "notes", _text(self.notes, "access.notes", required=False))
        if not isinstance(self.requires_authentication, bool):
            raise DatasetMetadataError("access.requires_authentication must be boolean")
        if not isinstance(self.requires_acceptance, bool):
            raise DatasetMetadataError("access.requires_acceptance must be boolean")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AccessSource:
        if not isinstance(value, Mapping):
            raise DatasetMetadataError("each access source must be an object")
        raw_kind = _text(value.get("kind"), "access.kind")
        assert raw_kind is not None
        kind = AccessKind.parse(raw_kind)
        url = _url(value.get("url"), "access.url")
        assert url is not None
        return cls(
            kind=kind,
            url=url,
            requires_authentication=value.get("requires_authentication", False),
            requires_acceptance=value.get("requires_acceptance", False),
            notes=value.get("notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "url": self.url,
            "requires_authentication": self.requires_authentication,
            "requires_acceptance": self.requires_acceptance,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class DatasetMetadata:
    """Validated identity and provenance for a dataset, collection, or reference index."""

    dataset_id: str
    name: str
    description: str
    homepage: str
    provenance: ProvenanceInfo
    license: LicenseInfo
    citations: tuple[Citation, ...]
    access: tuple[AccessSource, ...]
    modalities: tuple[Modality, ...] = ()
    sensors: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    formats: tuple[str, ...] = ()
    support_level: SupportLevel = SupportLevel.CATALOGED
    version: str | None = None
    revision: str | None = None
    approximate_size_bytes: int | None = None
    checksums: Mapping[str, str] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    resource_kind: ResourceKind = ResourceKind.DATASET

    def __post_init__(self) -> None:
        dataset_id = _text(self.dataset_id, "dataset_id")
        assert dataset_id is not None
        if not _DATASET_ID.fullmatch(dataset_id):
            raise DatasetMetadataError(
                "dataset_id must start with a lowercase letter or digit and contain only "
                "lowercase letters, digits, dots, underscores, or hyphens"
            )
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "name", _text(self.name, "name"))
        object.__setattr__(self, "description", _text(self.description, "description"))
        object.__setattr__(self, "homepage", _url(self.homepage, "homepage"))
        if not isinstance(self.provenance, ProvenanceInfo):
            raise DatasetMetadataError("provenance must be a ProvenanceInfo")
        if not isinstance(self.license, LicenseInfo):
            raise DatasetMetadataError("license must be a LicenseInfo")
        if not self.citations or not all(isinstance(item, Citation) for item in self.citations):
            raise DatasetMetadataError("citations must contain at least one Citation")
        if not self.access or not all(isinstance(item, AccessSource) for item in self.access):
            raise DatasetMetadataError("access must contain at least one AccessSource")
        object.__setattr__(self, "citations", tuple(self.citations))
        object.__setattr__(self, "access", tuple(self.access))

        if isinstance(self.modalities, str) or not isinstance(self.modalities, Sequence):
            raise DatasetMetadataError("modalities must be a sequence of strings")
        parsed_modalities: list[Modality] = []
        for value in self.modalities:
            try:
                modality = Modality.parse(value)
            except (KeyError, ValueError) as exc:
                raise DatasetMetadataError(f"Unknown modality {value!r}") from exc
            if modality is Modality.UNKNOWN:
                raise DatasetMetadataError("metadata modalities must not contain 'unknown'")
            if modality not in parsed_modalities:
                parsed_modalities.append(modality)
        object.__setattr__(self, "modalities", tuple(parsed_modalities))
        object.__setattr__(self, "sensors", _strings(self.sensors, "sensors"))
        object.__setattr__(self, "tasks", _strings(self.tasks, "tasks"))
        object.__setattr__(self, "formats", _strings(self.formats, "formats"))
        object.__setattr__(self, "resource_kind", ResourceKind.parse(self.resource_kind))
        object.__setattr__(self, "limitations", _strings(self.limitations, "limitations"))
        aliases = tuple(dict.fromkeys(alias.lower() for alias in _strings(self.aliases, "aliases")))
        aliases = tuple(alias for alias in aliases if alias != self.dataset_id)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "support_level", SupportLevel.parse(self.support_level))
        object.__setattr__(self, "version", _text(self.version, "version", required=False))
        object.__setattr__(self, "revision", _text(self.revision, "revision", required=False))

        if self.approximate_size_bytes is not None:
            if not isinstance(self.approximate_size_bytes, int) or isinstance(
                self.approximate_size_bytes, bool
            ):
                raise DatasetMetadataError("approximate_size_bytes must be an integer")
            if self.approximate_size_bytes < 0:
                raise DatasetMetadataError("approximate_size_bytes must be non-negative")

        normalized_checksums: dict[str, str] = {}
        if not isinstance(self.checksums, Mapping):
            raise DatasetMetadataError("checksums must be an object mapping asset names to SHA-256")
        for raw_name, raw_checksum in self.checksums.items():
            name = _text(raw_name, "checksums key")
            checksum = _text(raw_checksum, f"checksums[{name!r}]")
            assert name is not None and checksum is not None
            checksum = checksum.lower()
            if len(checksum) != 64 or any(char not in string.hexdigits for char in checksum):
                raise DatasetMetadataError(f"checksums[{name!r}] must be a SHA-256 hex digest")
            normalized_checksums[name] = checksum
        object.__setattr__(self, "checksums", MappingProxyType(normalized_checksums))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DatasetMetadata:
        """Validate a JSON-compatible metadata object."""
        if not isinstance(value, Mapping):
            raise DatasetMetadataError("dataset metadata must be an object")
        version = value.get("schema_version", DATASET_METADATA_VERSION)
        if version != DATASET_METADATA_VERSION:
            raise DatasetMetadataError(
                f"Unsupported dataset metadata schema {version!r}; "
                f"expected {DATASET_METADATA_VERSION!r}"
            )
        try:
            provenance = ProvenanceInfo.from_dict(value["provenance"])
            license_info = LicenseInfo.from_dict(value["license"])
            citations = tuple(Citation.from_dict(item) for item in value["citations"])
            access = tuple(AccessSource.from_dict(item) for item in value["access"])
        except KeyError as exc:
            raise DatasetMetadataError(f"Missing required metadata field: {exc.args[0]}") from exc
        except TypeError as exc:
            raise DatasetMetadataError("citations and access must be arrays") from exc
        dataset_id = _text(value.get("dataset_id"), "dataset_id")
        name = _text(value.get("name"), "name")
        description = _text(value.get("description"), "description")
        homepage = _url(value.get("homepage"), "homepage")
        assert dataset_id is not None
        assert name is not None
        assert description is not None
        assert homepage is not None
        return cls(
            dataset_id=dataset_id,
            name=name,
            description=description,
            homepage=homepage,
            provenance=provenance,
            license=license_info,
            citations=citations,
            access=access,
            modalities=value.get("modalities", ()),
            sensors=value.get("sensors", ()),
            tasks=value.get("tasks", ()),
            formats=value.get("formats", ()),
            resource_kind=value.get("resource_kind", ResourceKind.DATASET),
            support_level=value.get("support_level", SupportLevel.CATALOGED),
            version=value.get("version"),
            revision=value.get("revision"),
            approximate_size_bytes=value.get("approximate_size_bytes"),
            checksums=value.get("checksums", {}),
            limitations=value.get("limitations", ()),
            aliases=value.get("aliases", ()),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return metadata in the stable JSON-compatible interchange schema."""
        return {
            "schema_version": DATASET_METADATA_VERSION,
            "dataset_id": self.dataset_id,
            "name": self.name,
            "description": self.description,
            "homepage": self.homepage,
            "provenance": self.provenance.to_dict(),
            "version": self.version,
            "revision": self.revision,
            "license": self.license.to_dict(),
            "citations": [citation.to_dict() for citation in self.citations],
            "access": [source.to_dict() for source in self.access],
            "modalities": [modality.value for modality in self.modalities],
            "sensors": list(self.sensors),
            "tasks": list(self.tasks),
            "formats": list(self.formats),
            "resource_kind": self.resource_kind.value,
            "support_level": self.support_level.value,
            "approximate_size_bytes": self.approximate_size_bytes,
            "checksums": dict(self.checksums),
            "limitations": list(self.limitations),
            "aliases": list(self.aliases),
        }


def validate_metadata(value: DatasetMetadata | Mapping[str, Any]) -> DatasetMetadata:
    """Return a validated metadata instance or raise :class:`DatasetMetadataError`."""
    return value if isinstance(value, DatasetMetadata) else DatasetMetadata.from_dict(value)
