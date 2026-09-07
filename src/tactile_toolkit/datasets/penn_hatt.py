"""Bounded local XML reader for Penn HaTT recorded exploration data."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from tactile_toolkit.datasets.base import DatasetAdapter
from tactile_toolkit.datasets.errors import DatasetAccessError, DatasetValidationError
from tactile_toolkit.datasets.metadata import (
    AccessKind,
    AccessSource,
    Citation,
    DatasetMetadata,
    LicenseInfo,
    ProvenanceInfo,
    SupportLevel,
)
from tactile_toolkit.datasets.registry import dataset_adapter
from tactile_toolkit.model import SampleKind, TactileObservation, TactileSample
from tactile_toolkit.types import Modality

PENN_HATT_REPORT_URL = (
    "https://repository.upenn.edu/bitstreams/960863b6-df14-4770-9068-b1b2bf6a50f1/download"
)

PENN_HATT_METADATA = DatasetMetadata(
    dataset_id="penn-hatt",
    name="Penn Haptic Texture Toolkit (HaTT)",
    description=(
        "A University of Pennsylvania collection of recorded explorations and data-driven "
        "models for rendering 100 homogeneous, isotropic surface textures."
    ),
    homepage=PENN_HATT_REPORT_URL,
    version="technical report dated 2014-02-12",
    provenance=ProvenanceInfo(
        publisher="University of Pennsylvania",
        source_url=PENN_HATT_REPORT_URL,
        creators=(
            "Heather Culbertson",
            "Juan José López Delgado",
            "Katherine J. Kuchenbecker",
        ),
        institutions=("University of Pennsylvania",),
        notes=(
            "The local reader implements only the recorded-data XML fields documented in the "
            "official technical report; model and rendering packages are not interpreted."
        ),
    ),
    license=LicenseInfo(
        name="Penn HaTT noncommercial research license",
        url=PENN_HATT_REPORT_URL,
        allows_redistribution=None,
        allows_commercial_use=False,
        requires_attribution=True,
        notes=(
            "The report permits attributed noncommercial research use and says its attached "
            "University of Pennsylvania license controls; it is not a standard open-data license."
        ),
    ),
    citations=(
        Citation(
            title=(
                "The Penn Haptic Texture Toolkit for Modeling, Rendering, and Evaluating "
                "Haptic Virtual Textures"
            ),
            url=PENN_HATT_REPORT_URL,
            authors=(
                "Heather Culbertson",
                "Juan José López Delgado",
                "Katherine J. Kuchenbecker",
            ),
            year=2014,
        ),
    ),
    access=(
        AccessSource(
            AccessKind.MANUAL,
            PENN_HATT_REPORT_URL,
            notes=(
                "The repository preserves the report and license, but a current official data "
                "archive and checksum manifest could not be verified; provide a lawful local copy."
            ),
        ),
    ),
    modalities=(Modality.VIBRATION, Modality.WRENCH, Modality.POSE, Modality.VISION),
    sensors=("three-axis accelerometer", "three-axis force sensor", "position sensor"),
    tasks=("haptic texture modeling", "texture rendering", "texture recognition"),
    formats=("xml", "image", "matlab", "native rendering models"),
    support_level=SupportLevel.LOADABLE,
    limitations=(
        "Use is restricted to attributed noncommercial research under the publisher's terms.",
        "The adapter reads recorded-data XML only; it does not parse or render friction and "
        "vibration models.",
        "No active official archive, immutable revision, complete manifest, size, or checksum "
        "was verified.",
        "The publisher does not define canonical machine-learning splits.",
    ),
    aliases=("hatt", "penn-haptic-texture-toolkit"),
)

_NUMBER = re.compile(r"[\s,;]+")
_DECLARATION = re.compile(rb"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)
_TAG_CLEANER = re.compile(r"[^a-z0-9]+")
_FAMILIES: dict[str, tuple[str, Modality, tuple[str, ...], tuple[str, ...]]] = {
    "acceleration": (
        "acceleration",
        Modality.VIBRATION,
        ("accel", "acceleration"),
        ("accelunits", "accelerationunits"),
    ),
    "force": ("force", Modality.WRENCH, ("force",), ("forceunits",)),
    "position": ("position", Modality.POSE, ("position",), ("positionunits",)),
}


def _tag(element: ET.Element) -> str:
    local = element.tag.rsplit("}", 1)[-1]
    return _TAG_CLEANER.sub("", local.lower())


def _numbers(element: ET.Element, context: str) -> np.ndarray:
    text = (element.text or "").strip()
    if not text:
        return np.empty(0, dtype=np.float64)
    try:
        values = np.asarray([float(item) for item in _NUMBER.split(text) if item], dtype=np.float64)
    except ValueError as exc:
        raise DatasetValidationError(f"Penn HaTT {context} contains a non-numeric value") from exc
    if not np.isfinite(values).all():
        raise DatasetValidationError(f"Penn HaTT {context} contains a non-finite value")
    return values


def _first_text(elements: list[ET.Element], names: tuple[str, ...]) -> str | None:
    wanted = set(names)
    for element in elements:
        if _tag(element) in wanted and (element.text or "").strip():
            return (element.text or "").strip()
    return None


def _sample_rate(elements: list[ET.Element], path: Path) -> float:
    value = _first_text(elements, ("samplerate", "samplingrate", "samplefrequency"))
    if value is None:
        raise DatasetValidationError(f"Penn HaTT recording has no SampleRate: {path}")
    try:
        sample_rate = float(value)
    except ValueError as exc:
        raise DatasetValidationError(f"Penn HaTT SampleRate is not numeric in {path}") from exc
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise DatasetValidationError(f"Penn HaTT SampleRate must be finite and positive in {path}")
    return sample_rate


def _component(
    elements: list[ET.Element], prefixes: tuple[str, ...], axis: str, context: str
) -> np.ndarray | None:
    compound = {f"{prefix}{axis}" for prefix in prefixes}
    compound_values = [
        _numbers(element, context) for element in elements if _tag(element) in compound
    ]
    compound_values = [array for array in compound_values if array.size]
    if compound_values:
        return np.concatenate(compound_values)
    nested: list[np.ndarray] = []
    for parent in elements:
        if _tag(parent) not in prefixes:
            continue
        for child in list(parent):
            if _tag(child) == axis:
                values = _numbers(child, context)
                if values.size:
                    nested.append(values)
    return np.concatenate(nested) if nested else None


def _vector(
    elements: list[ET.Element], prefixes: tuple[str, ...], context: str
) -> np.ndarray | None:
    components = [_component(elements, prefixes, axis, f"{context}.{axis}") for axis in "xyz"]
    present = [item is not None for item in components]
    if not any(present):
        return None
    if not all(present):
        raise DatasetValidationError(f"Penn HaTT {context} must provide x, y, and z together")
    vectors = [item for item in components if item is not None]
    lengths = {item.shape[0] for item in vectors}
    if len(lengths) != 1:
        raise DatasetValidationError(f"Penn HaTT {context} component lengths do not match")
    return np.column_stack(vectors)


def _scalar(elements: list[ET.Element], names: tuple[str, ...], context: str) -> np.ndarray | None:
    values = [_numbers(element, context) for element in elements if _tag(element) in names]
    values = [array for array in values if array.size]
    return np.concatenate(values)[:, None] if values else None


def _material_id(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    candidate = relative.parent.name if relative.parent != Path(".") else path.stem
    value = candidate.strip()
    if not value:
        raise DatasetValidationError(f"Could not derive Penn HaTT material identity from {path}")
    return value


@dataset_adapter(PENN_HATT_METADATA)
class PennHattAdapter(DatasetAdapter):
    """Yield one normalized sequence per documented HaTT recorded-data XML file."""

    def __init__(self, root: str | Path | None = None, *, max_xml_bytes: int = 64_000_000):
        super().__init__(root)
        if max_xml_bytes <= 0:
            raise ValueError("max_xml_bytes must be positive")
        self.max_xml_bytes = max_xml_bytes

    def _parse(self, path: Path, root: Path) -> TactileSample | None:
        size = path.stat().st_size
        if size > self.max_xml_bytes:
            raise DatasetAccessError(
                f"Penn HaTT XML exceeds max_xml_bytes ({size} > {self.max_xml_bytes}): {path}"
            )
        payload = path.read_bytes()
        if _DECLARATION.search(payload):
            raise DatasetValidationError(
                f"Penn HaTT XML declarations for DTD/entities are not allowed: {path}"
            )
        try:
            document = ET.fromstring(payload)
        except ET.ParseError as exc:
            raise DatasetValidationError(f"Penn HaTT XML is malformed: {path}: {exc}") from exc
        elements = list(document.iter())

        candidates: list[tuple[str, Modality, np.ndarray | None, str | None, tuple[str, ...]]] = []
        for key, (name, modality, prefixes, unit_names) in _FAMILIES.items():
            candidates.append(
                (
                    name,
                    modality,
                    _vector(elements, prefixes, key),
                    _first_text(elements, unit_names),
                    prefixes,
                )
            )
        candidates.extend(
            (
                (
                    "acceleration.combined",
                    Modality.VIBRATION,
                    _scalar(elements, ("accel", "acceleration"), "acceleration.combined"),
                    _first_text(elements, ("accelunits", "accelerationunits")),
                    (),
                ),
                (
                    "force.normal_tangential",
                    Modality.WRENCH,
                    None,
                    _first_text(elements, ("forceunits",)),
                    (),
                ),
                (
                    "speed",
                    Modality.POSE,
                    _scalar(elements, ("speed",), "speed"),
                    _first_text(elements, ("speedunits",)),
                    (),
                ),
            )
        )
        normal = _scalar(elements, ("forcenormal",), "force.normal")
        tangential = _scalar(elements, ("forcetangential",), "force.tangential")
        if (normal is None) != (tangential is None):
            raise DatasetValidationError(
                f"Penn HaTT force normal and tangential channels must occur together: {path}"
            )
        if normal is not None and tangential is not None:
            if normal.shape[0] != tangential.shape[0]:
                raise DatasetValidationError(
                    f"Penn HaTT force channel lengths do not match: {path}"
                )
            for index, candidate in enumerate(candidates):
                if candidate[0] == "force.normal_tangential":
                    candidates[index] = (
                        *candidate[:2],
                        np.column_stack((normal[:, 0], tangential[:, 0])),
                        *candidate[3:],
                    )
                    break

        present = [candidate for candidate in candidates if candidate[2] is not None]
        if not present:
            return None
        sample_rate = _sample_rate(elements, path)
        observations: dict[str, TactileObservation] = {}
        for name, modality, values, unit, _ in present:
            assert values is not None
            timestamps = np.arange(values.shape[0], dtype=np.float64) / sample_rate
            observations[name] = TactileObservation(
                modality,
                values,
                sensor="Penn HaTT recording apparatus",
                timestamps=timestamps,
                unit=unit,
                metadata={"sample_rate_hz": sample_rate},
            )

        relative = path.relative_to(root).as_posix()
        material = _material_id(path, root)
        return TactileSample(
            sample_id=relative,
            observations=observations,
            kind=SampleKind.SEQUENCE,
            labels={"material": material},
            task="haptic texture modeling",
            group_id=relative.rsplit(".", 1)[0],
            metadata={
                "source_path": relative,
                "source_format": "Penn HaTT recorded-data XML",
                "license": PENN_HATT_METADATA.license.name,
            },
        )

    def iter_samples(self, *, split: str | None = None) -> Iterator[TactileSample]:
        if split is not None:
            raise DatasetAccessError("Penn HaTT does not publish canonical machine-learning splits")
        root = self.require_root()
        if not root.is_dir():
            raise DatasetAccessError(f"Penn HaTT root must be a directory: {root}")
        found = False
        for path in sorted(root.rglob("*.xml")):
            if not path.is_file():
                continue
            try:
                path.resolve().relative_to(root.resolve())
            except ValueError as exc:
                raise DatasetValidationError(
                    f"Penn HaTT XML path escapes the dataset root: {path}"
                ) from exc
            sample = self._parse(path, root)
            if sample is not None:
                found = True
                yield sample
        if not found:
            raise DatasetAccessError(
                f"No documented Penn HaTT recorded-data XML files were found below {root}"
            )
