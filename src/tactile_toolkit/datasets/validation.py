"""Bounded metadata, sample, and local-asset validation for dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

from tactile_toolkit.datasets.download import verify_file
from tactile_toolkit.datasets.errors import DatasetValidationError
from tactile_toolkit.model import AssetReference, TactileSample

if TYPE_CHECKING:
    from tactile_toolkit.datasets.base import DatasetAdapter


@dataclass
class DatasetValidationReport:
    """Actionable result of validating a bounded adapter sample."""

    dataset_id: str
    checked_samples: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        status = "OK" if self.ok else "FAILED"
        lines = [
            f"dataset {self.dataset_id}: {status}",
            f"  checked samples: {self.checked_samples}",
        ]
        for label, items in (("errors", self.errors), ("warnings", self.warnings)):
            if items:
                lines.append(f"  {label}:")
                lines.extend(f"    - {item}" for item in items)
        return "\n".join(lines)

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise DatasetValidationError(self.summary())


def _local_asset_path(
    reference: AssetReference, root: Path | None
) -> tuple[Path | None, str | None]:
    parsed = urlparse(reference.uri)
    if parsed.scheme and parsed.scheme != "file":
        return None, None
    if parsed.scheme == "file":
        candidate = Path(unquote(parsed.path))
    else:
        candidate = Path(reference.uri)
        if not candidate.is_absolute():
            if root is None:
                return None, "relative asset has no adapter root"
            candidate = root / candidate

    if root is not None:
        resolved_root = root.resolve()
        resolved_candidate = candidate.resolve()
        if not resolved_candidate.is_relative_to(resolved_root):
            return None, f"asset escapes adapter root: {reference.uri}"
        candidate = resolved_candidate
    return candidate, None


def _validate_assets(
    sample: TactileSample, root: Path | None, report: DatasetValidationReport
) -> None:
    for name, observation in sample.observations.items():
        if not isinstance(observation.data, AssetReference):
            continue
        path, error = _local_asset_path(observation.data, root)
        if error is not None:
            report.errors.append(f"sample {sample.sample_id} observation {name}: {error}")
            continue
        if path is None:
            continue
        # For a container reference these values describe the selected member, not the outer
        # TAR/HDF5/NPZ file. The adapter validates member metadata while enumerating; this pass
        # still verifies that the local container exists and remains inside the adapter root.
        member_reference = observation.data.member is not None
        integrity = verify_file(
            path,
            sha256=None if member_reference else observation.data.sha256,
            size_bytes=None if member_reference else observation.data.size_bytes,
        )
        for item in integrity.errors:
            report.errors.append(f"sample {sample.sample_id} observation {name}: {item}")


def validate_adapter(
    adapter: DatasetAdapter,
    *,
    split: str | None = None,
    limit: int = 32,
    check_assets: bool = False,
) -> DatasetValidationReport:
    """Validate metadata and at most ``limit`` lazy samples from ``adapter``."""
    if limit <= 0:
        raise ValueError("validation limit must be positive")

    metadata = adapter.metadata
    report = DatasetValidationReport(metadata.dataset_id)
    declared_modalities = set(metadata.modalities)
    seen_ids: set[str] = set()

    try:
        iterator = adapter.iter_samples(split=split)
        for position, sample in enumerate(islice(iterator, limit)):
            report.checked_samples = position + 1
            if not isinstance(sample, TactileSample):
                report.errors.append(
                    f"sample {position}: adapter yielded {type(sample).__name__}, "
                    "expected TactileSample"
                )
                continue
            if sample.sample_id in seen_ids:
                report.errors.append(f"duplicate sample_id in checked prefix: {sample.sample_id}")
            seen_ids.add(sample.sample_id)

            undeclared = sample.modalities - declared_modalities
            if declared_modalities and undeclared:
                values = ", ".join(sorted(modality.value for modality in undeclared))
                report.errors.append(
                    f"sample {sample.sample_id} uses undeclared modalities: {values}"
                )
            if split is not None:
                if sample.split is None:
                    report.warnings.append(
                        f"sample {sample.sample_id} has no split while validating split {split!r}"
                    )
                elif sample.split != split:
                    report.errors.append(
                        f"sample {sample.sample_id} belongs to split {sample.split!r}, "
                        f"not {split!r}"
                    )
            if check_assets:
                _validate_assets(sample, adapter.root, report)
    except Exception as exc:  # noqa: BLE001 - adapter failures belong in the report
        report.errors.append(f"adapter iteration failed: {type(exc).__name__}: {exc}")

    if report.checked_samples == 0 and not report.errors:
        report.warnings.append("adapter yielded no samples in the checked selection")
    return report
