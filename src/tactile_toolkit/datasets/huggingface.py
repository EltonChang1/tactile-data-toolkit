"""Explicit, revision-pinned acquisition from the Hugging Face Hub."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tactile_toolkit.datasets.errors import DatasetAccessError

_LARGE_DOWNLOAD_BYTES = 10_000_000_000


@dataclass(frozen=True)
class HuggingFaceSnapshot:
    """A publisher-controlled dataset snapshot with an immutable revision."""

    repo_id: str
    revision: str
    approximate_size_bytes: int
    browser_url: str

    def __post_init__(self) -> None:
        if self.repo_id.count("/") != 1 or any(
            part in {"", ".", ".."} for part in self.repo_id.split("/")
        ):
            raise ValueError("repo_id must have the form 'owner/dataset'")
        if not self.revision or not self.revision.strip():
            raise ValueError("revision must not be empty")
        if self.approximate_size_bytes < 0:
            raise ValueError("approximate_size_bytes must be non-negative")
        if not self.browser_url.startswith("https://huggingface.co/datasets/"):
            raise ValueError("browser_url must be an official Hugging Face dataset URL")

    @property
    def approximate_size_gb(self) -> float:
        """Return the publisher snapshot size in decimal gigabytes."""
        return self.approximate_size_bytes / 1_000_000_000

    def download(
        self,
        local_dir: str | Path,
        *,
        allow_patterns: str | Sequence[str] | None = None,
        token: str | bool | None = None,
        confirm_large_download: bool = False,
        **kwargs: Any,
    ) -> Path:
        """Download through the official client after an explicit large-data confirmation.

        ``allow_patterns`` can restrict FreeTacMan to selected task directories. FoTa's
        multi-volume ZIP requires every data volume, so partial pattern sets are not useful.
        """
        if self.approximate_size_bytes >= _LARGE_DOWNLOAD_BYTES and not confirm_large_download:
            raise DatasetAccessError(
                f"{self.repo_id} is approximately {self.approximate_size_gb:.1f} GB; inspect "
                "the official dataset page and pass confirm_large_download=True to continue"
            )
        destination = Path(local_dir).expanduser()
        if destination.exists() and not destination.is_dir():
            raise DatasetAccessError(f"Hugging Face local_dir is not a directory: {destination}")

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise DatasetAccessError(
                "Hugging Face acquisition requires the optional official client; install "
                "huggingface-hub>=0.24"
            ) from exc

        try:
            result = snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                revision=self.revision,
                local_dir=destination,
                allow_patterns=allow_patterns,
                token=token,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - normalize optional client failures
            raise DatasetAccessError(
                f"Official Hugging Face download failed for {self.repo_id}@{self.revision}: {exc}"
            ) from exc
        return Path(result)
