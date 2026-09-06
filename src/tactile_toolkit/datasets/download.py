"""Explicit, resumable HTTP acquisition and local integrity verification."""

from __future__ import annotations

import hashlib
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from tactile_toolkit.datasets.errors import DatasetAccessError, DatasetIntegrityError

_DATASET_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+|\*)$")
_DEFAULT_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class IntegrityReport:
    """Expected and observed integrity facts for one local file."""

    path: Path
    actual_size_bytes: int | None
    actual_sha256: str | None
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if self.errors:
            details = "; ".join(self.errors)
            raise DatasetIntegrityError(f"Integrity check failed for {self.path}: {details}")


def _expected_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("sha256 must be a string")
    checksum = value.lower()
    if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
        raise ValueError("sha256 must contain 64 hexadecimal characters")
    return checksum


def _expected_size(value: int | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("size_bytes must be an integer")
    if value < 0:
        raise ValueError("size_bytes must be non-negative")
    return value


def _promote(partial: Path, target: Path) -> None:
    try:
        os.replace(partial, target)
    except OSError as exc:
        raise DatasetAccessError(f"Could not finalize cache file {target}: {exc}") from exc


def sha256_file(path: str | Path, *, chunk_size: int = _DEFAULT_CHUNK_BYTES) -> str:
    """Hash ``path`` incrementally without loading it into memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def verify_file(
    path: str | Path,
    *,
    sha256: str | None = None,
    size_bytes: int | None = None,
    chunk_size: int = _DEFAULT_CHUNK_BYTES,
) -> IntegrityReport:
    """Validate existence, optional byte size, and optional SHA-256 digest."""
    expected_sha256 = _expected_sha256(sha256)
    size_bytes = _expected_size(size_bytes)

    target = Path(path)
    if not target.is_file():
        return IntegrityReport(
            target, None, None, ("file does not exist or is not a regular file",)
        )

    actual_size = target.stat().st_size
    errors: list[str] = []
    if size_bytes is not None and actual_size != size_bytes:
        errors.append(f"expected {size_bytes} bytes, found {actual_size}")

    actual_sha256 = sha256_file(target, chunk_size=chunk_size) if expected_sha256 else None
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        errors.append(f"expected SHA-256 {expected_sha256}, found {actual_sha256}")
    return IntegrityReport(target, actual_size, actual_sha256, tuple(errors))


def default_cache_dir() -> Path:
    """Return the platform-appropriate cache root without creating it."""
    explicit = os.environ.get("TACTILE_TOOLKIT_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "tactile-toolkit"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "tactile-toolkit"
    return Path.home() / ".cache" / "tactile-toolkit"


def _safe_dataset_id(dataset_id: str) -> str:
    value = dataset_id.strip().lower()
    if not _DATASET_COMPONENT.fullmatch(value):
        raise ValueError(
            "dataset_id must contain only lowercase letters, digits, dots, underscores, or hyphens"
        )
    return value


def _safe_filename(filename: str) -> str:
    value = filename.strip()
    if not value or value in {".", ".."}:
        raise ValueError("download filename must not be empty")
    if Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError("download filename must be a single safe path component")
    return value


def _url_filename(url: str) -> str:
    value = Path(unquote(urlparse(url).path)).name
    if not value:
        raise DatasetAccessError("Download URL has no filename; pass filename= explicitly")
    return _safe_filename(value)


ProgressCallback = Callable[[int, int | None], None]


class DatasetCache:
    """Namespaced cache with verified, atomic downloads and reusable partial files."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root).expanduser() if root is not None else default_cache_dir()

    def dataset_dir(self, dataset_id: str) -> Path:
        path = self.root / _safe_dataset_id(dataset_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def path_for(self, dataset_id: str, filename: str) -> Path:
        return self.dataset_dir(dataset_id) / _safe_filename(filename)

    def download(
        self,
        url: str,
        dataset_id: str,
        *,
        filename: str | None = None,
        sha256: str | None = None,
        size_bytes: int | None = None,
        resume: bool = True,
        timeout_s: float = 30.0,
        chunk_size: int = _DEFAULT_CHUNK_BYTES,
        progress: ProgressCallback | None = None,
    ) -> Path:
        """Download one HTTP(S) asset, retaining ``.part`` data after interruptions."""
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise DatasetAccessError("Automatic downloads require an absolute HTTP(S) URL")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        expected_sha256 = _expected_sha256(sha256)
        size_bytes = _expected_size(size_bytes)

        target = self.path_for(dataset_id, filename or _url_filename(url))
        if target.exists():
            existing = verify_file(
                target,
                sha256=expected_sha256,
                size_bytes=size_bytes,
                chunk_size=chunk_size,
            )
            if existing.ok:
                return target
            existing.raise_if_invalid()

        partial = target.with_name(f"{target.name}.part")
        offset = partial.stat().st_size if resume and partial.is_file() else 0
        headers = {"User-Agent": "tactile-toolkit/0.1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(url, headers=headers)

        try:
            with urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - scheme checked
                final_url = response.geturl()
                if urlparse(final_url).scheme not in {"http", "https"}:
                    raise DatasetAccessError(
                        f"Download redirected to unsupported URL scheme: {final_url}"
                    )
                status = getattr(response, "status", response.getcode())
                if status not in {200, 206}:
                    raise DatasetAccessError(f"Download returned HTTP {status}: {url}")

                mode = "wb"
                written = 0
                if offset and status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = _CONTENT_RANGE.fullmatch(content_range)
                    if match is None or int(match.group(1)) != offset:
                        raise DatasetAccessError(
                            "Server returned an invalid Content-Range for resume: "
                            f"{content_range!r}"
                        )
                    mode = "ab"
                    written = offset
                elif offset:
                    # The server ignored Range. Restart this cache-owned partial file safely.
                    mode = "wb"

                total = size_bytes
                if total is None:
                    content_length = response.headers.get("Content-Length")
                    if content_length and content_length.isdigit():
                        total = written + int(content_length)
                with partial.open(mode) as handle:
                    while block := response.read(chunk_size):
                        handle.write(block)
                        written += len(block)
                        if progress is not None:
                            progress(written, total)
        except HTTPError as exc:
            if (
                exc.code == 416
                and offset
                and (expected_sha256 is not None or size_bytes is not None)
            ):
                completed = verify_file(
                    partial,
                    sha256=expected_sha256,
                    size_bytes=size_bytes,
                    chunk_size=chunk_size,
                )
                if completed.ok:
                    _promote(partial, target)
                    return target
            raise DatasetAccessError(f"Download failed with HTTP {exc.code}: {url}") from exc
        except URLError as exc:
            raise DatasetAccessError(f"Download failed for {url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise DatasetAccessError(f"Download timed out for {url}") from exc
        except OSError as exc:
            raise DatasetAccessError(
                f"Dataset download I/O failed for {url} using {partial}: {exc}"
            ) from exc

        integrity = verify_file(
            partial,
            sha256=expected_sha256,
            size_bytes=size_bytes,
            chunk_size=chunk_size,
        )
        integrity.raise_if_invalid()
        _promote(partial, target)
        return target
