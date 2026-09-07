"""Verified acquisition and guarded local reader for Mendeley tactile textures V1."""

from __future__ import annotations

import pickle
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from tactile_toolkit.datasets.base import DatasetAdapter
from tactile_toolkit.datasets.download import DatasetCache
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

MENDELEY_TEXTURES_DOI = "10.17632/n666tk4mw9.1"
MENDELEY_TEXTURES_FILENAME = "multimodal-tactile-texture-dataset.zip"
MENDELEY_TEXTURES_SIZE_BYTES = 3_831_798_836
MENDELEY_TEXTURES_SHA256 = "56468a6bc7191b46e12f09fece605d117bf29e8c678ae458b885e01a41917572"
MENDELEY_TEXTURES_URL = (
    "https://data.mendeley.com/public-files/datasets/n666tk4mw9/files/"
    "70c5500e-2ef3-40af-a8f4-d4eac7bb3119/file_downloaded"
)
MENDELEY_NOTEBOOK_FILENAME = "reading-pickle-file.ipynb"
MENDELEY_NOTEBOOK_SIZE_BYTES = 29_865
MENDELEY_NOTEBOOK_SHA256 = "aaa43ffab84767e3b0589be84c64e9135a90fc9f60c044293c354a2518992176"
MENDELEY_NOTEBOOK_URL = (
    "https://data.mendeley.com/public-files/datasets/n666tk4mw9/files/"
    "1601963b-89ed-4b28-beb9-672b20e59040/file_downloaded"
)
MENDELEY_TEXTURES_API_URL = (
    "https://data.mendeley.com/public-api/datasets/n666tk4mw9/files?folder_id=root&version=1"
)

MENDELEY_TEXTURES_METADATA = DatasetMetadata(
    dataset_id="mendeley-tactile-textures",
    name="Multimodal Tactile Texture Dataset",
    description=(
        "A pressure-and-IMU dataset for classifying 12 textures during dynamic finger "
        "exploration at three commanded velocities."
    ),
    homepage="https://data.mendeley.com/datasets/n666tk4mw9/1",
    version="V1",
    revision="published 2023-08-15",
    provenance=ProvenanceInfo(
        publisher="Mendeley Data",
        source_url="https://doi.org/10.17632/n666tk4mw9.1",
        creators=(
            "Bruno Monteiro Rocha Lima",
            "Thiago Eustaquio Alves de Oliveira",
            "Vinicius Prado da Fonseca",
        ),
        institutions=(
            "University of Ottawa",
            "Memorial University of Newfoundland",
            "Lakehead University",
        ),
        notes=(
            "The immutable ZIP README and directory names identify speeds 30, 35, and 40 mm/s; "
            "the Mendeley landing-page prose instead says 30, 40, and 45 mm/s."
        ),
    ),
    license=LicenseInfo(
        name="Creative Commons Attribution 4.0 International",
        spdx_id="CC-BY-4.0",
        url="https://creativecommons.org/licenses/by/4.0/",
        allows_redistribution=True,
        allows_commercial_use=True,
        requires_attribution=True,
        notes="Mendeley Data version 1 declares CC BY 4.0.",
    ),
    citations=(
        Citation(
            title="Multimodal Tactile Texture Dataset",
            url="https://doi.org/10.17632/n666tk4mw9.1",
            authors=(
                "Bruno Monteiro Rocha Lima",
                "Thiago Eustaquio Alves de Oliveira",
                "Vinicius Prado da Fonseca",
            ),
            year=2023,
            doi=MENDELEY_TEXTURES_DOI,
        ),
        Citation(
            title=(
                "Classification of Textures using a Tactile-Enabled Finger in Dynamic "
                "Exploration Tasks"
            ),
            url="https://doi.org/10.1109/SENSORS47087.2021.9639755",
            authors=(
                "Bruno Monteiro Rocha Lima",
                "Thiago Eustaquio Alves de Oliveira",
                "Vinicius Prado da Fonseca",
            ),
            year=2021,
            doi="10.1109/SENSORS47087.2021.9639755",
        ),
    ),
    access=(
        AccessSource(
            AccessKind.HTTP,
            MENDELEY_TEXTURES_URL,
            notes=(
                f"Immutable V1 ZIP: {MENDELEY_TEXTURES_SIZE_BYTES} bytes, publisher SHA-256 "
                f"{MENDELEY_TEXTURES_SHA256}; explicit large-download confirmation is required."
            ),
        ),
        AccessSource(
            AccessKind.REFERENCE,
            MENDELEY_TEXTURES_API_URL,
            notes=(
                "Official versioned file API exposing asset identifiers, byte sizes, and SHA-256."
            ),
        ),
        AccessSource(
            AccessKind.HTTP,
            MENDELEY_NOTEBOOK_URL,
            notes=(
                f"Publisher reader notebook: {MENDELEY_NOTEBOOK_SIZE_BYTES} bytes, SHA-256 "
                f"{MENDELEY_NOTEBOOK_SHA256}."
            ),
        ),
    ),
    modalities=(Modality.PRESSURE, Modality.IMU),
    sensors=("barometer", "inertial measurement unit", "tactile-enabled finger"),
    tasks=("texture classification",),
    formats=("zip", "python pickle", "jupyter notebook"),
    support_level=SupportLevel.LOADABLE,
    approximate_size_bytes=3_831_828_701,
    checksums={
        MENDELEY_TEXTURES_FILENAME: MENDELEY_TEXTURES_SHA256,
        MENDELEY_NOTEBOOK_FILENAME: MENDELEY_NOTEBOOK_SHA256,
    },
    limitations=(
        "Native pickle files can execute code and are rejected unless trust_pickle=True.",
        "The adapter requires an extracted local archive and intentionally ignores derived "
        "single-axis folders.",
        "The release does not document physical units or canonical train/validation/test splits.",
        "Publisher prose says 30, 40, and 45 mm/s, while the immutable artifact says 30, 35, "
        "and 40 mm/s; the adapter follows the artifact.",
    ),
    aliases=("multimodal-tactile-texture-dataset", "mendeley-textures"),
)

_SPEED = re.compile(r"^pickles_(30|35|40)$")
_TEXTURE = re.compile(r"^texture_(0[1-9]|1[0-2])$")
_BARO_COLUMNS = ("baro",)
_IMU_COLUMNS = (
    "imu_ax",
    "imu_ay",
    "imu_az",
    "imu_gx",
    "imu_gy",
    "imu_gz",
    "imu_mx",
    "imu_my",
    "imu_mz",
)


def download_mendeley_textures(
    cache: DatasetCache | None = None,
    *,
    confirm_large_download: bool = False,
    timeout_s: float = 30.0,
) -> Path:
    """Explicitly download and verify the 3.83 GB immutable V1 ZIP."""
    if not confirm_large_download:
        raise DatasetAccessError(
            "Mendeley tactile textures V1 is 3.83 GB; inspect the official dataset page and "
            "pass confirm_large_download=True to continue"
        )
    target_cache = cache or DatasetCache()
    return target_cache.download(
        MENDELEY_TEXTURES_URL,
        MENDELEY_TEXTURES_METADATA.dataset_id,
        filename=MENDELEY_TEXTURES_FILENAME,
        sha256=MENDELEY_TEXTURES_SHA256,
        size_bytes=MENDELEY_TEXTURES_SIZE_BYTES,
        timeout_s=timeout_s,
    )


def _inside(path: Path, root: Path) -> Path:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise DatasetValidationError(
            f"Mendeley texture path escapes the dataset root: {path}"
        ) from exc
    return path


def _read_trusted_pickle(path: Path) -> Any:
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)  # noqa: S301 - gated by trust_pickle
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == "pandas" or exc.name.startswith("pandas.")):
            raise DatasetAccessError(
                "The official Mendeley pickles require a compatible pandas installation; "
                "install pandas only after reviewing and trusting the source files"
            ) from exc
        raise DatasetAccessError(
            f"Could not import a module required by trusted pickle {path}: {exc}"
        ) from exc
    except (OSError, pickle.PickleError, EOFError) as exc:
        raise DatasetValidationError(
            f"Could not read Mendeley texture pickle {path}: {exc}"
        ) from exc
    except Exception as exc:
        raise DatasetValidationError(
            f"Could not decode trusted Mendeley texture pickle {path}: {exc}"
        ) from exc


def _column_names(frame: Any, path: Path) -> tuple[str, ...]:
    if isinstance(frame, Mapping):
        return tuple(str(key) for key in frame if not str(key).startswith("__"))
    columns = getattr(frame, "columns", None)
    if columns is None:
        raise DatasetValidationError(f"Mendeley texture pickle is not a table or mapping: {path}")
    return tuple(str(key) for key in columns)


def _column(frame: Any, name: str, path: Path) -> np.ndarray:
    try:
        value = frame[name]
    except (KeyError, TypeError, IndexError) as exc:
        raise DatasetValidationError(
            f"Mendeley texture pickle {path} is missing column {name!r}"
        ) from exc
    array = np.asarray(value)
    if array.ndim != 1 or array.size == 0:
        raise DatasetValidationError(
            f"Mendeley texture column {name!r} must be a non-empty vector: {path}"
        )
    try:
        array = array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise DatasetValidationError(
            f"Mendeley texture column {name!r} must be numeric: {path}"
        ) from exc
    if not np.isfinite(array).all():
        raise DatasetValidationError(
            f"Mendeley texture column {name!r} contains non-finite values: {path}"
        )
    return array


def _timestamps(frame: Any, expected: int, path: Path) -> np.ndarray | None:
    if isinstance(frame, Mapping):
        index = frame.get("__timestamps_s__")
        if index is None:
            return None
    else:
        index = getattr(frame, "index", None)
        if index is None:
            return None
    values = np.asarray(index)
    if values.ndim != 1 or values.shape[0] != expected:
        raise DatasetValidationError(f"Mendeley texture index length does not match data: {path}")
    if np.issubdtype(values.dtype, np.datetime64):
        nanoseconds = values.astype("datetime64[ns]").astype(np.int64)
        if np.any(nanoseconds == np.iinfo(np.int64).min):
            raise DatasetValidationError(
                f"Mendeley texture index contains missing timestamps: {path}"
            )
        result = (nanoseconds - nanoseconds[0]).astype(np.float64) / 1_000_000_000
    elif isinstance(frame, Mapping):
        try:
            result = values.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise DatasetValidationError(
                f"Mendeley fixture timestamps must be numeric: {path}"
            ) from exc
        result = result - result[0]
    else:
        # The official data use DatetimeIndex. Numeric RangeIndex values have no declared unit.
        return None
    if not np.isfinite(result).all() or np.any(np.diff(result) < 0):
        raise DatasetValidationError(
            f"Mendeley texture timestamps must be finite and ordered: {path}"
        )
    return result


def _table(
    frame: Any, expected_columns: tuple[str, ...], path: Path
) -> tuple[np.ndarray, np.ndarray | None]:
    columns = _column_names(frame, path)
    missing = [name for name in expected_columns if name not in columns]
    if missing:
        raise DatasetValidationError(
            f"Mendeley texture pickle {path} is missing columns: {', '.join(missing)}"
        )
    arrays = [_column(frame, name, path) for name in expected_columns]
    lengths = {array.shape[0] for array in arrays}
    if len(lengths) != 1:
        raise DatasetValidationError(f"Mendeley texture column lengths do not match: {path}")
    data = np.column_stack(arrays)
    return data, _timestamps(frame, data.shape[0], path)


@dataset_adapter(MENDELEY_TEXTURES_METADATA)
class MendeleyTexturesAdapter(DatasetAdapter):
    """Read primary paired pressure/IMU trials from an extracted trusted V1 archive."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        trust_pickle: bool = False,
        max_pickle_bytes: int = 64_000_000,
    ):
        super().__init__(root)
        self.trust_pickle = trust_pickle
        if max_pickle_bytes <= 0:
            raise ValueError("max_pickle_bytes must be positive")
        self.max_pickle_bytes = max_pickle_bytes

    def _layout_root(self, root: Path) -> Path:
        if root.name.startswith("pickles_") and _SPEED.fullmatch(root.name):
            return root.parent
        candidates = {path.parent for path in root.rglob("pickles_30") if path.is_dir()}
        if not candidates:
            candidates = {path.parent for path in root.rglob("pickles_35") if path.is_dir()} | {
                path.parent for path in root.rglob("pickles_40") if path.is_dir()
            }
        if len(candidates) != 1:
            raise DatasetAccessError(
                "Expected one extracted Mendeley V1 layout containing pickles_30/35/40 below "
                f"{root}; "
                f"found {len(candidates)}"
            )
        return candidates.pop()

    def _load(self, path: Path) -> Any:
        if not path.is_file():
            raise DatasetAccessError(f"Mendeley paired pickle is missing: {path}")
        size = path.stat().st_size
        if size > self.max_pickle_bytes:
            raise DatasetAccessError(
                "Mendeley pickle exceeds max_pickle_bytes "
                f"({size} > {self.max_pickle_bytes}): {path}"
            )
        return _read_trusted_pickle(path)

    def iter_samples(self, *, split: str | None = None) -> Iterator[TactileSample]:
        if split is not None:
            raise DatasetAccessError(
                "Mendeley tactile textures does not publish canonical machine-learning splits"
            )
        if not self.trust_pickle:
            raise DatasetAccessError(
                "Mendeley tactile textures uses executable Python pickle files; review the "
                "official CC BY 4.0 archive and pass trust_pickle=True only for files you trust"
            )
        root = self.require_root()
        if not root.is_dir():
            raise DatasetAccessError(
                f"Mendeley texture root must be an extracted directory: {root}"
            )
        layout = self._layout_root(root)
        found = False
        for speed_dir in sorted(path for path in layout.glob("pickles_*") if path.is_dir()):
            speed_match = _SPEED.fullmatch(speed_dir.name)
            if speed_match is None:
                continue
            speed = int(speed_match.group(1))
            for texture_dir in sorted(
                path for path in speed_dir.glob("texture_*") if path.is_dir()
            ):
                texture_match = _TEXTURE.fullmatch(texture_dir.name)
                if texture_match is None:
                    raise DatasetValidationError(
                        f"Unexpected Mendeley texture directory for V1: {texture_dir}"
                    )
                baro_dir = texture_dir / "full_baro"
                imu_dir = texture_dir / "full_imu"
                if not baro_dir.is_dir() or not imu_dir.is_dir():
                    raise DatasetAccessError(
                        f"Mendeley texture {texture_dir} must contain full_baro and full_imu"
                    )
                for baro_path in sorted(baro_dir.glob("baro*.pkl")):
                    suffix = baro_path.name.removeprefix("baro")
                    if not suffix or Path(suffix).name != suffix:
                        raise DatasetValidationError(
                            f"Unsafe Mendeley trial filename: {baro_path.name}"
                        )
                    imu_path = imu_dir / f"imu{suffix}"
                    _inside(baro_path, layout)
                    _inside(imu_path, layout)
                    barometer, baro_timestamps = _table(
                        self._load(baro_path), _BARO_COLUMNS, baro_path
                    )
                    imu, imu_timestamps = _table(self._load(imu_path), _IMU_COLUMNS, imu_path)
                    relative = baro_path.relative_to(layout).as_posix()
                    trial = Path(suffix).stem
                    sample_id = f"{speed_dir.name}/{texture_dir.name}/{trial}"
                    found = True
                    yield TactileSample(
                        sample_id=sample_id,
                        observations={
                            "pressure.barometer": TactileObservation(
                                Modality.PRESSURE,
                                barometer,
                                sensor="barometer",
                                timestamps=baro_timestamps,
                                metadata={
                                    "columns": list(_BARO_COLUMNS),
                                    "unit_status": "not published",
                                },
                            ),
                            "imu": TactileObservation(
                                Modality.IMU,
                                imu,
                                sensor="inertial measurement unit",
                                timestamps=imu_timestamps,
                                metadata={
                                    "columns": list(_IMU_COLUMNS),
                                    "unit_status": "not published",
                                },
                            ),
                        },
                        kind=SampleKind.SEQUENCE,
                        labels={
                            "texture": texture_dir.name,
                            "texture_index": int(texture_match.group(1)),
                            "exploration_speed_mm_s": speed,
                            "trial": trial,
                        },
                        task="texture classification",
                        group_id=sample_id,
                        metadata={
                            "source_paths": [
                                relative,
                                imu_path.relative_to(layout).as_posix(),
                            ],
                            "source_release": "Mendeley Data V1",
                            "native_serialization": "trusted Python pickle",
                        },
                    )
        if not found:
            raise DatasetAccessError(
                f"No paired full_baro/full_imu Mendeley V1 trials were found below {layout}"
            )
