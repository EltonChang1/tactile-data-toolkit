"""Open-Tactile-Schema constants and validator."""

from __future__ import annotations

import numpy as np
import pytest

from tactile_toolkit import SCHEMA, SCHEMA_VERSION, validate
from tactile_toolkit.schema import (
    CONTACT_MASK,
    NORMAL_FORCE,
    POINT_CLOUD,
    RAW_IMAGE,
    SHEAR_FORCE,
    TIMESTAMPS,
    WRENCH,
    SchemaValidationError,
)
from tactile_toolkit.types import Modality, TactileChunk


def _valid_arrays(t: int = 4, n: int = 8, h: int = 16, w: int = 20) -> dict[str, np.ndarray]:
    return {
        TIMESTAMPS: np.linspace(0, 1, t, dtype=np.float64),
        CONTACT_MASK: np.zeros((t, n), dtype=bool),
        NORMAL_FORCE: np.zeros((t, n), dtype=np.float32),
        SHEAR_FORCE: np.zeros((t, n, 2), dtype=np.float32),
        POINT_CLOUD: np.zeros((t, n, 6), dtype=np.float32),
        RAW_IMAGE: np.zeros((t, h, w, 3), dtype=np.uint8),
        WRENCH: np.zeros((t, 6), dtype=np.float32),
    }


def test_schema_version_and_required_core_fields():
    assert SCHEMA_VERSION.startswith("open-tactile-schema/")
    required = SCHEMA.required_paths([Modality.VISION_TACTILE, Modality.WRENCH])
    core = (TIMESTAMPS, CONTACT_MASK, NORMAL_FORCE, SHEAR_FORCE, POINT_CLOUD, RAW_IMAGE, WRENCH)
    for path in core:
        assert path in required
    assert RAW_IMAGE not in SCHEMA.required_paths([Modality.TAXEL])


def test_schema_table_lists_every_field():
    rows = SCHEMA.as_table()
    paths = {r["path"] for r in rows}
    assert set(SCHEMA.paths) == paths


def test_validate_accepts_complete_mapping():
    report = validate(_valid_arrays(), modalities=[Modality.VISION_TACTILE, Modality.WRENCH])
    assert report.ok
    assert report.dims["T"] == 4
    assert report.dims["N"] == 8
    report.raise_if_invalid()


def test_validate_reports_missing_fields():
    arrays = _valid_arrays()
    del arrays[POINT_CLOUD]
    report = validate(arrays, modalities=[Modality.VISION_TACTILE])
    assert not report.ok
    assert POINT_CLOUD in report.missing
    with pytest.raises(SchemaValidationError):
        report.raise_if_invalid()


def test_validate_reports_dtype_and_shape_errors():
    arrays = _valid_arrays()
    arrays[NORMAL_FORCE] = arrays[NORMAL_FORCE].astype(np.float64)
    arrays[SHEAR_FORCE] = np.zeros((4, 8), dtype=np.float32)
    report = validate(arrays, modalities=[Modality.VISION_TACTILE])
    assert any(NORMAL_FORCE in item for item in report.dtype_mismatch)
    assert any(SHEAR_FORCE in item for item in report.shape_mismatch)


def test_validate_detects_inconsistent_n():
    arrays = _valid_arrays()
    arrays[CONTACT_MASK] = np.zeros((4, 3), dtype=bool)
    report = validate(arrays, modalities=[Modality.VISION_TACTILE])
    assert report.dim_conflicts


def test_validate_nested_mapping_and_unknown_paths():
    arrays = {
        "timestamps": _valid_arrays()[TIMESTAMPS],
        "observation": {
            "tactile": {
                "contact_mask": _valid_arrays()[CONTACT_MASK],
                "normal_force": _valid_arrays()[NORMAL_FORCE],
                "shear_force": _valid_arrays()[SHEAR_FORCE],
                "point_cloud": _valid_arrays()[POINT_CLOUD],
                "raw_image": _valid_arrays()[RAW_IMAGE],
                "extra_debug": np.zeros((4,), dtype=np.float32),
            }
        },
    }
    report = validate(arrays, modalities=[Modality.VISION_TACTILE], check_unknown=True)
    assert report.ok
    assert any("extra_debug" in p for p in report.unknown)


def test_tactile_chunk_enforces_shared_time_axis():
    chunk = TactileChunk({TIMESTAMPS: np.arange(3, dtype=np.float64)})
    with pytest.raises(ValueError):
        chunk[NORMAL_FORCE] = np.zeros((2, 4), dtype=np.float32)
    chunk[NORMAL_FORCE] = np.zeros((3, 4), dtype=np.float32)
    kept = chunk.select(np.array([True, False, True]))
    assert kept.num_frames == 2
