"""Open-Tactile-Schema: the sensor-agnostic layout every exported dataset conforms to.

The schema is a flat list of :class:`FieldSpec` entries keyed by hierarchical
path. Dimensions are symbolic (``T`` frames, ``N`` surface elements, ``H``/``W``
image size) and are checked for consistency across fields during validation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tactile_toolkit.types import Modality, normalize_path

SCHEMA_VERSION = "open-tactile-schema/0.2"

Dim = str | int


@dataclass(frozen=True)
class FieldSpec:
    path: str
    dtype: np.dtype
    shape: tuple[Dim, ...]
    unit: str
    description: str
    required: bool = True
    modalities: tuple[Modality, ...] = ()
    """If non-empty, the field is only required when one of these modalities is present."""

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    def is_required_for(self, modalities: Iterable[Modality] | None) -> bool:
        if not self.required:
            return False
        if not self.modalities:
            return True
        if modalities is None:
            return False
        return any(m in self.modalities for m in modalities)


TIMESTAMPS = "/timestamps"
CONTACT_MASK = "/observation/tactile/contact_mask"
NORMAL_FORCE = "/observation/tactile/normal_force"
SHEAR_FORCE = "/observation/tactile/shear_force"
POINT_CLOUD = "/observation/tactile/point_cloud"
RAW_IMAGE = "/observation/tactile/raw_image"
DEPTH_MAP = "/observation/tactile/depth_map"
PRESSURE_MAP = "/observation/tactile/pressure_map"
WORKSPACE_IMAGE = "/observation/image"
LANGUAGE_EMBEDDING = "/observation/natural_language_embedding"
WRENCH = "/observation/wrench"
POSE = "/observation/pose"
ACTION = "/action"
IS_FIRST = "/is_first"
IS_LAST = "/is_last"
IS_TERMINAL = "/is_terminal"

FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        TIMESTAMPS,
        np.dtype("float64"),
        ("T",),
        "s",
        "Sensor timestamp of each frame in seconds (master clock).",
    ),
    FieldSpec(
        CONTACT_MASK,
        np.dtype("bool"),
        ("T", "N"),
        "binary",
        "Boolean mask indicating active contact per taxel/pixel element.",
        modalities=(Modality.VISION_TACTILE, Modality.TAXEL),
    ),
    FieldSpec(
        NORMAL_FORCE,
        np.dtype("float32"),
        ("T", "N"),
        "N",
        "Calculated or measured normal force (z-axis pressure) per element.",
        modalities=(Modality.VISION_TACTILE, Modality.TAXEL),
    ),
    FieldSpec(
        SHEAR_FORCE,
        np.dtype("float32"),
        ("T", "N", 2),
        "N",
        "Shear force vectors along the surface plane (fx, fy).",
        modalities=(Modality.VISION_TACTILE, Modality.TAXEL),
    ),
    FieldSpec(
        POINT_CLOUD,
        np.dtype("float32"),
        ("T", "N", 6),
        "m, N",
        "Spatial 3D points (x, y, z) with force components (fx, fy, fz).",
        modalities=(Modality.VISION_TACTILE, Modality.TAXEL),
    ),
    FieldSpec(
        RAW_IMAGE,
        np.dtype("uint8"),
        ("T", "H", "W", 3),
        "RGB",
        "Raw gel camera frame (vision-tactile sensors only).",
        modalities=(Modality.VISION_TACTILE,),
    ),
    FieldSpec(
        DEPTH_MAP,
        np.dtype("float32"),
        ("T", "H", "W"),
        "m",
        "Reconstructed gel indentation depth (vision-tactile sensors only).",
        required=False,
        modalities=(Modality.VISION_TACTILE,),
    ),
    FieldSpec(
        PRESSURE_MAP,
        np.dtype("float32"),
        ("T", "GH", "GW"),
        "N",
        "Normal force interpolated onto a regular grid (taxel arrays only).",
        required=False,
        modalities=(Modality.TAXEL,),
    ),
    FieldSpec(
        WORKSPACE_IMAGE,
        np.dtype("uint8"),
        ("T", "OH", "OW", 3),
        "RGB",
        "Primary workspace RGB image used by Open X-Embodiment / RT-X policies.",
        modalities=(Modality.VISION,),
    ),
    FieldSpec(
        LANGUAGE_EMBEDDING,
        np.dtype("float32"),
        ("T", 512),
        "embedding",
        "Per-step 512-D language embedding used by the released RT-1-X model.",
        required=False,
    ),
    FieldSpec(
        WRENCH,
        np.dtype("float32"),
        ("T", 6),
        "N, N*m",
        "Global 6-axis force/torque wrench at the wrist (Fx, Fy, Fz, Tx, Ty, Tz).",
        modalities=(Modality.WRENCH,),
    ),
    FieldSpec(
        POSE,
        np.dtype("float32"),
        ("T", 7),
        "m, unit quaternion",
        "End-effector pose (x, y, z, qx, qy, qz, qw).",
        required=False,
        modalities=(Modality.POSE,),
    ),
    FieldSpec(
        ACTION,
        np.dtype("float32"),
        ("T", 7),
        "mixed",
        "Canonical RT-X action: (x, y, z, roll, pitch, yaw, gripper closedness).",
        modalities=(Modality.ROBOT_ACTION,),
    ),
    FieldSpec(
        IS_FIRST,
        np.dtype("bool"),
        ("T",),
        "binary",
        "RLDS marker indicating the first step in an episode.",
        required=False,
    ),
    FieldSpec(
        IS_LAST,
        np.dtype("bool"),
        ("T",),
        "binary",
        "RLDS marker indicating the final recorded step in an episode.",
        required=False,
    ),
    FieldSpec(
        IS_TERMINAL,
        np.dtype("bool"),
        ("T",),
        "binary",
        "RLDS marker distinguishing terminal episodes from truncated episodes.",
        required=False,
    ),
)


class OpenTactileSchema:
    """Programmatic view of the schema."""

    version = SCHEMA_VERSION

    def __init__(self, fields: Iterable[FieldSpec] = FIELDS):
        self._fields = {f.path: f for f in fields}

    def __iter__(self):
        return iter(self._fields.values())

    def __getitem__(self, path: str) -> FieldSpec:
        return self._fields[normalize_path(path)]

    def __contains__(self, path: object) -> bool:
        return isinstance(path, str) and normalize_path(path) in self._fields

    @property
    def paths(self) -> list[str]:
        return list(self._fields)

    def required_paths(self, modalities: Iterable[Modality] | None) -> list[str]:
        mods = list(modalities) if modalities is not None else None
        return [f.path for f in self._fields.values() if f.is_required_for(mods)]

    def as_table(self) -> list[dict[str, str]]:
        rows = []
        for f in self._fields.values():
            rows.append(
                {
                    "path": f.path,
                    "dtype": f.dtype.name,
                    "shape": "(" + ", ".join(str(d) for d in f.shape) + ")",
                    "unit": f.unit,
                    "required": "yes"
                    if f.required and not f.modalities
                    else ("if " + "/".join(m.value for m in f.modalities) if f.required else "no"),
                    "description": f.description,
                }
            )
        return rows


SCHEMA = OpenTactileSchema()


@dataclass
class ValidationReport:
    missing: list[str] = field(default_factory=list)
    dtype_mismatch: list[str] = field(default_factory=list)
    shape_mismatch: list[str] = field(default_factory=list)
    dim_conflicts: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    dims: dict[str, int] = field(default_factory=dict)
    checked: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.missing or self.dtype_mismatch or self.shape_mismatch or self.dim_conflicts
        )

    def summary(self) -> str:
        lines = [f"schema {SCHEMA_VERSION}: {'OK' if self.ok else 'FAILED'}"]
        lines.append(f"  checked fields : {len(self.checked)}")
        lines.append("  dims           : " + ", ".join(f"{k}={v}" for k, v in self.dims.items()))
        for label, items in (
            ("missing", self.missing),
            ("dtype mismatch", self.dtype_mismatch),
            ("shape mismatch", self.shape_mismatch),
            ("dim conflicts", self.dim_conflicts),
            ("unknown fields", self.unknown),
        ):
            if items:
                lines.append(f"  {label}:")
                lines.extend(f"    - {item}" for item in items)
        return "\n".join(lines)

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise SchemaValidationError(self.summary())


class SchemaValidationError(ValueError):
    pass


def _lookup(group: Any, path: str) -> Any | None:
    """Fetch an array-like at ``path`` from a zarr/h5py group or a mapping of arrays."""
    rel = path.lstrip("/")
    try:
        return group[rel]
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    try:
        return group[path]
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    node = group
    for part in rel.split("/"):
        try:
            node = node[part]
        except (KeyError, TypeError, ValueError, AttributeError):
            return None
    return node


def _iter_leaf_paths(group: Any, prefix: str = "") -> Iterable[str]:
    """Enumerate array paths in a zarr/h5py group or a flat/nested mapping."""
    if isinstance(group, Mapping) and not hasattr(group, "attrs"):
        for k, v in group.items():
            p = f"{prefix}/{k}"
            if isinstance(v, Mapping):
                yield from _iter_leaf_paths(v, p)
            else:
                yield normalize_path(p)
        return
    try:
        items = list(group.items())
    except Exception:  # noqa: BLE001 - foreign group types
        return
    for k, v in items:
        p = f"{prefix}/{k}"
        if hasattr(v, "shape") and hasattr(v, "dtype"):
            yield normalize_path(p)
        elif hasattr(v, "items"):
            yield from _iter_leaf_paths(v, p)


def _infer_modalities(group: Any) -> list[Modality] | None:
    attrs = getattr(group, "attrs", None)
    if attrs is None:
        return None
    try:
        raw = attrs.get("modalities")
    except Exception:  # noqa: BLE001
        return None
    if not raw:
        return None
    return [Modality.parse(m) for m in raw]


def validate(
    group: Any,
    modalities: Iterable[Modality | str] | None = None,
    schema: OpenTactileSchema = SCHEMA,
    check_unknown: bool = False,
) -> ValidationReport:
    """Validate a dataset group (zarr, h5py, or mapping of arrays) against the schema.

    Parameters
    ----------
    group:
        Root group of the dataset or a mapping of ``{path: array}``.
    modalities:
        Modalities present in the dataset. When omitted the ``modalities``
        attribute of the group is used; if unavailable, modality-conditional
        fields are only checked when present.
    check_unknown:
        Also report array paths that are not part of the schema.
    """
    report = ValidationReport()
    mods: list[Modality] | None
    if modalities is not None:
        mods = [Modality.parse(m) for m in modalities]
    else:
        mods = _infer_modalities(group)

    dims: dict[str, tuple[int, str]] = {}

    for spec in schema:
        arr = _lookup(group, spec.path)
        if arr is None or not hasattr(arr, "shape"):
            if spec.is_required_for(mods):
                report.missing.append(spec.path)
            continue
        report.checked.append(spec.path)

        actual_dtype = np.dtype(arr.dtype)
        if actual_dtype != spec.dtype:
            report.dtype_mismatch.append(
                f"{spec.path}: expected {spec.dtype.name}, found {actual_dtype.name}"
            )

        shape = tuple(int(s) for s in arr.shape)
        if len(shape) != len(spec.shape):
            report.shape_mismatch.append(
                f"{spec.path}: expected rank {len(spec.shape)} {spec.shape}, found {shape}"
            )
            continue
        for actual, expected in zip(shape, spec.shape):
            if isinstance(expected, int):
                if actual != expected:
                    report.shape_mismatch.append(
                        f"{spec.path}: expected shape {spec.shape}, found {shape}"
                    )
                    break
            else:
                prev = dims.get(expected)
                if prev is None:
                    dims[expected] = (actual, spec.path)
                elif prev[0] != actual:
                    report.dim_conflicts.append(
                        f"{expected}: {prev[1]} has {prev[0]} but {spec.path} has {actual}"
                    )

    report.dims = {k: v[0] for k, v in dims.items()}

    if check_unknown:
        known = set(schema.paths)
        for p in _iter_leaf_paths(group):
            if p not in known:
                report.unknown.append(p)

    return report
