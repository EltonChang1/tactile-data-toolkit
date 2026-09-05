"""Tactile Data Toolkit.

Universal ingestion, auto-calibration, and standardized dataset export for
robotic tactile sensors (vision-tactile gel fingers, taxel arrays, and
force/torque wrenches).
"""

from tactile_toolkit.schema import SCHEMA, SCHEMA_VERSION, OpenTactileSchema, validate
from tactile_toolkit.types import Modality, SensorMetadata, Stream, TactileChunk

__version__ = "0.1.0"

__all__ = [
    "SCHEMA",
    "SCHEMA_VERSION",
    "ConvertPipeline",
    "Modality",
    "OpenTactileSchema",
    "PipelineConfig",
    "SensorMetadata",
    "Stream",
    "TactileChunk",
    "__version__",
    "validate",
]


def __getattr__(name: str):
    # Heavy submodules are imported on first use to keep ``import tactile_toolkit`` light.
    if name in ("ConvertPipeline", "PipelineConfig"):
        from tactile_toolkit import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)
