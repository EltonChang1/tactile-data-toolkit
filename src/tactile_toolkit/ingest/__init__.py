"""Unified ingestion of raw tactile logs (MCAP, ROS bags, CSV / raw tensors)."""

from tactile_toolkit.ingest.align import TimeAligner
from tactile_toolkit.ingest.base import BaseReader, UnsupportedLogError
from tactile_toolkit.ingest.csv_reader import CsvReader, CsvSpec
from tactile_toolkit.ingest.modality import ModalityResolver, TopicMap
from tactile_toolkit.ingest.qa import QaConfig, QaReport, QualityGate
from tactile_toolkit.ingest.registry import detect_format, open_log, register_reader

__all__ = [
    "BaseReader",
    "CsvReader",
    "CsvSpec",
    "ModalityResolver",
    "QaConfig",
    "QaReport",
    "QualityGate",
    "TimeAligner",
    "TopicMap",
    "UnsupportedLogError",
    "detect_format",
    "open_log",
    "register_reader",
]


def __getattr__(name: str):
    # Lazy imports keep optional ROS backends off the import path until needed.
    if name == "McapReader":
        from tactile_toolkit.ingest.mcap_reader import McapReader

        return McapReader
    if name == "BagReader":
        from tactile_toolkit.ingest.bag_reader import BagReader

        return BagReader
    raise AttributeError(name)
