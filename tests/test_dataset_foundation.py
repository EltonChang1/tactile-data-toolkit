"""Shared dataset metadata, registry, adapter, validation, and cache behavior."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tactile_toolkit.datasets import (
    DatasetAccessError,
    DatasetAdapter,
    DatasetCache,
    DatasetIntegrityError,
    DatasetMetadata,
    DatasetMetadataError,
    DatasetRegistry,
    DatasetUnavailableError,
    DatasetValidationError,
    SupportLevel,
    verify_file,
)
from tactile_toolkit.model import AssetReference, TactileObservation, TactileSample
from tactile_toolkit.types import Modality

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "datasets" / "minimal"


def _metadata() -> DatasetMetadata:
    return DatasetMetadata.from_dict(json.loads((FIXTURE_ROOT / "metadata.json").read_text()))


class MinimalAdapter(DatasetAdapter):
    METADATA = _metadata()

    def iter_samples(self, *, split: str | None = None) -> Iterator[TactileSample]:
        root = self.require_root()
        with (root / "manifest.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if split is not None and row["split"] != split:
                    continue
                yield TactileSample(
                    sample_id=row["sample_id"],
                    observations={
                        "pressure": TactileObservation(
                            Modality.PRESSURE,
                            AssetReference(
                                row["asset"],
                                media_type="text/csv",
                                size_bytes=(root / row["asset"]).stat().st_size,
                            ),
                            sensor="Synthetic 2x2 taxel",
                        )
                    },
                    labels={"texture": row["texture"]},
                    task="texture classification",
                    split=row["split"],
                    group_id=row["sample_id"],
                )


def test_metadata_fixture_roundtrips_without_losing_unknown_permissions():
    metadata = _metadata()
    assert metadata.dataset_id == "minimal-touch"
    assert metadata.support_level is SupportLevel.LOADABLE
    assert metadata.modalities == (Modality.PRESSURE,)
    assert DatasetMetadata.from_dict(metadata.to_dict()) == metadata

    value = metadata.to_dict()
    value["license"]["allows_commercial_use"] = None
    assert DatasetMetadata.from_dict(value).license.allows_commercial_use is None


def test_metadata_rejects_schema_and_license_errors():
    value = _metadata().to_dict()
    value["schema_version"] = "future/9"
    with pytest.raises(DatasetMetadataError, match="Unsupported dataset metadata schema"):
        DatasetMetadata.from_dict(value)

    value = _metadata().to_dict()
    value["license"]["allows_redistribution"] = "probably"
    with pytest.raises(DatasetMetadataError, match="true, false, or null"):
        DatasetMetadata.from_dict(value)


def test_registry_resolves_aliases_filters_support_and_opens_adapter():
    metadata = _metadata()
    registry = DatasetRegistry()
    registry.register(metadata, MinimalAdapter)

    assert "MINIMAL" in registry
    assert registry.metadata("fixture-touch") is metadata
    assert registry.list(minimum_support="loadable") == [metadata]
    assert registry.list(minimum_support="convertible") == []

    adapter = registry.open("minimal", root=FIXTURE_ROOT)
    samples = list(adapter.iter_samples(split="train"))
    assert [sample.sample_id for sample in samples] == ["sample-0"]
    assert samples[0].is_lazy


def test_registry_rejects_alias_collision_and_metadata_only_open():
    metadata = _metadata()
    registry = DatasetRegistry()
    registry.register(metadata)
    with pytest.raises(DatasetUnavailableError, match="has no adapter"):
        registry.open("minimal")

    collision = replace(metadata, dataset_id="another-touch", aliases=("minimal",))
    with pytest.raises(DatasetMetadataError, match="already registered"):
        registry.register(collision)


def test_adapter_validation_is_bounded_and_checks_local_assets():
    adapter = MinimalAdapter(FIXTURE_ROOT)
    report = adapter.validate(limit=1, check_assets=True)
    assert report.ok, report.summary()
    assert report.checked_samples == 1

    split_report = adapter.validate(split="validation", limit=5, check_assets=True)
    assert split_report.ok, split_report.summary()
    assert split_report.checked_samples == 1


def test_adapter_validation_does_not_prefetch_beyond_limit():
    emitted: list[str] = []

    class CountingAdapter(MinimalAdapter):
        def iter_samples(self, *, split: str | None = None) -> Iterator[TactileSample]:
            for sample in super().iter_samples(split=split):
                emitted.append(sample.sample_id)
                yield sample

    report = CountingAdapter(FIXTURE_ROOT).validate(limit=1)
    assert report.ok
    assert emitted == ["sample-0"]


def test_adapter_missing_root_has_actionable_error():
    with pytest.raises(DatasetAccessError, match=r"pass root=\.\.\."):
        MinimalAdapter().require_root()


def test_adapter_validation_reports_duplicate_ids_and_undeclared_modality():
    class BrokenAdapter(MinimalAdapter):
        def iter_samples(self, *, split: str | None = None) -> Iterator[TactileSample]:
            sample = next(super().iter_samples(split="train"))
            sample.observations["imu"] = TactileObservation(
                Modality.IMU,
                AssetReference("touch/000.csv", size_bytes=16),
            )
            yield sample
            yield sample

    report = BrokenAdapter(FIXTURE_ROOT).validate(limit=2)
    assert not report.ok
    assert any("undeclared modalities: imu" in error for error in report.errors)
    assert any("duplicate sample_id" in error for error in report.errors)
    with pytest.raises(DatasetValidationError, match="duplicate sample_id"):
        report.raise_if_invalid()


def test_verify_file_reports_size_and_checksum(tmp_path):
    path = tmp_path / "asset.bin"
    payload = b"tactile-fixture-data"
    path.write_bytes(payload)
    checksum = hashlib.sha256(payload).hexdigest()

    report = verify_file(path, size_bytes=len(payload), sha256=checksum, chunk_size=3)
    assert report.ok
    assert report.actual_sha256 == checksum

    bad = verify_file(path, size_bytes=len(payload) + 1, sha256="0" * 64)
    assert not bad.ok
    assert len(bad.errors) == 2
    with pytest.raises(DatasetIntegrityError, match="Integrity check failed"):
        bad.raise_if_invalid()


@pytest.fixture
def range_server():
    payload = b"0123456789abcdefghijklmnopqrstuvwxyz"

    class Handler(BaseHTTPRequestHandler):
        ranges: list[str | None] = []

        def do_GET(self):  # noqa: N802 - stdlib callback name
            if self.path not in {"/asset.bin", "/ignore-range.bin"}:
                self.send_error(404)
                return
            requested = self.headers.get("Range")
            type(self).ranges.append(requested)
            start = 0
            status = 200
            if requested is not None and self.path == "/asset.bin":
                start = int(requested.removeprefix("bytes=").removesuffix("-"))
                if start >= len(payload):
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{len(payload)}")
                    self.end_headers()
                    return
                status = 206
            body = payload[start:]
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            if status == 206:
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}"
                )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002 - stdlib callback signature
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/asset.bin", payload, Handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_cache_resumes_partial_download_and_reuses_verified_target(tmp_path, range_server):
    url, payload, handler = range_server
    cache = DatasetCache(tmp_path / "cache")
    target = cache.path_for("fixture", "asset.bin")
    partial = target.with_name("asset.bin.part")
    partial.write_bytes(payload[:7])
    progress: list[tuple[int, int | None]] = []

    result = cache.download(
        url,
        "fixture",
        filename="asset.bin",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        chunk_size=5,
        progress=lambda done, total: progress.append((done, total)),
    )
    assert result.read_bytes() == payload
    assert handler.ranges == ["bytes=7-"]
    assert progress[-1] == (len(payload), len(payload))
    assert not partial.exists()

    requests_before = len(handler.ranges)
    assert (
        cache.download(
            url,
            "fixture",
            filename="asset.bin",
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        == result
    )
    assert len(handler.ranges) == requests_before


def test_cache_keeps_failed_integrity_download_as_partial(tmp_path, range_server):
    url, payload, _ = range_server
    cache = DatasetCache(tmp_path / "cache")
    with pytest.raises(DatasetIntegrityError, match="SHA-256"):
        cache.download(
            url,
            "fixture",
            filename="bad.bin",
            size_bytes=len(payload),
            sha256="0" * 64,
        )
    assert not cache.path_for("fixture", "bad.bin").exists()
    assert cache.path_for("fixture", "bad.bin.part").read_bytes() == payload


def test_cache_restarts_when_server_ignores_range(tmp_path, range_server):
    url, payload, handler = range_server
    url = url.replace("asset.bin", "ignore-range.bin")
    cache = DatasetCache(tmp_path / "cache")
    target = cache.path_for("fixture", "asset.bin")
    target.with_name("asset.bin.part").write_bytes(b"stale-prefix")

    result = cache.download(
        url,
        "fixture",
        filename="asset.bin",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert result.read_bytes() == payload
    assert handler.ranges == ["bytes=12-"]


def test_cache_rejects_path_traversal(tmp_path):
    cache = DatasetCache(tmp_path / "cache")
    with pytest.raises(ValueError, match="single safe path component"):
        cache.path_for("fixture", "../outside.bin")
    with pytest.raises(ValueError, match="size_bytes must be non-negative"):
        cache.download("https://example.invalid/data.bin", "fixture", size_bytes=-1)
