"""PyTorch DataLoader streaming over an exported Zarr store."""

from __future__ import annotations

import pytest

from tactile_toolkit.calibration.gelsight import GelSightConfig
from tactile_toolkit.dataset import TactileZarrDataset, make_dataloader
from tactile_toolkit.pipeline import ConvertPipeline, PipelineConfig
from tactile_toolkit.schema import NORMAL_FORCE, POINT_CLOUD, TIMESTAMPS, WRENCH


def test_zarr_dataset_single_process(tmp_out, vision_reader):
    zarr_path = tmp_out / "dl.zarr"
    ConvertPipeline(
        PipelineConfig(
            chunk_frames=8,
            gelsight=GelSightConfig(stride=8, track_markers=False, store_raw_image=False),
        )
    ).convert(vision_reader, [("zarr", zarr_path)])

    ds = TactileZarrDataset(zarr_path, keys=(NORMAL_FORCE, POINT_CLOUD, WRENCH, TIMESTAMPS))
    assert len(ds) > 0
    sample = ds[0]
    assert NORMAL_FORCE in sample
    assert sample[NORMAL_FORCE].ndim == 1
    windowed = TactileZarrDataset(zarr_path, keys=(NORMAL_FORCE,), window=3, stride=2)
    item = windowed[0]
    assert item[NORMAL_FORCE].shape[0] == 3


def test_dataloader_multiprocess_workers(tmp_out, vision_reader):
    torch = pytest.importorskip("torch")
    zarr_path = tmp_out / "dl_workers.zarr"
    ConvertPipeline(
        PipelineConfig(
            chunk_frames=8,
            gelsight=GelSightConfig(stride=8, track_markers=False, store_raw_image=False),
        )
    ).convert(vision_reader, [("zarr", zarr_path)])

    ds = TactileZarrDataset(zarr_path, keys=(NORMAL_FORCE, WRENCH, TIMESTAMPS))
    loader = make_dataloader(ds, batch_size=4, num_workers=2)
    batches = 0
    frames = 0
    for batch in loader:
        batches += 1
        frames += int(batch[NORMAL_FORCE].shape[0])
        assert isinstance(batch[NORMAL_FORCE], torch.Tensor)
        if batches >= 3:
            break
    assert frames > 0
    assert batches >= 1
