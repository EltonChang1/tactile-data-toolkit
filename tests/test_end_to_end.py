"""End-to-end conversion: synthetic logs -> Zarr / HDF5 / LeRobot, then validate."""

from __future__ import annotations

from tactile_toolkit.calibration.gelsight import GelSightConfig
from tactile_toolkit.calibration.taxel import TaxelConfig
from tactile_toolkit.export import infer_format, open_zarr, schema_view
from tactile_toolkit.pipeline import ConvertPipeline, PipelineConfig
from tactile_toolkit.schema import validate
from tactile_toolkit.synthetic import (
    SyntheticReader,
    SyntheticScenario,
    write_csv_dir,
    write_mcap,
    write_ros1_bag,
)


def _vision_cfg() -> PipelineConfig:
    return PipelineConfig(
        chunk_frames=8,
        gelsight=GelSightConfig(
            stride=8,
            store_depth_map=True,
            store_raw_image=True,
            track_markers=False,
        ),
    )


def test_in_memory_vision_to_all_formats(tmp_out, vision_reader):
    pipeline = ConvertPipeline(_vision_cfg())
    result = pipeline.convert(
        vision_reader,
        [
            ("zarr", tmp_out / "traj.zarr"),
            ("hdf5", tmp_out / "traj.h5"),
            ("lerobot", tmp_out / "traj_lerobot"),
        ],
    )
    assert result.frames_out > 0
    assert result.frames_in >= result.frames_out

    zarr_report = validate(open_zarr(tmp_out / "traj.zarr"))
    assert zarr_report.ok, zarr_report.summary()
    hdf_report = validate(schema_view(tmp_out / "traj.h5"))
    assert hdf_report.ok, hdf_report.summary()
    assert infer_format(tmp_out / "traj_lerobot") == "lerobot"
    assert (tmp_out / "traj_lerobot" / "meta" / "info.json").exists()


def test_mcap_roundtrip(tmp_out, small_vision_scenario):
    mcap = write_mcap(tmp_out / "input.mcap", small_vision_scenario)
    pipeline = ConvertPipeline(_vision_cfg())
    result = pipeline.convert(mcap, [("zarr", tmp_out / "from_mcap.zarr")])
    assert result.frames_out > 0
    report = validate(open_zarr(tmp_out / "from_mcap.zarr"))
    assert report.ok, report.summary()


def test_ros1_bag_roundtrip(tmp_out, small_vision_scenario):
    bag = write_ros1_bag(tmp_out / "input.bag", small_vision_scenario)
    pipeline = ConvertPipeline(_vision_cfg())
    result = pipeline.convert(bag, [("zarr", tmp_out / "from_bag.zarr")])
    assert result.frames_out > 0
    report = validate(open_zarr(tmp_out / "from_bag.zarr"))
    assert report.ok, report.summary()


def test_taxel_csv_to_hdf5(tmp_out, small_taxel_scenario):
    csv_dir = write_csv_dir(tmp_out / "csv_log", small_taxel_scenario)
    pipeline = ConvertPipeline(
        PipelineConfig(
            chunk_frames=16,
            taxel=TaxelConfig(idle_frames=5, rows=8, cols=8, pressure_map_shape=(12, 12)),
        )
    )
    result = pipeline.convert(csv_dir, [("hdf5", tmp_out / "taxels.h5")])
    assert result.frames_out > 0
    report = validate(schema_view(tmp_out / "taxels.h5"))
    assert report.ok, report.summary()


def test_quality_gate_drops_injected_duplicates(tmp_out):
    scenario = SyntheticScenario(
        duration_s=1.0,
        image_fps=30.0,
        image_height=48,
        image_width=64,
        include_taxels=False,
        include_wrench=False,
        duplicate_frame_every=5,
        seed=9,
    )
    pipeline = ConvertPipeline(_vision_cfg())
    result = pipeline.run(SyntheticReader(scenario), writers=[])
    assert result.qa.dropped_duplicate > 0
    assert result.frames_out < result.frames_in
