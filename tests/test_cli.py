"""CLI entry points: convert, inspect, validate, synth, schema."""

from __future__ import annotations

import json

from tactile_toolkit.cli import main


def test_cli_schema_table(capsys):
    assert main(["schema"]) == 0
    out = capsys.readouterr().out
    assert "open-tactile-schema" in out
    assert "/observation/tactile/normal_force" in out


def test_cli_synth_inspect_convert_validate(tmp_path, capsys):
    mcap = tmp_path / "scene.mcap"
    assert (
        main(
            [
                "synth",
                str(mcap),
                "--format",
                "mcap",
                "--duration",
                "0.4",
                "--fps",
                "20",
                "--size",
                "32x40",
                "--no-taxels",
                "--seed",
                "1",
            ]
        )
        == 0
    )
    assert mcap.exists()
    capsys.readouterr()

    assert main(["inspect", str(mcap), "--json"]) == 0
    desc = json.loads(capsys.readouterr().out)
    assert desc["kind"] == "raw_log"
    assert desc["master_topic"]

    zarr = tmp_path / "out.zarr"
    assert (
        main(
            [
                "convert",
                str(mcap),
                "-o",
                str(zarr),
                "--chunk-frames",
                "8",
                "--stride",
                "8",
                "--no-markers",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["validate", str(zarr)]) == 0
    capsys.readouterr()
    assert main(["inspect", str(zarr), "--json"]) == 0
    zdesc = json.loads(capsys.readouterr().out)
    assert zdesc["kind"] == "zarr"
    assert zdesc["attrs"]["num_frames"] > 0
