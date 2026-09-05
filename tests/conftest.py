"""Shared fixtures for the tactile-toolkit test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from tactile_toolkit.synthetic import SyntheticReader, SyntheticScenario


@pytest.fixture
def small_vision_scenario() -> SyntheticScenario:
    return SyntheticScenario(
        duration_s=1.0,
        image_fps=30.0,
        image_height=64,
        image_width=80,
        wrench_rate_hz=200.0,
        include_gelsight=True,
        include_taxels=False,
        include_wrench=True,
        include_pose=False,
        seed=7,
    )


@pytest.fixture
def small_taxel_scenario() -> SyntheticScenario:
    return SyntheticScenario(
        duration_s=1.0,
        image_fps=30.0,
        taxel_rate_hz=50.0,
        include_gelsight=False,
        include_taxels=True,
        include_wrench=True,
        include_pose=True,
        taxel_rows=8,
        taxel_cols=8,
        seed=11,
    )


@pytest.fixture
def vision_reader(small_vision_scenario: SyntheticScenario) -> SyntheticReader:
    return SyntheticReader(small_vision_scenario)


@pytest.fixture
def taxel_reader(small_taxel_scenario: SyntheticScenario) -> SyntheticReader:
    return SyntheticReader(small_taxel_scenario)


@pytest.fixture
def tmp_out(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    out.mkdir()
    return out
