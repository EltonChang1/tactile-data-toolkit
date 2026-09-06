# Contributing

## Setup

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format .
uv run pytest
```

Python 3.11 or 3.12. The `dev` extra pulls in pytest, ruff, mypy, psutil, and torch.

## Layout

- `src/tactile_toolkit/ingest` — readers, alignment, quality gate
- `src/tactile_toolkit/calibration` — GelSight and taxel pipelines
- `src/tactile_toolkit/export` — Zarr, HDF5, LeRobot writers
- `src/tactile_toolkit/dataset` — PyTorch loader
- `src/tactile_toolkit/datasets` — published-dataset metadata, adapters, registry, and acquisition
- `tests/` — unit and end-to-end coverage on synthetic logs
- `docs/` — architecture and schema

Keep conversions streaming. A change that loads an entire image trajectory into RAM will fail the 10,000-frame memory target.

## Tests

Prefer synthetic data from `tactile_toolkit.synthetic` over checked-in binaries. End-to-end tests should write Zarr or HDF5 and run `schema.validate`. Mark long benchmarks with `@pytest.mark.slow`.

Published-dataset integrations must follow [docs/adding-datasets.md](docs/adding-datasets.md), use tiny synthetic fixtures, and keep third-party license and provenance separate from this repository's Apache-2.0 license.

## Pull requests

Small, reviewable commits. Describe why the change exists. Do not add generated datasets, virtualenvs, or editor directories.
