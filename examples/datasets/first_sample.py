"""Discover published resources and inspect one normalized local sample."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tactile_toolkit.datasets import DATASETS, DatasetError, list_datasets, open_dataset
from tactile_toolkit.model import AssetReference, TactileObservation


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, np.ndarray):
        if value.size <= 64:
            return [_json_ready(item) for item in value.tolist()]
        return {"shape": list(value.shape), "dtype": str(value.dtype), "omitted": "large array"}
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _observation_summary(observation: TactileObservation) -> dict[str, Any]:
    data = observation.data
    if isinstance(data, AssetReference):
        payload: dict[str, Any] = {"storage": "lazy_asset", **data.to_dict()}
    elif isinstance(data, np.ndarray):
        payload = {
            "storage": "array",
            "shape": list(data.shape),
            "dtype": str(data.dtype),
        }
    else:
        payload = {"storage": "text", "characters": len(data)}
    payload.update(
        modality=observation.modality.value,
        sensor=observation.sensor,
        unit=observation.unit,
    )
    return payload


def _list_resources() -> None:
    print("dataset_id\tresource_kind\tsupport\tsample_adapter")
    for metadata in list_datasets():
        has_adapter = DATASETS.entry(metadata.dataset_id).factory is not None
        print(
            f"{metadata.dataset_id}\t{metadata.resource_kind.value}\t"
            f"{metadata.support_level.value}\t{'yes' if has_adapter else 'no'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_id", nargs="?", help="registered ID or alias")
    parser.add_argument("root", nargs="?", type=Path, help="user-provided local dataset root")
    parser.add_argument("--list", action="store_true", help="list resources without loading data")
    parser.add_argument("--split", help="publisher-provided split, when supported")
    parser.add_argument(
        "--trust-pickle",
        action="store_true",
        help="allow code-capable serialization only after verifying and trusting the source",
    )
    args = parser.parse_args()

    if args.list:
        _list_resources()
        return
    if args.dataset_id is None or args.root is None:
        parser.error("dataset_id and root are required unless --list is used")

    kwargs: dict[str, Any] = {"root": args.root}
    if args.trust_pickle:
        kwargs["trust_pickle"] = True
    try:
        adapter = open_dataset(args.dataset_id, **kwargs)
        sample = next(adapter.iter_samples(split=args.split))
    except StopIteration:
        parser.exit(1, f"No samples found for {args.dataset_id!r}\n")
    except DatasetError as exc:
        parser.exit(2, f"{exc}\n")

    result = {
        "dataset": {
            "dataset_id": adapter.metadata.dataset_id,
            "name": adapter.metadata.name,
            "resource_kind": adapter.metadata.resource_kind.value,
            "support": adapter.metadata.support_level.value,
            "license": adapter.metadata.license.name,
            "source": adapter.metadata.provenance.source_url,
        },
        "sample": {
            "sample_id": sample.sample_id,
            "kind": sample.kind.value,
            "task": sample.task,
            "split": sample.split,
            "group_id": sample.group_id,
            "labels": _json_ready(sample.labels),
            "observations": {
                name: _observation_summary(observation)
                for name, observation in sample.observations.items()
            },
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
