"""PyTorch ``Dataset`` over an exported Zarr store with multi-worker safe lazy opening."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from tactile_toolkit import schema as S
from tactile_toolkit.types import normalize_path

try:  # torch is optional at import time so the package works without it.
    import torch
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # pragma: no cover - exercised only without torch installed
    torch = None  # type: ignore[assignment]

    class _TorchDataset:  # type: ignore[no-redef]
        pass


DEFAULT_KEYS = (
    S.NORMAL_FORCE,
    S.SHEAR_FORCE,
    S.CONTACT_MASK,
    S.POINT_CLOUD,
    S.WRENCH,
    S.TIMESTAMPS,
)


class TactileZarrDataset(_TorchDataset):
    """Frame-indexed access to an Open-Tactile-Schema Zarr store.

    Parameters
    ----------
    path:
        Zarr store produced by :class:`tactile_toolkit.export.ZarrWriter`.
    keys:
        Schema paths to load. Missing keys are silently skipped unless ``strict``.
    window:
        Number of consecutive frames per sample. ``window > 1`` stacks frames
        along a new leading axis, e.g. for temporal models.
    stride:
        Step between sample start frames.
    transform:
        Optional callable applied to the sample dict.
    as_tensors:
        Convert arrays to ``torch.Tensor`` (default) or keep NumPy arrays.

    The store is opened lazily inside each worker process, so the dataset can be
    passed to ``DataLoader(num_workers>0)`` on every platform (fork or spawn).
    """

    def __init__(
        self,
        path: str | Path,
        keys: Iterable[str] | None = None,
        window: int = 1,
        stride: int = 1,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        as_tensors: bool = True,
        strict: bool = False,
    ):
        self.path = str(path)
        self.keys = [normalize_path(k) for k in (keys or DEFAULT_KEYS)]
        self.window = max(1, int(window))
        self.stride = max(1, int(stride))
        self.transform = transform
        self.as_tensors = as_tensors and torch is not None
        self.strict = strict
        self._root: Any = None
        self._arrays: dict[str, Any] | None = None
        self._length = self._probe_length()

    # ------------------------------------------------------------- internals
    @staticmethod
    def _lookup(root: Any, key: str) -> Any | None:
        rel = key.lstrip("/")
        try:
            node = root[rel]
        except (KeyError, TypeError, ValueError):
            return None
        return node if hasattr(node, "shape") else None

    def _probe_length(self) -> int:
        import zarr

        root = zarr.open_group(self.path, mode="r")
        ts = root[S.TIMESTAMPS.lstrip("/")]
        n = int(ts.shape[0])
        available = [k for k in self.keys if self._lookup(root, k) is not None]
        missing = [k for k in self.keys if k not in available]
        if missing and self.strict:
            raise KeyError(f"Keys missing from store: {missing}")
        self.keys = available
        return max(0, (n - self.window) // self.stride + 1)

    def _ensure_open(self) -> dict[str, Any]:
        if self._arrays is None:
            import zarr

            self._root = zarr.open_group(self.path, mode="r")
            arrays: dict[str, Any] = {}
            for k in self.keys:
                node = self._lookup(self._root, k)
                if node is not None:
                    arrays[k] = node
            self._arrays = arrays
        return self._arrays

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_root"] = None
        state["_arrays"] = None
        return state

    # --------------------------------------------------------------- dataset
    def __len__(self) -> int:
        return self._length

    @property
    def attrs(self) -> dict[str, Any]:
        self._ensure_open()
        return dict(self._root.attrs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        arrays = self._ensure_open()
        start = index * self.stride
        stop = start + self.window
        sample: dict[str, Any] = {}
        for key, arr in arrays.items():
            value = arr[start:stop]
            if self.window == 1:
                value = value[0]
            sample[key] = self._convert(value)
        sample["frame_index"] = self._convert(np.asarray(start, dtype=np.int64))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _convert(self, value: np.ndarray) -> Any:
        if not self.as_tensors:
            return value
        value = np.ascontiguousarray(value)
        if value.dtype == np.bool_:
            return torch.from_numpy(value)
        if value.dtype == np.float64:
            return torch.from_numpy(value)
        return torch.from_numpy(value)


def make_dataloader(
    dataset: TactileZarrDataset, batch_size: int = 32, num_workers: int = 0, **kwargs: Any
) -> Any:
    """Thin helper around ``torch.utils.data.DataLoader`` with sensible defaults."""
    if torch is None:
        raise ImportError("PyTorch is required for make_dataloader")
    from torch.utils.data import DataLoader

    kwargs.setdefault("shuffle", False)
    kwargs.setdefault("pin_memory", False)
    if num_workers > 0:
        kwargs.setdefault("persistent_workers", True)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, **kwargs)
