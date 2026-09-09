"""Open X-Embodiment (RLDS) and RT-1-X interoperability.

Open X-Embodiment datasets deliberately retain dataset-specific schemas.  This
module adapts an RLDS episode *after* any dataset-specific action transform into
the canonical seven-dimensional RT-X action used by Open-Tactile-Schema.  It
also provides the exact NumPy equivalent of the released JAX RT-1-X action
tokenizer and fixed-length, left-padded model windows.

TensorFlow is intentionally optional.  :func:`load_openx_dataset` imports
TensorFlow Datasets only when a local or ``gs://`` RLDS builder is requested;
the adapters themselves operate on ordinary mappings, NumPy arrays, or eager
TensorFlow values.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tactile_toolkit import schema as S
from tactile_toolkit.types import Modality, TactileChunk, normalize_path

OPEN_X_PROJECT_URL = "https://robotics-transformer-x.github.io/"
OPEN_X_REPOSITORY_URL = "https://github.com/google-deepmind/open_x_embodiment"
RT1X_DEFAULT_SEQUENCE_LENGTH = 15
RT1X_DEFAULT_VOCAB_SIZE = 512

RT1X_ACTION_WIDTHS: dict[str, int] = {
    "terminate_episode": 3,
    "world_vector": 3,
    "rotation_delta": 3,
    "gripper_closedness_action": 1,
    "base_displacement_vertical_rotation": 1,
    "base_displacement_vector": 2,
}


@dataclass(frozen=True)
class RT1XConfig:
    """Released RT-1-X preprocessing and action-tokenization parameters."""

    sequence_length: int = RT1X_DEFAULT_SEQUENCE_LENGTH
    image_size: tuple[int, int] = (300, 300)
    vocab_size: int = RT1X_DEFAULT_VOCAB_SIZE
    world_vector_range: tuple[float, float] = (-2.0, 2.0)
    rotation_range: tuple[float, float] = (-np.pi / 2, np.pi / 2)

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if min(self.image_size) <= 0:
            raise ValueError("image_size dimensions must be positive")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.world_vector_range[0] >= self.world_vector_range[1]:
            raise ValueError("world_vector_range must be increasing")
        if self.rotation_range[0] >= self.rotation_range[1]:
            raise ValueError("rotation_range must be increasing")


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _component(
    action: Mapping[str, Any], name: str, width: int, leading_shape: tuple[int, ...]
) -> np.ndarray:
    value = action.get(name)
    if value is None:
        return np.zeros((*leading_shape, width), dtype=np.float32)
    arr = _numpy(value).astype(np.float32, copy=False)
    if width == 1 and arr.shape == leading_shape:
        arr = arr[..., None]
    expected = (*leading_shape, width)
    if arr.shape != expected:
        raise ValueError(f"RT-1-X action '{name}' must have shape {expected}, got {arr.shape}")
    return arr


def _termination_action(
    value: Any | None, leading_shape: tuple[int, ...], is_last: Any | None
) -> np.ndarray:
    if value is not None:
        arr = _numpy(value)
        if arr.shape == (*leading_shape, 3):
            return arr.astype(np.int32, copy=False)
        if arr.shape == (*leading_shape, 1):
            arr = arr[..., 0]
        if arr.shape != leading_shape:
            raise ValueError(
                "RT-1-X action 'terminate_episode' must be scalar/batched flags or "
                f"one-hot shape {(*leading_shape, 3)}, got {arr.shape}"
            )
        terminate = arr.astype(bool, copy=False)
    elif is_last is not None:
        terminate = _numpy(is_last).astype(bool, copy=False)
        if terminate.shape != leading_shape:
            raise ValueError(f"is_last must have shape {leading_shape}, got {terminate.shape}")
    else:
        terminate = np.zeros(leading_shape, dtype=bool)

    # Official modes: 0 terminate, 1 arm + gripper, 2 mobile base.
    result = np.zeros((*leading_shape, 3), dtype=np.int32)
    result[..., 0] = terminate
    result[..., 1] = np.logical_not(terminate)
    return result


def canonicalize_rt1x_action(
    action: Mapping[str, Any] | np.ndarray,
    *,
    is_last: Any | None = None,
) -> dict[str, np.ndarray]:
    """Return the complete action dictionary expected by the released RT-1-X.

    A numeric input must end in seven values ordered as ``x, y, z, roll,
    pitch, yaw, gripper closedness``.  A mapping uses the official field names.
    Missing rotation, gripper, and mobile-base fields are zero-filled, matching
    the Open X-Embodiment training transform for unsupported robot dimensions.
    ``world_vector`` is the only required mapping field.
    """

    if not isinstance(action, Mapping):
        vector = _numpy(action).astype(np.float32, copy=False)
        if vector.ndim == 0 or vector.shape[-1] != 7:
            raise ValueError(f"Canonical RT-X action must end in 7 values, got {vector.shape}")
        leading_shape = vector.shape[:-1]
        action_map: Mapping[str, Any] = {
            "world_vector": vector[..., :3],
            "rotation_delta": vector[..., 3:6],
            "gripper_closedness_action": vector[..., 6:7],
        }
    else:
        if "world_vector" not in action:
            raise KeyError("RT-1-X action mapping requires 'world_vector'")
        world = _numpy(action["world_vector"])
        if world.ndim == 0 or world.shape[-1] != 3:
            raise ValueError(f"RT-1-X 'world_vector' must end in 3 values, got {world.shape}")
        leading_shape = world.shape[:-1]
        action_map = action

    result = {
        "world_vector": _component(action_map, "world_vector", 3, leading_shape),
        "rotation_delta": _component(action_map, "rotation_delta", 3, leading_shape),
        "gripper_closedness_action": _component(
            action_map, "gripper_closedness_action", 1, leading_shape
        ),
        "base_displacement_vertical_rotation": _component(
            action_map, "base_displacement_vertical_rotation", 1, leading_shape
        ),
        "base_displacement_vector": _component(
            action_map, "base_displacement_vector", 2, leading_shape
        ),
    }
    result["terminate_episode"] = _termination_action(
        action_map.get("terminate_episode"), leading_shape, is_last
    )
    return result


def rt1x_action_vector(action: Mapping[str, Any] | np.ndarray) -> np.ndarray:
    """Pack the arm-and-gripper part of an RT-1-X action as canonical 7-D values."""

    canonical = canonicalize_rt1x_action(action)
    return np.concatenate(
        [
            canonical["world_vector"],
            canonical["rotation_delta"],
            canonical["gripper_closedness_action"],
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def tokenize_rt1x_action(
    action: Mapping[str, Any] | np.ndarray,
    config: RT1XConfig | None = None,
    *,
    is_last: Any | None = None,
) -> np.ndarray:
    """Quantize an action into the released model's 11 categorical tokens."""

    cfg = config or RT1XConfig()
    canonical = canonicalize_rt1x_action(action, is_last=is_last)
    tokens = [np.argmax(canonical["terminate_episode"], axis=-1)[..., None].astype(np.int32)]
    ranges = (
        ("world_vector", cfg.world_vector_range),
        ("rotation_delta", cfg.rotation_range),
        ("gripper_closedness_action", (-1.0, 1.0)),
        ("base_displacement_vertical_rotation", (-np.pi, np.pi)),
        ("base_displacement_vector", (-1.0, 1.0)),
    )
    for name, (low, high) in ranges:
        value = np.clip(canonical[name], low, high)
        quantized = ((value - low) / (high - low) * (cfg.vocab_size - 1)).astype(np.int32)
        tokens.append(quantized)
    return np.concatenate(tokens, axis=-1)


def detokenize_rt1x_action(
    tokens: np.ndarray, config: RT1XConfig | None = None
) -> dict[str, np.ndarray]:
    """Invert RT-1-X tokens to the centre/edge value represented by each bin."""

    cfg = config or RT1XConfig()
    token_array = _numpy(tokens)
    if token_array.ndim == 0 or token_array.shape[-1] != 11:
        raise ValueError(f"RT-1-X tokens must end in 11 values, got {token_array.shape}")
    if np.any(token_array < 0) or np.any(token_array >= cfg.vocab_size):
        raise ValueError(f"RT-1-X tokens must be in [0, {cfg.vocab_size})")
    token_array = token_array.astype(np.int32, copy=False)
    if np.any(token_array[..., 0] > 2):
        raise ValueError("RT-1-X termination token must be one of 0, 1, or 2")
    terminate = np.eye(3, dtype=np.int32)[token_array[..., 0]]
    result: dict[str, np.ndarray] = {"terminate_episode": terminate}
    ranges = (
        ("world_vector", slice(1, 4), cfg.world_vector_range),
        ("rotation_delta", slice(4, 7), cfg.rotation_range),
        ("gripper_closedness_action", slice(7, 8), (-1.0, 1.0)),
        ("base_displacement_vertical_rotation", slice(8, 9), (-np.pi, np.pi)),
        ("base_displacement_vector", slice(9, 11), (-1.0, 1.0)),
    )
    for name, part, (low, high) in ranges:
        value = token_array[..., part].astype(np.float32)
        result[name] = value / (cfg.vocab_size - 1) * (high - low) + low
    return result


@dataclass(frozen=True)
class OpenXConfig:
    """Dataset-specific paths needed to adapt one Open X RLDS dataset."""

    fps: float = 3.0
    image_key: str = "image"
    language_instruction_key: str = "natural_language_instruction"
    language_embedding_key: str = "natural_language_embedding"
    timestamp_key: str | None = None
    action_transform: Callable[[Mapping[str, Any]], Mapping[str, Any] | np.ndarray] | None = None
    tactile_fields: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")


@dataclass
class OpenXEpisode:
    """One canonicalized RLDS episode plus the full RT-1-X action dictionary."""

    chunk: TactileChunk
    rtx_actions: dict[str, np.ndarray]
    instructions: tuple[str, ...]
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def instruction(self) -> str:
        return next((value for value in self.instructions if value), "")

    @property
    def modalities(self) -> list[Modality]:
        result = [Modality.VISION, Modality.ROBOT_ACTION]
        if S.RAW_IMAGE in self.chunk:
            result.append(Modality.VISION_TACTILE)
        elif any(
            key in self.chunk
            for key in (S.CONTACT_MASK, S.NORMAL_FORCE, S.SHEAR_FORCE, S.POINT_CLOUD)
        ):
            result.append(Modality.TAXEL)
        return result


def _decode_text(value: Any) -> str:
    arr = _numpy(value)
    if arr.ndim:
        if arr.size != 1:
            raise ValueError(f"Language instruction must be scalar, got shape {arr.shape}")
        value = arr.reshape(-1)[0]
    else:
        value = arr.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _mapping_length(tree: Mapping[str, Any]) -> int:
    for value in tree.values():
        if isinstance(value, Mapping):
            try:
                return _mapping_length(value)
            except ValueError:
                continue
        arr = _numpy(value)
        if arr.ndim:
            return int(arr.shape[0])
    raise ValueError("Could not infer the number of RLDS steps")


def _tree_index(tree: Mapping[str, Any], index: int) -> dict[str, Any]:
    return {
        key: _tree_index(value, index) if isinstance(value, Mapping) else _numpy(value)[index]
        for key, value in tree.items()
    }


def _iter_steps(episode: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    if "steps" not in episode:
        raise KeyError("RLDS episode requires a 'steps' field")
    steps = episode["steps"]
    if isinstance(steps, Mapping):
        for index in range(_mapping_length(steps)):
            yield _tree_index(steps, index)
        return
    if hasattr(steps, "as_numpy_iterator"):
        yield from steps.as_numpy_iterator()
        return
    yield from steps


def _feature(container: Mapping[str, Any], path: str) -> Any:
    node: Any = container
    for part in path.strip("/").split("/"):
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(path)
        node = node[part]
    return node


class OpenXEpisodeAdapter:
    """Convert heterogeneous RLDS episodes into canonical RT-X trajectory fields.

    Open X-Embodiment does not impose identical feature names or action units on
    every contributing dataset.  Configure image/language paths here and supply
    ``action_transform`` for datasets whose actions are not already canonical.
    The transform receives the whole source step and must return either the
    official RT-1-X action mapping or a canonical seven-value vector.
    """

    def __init__(self, config: OpenXConfig | None = None):
        self.config = config or OpenXConfig()

    def adapt(self, episode: Mapping[str, Any]) -> OpenXEpisode:
        cfg = self.config
        steps = list(_iter_steps(episode))
        if not steps:
            raise ValueError("Cannot adapt an empty RLDS episode")

        images: list[np.ndarray] = []
        embeddings: list[np.ndarray] = []
        instructions: list[str] = []
        timestamps: list[float] = []
        actions: dict[str, list[np.ndarray]] = {name: [] for name in RT1X_ACTION_WIDTHS}
        is_first: list[bool] = []
        is_last: list[bool] = []
        is_terminal: list[bool] = []
        tactile: dict[str, list[np.ndarray]] = {
            normalize_path(path): [] for path in cfg.tactile_fields
        }

        for index, step in enumerate(steps):
            observation = step.get("observation", {})
            if not isinstance(observation, Mapping):
                raise TypeError("RLDS step 'observation' must be a mapping")
            try:
                image = _feature(observation, cfg.image_key)
            except KeyError as exc:
                raise KeyError(
                    f"Workspace image '{cfg.image_key}' is missing from RLDS observation"
                ) from exc
            image_array = _numpy(image)
            if image_array.ndim != 3 or image_array.shape[-1] != 3:
                raise ValueError(
                    f"Workspace image must have shape (H, W, 3), got {image_array.shape}"
                )
            if image_array.dtype != np.uint8:
                raise TypeError(f"Workspace image must be uint8 RGB, got {image_array.dtype}")
            images.append(image_array)

            instruction = ""
            for container in (observation, step):
                try:
                    instruction = _decode_text(_feature(container, cfg.language_instruction_key))
                    break
                except KeyError:
                    continue
            instructions.append(instruction)

            try:
                embedding = _feature(observation, cfg.language_embedding_key)
            except KeyError:
                embedding = None
            if embedding is not None:
                embedding_array = _numpy(embedding).astype(np.float32, copy=False)
                if embedding_array.shape != (512,):
                    raise ValueError(
                        "RT-1-X natural_language_embedding must have shape (512,), "
                        f"got {embedding_array.shape}"
                    )
                embeddings.append(embedding_array)
            elif embeddings:
                raise ValueError("Language embedding is missing from only some RLDS steps")

            first = bool(_numpy(step.get("is_first", index == 0)).item())
            last = bool(_numpy(step.get("is_last", index == len(steps) - 1)).item())
            terminal = bool(_numpy(step.get("is_terminal", last)).item())
            is_first.append(first)
            is_last.append(last)
            is_terminal.append(terminal)

            raw_action = cfg.action_transform(step) if cfg.action_transform else step.get("action")
            if raw_action is None:
                raise KeyError("RLDS step requires 'action' or an OpenXConfig.action_transform")
            # When the source action has no explicit termination mode, only an
            # RLDS terminal (not a truncated is_last boundary) implies stop.
            canonical = canonicalize_rt1x_action(raw_action, is_last=terminal)
            for name in actions:
                actions[name].append(canonical[name])

            if cfg.timestamp_key is None:
                timestamps.append(index / cfg.fps)
            else:
                try:
                    timestamp = _feature(observation, cfg.timestamp_key)
                except KeyError:
                    timestamp = _feature(step, cfg.timestamp_key)
                timestamps.append(float(_numpy(timestamp).item()))

            for schema_path, source_path in cfg.tactile_fields.items():
                canonical_path = normalize_path(schema_path)
                try:
                    value = _feature(observation, source_path)
                except KeyError:
                    value = _feature(step, source_path)
                tactile[canonical_path].append(_numpy(value))

        action_arrays = {
            name: np.stack(values).astype(np.int32 if name == "terminate_episode" else np.float32)
            for name, values in actions.items()
        }
        arrays: dict[str, np.ndarray] = {
            S.TIMESTAMPS: np.asarray(timestamps, dtype=np.float64),
            S.WORKSPACE_IMAGE: np.stack(images).astype(np.uint8, copy=False),
            S.ACTION: rt1x_action_vector(action_arrays),
            S.IS_FIRST: np.asarray(is_first, dtype=bool),
            S.IS_LAST: np.asarray(is_last, dtype=bool),
            S.IS_TERMINAL: np.asarray(is_terminal, dtype=bool),
        }
        if embeddings:
            if len(embeddings) != len(steps):
                raise ValueError("Language embedding is missing from one or more RLDS steps")
            arrays[S.LANGUAGE_EMBEDDING] = np.stack(embeddings).astype(np.float32, copy=False)
        for path, values in tactile.items():
            arrays[path] = np.stack(values)

        source = {
            key: value
            for key, value in episode.items()
            if key != "steps" and isinstance(value, str | int | float | bool)
        }
        return OpenXEpisode(
            chunk=TactileChunk(arrays),
            rtx_actions=action_arrays,
            instructions=tuple(instructions),
            source=source,
        )


class RT1XWindowDataset:
    """Fixed-length, initial-zero-padded samples for released RT-1-X inference/training.

    The stock checkpoint consumes only the primary workspace image and the
    512-D language embedding.  ``tactile_keys`` can add synchronized touch
    fields for a modified multimodal policy without changing the stock
    observation dictionary.
    """

    def __init__(
        self,
        episode: OpenXEpisode,
        *,
        config: RT1XConfig | None = None,
        language_embedding: np.ndarray | None = None,
        tactile_keys: Iterable[str] = (),
    ):
        self.episode = episode
        self.config = config or RT1XConfig()
        self.tactile_keys = tuple(normalize_path(key) for key in tactile_keys)
        missing = [key for key in self.tactile_keys if key not in episode.chunk]
        if missing:
            raise KeyError(f"Tactile fields missing from episode: {missing}")
        if S.LANGUAGE_EMBEDDING in episode.chunk:
            self._language = episode.chunk[S.LANGUAGE_EMBEDDING]
        elif language_embedding is not None:
            value = _numpy(language_embedding).astype(np.float32, copy=False)
            if value.shape != (512,):
                raise ValueError(f"language_embedding must have shape (512,), got {value.shape}")
            self._language = np.broadcast_to(value, (episode.chunk.num_frames, 512))
        else:
            raise KeyError(
                "RT-1-X requires a 512-D language embedding; include "
                f"{S.LANGUAGE_EMBEDDING} or pass language_embedding"
            )

    def __len__(self) -> int:
        return self.episode.chunk.num_frames

    def _window(self, value: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
        length = self.config.sequence_length
        start = max(0, index - length + 1)
        real = value[start : index + 1]
        result = np.zeros((length, *value.shape[1:]), dtype=value.dtype)
        result[-len(real) :] = real
        mask = np.zeros(length, dtype=bool)
        mask[-len(real) :] = True
        return result, mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)

        images, mask = self._window(self.episode.chunk[S.WORKSPACE_IMAGE], index)
        target_h, target_w = self.config.image_size
        resized = np.empty((len(images), target_h, target_w, 3), dtype=np.float32)
        for frame_index, frame in enumerate(images):
            resized[frame_index] = cv2.resize(
                frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
        resized /= 255.0
        language, _ = self._window(self._language, index)
        last_steps, _ = self._window(self.episode.chunk[S.IS_LAST], index)

        action: dict[str, np.ndarray] = {}
        for name, values in self.episode.rtx_actions.items():
            action[name], _ = self._window(values, index)
        result: dict[str, Any] = {
            "observation": {
                "image": resized,
                "natural_language_embedding": language.astype(np.float32, copy=False),
                "natural_language_instruction": self.episode.instruction,
            },
            "action": action,
            "action_tokens": tokenize_rt1x_action(action, self.config),
            "valid_mask": mask,
            "action_valid_mask": np.logical_and(mask, np.logical_not(last_steps)),
            "frame_index": np.asarray(index, dtype=np.int64),
        }
        if self.tactile_keys:
            result["tactile"] = {
                key: self._window(self.episode.chunk[key], index)[0] for key in self.tactile_keys
            }
        return result


def load_openx_dataset(
    builder_dir: str | Path,
    *,
    split: str = "train",
    shuffle_files: bool = False,
    **as_dataset_kwargs: Any,
) -> Any:
    """Open an official local or GCS Open X-Embodiment RLDS builder lazily.

    Install the optional dependencies with ``pip install tactile-toolkit[openx]``.
    Dataset-specific feature/action mapping remains the caller's responsibility,
    as it is in the official Open X-Embodiment training example.
    """

    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "Open X-Embodiment loading requires the 'openx' extra: "
            "pip install 'tactile-toolkit[openx]'"
        ) from exc
    builder = tfds.builder_from_directory(builder_dir=str(builder_dir))
    return builder.as_dataset(
        split=split, shuffle_files=shuffle_files, **as_dataset_kwargs
    )


def iter_openx_episodes(
    dataset: Iterable[Mapping[str, Any]], adapter: OpenXEpisodeAdapter | None = None
) -> Iterator[OpenXEpisode]:
    """Adapt every raw RLDS episode from an already-open TFDS/iterable dataset."""

    episode_adapter = adapter or OpenXEpisodeAdapter()
    for episode in dataset:
        yield episode_adapter.adapt(episode)


__all__ = [
    "OPEN_X_PROJECT_URL",
    "OPEN_X_REPOSITORY_URL",
    "OpenXConfig",
    "OpenXEpisode",
    "OpenXEpisodeAdapter",
    "RT1XConfig",
    "RT1XWindowDataset",
    "canonicalize_rt1x_action",
    "detokenize_rt1x_action",
    "iter_openx_episodes",
    "load_openx_dataset",
    "rt1x_action_vector",
    "tokenize_rt1x_action",
]
