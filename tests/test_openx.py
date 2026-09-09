"""Open X-Embodiment episode adaptation and RT-1-X model inputs."""

from __future__ import annotations

import numpy as np
import pytest

from tactile_toolkit import schema as S
from tactile_toolkit.dataset import (
    OpenXConfig,
    OpenXEpisodeAdapter,
    RT1XConfig,
    RT1XWindowDataset,
    canonicalize_rt1x_action,
    detokenize_rt1x_action,
    rt1x_action_vector,
    tokenize_rt1x_action,
)
from tactile_toolkit.schema import validate
from tactile_toolkit.types import Modality


def _rlds_episode(t: int = 4) -> dict:
    steps = []
    for i in range(t):
        steps.append(
            {
                "observation": {
                    "image": np.full((8, 12, 3), i, dtype=np.uint8),
                    "natural_language_instruction": b"pick up the block",
                    "natural_language_embedding": np.full(512, i, dtype=np.float32),
                    "touch": np.full(6, i, dtype=np.float32),
                },
                "action": {
                    "world_vector": np.asarray([i, 0, 0], dtype=np.float32),
                    "rotation_delta": np.asarray([0, 0.1, 0], dtype=np.float32),
                    "gripper_closedness_action": np.asarray([(-1) ** i], dtype=np.float32),
                },
                "is_first": i == 0,
                "is_last": i == t - 1,
                "is_terminal": i == t - 1,
            }
        )
    return {"steps": steps, "episode_id": "demo-1"}


def test_openx_episode_adapter_canonicalizes_rlds_steps():
    adapter = OpenXEpisodeAdapter(
        OpenXConfig(tactile_fields={S.NORMAL_FORCE: "touch"})
    )
    episode = adapter.adapt(_rlds_episode())

    assert episode.instruction == "pick up the block"
    assert episode.source == {"episode_id": "demo-1"}
    assert episode.chunk[S.WORKSPACE_IMAGE].shape == (4, 8, 12, 3)
    assert episode.chunk[S.LANGUAGE_EMBEDDING].shape == (4, 512)
    assert episode.chunk[S.ACTION].shape == (4, 7)
    assert episode.chunk[S.NORMAL_FORCE].shape == (4, 6)
    assert episode.rtx_actions["base_displacement_vector"].shape == (4, 2)
    np.testing.assert_array_equal(episode.rtx_actions["terminate_episode"][-1], [1, 0, 0])
    assert episode.modalities == [
        Modality.VISION,
        Modality.ROBOT_ACTION,
        Modality.TAXEL,
    ]

    report = validate(
        episode.chunk,
        modalities=[Modality.VISION, Modality.ROBOT_ACTION],
    )
    assert report.ok, report.summary()


def test_openx_adapter_accepts_batched_step_mapping_and_vector_action():
    t = 3
    episode = {
        "steps": {
            "observation": {
                "image": np.zeros((t, 4, 5, 3), dtype=np.uint8),
                "natural_language_instruction": np.asarray([b"move"] * t),
            },
            "action": np.zeros((t, 7), dtype=np.float32),
            "is_first": np.asarray([True, False, False]),
            "is_last": np.asarray([False, False, True]),
            "is_terminal": np.asarray([False, False, False]),
        }
    }
    adapted = OpenXEpisodeAdapter().adapt(episode)
    assert adapted.chunk.num_frames == t
    np.testing.assert_array_equal(adapted.chunk[S.IS_LAST], [False, False, True])
    # A truncated RLDS episode is last but not a model-requested termination.
    np.testing.assert_array_equal(adapted.rtx_actions["terminate_episode"][-1], [0, 1, 0])


def test_rt1x_tokenizer_matches_ranges_and_round_trips_within_one_bin():
    cfg = RT1XConfig(vocab_size=512)
    action = {
        "world_vector": np.asarray([[-2.0, 0.0, 2.0]], dtype=np.float32),
        "rotation_delta": np.asarray([[-np.pi / 2, 0.0, np.pi / 2]], dtype=np.float32),
        "gripper_closedness_action": np.asarray([[1.0]], dtype=np.float32),
        "base_displacement_vertical_rotation": np.asarray([[0.0]], dtype=np.float32),
        "base_displacement_vector": np.asarray([[-1.0, 1.0]], dtype=np.float32),
        "terminate_episode": np.asarray([[0, 1, 0]], dtype=np.int32),
    }
    tokens = tokenize_rt1x_action(action, cfg)
    assert tokens.shape == (1, 11)
    np.testing.assert_array_equal(
        tokens[0, [0, 1, 3, 4, 6, 7, 9, 10]],
        [1, 0, 511, 0, 511, 511, 0, 511],
    )

    restored = detokenize_rt1x_action(tokens, cfg)
    for key in (
        "world_vector",
        "rotation_delta",
        "gripper_closedness_action",
        "base_displacement_vertical_rotation",
        "base_displacement_vector",
    ):
        width = {"world_vector": 4.0, "rotation_delta": np.pi}.get(key, 2.0)
        if key == "base_displacement_vertical_rotation":
            width = 2 * np.pi
        np.testing.assert_allclose(restored[key], action[key], atol=width / (cfg.vocab_size - 1))


def test_canonical_action_zero_fills_unsupported_dimensions():
    vector = np.arange(14, dtype=np.float32).reshape(2, 7)
    canonical = canonicalize_rt1x_action(vector, is_last=np.asarray([False, True]))
    np.testing.assert_array_equal(rt1x_action_vector(canonical), vector)
    np.testing.assert_array_equal(canonical["base_displacement_vector"], np.zeros((2, 2)))
    np.testing.assert_array_equal(canonical["terminate_episode"], [[0, 1, 0], [1, 0, 0]])


def test_rt1x_window_dataset_resizes_scales_and_left_pads():
    episode = OpenXEpisodeAdapter().adapt(_rlds_episode())
    dataset = RT1XWindowDataset(
        episode,
        config=RT1XConfig(sequence_length=3, image_size=(6, 7)),
    )

    first = dataset[0]
    assert first["observation"]["image"].shape == (3, 6, 7, 3)
    assert first["observation"]["image"].dtype == np.float32
    assert first["observation"]["natural_language_embedding"].shape == (3, 512)
    assert first["action_tokens"].shape == (3, 11)
    np.testing.assert_array_equal(first["valid_mask"], [False, False, True])
    np.testing.assert_array_equal(first["action_valid_mask"], [False, False, True])
    assert np.all(first["observation"]["image"][:2] == 0)

    last = dataset[-1]
    np.testing.assert_array_equal(last["valid_mask"], [True, True, True])
    np.testing.assert_array_equal(last["action_valid_mask"], [True, True, False])
    assert float(last["observation"]["image"][-1, 0, 0, 0]) == pytest.approx(3 / 255)


def test_rt1x_window_requires_language_embedding():
    raw = _rlds_episode()
    for step in raw["steps"]:
        del step["observation"]["natural_language_embedding"]
    episode = OpenXEpisodeAdapter().adapt(raw)
    with pytest.raises(KeyError, match="512-D language embedding"):
        RT1XWindowDataset(episode)
