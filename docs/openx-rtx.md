# Open X-Embodiment and RT-X

[Open X-Embodiment](https://robotics-transformer-x.github.io/) pools more than one million real-robot
trajectories from 60 datasets and 22 robot embodiments. Each contributing dataset uses the
[RLDS](https://github.com/google-research/rlds) episode envelope: an episode contains a sequence of
steps, and each step carries `is_first`, `is_last`, observations, actions, and optional terminal or
reward fields. Feature names, action units, coordinate conventions, and gripper semantics can still
differ by source dataset.

The project trained two cross-embodiment policies. RT-1-X is the RT-1 transformer trained on the
robotics mixture; RT-2-X is a vision-language model co-fine-tuned to emit actions as text tokens.
Their common arm action is seven values: translation `(x, y, z)`, rotation `(roll, pitch, yaw)`, and
gripper opening/closedness. Unsupported dimensions are zeroed during mixture training. The released
RT-1-X interface uses a 15-frame history, one workspace RGB image and a task string represented by a
512-D embedding, and 11 action tokens (termination mode, 3 translation, 3 rotation, gripper, mobile
base rotation, and 2 mobile-base translation).

## What this toolkit supports

- `OpenXEpisodeAdapter` converts an in-memory RLDS episode into `/observation/image`, `/action`, RLDS
  boundary markers, language embeddings when present, and any explicitly mapped tactile fields.
- `RT1XWindowDataset` creates initial-zero-padded model histories, converts workspace RGB to
  300×300 float images in `[0, 1]`, marks invalid padding/final-step actions, and preserves
  synchronized tactile windows for modified models.
- `tokenize_rt1x_action` and `detokenize_rt1x_action` are NumPy ports of the released JAX
  implementation, including RT-1-X's `[-2, 2]` translation range, `[-π/2, π/2]` rotation range,
  512-bin vocabulary, and 11-token ordering.
- Zarr stores the new schema fields directly. HDF5 writes `/action` to robomimic's `actions`
  dataset. LeRobot writes workspace RGB as `observation.images.workspace` MP4 and the 7-D action as
  a Parquet feature.

The toolkit does not bundle model weights or claim that the released checkpoint consumes tactile
input. The stock RT-1-X checkpoint only accepts the primary workspace camera and language
embedding. Passing `tactile_keys` exposes aligned touch to a model you extend or fine-tune.

## Load and adapt RLDS

Install TensorFlow support only if you want the loader:

```bash
pip install -e ".[openx]"
```

Then open an official local or Google Cloud TFDS builder:

```python
from tactile_toolkit.dataset import (
    OpenXConfig,
    OpenXEpisodeAdapter,
    RT1XWindowDataset,
    iter_openx_episodes,
    load_openx_dataset,
)

raw = load_openx_dataset(
    "gs://gresearch/robotics/fractal20220817_data/0.1.0",
    split="train[:1]",
)
adapter = OpenXEpisodeAdapter(OpenXConfig(fps=3.0))
episode = next(iter_openx_episodes(raw, adapter))
model_sample = RT1XWindowDataset(episode)[-1]
```

The default adapter expects the already-standardized field names used by RT-1-X:

```text
observation/image
observation/natural_language_instruction
observation/natural_language_embedding
action/world_vector
action/rotation_delta
action/gripper_closedness_action
```

For another contributor dataset, supply the correct image path and a transform based on that
dataset's own documentation and the official training notebook:

```python
def map_action(step):
    source = step["action"]
    return {
        "world_vector": source[:3],
        "rotation_delta": source[3:6],
        "gripper_closedness_action": source[6:7],
    }

adapter = OpenXEpisodeAdapter(
    OpenXConfig(image_key="front_rgb", action_transform=map_action)
)
```

Do not reuse this illustrative transform unless the source dataset documents the same order, units,
frame, and gripper sign. The official training example uses separate transforms and range scaling
for different contributors for exactly this reason.

## Official checkpoint

The [official repository](https://github.com/google-deepmind/open_x_embodiment) publishes inference
code and the RT-1-X JAX checkpoint download command:

```bash
gsutil -m cp -r \
  gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_jax .
```

Use `model_sample["observation"]` and `model_sample["action_tokens"]` with that code. Generate the
512-D language embedding using the same encoder as the checkpoint; a semantically similar but
different embedding model is not checkpoint-compatible.

## Sources

- [Project page and paper](https://robotics-transformer-x.github.io/)
- [Official Open X-Embodiment code and checkpoints](https://github.com/google-deepmind/open_x_embodiment)
- [RLDS format](https://github.com/google-research/rlds)
