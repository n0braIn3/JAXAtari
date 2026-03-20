import os
from typing import Dict, Union
import jax
import jax.numpy as jnp
import numpy as np
import wandb

from safetensors.flax import save_file, load_file
from flax.traverse_util import flatten_dict, unflatten_dict

def video_callback(states, dones, step, renderer, mod=False):
    """
    Starting a new thread to render video so that it doesn't block training.
    """
    # video_thread = threading.Thread(target=video_renderer, args=(states, dones, step, renderer, mod))
    # video_thread.start()
    try:
        video_renderer(states, dones, step, renderer, mod)
    except Exception as exc:
        # Never crash training/evaluation due to a best-effort debug video.
        print(f"Video rendering skipped due to callback error: {type(exc).__name__}: {exc}")

def video_renderer(states, dones, step, renderer, mod):
    print("Rendering video...")
    video_folder = f"{wandb.run.dir}/media/videos/"
    os.makedirs(video_folder, exist_ok=True)

    while hasattr(states, 'atari_state'):
        states = states.atari_state

    if hasattr(states, 'env_state'):
        states = states.env_state

    # num_states is where the first done is True; if there is no done signal,
    # use the full trajectory length from the first leaf in the state pytree.
    done_idx = int(jax.device_get(jnp.argmax(dones)))
    state_leaves = jax.tree_util.tree_leaves(states)
    if not state_leaves:
        print("Video skipped: no state leaves available.")
        return
    total_steps = int(np.shape(jax.device_get(state_leaves[0]))[0])
    num_states = done_idx if done_idx > 0 else total_steps
    max_steps = max(1, int(os.environ.get("VIDEO_MAX_STEPS", "1000")))
    num_states = min(num_states, max_steps)

    # Optional stride to reduce rendering/memory pressure on long traces.
    frame_stride = max(1, int(os.environ.get("VIDEO_FRAME_STRIDE", "4")))

    # Slice only along the leading time dimension and keep scalar leaves valid for vmap.
    def _slice_time_axis(x):
        x_arr = jnp.asarray(x)
        if x_arr.ndim == 0:
            return jnp.repeat(x_arr[None], num_states, axis=0)[::frame_stride]
        return x_arr[:num_states:frame_stride]

    # select every 4th frame (and only the first num_states)
    states_reduced = jax.tree_util.tree_map(_slice_time_axis, states)
    rasters = jax.vmap(renderer.render)(states_reduced)
    frames = np.array(rasters, dtype=np.uint8)
    # shape currently is (N, W, H, 3)
    # but should be (N, 3, W, H)
    frames = np.transpose(frames, (0, 3, 1, 2))

    fps = 30 
    video = wandb.Video(frames, fps=fps, format="mp4")
    name = f"video_{step}"
    if mod:
        name = f"video_{step}_mod"
    wandb.log({name: video})
    print("Video done.")

def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
    flattened_dict = flatten_dict(params, sep=',')
    save_file(flattened_dict, filename)

def load_params(filename:Union[str, os.PathLike]) -> Dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")
