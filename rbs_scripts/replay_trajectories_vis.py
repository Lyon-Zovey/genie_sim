#!/usr/bin/env python3
"""
Replay recorded Metaworld trajectories from H5/JSON files.

Two replay modes (controlled by --mode):
  state   (default)  Directly set qpos/qvel at every frame from env_states/.
                     Guaranteed pixel-perfect reproduction of the original rollout.
  action             Re-execute recorded actions through the environment.
                     Useful to verify policy determinism; may diverge slightly.

Two render modes (controlled by --render):
  human   (default)  Open a live MuJoCo viewer window (requires DISPLAY).
  video              Render offscreen and save an MP4 next to the H5 file.

Usage examples:
  # Watch traj_0 in GUI
  python scripts/replay_trajectories.py \
      --h5  rollout_data/door-lock-v3/trajectory.state.mocap_xyz.mujoco_cpu.h5 \
      --json rollout_data/door-lock-v3/trajectory.state.mocap_xyz.mujoco_cpu.json

  # Watch traj_3 only
  python scripts/replay_trajectories.py \
      --h5  rollout_data/door-lock-v3/trajectory.state.mocap_xyz.mujoco_cpu.h5 \
      --json rollout_data/door-lock-v3/trajectory.state.mocap_xyz.mujoco_cpu.json \
      --traj-id 3

  # Save all successful trajectories as MP4s (headless-friendly)
  python scripts/replay_trajectories.py \
      --h5  rollout_data/door-lock-v3/trajectory.state.mocap_xyz.mujoco_cpu.h5 \
      --json rollout_data/door-lock-v3/trajectory.state.mocap_xyz.mujoco_cpu.json \
      --render video --all-trajs --success-only

  # Re-run via recorded actions at 2x speed in GUI
  python scripts/replay_trajectories.py \
      --h5  rollout_data/door-lock-v3/trajectory.state.mocap_xyz.mujoco_cpu.h5 \
      --json rollout_data/door-lock-v3/trajectory.state.mocap_xyz.mujoco_cpu.json \
      --mode action --fps 60
"""

import argparse
import json
import os
import time

import h5py
import numpy as np

import metaworld
from metaworld.policies import ENV_POLICY_MAP  # only needed for action-replay label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def build_env(env_name: str, seed: int, render_mode: str, camera: str, width: int, height: int):
    """Construct and return a ready-to-use Metaworld env with a dummy task set."""
    mt1 = metaworld.MT1(env_name, seed=seed)
    env_cls = mt1.train_classes[env_name]
    env = env_cls(
        render_mode=render_mode,
        camera_name=camera,
        width=width,
        height=height,
    )
    # Set any task so internal guards pass; states will be overwritten anyway.
    env.set_task(mt1.train_tasks[0])
    env.reset()
    return env, mt1


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------

def _state_replay(env, h5f: h5py.File, traj_id: int, fps: float):
    """Set qpos/qvel (and target_pos when available) directly at each frame."""
    grp        = h5f[f"traj_{traj_id}"]
    qpos       = grp["env_states/qpos"][:]   # (T+1, qpos_dim)
    qvel       = grp["env_states/qvel"][:]   # (T+1, qvel_dim)
    has_target = "env_states/target_pos" in grp
    target_pos = grp["env_states/target_pos"][:] if has_target else None
    T          = qpos.shape[0]

    dt = 1.0 / fps
    frames = []

    for t in range(T):
        env.set_env_state((qpos[t], qvel[t]))
        if target_pos is not None:
            env._target_pos = target_pos[t].astype(np.float64)
        frame = env.render()
        if frame is not None:          # rgb_array mode
            frames.append(frame)
        time.sleep(dt)

    return frames


def _action_replay(env, h5f: h5py.File, traj_id: int, fps: float, mt1, env_name: str, seed: int, episodes: list):
    """Re-run recorded actions through env.step()."""
    grp      = h5f[f"traj_{traj_id}"]
    actions  = grp["actions"][:]   # (T, 4)
    T        = actions.shape[0]

    # Use stored task_idx when available (ensures correct goal is set)
    task_idx = episodes[traj_id].get("task_idx", traj_id % len(mt1.train_tasks))
    env.set_task(mt1.train_tasks[task_idx])
    env.reset()

    dt = 1.0 / fps
    frames = []

    for t in range(T):
        obs, reward, terminated, truncated, info = env.step(actions[t])
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        time.sleep(dt)
        if terminated or truncated:
            break

    return frames


# ---------------------------------------------------------------------------
# Save video
# ---------------------------------------------------------------------------

def save_video(frames: list, out_path: str, fps: float) -> None:
    try:
        import imageio.v3 as iio
        iio.imwrite(out_path, np.stack(frames, axis=0), fps=int(fps))
        print(f"    Saved video → {out_path}")
    except ImportError:
        # fallback: try imageio v2 API
        import imageio
        with imageio.get_writer(out_path, fps=int(fps)) as w:
            for f in frames:
                w.append_data(f)
        print(f"    Saved video → {out_path}")


# ---------------------------------------------------------------------------
# Main replay loop
# ---------------------------------------------------------------------------

def replay(args):
    json_data = load_json(args.json)
    env_info  = json_data["env_info"]
    env_name  = env_info["env_id"]
    seed      = env_info["env_kwargs"].get("seed", 42)
    episodes  = json_data["episodes"]

    # Decide which traj ids to show
    if args.all_trajs:
        traj_ids = list(range(len(episodes)))
    else:
        traj_ids = [args.traj_id]

    if args.success_only:
        traj_ids = [i for i in traj_ids if episodes[i]["success"]]
        print(f"  (--success-only: {len(traj_ids)} successful trajectories)")

    render_mode = "rgb_array" if args.render == "video" else "human"

    print(f"Env       : {env_name}")
    print(f"Mode      : {args.mode}  |  Render : {args.render}  |  FPS : {args.fps}")
    print(f"Trajs     : {traj_ids}")
    print()

    env, mt1 = build_env(env_name, seed, render_mode, args.camera, args.width, args.height)

    with h5py.File(args.h5, "r") as h5f:
        for traj_id in traj_ids:
            ep = episodes[traj_id]
            print(
                f"  traj_{traj_id:04d}  steps={ep['elapsed_steps']:3d}  "
                f"success={ep['success']}"
            )

            if args.mode == "state":
                frames = _state_replay(env, h5f, traj_id, args.fps)
            else:
                frames = _action_replay(env, h5f, traj_id, args.fps, mt1, env_name, seed, episodes)

            if args.render == "video" and frames:
                out_path = os.path.join(
                    os.path.dirname(args.h5),
                    f"replay_traj{traj_id:04d}.mp4",
                )
                save_video(frames, out_path, args.fps)

            if args.render == "human" and len(traj_ids) > 1:
                print("    (press Ctrl+C to skip to next trajectory)")
                time.sleep(1.0)

    env.close()
    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Replay Metaworld trajectories from H5/JSON.")

    p.add_argument("--h5",   required=True, help="Path to trajectories.h5")
    p.add_argument("--json", required=True, help="Path to trajectories.json")

    p.add_argument(
        "--traj-id", type=int, default=0,
        help="Index of the trajectory to replay (default: 0). Ignored if --all-trajs.",
    )
    p.add_argument(
        "--all-trajs", action="store_true",
        help="Replay all trajectories one by one.",
    )
    p.add_argument(
        "--mode", choices=["state", "action"], default="state",
        help="state: restore qpos/qvel directly (exact); action: re-run actions.",
    )
    p.add_argument(
        "--render", choices=["human", "video"], default="human",
        help="human: live GUI window; video: save MP4 files.",
    )
    p.add_argument(
        "--success-only", action="store_true",
        help="Only replay trajectories where success=True.",
    )
    p.add_argument("--fps",    type=float, default=30.0, help="Playback / video FPS.")
    p.add_argument("--camera", type=str,   default="corner",
                   help="Camera name: corner | topview | behindGripper | gripperPOV")
    p.add_argument("--width",  type=int,   default=640)
    p.add_argument("--height", type=int,   default=480)

    return p.parse_args()


if __name__ == "__main__":
    replay(parse_args())
