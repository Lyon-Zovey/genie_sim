#!/usr/bin/env python3
"""
Rollout Metaworld scripted policies and save trajectories in MIKASA-Robo-aligned format.

Output layout:
  <output_dir>/<env_name>/
      trajectory.state.mocap_xyz.mujoco_cpu.h5
      trajectory.state.mocap_xyz.mujoco_cpu.json

H5 layout (per traj_N):
  obs/
    agent/
      qpos          (T+1, qpos_dim)  float64   full MuJoCo qpos (robot + objects)
      qvel          (T+1, qvel_dim)  float64   full MuJoCo qvel
    extra/
      tcp_pos       (T+1, 3)         float32   end-effector XYZ
      tcp_gripper   (T+1, 1)         float32   gripper opening (normalised 0-1)
      obj_pos       (T+1, 3)         float32   object XYZ
      obj_quat      (T+1, 4)         float32   object quaternion (w,x,y,z)
      goal_pos      (T+1, 3)         float32   target/goal XYZ
      flat_obs      (T+1, 39)        float32   full 39-dim state obs (compat.)
  actions           (T, 4)           float32
  rewards           (T,)             float32
  terminated        (T,)             bool
  truncated         (T,)             bool
  success           (T,)             bool
  env_states/
    qpos            (T+1, qpos_dim)  float64   same as obs/agent/qpos
    qvel            (T+1, qvel_dim)  float64   same as obs/agent/qvel
    target_pos      (T+1, 3)         float32   goal xyz (for exact replay)

JSON: mirrors MIKASA-Robo rbs_record.py schema

39-dim flat obs layout (Metaworld convention):
  [0:3]   curr hand XYZ
  [3]     curr gripper distance (normalised)
  [4:7]   curr obj XYZ
  [7:11]  curr obj quaternion (x,y,z,w – MuJoCo convention in obs)
  [11:18] padding zeros (single-object envs)
  [18:21] prev hand XYZ  (frame-stacked)
  [21]    prev gripper
  [22:25] prev obj XYZ
  [25:29] prev obj quaternion
  [29:36] prev padding
  [36:39] goal XYZ

Usage:
  python scripts/rollout_scripted_policy.py --env-name reach-v3 --num-episodes 50
  python scripts/rollout_scripted_policy.py --all-envs --num-episodes 50
"""

import argparse
import json
import os
import subprocess

import h5py
import numpy as np

import metaworld
from metaworld.policies import ENV_POLICY_MAP

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRAJ_BASENAME = "trajectory.state.mocap_xyz.mujoco_cpu"


def get_commit_info() -> dict:
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return {"git_commit": commit}
    except Exception:
        return {}


def decompose_obs(obs_flat: np.ndarray, env) -> dict:
    """
    Split the 39-dim Metaworld flat observation into a nested dict that mirrors
    MIKASA-Robo's obs/agent/* and obs/extra/* structure.

    obs/agent/qpos and qvel are taken directly from the MuJoCo data object so
    they are guaranteed to be consistent with env_states/qpos and qvel.
    """
    return {
        "agent": {
            "qpos": np.copy(env.data.qpos).astype(np.float64),
            "qvel": np.copy(env.data.qvel).astype(np.float64),
        },
        "extra": {
            "tcp_pos":     obs_flat[0:3].astype(np.float32),
            "tcp_gripper": obs_flat[3:4].astype(np.float32),
            "obj_pos":     obs_flat[4:7].astype(np.float32),
            "obj_quat":    obs_flat[7:11].astype(np.float32),
            "goal_pos":    obs_flat[36:39].astype(np.float32),
            "flat_obs":    obs_flat.astype(np.float32),
        },
    }


def stack_nested(lst: list) -> dict | np.ndarray:
    """Recursively stack a list of (nested) dicts into a dict of arrays."""
    if isinstance(lst[0], dict):
        return {k: stack_nested([x[k] for x in lst]) for k in lst[0]}
    return np.stack(lst, axis=0)


def recursive_add_to_h5(group: h5py.Group, data: dict | np.ndarray, key: str) -> None:
    """Write a nested-dict / array tree into an h5py group (with gzip compression)."""
    if isinstance(data, dict):
        subgrp = group.require_group(key)
        for k, v in data.items():
            recursive_add_to_h5(subgrp, v, k)
    else:
        group.create_dataset(
            key,
            data=np.asarray(data),
            compression="gzip",
            compression_opts=4,
        )


# ---------------------------------------------------------------------------
# Core rollout
# ---------------------------------------------------------------------------

def rollout_single_env(
    env_name: str,
    num_episodes: int,
    seed: int,
    max_steps: int,
    stop_on_success: bool = True,
) -> list[dict]:
    """
    Collect `num_episodes` successful rollouts of the scripted policy for `env_name`.
    Returns a list of trajectory dicts with raw numpy arrays and JSON scalars.
    """
    if env_name not in ENV_POLICY_MAP:
        raise ValueError(
            f"'{env_name}' has no scripted policy. "
            f"Available: {sorted(ENV_POLICY_MAP.keys())}"
        )

    mt1 = metaworld.MT1(env_name, seed=seed)
    env_cls = mt1.train_classes[env_name]
    env = env_cls()
    env.seed(seed)

    policy = ENV_POLICY_MAP[env_name]()
    tasks = mt1.train_tasks
    trajectories = []

    attempt_idx = 0
    while len(trajectories) < num_episodes:
        task_idx = attempt_idx % len(tasks)
        env.set_task(tasks[task_idx])
        obs, _ = env.reset()

        # Capture state right after reset (t=0)
        qpos_init, qvel_init = env.get_env_state()
        target_init = np.copy(env._target_pos).astype(np.float32)

        obs_buf         = [decompose_obs(obs, env)]
        qpos_buf        = [qpos_init.copy().astype(np.float64)]
        qvel_buf        = [qvel_init.copy().astype(np.float64)]
        target_buf      = [target_init]
        action_buf      = []
        reward_buf      = []
        terminated_buf  = []
        truncated_buf   = []
        success_buf     = []

        step_info: dict = {}
        done = False
        t = 0

        while t < max_steps and not done:
            action = policy.get_action(obs)
            obs, reward, terminated, truncated, step_info = env.step(action)

            qpos, qvel = env.get_env_state()
            success = bool(step_info.get("success", 0))

            obs_buf.append(decompose_obs(obs, env))
            qpos_buf.append(qpos.copy().astype(np.float64))
            qvel_buf.append(qvel.copy().astype(np.float64))
            target_buf.append(np.copy(env._target_pos).astype(np.float32))

            action_buf.append(action.astype(np.float32))
            reward_buf.append(float(reward))
            terminated_buf.append(bool(terminated))
            truncated_buf.append(bool(truncated))
            success_buf.append(success)

            done = terminated or truncated or (stop_on_success and success)
            t += 1

        ep_success = bool(step_info.get("success", False))

        # Stack obs list into nested dict of arrays
        obs_stacked = stack_nested(obs_buf)  # dict of (T+1, ...) arrays

        if ep_success:
            trajectories.append(
                {
                    # JSON scalars
                    "episode_seed": seed + attempt_idx,
                    "task_idx":     task_idx,
                    "elapsed_steps": t,
                    "success":      ep_success,
                    # H5 arrays
                    "obs":          obs_stacked,                                    # nested dict
                    "actions":      np.stack(action_buf, axis=0),                  # (T, 4)
                    "rewards":      np.array(reward_buf,     dtype=np.float32),    # (T,)
                    "terminated":   np.array(terminated_buf, dtype=bool),          # (T,)
                    "truncated":    np.array(truncated_buf,  dtype=bool),          # (T,)
                    "success_seq":  np.array(success_buf,    dtype=bool),          # (T,)
                    "qpos":         np.stack(qpos_buf,       axis=0),              # (T+1, Dq)
                    "qvel":         np.stack(qvel_buf,       axis=0),              # (T+1, Dv)
                    "target_pos":   np.stack(target_buf,     axis=0),              # (T+1, 3)
                }
            )

        attempt_idx += 1

    env.close()
    return trajectories


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_trajectories(
    trajectories: list[dict],
    env_name: str,
    seed: int,
    max_steps: int,
    output_dir: str,
    source_type: str = "scripted",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    h5_path   = os.path.join(output_dir, f"{TRAJ_BASENAME}.h5")
    json_path = os.path.join(output_dir, f"{TRAJ_BASENAME}.json")

    json_data: dict = {
        "env_info": {
            "env_id": env_name,
            "env_kwargs": {
                "obs_mode":    "state",
                "control_mode": "mocap_xyz",
                "sim_backend":  "mujoco_cpu",
                "seed":         seed,
            },
            "max_episode_steps": max_steps,
        },
        "commit_info":  get_commit_info(),
        "source_type":  source_type,
        "source_desc":  (
            f"Metaworld scripted policy rollout "
            f"(env={env_name}, seed={seed})"
        ),
        "episodes": [],
    }

    with h5py.File(h5_path, "w") as h5f:
        for ep_idx, traj in enumerate(trajectories):
            grp = h5f.create_group(f"traj_{ep_idx}")

            # obs/ – nested: agent/{qpos,qvel}, extra/{tcp_pos,...,flat_obs}
            recursive_add_to_h5(grp, traj["obs"], "obs")

            # transitions
            recursive_add_to_h5(grp, traj["actions"],    "actions")
            recursive_add_to_h5(grp, traj["rewards"],    "rewards")
            recursive_add_to_h5(grp, traj["terminated"], "terminated")
            recursive_add_to_h5(grp, traj["truncated"],  "truncated")
            recursive_add_to_h5(grp, traj["success_seq"], "success")

            # env_states/ – stores full physics state for exact state-replay
            es = {
                "qpos":       traj["qpos"],
                "qvel":       traj["qvel"],
                "target_pos": traj["target_pos"],
            }
            recursive_add_to_h5(grp, es, "env_states")

            json_data["episodes"].append(
                {
                    "episode_id":    ep_idx,
                    "episode_seed":  traj["episode_seed"],
                    "task_idx":      traj["task_idx"],
                    "control_mode":  "mocap_xyz",
                    "elapsed_steps": traj["elapsed_steps"],
                    "reset_kwargs":  {},
                    "success":       traj["success"],
                }
            )

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    n_total = len(trajectories)
    print(f"  Saved {n_total} successful episodes → {output_dir}")
    print(f"  H5:   {h5_path}")
    print(f"  JSON: {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rollout Metaworld scripted policies → MIKASA-Robo format."
    )
    parser.add_argument(
        "--env-name", type=str, default="door-lock-v3",
        help="Metaworld task name (e.g. reach-v3). Ignored with --all-envs.",
    )
    parser.add_argument(
        "--all-envs", action="store_true",
        help="Rollout all envs that have a scripted policy.",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=50,
        help="Number of episodes to collect per environment.",
    )
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Hard step limit per episode.")
    parser.add_argument(
        "--no-stop-on-success", action="store_true",
        help="Keep stepping even after success (full horizon).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="rollout_data",
        help="Root output directory. Data goes into <output_dir>/<env_name>/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_names = list(ENV_POLICY_MAP.keys()) if args.all_envs else [args.env_name]

    for env_name in env_names:
        print(f"\n{'='*55}")
        print(f"  env : {env_name}")
        print(f"  target successful eps : {args.num_episodes}   seed : {args.seed}")
        print(f"{'='*55}")

        trajs = rollout_single_env(
            env_name=env_name,
            num_episodes=args.num_episodes,
            seed=args.seed,
            max_steps=args.max_steps,
            stop_on_success=not args.no_stop_on_success,
        )
        out_dir = os.path.join(args.output_dir, env_name)
        save_trajectories(
            trajectories=trajs,
            env_name=env_name,
            seed=args.seed,
            max_steps=args.max_steps,
            output_dir=out_dir,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
