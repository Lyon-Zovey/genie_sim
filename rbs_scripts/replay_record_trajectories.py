#!/usr/bin/env python3
"""
Replay Metaworld trajectories with simultaneous camera-data recording.

Reads:
  trajectory.state.mocap_xyz.mujoco_cpu.h5
  trajectory.state.mocap_xyz.mujoco_cpu.json

Writes (inside <output_dir>/):
  trajectory.rgb+depth+segmentation.mocap_xyz.mujoco_cpu.h5
  trajectory.rgb+depth+segmentation.mocap_xyz.mujoco_cpu.json
  camera_data/
    traj_N/
      rgb.mp4                  H×W×3  uint8
      depth_video.npy          (T+1, H, W)  float16  metres
      seg.npy                  (T+1, H, W)  int32    MuJoCo body-IDs (-1 = bg)
      cam_poses.npy            (T+1, 4, 4)  float32  cam-to-world (OpenGL/SAPIEN conv.)
      cam_intrinsics.npy       (3, 3)       float32  pinhole K matrix
      traj_N.h5                copy of main traj_N group + id_poses/

id_poses/ layout (within each traj_N group):
  .attrs  {str(body_id): "body:<name>", ...}
  <body_id>/
    .attrs  {name: "body:<name>", seg_id: body_id}
    position          (T+1, 3)  float32  world-frame position
    quaternion        (T+1, 4)  float32  (w,x,y,z)
    camera_position   (T+1, 3)  float32  camera centre in world frame
    camera_quaternion (T+1, 4)  float32  (w,x,y,z)

Replay uses env_states (qpos + qvel + target_pos) for exact frame-by-frame
reproduction – no policy is needed.

Usage:
  python scripts/replay_record_trajectories.py \\
      --h5   rollout_data/reach-v3/trajectory.state.mocap_xyz.mujoco_cpu.h5 \\
      --json rollout_data/reach-v3/trajectory.state.mocap_xyz.mujoco_cpu.json \\
      --output-dir replay_data/reach-v3

  # only record successful trajectories
  python scripts/replay_record_trajectories.py \\
      --h5   rollout_data/reach-v3/trajectory.state.mocap_xyz.mujoco_cpu.h5 \\
      --json rollout_data/reach-v3/trajectory.state.mocap_xyz.mujoco_cpu.json \\
      --output-dir replay_data/reach-v3 --success-only

  # specify camera / resolution
  python scripts/replay_record_trajectories.py \\
      --h5  ... --json ... --output-dir ... \\
      --camera corner --width 640 --height 480 --fps 30
"""

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import h5py
import numpy as np

import mujoco
import metaworld

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OUT_OBS_MODE    = "rgb+depth+segmentation"
OUT_CTRL_MODE   = "mocap_xyz"
OUT_BACKEND     = "mujoco_cpu"
OUT_BASENAME    = f"trajectory.{OUT_OBS_MODE}.{OUT_CTRL_MODE}.{OUT_BACKEND}"


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def mat3_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix → quaternion (w, x, y, z) via scipy."""
    try:
        from scipy.spatial.transform import Rotation
        xyzw = Rotation.from_matrix(R.astype(np.float64)).as_quat()
        return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)
    except ImportError:
        # Shepperd's method (no scipy)
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            return np.array([0.25 / s,
                             (R[2, 1] - R[1, 2]) * s,
                             (R[0, 2] - R[2, 0]) * s,
                             (R[1, 0] - R[0, 1]) * s], dtype=np.float32)
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                             (R[0, 1] + R[1, 0]) / s,
                             (R[0, 2] + R[2, 0]) / s], dtype=np.float32)
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                             0.25 * s, (R[1, 2] + R[2, 1]) / s], dtype=np.float32)
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                             (R[1, 2] + R[2, 1]) / s, 0.25 * s], dtype=np.float32)


def build_env(env_name: str, seed: int, width: int, height: int):
    """Build a Metaworld env.

    We pass render_mode="rgb_array" together with the target resolution so that
    gymnasium's MujocoRenderer expands the model's offscreen framebuffer to at
    least (width, height) before we create our own mujoco.Renderer instances.
    Without this, mujoco.Renderer raises:
      ValueError: Image width N > framebuffer width M
    We do NOT call env.render() afterwards – all rendering goes through the
    three standalone mujoco.Renderer objects in main().
    """
    mt1 = metaworld.MT1(env_name, seed=seed)
    env_cls = mt1.train_classes[env_name]
    env = env_cls(render_mode="rgb_array", width=width, height=height)
    env.seed(seed)
    env.set_task(mt1.train_tasks[0])
    env.reset()
    return env, mt1


def get_camera_id(model, cam_name: str) -> int:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise ValueError(
            f"Camera '{cam_name}' not found in model. "
            "Available: corner, topview, behindGripper, gripperPOV"
        )
    return cam_id


def compute_intrinsics(model, cam_id: int, width: int, height: int,
                       fovy_deg: float | None = None) -> np.ndarray:
    """Pinhole K matrix from MuJoCo camera FOV (square pixels assumed).

    If fovy_deg is provided it overrides the model's stored value (useful when
    the camera pose has been overridden at runtime via override_camera_pose()).
    """
    if fovy_deg is None:
        fovy_deg = float(model.cam_fovy[cam_id])
    fy = (height / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
    fx = fy  # MuJoCo renders with square pixels
    cx = width  / 2.0
    cy = height / 2.0
    return np.array([[fx, 0., cx],
                     [0., fy, cy],
                     [0., 0., 1.]], dtype=np.float32)


def _lookat_to_rotation(pos: np.ndarray, lookat: np.ndarray,
                        up: np.ndarray | None = None) -> np.ndarray:
    """Compute a 3×3 cam-to-world rotation matrix (OpenGL convention).

    In OpenGL convention the camera looks along its *-z* axis, +y is up, +x
    is right.  MuJoCo's cam_xmat stores columns [right | up | back] in world
    coordinates (where back = +z_cam in world = the direction the camera
    points *away* from).

    Returns the 3×3 matrix whose columns are [right, up_corrected, back].
    """
    if up is None:
        up = np.array([0., 0., 1.], dtype=np.float64)
    forward = (lookat - pos).astype(np.float64)
    forward /= np.linalg.norm(forward)          # unit vector toward target
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:            # forward is parallel to up
        up = np.array([0., 1., 0.], dtype=np.float64)
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up_corrected = np.cross(right, forward)     # re-orthogonalise
    up_corrected /= np.linalg.norm(up_corrected)
    back = -forward                             # camera looks along -z in world
    # cam_xmat columns: right, up, back  (OpenGL / SAPIEN convention)
    return np.stack([right, up_corrected, back], axis=1).astype(np.float32)


def override_camera_pose(model, cam_id: int,
                         pos: np.ndarray,
                         lookat: np.ndarray | None = None,
                         quat_wxyz: np.ndarray | None = None,
                         fovy_deg: float | None = None) -> None:
    """Override camera position/orientation in the MuJoCo model (in-place).

    Exactly one of `lookat` or `quat_wxyz` must be provided to set orientation.

    Args:
        model:      mujoco.MjModel
        cam_id:     camera index from get_camera_id()
        pos:        (3,) world-frame camera position
        lookat:     (3,) world-frame point the camera points at
        quat_wxyz:  (4,) orientation quaternion (w, x, y, z)
        fovy_deg:   vertical field-of-view in degrees (optional)

    MuJoCo stores cam_pos0 / cam_mat0 as the *initial* pose and copies them
    to cam_xpos / cam_xmat every time mj_kinematics() is called (for fixed
    cameras with mode=0).  We overwrite cam_pos0 and cam_mat0 so the change
    persists across all subsequent render calls without needing env.reset().
    """
    model.cam_pos0[cam_id] = pos.astype(np.float64)

    if lookat is not None:
        R = _lookat_to_rotation(pos, np.asarray(lookat, np.float64))
        # cam_mat0 is stored row-major (9 floats); columns = [right, up, back]
        model.cam_mat0[cam_id] = R.T.reshape(9)   # store as flattened row-major
    elif quat_wxyz is not None:
        try:
            from scipy.spatial.transform import Rotation
            w, x, y, z = quat_wxyz
            R = Rotation.from_quat([x, y, z, w]).as_matrix().astype(np.float32)
        except ImportError:
            raise RuntimeError("scipy is required for quat_wxyz override; "
                               "install it or use --cam-lookat instead.")
        model.cam_mat0[cam_id] = R.T.reshape(9)

    if fovy_deg is not None:
        model.cam_fovy[cam_id] = float(fovy_deg)


def get_cam_pose(data, cam_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (position (3,), rotation_matrix (3,3)) of camera in world frame.

    MuJoCo's cam_xmat uses OpenGL camera convention:
      +x right, +y up, -z forward  (z points *backward* out of lens)
    which matches the cam2world_gl convention used by MIKASA / SAPIEN.
    """
    cam_pos  = np.copy(data.cam_xpos[cam_id]).astype(np.float32)         # (3,)
    cam_xmat = np.copy(data.cam_xmat[cam_id]).reshape(3, 3).astype(np.float32)  # (3,3)
    return cam_pos, cam_xmat


def render_rgb(renderer: mujoco.Renderer, data, cam_id: int) -> np.ndarray:
    """Render RGB (H, W, 3) uint8.

    mujoco.Renderer internally applies np.flipud() (EGL/OSMesa/GLFW backends),
    so row-0 = top of image = camera +y direction.
    For Metaworld's corner camera, the y-axis is (-0.2, -0.2, -1) – mostly
    *downward* in world coordinates – so after that internal flip, world-down
    ends up at the image top, making the scene appear upside-down.
    We apply one more [::-1] to put world-up back at the image top.
    """
    renderer.update_scene(data, camera=cam_id)
    return renderer.render()[::-1].copy()   # (H, W, 3) uint8


def render_depth(renderer: mujoco.Renderer, data, cam_id: int) -> np.ndarray:
    """Render depth (H, W) float32 metres.  Same extra flip as render_rgb."""
    renderer.update_scene(data, camera=cam_id)
    return renderer.render()[::-1].copy()   # (H, W) float32


def render_seg_body_ids(renderer: mujoco.Renderer, data, cam_id: int,
                        model) -> np.ndarray:
    """Render per-pixel MuJoCo body IDs (0 = background).  Same extra flip.

    mujoco.Renderer with enable_segmentation_rendering() returns (H, W, 2) int32:
      [..., 0] = objid    – object index within its type (geom_id when type==GEOM)
      [..., 1] = objtype  – mjtObj enum value (mjOBJ_GEOM=5, mjOBJ_SITE=6, …)
    Background pixels have both channels == -1.

    Background is mapped to 0 (world-body ID) so that convert_camera_depths.py
    can identify background with ``seg_flat == 0``.  In practice no geom is
    ever assigned to body-0 (the MuJoCo world body), so 0 is unambiguous.
    """
    renderer.update_scene(data, camera=cam_id)
    seg_raw = renderer.render()[::-1]   # (H, W, 2) int32, flip corrects orientation

    # Select pixels where a visible geom was rendered
    geom_mask = (
        (seg_raw[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) &   # channel 1 = objtype
        (seg_raw[..., 0] >= 0)                              # channel 0 = objid (valid)
    )
    # Safely map geom_id → body_id via model.geom_bodyid
    safe_geom_ids = np.clip(seg_raw[..., 0], 0, model.ngeom - 1)
    body_ids      = model.geom_bodyid[safe_geom_ids]
    return np.where(geom_mask, body_ids, 0).astype(np.int32)  # (H, W)


def save_rgb_video(frames: list[np.ndarray], path: str, fps: float) -> None:
    """Save list of (H,W,3) uint8 frames as MP4.

    Tries h264_nvenc (GPU) first, falls back to libx264 (CPU) if unavailable.
    """
    import subprocess as _sp
    frames_arr = np.stack(frames, axis=0)  # (T,H,W,3) uint8
    T, H, W, _ = frames_arr.shape

    def _ffmpeg_encode(codec: str) -> bool:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(int(fps)), "-i", "pipe:0",
            "-c:v", codec,
        ]
        if codec == "h264_nvenc":
            cmd += ["-preset", "p4", "-rc", "constqp", "-qp", "18", "-pix_fmt", "yuv420p"]
        else:
            cmd += ["-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p"]
        cmd.append(path)
        p = _sp.Popen(cmd, stdin=_sp.PIPE, stdout=_sp.DEVNULL, stderr=_sp.PIPE)
        _, err = p.communicate(input=frames_arr.tobytes())
        return p.returncode == 0

    if not _ffmpeg_encode("h264_nvenc"):
        ok = _ffmpeg_encode("libx264")
        if not ok:
            raise RuntimeError(f"ffmpeg encode failed for both h264_nvenc and libx264")


# ---------------------------------------------------------------------------
# Core replay + record
# ---------------------------------------------------------------------------

def replay_and_record_traj(
    env,
    mt1,
    in_h5f:     h5py.File,
    out_h5f:    h5py.File,
    ep_idx:     int,
    episode:    dict,
    cam_id:     int,
    cam_name:   str,
    cam_K:      np.ndarray,
    rgb_ren:    mujoco.Renderer,
    depth_ren:  mujoco.Renderer,
    seg_ren:    mujoco.Renderer,
    cam_data_root: Path,
    fps:        float,
    env_name:   str = "",
) -> None:
    """
    Replay one trajectory from env_states, recording camera data and id_poses.
    Writes into out_h5f[f"traj_{ep_idx}"] and camera_data/traj_{ep_idx}/.
    """
    traj_key = f"traj_{ep_idx}"
    if traj_key not in in_h5f:
        print(f"  [SKIP] {traj_key} not found in input H5")
        return

    in_grp = in_h5f[traj_key]
    qpos_seq   = in_grp["env_states/qpos"][:]        # (T+1, Dq)
    qvel_seq   = in_grp["env_states/qvel"][:]        # (T+1, Dv)
    has_target = "env_states/target_pos" in in_grp
    target_seq = in_grp["env_states/target_pos"][:] if has_target else None
    T_plus_1   = qpos_seq.shape[0]

    # ------------------------------------------------------------------ env setup
    task_idx = episode.get("task_idx", ep_idx % len(mt1.train_tasks))
    env.set_task(mt1.train_tasks[task_idx])
    env.reset()

    # ------------------------------------------------------------------ buffers
    rgb_frames   : list[np.ndarray] = []
    depth_frames : list[np.ndarray] = []
    seg_frames   : list[np.ndarray] = []
    cam_poses    : list[np.ndarray] = []  # (4,4)

    n_bodies = env.model.nbody
    # body 0 = world; track bodies 1..n_bodies-1
    body_ids_tracked = list(range(1, n_bodies))
    id_buf: dict[int, dict[str, list]] = {
        bid: {"position": [], "quaternion": [],
              "camera_position": [], "camera_quaternion": []}
        for bid in body_ids_tracked
    }

    # ------------------------------------------------------------------ loop over frames
    for t in range(T_plus_1):
        env.set_env_state((qpos_seq[t], qvel_seq[t]))
        if target_seq is not None:
            env._target_pos = target_seq[t].astype(np.float64)

        # RGB – use mujoco.Renderer directly (no gymnasium flip, see render_rgb docstring)
        rgb = render_rgb(rgb_ren, env.data, cam_id)                # (H, W, 3) uint8
        rgb_frames.append(rgb)

        # Depth (metres, float32)
        # Mask far-clipping-plane background pixels (sky / empty space) to 0 so
        # that convert_camera_depths.py's ``valid = z > 0`` filter removes them.
        depth_raw = render_depth(depth_ren, env.data, cam_id)
        if depth_raw.size > 0:
            far_thresh = float(depth_raw.max()) * 0.995
            depth_raw = np.where(depth_raw >= far_thresh, 0.0, depth_raw)
        depth_frames.append(depth_raw)

        # Segmentation body-IDs (background = 0)
        seg = render_seg_body_ids(seg_ren, env.data, cam_id, env.model)
        seg_frames.append(seg)

        # Camera pose (cam-to-world 4×4, OpenGL convention)
        cam_pos, cam_xmat = get_cam_pose(env.data, cam_id)
        T_c2w = np.eye(4, dtype=np.float32)
        T_c2w[:3, :3] = cam_xmat
        T_c2w[:3, 3]  = cam_pos
        cam_poses.append(T_c2w)

        # World-to-camera transform (R_w2c = R_c2w.T for orthonormal matrix)
        R_w2c = cam_xmat.T                          # (3,3)
        t_w2c = -(R_w2c @ cam_pos)                  # (3,)

        # id_poses: per-body world-frame pose AND camera-frame pose
        # camera_position/camera_quaternion = body's pose expressed in camera
        # coordinates, as expected by convert_camera_depths.py for anchor tracking.
        for bid in body_ids_tracked:
            # world-frame pose
            pos_world  = np.copy(env.data.xpos [bid]).astype(np.float32)  # (3,)
            quat_world = np.copy(env.data.xquat[bid]).astype(np.float32)  # (4,) w,x,y,z
            id_buf[bid]["position"].append(pos_world)
            id_buf[bid]["quaternion"].append(quat_world)

            # camera-frame pose
            pos_cam  = (R_w2c @ pos_world + t_w2c).astype(np.float32)    # (3,)
            body_xmat = np.copy(env.data.xmat[bid]).reshape(3, 3).astype(np.float32)
            R_body_cam = (R_w2c @ body_xmat).astype(np.float32)           # body→cam rot
            quat_cam   = mat3_to_quat_wxyz(R_body_cam)                    # (4,) w,x,y,z
            id_buf[bid]["camera_position"].append(pos_cam)
            id_buf[bid]["camera_quaternion"].append(quat_cam)

    # ------------------------------------------------------------------ save camera_data/traj_N/
    cam_traj_dir = cam_data_root / traj_key
    cam_traj_dir.mkdir(parents=True, exist_ok=True)

    # rgb.mp4
    save_rgb_video(rgb_frames, str(cam_traj_dir / "rgb.mp4"), fps)

    # depth_video.npy  (T+1, H, W) float16 metres
    np.save(str(cam_traj_dir / "depth_video.npy"),
            np.stack(depth_frames, axis=0).astype(np.float16))

    # seg.npy  (T+1, H, W) int32
    np.save(str(cam_traj_dir / "seg.npy"),
            np.stack(seg_frames, axis=0))

    # cam_poses.npy  (T+1, 4, 4) float32
    np.save(str(cam_traj_dir / "cam_poses.npy"),
            np.stack(cam_poses, axis=0))

    # cam_intrinsics.npy  (3, 3) float32  (constant for fixed camera)
    np.save(str(cam_traj_dir / "cam_intrinsics.npy"), cam_K)

    # camera_name.txt  – which camera was used for this trajectory
    (cam_traj_dir / "camera_name.txt").write_text(cam_name + "\n")

    # traj_task.json  – task metadata for downstream consumers
    traj_task = {
        "task_id":   env_name,
        "traj_name": traj_key,
        "actors": [
            {"seg_id": bid, "name": f"body:{env.model.body(bid).name}"}
            for bid in body_ids_tracked
            if not env.model.body(bid).name.startswith("link:")
        ],
        "links": [
            {"seg_id": bid, "name": f"link:{env.model.body(bid).name}"}
            for bid in body_ids_tracked
            if env.model.body(bid).name.startswith("link:")
        ],
    }
    with open(str(cam_traj_dir / "traj_task.json"), "w", encoding="utf-8") as _jf:
        json.dump(traj_task, _jf, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ write main H5 traj group
    # Copy all datasets from input (obs, actions, rewards, env_states, …)
    in_h5f.copy(traj_key, out_h5f)
    out_grp = out_h5f[traj_key]
    out_grp.attrs["camera_name"] = cam_name

    # id_poses/ group (MIKASA-compatible)
    id_grp = out_grp.create_group("id_poses", track_order=True)
    for bid in body_ids_tracked:
        body_name = env.model.body(bid).name
        id_grp.attrs[str(bid)] = f"body:{body_name}"

    for bid in body_ids_tracked:
        body_name = env.model.body(bid).name
        seg_grp = id_grp.create_group(str(bid), track_order=True)
        seg_grp.attrs["name"]   = f"body:{body_name}"
        seg_grp.attrs["seg_id"] = bid
        buf = id_buf[bid]
        seg_grp.create_dataset("position",
            data=np.array(buf["position"],          dtype=np.float32))
        seg_grp.create_dataset("quaternion",
            data=np.array(buf["quaternion"],         dtype=np.float32))
        seg_grp.create_dataset("camera_position",
            data=np.array(buf["camera_position"],    dtype=np.float32))
        seg_grp.create_dataset("camera_quaternion",
            data=np.array(buf["camera_quaternion"],  dtype=np.float32))

    # ------------------------------------------------------------------ per-episode traj_N.h5
    per_h5_path = cam_traj_dir / f"{traj_key}.h5"
    with h5py.File(str(per_h5_path), "w") as per_h5:
        try:
            out_h5f.copy(traj_key, per_h5)
        except Exception as e:
            print(f"  [WARN] per-episode H5 copy failed ({e}); writing children individually")
            for k in out_grp.keys():
                out_h5f.copy(f"{traj_key}/{k}", per_h5, name=k)

    print(
        f"  traj_{ep_idx:04d}  steps={episode['elapsed_steps']:3d}"
        f"  success={episode['success']}"
        f"  camera={cam_name}"
        f"  → {cam_traj_dir}"
    )


# ---------------------------------------------------------------------------
# Multi-camera single-replay variant
# ---------------------------------------------------------------------------

def replay_and_record_traj_multicam(
    env,
    mt1,
    in_h5f:      h5py.File,
    out_h5f:     h5py.File,
    ep_idx:      int,
    episode:     dict,
    cam_configs: list,   # list of {"cam_id": int, "cam_name": str, "cam_K": np.ndarray}
    rgb_ren:     mujoco.Renderer,
    depth_ren:   mujoco.Renderer,
    seg_ren:     mujoco.Renderer,
    cam_data_root: Path,
    fps:         float,
    env_name:    str = "",
) -> None:
    """Replay one trajectory, recording all cameras in a single env-state pass.

    Each camera's files are written to cam_data_root/traj_N/<cam_name>/.
    Body id_poses are computed in world frame once per frame, then projected
    into each camera's frame separately.
    """
    traj_key = f"traj_{ep_idx}"
    if traj_key not in in_h5f:
        print(f"  [SKIP] {traj_key} not found in input H5")
        return

    in_grp = in_h5f[traj_key]
    qpos_seq   = in_grp["env_states/qpos"][:]
    qvel_seq   = in_grp["env_states/qvel"][:]
    has_target = "env_states/target_pos" in in_grp
    target_seq = in_grp["env_states/target_pos"][:] if has_target else None
    T_plus_1   = qpos_seq.shape[0]

    task_idx = episode.get("task_idx", ep_idx % len(mt1.train_tasks))
    env.set_task(mt1.train_tasks[task_idx])
    env.reset()

    n_bodies = env.model.nbody
    body_ids_tracked = list(range(1, n_bodies))

    # Per-camera accumulators
    cam_names = [c["cam_name"] for c in cam_configs]
    rgb_frames   = {n: [] for n in cam_names}
    depth_frames = {n: [] for n in cam_names}
    seg_frames   = {n: [] for n in cam_names}
    cam_poses    = {n: [] for n in cam_names}
    # Per-camera body cam-frame buffers
    id_cam_pos  = {n: {bid: [] for bid in body_ids_tracked} for n in cam_names}
    id_cam_quat = {n: {bid: [] for bid in body_ids_tracked} for n in cam_names}
    # Shared world-frame body buffers
    id_buf_world: dict[int, dict[str, list]] = {
        bid: {"position": [], "quaternion": []}
        for bid in body_ids_tracked
    }

    for t in range(T_plus_1):
        env.set_env_state((qpos_seq[t], qvel_seq[t]))
        if target_seq is not None:
            env._target_pos = target_seq[t].astype(np.float64)

        # Collect world-frame body poses once per frame (camera-independent)
        for bid in body_ids_tracked:
            id_buf_world[bid]["position"].append(
                np.copy(env.data.xpos[bid]).astype(np.float32))
            id_buf_world[bid]["quaternion"].append(
                np.copy(env.data.xquat[bid]).astype(np.float32))

        # Render all cameras for this frame
        for cfg in cam_configs:
            cid   = cfg["cam_id"]
            cname = cfg["cam_name"]

            rgb = render_rgb(rgb_ren, env.data, cid)
            rgb_frames[cname].append(rgb)

            depth_raw = render_depth(depth_ren, env.data, cid)
            if depth_raw.size > 0:
                far_thresh = float(depth_raw.max()) * 0.995
                depth_raw = np.where(depth_raw >= far_thresh, 0.0, depth_raw)
            depth_frames[cname].append(depth_raw)

            seg = render_seg_body_ids(seg_ren, env.data, cid, env.model)
            seg_frames[cname].append(seg)

            cam_pos_v, cam_xmat = get_cam_pose(env.data, cid)
            T_c2w = np.eye(4, dtype=np.float32)
            T_c2w[:3, :3] = cam_xmat
            T_c2w[:3, 3]  = cam_pos_v
            cam_poses[cname].append(T_c2w)

            R_w2c = cam_xmat.T
            t_w2c = -(R_w2c @ cam_pos_v)

            for bid in body_ids_tracked:
                pos_world  = id_buf_world[bid]["position"][-1]
                body_xmat  = np.copy(env.data.xmat[bid]).reshape(3, 3).astype(np.float32)
                pos_cam    = (R_w2c @ pos_world + t_w2c).astype(np.float32)
                R_body_cam = (R_w2c @ body_xmat).astype(np.float32)
                quat_cam   = mat3_to_quat_wxyz(R_body_cam)
                id_cam_pos[cname][bid].append(pos_cam)
                id_cam_quat[cname][bid].append(quat_cam)

    # Copy input traj to main H5 once (use world-frame id_poses, first camera
    # for camera_position/quaternion so the file stays schema-compatible)
    in_h5f.copy(traj_key, out_h5f)
    out_grp = out_h5f[traj_key]
    first_cam_name = cam_configs[0]["cam_name"]
    out_grp.attrs["camera_name"] = "+".join(cam_names)

    id_grp = out_grp.create_group("id_poses", track_order=True)
    for bid in body_ids_tracked:
        body_name = env.model.body(bid).name
        id_grp.attrs[str(bid)] = f"body:{body_name}"
    for bid in body_ids_tracked:
        body_name = env.model.body(bid).name
        sg = id_grp.create_group(str(bid), track_order=True)
        sg.attrs["name"]   = f"body:{body_name}"
        sg.attrs["seg_id"] = bid
        sg.create_dataset("position",          data=np.array(id_buf_world[bid]["position"],       dtype=np.float32))
        sg.create_dataset("quaternion",        data=np.array(id_buf_world[bid]["quaternion"],      dtype=np.float32))
        sg.create_dataset("camera_position",   data=np.array(id_cam_pos[first_cam_name][bid],     dtype=np.float32))
        sg.create_dataset("camera_quaternion", data=np.array(id_cam_quat[first_cam_name][bid],    dtype=np.float32))

    # Write per-camera directories and per-camera H5 files
    traj_root = cam_data_root / traj_key
    for cfg in cam_configs:
        cname = cfg["cam_name"]
        cam_K = cfg["cam_K"]

        cam_dir = traj_root / cname
        cam_dir.mkdir(parents=True, exist_ok=True)

        save_rgb_video(rgb_frames[cname], str(cam_dir / "rgb.mp4"), fps)
        np.save(str(cam_dir / "depth_video.npy"),
                np.stack(depth_frames[cname], axis=0).astype(np.float16))
        np.save(str(cam_dir / "seg.npy"),
                np.stack(seg_frames[cname], axis=0))
        np.save(str(cam_dir / "cam_poses.npy"),
                np.stack(cam_poses[cname], axis=0))
        np.save(str(cam_dir / "cam_intrinsics.npy"), cam_K)
        (cam_dir / "camera_name.txt").write_text(cname + "\n")

        traj_task = {
            "task_id":   env_name,
            "traj_name": traj_key,
            "actors": [
                {"seg_id": bid, "name": f"body:{env.model.body(bid).name}"}
                for bid in body_ids_tracked
                if not env.model.body(bid).name.startswith("link:")
            ],
            "links": [
                {"seg_id": bid, "name": f"link:{env.model.body(bid).name}"}
                for bid in body_ids_tracked
                if env.model.body(bid).name.startswith("link:")
            ],
        }
        with open(str(cam_dir / "traj_task.json"), "w", encoding="utf-8") as _jf:
            json.dump(traj_task, _jf, ensure_ascii=False, indent=2)

        # Per-camera H5: camera-specific id_poses (camera_position/quaternion)
        per_h5_path = cam_dir / f"{traj_key}.h5"
        with h5py.File(str(per_h5_path), "w") as ph5:
            in_h5f.copy(traj_key, ph5)
            pg = ph5[traj_key]
            pg.attrs["camera_name"] = cname
            pid_grp = pg.create_group("id_poses", track_order=True)
            for bid in body_ids_tracked:
                body_name = env.model.body(bid).name
                pid_grp.attrs[str(bid)] = f"body:{body_name}"
            for bid in body_ids_tracked:
                body_name = env.model.body(bid).name
                sg = pid_grp.create_group(str(bid), track_order=True)
                sg.attrs["name"]   = f"body:{body_name}"
                sg.attrs["seg_id"] = bid
                sg.create_dataset("position",          data=np.array(id_buf_world[bid]["position"],   dtype=np.float32))
                sg.create_dataset("quaternion",        data=np.array(id_buf_world[bid]["quaternion"],  dtype=np.float32))
                sg.create_dataset("camera_position",   data=np.array(id_cam_pos[cname][bid],          dtype=np.float32))
                sg.create_dataset("camera_quaternion", data=np.array(id_cam_quat[cname][bid],         dtype=np.float32))

    print(
        f"  traj_{ep_idx:04d}  steps={episode['elapsed_steps']:3d}"
        f"  success={episode['success']}"
        f"  cameras={cam_names}"
        f"  → {traj_root}"
    )


# ---------------------------------------------------------------------------
# Parallel helpers (used when --num-procs > 1)
# ---------------------------------------------------------------------------

def _build_worker_cmd(
    args: argparse.Namespace, worker_id: int, w_start: int, w_end: int
) -> list[str]:
    """Build the subprocess argv list for one worker process."""
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--h5",          args.h5,
        "--json",        args.json,
        "--output-dir",  args.output_dir,
        "--fps",         str(args.fps),
        "--width",       str(args.width),
        "--height",      str(args.height),
        "--num-procs",   "1",
        "--_worker-id",    str(worker_id),
        "--_worker-start", str(w_start),
        "--_worker-end",   str(w_end),
    ]
    if args.all_trajs:
        cmd.append("--all-trajs")
    else:
        cmd += ["--traj-id", str(args.traj_id)]
    if args.success_only:
        cmd.append("--success-only")
    if getattr(args, "multi_cameras", None):
        cmd += ["--multi-cameras"] + args.multi_cameras
    elif args.random_camera:
        cmd.append("--random-camera")
        cmd += ["--cameras"] + args.cameras
    else:
        cmd += ["--camera", args.camera]
    if args.cam_pos is not None:
        cmd += ["--cam-pos"] + [str(v) for v in args.cam_pos]
    if args.cam_lookat is not None:
        cmd += ["--cam-lookat"] + [str(v) for v in args.cam_lookat]
    if args.cam_fovy is not None:
        cmd += ["--cam-fovy", str(args.cam_fovy)]
    return cmd


def _spawn_workers(args: argparse.Namespace, selected: list[int]) -> None:
    """Split *selected* across args.num_procs workers and merge outputs.

    Each worker writes:
      * _tmp_worker_K.h5          – its traj groups
      * _tmp_worker_K_eps.json    – its episode list
      * camera_data/traj_N/…      – shared dir, no conflict (different N per worker)
    This function merges the H5 and episode lists into the final outputs.
    """
    import math
    import subprocess

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(selected) == 0:
        raise RuntimeError("No trajectories selected for replay")

    n = min(args.num_procs, len(selected))
    chunk = math.ceil(len(selected) / n)
    slices = [
        (i * chunk, min((i + 1) * chunk, len(selected)))
        for i in range(n) if i * chunk < len(selected)
    ]

    print(f"\n[parallel] {len(selected)} trajs  →  {len(slices)} workers")

    procs: list[tuple[int, subprocess.Popen]] = []
    log_handles = []
    for k, (ws, we) in enumerate(slices):
        cmd = _build_worker_cmd(args, k, ws, we)
        log_path = out_dir / f"_tmp_worker_{k}.log"
        lf = open(str(log_path), "w")
        print(f"  worker {k}: selected[{ws}:{we}] ({we - ws} trajs)  log={log_path.name}")
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        procs.append((k, p))
        log_handles.append(lf)

    print()
    failed = []
    for (k, p), lf in zip(procs, log_handles):
        rc = p.wait()
        lf.close()
        if rc != 0:
            failed.append(k)
            print(f"  [ERROR] worker {k} exited {rc}  → {out_dir / f'_tmp_worker_{k}.log'}")
        else:
            print(f"  [OK]    worker {k}")

    if failed:
        raise RuntimeError(f"Worker(s) {failed} failed; check _tmp_worker_K.log")

    # ── Merge H5 ────────────────────────────────────────────────────────────
    out_h5_path = out_dir / f"{OUT_BASENAME}.h5"
    print(f"\n[merge] H5  → {out_h5_path}")
    with h5py.File(str(out_h5_path), "w") as dst:
        for k, _ in enumerate(slices):
            tmp_h5 = out_dir / f"_tmp_worker_{k}.h5"
            with h5py.File(str(tmp_h5), "r") as src:
                for key in sorted(src.keys()):   # traj_0, traj_1, …
                    src.copy(key, dst)
            tmp_h5.unlink()

    # ── Merge JSON episodes ─────────────────────────────────────────────────
    all_episodes: list[dict] = []
    for k, _ in enumerate(slices):
        tmp_eps = out_dir / f"_tmp_worker_{k}_eps.json"
        with open(str(tmp_eps)) as f:
            all_episodes.extend(json.load(f))
        tmp_eps.unlink()

    json_data  = load_json(args.json)
    env_info   = json_data["env_info"]
    out_json   = copy.deepcopy(json_data)
    out_json["env_info"]["env_kwargs"]["obs_mode"] = OUT_OBS_MODE
    cam_desc = (
        "multi:" + "+".join(args.multi_cameras) if getattr(args, "multi_cameras", None)
        else ("random:" + "+".join(args.cameras)) if args.random_camera
        else args.camera
    )
    out_json["source_desc"] = (
        f"Metaworld scripted policy replay+record "
        f"(env={env_info['env_id']}, camera={cam_desc})"
    )
    out_json["episodes"] = all_episodes
    out_json_path = out_dir / f"{OUT_BASENAME}.json"
    with open(str(out_json_path), "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"[merge] JSON → {out_json_path}")

    # ── Clean up worker logs (empty on success) ─────────────────────────────
    for k, _ in enumerate(slices):
        lp = out_dir / f"_tmp_worker_{k}.log"
        if lp.exists():
            lp.unlink()

    print(f"\nOutput H5   : {out_h5_path}")
    print(f"Output JSON : {out_json_path}")
    print(f"Camera data : {out_dir / 'camera_data'}")
    print("\nDone.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    json_data  = load_json(args.json)
    env_info   = json_data["env_info"]
    env_name   = env_info["env_id"]
    seed       = env_info["env_kwargs"].get("seed", 42)
    episodes   = json_data["episodes"]

    # Filter trajectories
    if args.all_trajs:
        selected = list(range(len(episodes)))
    else:
        selected = [args.traj_id]

    if args.success_only:
        selected = [i for i in selected if episodes[i]["success"]]
        print(f"  (--success-only: {len(selected)} successful trajectories)")

    if not selected:
        print("No trajectories to process. Exiting.")
        return

    # ── Parallel dispatch / worker slice ────────────────────────────────────
    if args.num_procs > 1 and args._worker_id is None:
        _spawn_workers(args, selected)
        return
    if args._worker_id is not None:
        selected = selected[args._worker_start : args._worker_end]

    # Output paths
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_data_root = out_dir / "camera_data"
    cam_data_root.mkdir(exist_ok=True)
    if args._worker_id is not None:
        out_h5_path = out_dir / f"_tmp_worker_{args._worker_id}.h5"
    else:
        out_h5_path = out_dir / f"{OUT_BASENAME}.h5"
    out_json_path = out_dir / f"{OUT_BASENAME}.json"

    # Build env + aux renderers
    print(f"\nBuilding env  : {env_name}  (seed={seed})")
    env, mt1 = build_env(env_name, seed, args.width, args.height)

    # Camera pool for (optional) per-trajectory randomisation
    camera_pool = args.cameras  # list[str], always set by argparse default
    multi_cameras = getattr(args, "multi_cameras", None) or []
    if multi_cameras:
        print(f"Camera mode   : MULTI  cameras={multi_cameras}")
    elif args.random_camera:
        print(f"Camera mode   : RANDOM  pool={camera_pool}")
    else:
        print(f"Camera mode   : fixed={args.camera}")
    print(f"Resolution    : {args.width}×{args.height}")
    print(f"Trajectories  : {selected}\n")

    # Optional runtime camera pose override (only valid for fixed-camera mode)
    if args.cam_pos is not None:
        if args.random_camera:
            print("[WARN] --cam-pos / --cam-lookat are ignored in --random-camera mode")
        else:
            cam_id_override = get_camera_id(env.model, args.camera)
            cam_pos_arr = np.array(args.cam_pos, dtype=np.float64)
            lookat_arr  = np.array(args.cam_lookat, dtype=np.float64) if args.cam_lookat else None
            override_camera_pose(
                env.model, cam_id_override,
                pos=cam_pos_arr,
                lookat=lookat_arr,
                fovy_deg=args.cam_fovy,
            )
            print(f"Camera pose overridden: pos={args.cam_pos}  "
                  f"lookat={args.cam_lookat}  fovy={args.cam_fovy}")

    # Three independent mujoco.Renderer instances – all use the same no-flip
    # convention so RGB / depth / seg are spatially consistent with each other.
    # They are camera-agnostic; cam_id is passed per render call.
    rgb_ren   = mujoco.Renderer(env.model, height=args.height, width=args.width)

    depth_ren = mujoco.Renderer(env.model, height=args.height, width=args.width)
    depth_ren.enable_depth_rendering()

    seg_ren = mujoco.Renderer(env.model, height=args.height, width=args.width)
    seg_ren.enable_segmentation_rendering()

    # Output JSON (copy + update obs_mode)
    out_json = copy.deepcopy(json_data)
    out_json["env_info"]["env_kwargs"]["obs_mode"] = OUT_OBS_MODE
    camera_desc = (
        "multi:" + "+".join(multi_cameras) if multi_cameras
        else "random:" + "+".join(camera_pool) if args.random_camera
        else args.camera
    )
    out_json["source_desc"] = (
        f"Metaworld scripted policy replay+record "
        f"(env={env_name}, seed={seed}, camera={camera_desc})"
    )

    # Track per-trajectory camera assignment so we can embed it in the JSON
    cam_name_per_traj: dict[int, str] = {}

    # Process trajectories
    with (
        h5py.File(args.h5,           "r") as in_h5f,
        h5py.File(str(out_h5_path),  "w") as out_h5f,
    ):
        for ep_idx in selected:
            if multi_cameras:
                cam_configs = []
                for cname in multi_cameras:
                    cid = get_camera_id(env.model, cname)
                    cK  = compute_intrinsics(env.model, cid, args.width, args.height)
                    cam_configs.append({"cam_id": cid, "cam_name": cname, "cam_K": cK})
                replay_and_record_traj_multicam(
                    env=env,
                    mt1=mt1,
                    in_h5f=in_h5f,
                    out_h5f=out_h5f,
                    ep_idx=ep_idx,
                    episode=episodes[ep_idx],
                    cam_configs=cam_configs,
                    rgb_ren=rgb_ren,
                    depth_ren=depth_ren,
                    seg_ren=seg_ren,
                    cam_data_root=cam_data_root,
                    fps=args.fps,
                    env_name=env_name,
                )
                cam_name_per_traj[ep_idx] = "+".join(multi_cameras)
            else:
                # --- per-trajectory camera selection ---
                if args.random_camera:
                    cam_name = random.choice(camera_pool)
                else:
                    cam_name = args.camera
                cam_name_per_traj[ep_idx] = cam_name

                cam_id = get_camera_id(env.model, cam_name)
                cam_K  = compute_intrinsics(env.model, cam_id, args.width, args.height)

                replay_and_record_traj(
                    env=env,
                    mt1=mt1,
                    in_h5f=in_h5f,
                    out_h5f=out_h5f,
                    ep_idx=ep_idx,
                    episode=episodes[ep_idx],
                    cam_id=cam_id,
                    cam_name=cam_name,
                    cam_K=cam_K,
                    rgb_ren=rgb_ren,
                    depth_ren=depth_ren,
                    seg_ren=seg_ren,
                    cam_data_root=cam_data_root,
                    fps=args.fps,
                    env_name=env_name,
                )

    # Write output JSON – embed the chosen camera name into each episode record
    out_episodes = []
    for i in selected:
        ep = dict(episodes[i])
        ep["camera"] = cam_name_per_traj[i]
        out_episodes.append(ep)
    if args._worker_id is not None:
        # Worker mode: write only the episodes list; orchestrator merges.
        tmp_eps = out_dir / f"_tmp_worker_{args._worker_id}_eps.json"
        with open(str(tmp_eps), "w") as f:
            json.dump(out_episodes, f, indent=2)
    else:
        out_json["episodes"] = out_episodes
        with open(str(out_json_path), "w") as f:
            json.dump(out_json, f, indent=2)

    rgb_ren.close()
    depth_ren.close()
    seg_ren.close()
    env.close()

    print(f"\nOutput H5   : {out_h5_path}")
    print(f"Output JSON : {out_json_path}")
    print(f"Camera data : {cam_data_root}")
    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay Metaworld trajectories and record RGB/depth/seg/id_poses."
    )
    p.add_argument("--h5",   required=True,
                   help="Path to trajectory.state.mocap_xyz.mujoco_cpu.h5")
    p.add_argument("--json", required=True,
                   help="Path to trajectory.state.mocap_xyz.mujoco_cpu.json")
    p.add_argument("--output-dir", required=True,
                   help="Directory where output H5, JSON and camera_data/ are written.")
    p.add_argument(
        "--traj-id", type=int, default=0,
        help="Index of the single trajectory to process (ignored with --all-trajs).",
    )
    p.add_argument(
        "--all-trajs", action="store_true",
        help="Process all trajectories in the file.",
    )
    p.add_argument(
        "--success-only", action="store_true",
        help="Only process trajectories where success=True.",
    )
    p.add_argument("--fps",    type=float, default=30.0, help="FPS for rgb.mp4.")
    p.add_argument(
        "--multi-cameras", type=str, nargs="+", default=None,
        metavar="CAM",
        help="Record multiple cameras in one replay pass. Each camera gets its own "
             "camera_data/traj_N/<cam_name>/ subdir with full sceneflow-ready data. "
             "E.g. --multi-cameras corner corner2 corner3",
    )
    p.add_argument("--camera", type=str,   default="corner",
                   help="MuJoCo camera name (used when --random-camera is NOT set): "
                        "corner | topview | corner2 | corner3 | corner4")
    p.add_argument(
        "--random-camera", action="store_true",
        help="Randomly pick a camera from --cameras for each trajectory independently.",
    )
    p.add_argument(
        "--cameras", type=str, nargs="+",
        default=["corner", "corner2", "corner3"],
        metavar="CAM",
        help="Camera pool used with --random-camera. "
             "Default: corner corner2 corner3",
    )
    p.add_argument("--width",  type=int,   default=640)
    p.add_argument("--height", type=int,   default=480)

    # ---- optional camera pose override ----------------------------------------
    cam = p.add_argument_group(
        "camera pose override",
        "Override the selected camera's position/orientation at runtime. "
        "If --cam-pos is given, --cam-lookat is also required (or the camera "
        "keeps its original orientation from the XML)."
    )
    cam.add_argument(
        "--cam-pos", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="Camera position in world frame, e.g. --cam-pos -1.1 -0.4 0.6",
    )
    cam.add_argument(
        "--cam-lookat", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="World-frame point the camera looks at, e.g. --cam-lookat 0 0.6 0.2",
    )
    cam.add_argument(
        "--cam-fovy", type=float, default=None,
        help="Override vertical field-of-view in degrees (default: use XML value).",
    )

    # ---- parallel replay ------------------------------------------------
    p.add_argument(
        "--num-procs", type=int, default=1,
        help="Parallel worker sub-processes for trajectory replay (default: 1).",
    )
    # Internal args used by _spawn_workers(); not for direct invocation.
    p.add_argument("--_worker-id",    type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker-start", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker-end",   type=int, default=None, help=argparse.SUPPRESS)

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
