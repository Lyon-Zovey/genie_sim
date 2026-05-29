import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import h5py
import numpy as np


def depth_to_camera_points(
    depth: np.ndarray, K: np.ndarray, far_mm: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one depth image to camera-space points.

    Returns:
        pts_cam: (N,3)
        valid_mask: (H*W,) bool mask used during flattening
    """
    H, W = depth.shape
    if far_mm is not None:
        depth = np.where((depth == 0) | (depth == -32768), far_mm, depth)

    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v, indexing="xy")

    z_mm = depth.reshape(-1).astype(np.float32)
    valid = (z_mm > 0) & np.isfinite(z_mm)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), valid

    uu = uu.reshape(-1)[valid].astype(np.float32)
    vv = vv.reshape(-1)[valid].astype(np.float32)
    z_m = z_mm[valid]

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    x = (uu - cx) * z_m / fx
    # render_depth applies [::-1] so row-0 = camera -y; y increases with v
    y = (vv - cy) * z_m / fy
    z_cam = -z_m
    return np.stack([x, y, z_cam], axis=1), valid


def quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    q = quat
    if q.ndim == 1:
        q = q[None, ...]
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    rot = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    rot[:, 0, 0] = 1 - 2 * (yy + zz)
    rot[:, 0, 1] = 2 * (xy - wz)
    rot[:, 0, 2] = 2 * (xz + wy)
    rot[:, 1, 0] = 2 * (xy + wz)
    rot[:, 1, 1] = 1 - 2 * (xx + zz)
    rot[:, 1, 2] = 2 * (yz - wx)
    rot[:, 2, 0] = 2 * (xz - wy)
    rot[:, 2, 1] = 2 * (yz + wx)
    rot[:, 2, 2] = 1 - 2 * (xx + yy)
    if rot.shape[0] == 1:
        return rot[0]
    return rot


def find_h5_with_id_poses(folder: Path):
    for p in folder.glob("*.h5"):
        try:
            with h5py.File(p, "r") as f:
                for k in f.keys():
                    g = f[k]
                    if isinstance(g, h5py.Group) and "id_poses" in g.keys():
                        return p, k
        except Exception:
            continue
    return None, None


def _load_tracking_context(folder: Path):
    """Load reusable tracking context once per folder (for speed).

    The full ``seg`` array is kept in the context so that each call to
    ``track_anchor_file_exact`` can use ``seg[anchor_idx]`` as the
    pixel→sid mapping — matching the MIKASA-Robo convention.  Using a
    hard-coded ``seg[0]`` for every anchor is wrong: it makes all five
    sceneflow files look identical and silently drops actors that were not
    visible at frame 0.
    """
    seg_candidates = [folder / "seg.npy", folder / "segmentation.npy"]
    seg_path = None
    for c in seg_candidates:
        if c.exists():
            seg_path = c
            break
    if seg_path is None:
        raise FileNotFoundError(f"seg.npy not found in {folder}")

    seg_all = np.load(seg_path, allow_pickle=False)  # (T, H, W) or (H, W)
    if seg_all.ndim == 2:
        seg_all = seg_all[None, ...]                  # treat as single frame
    pixel_shape = (int(seg_all.shape[1]), int(seg_all.shape[2]))

    h5_path, traj_group = find_h5_with_id_poses(folder)
    if h5_path is None:
        raise FileNotFoundError(f"No .h5 with id_poses found in {folder}")

    sid_data = {}
    T = None
    with h5py.File(h5_path, "r") as f_h5:
        if traj_group is None:
            raise RuntimeError("traj_group is None")
        traj = f_h5[traj_group]
        if not isinstance(traj, h5py.Group):
            raise RuntimeError(f"{traj_group} is not group")
        id_poses = traj["id_poses"]
        if not isinstance(id_poses, h5py.Group):
            raise RuntimeError("id_poses is not group")

        for key in id_poses.keys():
            g = id_poses[key]
            if not isinstance(g, h5py.Group):
                continue

            if "camera_position" in g and "camera_quaternion" in g:
                pos_arr = np.asarray(g["camera_position"])
                quat_arr = np.asarray(g["camera_quaternion"])
            elif "position" in g and "quaternion" in g:
                pos_arr = np.asarray(g["position"])
                quat_arr = np.asarray(g["quaternion"])
            else:
                continue

            if pos_arr.ndim == 3 and pos_arr.shape[1] > 1:
                pos_arr = pos_arr[:, 0, :]
            if quat_arr.ndim == 3 and quat_arr.shape[1] > 1:
                quat_arr = quat_arr[:, 0, :]

            pos_arr = pos_arr.astype(np.float32)
            quat_arr = quat_arr.astype(np.float32)

            if T is None:
                T = int(pos_arr.shape[0])

            sid_data[str(int(key))] = {
                "pos": pos_arr,
                "rot": quat_to_rot_matrix(quat_arr),
            }

    if T is None:
        raise RuntimeError("Cannot determine T from id_poses")

    return {
        "seg_all": seg_all,        # full (T, H, W) — per-anchor slice in tracker
        "pixel_shape": pixel_shape,
        "sid_data": sid_data,
        "T": int(T),
    }


def track_anchor_file_exact(
    anchor_path: Path,
    out_path: Path,
    context: dict | None = None,
    anchor_array: np.ndarray | None = None,
    anchor_idx: int = 0,
):
    """Track anchor frame forward in time, producing a sceneflow tensor.

    The output has shape ``(T - anchor_idx, H, W, 3)`` so that frame 0 of the
    tensor always corresponds to the anchor frame itself, and subsequent frames
    show where each pixel's 3D point moves as the scene evolves.

    Key fix: body-local coordinates are computed using the body pose at
    ``anchor_idx``, NOT always at t=0.  Using t=0 for a non-zero anchor causes
    all sceneflow files to look identical (they all reconstruct the scene from
    the t=0 body pose perspective).
    """
    folder = anchor_path.parent
    if anchor_array is None:
        anchor = np.load(anchor_path, allow_pickle=False)
    else:
        anchor = anchor_array

    if context is None:
        context = _load_tracking_context(folder)

    seg_all    = context["seg_all"]     # (T_seg, H, W)
    pixel_shape = context["pixel_shape"]
    sid_data   = context["sid_data"]
    T          = context["T"]

    # Use seg[anchor_idx] for the pixel→sid mapping, NOT seg[0].
    # This matches the MIKASA-Robo convention and ensures actors that first
    # appear after frame 0 (or move significantly) are tracked correctly.
    a = max(0, min(anchor_idx, int(seg_all.shape[0]) - 1))
    seg_flat    = seg_all[a].reshape(-1)
    unique_sids = np.unique(seg_flat)

    if anchor.ndim == 3 and anchor.shape[2] == 3 and pixel_shape is not None and anchor.shape[0:2] == pixel_shape:
        H, W = pixel_shape
        pts = anchor.reshape(-1, 3)
        pixel_shape = (H, W)
    elif anchor.ndim == 2 and anchor.shape[1] == 3:
        pts = anchor
        if pixel_shape is not None:
            H, W = pixel_shape
            if pts.shape[0] != seg_flat.shape[0]:
                raise ValueError("Anchor N does not match seg H*W")
            pixel_shape = (H, W)
        else:
            pixel_shape = None
    else:
        raise ValueError(f"Unsupported anchor/seg shapes: {anchor.shape}")

    # Clamp anchor_idx to valid range
    anchor_idx = max(0, min(anchor_idx, T - 1))

    N = pts.shape[0]
    p_local = np.full((N, 3), np.nan, dtype=np.float32)

    for sid in unique_sids:
        if sid == 0:
            continue
        sid_str = str(int(sid))
        if sid_str not in sid_data:
            continue

        pose = sid_data[sid_str]
        # Use the body pose AT the anchor frame (not always t=0)
        pos_ref_idx = min(anchor_idx, int(pose["pos"].shape[0]) - 1)
        pos_a = pose["pos"][pos_ref_idx]
        R_a   = pose["rot"][pos_ref_idx] if pose["rot"].ndim == 3 else pose["rot"]

        T_a = np.eye(4, dtype=np.float32)
        T_a[:3, :3] = R_a
        T_a[:3, 3]  = pos_a
        T_a_inv = np.linalg.inv(T_a)

        mask = seg_flat == sid
        if not np.any(mask):
            continue
        pts_sel = pts[mask]
        homo = np.concatenate(
            [pts_sel[:, :3], np.ones((pts_sel.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        local_sel = (T_a_inv @ homo.T).T[:, :3]
        p_local[mask] = local_sel

    # Output: FULL trajectory (T frames), matching MIKASA-Robo convention.
    # frames[t] gives camera-space XYZ of each anchor-frame pixel at timestep t.
    # This means ref00070 still outputs (T, H, W, 3) — its pixels' positions
    # over the whole trajectory (including past frames before the anchor).
    frames = np.zeros((T, N, 3), dtype=np.float32)
    # Background pixels (sid==0): static — replicate anchor-frame positions
    bg_mask = seg_flat == 0
    if np.any(bg_mask):
        frames[:, bg_mask, :] = pts[bg_mask][None, :, :]

    for sid in unique_sids:
        if sid == 0:
            continue
        sid_str = str(int(sid))
        mask = seg_flat == sid
        if sid_str not in sid_data or not np.any(mask):
            continue

        pose = sid_data[sid_str]
        R_ts = pose["rot"]
        t_ts = pose["pos"]
        local_sel = p_local[mask]
        if np.isnan(local_sel).all():
            continue
        local_h = np.concatenate(
            [local_sel, np.ones((local_sel.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        for t in range(T):
            R    = R_ts[t] if R_ts.ndim == 3 else R_ts
            tvec = t_ts[t]
            Tt = np.eye(4, dtype=np.float32)
            Tt[:3, :3] = R
            Tt[:3, 3]  = tvec
            res = (Tt @ local_h.T).T[:, :3]
            frames[t, mask, :] = res

    if pixel_shape is not None:
        H, W = pixel_shape
        frames = frames.reshape((T, H, W, 3))

    np.save(out_path, frames.astype(np.float32))
    print(f"Tracked -> {out_path} shape={frames.shape} ref_frame={anchor_idx}")

def process_folder(folder: Path, out_name_template: str = "scene_point_flow_ref{idx:05d}.anchor.npy"):
    depth_path = folder / "depth_video.npy"
    if not depth_path.exists():
        return 0
    depth = np.load(depth_path, allow_pickle=False)
    # depth can be (T,H,W) or (H,W)
    if depth.ndim == 3:
        T, H, W = depth.shape
    elif depth.ndim == 2:
        T = 1
        depth = depth[None, ...]
    else:
        raise ValueError(f"Unsupported depth shape: {depth.shape}")

    intr_path = folder / "cam_intrinsics.npy"
    if intr_path.exists():
        K_all = np.load(intr_path, allow_pickle=False)
        if K_all.ndim == 3:
            # per-frame intrinsics
            def get_K(i):
                return K_all[i]
        elif K_all.ndim == 2:
            def get_K(i):
                return K_all
        else:
            raise ValueError(f"Unsupported intrinsics shape: {K_all.shape}")
    else:
        # fallback: try to read a generic file or fail
        raise FileNotFoundError(f"Missing cam_intrinsics.npy in {folder}")

    # choose frames: 0, round(1/4*(T-1)), round(2/4*(T-1)), round(3/4*(T-1)), T-1
    idxs = []
    if T >= 1:
        for i in range(5):
            idx = int(round(i * (T - 1) / 4.0))
            idxs.append(idx)
        # unique and sorted
        idxs = sorted(set(idxs))

    saved = 0
    # 目录级缓存，避免每个 ref 都重复读取 seg/h5/id_poses
    tracking_context = _load_tracking_context(folder)

    for idx in idxs:
        K = get_K(idx)
        frame_depth = depth[idx]
        pts_cam, valid = depth_to_camera_points(frame_depth, K)

        # 保存为 (H, W, 3) 格式，而不是 (N, 3)
        anchor_hw = np.zeros((H, W, 3), dtype=np.float32)
        anchor_flat = anchor_hw.reshape(-1, 3)
        anchor_flat[valid] = pts_cam.astype(np.float32)

        out_path = folder / out_name_template.format(idx=idx)
        np.save(out_path, anchor_hw)
        print(f"Saved {out_path} ({anchor_hw.shape[0]}x{anchor_hw.shape[1]} points)")

        # 代码层面融合：每生成一个 anchor，立刻 tracking 成对应的 scene_point_flow_refXXXXX.npy
        if out_path.name.endswith(".anchor.npy"):
            tracked_out_path = out_path.with_name(out_path.name.replace(".anchor.npy", ".npy"))
        else:
            tracked_out_path = out_path.with_suffix(".npy")
        track_anchor_file_exact(
            out_path,
            tracked_out_path,
            context=tracking_context,
            anchor_array=anchor_hw,
            anchor_idx=idx,      # use the correct anchor frame index
        )

        saved += 1

    if saved > 0:
        # Write a _DONE sentinel so that flow_compress.py's --out_root mode
        # picks this directory up automatically.
        (folder / "_DONE").touch(exist_ok=True)

    return saved


def _process_folder_worker(folder_path: str):
    folder = Path(folder_path)
    try:
        n = process_folder(folder)
        return folder_path, n, None
    except Exception as e:
        return folder_path, 0, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="camera_data", help="camera_data 根目录，包含若干子文件夹")
    parser.add_argument("--workers", type=int, default=1, help="并行处理子目录的进程数，默认1")
    args = parser.parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    # 支持直接传入单个traj目录（包含 depth_video.npy 或 depth.npy）
    if (root / "depth_video.npy").exists() or (root / "depth.npy").exists():
        n = process_folder(root)
        print(f"Done. Saved {n} files.")
        return

    # Collect leaf directories that contain depth_video.npy.
    # Supports two layouts:
    #   flat:   camera_data/traj_N/depth_video.npy
    #   nested: camera_data/traj_N/<cam_name>/depth_video.npy  (multi-camera)
    folders: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "depth_video.npy").exists():
            # flat single-camera layout
            folders.append(str(child))
        else:
            # nested multi-camera layout: one level deeper
            for subchild in sorted(child.iterdir()):
                if subchild.is_dir() and (subchild / "depth_video.npy").exists():
                    folders.append(str(subchild))

    total = 0

    if args.workers <= 1:
        for fp in folders:
            p = Path(fp)
            try:
                n = process_folder(p)
                if n > 0:
                    total += n
            except Exception as e:
                print(f"Skipped {p}: {e}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_process_folder_worker, fp) for fp in folders]
            for fut in as_completed(futures):
                folder_path, n, err = fut.result()
                if err is not None:
                    print(f"Skipped {folder_path}: {err}")
                else:
                    total += n
    print(f"Done. Saved {total} files.")


if __name__ == "__main__":
    main()
