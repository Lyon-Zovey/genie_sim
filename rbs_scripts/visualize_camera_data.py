#!/usr/bin/env python3
"""
Visualize recorded camera data (RGB / depth / segmentation) from camera_data/.

Reads from:
  camera_data/traj_N/
    rgb.mp4
    depth_video.npy    (T, H, W)  float16  metres
    seg.npy            (T, H, W)  int32    body-IDs  (0 = background / world body)
    traj_N.h5          (optional)  id_poses with body names

Writes to the same traj_N/ directory:
  depth_vis.mp4        plasma-colourmap depth video
  seg_vis.mp4          body-ID colour-coded segmentation video
  combined.mp4         RGB | depth | seg  side-by-side

Usage:
  # Visualise traj_0 from a recorded dataset
  python scripts/visualize_camera_data.py \\
      --traj-dir dataset/hammer-v3/camera_data/traj_0

  # All trajs under camera_data/
  python scripts/visualize_camera_data.py \\
      --camera-data-dir dataset/hammer-v3/camera_data

  # Only write combined.mp4, skip individual videos
  python scripts/visualize_camera_data.py \\
      --traj-dir dataset/hammer-v3/camera_data/traj_0 \\
      --combined-only

  # Custom depth clip range (metres)
  python scripts/visualize_camera_data.py \\
      --traj-dir dataset/hammer-v3/camera_data/traj_0 \\
      --depth-min 0.3 --depth-max 2.0
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Colourmap helpers
# ---------------------------------------------------------------------------

def apply_colormap(arr_norm: np.ndarray, cmap_name: str = "plasma") -> np.ndarray:
    """Map normalised float array [0,1] → (H,W,3) uint8 using matplotlib cmap."""
    from matplotlib import colormaps
    cmap = colormaps[cmap_name]
    rgba = cmap(arr_norm)                          # (H, W, 4) float32  0-1
    return (rgba[..., :3] * 255).astype(np.uint8)  # (H, W, 3) uint8


# 40 visually distinct colours (RGB) for body IDs; background = black.
_PALETTE = np.array([
    [0,   0,   0  ],  # 0 = background / world body → dark
    [220, 50,  50 ],  # 1
    [50,  150, 220],  # 2
    [50,  220, 100],  # 3
    [220, 180, 50 ],  # 4
    [180, 50,  220],  # 5
    [50,  220, 200],  # 6
    [220, 100, 150],  # 7
    [130, 220, 50 ],  # 8
    [50,  100, 220],  # 9
    [220, 130, 50 ],  # 10
    [150, 50,  150],  # 11
    [50,  200, 150],  # 12
    [200, 80,  80 ],  # 13
    [80,  200, 80 ],  # 14
    [80,  80,  200],  # 15
    [200, 200, 80 ],  # 16
    [200, 80,  200],  # 17
    [80,  200, 200],  # 18
    [160, 80,  40 ],  # 19
    [40,  160, 80 ],  # 20
    [80,  40,  160],  # 21
    [160, 160, 40 ],  # 22
    [160, 40,  160],  # 23
    [40,  160, 160],  # 24
    [255, 140, 0  ],  # 25
    [0,   255, 140],  # 26
    [140, 0,   255],  # 27
    [255, 0,   140],  # 28
    [140, 255, 0  ],  # 29
    [0,   140, 255],  # 30
    [200, 200, 200],  # 31
    [100, 100, 100],  # 32
    [255, 200, 200],  # 33
    [200, 255, 200],  # 34
    [200, 200, 255],  # 35
    [255, 255, 150],  # 36
    [255, 150, 255],  # 37
    [150, 255, 255],  # 38
    [180, 120, 60 ],  # 39
], dtype=np.uint8)  # shape (40, 3)


def colourise_seg(seg: np.ndarray) -> np.ndarray:
    """
    Map (H, W) int32 body-ID array → (H, W, 3) uint8 colour image.
    Background / world body (id <= 0) → black.
    """
    H, W = seg.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    unique_ids = np.unique(seg)
    for uid in unique_ids:
        if uid <= 0:          # background (-1) or world body (0) → black
            continue
        colour = _PALETTE[uid % len(_PALETTE)]
        out[seg == uid] = colour
    return out


def colourise_depth(
    depth: np.ndarray,
    d_min: float | None = None,
    d_max: float | None = None,
    cmap: str = "plasma",
    bg_pct: float = 99.0,
) -> np.ndarray:
    """
    Map (H, W) float depth (metres) → (H, W, 3) uint8 colour image.

    Background / invalid pixels are stored as depth == 0 by
    replay_record_trajectories.py (far-clipping-plane pixels are zeroed out
    before saving).  Those pixels are rendered black.

    d_min / d_max default to the 1st / bg_pct-th percentile of valid pixels so
    that depth differences across the robot workspace are fully visible.
    """
    depth = depth.astype(np.float32)
    valid = depth > 0          # 0 = background / far-plane (already zeroed upstream)

    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    if d_min is None:
        d_min = float(np.percentile(depth[valid], 1))
    if d_max is None:
        d_max = float(np.percentile(depth[valid], bg_pct))
    if d_max <= d_min:
        d_max = d_min + 1e-4

    norm = np.clip((depth - d_min) / (d_max - d_min), 0.0, 1.0)
    coloured = apply_colormap(norm, cmap)
    coloured[~valid] = 0       # black for background pixels
    return coloured


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def load_rgb_frames(mp4_path: Path) -> np.ndarray | None:
    """Load an MP4 as (T, H, W, 3) uint8 array.  Returns None on failure."""
    try:
        import imageio.v3 as iio
        return iio.imread(str(mp4_path), plugin="pyav")   # (T, H, W, 3)
    except Exception:
        try:
            import imageio
            reader = imageio.get_reader(str(mp4_path))
            frames = [f for f in reader]
            reader.close()
            return np.stack(frames, axis=0)
        except Exception as e:
            print(f"  [WARN] Could not load {mp4_path}: {e}")
            return None


def save_video(frames: np.ndarray, path: Path, fps: float) -> None:
    """Save (T, H, W, 3) uint8 → MP4."""
    try:
        import imageio.v3 as iio
        iio.imwrite(str(path), frames, fps=int(fps))
    except Exception:
        import imageio
        with imageio.get_writer(str(path), fps=int(fps)) as w:
            for f in frames:
                w.append_data(f)
    print(f"    → {path}")


# ---------------------------------------------------------------------------
# Per-traj processing
# ---------------------------------------------------------------------------

def load_body_names(traj_dir: Path, traj_id: str) -> dict[int, str]:
    """Try to read id_poses body-name mapping from traj_N.h5."""
    h5_path = traj_dir / f"{traj_id}.h5"
    if not h5_path.exists():
        return {}
    try:
        with h5py.File(str(h5_path), "r") as f:
            grp_key = traj_id if traj_id in f else list(f.keys())[0]
            ip = f[grp_key].get("id_poses")
            if ip is None:
                return {}
            return {int(k): str(v) for k, v in ip.attrs.items()}
    except Exception:
        return {}


def process_traj(
    traj_dir: Path,
    fps: float,
    depth_min: float | None,
    depth_max: float | None,
    combined_only: bool,
) -> None:
    traj_id = traj_dir.name   # e.g. "traj_0"
    print(f"\n  Processing {traj_id}  ({traj_dir})")

    depth_path = traj_dir / "depth_video.npy"
    seg_path   = traj_dir / "seg.npy"
    rgb_path   = traj_dir / "rgb.mp4"

    if not depth_path.exists() or not seg_path.exists():
        print(f"  [SKIP] depth_video.npy or seg.npy not found")
        return

    depth_seq = np.load(str(depth_path)).astype(np.float32)  # (T, H, W)
    seg_seq   = np.load(str(seg_path))                         # (T, H, W)  int32
    T, H, W   = depth_seq.shape

    d_flat = depth_seq[depth_seq > 0].astype(np.float32)  # depth=0 is background
    d_vis_min = float(np.percentile(d_flat, 1))  if d_flat.size else 0.0
    d_vis_max = float(np.percentile(d_flat, 99)) if d_flat.size else 1.0
    bg_px = int((depth_seq == 0).sum())
    bg_pct_val = 100.0 * bg_px / depth_seq.size
    print(f"    frames={T}  res={W}×{H}")
    print(f"    depth  fg range (1–99 pct): [{d_vis_min:.3f}, {d_vis_max:.3f}] m  "
          f"background(depth=0): {bg_pct_val:.1f}%")
    body_names = load_body_names(traj_dir, traj_id)
    unique_ids = sorted(int(x) for x in np.unique(seg_seq) if x > 0)
    print(f"    seg    unique body-IDs: {unique_ids}")
    if body_names:
        for bid in unique_ids[:8]:
            name = body_names.get(bid, "?")
            print(f"      id={bid:3d}  {name}")
        if len(unique_ids) > 8:
            print(f"      ... ({len(unique_ids)} total)")

    # ---- coloured sequences ----
    depth_vis = np.stack(
        [colourise_depth(depth_seq[t], depth_min, depth_max) for t in range(T)],
        axis=0,
    )   # (T, H, W, 3)

    seg_vis = np.stack(
        [colourise_seg(seg_seq[t]) for t in range(T)],
        axis=0,
    )   # (T, H, W, 3)

    if not combined_only:
        save_video(depth_vis, traj_dir / "depth_vis.mp4", fps)
        save_video(seg_vis,   traj_dir / "seg_vis.mp4",   fps)

    # ---- combined (RGB | depth | seg) ----
    rgb_frames = load_rgb_frames(rgb_path)
    if rgb_frames is not None:
        # Align lengths (rgb might have slightly different count)
        n = min(len(rgb_frames), T)
        combined = np.concatenate(
            [rgb_frames[:n], depth_vis[:n], seg_vis[:n]], axis=2
        )   # (n, H, 3*W, 3)
        save_video(combined, traj_dir / "combined.mp4", fps)
    else:
        print("  [WARN] RGB video not available; skipping combined.mp4")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise depth & seg from recorded camera_data/."
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--traj-dir",
        help="Path to a single camera_data/traj_N/ directory.",
    )
    group.add_argument(
        "--camera-data-dir",
        help="Path to camera_data/ root; processes all traj_N/ subdirectories.",
    )
    p.add_argument("--fps",          type=float, default=30.0)
    p.add_argument("--depth-min",    type=float, default=None,
                   help="Depth clip minimum in metres (auto if omitted).")
    p.add_argument("--depth-max",    type=float, default=None,
                   help="Depth clip maximum in metres (auto if omitted).")
    p.add_argument("--combined-only", action="store_true",
                   help="Only write combined.mp4; skip depth_vis.mp4 and seg_vis.mp4.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.traj_dir:
        dirs = [Path(args.traj_dir)]
    else:
        root = Path(args.camera_data_dir)
        dirs = sorted(d for d in root.iterdir()
                      if d.is_dir() and d.name.startswith("traj_"))
        print(f"Found {len(dirs)} traj dirs under {root}")

    for d in dirs:
        process_traj(
            traj_dir=d,
            fps=args.fps,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            combined_only=args.combined_only,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
