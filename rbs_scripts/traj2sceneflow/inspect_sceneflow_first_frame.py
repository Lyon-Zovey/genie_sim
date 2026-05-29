#!/usr/bin/env python3
"""Render a recognizable sanity-check PNG for the first sceneflow anchor frame
of every trajectory under a `--gpu-batched` run.

Each trajectory gets a 4-panel figure that mirrors what dev_wjj's SAPIEN
viewer overlay shows, but as a static image:

    ┌──────────────────────────┬──────────────────────────┐
    │ (a) rgb anchor frame      │ (b) depth (sceneflow z)   │
    │     (what the camera saw) │     pixel-aligned heatmap │
    ├──────────────────────────┼──────────────────────────┤
    │ (c) 3D point cloud,       │ (d) 3D point cloud,       │
    │     RGB-colored, oblique  │     seg-id colored, oblique│
    │     (ground/table dropped)│     (mug-L/C/R distinct)  │
    └──────────────────────────┴──────────────────────────┘

Why this works for sceneflow: `scene_point_flow_ref<ANCHOR>.anchor.npy` is
shape (H, W, 3) — pixel-aligned with `rgb.mp4` frame ANCHOR — so it is
literally the image's *world-space* xyz at every pixel. Visualizing the z
channel as a heatmap (panel b) gives you the most direct "0th frame as an
image" view; panels (c)/(d) lift the same data to 3D for context.

Why we drop ground+table: empirically 94.9% of MIKASA-Robo base_camera
pixels belong to `actor:ground` (seg id 17) and `actor:table-workspace`
(seg id 16); without filtering they dominate every scatter plot and the
robot+mugs become invisible. The seg-id list is auto-built from the H5's
`id_poses.attrs` (no hard-coding).

Usage:
    # one specific trajectory
    python run_scripts/inspect_sceneflow_first_frame.py \
        --traj-dir /path/to/camera_data/traj_0

    # every trajectory under a save_dir
    python run_scripts/inspect_sceneflow_first_frame.py \
        --root /path/to/MIKASA-Robo/gpu_batched/ShellGameTouch-v0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# ─── data loading ───────────────────────────────────────────────────────────
def _find_all_anchors(traj_dir: Path) -> list[tuple[str, Path, Optional[Path]]]:
    """Return [(anchor_str, anchor_npy, pf_npy_or_None), ...] for *every*
    `scene_point_flow_ref<ANCHOR>.*` present in `traj_dir`, sorted by anchor
    frame index."""
    anchors: dict[str, tuple[Path, Optional[Path]]] = {}
    for f in traj_dir.glob("scene_point_flow_ref*.anchor.npy"):
        anchor = f.name[len("scene_point_flow_ref"):-len(".anchor.npy")]
        pf = traj_dir / f"scene_point_flow_ref{anchor}.npy"
        anchors[anchor] = (f, pf if pf.exists() else None)
    for f in traj_dir.glob("scene_point_flow_ref*.npy"):
        if ".anchor." in f.name:
            continue
        anchor = f.name[len("scene_point_flow_ref"):-len(".npy")]
        anchors.setdefault(anchor, (f, f))
    out = [(a, npy, pf) for a, (npy, pf) in anchors.items()]
    out.sort(key=lambda x: int("".join(c for c in x[0] if c.isdigit()) or "0"))
    return out


def _load_anchor_xyz(anchor_npy: Path, pf_npy: Optional[Path]) -> np.ndarray:
    """(H, W, 3) float32 world coords for the anchor frame."""
    if anchor_npy.exists():
        arr = np.load(anchor_npy)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        return arr.astype(np.float32)
    if pf_npy is not None and pf_npy.exists():
        return np.load(pf_npy)[0].astype(np.float32)
    raise FileNotFoundError(f"neither {anchor_npy} nor {pf_npy} exists")


def _read_rgb_frame(traj_dir: Path, frame_idx: int) -> Optional[np.ndarray]:
    """Best-effort grab of `rgb.mp4` frame `frame_idx` as (H, W, 3) uint8."""
    rgb_path = traj_dir / "rgb.mp4"
    if not rgb_path.exists():
        return None
    try:
        import imageio.v3 as iio                                           # noqa: WPS433
        return np.asarray(iio.imread(str(rgb_path), index=frame_idx))
    except Exception:
        pass
    try:
        import cv2                                                          # noqa: WPS433

        cap = cv2.VideoCapture(str(rgb_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if ok:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return None


def _load_seg_anchor(traj_dir: Path, frame_idx: int) -> Optional[np.ndarray]:
    """Anchor-frame seg as (H, W) int. Tries seg.npy first; falls back to
    seg.b2nd (decompressed via mikasa_robo_suite's helper)."""
    seg_npy = traj_dir / "seg.npy"
    if seg_npy.exists():
        seg = np.load(seg_npy)
        if seg.ndim == 4 and seg.shape[-1] == 1:
            seg = seg[..., 0]
        return np.asarray(seg[frame_idx]).astype(np.int32)
    seg_b2nd = traj_dir / "seg.b2nd"
    if seg_b2nd.exists():
        try:
            from mikasa_robo_suite.dataset_collectors.rbs_record.seg_compress import (  # noqa: WPS433
                decompress_seg,
            )

            seg = decompress_seg(seg_b2nd, None)
            if seg.ndim == 4 and seg.shape[-1] == 1:
                seg = seg[..., 0]
            return np.asarray(seg[frame_idx]).astype(np.int32)
        except Exception as exc:                                            # noqa: BLE001
            print(f"  [warn] seg.b2nd present but decompress failed: {exc}")
    return None


def _load_cam_pose(traj_dir: Path, frame_idx: int) -> Optional[np.ndarray]:
    """Load the 4×4 cam-to-world matrix for `frame_idx` from cam_poses.npy.
    Returns None if not found."""
    p = traj_dir / "cam_poses.npy"
    if not p.exists():
        return None
    poses = np.load(p)          # (T, 4, 4)
    if poses.ndim == 2:
        return poses.astype(np.float32)
    idx = min(frame_idx, poses.shape[0] - 1)
    return poses[idx].astype(np.float32)


def _anchor_to_world(xyz: np.ndarray, cam_pose: np.ndarray) -> np.ndarray:
    """Transform (H, W, 3) anchor points from camera space to world space.

    anchor.npy was computed from depth images that were vertically flipped
    ([::-1]) before saving.  The unprojection formula y = (cy - vv) * z / fy
    therefore produces a Y value whose sign is *inverted* relative to the
    physical camera Y axis.  We correct this before applying the cam-to-world
    rotation so that the resulting world coordinates match Metaworld's Z-up
    convention (table ≈ z=0, robot arm z > 0).
    """
    H, W = xyz.shape[:2]
    pts = xyz.reshape(-1, 3).copy()
    pts[:, 1] *= -1                         # correct the y-flip artifact
    R = cam_pose[:3, :3]                    # cam-to-world rotation
    t = cam_pose[:3, 3]                     # cam-to-world translation
    pts_world = (R @ pts.T).T + t
    return pts_world.reshape(H, W, 3)


def _load_id_to_name(traj_dir: Path) -> dict[int, str]:
    """`{seg_id: actor_name}` from any `traj_<i>.h5` in `traj_dir`. Returns
    empty dict if absent (then no semantic filtering is possible)."""
    try:
        import h5py                                                         # noqa: WPS433
    except Exception:
        return {}
    h5_files = list(traj_dir.glob("traj_*.h5"))
    if not h5_files:
        return {}
    out: dict[int, str] = {}
    with h5py.File(h5_files[0], "r") as hf:
        for top_key in hf.keys():
            grp = hf[top_key]
            if "id_poses" in grp and hasattr(grp["id_poses"], "attrs"):
                for k, v in grp["id_poses"].attrs.items():
                    try:
                        out[int(k)] = str(v)
                    except Exception:
                        continue
                break
    return out


# ─── filtering ──────────────────────────────────────────────────────────────
DEFAULT_DROP_PATTERNS = ("actor:ground", "ground-plane", "actor:table", "scene-builder")


def _foreground_mask(
    seg: Optional[np.ndarray],
    id_to_name: dict[int, str],
    drop_patterns: tuple[str, ...] = DEFAULT_DROP_PATTERNS,
) -> Optional[np.ndarray]:
    """Boolean (H, W) mask = True for *foreground* pixels (i.e. NOT matching
    any drop_pattern). Returns None if seg / id_to_name are unavailable."""
    if seg is None or not id_to_name:
        return None
    drop_patterns_lc = tuple(p.lower() for p in drop_patterns)
    drop_ids = {
        sid for sid, name in id_to_name.items()
        if any(p in name.lower() for p in drop_patterns_lc)
    }
    if not drop_ids:
        return None
    return ~np.isin(seg, list(drop_ids))


# ─── colour palette for seg-id overlay ──────────────────────────────────────
def _palette_color(seg_id: int) -> np.ndarray:
    """Stable distinct RGB ∈ [0,1] for any int id (HSV golden-angle wheel)."""
    h = (seg_id * 0.61803398875) % 1.0
    s, v = 0.85, 0.95
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    table = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    return np.asarray(table[i % 6], dtype=np.float32)


# ─── main panel ─────────────────────────────────────────────────────────────
def _make_panel(
    rgb: Optional[np.ndarray],
    xyz: np.ndarray,
    fg_mask: Optional[np.ndarray],
    seg: Optional[np.ndarray],
    id_to_name: dict[int, str],
    out_png: Path,
    title: str,
    max_3d_points: int = 60_000,
    no_filter: bool = False,
    cam_pose: Optional[np.ndarray] = None,
    flow_last: Optional[np.ndarray] = None,   # (H, W, 3) last frame of full flow
    rgb_last: Optional[np.ndarray] = None,    # last RGB frame for comparison
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    H, W = xyz.shape[:2]
    valid = np.isfinite(xyz).all(axis=-1)

    if no_filter:
        # Show every finite point; disable all background filtering
        fg_mask = valid
    elif fg_mask is None:
        # If no semantic foreground filter is available, fall back to a geometric
        # heuristic: drop anything farther than 1.2 m from the median-z plane.
        z = xyz[..., 2]
        z_med = float(np.nanmedian(z[valid]))
        fg_mask = valid & (np.abs(z - z_med) < 1.2)
    else:
        fg_mask = fg_mask & valid

    fig = plt.figure(figsize=(24, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.22, wspace=0.14)

    # ── (a) RGB anchor frame ────────────────────────────────────────────────
    ax_rgb = fig.add_subplot(gs[0, 0])
    if rgb is not None:
        ax_rgb.imshow(rgb)
        ax_rgb.set_title("(a) rgb  anchor frame")
    else:
        ax_rgb.text(0.5, 0.5, "rgb.mp4 unavailable", ha="center", va="center")
        ax_rgb.set_title("(a) rgb  anchor frame")
    ax_rgb.axis("off")

    # ── (b) depth (sceneflow z) heatmap, pixel-aligned with rgb ────────────
    ax_z = fig.add_subplot(gs[0, 1])
    z_disp = -xyz[..., 2].copy()          # negate: positive = farther away
    z_disp[~valid] = np.nan
    z_finite = z_disp[np.isfinite(z_disp)]
    if z_finite.size:
        vmin, vmax = np.percentile(z_finite, [1.0, 99.0])
    else:
        vmin, vmax = 0.0, 5.0
    im = ax_z.imshow(z_disp, cmap="turbo", vmin=vmin, vmax=vmax)
    ax_z.set_title(
        "(b) depth from anchor.npy  (bright = farther)"
    )
    ax_z.axis("off")
    cb = plt.colorbar(im, ax=ax_z, fraction=0.046, pad=0.02)
    cb.set_label("depth [m]")

    # ── (c) displacement heatmap  anchor→last frame ─────────────────────────
    ax_disp = fig.add_subplot(gs[0, 2])
    if flow_last is not None:
        disp_map = np.linalg.norm(flow_last - xyz, axis=-1)  # (H, W)
        disp_map[~valid] = np.nan
        d_valid = disp_map[np.isfinite(disp_map)]
        d_max = float(np.percentile(d_valid, 99.0)) if d_valid.size else 0.5
        im_d = ax_disp.imshow(disp_map, cmap="hot", vmin=0.0, vmax=d_max)
        cb_d = plt.colorbar(im_d, ax=ax_disp, fraction=0.046, pad=0.02)
        cb_d.set_label("3D disp [m]")
        moving_pct = 100.0 * float((disp_map > 0.02).sum()) / max(int(valid.sum()), 1)
        ax_disp.set_title(
            f"(c) 3D displacement  anchor→last\n"
            f"    bright = large motion;  >2cm: {moving_pct:.1f}% of pixels"
        )
    else:
        ax_disp.text(0.5, 0.5, "flow .npy unavailable\n(run flow_compress first)",
                     ha="center", va="center", fontsize=9)
        ax_disp.set_title("(c) 3D displacement  (unavailable)")
    ax_disp.axis("off")

    # ── 3D representation: prefer world space if cam_pose is available ─────
    # anchor.npy stores camera-space XYZ (z<0 = in front of camera).
    # Transform to world space so the plot shows the scene in a natural
    # Z-up orientation (robot base ≈ z=0, table z≈0, arm z>0).
    if cam_pose is not None:
        xyz_3d = _anchor_to_world(xyz, cam_pose)
        coord_label = "world"
        elev, azim = 30, -60        # oblique bird's-eye for Z-up world
    else:
        xyz_3d = xyz
        coord_label = "camera"
        elev, azim = 22, -65        # original angles for camera space

    # ── prep 3D points (foreground only) ───────────────────────────────────
    pts_fg = xyz_3d[fg_mask]
    n_fg = int(pts_fg.shape[0])
    if rgb is not None:
        rgb_fg = (rgb[fg_mask].astype(np.float32) / 255.0).clip(0.0, 1.0)
    else:
        rgb_fg = np.full_like(pts_fg, 0.7)

    if seg is not None:
        seg_fg = seg[fg_mask]
        seg_color_fg = np.stack([_palette_color(int(s)) for s in np.unique(seg_fg)])
        seg_id_to_color = {int(s): seg_color_fg[i] for i, s in enumerate(np.unique(seg_fg))}
        seg_rgb_fg = np.stack([seg_id_to_color[int(s)] for s in seg_fg])
    else:
        seg_rgb_fg = rgb_fg

    if n_fg > max_3d_points:
        idx = np.random.RandomState(0).choice(n_fg, max_3d_points, replace=False)
        pts_fg = pts_fg[idx]
        rgb_fg = rgb_fg[idx]
        seg_rgb_fg = seg_rgb_fg[idx]

    def _setup_3d(ax, title: str) -> None:
        ax.set_xlabel(f"{coord_label} x [m]")
        ax.set_ylabel(f"{coord_label} y [m]")
        ax.set_zlabel(f"{coord_label} z [m]")
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
        if pts_fg.shape[0] > 0:
            for axis_name, vals in zip("xyz", pts_fg.T):
                lo, hi = float(vals.min()), float(vals.max())
                pad = max((hi - lo) * 0.05, 0.02)
                getattr(ax, f"set_{axis_name}lim")(lo - pad, hi + pad)
            try:
                spans = pts_fg.max(0) - pts_fg.min(0)
                ax.set_box_aspect(tuple(spans + 1e-6))
            except Exception:
                pass

    # ── (d) 3D point cloud, RGB-colored ────────────────────────────────────
    ax_pc = fig.add_subplot(gs[1, 0], projection="3d")
    if n_fg:
        ax_pc.scatter(pts_fg[:, 0], pts_fg[:, 1], pts_fg[:, 2],
                      c=rgb_fg, s=2.0, marker=".", linewidths=0)
    _setup_3d(
        ax_pc,
        f"(d) 3D cloud, RGB-colored  [{coord_label} space]  N={n_fg:,}",
    )

    # ── (e) 3D point cloud, seg-id colored ─────────────────────────────────
    ax_id = fig.add_subplot(gs[1, 1], projection="3d")
    if n_fg:
        ax_id.scatter(pts_fg[:, 0], pts_fg[:, 1], pts_fg[:, 2],
                      c=seg_rgb_fg, s=2.0, marker=".", linewidths=0)
    _setup_3d(ax_id, f"(e) 3D cloud, seg-id colored  [{coord_label} space]")

    # legend for (e): show up to 12 most-populous seg ids
    if seg is not None and id_to_name and n_fg:
        seg_fg_full = seg[fg_mask]
        ids, counts = np.unique(seg_fg_full, return_counts=True)
        order = np.argsort(-counts)[:12]
        handles = []
        for i in order:
            sid = int(ids[i])
            name = id_to_name.get(sid, f"id {sid}")
            short = name.split("/")[-1].split("[")[0]
            color = _palette_color(sid)
            handles.append(
                Line2D([0], [0], marker="o", linestyle="", color=color,
                       markersize=6, label=f"{sid}: {short} (n={counts[i]})")
            )
        ax_id.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.05, 1.0),
                     fontsize=7, frameon=False)

    # ── (f) RGB last frame (comparison with anchor) ─────────────────────────
    ax_rgb_last = fig.add_subplot(gs[1, 2])
    if rgb_last is not None:
        ax_rgb_last.imshow(rgb_last)
        ax_rgb_last.set_title("(f) rgb  last frame  (compare with (a))")
    elif flow_last is not None:
        ax_rgb_last.text(0.5, 0.5, "rgb last frame\nnot loaded",
                         ha="center", va="center", fontsize=9)
        ax_rgb_last.set_title("(f) rgb  last frame")
    else:
        ax_rgb_last.text(0.5, 0.5, "flow .npy unavailable", ha="center", va="center", fontsize=9)
        ax_rgb_last.set_title("(f) rgb  last frame")
    ax_rgb_last.axis("off")

    # ── footer / global title ──────────────────────────────────────────────
    drop_pct = 100.0 * (1.0 - n_fg / max(int(valid.sum()), 1))
    fig.suptitle(
        f"{title}\n"
        f"sceneflow shape (H,W,3) = {xyz.shape}    "
        f"foreground points kept = {n_fg:,}  "
        f"(dropped {drop_pct:.1f}% as ground/table/scene-builder)",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─── per-traj driver ────────────────────────────────────────────────────────
def _process_traj(traj_dir: Path, no_filter: bool = False) -> int:
    """Render one PNG per anchor frame found in `traj_dir`.

    Returns the number of anchors successfully rendered.
    """
    anchors = _find_all_anchors(traj_dir)
    if not anchors:
        print(f"[SKIP] {traj_dir.name}: no scene_point_flow_ref*.anchor.npy / .npy found")
        return 0

    # id_poses attrs and rgb.mp4 only need to be opened once per traj
    id_to_name = _load_id_to_name(traj_dir)

    n_ok = 0
    for anchor, anchor_npy, pf_npy in anchors:
        try:
            anchor_idx = int("".join(c for c in anchor if c.isdigit()) or "0")
        except Exception:
            anchor_idx = 0
        try:
            xyz = _load_anchor_xyz(anchor_npy, pf_npy)
        except Exception as exc:                                            # noqa: BLE001
            print(f"[FAIL] {traj_dir.name} ref{anchor}: {exc}")
            continue
        rgb = _read_rgb_frame(traj_dir, anchor_idx)
        seg = _load_seg_anchor(traj_dir, anchor_idx)
        cam_pose = _load_cam_pose(traj_dir, anchor_idx)
        fg_mask = None if no_filter else _foreground_mask(seg, id_to_name)

        # Load full flow to compute displacement and last-frame comparison
        flow_last: Optional[np.ndarray] = None
        rgb_last: Optional[np.ndarray] = None
        if pf_npy is not None and pf_npy.exists():
            try:
                full_flow = np.load(str(pf_npy))  # (T, H, W, 3)
                if full_flow.ndim == 4 and full_flow.shape[0] > 1:
                    flow_last = full_flow[-1]      # (H, W, 3) last frame XYZ
                    last_idx = anchor_idx + full_flow.shape[0] - 1
                    rgb_last = _read_rgb_frame(traj_dir, last_idx)
            except Exception as exc:              # noqa: BLE001
                print(f"  [warn] could not load full flow {pf_npy.name}: {exc}")

        suffix = "_nofilter" if no_filter else ""
        out = traj_dir / f"_sceneflow_check_ref{anchor}{suffix}.png"
        _make_panel(
            rgb, xyz, fg_mask, seg, id_to_name, out,
            title=f"{traj_dir.name}  ref{anchor}  (frame {anchor_idx})"
                  + ("  [no filter]" if no_filter else ""),
            no_filter=no_filter,
            cam_pose=cam_pose,
            flow_last=flow_last,
            rgb_last=rgb_last,
        )
        extras = []
        if rgb is None:
            extras.append("rgb=missing")
        if seg is None:
            extras.append("seg=missing")
        if not id_to_name:
            extras.append("id_poses=missing")
        extra_s = f"  [{'; '.join(extras)}]" if extras else ""
        print(f"[OK]   {traj_dir.name} ref{anchor} (frame {anchor_idx}) → {out.name}{extra_s}")
        n_ok += 1
    return n_ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--traj-dir", type=Path, help="single camera_data/traj_<i> dir")
    g.add_argument("--root",     type=Path, help="save_dir; processes every traj under <root>/camera_data/")
    ap.add_argument(
        "--no-filter", action="store_true",
        help="Disable all background filtering (semantic and geometric); "
             "show the raw point cloud exactly as recorded. "
             "Output PNGs are named *_nofilter.png to avoid overwriting filtered ones.",
    )
    args = ap.parse_args()

    if args.traj_dir is not None:
        if not args.traj_dir.is_dir():
            print(f"--traj-dir does not exist: {args.traj_dir}", file=sys.stderr)
            return 2
        n_imgs = _process_traj(args.traj_dir, no_filter=args.no_filter)
        return 0 if n_imgs > 0 else 1

    cam_root = args.root / "camera_data" if (args.root / "camera_data").is_dir() else args.root
    if not cam_root.is_dir():
        print(f"no camera_data/ under {args.root}", file=sys.stderr)
        return 2
    trajs = sorted(p for p in cam_root.iterdir() if p.is_dir() and p.name.startswith("traj_"))
    if not trajs:
        print(f"no traj_* directories under {cam_root}", file=sys.stderr)
        return 2
    total_imgs = 0
    n_traj_ok = 0
    for p in trajs:
        k = _process_traj(p, no_filter=args.no_filter)
        total_imgs += k
        if k > 0:
            n_traj_ok += 1
    print(f"\n[done] {total_imgs} images rendered across {n_traj_ok}/{len(trajs)} trajectories")
    return 0 if total_imgs > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
