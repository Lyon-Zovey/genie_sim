"""
Pair-check visualization: RGB / depth aligned with sceneflow via pixel (u,v).

For each pixel (i,j), sceneflow stores a 3D point flow[t,i,j] in the OpenCV
camera frame and RGB stores a color rgb[t,i,j]. Coloring the sceneflow point
cloud by the SAME (i,j) of the RGB image should produce a recognizable scene.

Usage:
    python vis_rgb_sceneflow.py <traj_dir> --port 8093
    python vis_rgb_sceneflow.py <traj_dir> --port 8093 --full --color-mode seg

Opens viser web UI at http://localhost:<port>.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import trimesh
import viser
import viser.transforms as tf

sys.path.insert(0, str(Path(__file__).parent.parent / "traj2sceneflow"))
from flow_compress import decompress_one_flow



def load_first_flow(traj_dir: Path):
    anchors = sorted(traj_dir.glob("scene_point_flow_ref*.anchor.npy"))
    if not anchors:
        raise FileNotFoundError(f"No anchor files in {traj_dir}")
    anchor_path = anchors[0]
    anchor = np.load(str(anchor_path))
    stem = anchor_path.stem.replace(".anchor", "")
    vids = list(traj_dir.glob(f"{stem}*.mp4")) + list(traj_dir.glob(f"{stem}*.mkv"))
    if not vids:
        raise FileNotFoundError(f"No flow video for {stem}")
    return decompress_one_flow(vids[0], anchor)


def load_rgb_video(path: Path) -> np.ndarray:
    frames = iio.imread(str(path))
    if frames.ndim == 3:
        frames = frames[None]
    return frames.astype(np.uint8)


def load_depth_b2nd(path: Path) -> np.ndarray | None:
    try:
        import blosc2
    except ImportError:
        print("  blosc2 not installed; skipping depth")
        return None
    arr = blosc2.open(str(path))[:]
    return np.asarray(arr).astype(np.float32) / 1000.0  # int16 mm -> meters


def load_seg_b2nd(path: Path) -> np.ndarray | None:
    try:
        import blosc2
    except ImportError:
        return None
    arr = blosc2.open(str(path))[:]
    return np.asarray(arr)


def load_video_gray(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    frames = iio.imread(str(path))
    if frames.ndim == 4:
        frames = frames[..., 0]
    return frames.astype(np.uint8)


def seg_to_color(seg: np.ndarray) -> np.ndarray:
    """Map integer seg ids to a deterministic RGB color per id."""
    seg = seg.astype(np.int32)
    ids = seg.reshape(-1)
    rng = np.random.default_rng(0)
    palette = (rng.integers(40, 240, size=(seg.max() + 2, 3))).astype(np.uint8)
    palette[0] = (30, 30, 30)
    return palette[ids].reshape(*seg.shape, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj_dir", type=Path)
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--downsample", type=int, default=4,
                    help="Pixel-space downsample for point clouds")
    ap.add_argument("--no-depth", action="store_true",
                    help="Do not load/show depth point cloud")
    ap.add_argument("--no-flow", action="store_true",
                    help="Do not show sceneflow point cloud")
    ap.add_argument("--full", action="store_true",
                    help="Full end-to-end check: also load seg / target-obj mask / "
                         "obj_pose / mesh. seg & mask follow normal orientation; "
                         "intrinsics-fx/fy and cam2world stay unchanged.")
    ap.add_argument("--color-mode", choices=["rgb", "seg", "target"], default="rgb",
                    help="Color the flow/depth point cloud by RGB, seg id, or target-obj mask")
    ap.add_argument("--objflow-sample", type=int, default=2000,
                    help="Extra points sampled on mesh faces for object-flow "
                         "(in addition to raw vertices). 0 = vertices only.")
    ap.add_argument("--objflow-trails", type=int, default=40,
                    help="Number of vertex trajectories (polylines) to draw "
                         "from frame 0..current t. 0 = no trails.")
    args = ap.parse_args()

    traj_dir = args.traj_dir

    # ---- load everything ----
    print("Loading sceneflow...")
    flow = load_first_flow(traj_dir).astype(np.float32)  # (T,H,W,3), OpenGL cam
    T, H, W, _ = flow.shape
    print(f"  flow: {flow.shape}")

    print("Loading rgb.mp4 ...")
    rgb = load_rgb_video(traj_dir / "rgb.mp4")
    print(f"  rgb: {rgb.shape}")
    Tn = min(T, rgb.shape[0])
    flow = flow[:Tn]
    rgb = rgb[:Tn]

    depth = None
    if not args.no_depth:
        dp = traj_dir / "depth_video_int16mm_dt.b2nd"
        if dp.exists():
            print("Loading depth ...")
            depth = load_depth_b2nd(dp)
            if depth is not None:
                depth = depth[:Tn]
                print(f"  depth: {depth.shape}")

    K = np.load(traj_dir / "cam_intrinsics.npy").astype(np.float32)
    if K.ndim == 3:
        K = K[0]
    print(f"  K: {K.tolist()}")

    c2w_path = traj_dir / "cam2world_cv.npy"
    if not c2w_path.exists():
        c2w_path = traj_dir / "cam2world.npy"
    cam2world = np.load(str(c2w_path)).astype(np.float32)[:Tn]
    print(f"  cam2world: {cam2world.shape}")

    # ---- full-mode extras: seg / target mask / obj poses / meshes ----
    seg = None
    target_mask = None
    obj_poses: dict[str, np.ndarray] = {}
    obj_meshes: dict[str, trimesh.Trimesh] = {}
    if args.full:
        sp = traj_dir / "seg.b2nd"
        if sp.exists():
            print("Loading seg ...")
            seg = load_seg_b2nd(sp)
            if seg is not None:
                seg = seg[:Tn]
                if seg.ndim == 4:
                    seg = seg[..., 0]
                print(f"  seg: {seg.shape}")

        tm = None
        mz_files = sorted(traj_dir.glob("mask_*.npz"))
        if mz_files:
            mz = mz_files[0]
            z = np.load(mz)
            key = list(z.keys())[0]
            tm = z[key]
            print(f"  loaded mask from npz: {mz.name} (key={key})")
        if tm is None:
            tm = load_video_gray(traj_dir / "target_obj_mask.mp4")
            if tm is not None:
                print("  loaded mask from target_obj_mask.mp4 (lossy)")
        if tm is not None:
            target_mask = tm[:Tn].astype(np.uint8)
            print(f"  target_mask: {target_mask.shape}")

        for pf in sorted(traj_dir.glob("pose_*.npy")):
            if "_cv" in pf.stem:
                continue
            name = pf.stem.replace("pose_", "")
            obj_poses[name] = np.load(str(pf)).astype(np.float32)[:Tn]
            print(f"  pose_{name}: {obj_poses[name].shape}")

        for mp in sorted(traj_dir.glob("mesh_*.ply")):
            name = mp.stem.replace("mesh_", "")
            try:
                m = trimesh.load(str(mp), force="mesh")
                if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0:
                    obj_meshes[name] = m
                    print(f"  mesh_{name}: V={len(m.vertices)} F={len(m.faces)}")
            except Exception as e:
                print(f"  failed to load {mp.name}: {e}")

    print("  sceneflow is treated as OpenCV camera frame")

    # Now both flow values and RGB pixels live under the SAME (u,v) indexing
    # and the SAME OpenCV camera frame. cam2world is the OpenCV cam->world.

    # ---- precompute object flow: per-frame vertex world coords ----
    obj_flow_world: dict[str, np.ndarray] = {}   # name -> (T, N, 3)
    obj_flow_verts: dict[str, np.ndarray] = {}   # name -> (N, 3) body-local
    obj_flow_colors: dict[str, np.ndarray] = {}  # name -> (3,) uint8
    if args.full and obj_poses:
        palette = np.array([
            (255,  80,  80), ( 80, 200, 255), (120, 255, 120),
            (255, 200,  80), (220, 120, 255), ( 80, 255, 220),
        ], dtype=np.uint8)
        for i, (name, mesh) in enumerate(obj_meshes.items()):
            if name not in obj_poses:
                continue
            v = np.asarray(mesh.vertices, dtype=np.float32)
            if args.objflow_sample > 0 and len(mesh.faces) > 0:
                try:
                    extra, _ = trimesh.sample.sample_surface(mesh, args.objflow_sample)
                    v = np.concatenate([v, extra.astype(np.float32)], axis=0)
                except Exception as e:
                    print(f"  face-sample failed for {name}: {e}")
            obj_flow_verts[name] = v
            obj_flow_colors[name] = palette[i % len(palette)]
            T_b2c = obj_poses[name]                                       # (T,4,4)
            vh = np.concatenate([v, np.ones((v.shape[0], 1), np.float32)], 1)  # (N,4)
            p_cam = np.einsum('tij,nj->tni', T_b2c, vh)                   # (T,N,4)
            p_world = np.einsum('tij,tnj->tni', cam2world, p_cam)         # (T,N,4)
            obj_flow_world[name] = p_world[..., :3].astype(np.float32)
            print(f"  objflow[{name}]: N={v.shape[0]}, T={obj_flow_world[name].shape[0]}")

    ds = args.downsample
    flow_ds = flow[:, ::ds, ::ds, :]
    rgb_ds = rgb[:, ::ds, ::ds, :]
    seg_ds = seg[:, ::ds, ::ds] if seg is not None else None
    tgt_ds = target_mask[:, ::ds, ::ds] if target_mask is not None else None
    H_ds, W_ds = flow_ds.shape[1], flow_ds.shape[2]
    print(f"  downsampled to ({H_ds}, {W_ds})")

    if seg_ds is not None:
        seg_color_ds = seg_to_color(seg_ds)
    else:
        seg_color_ds = None

    # Precompute pixel grid for depth back-projection
    if depth is not None:
        depth_ds = depth[:, ::ds, ::ds]
        u = np.arange(W, dtype=np.float32)[::ds]
        v = np.arange(H, dtype=np.float32)[::ds]
        uu, vv = np.meshgrid(u, v)  # (H_ds, W_ds)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
    else:
        depth_ds = None

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"Viser: http://0.0.0.0:{args.port}")

    frame_slider = server.gui.add_slider("Frame", min=0, max=Tn - 1, step=1, initial_value=0)
    point_size_slider = server.gui.add_slider("Point size", min=0.001, max=0.03,
                                              step=0.001, initial_value=0.004)
    show_flow_cb = server.gui.add_checkbox("Show sceneflow pc", initial_value=not args.no_flow)
    show_depth_cb = server.gui.add_checkbox("Show depth pc",
                                            initial_value=(depth_ds is not None))
    show_obj_cb = server.gui.add_checkbox("Show obj_pose / mesh",
                                          initial_value=args.full and len(obj_poses) > 0)
    show_mask_pc_cb = server.gui.add_checkbox(
        "Show target_mask pc (red, mask-only flow points)",
        initial_value=(tgt_ds is not None),
    )
    show_mask_img_cb = server.gui.add_checkbox(
        "Show target_mask 2D image",
        initial_value=(target_mask is not None),
    )
    color_mode_dd = server.gui.add_dropdown(
        "Color mode", options=["rgb", "seg", "target"],
        initial_value=args.color_mode,
    )
    show_objflow_cb = server.gui.add_checkbox(
        "Show obj_flow (mesh verts via 6D pose)",
        initial_value=(len(obj_flow_world) > 0),
    )
    show_objflow_trails_cb = server.gui.add_checkbox(
        "Show obj_flow trails (vertex trajectories)",
        initial_value=(len(obj_flow_world) > 0 and args.objflow_trails > 0),
    )

    flow_handle = None
    depth_handle = None
    cam_handle = None
    mask_pc_handle = None
    mask_img_handle = None
    obj_frame_handles: dict[str, object] = {}
    mesh_handles: dict[str, object] = {}
    objflow_pc_handles: dict[str, object] = {}
    objflow_trail_handles: dict[str, object] = {}
    objflow_trail_idx: dict[str, np.ndarray] = {}
    for name, v in obj_flow_verts.items():
        n_trail = min(args.objflow_trails, v.shape[0])
        if n_trail > 0:
            rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
            objflow_trail_idx[name] = rng.choice(v.shape[0], n_trail, replace=False)

    def get_colors(t: int) -> np.ndarray:
        mode = color_mode_dd.value
        if mode == "seg" and seg_color_ds is not None:
            return seg_color_ds[t].reshape(-1, 3)
        if mode == "target" and tgt_ds is not None:
            m = tgt_ds[t]
            c = np.zeros((*m.shape, 3), dtype=np.uint8)
            c[m > 0] = (255, 60, 60)
            c[m == 0] = (60, 60, 60)
            return c.reshape(-1, 3)
        return rgb_ds[t].reshape(-1, 3)

    def render(t: int):
        nonlocal flow_handle, depth_handle, cam_handle, mask_pc_handle, mask_img_handle

        c2w = cam2world[t]
        R = c2w[:3, :3]
        tvec = c2w[:3, 3]

        if cam_handle is not None:
            cam_handle.remove()
        cam_handle = server.scene.add_frame(
            "/camera",
            wxyz=tf.SO3.from_matrix(R).wxyz,
            position=tvec,
            axes_length=0.05,
            axes_radius=0.002,
        )

        if flow_handle is not None:
            flow_handle.remove()
            flow_handle = None
        if show_flow_cb.value and not args.no_flow:
            pts_cam = flow_ds[t].reshape(-1, 3)
            colors = get_colors(t)
            valid = np.isfinite(pts_cam).all(axis=1) & (np.abs(pts_cam) < 100).all(axis=1)
            pts_cam = pts_cam[valid]
            colors = colors[valid]
            pts = (R @ pts_cam.T).T + tvec
            flow_handle = server.scene.add_point_cloud(
                "/flow_rgb", points=pts, colors=colors,
                point_size=point_size_slider.value,
            )

        if depth_handle is not None:
            depth_handle.remove()
            depth_handle = None
        if show_depth_cb.value and depth_ds is not None:
            d = depth_ds[t]
            x = (uu - cx) * d / fx
            y = (vv - cy) * d / fy
            z = d
            # OpenCV cam: x right, y down, z forward
            pts_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
            colors = get_colors(t)
            mask = (d > 0).reshape(-1) & np.isfinite(pts_cam).all(axis=1)
            pts_cam = pts_cam[mask]
            colors = colors[mask]
            pts = (R @ pts_cam.T).T + tvec
            depth_handle = server.scene.add_point_cloud(
                "/depth_rgb", points=pts, colors=colors,
                point_size=point_size_slider.value,
            )

        if mask_pc_handle is not None:
            mask_pc_handle.remove()
            mask_pc_handle = None
        if show_mask_pc_cb.value and tgt_ds is not None:
            m = tgt_ds[t] > 0
            pts_cam = flow_ds[t][m].reshape(-1, 3)
            valid = np.isfinite(pts_cam).all(axis=1) & (np.abs(pts_cam) < 100).all(axis=1)
            pts_cam = pts_cam[valid]
            if pts_cam.shape[0] > 0:
                pts = (R @ pts_cam.T).T + tvec
                colors = np.tile(np.array([[255, 30, 30]], dtype=np.uint8),
                                 (pts.shape[0], 1))
                mask_pc_handle = server.scene.add_point_cloud(
                    "/mask_pc", points=pts, colors=colors,
                    point_size=max(point_size_slider.value * 1.8, 0.006),
                )

        if mask_img_handle is not None:
            mask_img_handle.remove()
            mask_img_handle = None
        if show_mask_img_cb.value and target_mask is not None:
            m = target_mask[t]
            img = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
            img[m > 0] = (255, 60, 60)
            img[m == 0] = (40, 40, 40)
            mask_img_handle = server.scene.add_image(
                "/mask_img",
                image=img,
                render_width=0.8,
                render_height=0.8 * m.shape[0] / m.shape[1],
                position=(0.0, 0.0, -0.6),
                wxyz=(1.0, 0.0, 0.0, 0.0),
            )

        for name in list(obj_frame_handles.keys()):
            if not show_obj_cb.value or name not in obj_poses:
                obj_frame_handles[name].remove()
                del obj_frame_handles[name]
        for name in list(mesh_handles.keys()):
            if not show_obj_cb.value or name not in obj_meshes:
                mesh_handles[name].remove()
                del mesh_handles[name]
        if show_obj_cb.value:
            for name, pose_arr in obj_poses.items():
                pose_world = c2w @ pose_arr[t]
                Rw = pose_world[:3, :3]
                tw = pose_world[:3, 3]
                if name in obj_frame_handles:
                    obj_frame_handles[name].remove()
                obj_frame_handles[name] = server.scene.add_frame(
                    f"/obj_{name}",
                    wxyz=tf.SO3.from_matrix(Rw).wxyz,
                    position=tw,
                    axes_length=0.08, axes_radius=0.003,
                )
                mesh = obj_meshes.get(name)
                if mesh is not None:
                    if name in mesh_handles:
                        mesh_handles[name].remove()
                    mesh_handles[name] = server.scene.add_mesh_trimesh(
                        f"/mesh_{name}", mesh=mesh,
                        wxyz=tf.SO3.from_matrix(Rw).wxyz,
                        position=tw,
                    )

        for name in list(objflow_pc_handles.keys()):
            objflow_pc_handles[name].remove()
        objflow_pc_handles.clear()
        for name in list(objflow_trail_handles.keys()):
            objflow_trail_handles[name].remove()
        objflow_trail_handles.clear()

        if show_objflow_cb.value:
            for name, pts_TN3 in obj_flow_world.items():
                pts = pts_TN3[t]
                color = obj_flow_colors[name]
                colors = np.tile(color[None, :], (pts.shape[0], 1))
                objflow_pc_handles[name] = server.scene.add_point_cloud(
                    f"/objflow_{name}", points=pts, colors=colors,
                    point_size=max(point_size_slider.value * 1.2, 0.005),
                )

        if show_objflow_trails_cb.value and t > 0:
            for name, pts_TN3 in obj_flow_world.items():
                idx = objflow_trail_idx.get(name)
                if idx is None or idx.size == 0:
                    continue
                trail = pts_TN3[: t + 1, idx, :]               # (t+1, K, 3)
                segs = np.stack([trail[:-1], trail[1:]], axis=2)  # (t, K, 2, 3)
                segs = segs.reshape(-1, 2, 3)
                color = obj_flow_colors[name]
                seg_colors = np.tile(color[None, None, :], (segs.shape[0], 2, 1))
                objflow_trail_handles[name] = server.scene.add_line_segments(
                    f"/objflow_trail_{name}",
                    points=segs.astype(np.float32),
                    colors=seg_colors.astype(np.uint8),
                    line_width=2.0,
                )

    @frame_slider.on_update
    def _(_):
        render(int(frame_slider.value))

    @show_flow_cb.on_update
    def _(_):
        render(int(frame_slider.value))

    @show_depth_cb.on_update
    def _(_):
        render(int(frame_slider.value))

    @show_obj_cb.on_update
    def _(_):
        render(int(frame_slider.value))

    @show_mask_pc_cb.on_update
    def _(_):
        render(int(frame_slider.value))

    @show_mask_img_cb.on_update
    def _(_):
        render(int(frame_slider.value))

    @color_mode_dd.on_update
    def _(_):
        render(int(frame_slider.value))

    @show_objflow_cb.on_update
    def _(_):
        render(int(frame_slider.value))

    @show_objflow_trails_cb.on_update
    def _(_):
        render(int(frame_slider.value))

    @point_size_slider.on_update
    def _(_):
        render(int(frame_slider.value))

    render(0)

    print("\nHow to read this view:")
    print("  - flow_rgb : sceneflow points colored by rgb[i,j]. If (u,v)")
    print("               alignment is correct, the cloud looks like the real scene.")
    print("  - depth_rgb: depth back-projected with K, also colored by rgb[i,j].")
    print("               Should overlap flow_rgb.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
