#!/usr/bin/env python3
"""End-to-end conversion of one Metaworld traj_* dir to mikasa-style layout.

Inputs (must already exist in the traj dir):
    cam_poses.npy            (T, 4, 4) cam-to-world, OpenGL/RUB, ABSOLUTE
    cam_intrinsics.npy
    rgb.mp4
    seg.b2nd                 (T, H, W) int32 body-id per pixel
    traj_<N>.h5              MIKASA-compatible h5 with id_poses/<bid>/{position,
                              quaternion, camera_position, camera_quaternion}
    traj_task.json           actors = [{seg_id, name="body:<X>"}]

Outputs written into the traj dir:
    cam2world_cv.npy         (T, 4, 4) OpenCV cam-to-world ABSOLUTE
    cam2world_gl.npy         (T, 4, 4) OpenGL cam-to-world ABSOLUTE
    cam_poses.npy            (T, 4, 4) OpenCV, RELATIVE to cam0
    pose_<obj>.npy           (T, 4, 4) body->cam in OpenCV (per target body)
    mesh_<obj>.ply           binary little-endian PLY in body-local frame
    mask_<obj>.npz           (T, H, W) uint8 {0,255} target binary mask
    target_obj_mask.mp4      (T, H, W) binary video, same fps/size as rgb
    meta.json                exact spec required (overwrites prior meta.json)

Target body resolution uses scripts/target_objects.json + traj_task.json actors
(handles the "child_of" anonymous-child case).

Usage:
    python scripts/build_mikasa_format.py --traj-dir <path>
    python scripts/build_mikasa_format.py --root <dataset_root>      # serial
    python scripts/build_mikasa_format.py --root <dataset_root> -j N # parallel
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import blosc2
import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
FLIP4 = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
DEFAULT_MAPPING = REPO / "scripts" / "target_objects.json"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def safe_name(raw: str) -> str:
    s = str(raw)
    if s.startswith("body:"):
        s = s[len("body:"):]
    return _SAFE_NAME.sub("_", s)


def quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def inv44_batch(T: np.ndarray) -> np.ndarray:
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out


# ---------------------------------------------------------------------------
# Target body resolution (matches scripts/generate_target_obj_mask.py logic)
# ---------------------------------------------------------------------------

def resolve_targets(actors: list, mapping_entry: dict) -> list[dict]:
    """Return list of {seg_id, name, short} for each matched target body."""
    target = list(mapping_entry.get("target_bodies", []))
    child_of = set(mapping_entry.get("child_of", []))
    wanted = set(target)
    matched: list[dict] = []
    seen: set[int] = set()
    for i, a in enumerate(actors):
        if not a["name"].startswith("body:"):
            continue
        short = a["name"][len("body:"):]
        if short in wanted:
            if a["seg_id"] not in seen:
                matched.append({"seg_id": int(a["seg_id"]), "name": a["name"], "short": short})
                seen.add(a["seg_id"])
            if short in child_of and i + 1 < len(actors):
                nxt = actors[i + 1]
                if nxt["name"].startswith("body:") and nxt["seg_id"] not in seen:
                    nxt_short = nxt["name"][len("body:"):] or f"child_of_{short}"
                    matched.append({"seg_id": int(nxt["seg_id"]), "name": nxt["name"], "short": nxt_short})
                    seen.add(nxt["seg_id"])
    return matched


# ---------------------------------------------------------------------------
# Body->cam pose computation from H5
# ---------------------------------------------------------------------------

def compute_body_to_cam_cv(h5_path: Path, traj_key: str, bid: int) -> np.ndarray:
    """Read camera_position/quaternion (GL convention) from H5, return (T,4,4) body->cam in OpenCV."""
    with h5py.File(str(h5_path), "r") as f:
        g = f[f"{traj_key}/id_poses/{bid}"]
        cp = g["camera_position"][...]   # (T, 3) GL
        cq = g["camera_quaternion"][...]  # (T, 4) wxyz, GL body->cam

    T = cp.shape[0]
    out_gl = np.zeros((T, 4, 4), dtype=np.float32)
    for t in range(T):
        out_gl[t, :3, :3] = quat_wxyz_to_R(cq[t]).astype(np.float32)
        out_gl[t, :3, 3] = cp[t]
        out_gl[t, 3, 3] = 1.0
    out_cv = (FLIP4[None] @ out_gl).astype(np.float32)
    return out_cv


# ---------------------------------------------------------------------------
# Mesh extraction (target-only, walks descendants)
# ---------------------------------------------------------------------------

def _import_mujoco_metaworld():
    import mujoco
    import metaworld
    return mujoco, metaworld


def build_env_model(env_name: str):
    mujoco, metaworld = _import_mujoco_metaworld()
    mt1 = metaworld.MT1(env_name, seed=42)
    env = mt1.train_classes[env_name](render_mode="rgb_array", width=64, height=64)
    env.seed(42)
    env.set_task(mt1.train_tasks[0])
    env.reset()
    return env, env.model


# Reuse extract_meshes geom-to-body mesh logic
from extract_meshes import (  # noqa: E402
    box_mesh, sphere_mesh, ellipsoid_mesh, cylinder_mesh, capsule_mesh,
    extract_mesh_asset, geom_to_body_mesh, merge_body_meshes, write_ply_binary,
    collect_target_mesh, build_target_meshes,
)


# ---------------------------------------------------------------------------
# Mask extraction (npz + mp4)
# ---------------------------------------------------------------------------

def _ffprobe_fps(rgb_mp4: Path) -> float:
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate",
            "-of", "default=nokey=1:noprint_wrappers=1",
            str(rgb_mp4),
        ], text=True).strip()
        if "/" in out:
            n, d = out.split("/")
            d = float(d) if float(d) != 0 else 1.0
            return float(n) / d
        return float(out)
    except Exception:
        return 30.0


def write_mask_mp4(mask_t_h_w_uint8: np.ndarray, out_mp4: Path, fps: float) -> None:
    """Encode binary mask as monochrome H.264 mp4. mask is (T,H,W) uint8 in {0,255}."""
    T, H, W = mask_t_h_w_uint8.shape
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}",
        "-r", f"{fps}",
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "0", "-preset", "veryfast",
        str(out_mp4),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        p.stdin.write(mask_t_h_w_uint8.tobytes())
    finally:
        p.stdin.close()
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed ({rc}) writing {out_mp4}")


# ---------------------------------------------------------------------------
# Per-traj processing
# ---------------------------------------------------------------------------

def find_h5(traj_dir: Path) -> Path:
    cands = sorted(traj_dir.glob("*.h5"))
    if not cands:
        raise FileNotFoundError(f"no .h5 in {traj_dir}")
    if len(cands) > 1:
        raise RuntimeError(f"multiple .h5 in {traj_dir}: {[c.name for c in cands]}")
    return cands[0]


def build_camera_poses(traj_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (cam2world_cv, cam2world_gl, cam_poses_rel_cv) all (T,4,4) float32."""
    raw = traj_dir / "cam_poses.npy"
    if not raw.exists():
        # Maybe pipeline already ran — try cam2world_gl.npy as source-of-truth
        gl = traj_dir / "cam2world_gl.npy"
        if not gl.exists():
            raise FileNotFoundError(f"no cam_poses.npy or cam2world_gl.npy in {traj_dir}")
        cam2world_gl = np.load(str(gl)).astype(np.float32)
    else:
        cam2world_gl = np.load(str(raw)).astype(np.float32)
    cam2world_cv = (cam2world_gl @ FLIP4).astype(np.float32)
    cam_poses_rel_cv = (inv44_batch(cam2world_cv[:1]) @ cam2world_cv).astype(np.float32)
    cam_poses_rel_cv[0] = np.eye(4, dtype=np.float32)
    return cam2world_cv, cam2world_gl, cam_poses_rel_cv


def emit_target_files_h5(traj_dir: Path, h5_path: Path, traj_key: str,
                         targets: list[dict]) -> dict[str, str]:
    """Write pose_<obj>.npy (body->cam CV) for each target body. Returns name->filename map."""
    out: dict[str, str] = {}
    for t in targets:
        bid = int(t["seg_id"])
        short = t["short"]
        # Use safe_name for filename
        fn_short = safe_name(short)
        pose_cv = compute_body_to_cam_cv(h5_path, traj_key, bid)
        np.save(str(traj_dir / f"pose_{fn_short}.npy"), pose_cv)
        out[fn_short] = f"pose_{fn_short}.npy"
    return out


def emit_meshes(traj_dir: Path, env_model, target_short_names: list[str],
                overwrite: bool) -> dict:
    """Write mesh_<obj>.ply for each target body. Returns mesh meta dict."""
    targets = {safe_name(n) for n in target_short_names}
    body_meshes = build_target_meshes(env_model, targets)

    files_map: dict[str, str] = {}
    seg_id_map: dict[str, int] = {}
    n_verts: dict[str, int] = {}
    n_faces: dict[str, int] = {}

    for bid, entry in body_meshes.items():
        obj = entry["obj_name"]
        out_ply = traj_dir / f"mesh_{obj}.ply"
        if overwrite or not out_ply.exists():
            write_ply_binary(out_ply, entry["V"], entry["F"])
        files_map[obj] = out_ply.name
        seg_id_map[obj] = int(bid)
        n_verts[obj] = int(entry["V"].shape[0])
        n_faces[obj] = int(entry["F"].shape[0])

    return {
        "format": "binary_little_endian PLY; vertices are in body-local "
                  "frame (p_cam_h = T_body2cam[t] @ [v_body; 1]).",
        "files": files_map,
        "seg_ids": seg_id_map,
        "num_vertices": n_verts,
        "num_faces": n_faces,
    }


def emit_mask(traj_dir: Path, target_seg_ids: list[int],
              target_short_names: list[str], overwrite: bool) -> dict:
    """Write mask_<obj>.npz + target_obj_mask.mp4. Returns mask meta dict."""
    seg_path = traj_dir / "seg.b2nd"
    rgb_path = traj_dir / "rgb.mp4"
    if not seg_path.is_file():
        raise FileNotFoundError(f"missing seg.b2nd: {seg_path}")
    if not rgb_path.is_file():
        raise FileNotFoundError(f"missing rgb.mp4: {rgb_path}")

    seg = blosc2.open(str(seg_path))[:]   # (T, H, W) int32
    mask = np.isin(seg, target_seg_ids).astype(np.uint8) * 255
    T, H, W = mask.shape

    obj_safe = safe_name(target_short_names[0]) if target_short_names else "target"
    out_npz = traj_dir / f"mask_{obj_safe}.npz"
    if overwrite or not out_npz.is_file():
        np.savez_compressed(str(out_npz), mask=mask)

    out_mp4 = traj_dir / "target_obj_mask.mp4"
    fps = _ffprobe_fps(rgb_path)
    if overwrite or not out_mp4.is_file():
        write_mask_mp4(mask, out_mp4, fps)

    return {
        "npz_file": out_npz.name,
        "mp4_file": out_mp4.name,
        "format": "npz_uint8_binary + mp4_h264_mono",
        "binary_values": [0, 255],
        "num_frames": int(T),
        "height": int(H),
        "width": int(W),
        "fps": float(fps),
    }


def write_meta(traj_dir: Path, *, task_id: str, traj_name: str, T: int,
               cam_intrinsics_path: Path, target_objs: dict, target_seg_ids: list[int],
               body_pose_files: dict[str, str], mesh_meta: dict,
               mask_meta: dict, actors: list) -> None:
    """Write meta.json conforming to the required spec."""
    K = np.load(str(cam_intrinsics_path)).astype(np.float64) if cam_intrinsics_path.exists() else None

    meta = {
        "task_id": task_id,
        "traj_name": traj_name,
        "num_frames": int(T),
        "actors": actors,

        "coordinate_convention": {
            "sceneflow":     "opengl_camera",
            "anchor_points": "opengl_camera_ref_frame",
            "flow_vectors":  "opengl_camera_ref_frame",
        },
        "camera_convention": "opencv",
        "flow_convention":   "opengl",
        "camera_pose_layout": "cam2world absolute; cam_poses relative to cam0",
        "cam2world_file":    "cam2world_cv.npy",
        "cam2world_gl_file": "cam2world_gl.npy",
        "cam_poses_file":    "cam_poses.npy",
        "pose_layout":       "body->cam",

        "target_object": {
            "body_names": [t["short"] for t in target_objs],
            "seg_ids":    list(target_seg_ids),
        },
        "body_poses": {
            "format": "(T,4,4) float32 body-to-camera homogeneous transform in OpenCV camera frame",
            "files":  body_pose_files,
        },
        "meshes": mesh_meta,
        "target_obj_mask": mask_meta,
    }

    if K is not None:
        meta["cam_intrinsics_file"] = "cam_intrinsics.npy"
        meta["cam_intrinsics"] = K.tolist()

    (traj_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Top-level: process one traj (assumes env model handed in for speed)
# ---------------------------------------------------------------------------

def process_traj_with_model(traj_dir: Path, task_id: str, mapping_entry: dict,
                            env_model, overwrite: bool) -> str:
    h5_path = find_h5(traj_dir)
    traj_key = h5_path.stem  # e.g. "traj_0"

    legacy = traj_dir / "traj_task.json"
    meta_path = traj_dir / "meta.json"
    if legacy.exists():
        actors = json.loads(legacy.read_text())["actors"]
    elif meta_path.exists():
        actors = json.loads(meta_path.read_text()).get("actors", [])
    else:
        raise FileNotFoundError(f"no traj_task.json or meta.json in {traj_dir}")

    targets = resolve_targets(actors, mapping_entry)
    if not targets:
        return f"SKIP {traj_dir.name}: no matching target bodies"

    # 1) cam2world_{cv,gl}.npy + cam_poses.npy (relative CV)
    cam2world_cv, cam2world_gl, cam_poses_rel_cv = build_camera_poses(traj_dir)
    T = int(cam2world_cv.shape[0])

    np.save(str(traj_dir / "cam2world_cv.npy"), cam2world_cv)
    np.save(str(traj_dir / "cam2world_gl.npy"), cam2world_gl)
    np.save(str(traj_dir / "cam_poses.npy"),    cam_poses_rel_cv)

    # 2) pose_<obj>.npy from H5
    body_pose_files = emit_target_files_h5(traj_dir, h5_path, traj_key, targets)

    # 3) mesh_<obj>.ply
    target_shorts = [t["short"] for t in targets]
    mesh_meta = emit_meshes(traj_dir, env_model, target_shorts, overwrite)

    # 4) mask_<obj>.npz + target_obj_mask.mp4
    target_seg_ids = sorted({int(t["seg_id"]) for t in targets})
    mask_meta = emit_mask(traj_dir, target_seg_ids, target_shorts, overwrite)

    # 5) meta.json
    write_meta(
        traj_dir, task_id=task_id, traj_name=traj_dir.name, T=T,
        cam_intrinsics_path=traj_dir / "cam_intrinsics.npy",
        target_objs=targets, target_seg_ids=target_seg_ids,
        body_pose_files=body_pose_files, mesh_meta=mesh_meta,
        mask_meta=mask_meta, actors=actors,
    )

    if legacy.exists():
        legacy.unlink()

    return f"OK   {traj_dir.name}: targets={target_shorts} mesh_files={list(mesh_meta['files'].keys())}"


# ---------------------------------------------------------------------------
# Per-task driver: builds env once, processes all trajs sequentially.
# (One process per task in --root mode.)
# ---------------------------------------------------------------------------

def process_task(task_dir: Path, mapping: dict, overwrite: bool) -> dict:
    task_id = task_dir.name
    cam = task_dir / "camera_data"
    if task_id not in mapping:
        return {"task": task_id, "ok": 0, "skip": 0, "err": 0, "msg": "no mapping entry"}
    if not cam.is_dir():
        return {"task": task_id, "ok": 0, "skip": 0, "err": 0, "msg": "no camera_data"}

    trajs = sorted(
        (p for p in cam.iterdir() if p.is_dir() and p.name.startswith("traj_")),
        key=lambda p: int(p.name.split("_", 1)[1]) if p.name.split("_", 1)[1].isdigit() else 1 << 30,
    )
    if not trajs:
        return {"task": task_id, "ok": 0, "skip": 0, "err": 0, "msg": "no traj_*"}

    # Build env model ONCE for the whole task
    env, model = build_env_model(task_id)
    ok = skip = err = 0
    errors: list[str] = []
    try:
        for td in trajs:
            try:
                msg = process_traj_with_model(td, task_id, mapping[task_id], model, overwrite)
                if msg.startswith("OK"):
                    ok += 1
                else:
                    skip += 1
            except Exception as e:
                err += 1
                errors.append(f"{td.name}: {type(e).__name__}: {e}")
    finally:
        try:
            env.close()
        except Exception:
            pass
    return {"task": task_id, "ok": ok, "skip": skip, "err": err, "errors": errors[:5]}


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--traj-dir", type=Path, help="single traj_* dir")
    g.add_argument("--task-dir", type=Path, help="single <task>/ dir (contains camera_data/)")
    g.add_argument("--root",     type=Path, help="dataset root containing <task>/camera_data/traj_*/")
    ap.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("-j", "--jobs", type=int, default=1,
                    help="parallel tasks (one process per task; each builds its own env)")
    args = ap.parse_args()

    mapping_all = json.loads(args.mapping.read_text())
    mapping = {k: v for k, v in mapping_all.items() if not k.startswith("_")}

    if args.traj_dir is not None:
        td = args.traj_dir.resolve()
        task_id = td.parent.parent.name
        if task_id not in mapping:
            print(f"[ERR] no mapping for {task_id}")
            return 1
        env, model = build_env_model(task_id)
        try:
            msg = process_traj_with_model(td, task_id, mapping[task_id], model, args.overwrite)
        finally:
            try:
                env.close()
            except Exception:
                pass
        print(msg)
        return 0

    if args.task_dir is not None:
        r = process_task(args.task_dir.resolve(), mapping, args.overwrite)
        print(f"  {r['task']:32s} ok={r['ok']:4d} skip={r['skip']:4d} err={r['err']:3d}  "
              f"{r.get('msg','')}")
        for e in r.get("errors", []):
            print(f"    · {e}")
        return 0 if r["err"] == 0 else 1

    # --root
    root = args.root.resolve()
    task_dirs = sorted(p for p in root.iterdir()
                       if p.is_dir() and (p / "camera_data").is_dir())
    if not task_dirs:
        print(f"no <task>/camera_data found in {root}", file=sys.stderr)
        return 1

    print(f"[build_mikasa_format] {len(task_dirs)} tasks under {root}, jobs={args.jobs}")
    t0 = time.time()
    summary = []
    failures: list[str] = []

    if args.jobs <= 1:
        for tdir in task_dirs:
            r = process_task(tdir, mapping, args.overwrite)
            summary.append(r)
            print(f"  {r['task']:32s} ok={r['ok']:4d} skip={r['skip']:4d} err={r['err']:3d}  "
                  f"{r.get('msg','')}")
            for e in r.get("errors", []):
                failures.append(f"{r['task']}/{e}")
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(process_task, td, mapping, args.overwrite): td for td in task_dirs}
            for f in as_completed(futs):
                td = futs[f]
                try:
                    r = f.result()
                except Exception as e:
                    r = {"task": td.name, "ok": 0, "skip": 0, "err": 1,
                         "errors": [f"executor:{type(e).__name__}:{e}"]}
                summary.append(r)
                print(f"  {r['task']:32s} ok={r['ok']:4d} skip={r['skip']:4d} err={r['err']:3d}  "
                      f"{r.get('msg','')}")
                for e in r.get("errors", []):
                    failures.append(f"{r['task']}/{e}")

    total_ok   = sum(s["ok"]   for s in summary)
    total_skip = sum(s["skip"] for s in summary)
    total_err  = sum(s["err"]  for s in summary)
    print(f"\n[summary] ok={total_ok} skip={total_skip} err={total_err}  "
          f"elapsed={(time.time()-t0)/60:.1f}m")
    if failures:
        print(f"[failures] showing first 20 of {len(failures)}:")
        for x in failures[:20]:
            print(f"  · {x}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
