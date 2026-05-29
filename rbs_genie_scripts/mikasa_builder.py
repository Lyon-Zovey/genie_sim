"""Step-5 replacement for Isaac Sim / GenieSim datasets.

Generates all MIKASA-format derived files for one traj_N/ directory while
Isaac Sim is still running (called from SceneFlowRecorder.flush), so no
second simulator launch is needed.

Outputs written into traj_N/ (same spec as rbs_scripts/build_mikasa_format.py):

    cam2world_cv.npy        (T,4,4) OpenCV cam-to-world absolute
    cam2world_gl.npy        (T,4,4) OpenGL cam-to-world absolute
    cam_poses.npy           (T,4,4) OpenCV relative to cam0  [overwrites]
    pose_<obj>.npy          (T,4,4) body→cam OpenCV, per target body
    mesh_<obj>.ply          body-local PLY, per target body
    mask_<obj>.npz          (T,H,W) uint8 {0,255}
    target_obj_mask.mp4     same as mask but H.264 mono video
    meta.json               full metadata spec

Isaac Sim mesh extraction
--------------------------
USD assets live in $SIM_ASSETS/objects/<...>/<asset>/  and the prim path
inside the running stage looks like  /World/<object_id>/...
We extract the mesh by iterating UsdGeom.Mesh prims under the object's root
prim, collecting all (vertex, face) data in body-local frame, and writing a
binary little-endian PLY.

Coordinate conventions
-----------------------
cam_poses.npy (input from SceneFlowRecorder) is OpenGL cam-to-world.
FLIP4 = diag(1,-1,-1,1) converts OpenGL ↔ OpenCV.
"""

from __future__ import annotations

import json
import re
import struct
import subprocess
from pathlib import Path

import numpy as np

# diag(1,-1,-1,1) — converts between OpenGL and OpenCV camera frame
FLIP4 = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_name(raw: str) -> str:
    s = str(raw)
    for prefix in ("body:", "link:"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return _SAFE_RE.sub("_", s)


# ---------------------------------------------------------------------------
# PLY writer (binary little-endian, float32 vertices + int32 faces)
# ---------------------------------------------------------------------------

def write_ply(path: Path, V: np.ndarray, F: np.ndarray) -> None:
    V = np.ascontiguousarray(V, dtype=np.float32)
    F = np.ascontiguousarray(F, dtype=np.int32)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {V.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {F.shape[0]}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    face_dtype = np.dtype([("n", "u1"), ("idx", "<i4", (3,))])
    face_arr = np.empty(F.shape[0], dtype=face_dtype)
    face_arr["n"] = 3
    face_arr["idx"] = F
    with open(path, "wb") as fp:
        fp.write(header.encode("ascii"))
        fp.write(V.tobytes(order="C"))
        fp.write(face_arr.tobytes(order="C"))


# ---------------------------------------------------------------------------
# Mesh extraction from the live USD stage (runs inside Isaac Sim process)
# ---------------------------------------------------------------------------

def _extract_mesh_from_stage(object_root_prim_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Walk all UsdGeom.Mesh prims under object_root_prim_path and merge into
    one body-local mesh.  Returns (V float32 (N,3), F int32 (M,3)) or None."""
    try:
        import omni.usd
        from pxr import UsdGeom, Gf, Usd
    except ImportError:
        return None

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None

    root_prim = stage.GetPrimAtPath(object_root_prim_path)
    if not root_prim.IsValid():
        return None

    # Collect the world-space transform of the root body so we can express
    # vertices in body-local frame (same convention as MuJoCo extract_meshes).
    root_xformable = UsdGeom.Xformable(root_prim)
    time = Usd.TimeCode.Default()
    root_world_mat = np.array(root_xformable.ComputeLocalToWorldTransform(time), dtype=np.float64).T  # 4×4

    all_V: list[np.ndarray] = []
    all_F: list[np.ndarray] = []
    vert_offset = 0

    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)

        points_attr = mesh.GetPointsAttr().Get(time)
        if points_attr is None or len(points_attr) == 0:
            continue
        V_local = np.array(points_attr, dtype=np.float64)  # (N, 3)

        face_vc = mesh.GetFaceVertexCountsAttr().Get(time)
        face_vi = mesh.GetFaceVertexIndicesAttr().Get(time)
        if face_vc is None or face_vi is None:
            continue

        # Triangulate (USD meshes may have quads or n-gons)
        triangles: list[list[int]] = []
        vi_list = list(face_vi)
        idx = 0
        for n_verts in face_vc:
            fan = vi_list[idx:idx + n_verts]
            for k in range(1, n_verts - 1):
                triangles.append([fan[0], fan[k], fan[k + 1]])
            idx += n_verts
        if not triangles:
            continue
        F_local = np.array(triangles, dtype=np.int32)

        # Transform vertices from prim-local → world, then → root-body-local
        prim_xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time)
        prim_world = np.array(prim_xform, dtype=np.float64).T  # 4×4

        # world = prim_world @ [V; 1]
        V_h = np.hstack([V_local, np.ones((len(V_local), 1), dtype=np.float64)])
        V_world = (prim_world @ V_h.T).T[:, :3]

        # body-local = inv(root_world) @ [V_world; 1]
        root_inv = np.linalg.inv(root_world_mat)
        V_body = (root_inv @ np.hstack([V_world, np.ones((len(V_world), 1))]).T).T[:, :3]

        all_V.append(V_body.astype(np.float32))
        all_F.append((F_local + vert_offset).astype(np.int32))
        vert_offset += len(V_body)

    if not all_V:
        return None

    return np.concatenate(all_V, axis=0), np.concatenate(all_F, axis=0)


def extract_object_mesh(prim_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Public entry-point — try live USD stage first, return None on failure."""
    return _extract_mesh_from_stage(prim_path)


# ---------------------------------------------------------------------------
# Camera-pose helpers
# ---------------------------------------------------------------------------

def build_camera_poses(cam_poses_gl: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Input:  cam_poses_gl  (T,4,4) float32  OpenGL cam-to-world absolute
                           (as stored in cam_poses.npy by SceneFlowRecorder)
    Output: (cam2world_cv, cam2world_gl, cam_poses_rel_cv)  all (T,4,4) float32
    """
    cam2world_gl = cam_poses_gl.astype(np.float32)
    cam2world_cv = (cam2world_gl @ FLIP4).astype(np.float32)

    inv0 = _inv44(cam2world_cv[0:1])                    # (1,4,4)
    cam_poses_rel_cv = (inv0 @ cam2world_cv).astype(np.float32)
    cam_poses_rel_cv[0] = np.eye(4, dtype=np.float32)   # enforce identity
    return cam2world_cv, cam2world_gl, cam_poses_rel_cv


def _inv44(T: np.ndarray) -> np.ndarray:
    """Batch invert (N,4,4) rigid transforms (R, t) without linalg.inv."""
    R  = T[..., :3, :3]
    t  = T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3,  3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out


# ---------------------------------------------------------------------------
# Body → camera pose from H5
# ---------------------------------------------------------------------------

def compute_body_to_cam_cv(
    h5_path: Path,
    traj_key: str,
    seg_id: int,
) -> np.ndarray:
    """Read camera_position / camera_quaternion (OpenGL) from H5,
    return (T,4,4) float32 body→cam in OpenCV."""
    import h5py
    from scipy.spatial.transform import Rotation

    with h5py.File(str(h5_path), "r") as f:
        g = f[f"{traj_key}/id_poses/{seg_id}"]
        cp = g["camera_position"][...]    # (T,3) OpenGL
        cq = g["camera_quaternion"][...]  # (T,4) wxyz OpenGL body→cam

    T_frames = cp.shape[0]
    out_gl = np.zeros((T_frames, 4, 4), dtype=np.float32)
    for t in range(T_frames):
        w, x, y, z = cq[t]
        R = Rotation.from_quat([x, y, z, w]).as_matrix().astype(np.float32)
        out_gl[t, :3, :3] = R
        out_gl[t, :3,  3] = cp[t]
        out_gl[t,  3,  3] = 1.0
    # OpenGL body→cam → OpenCV body→cam: left-multiply FLIP4
    out_cv = (FLIP4[None] @ out_gl).astype(np.float32)
    return out_cv


# ---------------------------------------------------------------------------
# Mask generation
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
        return 16.0


def _write_mask_mp4(mask: np.ndarray, out_mp4: Path, fps: float) -> None:
    """Encode (T,H,W) uint8 {0,255} as mono H.264 mp4."""
    T, H, W = mask.shape
    raw = mask.tobytes()

    def _encode(codec: str) -> bool:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "gray",
            "-s", f"{W}x{H}", "-r", str(int(fps)), "-i", "-",
            "-c:v", codec,
        ]
        if codec == "h264_nvenc":
            cmd += ["-preset", "p4", "-rc", "constqp", "-qp", "0", "-pix_fmt", "yuv420p"]
        else:
            cmd += ["-crf", "0", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
        cmd.append(str(out_mp4))
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr = p.communicate(input=raw)
            if p.returncode != 0:
                import logging
                logging.getLogger(__name__).warning(
                    f"ffmpeg {codec} failed (rc={p.returncode}): {stderr.decode(errors='replace')[:300]}"
                )
                return False
            return True
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"ffmpeg {codec} exception: {e}")
            return False

    if not _encode("h264_nvenc"):
        if not _encode("libx264"):
            raise RuntimeError(f"ffmpeg mask encode failed: {out_mp4}")


def build_mask(
    seg: np.ndarray,       # (T,H,W) int32
    target_seg_ids: list[int],
    traj_dir: Path,
    obj_safe_name: str,
    fps: float,
    overwrite: bool = False,
) -> dict:
    mask = np.isin(seg, target_seg_ids).astype(np.uint8) * 255
    T, H, W = mask.shape

    npz_path = traj_dir / f"mask_{obj_safe_name}.npz"
    mp4_path = traj_dir / "target_obj_mask.mp4"

    if overwrite or not npz_path.exists():
        np.savez_compressed(str(npz_path), mask=mask)
    if overwrite or not mp4_path.exists():
        _write_mask_mp4(mask, mp4_path, fps)

    return {
        "npz_file": npz_path.name,
        "mp4_file": mp4_path.name,
        "format": "npz_uint8_binary + mp4_h264_mono",
        "binary_values": [0, 255],
        "num_frames": int(T),
        "height": int(H),
        "width": int(W),
        "fps": float(fps),
    }


# ---------------------------------------------------------------------------
# meta.json writer
# ---------------------------------------------------------------------------

def write_meta(
    traj_dir: Path,
    *,
    task_id: str,
    traj_name: str,
    T: int,
    actors: list,
    cam_intrinsics_path: Path,
    target_objs: list[dict],       # [{seg_id, name, short}]
    body_pose_files: dict[str, str],
    mesh_meta: dict,
    mask_meta: dict,
) -> None:
    K = None
    if cam_intrinsics_path.exists():
        K = np.load(str(cam_intrinsics_path)).astype(np.float64)

    meta: dict = {
        "task_id": task_id,
        "traj_name": traj_name,
        "num_frames": int(T),
        "actors": actors,

        "coordinate_convention": {
            "sceneflow":     "opengl_camera",
            "anchor_points": "opengl_camera_ref_frame",
            "flow_vectors":  "opengl_camera_ref_frame",
        },
        "camera_convention":   "opencv",
        "flow_convention":     "opengl",
        "camera_pose_layout":  "cam2world absolute; cam_poses relative to cam0",
        "cam2world_file":      "cam2world_cv.npy",
        "cam2world_gl_file":   "cam2world_gl.npy",
        "cam_poses_file":      "cam_poses.npy",
        "pose_layout":         "body->cam",

        "target_object": {
            "body_names": [t["short"] for t in target_objs],
            "seg_ids":    [t["seg_id"] for t in target_objs],
        },
        "body_poses": {
            "format": "(T,4,4) float32 body-to-camera homogeneous transform in OpenCV camera frame",
            "files":  body_pose_files,
        },
        "meshes":          mesh_meta,
        "target_obj_mask": mask_meta,
    }

    if K is not None:
        meta["cam_intrinsics_file"] = "cam_intrinsics.npy"
        meta["cam_intrinsics"] = K.tolist()

    (traj_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Top-level entry point — called from SceneFlowRecorder.flush()
# ---------------------------------------------------------------------------

class MikasaBuilder:
    """Builds all Step-5 MIKASA outputs for a single traj_N/ directory.

    Parameters
    ----------
    traj_dir : Path
        The traj_N/ directory that SceneFlowRecorder just wrote.
    actors : list[dict]
        [{"seg_id": int, "name": "body:<prim_name>"}, ...]
        Taken from traj_task.json / SceneFlowRecorder._obj_bufs.
    target_seg_ids : list[int]
        Seg IDs that belong to the *target* object(s) to track.
        If None, all actors are treated as targets.
    object_prim_paths : dict[int, str]
        {seg_id: "/World/<prim_path>"} for live USD mesh extraction.
    seg_array : np.ndarray or None
        (T,H,W) int32 segmentation, if already in memory (avoids re-loading
        seg.b2nd from disk).
    overwrite : bool
        Overwrite existing output files.
    """

    def __init__(
        self,
        traj_dir: Path,
        actors: list[dict],
        target_seg_ids: list[int] | None,
        object_prim_paths: dict[int, str] | None = None,
        seg_array: np.ndarray | None = None,
        overwrite: bool = False,
    ) -> None:
        self.traj_dir = Path(traj_dir)
        self.actors = actors
        self.overwrite = overwrite
        self._seg_array = seg_array

        # Build seg_id → actor name mapping
        self._id_to_name: dict[int, str] = {
            a["seg_id"]: a["name"] for a in actors
        }

        if target_seg_ids is None:
            self.target_seg_ids = [a["seg_id"] for a in actors]
        else:
            self.target_seg_ids = list(target_seg_ids)

        self.object_prim_paths = object_prim_paths or {}

    # ------------------------------------------------------------------

    def run(self) -> None:
        traj_dir = self.traj_dir
        traj_key = traj_dir.name  # "traj_0"

        # 1) Camera poses ─────────────────────────────────────────────
        raw_gl = np.load(str(traj_dir / "cam_poses.npy")).astype(np.float32)
        cam2world_cv, cam2world_gl, cam_poses_rel_cv = build_camera_poses(raw_gl)
        T = int(cam2world_cv.shape[0])

        np.save(str(traj_dir / "cam2world_cv.npy"), cam2world_cv)
        np.save(str(traj_dir / "cam2world_gl.npy"), cam2world_gl)
        np.save(str(traj_dir / "cam_poses.npy"),    cam_poses_rel_cv)  # overwrite

        # 2) Body → cam poses (from H5) ────────────────────────────────
        h5_candidates = sorted(traj_dir.glob("*.h5"))
        if not h5_candidates:
            raise FileNotFoundError(f"no .h5 in {traj_dir}")
        h5_path = h5_candidates[0]

        body_pose_files: dict[str, str] = {}
        for seg_id in self.target_seg_ids:
            name = self._id_to_name.get(seg_id, f"obj_{seg_id}")
            short = safe_name(name)
            pose_cv = compute_body_to_cam_cv(h5_path, traj_key, seg_id)
            out_name = f"pose_{short}.npy"
            np.save(str(traj_dir / out_name), pose_cv)
            body_pose_files[short] = out_name

        # 3) Meshes (live USD stage) ────────────────────────────────────
        mesh_files: dict[str, str] = {}
        mesh_seg_ids: dict[str, int] = {}
        mesh_n_verts: dict[str, int] = {}
        mesh_n_faces: dict[str, int] = {}

        for seg_id in self.target_seg_ids:
            name = self._id_to_name.get(seg_id, f"obj_{seg_id}")
            short = safe_name(name)
            ply_path = traj_dir / f"mesh_{short}.ply"

            if not ply_path.exists() or self.overwrite:
                prim_path = self.object_prim_paths.get(seg_id)
                result = extract_object_mesh(prim_path) if prim_path else None

                if result is not None:
                    V, F = result
                    write_ply(ply_path, V, F)
                    mesh_files[short]    = ply_path.name
                    mesh_seg_ids[short]  = int(seg_id)
                    mesh_n_verts[short]  = int(V.shape[0])
                    mesh_n_faces[short]  = int(F.shape[0])
                # If extraction failed, skip this object silently (no .ply written)
            else:
                import struct as _s
                # Read existing PLY header to get counts
                with open(ply_path, "rb") as fp:
                    hdr = b""
                    while b"end_header" not in hdr:
                        hdr += fp.read(256)
                hdr_txt = hdr[:hdr.index(b"end_header")].decode("ascii", errors="replace")
                nv = next((int(l.split()[-1]) for l in hdr_txt.splitlines() if l.startswith("element vertex")), 0)
                nf = next((int(l.split()[-1]) for l in hdr_txt.splitlines() if l.startswith("element face")), 0)
                mesh_files[short]   = ply_path.name
                mesh_seg_ids[short] = int(seg_id)
                mesh_n_verts[short] = nv
                mesh_n_faces[short] = nf

        mesh_meta = {
            "format": "binary_little_endian PLY; vertices are in body-local "
                      "frame (p_cam_h = T_body2cam[t] @ [v_body; 1]).",
            "files":        mesh_files,
            "seg_ids":      mesh_seg_ids,
            "num_vertices": mesh_n_verts,
            "num_faces":    mesh_n_faces,
        }

        # 4) Mask ──────────────────────────────────────────────────────
        seg = self._load_seg(traj_dir)
        fps = _ffprobe_fps(traj_dir / "rgb.mp4")
        first_short = safe_name(
            self._id_to_name.get(self.target_seg_ids[0], "target")
        ) if self.target_seg_ids else "target"
        mask_meta = build_mask(
            seg, self.target_seg_ids, traj_dir, first_short, fps, self.overwrite
        )

        # 5) meta.json ─────────────────────────────────────────────────
        target_objs = [
            {"seg_id": sid,
             "name":   self._id_to_name.get(sid, f"body:obj_{sid}"),
             "short":  safe_name(self._id_to_name.get(sid, f"obj_{sid}"))}
            for sid in self.target_seg_ids
        ]
        task_id = self._read_task_id(traj_dir)
        write_meta(
            traj_dir,
            task_id=task_id,
            traj_name=traj_key,
            T=T,
            actors=self.actors,
            cam_intrinsics_path=traj_dir / "cam_intrinsics.npy",
            target_objs=target_objs,
            body_pose_files=body_pose_files,
            mesh_meta=mesh_meta,
            mask_meta=mask_meta,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_seg(self, traj_dir: Path) -> np.ndarray:
        if self._seg_array is not None:
            return self._seg_array

        b2nd = traj_dir / "seg.b2nd"
        npy  = traj_dir / "seg.npy"

        if b2nd.exists():
            import blosc2
            return blosc2.open(str(b2nd))[:]
        if npy.exists():
            return np.load(str(npy))
        raise FileNotFoundError(f"no seg.b2nd or seg.npy in {traj_dir}")

    def _read_task_id(self, traj_dir: Path) -> str:
        for name in ("traj_task.json", "meta.json"):
            p = traj_dir / name
            if p.exists():
                d = json.loads(p.read_text())
                tid = d.get("task_id") or d.get("task") or ""
                if tid:
                    return str(tid)
        return traj_dir.parent.parent.name  # fallback: task directory name


# ---------------------------------------------------------------------------
# Standalone CLI (run outside Isaac Sim, mesh step is skipped if USD unavailable)
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse, sys, time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ap = argparse.ArgumentParser(
        description="Generate MIKASA Step-5 outputs for GenieSim trajectories."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--traj-dir",  type=Path, help="single traj_N/ dir")
    g.add_argument("--task-dir",  type=Path, help="<task>/ dir containing camera_data/")
    g.add_argument("--root",      type=Path, help="dataset root containing <task>/camera_data/")
    ap.add_argument("--target-seg-ids", type=int, nargs="+", default=None,
                    help="target seg IDs (default: all actors)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("-j", "--jobs", type=int, default=1)
    args = ap.parse_args()

    def _actors_from_traj(td: Path) -> list[dict]:
        for name in ("traj_task.json", "meta.json"):
            p = td / name
            if p.exists():
                d = json.loads(p.read_text())
                if "actors" in d:
                    return d["actors"]
        return []

    def _process_one(td: Path) -> str:
        actors = _actors_from_traj(td)
        if not actors:
            return f"SKIP {td}: no actors"
        bld = MikasaBuilder(
            traj_dir=td,
            actors=actors,
            target_seg_ids=args.target_seg_ids,
            overwrite=args.overwrite,
        )
        bld.run()
        return f"OK   {td.name}"

    if args.traj_dir:
        print(_process_one(args.traj_dir.resolve()))
        return

    if args.task_dir:
        cam = args.task_dir.resolve() / "camera_data"
        trajs = sorted(p for p in cam.iterdir() if p.is_dir() and p.name.startswith("traj_"))
    else:
        root = args.root.resolve()
        trajs = sorted(
            td
            for task in root.iterdir() if (task / "camera_data").is_dir()
            for td in (task / "camera_data").iterdir()
            if td.is_dir() and td.name.startswith("traj_")
        )

    if not trajs:
        print("no traj dirs found", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    ok = err = 0

    if args.jobs <= 1:
        for td in trajs:
            try:
                print(_process_one(td))
                ok += 1
            except Exception as e:
                print(f"ERR  {td}: {e}")
                err += 1
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_process_one, td): td for td in trajs}
            for fut in as_completed(futs):
                try:
                    print(fut.result())
                    ok += 1
                except Exception as e:
                    print(f"ERR  {futs[fut]}: {e}")
                    err += 1

    print(f"\n[summary] ok={ok} err={err}  elapsed={(time.time()-t0)/60:.1f}m")
    if err:
        sys.exit(1)


if __name__ == "__main__":
    _main()
