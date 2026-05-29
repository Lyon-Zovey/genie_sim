#!/usr/bin/env python3
"""Extract per-body PLY meshes baked into the body frame.

For each traj_N/ writes:
  mesh_<obj>.ply        binary little-endian PLY (vertex + face)
                        vertices are in the body's local frame already, i.e.
                        p_cam_h = T_body2cam[t] @ [v_body; 1]
and updates traj_N/meta.json "meshes" section indexed by name.

Only bodies listed in meta.json["target_object"]["body_names"] are exported.

<obj> is the body name with the leading "body:" prefix stripped.

Geom types supported:
  MESH       → uses the original mesh asset transformed by mesh_quat/pos/scale
               then by geom_quat/pos
  BOX        → 12 triangles
  SPHERE     → icosphere subdivided to ~80 faces, scaled to radius
  CAPSULE    → cylinder + 2 hemispheres
  CYLINDER   → tessellated cylinder with caps
  ELLIPSOID  → icosphere scaled non-uniformly
  PLANE      → skipped (infinite, no useful mesh)
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import traceback
from pathlib import Path

import numpy as np

import mujoco
import metaworld


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_name(raw: str) -> str:
    if raw.startswith("body:"):
        raw = raw[len("body:"):]
    elif raw.startswith("link:"):
        raw = raw[len("link:"):]
    return _SAFE_NAME.sub("_", raw)


def quat_wxyz_to_R(q: np.ndarray | list[float]) -> np.ndarray:
    """(w,x,y,z) -> (3,3) rotation matrix, robust to non-unit input."""
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Primitive mesh generators (vertices in geom-local frame BEFORE geom_pos/quat)
# All produce (V, F) where V is (N,3) float64, F is (M,3) int32 triangles.
# ---------------------------------------------------------------------------

def box_mesh(half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hx, hy, hz = float(half[0]), float(half[1]), float(half[2])
    V = np.array([
        [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
        [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
    ], dtype=np.float64)
    F = np.array([
        [0,2,1],[0,3,2],   # -z
        [4,5,6],[4,6,7],   # +z
        [0,1,5],[0,5,4],   # -y
        [3,7,6],[3,6,2],   # +y
        [0,4,7],[0,7,3],   # -x
        [1,2,6],[1,6,5],   # +x
    ], dtype=np.int32)
    return V, F


def icosphere(subdiv: int = 1) -> tuple[np.ndarray, np.ndarray]:
    t = (1.0 + np.sqrt(5.0)) / 2.0
    V = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=np.float64)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    F = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=np.int32)
    for _ in range(subdiv):
        edge_cache: dict[tuple[int,int], int] = {}
        V_list = V.tolist()
        new_F = []
        def midpoint(a: int, b: int) -> int:
            k = (a, b) if a < b else (b, a)
            if k in edge_cache:
                return edge_cache[k]
            mid = (V[a] + V[b]) / 2.0
            mid /= np.linalg.norm(mid)
            V_list.append(mid.tolist())
            idx = len(V_list) - 1
            edge_cache[k] = idx
            return idx
        for tri in F:
            a, b, c = (int(x) for x in tri)
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_F.extend([[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]])
        V = np.asarray(V_list, dtype=np.float64)
        F = np.asarray(new_F, dtype=np.int32)
    return V, F


def sphere_mesh(radius: float) -> tuple[np.ndarray, np.ndarray]:
    V, F = icosphere(subdiv=1)
    return V * float(radius), F


def ellipsoid_mesh(half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    V, F = icosphere(subdiv=1)
    V = V * np.asarray(half, dtype=np.float64)[None, :]
    return V, F


def cylinder_mesh(radius: float, half_height: float,
                  segs: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Cylinder along local z, length=2*half_height."""
    r = float(radius); h = float(half_height)
    angles = np.linspace(0.0, 2.0 * np.pi, segs, endpoint=False)
    cx, cy = np.cos(angles) * r, np.sin(angles) * r
    bottom = np.stack([cx, cy, -np.full(segs, h)], axis=1)
    top    = np.stack([cx, cy,  np.full(segs, h)], axis=1)
    bot_c = np.array([[0., 0., -h]])
    top_c = np.array([[0., 0.,  h]])
    V = np.concatenate([bottom, top, bot_c, top_c], axis=0)
    bc, tc = 2 * segs, 2 * segs + 1
    F: list[list[int]] = []
    for i in range(segs):
        j = (i + 1) % segs
        F.append([i, j, segs + j])
        F.append([i, segs + j, segs + i])
        F.append([bc, j, i])
        F.append([tc, segs + i, segs + j])
    return V, np.asarray(F, dtype=np.int32)


def capsule_mesh(radius: float, half_height: float,
                 segs: int = 16, hemi_rings: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Capsule along local z: cylinder of length 2*half_height + two hemispheres."""
    r = float(radius); h = float(half_height)
    cyl_V, cyl_F = cylinder_mesh(r, h, segs=segs)
    cyl_V = cyl_V[: 2 * segs]
    cyl_F_clean: list[list[int]] = []
    for tri in cyl_F:
        if 2*segs in tri or 2*segs+1 in tri:
            continue
        cyl_F_clean.append(tri.tolist())
    cyl_F = np.asarray(cyl_F_clean, dtype=np.int32)

    def hemisphere(top: bool) -> tuple[np.ndarray, np.ndarray]:
        rings = hemi_rings
        thetas = np.linspace(0.0, np.pi / 2.0, rings + 1)[1:]
        phis   = np.linspace(0.0, 2.0 * np.pi, segs, endpoint=False)
        verts: list[list[float]] = []
        for th in thetas:
            zr = np.cos(th) * r
            rr = np.sin(th) * r
            for ph in phis:
                verts.append([rr * np.cos(ph), rr * np.sin(ph),
                              (h + zr) if top else (-h - zr)])
        pole = [[0.0, 0.0, (h + r) if top else (-h - r)]]
        V = np.asarray(verts + pole, dtype=np.float64)
        faces: list[list[int]] = []
        for k in range(rings - 1):
            for i in range(segs):
                a = k * segs + i
                b = k * segs + (i + 1) % segs
                c = (k + 1) * segs + i
                d = (k + 1) * segs + (i + 1) % segs
                if top:
                    faces.append([a, c, d]); faces.append([a, d, b])
                else:
                    faces.append([a, d, c]); faces.append([a, b, d])
        pole_idx = len(V) - 1
        last_ring = (rings - 1) * segs
        for i in range(segs):
            a = last_ring + i
            b = last_ring + (i + 1) % segs
            if top:
                faces.append([a, pole_idx, b])
            else:
                faces.append([a, b, pole_idx])
        return V, np.asarray(faces, dtype=np.int32)

    V_top, F_top = hemisphere(top=True)
    V_bot, F_bot = hemisphere(top=False)

    n_cyl = cyl_V.shape[0]
    n_top = V_top.shape[0]

    bottom_ring_cyl = np.arange(0, segs)
    top_ring_cyl    = np.arange(segs, 2 * segs)

    bot_first_ring_local = np.arange(0, segs)
    top_first_ring_local = np.arange(0, segs)

    F_top_off = F_top + n_cyl
    F_bot_off = F_bot + n_cyl + n_top

    stitch: list[list[int]] = []
    for i in range(segs):
        j = (i + 1) % segs
        a = int(top_ring_cyl[i])
        b = int(top_ring_cyl[j])
        c = int(top_first_ring_local[i]) + n_cyl
        d = int(top_first_ring_local[j]) + n_cyl
        stitch.append([a, b, d])
        stitch.append([a, d, c])
    for i in range(segs):
        j = (i + 1) % segs
        a = int(bottom_ring_cyl[i])
        b = int(bottom_ring_cyl[j])
        c = int(bot_first_ring_local[i]) + n_cyl + n_top
        d = int(bot_first_ring_local[j]) + n_cyl + n_top
        stitch.append([a, d, b])
        stitch.append([a, c, d])

    V = np.concatenate([cyl_V, V_top, V_bot], axis=0)
    F = np.concatenate([cyl_F, F_top_off, F_bot_off,
                        np.asarray(stitch, dtype=np.int32)], axis=0)
    return V, F


def extract_mesh_asset(model, mesh_id: int) -> tuple[np.ndarray, np.ndarray]:
    v_a = int(model.mesh_vertadr[mesh_id])
    v_n = int(model.mesh_vertnum[mesh_id])
    f_a = int(model.mesh_faceadr[mesh_id])
    f_n = int(model.mesh_facenum[mesh_id])
    V = np.asarray(model.mesh_vert[v_a:v_a + v_n], dtype=np.float64)
    F = np.asarray(model.mesh_face[f_a:f_a + f_n], dtype=np.int32)
    return V, F


def geom_to_body_mesh(model, gid: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Build (V, F) for one geom in BODY-LOCAL coordinates.  Returns None for PLANE."""
    gtype = int(model.geom_type[gid])
    size  = np.asarray(model.geom_size[gid], dtype=np.float64)

    if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return None
    if gtype == int(mujoco.mjtGeom.mjGEOM_HFIELD):
        return None

    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        V, F = box_mesh(size[:3])
    elif gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        V, F = sphere_mesh(size[0])
    elif gtype == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        V, F = ellipsoid_mesh(size[:3])
    elif gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        V, F = cylinder_mesh(size[0], size[1])
    elif gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        V, F = capsule_mesh(size[0], size[1])
    elif gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = int(model.geom_dataid[gid])
        if mid < 0:
            return None
        V_raw, F = extract_mesh_asset(model, mid)
        scale = np.asarray(model.mesh_scale[mid], dtype=np.float64)
        R_mesh = quat_wxyz_to_R(model.mesh_quat[mid])
        t_mesh = np.asarray(model.mesh_pos[mid], dtype=np.float64)
        V = (R_mesh @ (V_raw * scale).T).T + t_mesh
    else:
        return None

    R_geom = quat_wxyz_to_R(model.geom_quat[gid])
    t_geom = np.asarray(model.geom_pos[gid], dtype=np.float64)
    V_body = (R_geom @ V.T).T + t_geom
    return V_body.astype(np.float32), F.astype(np.int32)


def merge_body_meshes(parts: list[tuple[np.ndarray, np.ndarray]]
                      ) -> tuple[np.ndarray, np.ndarray]:
    if not parts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
    V_all: list[np.ndarray] = []
    F_all: list[np.ndarray] = []
    offset = 0
    for V, F in parts:
        V_all.append(V)
        F_all.append(F + offset)
        offset += V.shape[0]
    return (np.concatenate(V_all, axis=0).astype(np.float32),
            np.concatenate(F_all, axis=0).astype(np.int32))


def write_ply_binary(path: Path, V: np.ndarray, F: np.ndarray) -> None:
    """Binary little-endian PLY (float32 vertex, int32 indices)."""
    V = np.ascontiguousarray(V, dtype=np.float32)
    F = np.ascontiguousarray(F, dtype=np.int32)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {V.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {F.shape[0]}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    face_dtype = np.dtype([("n", "u1"), ("idx", "<i4", (3,))])
    face_arr = np.empty(F.shape[0], dtype=face_dtype)
    face_arr["n"]   = 3
    face_arr["idx"] = F
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(V.tobytes(order="C"))
        f.write(face_arr.tobytes(order="C"))


def build_env_model(env_name: str):
    mt1 = metaworld.MT1(env_name, seed=42)
    env = mt1.train_classes[env_name](render_mode="rgb_array", width=64, height=64)
    env.seed(42)
    env.set_task(mt1.train_tasks[0])
    env.reset()
    return env, env.model


def build_body_meshes(model) -> dict[int, dict]:
    """Return {bid: {"obj_name": str, "V": (N,3) f32, "F": (M,3) i32}}.

    Only bodies whose merged mesh has > 0 faces are included.
    """
    used_names: dict[str, int] = {}
    out: dict[int, dict] = {}
    for bid in range(model.nbody):
        raw_name = model.body(bid).name or f"body_{bid}"
        obj_name = safe_name(raw_name) or f"body_{bid}"
        if obj_name in used_names:
            used_names[obj_name] += 1
            obj_name = f"{obj_name}_{used_names[obj_name]}"
        else:
            used_names[obj_name] = 0

        a = int(model.body_geomadr[bid])
        n = int(model.body_geomnum[bid])
        parts: list[tuple[np.ndarray, np.ndarray]] = []
        for k in range(n):
            gid = a + k
            res = geom_to_body_mesh(model, gid)
            if res is None:
                continue
            parts.append(res)
        if not parts:
            continue
        V, F = merge_body_meshes(parts)
        if F.shape[0] == 0:
            continue
        out[bid] = {"obj_name": obj_name, "V": V, "F": F}
    return out


def collect_target_mesh(model, root_bid: int) -> tuple[np.ndarray, np.ndarray]:
    """Merge geoms of `root_bid` and all its descendants into root-body-local frame.

    Each descendant body has a pose (body_pos, body_quat) expressed in its parent's
    local frame, so the chain T_root_from_d = T_root_from_p @ ... @ T_p_from_d
    transforms its geoms back into root-body-local coords.
    """
    children: dict[int, list[int]] = {}
    for c in range(model.nbody):
        p = int(model.body_parentid[c])
        if c == p:
            continue
        children.setdefault(p, []).append(c)

    parts: list[tuple[np.ndarray, np.ndarray]] = []

    def walk(bid: int, T_root_from_b: np.ndarray) -> None:
        a = int(model.body_geomadr[bid])
        n = int(model.body_geomnum[bid])
        R_rb = T_root_from_b[:3, :3]
        t_rb = T_root_from_b[:3, 3]
        for k in range(n):
            res = geom_to_body_mesh(model, a + k)
            if res is None:
                continue
            V_b, F = res
            V_root = (R_rb @ V_b.T.astype(np.float64)).T + t_rb
            parts.append((V_root.astype(np.float32), F))
        for c in children.get(bid, []):
            R_bc = quat_wxyz_to_R(model.body_quat[c])
            t_bc = np.asarray(model.body_pos[c], dtype=np.float64)
            T_b_from_c = np.eye(4, dtype=np.float64)
            T_b_from_c[:3, :3] = R_bc
            T_b_from_c[:3, 3] = t_bc
            walk(c, T_root_from_b @ T_b_from_c)

    walk(root_bid, np.eye(4, dtype=np.float64))
    return merge_body_meshes(parts)


def build_target_meshes(model, target_names: set[str]) -> dict[int, dict]:
    """Like build_body_meshes but restricted to `target_names`, and pulling in
    descendant-body geoms transformed into the target body's local frame.
    """
    out: dict[int, dict] = {}
    for bid in range(model.nbody):
        raw_name = model.body(bid).name or ""
        obj_name = safe_name(raw_name)
        if obj_name not in target_names:
            continue
        V, F = collect_target_mesh(model, bid)
        if F.shape[0] == 0:
            continue
        out[bid] = {"obj_name": obj_name, "V": V, "F": F}
    return out


def process_env(env_root: Path, env_name: str | None = None,
                overwrite: bool = False) -> str:
    env_root = env_root.resolve()
    if env_name is None:
        env_name = env_root.name

    cam_data_dir = env_root / "camera_data"
    traj_dirs = sorted(p for p in cam_data_dir.glob("traj_*") if p.is_dir())
    if not traj_dirs:
        return f"{env_name}: no traj dirs"

    target_names: set[str] = set()
    for traj_dir in traj_dirs:
        meta_path = traj_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        for n in meta.get("target_object", {}).get("body_names", []):
            target_names.add(safe_name(n))
    if not target_names:
        return f"{env_name}: no target_object.body_names found in any meta.json"

    env, model = build_env_model(env_name)
    try:
        body_meshes = build_target_meshes(model, target_names)
    finally:
        try:
            env.close()
        except Exception:
            pass

    n_traj_updated = 0
    for traj_dir in traj_dirs:
        meta_path = traj_dir / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        traj_targets = {
            safe_name(n)
            for n in meta.get("target_object", {}).get("body_names", [])
        }
        if not traj_targets:
            continue

        files_map: dict[str, str] = {}
        seg_id_map: dict[str, int] = {}
        n_verts: dict[str, int] = {}
        n_faces: dict[str, int] = {}
        for bid, entry in body_meshes.items():
            obj = entry["obj_name"]
            if obj not in traj_targets:
                continue
            out_ply = traj_dir / f"mesh_{obj}.ply"
            if not (out_ply.exists() and not overwrite):
                write_ply_binary(out_ply, entry["V"], entry["F"])
            files_map[obj] = out_ply.name
            seg_id_map[obj] = int(bid)
            n_verts[obj] = int(entry["V"].shape[0])
            n_faces[obj] = int(entry["F"].shape[0])

        meta["meshes"] = {
            "format": "binary_little_endian PLY; vertices are in body-local "
                      "frame (p_cam_h = T_body2cam[t] @ [v_body; 1]).",
            "files":   files_map,
            "seg_ids": seg_id_map,
            "num_vertices": n_verts,
            "num_faces":    n_faces,
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        n_traj_updated += 1

    return (f"{env_name}: target_bodies={sorted(target_names)}  "
            f"meshes_built={len(body_meshes)}  trajs_updated={n_traj_updated}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--env-dir",  type=Path)
    g.add_argument("--root",     type=Path)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.env_dir is not None:
        env_dirs = [args.env_dir]
    else:
        env_dirs = [p for p in sorted(args.root.iterdir())
                    if p.is_dir() and (p / "camera_data").is_dir()]

    if not env_dirs:
        print("[extract_meshes] no env dirs found", file=sys.stderr)
        sys.exit(1)

    print(f"[extract_meshes] {len(env_dirs)} env(s)")
    n_ok = n_err = 0
    for env_root in env_dirs:
        try:
            status = process_env(env_root, env_name=env_root.name,
                                 overwrite=args.overwrite)
            print(f"  [OK ] {status}")
            n_ok += 1
        except Exception as e:
            print(f"  [ERR] {env_root.name}: {e}")
            traceback.print_exc()
            n_err += 1

    print(f"[extract_meshes] ok={n_ok}  err={n_err}")
    if n_err:
        sys.exit(2)


if __name__ == "__main__":
    main()
