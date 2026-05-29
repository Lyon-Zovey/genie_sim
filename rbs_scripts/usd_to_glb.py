#!/usr/bin/env python3
"""
Convert rigid-body object USD assets to GLB + metadata JSON.

Per-object output layout:
  <output_dir>/<object_id>/
    model.glb          – geometry + diffuse texture URI
    metadata.json      – physics params, bbox, up_axis, place/grasp interaction data

Grasp data (grasp_pose.pkl) is stored in metadata.json as:
  grasp_poses: list of {"matrix": [[4x4]], "gripper_width": float}
  (top-N by gripper width, default 50; use --grasp-limit 0 for all)

interaction.json place points are preserved verbatim under:
  place_points: { "active": {...}, "passive": {...} }
"""

import argparse
import ast
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

try:
    from pxr import Usd, UsdGeom, UsdShade, Sdf
except ImportError:
    sys.exit("pxr not found – run: pip install usd-core")

try:
    import pygltflib
    from pygltflib import (
        GLTF2, Scene, Node, Mesh as GMesh, Primitive, Buffer,
        BufferView, Accessor, Material as GMaterial, Texture, Image,
        Sampler, TextureInfo, PbrMetallicRoughness,
        ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER,
        FLOAT, UNSIGNED_INT,
        SCALAR, VEC2, VEC3,
    )
except ImportError:
    sys.exit("pygltflib not found – run: pip install pygltflib")


ASSETS_ROOT = Path("/home/sheng/genie_sim/source/geniesim/assets")
INTERACTION_DIR = ASSETS_ROOT / "interaction"


# ─── item.py parser ───────────────────────────────────────────────────────────

def parse_item_py(path: str) -> dict:
    """Safely eval item.py (which is a bare dict literal at module level)."""
    try:
        src = open(path).read()
        # item.py contains a single dict expression
        tree = ast.parse(src, mode="eval")
        return ast.literal_eval(tree)
    except Exception:
        return {}


# ─── interaction data ─────────────────────────────────────────────────────────

def load_interaction(object_id: str, grasp_limit: int) -> dict:
    """
    Load interaction.json and grasp_pose.pkl for object_id.
    Returns dict with keys: place_points, grasp_poses (may be empty).
    """
    result = {"place_points": {}, "grasp_poses": []}
    idir = INTERACTION_DIR / object_id
    if not idir.exists():
        return result

    # place points
    interaction_json = idir / "interaction.json"
    if interaction_json.exists():
        try:
            raw = json.loads(interaction_json.read_text())
            ia = raw.get("interaction", {})
            place_points = {}
            for role in ("active", "passive"):
                place_data = ia.get(role, {}).get("place", {})
                if place_data:
                    place_points[role] = place_data
            result["place_points"] = place_points
        except Exception as e:
            print(f"    [WARN] interaction.json parse error: {e}")

    # grasp poses
    pkl_path = idir / "grasp_pose" / "grasp_pose.pkl"
    if pkl_path.exists():
        try:
            data = pickle.load(open(pkl_path, "rb"))
            poses = data.get("grasp_pose")   # shape (N, 4, 4)
            widths = data.get("width")        # shape (N,)
            if poses is not None and widths is not None:
                n = len(poses)
                limit = n if grasp_limit <= 0 else min(grasp_limit, n)
                # Sort descending by gripper width and take top-limit
                order = np.argsort(widths)[::-1][:limit]
                grasp_list = []
                for i in order:
                    grasp_list.append({
                        "matrix": poses[i].tolist(),
                        "gripper_width": float(widths[i]),
                    })
                result["grasp_poses"] = grasp_list
        except Exception as e:
            print(f"    [WARN] grasp_pose.pkl parse error: {e}")

    return result


# ─── USD mesh extraction ───────────────────────────────────────────────────────

def find_geo_mesh(stage: Usd.Stage):
    for p in stage.Traverse():
        if p.GetTypeName() == "Mesh":
            path_str = str(p.GetPath()).lower()
            if "visual" in path_str or "body" in path_str:
                return p
    # fallback: first mesh
    for p in stage.Traverse():
        if p.GetTypeName() == "Mesh":
            return p
    return None


def extract_mesh_data(prim) -> dict:
    mesh = UsdGeom.Mesh(prim)
    points_usd = mesh.GetPointsAttr().Get()
    indices_usd = mesh.GetFaceVertexIndicesAttr().Get()
    face_counts = mesh.GetFaceVertexCountsAttr().Get()

    if points_usd is None or indices_usd is None or face_counts is None:
        raise ValueError(f"Mesh {prim.GetPath()} missing required attributes")

    # Triangulate
    tri_indices = []
    cursor = 0
    for count in face_counts:
        verts = list(indices_usd[cursor: cursor + count])
        for i in range(1, count - 1):
            tri_indices.extend([verts[0], verts[i], verts[i + 1]])
        cursor += count

    positions = np.array([[p[0], p[1], p[2]] for p in points_usd], dtype=np.float32)
    flat_pos = positions[tri_indices]

    # UV (faceVarying st primvar)
    flat_uv = None
    api = UsdGeom.PrimvarsAPI(prim)
    for pv in api.GetPrimvars():
        if pv.GetPrimvarName() not in ("st", "uv", "UVMap"):
            continue
        raw = pv.Get()
        if raw is None:
            break
        raw_np = np.array([[u[0], u[1]] for u in raw], dtype=np.float32)
        interp = pv.GetInterpolation()
        uv_idx = pv.GetIndices()

        if interp == "faceVarying":
            if uv_idx is not None and len(uv_idx) > 0:
                tri_uv = []
                cursor = 0
                for count in face_counts:
                    verts = list(uv_idx[cursor: cursor + count])
                    for i in range(1, count - 1):
                        tri_uv.extend([verts[0], verts[i], verts[i + 1]])
                    cursor += count
                flat_uv = raw_np[tri_uv]
            else:
                tri_uv = []
                cursor = 0
                for count in face_counts:
                    idxs = list(range(cursor, cursor + count))
                    for i in range(1, count - 1):
                        tri_uv.extend([idxs[0], idxs[i], idxs[i + 1]])
                    cursor += count
                flat_uv = raw_np[tri_uv]
        elif interp == "vertex":
            flat_uv = raw_np[tri_indices]

        if flat_uv is not None:
            flat_uv[:, 1] = 1.0 - flat_uv[:, 1]   # USD V → GLTF V flip
        break

    return {
        "positions": flat_pos,
        "uvs": flat_uv,
        "indices": np.arange(len(flat_pos), dtype=np.uint32),
    }


def extract_material(stage: Usd.Stage, prim) -> dict:
    result = {"diffuse": None, "base_color": (0.5, 0.5, 0.5, 1.0)}
    usd_dir = Path(stage.GetRootLayer().realPath).parent

    binding = UsdShade.MaterialBindingAPI(prim)
    mat, _ = binding.ComputeBoundMaterial()
    if mat is None:
        return result

    for output in mat.GetSurfaceOutputs():
        for src_list in output.GetConnectedSources():
            for src in src_list:
                shader = UsdShade.Shader(src.source.GetPrim())
                if not shader:
                    continue
                inputs = {i.GetBaseName(): i for i in shader.GetInputs()}

                def tex_abs(key):
                    if key not in inputs:
                        return None
                    v = inputs[key].Get()
                    if v is None:
                        return None
                    p = Path(str(v).strip("@"))
                    if not p.is_absolute():
                        p = usd_dir / p
                    return str(p) if p.exists() else None

                result["diffuse"] = tex_abs("diffuse_texture")
                if "diffuse_color_constant" in inputs:
                    c = inputs["diffuse_color_constant"].Get()
                    if c:
                        result["base_color"] = (float(c[0]), float(c[1]), float(c[2]), 1.0)
                return result
    return result


# ─── GLB builder ──────────────────────────────────────────────────────────────

def build_glb(mesh_data: dict, mat_info: dict, output_path: str):
    gltf = GLTF2()
    gltf.scene = 0
    gltf.scenes = [Scene(nodes=[0])]
    gltf.nodes = [Node(mesh=0)]

    positions = mesh_data["positions"]
    uvs = mesh_data["uvs"]
    indices = mesh_data["indices"]

    bin_data = bytearray()

    def add_bv(data: bytes, target=None) -> int:
        offset = len(bin_data)
        pad = (4 - len(data) % 4) % 4
        bin_data.extend(data + b"\x00" * pad)
        gltf.bufferViews.append(
            BufferView(buffer=0, byteOffset=offset, byteLength=len(data), target=target))
        return len(gltf.bufferViews) - 1

    def add_acc(bv, comp_type, count, type_str, mn=None, mx=None) -> int:
        acc = Accessor(bufferView=bv, byteOffset=0, componentType=comp_type,
                       count=count, type=type_str)
        if mn is not None:
            acc.min = mn
        if mx is not None:
            acc.max = mx
        gltf.accessors.append(acc)
        return len(gltf.accessors) - 1

    idx_acc = add_acc(add_bv(indices.tobytes(), ELEMENT_ARRAY_BUFFER),
                      UNSIGNED_INT, len(indices), SCALAR)
    pos_acc = add_acc(add_bv(positions.tobytes(), ARRAY_BUFFER),
                      FLOAT, len(positions), VEC3,
                      positions.min(axis=0).tolist(),
                      positions.max(axis=0).tolist())

    attributes = {"POSITION": pos_acc}
    if uvs is not None:
        uv_acc = add_acc(add_bv(uvs.tobytes(), ARRAY_BUFFER),
                         FLOAT, len(uvs), VEC2)
        attributes["TEXCOORD_0"] = uv_acc

    gmat = GMaterial()
    gmat.pbrMetallicRoughness = PbrMetallicRoughness()
    gmat.pbrMetallicRoughness.baseColorFactor = list(mat_info["base_color"])

    if mat_info["diffuse"] and uvs is not None:
        tex_path = mat_info["diffuse"]
        ext = Path(tex_path).suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        tex_bytes = Path(tex_path).read_bytes()
        tex_bv = add_bv(tex_bytes)   # no target for image buffer views
        gltf.images.append(Image(bufferView=tex_bv, mimeType=mime))
        gltf.samplers.append(Sampler())
        gltf.textures.append(
            Texture(source=len(gltf.images) - 1, sampler=len(gltf.samplers) - 1))
        gmat.pbrMetallicRoughness.baseColorTexture = TextureInfo(
            index=len(gltf.textures) - 1)

    gltf.materials = [gmat]
    gltf.meshes = [GMesh(primitives=[Primitive(
        attributes=attributes, indices=idx_acc, material=0)])]
    gltf.buffers = [Buffer(byteLength=len(bin_data))]
    gltf.set_binary_blob(bytes(bin_data))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    gltf.save_binary(output_path)


# ─── metadata builder ─────────────────────────────────────────────────────────

def build_metadata(object_id: str, item: dict, obj_params: dict,
                   interaction: dict) -> dict:
    meta = {
        "object_id": object_id,
        "up_axis": item.get("up_axis", "y"),
        "has_joint": item.get("has_joint", False),
        "has_articulation": item.get("has_articulation", False),
        # bounding box from item.py shapes
        "bbox": next(
            (s for s in item.get("shapes", []) if s.get("name") == "bbox"), {}),
        # physical properties from object_parameters.json
        "physics": {
            "mass": obj_params.get("mass"),
            "scale": obj_params.get("scale"),
            "size": obj_params.get("size"),
            "unit": obj_params.get("unit"),
            "material_options": obj_params.get("materialOptions", []),
        },
        # interaction: place contact points + grasp poses
        "place_points": interaction["place_points"],
        "grasp_poses": interaction["grasp_poses"],
    }
    return meta


# ─── per-object conversion ────────────────────────────────────────────────────

def convert_object(usd_path: str, output_dir: str, object_id: str,
                   grasp_limit: int) -> bool:
    obj_dir = Path(usd_path).parent

    # Load side-car files
    item = parse_item_py(str(obj_dir / "item.py"))
    try:
        obj_params = json.loads((obj_dir / "object_parameters.json").read_text())
    except Exception:
        obj_params = {}

    interaction = load_interaction(object_id, grasp_limit)

    # USD → mesh
    try:
        stage = Usd.Stage.Open(usd_path)
    except Exception as e:
        print(f"  [ERROR] Cannot open USD: {e}")
        return False

    prim = find_geo_mesh(stage)
    if prim is None:
        print(f"  [SKIP] No geometry mesh found")
        return False

    try:
        mesh_data = extract_mesh_data(prim)
    except Exception as e:
        print(f"  [ERROR] Mesh extraction: {e}")
        return False

    mat_info = extract_material(stage, prim)

    # Write GLB
    out_dir = Path(output_dir) / object_id
    glb_path = str(out_dir / "model.glb")
    try:
        build_glb(mesh_data, mat_info, glb_path)
    except Exception as e:
        print(f"  [ERROR] GLB build: {e}")
        return False

    # Write metadata JSON
    meta = build_metadata(object_id, item, obj_params, interaction)
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))

    glb_kb = os.path.getsize(glb_path) / 1024
    n_grasp = len(interaction["grasp_poses"])
    n_place = sum(
        len(pts) for role in interaction["place_points"].values()
        for pts in role.values()
    )
    tex = os.path.basename(mat_info["diffuse"]) if mat_info["diffuse"] else "none"
    print(f"  [OK] {glb_path}  "
          f"({glb_kb:.0f} KB, tex={tex}, "
          f"grasp={n_grasp}, place_pts={n_place})")
    return True


# ─── candidate collection ─────────────────────────────────────────────────────

def collect_rigid_body_usds(assets_root: str, limit: int) -> list:
    results = []
    for root, dirs, files in os.walk(assets_root):
        dirs.sort()
        if "Aligned.usd" not in files:
            continue
        item_py = os.path.join(root, "item.py")
        if os.path.exists(item_py):
            item = parse_item_py(item_py)
            if item.get("has_joint", False):
                continue
        object_id = os.path.basename(root)
        results.append((os.path.join(root, "Aligned.usd"), object_id))
        if len(results) >= limit:
            break
    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert USD rigid-body assets to GLB + metadata JSON")
    parser.add_argument(
        "--assets-root",
        default=str(ASSETS_ROOT / "objects"),
        help="Root dir to search for Aligned.usd files",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/sheng/genie_sim/rbs_scripts/output_glb",
        help="Output root directory",
    )
    parser.add_argument("--limit", type=int, default=10,
                        help="Max objects to convert (0 = all)")
    parser.add_argument("--grasp-limit", type=int, default=50,
                        help="Max grasp poses per object stored in metadata "
                             "(0 = all, can be tens of thousands)")
    parser.add_argument("--usd-list", nargs="+",
                        help="Explicit Aligned.usd paths (overrides --assets-root)")
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else 10**9

    if args.usd_list:
        candidates = [(p, Path(p).parent.name) for p in args.usd_list]
    else:
        candidates = collect_rigid_body_usds(args.assets_root, limit)

    if not candidates:
        print("No candidates found.")
        return

    print(f"Converting {len(candidates)} objects → {args.output_dir}\n")
    ok = fail = 0
    for i, (usd_path, object_id) in enumerate(candidates, 1):
        print(f"[{i}/{len(candidates)}] {object_id}")
        if convert_object(usd_path, args.output_dir, object_id, args.grasp_limit):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} succeeded, {fail} failed.")


if __name__ == "__main__":
    main()
