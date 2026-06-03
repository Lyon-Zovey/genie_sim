#!/usr/bin/env python3
"""Render seg_vis.mp4 + seg_legend.json for an existing traj_N directory.

Useful when you want to eyeball "did the gripper actually get a seg mask?"
without re-collecting data.

Usage:
    python render_seg_vis.py <traj_dir> [--fps 16] [--alpha 0.55]

Reads:  rgb.mp4  seg.npy  traj_task.json
Writes: seg_vis.mp4  seg_legend.json
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from sceneflow_recorder import render_seg_overlay_video


def _decode_rgb_mp4(path: Path, n_frames: int, h: int, w: int) -> np.ndarray:
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", str(path),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-vsync", "0",
        "pipe:1",
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    raw = np.frombuffer(p.stdout, dtype=np.uint8)
    expected = n_frames * h * w * 3
    if raw.size < expected:
        raise RuntimeError(
            f"rgb.mp4 yielded {raw.size} bytes, need >= {expected} "
            f"({n_frames}x{h}x{w}x3)"
        )
    return raw[:expected].reshape(n_frames, h, w, 3)


def _build_sid_to_name(traj_dir: Path) -> dict:
    j = json.load(open(traj_dir / "traj_task.json"))
    out = {}
    for a in j.get("actors", []):
        sid = int(a["seg_id"])
        # actors store "body:<leaf_name>"; strip the "body:" prefix
        name = a.get("name", "").split(":", 1)[-1] or f"sid{sid}"
        out[sid] = name
    return out


def _load_seg(traj_dir: Path) -> np.ndarray:
    """Load seg as (T,H,W) int32 from either seg.npy (recorder raw) or
    seg.b2nd (compressed by flow_compress / point_compress / seg_compress)."""
    npy = traj_dir / "seg.npy"
    if npy.exists():
        return np.load(npy)
    b2nd = traj_dir / "seg.b2nd"
    if b2nd.exists():
        import blosc2
        arr = blosc2.open(str(b2nd))
        return np.asarray(arr[:])
    raise FileNotFoundError(
        f"neither seg.npy nor seg.b2nd found in {traj_dir}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj_dir", type=Path)
    ap.add_argument("--fps", type=float, default=16.0)
    ap.add_argument("--alpha", type=float, default=0.55)
    args = ap.parse_args()

    td = args.traj_dir.resolve()
    if not td.is_dir():
        raise SystemExit(f"not a dir: {td}")

    seg = _load_seg(td).astype(np.int32)                             # (T,H,W)
    T, H, W = seg.shape
    rgb = _decode_rgb_mp4(td / "rgb.mp4", T, H, W)
    sid_to_name = _build_sid_to_name(td)

    out_video = td / "seg_vis.mp4"
    render_seg_overlay_video(
        rgb, seg, str(out_video), sid_to_name,
        fps=args.fps, alpha=args.alpha,
    )

    legend = {
        "sid_to_name": {str(k): v for k, v in sorted(sid_to_name.items())},
        "sid_total_pixels": {
            str(int(s)): int(c)
            for s, c in zip(*np.unique(seg, return_counts=True))
        },
        "sid_visible_frames": {
            str(int(s)): int(((seg == s).reshape(T, -1).sum(-1) > 0).sum())
            for s in sid_to_name
        },
    }
    with open(td / "seg_legend.json", "w", encoding="utf-8") as f:
        json.dump(legend, f, ensure_ascii=False, indent=2)

    print(f"wrote {out_video}")
    print(f"wrote {td / 'seg_legend.json'}")
    print("--- coverage summary ---")
    for sid, name in sorted(sid_to_name.items()):
        n_pix = int((seg == sid).sum())
        n_frm = int(((seg == sid).reshape(T, -1).sum(-1) > 0).sum())
        flag = "✅" if n_pix > 0 else "❌ (NO PIXELS)"
        print(f"  sid={sid:>2}  {name:<40s}  pixels={n_pix:>9d}  frames={n_frm}/{T}  {flag}")


if __name__ == "__main__":
    main()
