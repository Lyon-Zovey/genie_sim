# SceneFlow data recorder for GenieSim (Isaac Sim backend).
#
# Mirrors what rbs_scripts/replay_record_trajectories.py does for the SAPIEN/
# MuJoCo backend: every simulation frame the caller pushes RGB, depth, seg,
# camera pose and object poses into this recorder; at trajectory end flush()
# writes the camera_data/traj_N/ directory that the rest of the sceneflow
# pipeline (convert_camera_depths → flow_compress → build_mikasa_format) expects.
#
# Coordinate conventions
# ----------------------
# Isaac Sim Camera prims (USD Camera schema) follow the OpenGL convention:
# the camera's local +Z axis points BACKWARD (out of the scene) and the
# camera looks along its local -Z. So `cam_xform.get_world_pose()` returns
# an OpenGL cam-to-world transform directly. (The previous notes in this
# file calling it "USD/RDF" were wrong — RDF is what Replicator's depth
# uses for distance-along-+Z, not what the Camera prim's pose returns.)
#
# The MIKASA / sceneflow pipeline downstream of us is entirely OpenCV:
# anchor.npy uses z>0 forward, depth unprojection uses z_cam = +z_m,
# cam2world_cv.npy is the consumed file. To keep one convention end-to-end
# we convert GL → CV at the recorder source and never write GL again,
# except as a derived helper file (cam2world_gl.npy) for legacy consumers.
#
# FLIP4 = diag(1, -1, -1, 1) is the GL ↔ CV converter (it is its own
# inverse). For a cam-to-world matrix:
#   T_cv = T_gl @ FLIP4
# All on-disk products written by this recorder are in OpenCV:
#   cam_poses.npy                          OpenCV cam-to-world (T,4,4)
#   id_poses/<sid>/camera_position         OpenCV body→cam translation
#   id_poses/<sid>/camera_quaternion       OpenCV body→cam rotation (wxyz)
# id_poses/<sid>/position and quaternion are world-frame and convention-free.

import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

# diag(1,-1,-1,1) — flips Y and Z axes to convert OpenGL ↔ OpenCV camera frame
# (its own inverse: applying it twice is the identity).
FLIP4 = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def _mat3_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(R.astype(np.float64)).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)


def _save_rgb_video(frames: list, path: str, fps: float) -> None:
    frames_arr = np.stack(frames, axis=0)
    T, H, W, _ = frames_arr.shape

    def _encode(codec: str) -> bool:
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
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        p.communicate(input=frames_arr.tobytes())
        return p.returncode == 0

    if not _encode("h264_nvenc"):
        if not _encode("libx264"):
            raise RuntimeError(f"ffmpeg encode failed for {path}")


# Stable color palette: sid 0 always black (background); sids 1..N use a
# tab20-like palette so two consecutive sids are always visually distinct.
_PALETTE_RGB = np.array([
    [  0,   0,   0],   # 0 background
    [ 31, 119, 180],   # 1
    [255, 127,  14],   # 2
    [ 44, 160,  44],   # 3
    [214,  39,  40],   # 4
    [148, 103, 189],   # 5
    [140,  86,  75],   # 6
    [227, 119, 194],   # 7
    [127, 127, 127],   # 8
    [188, 189,  34],   # 9
    [ 23, 190, 207],   # 10
    [174, 199, 232],   # 11
    [255, 152, 150],   # 12
    [152, 223, 138],   # 13
    [197, 176, 213],   # 14
    [196, 156, 148],   # 15
    [247, 182, 210],   # 16
    [199, 199, 199],   # 17
    [219, 219, 141],   # 18
    [158, 218, 229],   # 19
], dtype=np.uint8)


def _seg_color_lut(max_sid: int) -> np.ndarray:
    """Return (max_sid+1, 3) uint8 LUT mapping sid -> RGB."""
    n = max_sid + 1
    if n <= len(_PALETTE_RGB):
        return _PALETTE_RGB[:n].copy()
    extra = n - len(_PALETTE_RGB)
    rng = np.random.default_rng(0xC0FFEE)
    extra_rgb = rng.integers(40, 230, size=(extra, 3), dtype=np.uint8)
    return np.concatenate([_PALETTE_RGB, extra_rgb], axis=0)


def render_seg_overlay_video(
    rgb_frames: np.ndarray,
    seg_frames: np.ndarray,
    out_path: str,
    sid_to_name: dict,
    fps: float = 16.0,
    alpha: float = 0.55,
) -> None:
    """Write an MP4 with RGB overlaid by seg colors and per-sid text labels.

    rgb_frames : (T,H,W,3) uint8
    seg_frames : (T,H,W)   int(any width) — sids; 0 means background
    sid_to_name: {int sid: str leaf_name}, sid=0 is auto-labelled "background"
    """
    try:
        import cv2  # only used for centroid placement + putText
    except ImportError:
        cv2 = None  # text labels disabled but overlay still works

    rgb_frames = np.asarray(rgb_frames)
    seg_frames = np.asarray(seg_frames).astype(np.int32)
    assert rgb_frames.shape[:3] == seg_frames.shape[:3], (
        f"rgb {rgb_frames.shape} / seg {seg_frames.shape} shape mismatch"
    )
    T, H, W = seg_frames.shape

    max_sid = int(max(seg_frames.max(initial=0), max(sid_to_name.keys() or [0])))
    lut = _seg_color_lut(max_sid)
    seg_clamped = np.clip(seg_frames, 0, max_sid)
    seg_rgb = lut[seg_clamped]                          # (T,H,W,3)

    # Alpha blend (skip pixels where seg==0 — let RGB show through unchanged
    # so the background scene stays visible)
    rgb_f = rgb_frames.astype(np.float32)
    seg_f = seg_rgb.astype(np.float32)
    fg_mask = (seg_frames > 0)[..., None].astype(np.float32)
    blended = rgb_f * (1.0 - alpha * fg_mask) + seg_f * (alpha * fg_mask)
    blended = blended.clip(0, 255).astype(np.uint8)

    # Per-frame: write each visible sid's leaf-name once at its centroid
    out_frames = []
    for t in range(T):
        img = blended[t].copy()
        seg_t = seg_frames[t]
        unique_sids = np.unique(seg_t)
        if cv2 is not None:
            for sid in unique_sids:
                if sid <= 0:
                    continue
                ys, xs = np.where(seg_t == sid)
                if ys.size < 30:                # ignore noise specks
                    continue
                cy, cx = int(ys.mean()), int(xs.mean())
                label = sid_to_name.get(int(sid), f"sid={sid}")
                text = f"{sid}:{label}"
                cv2.putText(img, text, (cx - 4, cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, text, (cx - 4, cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Top-left legend: list every tracked sid (visible or not), green
            # if present in this frame, red if missing — instantly shows the
            # "夹爪没 mask" case.
            present = set(unique_sids.tolist())
            for i, (sid, name) in enumerate(sorted(sid_to_name.items())):
                color = (0, 200, 0) if sid in present else (220, 60, 60)
                cv2.putText(img, f"{sid}:{name}", (8, 16 + i * 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, f"{sid}:{name}", (8, 16 + i * 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        out_frames.append(img)

    _save_rgb_video(out_frames, out_path, fps)


class SceneFlowRecorder:
    """Per-trajectory frame buffer + disk writer.

    Usage (inside command_controller.py)::

        # on start_recording
        recorder = SceneFlowRecorder(
            output_root="recording_data/<task>/camera_data",
            traj_idx=self.loop_count,
            camera_prim_list=[...],
            fps=self.fps,
            task_id=self.task_name,
        )
        recorder.init_annotators()   # called once after Isaac Sim is ready

        # each rendered frame
        recorder.capture_frame(object_poses)  # reads annotators + world poses

        # on task success
        recorder.flush()
    """

    def __init__(
        self,
        output_root: str,
        traj_idx: int,
        camera_prim_list: list,
        fps: float,
        task_id: str = "",
        object_prim_paths: list = None,
        camera_resolutions: dict = None,
        target_prim_paths: list = None,
        prim_to_seg_id: dict = None,
    ):
        self.traj_key = f"traj_{traj_idx}"
        self.traj_dir = Path(output_root) / self.traj_key
        self.traj_dir.mkdir(parents=True, exist_ok=True)

        self.fps = fps
        self.task_id = task_id
        self.camera_prim_list = camera_prim_list
        self.object_prim_paths = object_prim_paths or []
        # {prim_path: (width, height)} — used in init_annotators to set resolution
        self.camera_resolutions = camera_resolutions or {}

        # Subset of object_prim_paths that are the *target* objects for MIKASA outputs.
        # If None, all object_prim_paths are treated as targets.
        self.target_prim_paths: list = target_prim_paths if target_prim_paths is not None else self.object_prim_paths

        # per-camera buffers  {prim_path: {field: list}}
        self._bufs: dict[str, dict] = {}
        for prim in camera_prim_list:
            self._bufs[prim] = {
                "rgb": [],
                "depth": [],
                "seg": [],
                "cam_poses": [],
                "cam_K": None,
            }

        # omni.replicator annotators, initialised once in init_annotators()
        # {prim_path: {"rgb": ann, "depth": ann, "seg": ann, "cam_K": np.ndarray}}
        self._annotators: dict[str, dict] = {}

        # per-object world-frame pose buffers  {prim_path: {field: list}}
        self._obj_bufs: dict[str, dict] = {
            p: {"position": [], "quaternion": []}
            for p in self.object_prim_paths
        }

        # seg_id mapping: prim_path → integer ID (populated by caller or inferred at flush)
        self._prim_to_seg_id: dict[str, int] = dict(prim_to_seg_id) if prim_to_seg_id else {}

        # One-shot diagnostic: dump idToLabels + remap-LUT coverage on the
        # first captured frame so we can see, post-mortem, which prims got a
        # working SemanticsAPI and which silently fell back to "robot"/"".
        self._dumped_seg_diag: bool = False

    # ------------------------------------------------------------------
    # One-time annotator setup (call once after Isaac Sim stage is ready)
    # ------------------------------------------------------------------

    # Target capture resolution for all cameras — matches rgb.mp4 output size.
    # Using the native sensor resolution would blow up frame-buffer memory
    # (e.g. 1920×1536 × 5 cams × ~350 frames × RGB+depth+seg ≈ 30+ GB).
    CAPTURE_WIDTH  = 832
    CAPTURE_HEIGHT = 480

    def init_annotators(self) -> None:
        """Create render-product annotators for each camera — call exactly once."""
        import omni.replicator.core as rep
        from isaacsim.sensors.camera import Camera

        for cam_prim in self.camera_prim_list:
            cam = Camera(prim_path=cam_prim, resolution=[self.CAPTURE_WIDTH, self.CAPTURE_HEIGHT])
            cam.initialize()

            focal_length        = cam.get_focal_length()
            horizontal_aperture = cam.get_horizontal_aperture()
            fx  = self.CAPTURE_WIDTH  * focal_length / horizontal_aperture
            fy  = self.CAPTURE_HEIGHT * focal_length / (
                horizontal_aperture * self.CAPTURE_HEIGHT / self.CAPTURE_WIDTH
            )
            cam_K = np.array(
                [[fx, 0., self.CAPTURE_WIDTH * 0.5],
                 [0., fy, self.CAPTURE_HEIGHT * 0.5],
                 [0., 0., 1.]],
                dtype=np.float32,
            )

            rp = cam._render_product_path

            ann_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            ann_rgb.attach([rp])

            ann_depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            ann_depth.attach([rp])

            ann_seg = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
            ann_seg.attach([rp])

            self._annotators[cam_prim] = {
                "rgb":   ann_rgb,
                "depth": ann_depth,
                "seg":   ann_seg,
                "cam_K": cam_K,
            }
            # Store intrinsics into the frame buffer now
            self._bufs[cam_prim]["cam_K"] = cam_K

    # ------------------------------------------------------------------
    # Per-frame API
    # ------------------------------------------------------------------

    def capture_frame(self, object_poses: dict = None) -> None:
        """Read annotators + world poses for one frame and buffer the results.

        Parameters
        ----------
        object_poses : dict {prim_path: (4,4) float32 world-frame pose}
                       Pass every tracked object's current world pose.
        """
        from scipy.spatial.transform import Rotation as _R

        for cam_prim, anns in self._annotators.items():
            buf = self._bufs[cam_prim]

            # ── RGB ──────────────────────────────────────────────────
            rgba = anns["rgb"].get_data()
            if rgba is None or rgba.size == 0:
                continue
            rgb = rgba[:, :, :3].astype(np.uint8)

            # ── Depth ────────────────────────────────────────────────
            depth = anns["depth"].get_data()
            if depth is None or depth.size == 0:
                continue
            depth = depth.astype(np.float32)

            # ── Segmentation ─────────────────────────────────────────
            seg_data = anns["seg"].get_data()
            H, W = depth.shape
            if seg_data is not None and isinstance(seg_data, dict):
                id_img = seg_data.get("data")
                if id_img is not None and id_img.size > 0:
                    if id_img.ndim == 3 and id_img.shape[2] >= 4:
                        id_img = id_img.astype(np.uint32)
                        seg = (
                            id_img[:, :, 0]
                            | (id_img[:, :, 1] << 8)
                            | (id_img[:, :, 2] << 16)
                            | (id_img[:, :, 3] << 24)
                        ).astype(np.int32)
                    else:
                        seg = id_img.astype(np.int32)
                    # Remap Isaac Sim instance IDs → our sequential seg IDs using
                    # the idToLabels dict provided by the annotator.  Without this,
                    # Isaac Sim's own ID=1 (typically background/table/robot) would
                    # collide with our prim_to_seg_id[first_object]=1, causing the
                    # mask to cover most of the frame.
                    id_to_labels = seg_data.get("info", {}).get("idToLabels", {})
                    if not self._dumped_seg_diag:
                        self._dump_seg_diag(id_to_labels, seg)
                        self._dumped_seg_diag = True
                    if id_to_labels and self._prim_to_seg_id:
                        seg = self._remap_seg(seg, id_to_labels)
                else:
                    seg = np.zeros((H, W), dtype=np.int32)
            else:
                seg = np.zeros((H, W), dtype=np.int32)
            seg[seg < 0] = 0

            # ── Camera pose (USD cam-to-world) ────────────────────────
            from isaacsim.core.prims import SingleXFormPrim as _XFormPrim
            cam_xform = _XFormPrim(prim_path=cam_prim)
            cam_pos, cam_quat_wxyz = cam_xform.get_world_pose()
            cam_rot = _R.from_quat([
                cam_quat_wxyz[1], cam_quat_wxyz[2],
                cam_quat_wxyz[3], cam_quat_wxyz[0],
            ]).as_matrix().astype(np.float32)
            cam_T_world = np.eye(4, dtype=np.float32)
            cam_T_world[:3, :3] = cam_rot
            cam_T_world[:3, 3]  = cam_pos.astype(np.float32)

            buf["rgb"].append(rgb)
            buf["depth"].append(depth)
            buf["seg"].append(seg)
            buf["cam_poses"].append(cam_T_world)

        # ── Object world poses (camera-independent, record once) ──────
        if object_poses:
            # Use first camera's frame count as reference
            ref_len = len(next(iter(self._bufs.values()))["rgb"])
            for prim_path, pose_T in object_poses.items():
                if prim_path not in self._obj_bufs:
                    self._obj_bufs[prim_path] = {"position": [], "quaternion": []}
                ob = self._obj_bufs[prim_path]
                if len(ob["position"]) < ref_len:
                    T = np.asarray(pose_T, dtype=np.float32)
                    ob["position"].append(T[:3, 3])
                    ob["quaternion"].append(_mat3_to_quat_wxyz(T[:3, :3]))

    def append_frame(
        self,
        camera_prim: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        seg: np.ndarray,
        cam_T_world: np.ndarray,
        cam_K: np.ndarray,
        object_poses: dict = None,
    ) -> None:
        """Legacy per-camera append — kept for compatibility."""
        if camera_prim not in self._bufs:
            return
        buf = self._bufs[camera_prim]
        buf["rgb"].append(rgb.astype(np.uint8))
        buf["depth"].append(depth.astype(np.float32))
        seg_clean = seg.astype(np.int32)
        seg_clean[seg_clean < 0] = 0
        buf["seg"].append(seg_clean)
        buf["cam_poses"].append(cam_T_world.astype(np.float32))
        if buf["cam_K"] is None:
            buf["cam_K"] = cam_K.astype(np.float32)
        if object_poses:
            for prim_path, pose in object_poses.items():
                if prim_path not in self._obj_bufs:
                    self._obj_bufs[prim_path] = {"position": [], "quaternion": []}
                ob = self._obj_bufs[prim_path]
                T = np.asarray(pose, dtype=np.float32)
                if len(ob["position"]) < len(buf["cam_poses"]):
                    ob["position"].append(T[:3, 3])
                    ob["quaternion"].append(_mat3_to_quat_wxyz(T[:3, :3]))

    # ------------------------------------------------------------------
    # Flush to disk
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Write all buffered data to traj_dir.  Idempotent if called once."""
        multi_cam = len(self._bufs) > 1
        for cam_prim, buf in self._bufs.items():
            if not buf["rgb"]:
                continue
            cam_name = cam_prim.split("/")[-1]
            if multi_cam:
                out_dir = self.traj_dir / cam_name
                out_dir.mkdir(exist_ok=True)
            else:
                out_dir = self.traj_dir

            self._write_camera_data(buf, out_dir, cam_name)

        self._write_traj_h5()
        self._write_traj_task_json()

        # For multi-camera layout, copy h5 and traj_task.json into every
        # camera sub-directory so that downstream scripts (convert_camera_depths,
        # build_mikasa_format) can find them regardless of which sub-dir they
        # walk into.
        if multi_cam:
            import shutil as _shutil
            h5_src   = self.traj_dir / f"{self.traj_key}.h5"
            json_src = self.traj_dir / "traj_task.json"
            for cam_prim, buf in self._bufs.items():
                if not buf["rgb"]:
                    continue
                cam_name = cam_prim.split("/")[-1]
                cam_dir  = self.traj_dir / cam_name
                if h5_src.exists():
                    _shutil.copy2(str(h5_src), str(cam_dir / h5_src.name))
                if json_src.exists():
                    _shutil.copy2(str(json_src), str(cam_dir / "traj_task.json"))

        # ── Step-5 MIKASA outputs (while Isaac Sim is still alive for mesh USD) ──
        self._build_mikasa_outputs()

        # Release frame buffers immediately to free memory
        for buf in self._bufs.values():
            buf["rgb"].clear()
            buf["depth"].clear()
            buf["seg"].clear()
            buf["cam_poses"].clear()
        for ob in self._obj_bufs.values():
            ob["position"].clear()
            ob["quaternion"].clear()

    # ------------------------------------------------------------------
    # MIKASA Step-5 builder
    # ------------------------------------------------------------------

    def _build_mikasa_outputs(self) -> None:
        """Call MikasaBuilder while the USD stage is still live (for mesh extraction).

        The seg array is passed in-memory from the first camera's buffer so that
        seg.b2nd does not need to be read back from disk.
        """
        try:
            import os as _os
            import sys as _sys
            _here = _os.path.dirname(_os.path.abspath(__file__))
            if _here not in _sys.path:
                _sys.path.insert(0, _here)
            from mikasa_builder import MikasaBuilder
        except ImportError as e:
            import logging
            logging.getLogger(__name__).warning(
                f"MikasaBuilder not available — skipping Step-5 outputs: {e}"
            )
            return

        # Build actors list from _obj_bufs (same as traj_task.json)
        actors = []
        for idx, prim_path in enumerate(self._obj_bufs.keys()):
            prim_name = prim_path.split("/")[-1]
            seg_id = self._prim_to_seg_id.get(prim_path, idx + 1)
            actors.append({"seg_id": seg_id, "name": f"body:{prim_name}"})

        # Target seg IDs — use explicitly set target_prim_paths if provided
        if self.target_prim_paths is not self.object_prim_paths:
            target_seg_ids = [
                self._prim_to_seg_id.get(p, i + 1)
                for i, p in enumerate(self.object_prim_paths)
                if p in set(self.target_prim_paths)
            ] or None
        else:
            target_seg_ids = None  # MikasaBuilder uses all actors

        # {seg_id: prim_path} for live USD mesh extraction
        seg_id_to_prim = {
            self._prim_to_seg_id.get(p, i + 1): p
            for i, p in enumerate(self.object_prim_paths)
        }

        # Grab seg array from first camera buffer (still in memory at this point)
        seg_array = None
        for buf in self._bufs.values():
            if buf["seg"]:
                seg_array = np.stack(buf["seg"], axis=0)
                break

        try:
            builder = MikasaBuilder(
                traj_dir=self.traj_dir,
                actors=actors,
                target_seg_ids=target_seg_ids,
                object_prim_paths=seg_id_to_prim,
                seg_array=seg_array,
                overwrite=False,
            )
            builder.run()
        except Exception as _e:
            import logging, traceback
            logging.getLogger(__name__).error(
                f"MikasaBuilder failed for {self.traj_dir}: {_e}\n{traceback.format_exc()}"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remap_seg(self, seg: np.ndarray, id_to_labels: dict) -> np.ndarray:
        """Remap Isaac Sim instance pixel IDs to our sequential prim_to_seg_id values.

        Isaac Sim assigns arbitrary integer IDs to prims.  id_to_labels is the
        {str(isaac_id): {"class": label_name}} dict from the annotator info.
        We build a lookup table: isaac_id → our_seq_id (0 = background/unknown).
        """
        # Build label_name → our_seq_id from _prim_to_seg_id
        # prim paths are like "/World/Objects/geniesim_2025_billiards_blue"
        # label_name registered in Isaac Sim is the last segment of the prim path
        label_to_seq: dict[str, int] = {}
        for prim_path, seq_id in self._prim_to_seg_id.items():
            label = prim_path.split("/")[-1]
            label_to_seq[label] = seq_id

        # Build LUT: isaac_id → seq_id
        # Replicator may concatenate multiple ancestor classes with ',' (e.g.
        # an ancestor /G1 carrying class="robot" plus a descendant link
        # carrying class="gripper_l_center_link" → "gripper_l_center_link,robot").
        # We split on ',' and pick the most specific tracked leaf-name match
        # (the one in label_to_seq); anything else collapses to background.
        max_isaac_id = max((int(k) for k in id_to_labels), default=0)
        lut = np.zeros(max_isaac_id + 2, dtype=np.int32)  # default 0 = background
        for isaac_id_str, entry in id_to_labels.items():
            try:
                isaac_id = int(isaac_id_str)
            except (ValueError, TypeError):
                continue
            label = entry.get("class", "") if isinstance(entry, dict) else str(entry)
            parts = [p.strip() for p in label.split(",") if p.strip()]
            seq_id = 0
            for part in parts:
                if part in label_to_seq:
                    seq_id = label_to_seq[part]
                    break
            if 0 <= isaac_id < len(lut):
                lut[isaac_id] = seq_id

        # Apply LUT — clamp out-of-range values to 0
        remapped = np.where(seg < len(lut), lut[np.clip(seg, 0, len(lut) - 1)], 0)
        return remapped.astype(np.int32)

    def _dump_seg_diag(self, id_to_labels: dict, seg: np.ndarray) -> None:
        """One-shot post-mortem: which Isaac labels did we actually receive,
        and which of our tracked prim leaf-names matched?"""
        import logging
        log = logging.getLogger(__name__)

        labels_in_frame = set()
        for k, v in (id_to_labels or {}).items():
            label = v.get("class", "") if isinstance(v, dict) else str(v)
            for part in label.split(","):
                part = part.strip()
                if part:
                    labels_in_frame.add(part)

        wanted = {p.split("/")[-1]: p for p in self._prim_to_seg_id}
        matched = sorted(set(wanted) & labels_in_frame)
        missing = sorted(set(wanted) - labels_in_frame)

        unique_ids, counts = np.unique(seg, return_counts=True)
        top = sorted(zip(unique_ids.tolist(), counts.tolist()),
                     key=lambda x: -x[1])[:8]

        log.warning(
            "[SceneFlow seg diag] idToLabels=%s",
            {k: v.get("class", v) if isinstance(v, dict) else v
             for k, v in (id_to_labels or {}).items()},
        )
        log.warning("[SceneFlow seg diag] tracked leaf-names = %s", sorted(wanted))
        log.warning("[SceneFlow seg diag] matched in frame    = %s", matched)
        log.warning("[SceneFlow seg diag] MISSING from frame  = %s", missing)
        log.warning("[SceneFlow seg diag] raw seg top-8 (id,px) = %s", top)

    def _write_camera_data(self, buf: dict, out_dir: Path, cam_name: str) -> None:
        rgb_list   = buf["rgb"]
        depth_list = buf["depth"]
        seg_list   = buf["seg"]
        pose_list  = buf["cam_poses"]
        cam_K      = buf["cam_K"]

        T = len(rgb_list)
        H, W = depth_list[0].shape

        # ── rgb.mp4 ──────────────────────────────────────────────────────
        _save_rgb_video(rgb_list, str(out_dir / "rgb.mp4"), 16.0)

        # ── depth_video.npy  (T,H,W) float16 metres, OpenCV convention ──
        # Replicator's distance_to_image_plane is positive distance along the
        # camera's +Z forward axis, which is exactly OpenCV. We store raw
        # positive metres and unproject with z_cam = +z_m in
        # convert_camera_depths.py.
        depth_arr = np.stack(depth_list, axis=0).astype(np.float16)  # (T,H,W)
        np.save(str(out_dir / "depth_video.npy"), depth_arr)

        # ── seg.npy  (T,H,W) int32 ───────────────────────────────────────
        seg_arr = np.stack(seg_list, axis=0)  # (T,H,W) int32
        np.save(str(out_dir / "seg.npy"), seg_arr)

        # ── cam_poses.npy  (T,4,4) float32 OpenCV cam-to-world ──────────
        # Isaac Sim Camera.get_world_pose() is OpenGL (cam looks down -Z).
        # We convert to OpenCV at the source so every on-disk product (this
        # file, depth, anchor, cam2world_cv, id_poses/camera_*) shares one
        # convention.  mikasa_builder.build_camera_poses consumes this file
        # as OpenCV directly.
        poses_gl = np.stack(pose_list, axis=0).astype(np.float32)  # (T,4,4)
        poses_cv = (poses_gl @ FLIP4).astype(np.float32)           # (T,4,4)
        np.save(str(out_dir / "cam_poses.npy"), poses_cv)

        # ── cam_intrinsics.npy  (3,3) float32 ───────────────────────────
        np.save(str(out_dir / "cam_intrinsics.npy"), cam_K)

        # ── camera_name.txt ──────────────────────────────────────────────
        (out_dir / "camera_name.txt").write_text(cam_name + "\n")

        # ── seg_vis.mp4 + seg_legend.json (debug aids) ───────────────────
        # seg_vis.mp4: rgb with per-sid color overlay + leaf-name labels at
        # blob centroids + a top-left legend that goes red whenever a tracked
        # sid is missing from the frame. Lets you eyeball "is the gripper
        # actually being segmented?" without writing any extra script.
        sid_to_name = {
            self._prim_to_seg_id[p]: p.split("/")[-1]
            for p in self._prim_to_seg_id
        }
        try:
            rgb_arr = np.stack(rgb_list, axis=0)
            seg_arr_full = np.stack(seg_list, axis=0)
            render_seg_overlay_video(
                rgb_arr, seg_arr_full,
                str(out_dir / "seg_vis.mp4"),
                sid_to_name, fps=16.0,
            )
        except Exception as _e:
            import logging
            logging.getLogger(__name__).warning(
                f"seg_vis.mp4 render failed (non-fatal): {_e}"
            )

        legend = {
            "sid_to_name": {str(k): v for k, v in sorted(sid_to_name.items())},
            "sid_to_prim": {
                str(self._prim_to_seg_id[p]): p for p in self._prim_to_seg_id
            },
        }
        with open(str(out_dir / "seg_legend.json"), "w", encoding="utf-8") as f:
            json.dump(legend, f, ensure_ascii=False, indent=2)

    def _write_traj_h5(self) -> None:
        """Write traj_N.h5 with id_poses/ group (MIKASA-compatible).

        camera_position / camera_quaternion are stored in OpenCV body→cam,
        matching anchor.npy / depth / cam2world_cv.npy on disk.
        """
        # Use the first camera to determine T
        first_buf = next(iter(self._bufs.values()))
        if not first_buf["rgb"]:
            return
        T = len(first_buf["rgb"])

        # buf["cam_poses"] is OpenGL cam-to-world (Isaac Camera native).
        # Convert to OpenCV once; world→cam is then derived from poses_cv so
        # that the body→cam translations / rotations stored under id_poses
        # are in OpenCV (the same convention as anchor unprojection).
        poses_gl = np.stack(first_buf["cam_poses"], axis=0).astype(np.float32)  # (T,4,4)
        poses_cv = (poses_gl @ FLIP4).astype(np.float32)                        # (T,4,4)

        h5_path = self.traj_dir / f"{self.traj_key}.h5"
        with h5py.File(str(h5_path), "w") as f:
            grp = f.create_group(self.traj_key, track_order=True)

            if not self._obj_bufs:
                return

            id_grp = grp.create_group("id_poses", track_order=True)

            for idx, (prim_path, ob) in enumerate(self._obj_bufs.items()):
                if not ob["position"]:
                    continue
                prim_name = prim_path.split("/")[-1]
                seg_id = self._prim_to_seg_id.get(prim_path, idx + 1)

                id_grp.attrs[str(seg_id)] = f"body:{prim_name}"

                sg = id_grp.create_group(str(seg_id), track_order=True)
                sg.attrs["name"]   = f"body:{prim_name}"
                sg.attrs["seg_id"] = seg_id

                pos_world  = np.array(ob["position"],   dtype=np.float32)   # (T,3)
                quat_world = np.array(ob["quaternion"], dtype=np.float32)    # (T,4) wxyz
                sg.create_dataset("position",   data=pos_world)
                sg.create_dataset("quaternion", data=quat_world)

                cam_pos_list  = []
                cam_quat_list = []
                for t in range(min(T, len(ob["position"]))):
                    R_c2w = poses_cv[t, :3, :3]
                    t_c2w = poses_cv[t, :3,  3]
                    R_w2c = R_c2w.T
                    t_w2c = -(R_w2c @ t_c2w)

                    p_cam = (R_w2c @ pos_world[t] + t_w2c).astype(np.float32)
                    R_body_world = Rotation.from_quat([
                        quat_world[t, 1], quat_world[t, 2],
                        quat_world[t, 3], quat_world[t, 0],
                    ]).as_matrix().astype(np.float32)
                    R_body_cam = (R_w2c @ R_body_world).astype(np.float32)
                    q_cam = _mat3_to_quat_wxyz(R_body_cam)

                    cam_pos_list.append(p_cam)
                    cam_quat_list.append(q_cam)

                sg.create_dataset("camera_position",   data=np.array(cam_pos_list,  dtype=np.float32))
                sg.create_dataset("camera_quaternion", data=np.array(cam_quat_list, dtype=np.float32))

    def _write_traj_task_json(self) -> None:
        actors = []
        for idx, prim_path in enumerate(self._obj_bufs.keys()):
            prim_name = prim_path.split("/")[-1]
            seg_id = self._prim_to_seg_id.get(prim_path, idx + 1)
            actors.append({"seg_id": seg_id, "name": f"body:{prim_name}"})

        traj_task = {
            "task_id":   self.task_id,
            "traj_name": self.traj_key,
            "actors":    actors,
        }
        with open(str(self.traj_dir / "traj_task.json"), "w", encoding="utf-8") as f:
            json.dump(traj_task, f, ensure_ascii=False, indent=2)
