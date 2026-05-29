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
# Isaac Sim cameras use a USD / "RDF" (right-down-forward) convention where
# +Z points into the scene.  The sceneflow pipeline expects OpenGL / "RUB"
# (right-up-backward): +X right, +Y up, -Z forward (cam looks along -Z).
#
# FLIP4 = diag(1, -1, -1, 1)  converts USD cam-to-world → OpenGL cam-to-world:
#   T_gl = T_usd @ FLIP4
# The same flip applies to any point expressed in camera space:
#   p_gl = p_usd * [1, -1, -1]
#
# All poses stored in cam_poses.npy / id_poses are in OpenGL convention so
# that downstream scripts need zero changes.

import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

# diag(1,-1,-1,1) — flips Y and Z axes to convert USD→OpenGL camera frame
FLIP4 = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
FLIP3 = np.array([1.0, -1.0, -1.0], dtype=np.float32)


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

        # ── depth_video.npy  (T,H,W) float16 metres, OpenGL convention ──
        # Depth from Isaac Sim is positive distance along +Z (USD/RDF).
        # OpenGL convention: camera looks along -Z, so depth values are the
        # same magnitude but we don't negate the scalars — the sign change is
        # already handled by convert_camera_depths.py using z_cam = -z_m.
        # We store raw positive metres; background=0 already set above.
        depth_arr = np.stack(depth_list, axis=0).astype(np.float16)  # (T,H,W)
        np.save(str(out_dir / "depth_video.npy"), depth_arr)

        # ── seg.npy  (T,H,W) int32 ───────────────────────────────────────
        seg_arr = np.stack(seg_list, axis=0)  # (T,H,W) int32
        np.save(str(out_dir / "seg.npy"), seg_arr)

        # ── cam_poses.npy  (T,4,4) float32 OpenGL cam-to-world ──────────
        # Isaac Sim cam-to-world is in USD/RDF.  Convert to OpenGL by
        # right-multiplying with FLIP4:  T_gl = T_usd @ FLIP4
        poses_usd = np.stack(pose_list, axis=0)           # (T,4,4)
        poses_gl  = poses_usd @ FLIP4                     # (T,4,4)
        np.save(str(out_dir / "cam_poses.npy"), poses_gl.astype(np.float32))

        # ── cam_intrinsics.npy  (3,3) float32 ───────────────────────────
        np.save(str(out_dir / "cam_intrinsics.npy"), cam_K)

        # ── camera_name.txt ──────────────────────────────────────────────
        (out_dir / "camera_name.txt").write_text(cam_name + "\n")

    def _write_traj_h5(self) -> None:
        """Write traj_N.h5 with id_poses/ group (MIKASA-compatible)."""
        # Use the first camera to determine T
        first_buf = next(iter(self._bufs.values()))
        if not first_buf["rgb"]:
            return
        T = len(first_buf["rgb"])
        poses_usd = np.stack(first_buf["cam_poses"], axis=0)  # (T,4,4) USD
        # cam-to-world in OpenGL for computing world→cam transforms
        poses_gl  = (poses_usd @ FLIP4).astype(np.float32)    # (T,4,4)

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

                # camera-frame pose: p_cam_gl = R_w2c_gl @ p_world + t_w2c_gl
                # R_w2c_gl = poses_gl[:, :3, :3].T (per frame)
                cam_pos_list  = []
                cam_quat_list = []
                for t in range(min(T, len(ob["position"]))):
                    R_c2w = poses_gl[t, :3, :3]  # (3,3) OpenGL
                    t_c2w = poses_gl[t, :3,  3]  # (3,)
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
