"""
Replay recorded trajectories from a LeRobot-format dataset in Isaac Sim via ROS2.

仿真器通过 /joint_command topic (sensor_msgs/JointState) 接收关节指令。
需要先启动仿真器（enable_ros: true），再运行本脚本。

Usage (容器内):
    # 终端1: 启动仿真器
    geniesim --config source/geniesim/config/if_pick_billiards_color.yaml

    # 终端2: 跑 replay
    omni_python rbs_scripts/replay_from_dataset.py \\
        --dataset data/instruction/pick_billards_color \\
        --episode 0

    # 只看数据不连仿真:
    omni_python rbs_scripts/replay_from_dataset.py \\
        --dataset data/instruction/pick_billards_color \\
        --episode 0 --dry-run

Action layout (40-dim):
    [0]      action/left_effector/position   (left gripper 0~1)
    [1]      action/right_effector/position  (right gripper 0~1)
    [2:8]    action/end/position             (left+right EEF xyz)
    [8:16]   action/end/orientation          (left+right EEF quat)
    [16:30]  action/joint/position           (14 arm joints: 7L + 7R)
    [30:33]  action/head/position            (3 head joints)
    [33:38]  action/waist/position           (5 waist joints)
    [38:40]  action/robot/velocity
"""

import argparse
import json
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np

# 让 omni_python 能找到 rclpy 和 sensor_msgs
_ros_bridge = "/isaac-sim/exts/isaacsim.ros2.bridge/jazzy"
os.environ["LD_LIBRARY_PATH"] = _ros_bridge + "/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, _ros_bridge + "/rclpy")

sys.path.append(str(Path(__file__).resolve().parent.parent / "source"))

from geniesim.utils.name_utils import (
    G2_DUAL_ARM_JOINT_NAMES,
    G2_HEAD_JOINT_NAMES,
    G2_WAIST_JOINT_NAMES,
    OMNIPICKER_AJ_NAMES,
)

LIMIT_VAL = 0.78


def relabel_gripper(raw):
    """将数据集中 0~1 的 gripper 值转换为仿真器角度 (同 infer_post_process.py)"""
    return np.array([(1 - raw[0]) * LIMIT_VAL, (1 - raw[1]) * LIMIT_VAL])


def load_info(dataset_dir: str) -> dict:
    meta_tar = os.path.join(dataset_dir, "meta.tar.gz.000")
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(meta_tar, "r:gz") as tf:
            tf.extractall(tmp)
        with open(os.path.join(tmp, "meta", "info.json")) as f:
            return json.load(f)


def load_episode(dataset_dir: str, info: dict, episode_idx: int) -> np.ndarray:
    """返回 (T, 40) 的 action 数组"""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("请先安装: /isaac-sim/python.sh -m pip install pandas pyarrow")

    data_tar = os.path.join(dataset_dir, "data.tar.gz.000")
    chunk = episode_idx // info["chunks_size"]
    parquet_path = info["data_path"].format(
        episode_chunk=chunk, episode_index=episode_idx
    )

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(data_tar, "r:gz") as tf:
            try:
                member = tf.getmember(parquet_path)
            except KeyError:
                raise ValueError(f"Episode {episode_idx} 不存在 (找 '{parquet_path}')")
            tf.extract(member, tmp)
        import pandas as pd
        df = pd.read_parquet(os.path.join(tmp, parquet_path))

    return np.stack(df["action"].values).astype(np.float32)


def replay(dataset_dir: str, episode_idx: int, fps: int, dry_run: bool):
    print(f"[replay] 加载数据集: {dataset_dir}")
    info = load_info(dataset_dir)
    print(f"[replay] robot_type={info['robot_type']}, total_episodes={info['total_episodes']}, fps={info['fps']}")

    actions = load_episode(dataset_dir, info, episode_idx)
    print(f"[replay] Episode {episode_idx}: {len(actions)} frames")

    if dry_run:
        print("[replay] Dry-run，前5帧 arm joints:")
        for i, a in enumerate(actions[:5]):
            print(f"  frame {i}  arm={np.round(a[16:30], 3)}  gripper={np.round(a[0:2], 3)}")
        return

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    # 关节名顺序与 action 切片对应
    joint_names = (
        G2_DUAL_ARM_JOINT_NAMES   # [16:30]
        + list(OMNIPICKER_AJ_NAMES)  # gripper, relabeled from [0:2]
        + G2_HEAD_JOINT_NAMES     # [30:33]
        + G2_WAIST_JOINT_NAMES    # [33:38]
    )

    rclpy.init()
    node = Node("replay_node")
    pub = node.create_publisher(JointState, "/joint_command", 1)

    print(f"[replay] 开始以 {fps} fps 发送关节指令 ...")
    dt = 1.0 / fps

    for frame_idx, action in enumerate(actions):
        t0 = time.time()

        arm_cmd    = action[16:30].tolist()
        gripper_cmd = relabel_gripper(action[0:2]).tolist()
        head_cmd   = action[30:33].tolist()
        waist_cmd  = action[33:38].tolist()

        positions = arm_cmd + gripper_cmd + head_cmd + waist_cmd

        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.name = joint_names
        msg.position = positions
        pub.publish(msg)

        elapsed = time.time() - t0
        sleep_t = dt - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

        if frame_idx % 30 == 0:
            print(f"  frame {frame_idx:4d}/{len(actions)}", flush=True)

    print("[replay] 完成.")
    node.destroy_node()
    rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="从数据集 replay episode 到 GenieSim")
    parser.add_argument("--dataset", required=True, help="数据集路径 (含 *.tar.gz.000)")
    parser.add_argument("--episode", type=int, default=0, help="episode 序号")
    parser.add_argument("--fps", type=int, default=30, help="回放帧率")
    parser.add_argument("--dry-run", action="store_true", help="只打印数据，不连仿真")
    args = parser.parse_args()

    replay(
        dataset_dir=args.dataset,
        episode_idx=args.episode,
        fps=args.fps,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
