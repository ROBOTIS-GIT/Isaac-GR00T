#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Robotis DDS Python SDK + DDS Inference Server
# High-level wrapper for DDS-based robot communication + Inference integration
#
# Author: Heewon Lee, Dongyun Kim
# License: Apache 2.0

import time
import json
import torch
import numpy as np

from robotis_dds_python.robotis_dds_sdk.robotis_dds_sdk import RobotisDDSSDK
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy


# ==============================================================
# System Config
# ==============================================================

CONFIG_PATH = "config.json"
POLICY_PATH = "/workspace/checkpoints/ROBOTIS/ffw_bg2_rev4_pick_coffee_bottle_env5_1_to_34_joint_fix_40k"
ROBOT_TYPE = "ffw_bg2"
EMBODIMENT_TAG = "new_embodiment"
DENOISING_STEPS = 4


# Load camera keys from config.json
def load_camera_keys():
    cfg = json.load(open(CONFIG_PATH))
    cam_cfg = cfg.get("camera_topics", {})
    return list(cam_cfg.keys())

CAMERA_KEYS = load_camera_keys()
# Example:
# If config.json has:
#   { "camera_topics": { "cam_head": {...}, "cam_left": {...} } }
# CAMERA_KEYS will be ["cam_head", "cam_left"]
#
# ➜ To add more cameras: just add more entries in config.json
#   No code modification needed.


# ==============================================================
# Load GR00T Policy
# ==============================================================

def load_policy():
    print("[Policy] Loading...")
    cfg = load_data_config(ROBOT_TYPE)

    policy = Gr00tPolicy(
        model_path=POLICY_PATH,
        modality_config=cfg.modality_config(),
        modality_transform=cfg.transform(),
        embodiment_tag=EMBODIMENT_TAG,
        denoising_steps=DENOISING_STEPS,
    )
    print("[Policy] Loaded.")
    return policy


# ==============================================================
# Build GR00T Input
# ==============================================================

def build_gr00t_input(imgs, odom, joint):

    def to_4d(x):
        x = np.asarray(x)
        if x.ndim == 3:
            return x[None]
        if x.ndim == 4:
            return x
        print("[ERROR] Image ndim invalid:", x.ndim)
        return None

    data = {}

    # Add all camera images automatically
    for key, img in imgs.items():
        img4d = to_4d(img)
        if img4d is not None:
            data[f"video.{key}"] = img4d

    # Robot base state
    state_vec = np.array([
        odom["x"], odom["y"], odom["theta"],
        odom["linear_vel"], odom["angular_vel"]
    ], dtype=np.float32)
    data["state.robot"] = state_vec[None]

    # Joint states
    pos = np.array(joint["position"], dtype=np.float32)
    data["state.joints"] = pos[None]

    # Arms (GR00T requires left_arm/right_arm with 8-DoF)
    left7 = pos[0:7]
    right7 = pos[7:14]

    data["state.left_arm"] = np.concatenate([left7,  [0.0]]).reshape(1, -1)
    data["state.right_arm"] = np.concatenate([right7, [0.0]]).reshape(1, -1)

    return data


# ==============================================================
# Apply GR00T Action → DDS Commands
# ==============================================================

def apply_action_to_robot(action, rds):
    if not isinstance(action, dict):
        return

    if "action.left_arm" in action and "action.right_arm" in action:
        left = action["action.left_arm"][0]
        right = action["action.right_arm"][0]
        full = np.concatenate([left, right])
        print("[APPLY] Arm:", full)
        rds.send_joint_trajectory(list(full))

    if "cmd_vel" in action:
        vx, wz = action["cmd_vel"]
        print("[APPLY] Base:", vx, wz)
        rds.send_cmd_vel(vx, wz)


# ==============================================================
# Inference Runner
# ==============================================================

class DdsGr00tInferenceRunner:

    def __init__(self, domain_id=30):
        print("[Runner] Initializing DDS SDK...")
        self.rds = RobotisDDSSDK(
            domain_id=domain_id,
            camera_config_path=CONFIG_PATH,
        )

        print("[Runner] Loading GR00T policy...")
        self.policy = load_policy()

        # Previous sensor cache (used for fresh-check)
        self.prev = {}
        print("[Runner] Ready.")

    def _fresh(self, now, prev):
        if now is None:
            return False
        if prev is None:
            return True
        if isinstance(now, np.ndarray):
            return not np.array_equal(now, prev)
        return now != prev

    def run(self):

        print("\n==============================")
        print("  GR00T DDS Inference Runner")
        print("==============================\n")

        while True:

            # 1) Read all camera images defined in config.json
            imgs = self.rds.get_images(CAMERA_KEYS)

            # Wait for all cameras to receive fresh data
            cam_fresh = True
            for key in CAMERA_KEYS:
                if not self._fresh(imgs.get(key), self.prev.get(key)):
                    cam_fresh = False
                    break
            if not cam_fresh:
                print("[Runner] Waiting: camera not updating")
                time.sleep(0.1)
                continue

            # 2) Read odometry & joint states
            odom = self.rds.get_odometry()
            joint = self.rds.get_joint_state()

            if odom is None or joint is None:
                print("[Runner] Waiting: odom/joint missing")
                time.sleep(0.1)
                continue

            if not self._fresh(odom, self.prev.get("odom")):
                print("[Runner] Waiting: odometry stale")
                time.sleep(0.1)
                continue

            if not self._fresh(joint, self.prev.get("joint")):
                print("[Runner] Waiting: joint stale")
                time.sleep(0.1)
                continue

            # 3) Build GR00T input
            data = build_gr00t_input(imgs, odom, joint)
            if data is None:
                print("[Runner] GR00T input error")
                time.sleep(0.05)
                continue

            # 4) Inference
            with torch.no_grad():
                action = self.policy.get_action(data)

            # 5) Apply action to robot
            apply_action_to_robot(action, self.rds)

            # 6) Save previous sensor values
            for key in CAMERA_KEYS:
                self.prev[key] = imgs.get(key)

            self.prev["odom"] = odom
            self.prev["joint"] = joint


# ==============================================================
# Entry Point
# ==============================================================

def main():
    runner = DdsGr00tInferenceRunner(domain_id=30)
    runner.run()


if __name__ == "__main__":
    main()
