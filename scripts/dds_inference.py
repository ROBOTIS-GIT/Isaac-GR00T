#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


# ==============================================================
# Load config.json (camera + arm)
# ==============================================================

def load_config():
    return json.load(open(CONFIG_PATH))

CFG = load_config()

CAMERA_CONFIG = CFG.get("camera_topics", {})
ARM_CONFIG = CFG.get("arm_publishers", {})

CAMERA_KEYS = list(CAMERA_CONFIG.keys())


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
        if x.ndim == 3:  # HWC
            return x[None]
        if x.ndim == 4:  # BHWC
            return x
        print("[ERROR] Invalid ndim:", x.ndim)
        return None

    data = {}

    # Camera images
    for key, img in imgs.items():
        img4d = to_4d(img)
        if img4d is not None:
            data[f"video.{key}"] = img4d

    # Base state
    state_vec = np.array([
        odom["x"], odom["y"], odom["theta"],
        odom["linear_vel"], odom["angular_vel"]
    ], dtype=np.float32)
    data["state.robot"] = state_vec[None]

    # Joint states
    pos = np.array(joint["position"], dtype=np.float32)
    data["state.joints"] = pos[None]

    # GR00T arms
    left7 = pos[0:7]
    right7 = pos[7:14]

    data["state.left_arm"] = np.concatenate([left7, [0.0]]).reshape(1, -1)
    data["state.right_arm"] = np.concatenate([right7, [0.0]]).reshape(1, -1)

    return data


# ==============================================================
# Apply GR00T Action → DDS Commands (LEFT/RIGHT)
# ==============================================================

def apply_action_to_robot(action, rds):
    if not isinstance(action, dict):
        return

    left = action.get("action.left_arm")
    right = action.get("action.right_arm")

    if left is not None or right is not None:
        print("\n================ APPLY ACTION ================")
        if left is not None:
            print(" LEFT ARM :", left[0])
        if right is not None:
            print(" RIGHT ARM:", right[0])
        print("==============================================\n")

    # Publish
    if left is not None:
        rds.send_arm_trajectory("left", list(left[0]))

    if right is not None:
        rds.send_arm_trajectory("right", list(right[0]))


# ==============================================================
# Runner
# ==============================================================

class DdsGr00tInferenceRunner:

    def __init__(self, domain_id=30):
        print("[Runner] Initializing DDS SDK...")
        self.rds = RobotisDDSSDK(domain_id=domain_id)

        # Register cameras
        print("[Runner] Registering cameras...")
        for key, info in CAMERA_CONFIG.items():
            topic = info["topic"]
            msg_type = info.get("type", "CompressedImage_")
            self.rds.register_camera(key, topic, msg_type)

        # Register arm publishers
        print("[Runner] Registering arm publishers...")
        for arm, topic in ARM_CONFIG.items():
            self.rds.register_arm_publisher(arm, topic)

        print("[Runner] Loading GR00T policy...")
        self.policy = load_policy()
        self.prev = {}
        print("[Runner] Ready.")

    # fresh-check
    def _fresh(self, now, prev):
        if now is None:
            return False
        if prev is None:
            return True
        if isinstance(now, np.ndarray):
            return not np.array_equal(now, prev)
        return now != prev

    # ==========================================================
    # Loop
    # ==========================================================

    def run(self):
        print("\n==============================")
        print("  GR00T DDS Inference Runner")
        print("==============================\n")

        while True:

            imgs = self.rds.get_images(CAMERA_KEYS)

            # Missing camera logs
            missing_cam = False
            for k in CAMERA_KEYS:
                if imgs.get(k) is None:
                    print(f"[WAIT] Camera '{k}' has NO DATA.")
                    missing_cam = True
            if missing_cam:
                time.sleep(0.02)
                continue

            # Camera freshness
            cam_fresh = True
            for k in CAMERA_KEYS:
                if not self._fresh(imgs.get(k), self.prev.get(k)):
                    cam_fresh = False
            if not cam_fresh:
                print("[WAIT] Camera data stale.")
                time.sleep(0.02)
                continue

            # Odometry & joint
            odom = self.rds.get_odometry()
            joint = self.rds.get_joint_state()

            if odom is None:
                print("[WAIT] NO odometry data.")
            if joint is None:
                print("[WAIT] NO joint_state data.")
            if odom is None or joint is None:
                time.sleep(0.02)
                continue

            # Freshness check
            if not self._fresh(odom, self.prev.get("odom")):
                print("[WAIT] Odometry stale.")
                time.sleep(0.02)
                continue

            if not self._fresh(joint, self.prev.get("joint")):
                print("[WAIT] Joint state stale.")
                time.sleep(0.02)
                continue

            # Prepare GR00T input
            data = build_gr00t_input(imgs, odom, joint)

            # Inference
            with torch.no_grad():
                action = self.policy.get_action(data)

            apply_action_to_robot(action, self.rds)

            # Save previous
            for k in CAMERA_KEYS:
                self.prev[k] = imgs.get(k)
            self.prev["odom"] = odom
            self.prev["joint"] = joint


# ==============================================================
# Entry
# ==============================================================

def main():
    runner = DdsGr00tInferenceRunner(domain_id=30)
    runner.run()


if __name__ == "__main__":
    main()
