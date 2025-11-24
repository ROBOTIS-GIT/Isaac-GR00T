#!/usr/bin/env python3
#
# Copyright 2025 ROBOTIS
#

"""
DDS Sensor → GR00T Policy Inference → DDS Command Publisher
Full Pipeline Runner (minimal patch version, no SDK modifications)
"""

import time
import torch
import numpy as np

from robotis_dds_python.robotis_dds_sdk.robotis_dds_sdk import RobotisDDSSDK
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy


# ==============================================================
# Hardcoded Settings
# ==============================================================

POLICY_TYPE = "GR00T_N1_5"
POLICY_PATH = "/workspace/checkpoints/ROBOTIS/ffw_bg2_rev4_pick_coffee_bottle_env5_1_to_34_joint_fix_40k"
ROBOT_TYPE = "ffw_bg2"
EMBODIMENT_TAG = "new_embodiment"
DENOISING_STEPS = 4


# ==============================================================
# Load GR00T Policy
# ==============================================================

def load_policy():
    print("[Policy] Loading GR00T policy...")
    data_config = load_data_config(ROBOT_TYPE)

    policy = Gr00tPolicy(
        model_path=POLICY_PATH,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=EMBODIMENT_TAG,
        denoising_steps=DENOISING_STEPS,
    )
    print("[Policy] Loaded.")
    return policy


# ==============================================================
# Convert DDS Input → GR00T Input
# ==============================================================

def build_gr00t_input(rds: RobotisDDSSDK):

    def to_4d(x):
        if x is None:
            return None
        x = np.asarray(x)
        if x.ndim == 3:
            return x[None, ...]
        if x.ndim == 4:
            return x
        print(f"[ERROR] Invalid image ndim={x.ndim}")
        return None

    # Camera
    head = to_4d(rds.get_zed_left_image())
    if head is None:
        print("[ERROR] Missing cam_head → pause inference")
        return None

    left  = to_4d(rds.get_left_image())
    right = to_4d(rds.get_right_image())

    data = {"video.cam_head": head}
    if left is not None:
        data["video.cam_left"] = left
    if right is not None:
        data["video.cam_right"] = right

    # Odometry
    odom = rds.get_odometry()
    if odom is None:
        print("[ERROR] Missing odometry → pause inference")
        return None

    state_vec = np.array([
        odom["x"], odom["y"], odom["theta"],
        odom["linear_vel"], odom["angular_vel"]
    ], dtype=np.float32)
    data["state.robot"] = state_vec[None, :]

    # Joint State
    joint = rds.get_joint_state()
    if joint is None:
        print("[ERROR] Missing joint_state → pause inference")
        return None

    positions = np.array(joint["position"], dtype=np.float32)
    data["state.joints"] = positions[None, :]

    # Arms
    left7  = positions[0:7]
    right7 = positions[7:14]

    left8  = np.concatenate([left7,  np.array([0.0], dtype=np.float32)])
    right8 = np.concatenate([right7, np.array([0.0], dtype=np.float32)])

    data["state.left_arm"]  = left8[None, :]
    data["state.right_arm"] = right8[None, :]

    return data


# ==============================================================
# Apply GR00T Action → DDS Command
# ==============================================================

def apply_action_to_robot(action, rds: RobotisDDSSDK):

    if isinstance(action, dict):
        print("\n[ACTION DEBUG]")
        print(" - keys:", list(action.keys()))

        if "action.left_arm" in action:
            print(" - left_arm[0]:", action["action.left_arm"][0])
        if "action.right_arm" in action:
            print(" - right_arm[0]:", action["action.right_arm"][0])
        if "cmd_vel" in action:
            print(" - cmd_vel:", action["cmd_vel"])

    if isinstance(action, dict):
        # arm
        if "action.left_arm" in action and "action.right_arm" in action:
            left  = action["action.left_arm"][0]
            right = action["action.right_arm"][0]

            full = np.concatenate([left, right], axis=0)
            print("[APPLY] 16-DOF Arm Trajectory:", full)
            rds.send_joint_trajectory(list(full))

        # base
        if "cmd_vel" in action:
            vx, wz = action["cmd_vel"]
            rds.send_cmd_vel(vx, wz)

        return

    if isinstance(action, np.ndarray):
        vx = float(action[0])
        wz = float(action[1])
        rds.send_cmd_vel(vx, wz)


# ==============================================================
# Main Runner Loop
# ==============================================================

class DdsGr00tInferenceRunner:
    def __init__(self, domain_id=30):
        print("[Runner] Initializing DDS SDK...")
        self.rds = RobotisDDSSDK(domain_id=domain_id)

        print("[Runner] Loading policy...")
        self.policy = load_policy()

        # Track previous sensor values
        self.prev_img = None
        self.prev_odom = None
        self.prev_joint = None

        self.running = True
        print("[Runner] Ready.")

    def run(self):
        print("\n==============================")
        print("    GR00T DDS Inference Runner")
        print("==============================\n")

        while self.running:

            # -------------------------------------------------
            # 1) Read sensors
            # -------------------------------------------------
            img   = self.rds.get_zed_left_image()
            odom  = self.rds.get_odometry()
            joint = self.rds.get_joint_state()

            # -------------------------------------------------
            # 2) Check if fresh (changed) data exists
            # -------------------------------------------------
            # Camera
            if img is None:
                print("[Runner] No camera → waiting...")
                time.sleep(0.1)
                continue
            if self.prev_img is not None and np.array_equal(img, self.prev_img):
                print("[Runner] Camera not updating → waiting...")
                time.sleep(0.1)
                continue

            # Odometry
            if odom is None:
                print("[Runner] No odometry → waiting...")
                time.sleep(0.1)
                continue
            if self.prev_odom is not None and odom == self.prev_odom:
                print("[Runner] Odometry not updating → waiting...")
                time.sleep(0.1)
                continue

            # Joint states
            if joint is None:
                print("[Runner] No joint_state → waiting...")
                time.sleep(0.1)
                continue
            if self.prev_joint is not None and joint["position"] == self.prev_joint:
                print("[Runner] joint_state not updating → waiting...")
                time.sleep(0.1)
                continue

            # -------------------------------------------------
            # 3) Build model input
            # -------------------------------------------------
            data = build_gr00t_input(self.rds)
            if data is None:
                time.sleep(0.05)
                continue

            # -------------------------------------------------
            # 4) Inference
            # -------------------------------------------------
            with torch.no_grad():
                action = self.policy.get_action(data)

            # -------------------------------------------------
            # 5) Apply to robot
            # -------------------------------------------------
            apply_action_to_robot(action, self.rds)

            # -------------------------------------------------
            # 6) Save current sensor values
            # -------------------------------------------------
            self.prev_img = img
            self.prev_odom = odom
            self.prev_joint = joint["position"]


# ==============================================================
# Entry Point
# ==============================================================

def main():
    runner = DdsGr00tInferenceRunner(domain_id=30)
    runner.run()


if __name__ == "__main__":
    main()
