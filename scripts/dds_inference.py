#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Robotis DDS Python SDK + DDS Inference Server
# High-level wrapper for DDS-based robot communication + Inference integration
#
# Author: Heewon Lee, Dongyun Kim
# License: Apache 2.0

import time
import torch
import numpy as np

from robotis_dds_python.robotis_dds_sdk.robotis_dds_sdk import RobotisDDSSDK
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy


class DdsInference:
    """
    Runs a loop that:
      1) Reads sensor inputs (cameras, joints) from DDS via RobotisDDSSDK
      2) Converts inputs to GR00T model format
      3) Runs policy inference
      4) Sends arm trajectories back to the robot
    """

    def __init__(self, domain_id=30):
        # Initialize DDS SDK and GR00T policy
        print("[Init] DDS SDK")
        self.rds = RobotisDDSSDK(domain_id=domain_id, robot_type="ffw_bg2")
        print("[Init] Loading policy")
        self.policy = self.load_policy()

        # Previous snapshots to detect new data arrival
        self.prev_imgs = None
        self.prev_joint = None

        print("[Init] Ready")
        self.run()

    def load_policy(self):
        """
        Load GR00T policy with robot-specific config.
        Adjust paths/tags to match your model and robot setup.
        """
        POLICY_PATH = "/workspace/checkpoints/ROBOTIS/ffw_bg2_rev4_pick_coffee_bottle_env5_1_to_34_joint_fix_40k"
        ROBOT_TYPE = "ffw_bg2"
        EMBODIMENT_TAG = "new_embodiment"
        DENOISING_STEPS = 4

        cfg = load_data_config(ROBOT_TYPE)
        return Gr00tPolicy(
            model_path=POLICY_PATH,
            modality_config=cfg.modality_config(),
            modality_transform=cfg.transform(),
            embodiment_tag=EMBODIMENT_TAG,
            denoising_steps=DENOISING_STEPS,
        )

    def _fresh_imgs(self, now, prev):
        """
        Return True if image set is new or changed compared to previous.
        Uses shallow checks and array equality for frames.
        """
        if not now:
            return False
        if prev is None or now.keys() != prev.keys():
            return True
        for k in now.keys():
            a, b = now[k], prev[k]
            if a is None or b is None:
                return True
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                if not np.array_equal(a, b):
                    return True
            else:
                return True
        return False

    def _fresh_joint(self, now, prev):
        """
        Return True if joint positions changed or first time seen.
        """
        if now is None:
            return False
        if prev is None:
            return True
        now_pos = now.get("position")
        prev_pos = prev.get("position")
        if now_pos is None or prev_pos is None:
            return True
        now_pos = np.array(now_pos, dtype=np.float32)
        prev_pos = np.array(prev_pos, dtype=np.float32)
        if now_pos.shape != prev_pos.shape:
            return True
        return not np.array_equal(now_pos, prev_pos)

    def preprocess_action_input(self, imgs, joint):
        """
        Build GR00T input dict:
          - video.<camera_key>: (B,H,W,C)
          - state.joints: full joint vector
          - state.left_arm/right_arm: 7-DoF + pad to 8
        """
        def add_batch(x):
            x = np.asarray(x)
            return x[None] if x.ndim == 3 else x

        data = {}
        # Add all registered camera frames
        for key, img in imgs.items():
            data[f"video.{key}"] = add_batch(img)

        # Add joint states and split into left/right arms
        pos = np.array(joint["position"], dtype=np.float32)
        data["state.joints"] = pos[None]
        left7 = pos[0:7]
        right7 = pos[7:14]
        data["state.left_arm"] = np.concatenate([left7, [0.0]]).reshape(1, -1)
        data["state.right_arm"] = np.concatenate([right7, [0.0]]).reshape(1, -1)
        return data

    def run(self):
        """
        Main control loop:
          - Waits for required inputs
          - Skips cycle if no new data
          - Runs inference and sends arm trajectories
        Note:
          To add cameras, edit config.json (camera_topics) or call SDK register_camera before running.
        """
        print("\n=== GR00T DDS Inference Runner ===\n")
        while True:
            # Read sensor caches from SDK
            imgs = self.rds.get_images()
            joint = self.rds.get_joint_state()

            # Ensure all required cameras are present
            required = list(self.rds._camera_key_map.keys())
            missing = [k for k in required if k not in imgs or imgs[k] is None]
            if missing:
                print(f"[WAIT] Missing cameras: {', '.join(missing)}")
                time.sleep(0.05)
                continue

            # Ensure joint states are available
            if joint is None or joint.get("position") is None:
                print("[WAIT] JointState missing")
                time.sleep(0.05)
                continue

            # Skip until fresh data arrives
            imgs_fresh = self._fresh_imgs(imgs, self.prev_imgs)
            joint_fresh = self._fresh_joint(joint, self.prev_joint)
            if not imgs_fresh and not joint_fresh:
                print("[WAIT] No new data")
                time.sleep(0.02)
                continue

            # Inference
            print("[RUN] Inference")
            data = self.preprocess_action_input(imgs, joint)
            with torch.no_grad():
                action = self.policy.get_action(data)

            # Send arm trajectories back to robot
            left = action.get("action.left_arm")
            right = action.get("action.right_arm")
            if left is not None:
                self.rds.send_arm_trajectory("left", list(left[0]))
            if right is not None:
                self.rds.send_arm_trajectory("right", list(right[0]))

            # Update previous snapshots
            self.prev_imgs = imgs
            self.prev_joint = joint


def main():
    DdsInference(domain_id=30)


if __name__ == "__main__":
    main()
