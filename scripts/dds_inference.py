#!/usr/bin/env python3
#
# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Author: Heewon Lee

import time
import torch
import numpy as np

from robotis_dds_python.robotis_dds_sdk.robotis_dds_sdk import RobotisDDSSDK
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy


class DdsInference:
    # Path to trained GR00T checkpoint
    POLICY_PATH = "/workspace/checkpoints/ROBOTIS/ffw_bg2_rev4_pick_coffee_bottle_env5_1_to_34_joint_fix_40k"
    ROBOT_TYPE = "ffw_bg2"
    EMBODIMENT_TAG = "new_embodiment"
    DENOISING_STEPS = 4

    def __init__(self):
        """Initialize SDK, load policy, and start inference loop"""
        print("[Init] DDS SDK")

        # DDS SDK loads all sensor topics strictly from config.json
        # Modify config.json to change camera names or joint settings
        self.rds = RobotisDDSSDK(domain_id=20, robot_type=self.ROBOT_TYPE)

        print("[Init] Loading policy")
        self.policy = self.load_policy()

        # Store last-read sensors to detect freshness
        self.prev_imgs = None
        self.prev_joint = None

        print("[Init] Ready")
        self.run()

    def load_policy(
        self,
        policy_path=POLICY_PATH,
        embodiment_tag=EMBODIMENT_TAG,
        denoising_steps=DENOISING_STEPS,
        robot_type=ROBOT_TYPE,
    ):
        """Load GR00T policy model with modality config and transforms"""
        cfg = load_data_config(robot_type)
        return Gr00tPolicy(
            model_path=policy_path,
            modality_config=cfg.modality_config(),
            modality_transform=cfg.transform(),
            embodiment_tag=embodiment_tag,
            denoising_steps=denoising_steps,
        )

    def _fresh_imgs(self, now, prev):
        """Check if new camera frames arrived"""
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
        """Check if joint_state changed"""
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

        return not np.array_equal(now_pos, prev_pos)

    def preprocess_input(self, imgs, joint):
        """Convert DDS sensor data into GR00T input format"""

        def add_batch(x):
            """Add batch dimension if needed"""
            x = np.asarray(x)
            return x[None] if x.ndim == 3 else x

        data = {}

        # Cameras (keys come from config.json)
        for key, img in imgs.items():
            data[f"video.{key}"] = add_batch(img)

        # Joint data
        pos = np.array(joint["position"], dtype=np.float32)
        data["state.joints"] = pos[None]

        # ffw_bg2: first 8 = left arm, next 8 = right arm
        left7 = pos[0:8]
        right7 = pos[8:16]
        data["state.left_arm"] = left7.reshape(1, -1)
        data["state.right_arm"] = right7.reshape(1, -1)

        return data

    def run(self):
        """Main loop: Read sensors → Inference → Send trajectory"""
        print("\n=== GR00T DDS Inference Runner ===\n")

        while True:
            imgs = self.rds.get_images()
            joint = self.rds.get_joint_state()

            # Validate required cameras from config.json
            required = list(self.rds._camera_key_map.keys())
            missing = [k for k in required if k not in imgs or imgs[k] is None]
            if missing:
                print(f"[WAIT] Missing cameras: {', '.join(missing)}")
                time.sleep(0.1)
                continue

            if joint is None or joint.get("position") is None:
                print("[WAIT] JointState missing")
                time.sleep(0.1)
                continue

            # Check if ANY data is fresh
            imgs_fresh = self._fresh_imgs(imgs, self.prev_imgs)
            joint_fresh = self._fresh_joint(joint, self.prev_joint)
            
            if not imgs_fresh and not joint_fresh:
                time.sleep(0.01)
                continue

            print("[RUN] Inference")
            data = self.preprocess_input(imgs, joint)

            with torch.no_grad():
                action = self.policy.get_action(data)

            left = action.get("action.left_arm")
            right = action.get("action.right_arm")

            self.rds.send_arm_trajectory("left", list(left[0]))
            self.rds.send_arm_trajectory("right", list(right[0]))

            # Store previous cycle data
            self.prev_imgs = imgs
            self.prev_joint = joint


def main():
    """Entry point: run DdsInference instance"""
    DdsInference()


if __name__ == "__main__":
    main()
