#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
import torch
import numpy as np

from robotis_dds_python.robotis_dds_sdk.robotis_dds_sdk import RobotisDDSSDK
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy


class DdsInference:
    def __init__(self, domain_id=30):
        print("[Init] DDS SDK")
        self.rds = RobotisDDSSDK(domain_id=domain_id, robot_type="ffw_bg2")
        print("[Init] Loading policy")
        self.policy = self.load_policy()
        self.prev_imgs = None
        self.prev_joint = None
        print("[Init] Ready")
        self.run()

    def load_policy(self):
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
        def to_4d(x):
            x = np.asarray(x)
            return x[None] if x.ndim == 3 else x
        data = {}
        for key, img in imgs.items():
            data[f"video.{key}"] = to_4d(img)
        pos = np.array(joint["position"], dtype=np.float32)
        data["state.joints"] = pos[None]
        left7 = pos[0:7]
        right7 = pos[7:14]
        data["state.left_arm"] = np.concatenate([left7, [0.0]]).reshape(1, -1)
        data["state.right_arm"] = np.concatenate([right7, [0.0]]).reshape(1, -1)
        return data

    def run(self):
        print("\n=== GR00T DDS Inference Runner ===\n")
        while True:
            imgs = self.rds.get_images()
            joint = self.rds.get_joint_state()

            required = list(self.rds._camera_key_map.keys())
            missing = [k for k in required if k not in imgs or imgs[k] is None]
            if missing:
                print(f"[WAIT] Missing cameras: {', '.join(missing)}")
                time.sleep(0.05)
                continue

            if joint is None or joint.get("position") is None:
                print("[WAIT] JointState missing")
                time.sleep(0.05)
                continue

            imgs_fresh = self._fresh_imgs(imgs, self.prev_imgs)
            joint_fresh = self._fresh_joint(joint, self.prev_joint)
            if not imgs_fresh and not joint_fresh:
                print("[WAIT] No new data")
                time.sleep(0.02)
                continue

            print("[RUN] Inference")
            data = self.preprocess_action_input(imgs, joint)
            with torch.no_grad():
                action = self.policy.get_action(data)

            left = action.get("action.left_arm")
            right = action.get("action.right_arm")
            if left is not None:
                self.rds.send_arm_trajectory("left", list(left[0]))
            if right is not None:
                self.rds.send_arm_trajectory("right", list(right[0]))

            self.prev_imgs = imgs
            self.prev_joint = joint


def main():
    DdsInference(domain_id=30)


if __name__ == "__main__":
    main()
