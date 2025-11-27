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
        print("[Runner] Initializing DDS SDK...")

        # robot_type으로 config.json 기반 카메라/팔 자동 등록된 상태여야 함
        self.rds = RobotisDDSSDK(domain_id=domain_id, robot_type="ffw_bg2")

        print("[Runner] Loading GR00T policy...")
        self.policy = self.load_policy()

        # 이전 프레임/조인트 저장용
        self.prev_imgs = None
        self.prev_joint = None

        print("[Runner] Ready.")
        self.run()

    def load_policy(self):
        POLICY_PATH = "/workspace/checkpoints/ROBOTIS/ffw_bg2_rev4_pick_coffee_bottle_env5_1_to_34_joint_fix_40k"
        ROBOT_TYPE = "ffw_bg2"
        EMBODIMENT_TAG = "new_embodiment"
        DENOISING_STEPS = 4

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

    # ------------------------------------------------------------
    # Fresh 비교 함수 (이미지)
    # ------------------------------------------------------------
    def _fresh_imgs(self, now, prev):
        # 아무것도 없으면 fresh 아님
        if now is None or now == {}:
            return False
        # 이전이 없으면 fresh
        if prev is None:
            return True

        # key 셋이 달라지면 fresh
        if now.keys() != prev.keys():
            return True

        # 각 카메라별로 프레임 비교
        for k in now.keys():
            a = now[k]
            b = prev[k]
            if a is None or b is None:
                return True
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                if not np.array_equal(a, b):
                    return True
            else:
                # 예상치 못한 타입이면 그냥 fresh 된 걸로 봄
                return True

        # 완전히 동일 → fresh 아님 (stale)
        return False

    # ------------------------------------------------------------
    # Fresh 비교 함수 (조인트)
    # ------------------------------------------------------------
    def _fresh_joint(self, now, prev):
        if now is None:
            return False
        if prev is None:
            return True

        now_pos = np.array(now.get("position", []), dtype=np.float32)
        prev_pos = np.array(prev.get("position", []), dtype=np.float32)

        if now_pos.shape != prev_pos.shape:
            return True

        return not np.array_equal(now_pos, prev_pos)

    # ------------------------------------------------------------
    # GR00T Input 생성
    # ------------------------------------------------------------
    def preprocess_action_input(self, imgs, joint):

        def to_4d(x):
            x = np.asarray(x)
            if x.ndim == 3:      # HWC
                return x[None]   # (1,H,W,C)
            return x

        data = {}

        # video.cam_head, video.cam_left, video.cam_right ...
        for key, img in imgs.items():
            data[f"video.{key}"] = to_4d(img)

        # joint 상태
        pos = np.array(joint["position"], dtype=np.float32)
        data["state.joints"] = pos[None]

        # GR00T용 left/right arm state
        left7 = pos[0:7]
        right7 = pos[7:14]
        data["state.left_arm"] = np.concatenate([left7, [0.0]]).reshape(1, -1)
        data["state.right_arm"] = np.concatenate([right7, [0.0]]).reshape(1, -1)

        return data

    # ------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------
    def run(self):
        print("\n==============================")
        print("  GR00T DDS Inference Runner")
        print("==============================\n")

        while True:
            # config.json 기반으로 등록된 카메라 전부를 dict로 가져옴
            imgs = self.rds.get_images()
            joint = self.rds.get_joint_state()

            # ------------- (1) 카메라 모두 반드시 있어야 함 -------------
            required_keys = list(self.rds._camera_key_map.keys())
            missing = [k for k in required_keys if k not in imgs or imgs[k] is None]

            if missing:
                print(f"[WAIT] Missing cameras: {', '.join(missing)} → Inference paused")
                time.sleep(0.05)
                continue

            # ------------- (2) JointState 반드시 필요 -------------
            if joint is None:
                print("[WAIT] JointState missing → Inference paused")
                time.sleep(0.05)
                continue

            # ------------- (3) Fresh-check -------------
            imgs_fresh = self._fresh_imgs(imgs, self.prev_imgs)
            joint_fresh = self._fresh_joint(joint, self.prev_joint)

            # 둘 다 새 데이터가 아니면 → rosbag 안 들어오는 걸로 보고 그냥 대기
            if not imgs_fresh and not joint_fresh:
                print("[WAIT] No new data → Inference paused")
                time.sleep(0.02)
                continue

            # ------------- (4) Inference 실행 -------------
            print("[RUN] New data received → Running inference...")

            data = self.preprocess_action_input(imgs, joint)

            with torch.no_grad():
                action = self.policy.get_action(data)

            # ------------- (5) Apply action -------------
            left = action.get("action.left_arm")
            right = action.get("action.right_arm")

            if left is not None:
                self.rds.send_arm_trajectory("left", list(left[0]))
            if right is not None:
                self.rds.send_arm_trajectory("right", list(right[0]))

            # ------------- (6) Save previous -------------
            self.prev_imgs = imgs
            self.prev_joint = joint


def main():
    DdsInference(domain_id=30)


if __name__ == "__main__":
    main()
