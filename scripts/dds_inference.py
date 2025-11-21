#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DDS Sensor → GR00T Policy Inference → DDS Command Publisher
Full Pipeline Runner
"""

import time
import torch
import numpy as np

from robotis_dds_python.robotis_dds_sdk.robotis_dds_sdk import RobotisDDSSDK
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy


# ==============================================================
# 1) 하드코딩 설정
# ==============================================================

POLICY_TYPE = "GR00T_N1_5"
POLICY_PATH = "/workspace/checkpoints/ROBOTIS/ffw_bg2_rev4_pick_coffee_bottle_env5_1_to_34_joint_fix_40k"
ROBOT_TYPE = "ffw_bg2"
EMBODIMENT_TAG = "new_embodiment"
DENOISING_STEPS = 4

    # ==============================================================
# 2) 정책 로드
# ==============================================================

def load_policy():
    """
    DDS 통신 없이 로컬에서 직접 policy를 로딩.
    """
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
# 3) DDS 입력 → GR00T input 변환
# ==============================================================

def build_gr00t_input(rds: RobotisDDSSDK):
    """
    rds.get_xxx() → GR00T get_action() 입력 dict 형태로 변환.
    """

    rgb = rds.get_rgb_image()  # BGR ndarray (H,W,3)
    if rgb is None:
        return None

    odom = rds.get_odometry()
    joint_state = rds.get_joint_state()

    # ---- GR00T 모델이 요구하는 key 이름은 환경에 맞게 조정해야 함 ----
    data = {
        "rgb": rgb,
        "odom": odom,                     # {'x':..., 'y':..., ...} or None
        "joint_state": joint_state,       # dict[name→pos] or None
    }

    return data


# ==============================================================
# 4) GR00T 액션 → DDS 로봇 제어로 변환 & 실행
# ==============================================================

def apply_action_to_robot(action, rds: RobotisDDSSDK):
    """
    action: policy.get_action() 결과 (dict 또는 numpy)
    → 적절히 변환해서 DDS publish
    """

    if action is None:
        return

    # GR00T 구조에 맞게 조정 필요
    if isinstance(action, dict):
        # ------ 예시 1: cmd_vel ------
        if "cmd_vel" in action:
            vx, wz = action["cmd_vel"]
            rds.send_cmd_vel(vx, wz)

        # ------ 예시 2: joint trajectory ------
        if "joint_positions" in action:
            rds.send_joint_trajectory(action["joint_positions"])

    # GR00T가 numpy array로 리턴하는 경우 → 로봇에 맞게 여기서 처리
    elif isinstance(action, np.ndarray):
        # 예: 첫 2개를 cmd_vel로 사용
        vx = float(action[0])
        wz = float(action[1])
        rds.send_cmd_vel(vx, wz)


# ==============================================================
# 5) 메인 루프 클래스
# ==============================================================

class DdsGr00tInferenceRunner:
    def __init__(self, domain_id=30):
        print("[Runner] Initializing DDS SDK...")
        self.rds = RobotisDDSSDK(domain_id=domain_id)

        print("[Runner] Loading policy...")
        self.policy = load_policy()

        self.running = True
        print("[Runner] Ready.")

    def run(self):
        print("\n==============================")
        print("  GR00T DDS Inference Runner")
        print("==============================\n")

        while self.running:

            # 1) DDS → GR00T 입력 구성
            data = build_gr00t_input(self.rds)
            if data is None:
                time.sleep(0.01)
                continue

            # 2) inference
            with torch.no_grad():
                action = self.policy.get_action(data)

            # 3) 로그 출력
            print("[Inference] action:", action)

            # 4) GR00T action → DDS publish
            apply_action_to_robot(action, self.rds)


# ==============================================================
# 6) 엔트리 포인트
# ==============================================================

def main():
    runner = DdsGr00tInferenceRunner(domain_id=30)
    runner.run()


if __name__ == "__main__":
    main()
