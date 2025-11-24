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

    # ---------------------------------------------------
    # Helper: 이미지 → (1,H,W,C)
    # ---------------------------------------------------
    def to_4d_or_5d(x, name=""):
        rds._debug_image(x, name)
        if x is None:
            return None

        x = np.asarray(x)

        # 3D → 4D
        if x.ndim == 3:
            out = x[None, ...]
            rds._debug_image(out, name + "_to4d")
            return out

        # 4D는 그대로
        if x.ndim == 4:
            return x

        print(f"[ERROR] {name}: Invalid ndim = {x.ndim}")
        return None

    # ---------------------------------------------------
    # Camera Inputs
    # ---------------------------------------------------
    head  = to_4d_or_5d(rds.get_zed_left_image(),  "cam_head")
    left  = to_4d_or_5d(rds.get_left_image(),      "cam_left")
    right = to_4d_or_5d(rds.get_right_image(),     "cam_right")

    # head는 반드시 있어야 GR00T가 돌아감
    if head is None:
        print("[ERROR] cam_head is None → cannot run inference")
        return None

    data = {"video.cam_head": head}
    if left is not None:  data["video.cam_left"]  = left
    if right is not None: data["video.cam_right"] = right

    # ---------------------------------------------------
    # Odom state (없으면 기본값)
    # ---------------------------------------------------
    odom = rds.get_odometry()
    rds._debug_image(odom, "odom")

    if odom is None:
        print("[WARN] odom 없음 → default zero odom 사용")
        data["state.robot"] = np.zeros((1, 5), dtype=np.float32)
    else:
        state_vec = np.array([
            odom["x"], odom["y"], odom["theta"],
            odom["linear_vel"], odom["angular_vel"]
        ], dtype=np.float32)
        data["state.robot"] = state_vec[None, :]
        rds._debug_image(state_vec, "state.robot")

    # ---------------------------------------------------
    # Joint State (필수 입력)
    # ---------------------------------------------------
    joint = rds.get_joint_state()
    rds._debug_image(joint, "joint_state_dict")

    if joint is None:
        print("[WARN] joint_state 없음 → default zero joints 사용")
        positions = np.zeros((25,), dtype=np.float32)
    else:
        positions = np.array(joint["position"], dtype=np.float32)  # shape (25,)

    # full joint state: (1, 25)
    data["state.joints"] = positions[None, :]
    rds._debug_image(data["state.joints"], "state.joints")

    # ---------------------------------------------------
    # Left/Right arm (앞 7개 + pad 1)
    # ---------------------------------------------------
    # 왼팔 joint 7개
    left7 = positions[0:7]   # (7,)
    # 오른팔 joint 7개
    right7 = positions[7:14] # (7,)

    # GR00T 요구: 8차원 → pad 1개 추가
    left8  = np.concatenate([left7,  np.array([0.0], dtype=np.float32)])
    right8 = np.concatenate([right7, np.array([0.0], dtype=np.float32)])

    data["state.left_arm"]  = left8[None, :]   # (1, 8)
    data["state.right_arm"] = right8[None, :]  # (1, 8)

    rds._debug_image(data["state.left_arm"],  "state.left_arm")
    rds._debug_image(data["state.right_arm"], "state.right_arm")

    # ---------------------------------------------------
    # 최종 클린업 (numpy만 허용)
    # ---------------------------------------------------
    clean = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            clean[k] = v
        else:
            print(f"[SKIP] {k} removed (non-numpy)")

    return clean


# ==============================================================
# 4) GR00T 액션 → DDS 로봇 제어로 변환 & 실행
# ==============================================================

def apply_action_to_robot(action, rds: RobotisDDSSDK):
    """
    action: policy.get_action() 결과 (dict 또는 numpy)
    → 적절히 변환해서 DDS publish
    """

    print("\n================ APPLY ACTION DEBUG ================")

    # ---------------------------------------------------
    # 1) action type / keys / shapes 출력
    # ---------------------------------------------------
    print(f"[DEBUG] action type: {type(action)}")

    if isinstance(action, dict):
        print(f"[DEBUG] action keys: {list(action.keys())}")

        # 각 키에 대해 shape/값 로그 출력
        for k, v in action.items():
            if isinstance(v, np.ndarray):
                print(f"[DEBUG] {k}: shape={v.shape}, dtype={v.dtype}")
                print(f"[DEBUG] {k} sample (first row): {v[0] if v.ndim>1 else v[:8]}")
            else:
                print(f"[DEBUG] {k}: {v}")

    elif isinstance(action, np.ndarray):
        print(f"[DEBUG] ndarray action shape={action.shape}, dtype={action.dtype}")
        print(f"[DEBUG] action[:8] = {action[:8]}")
    else:
        print(f"[WARN] Unknown action format: {action}")
        return

    print("====================================================\n")

    # ---------------------------------------------------
    # 실제 로봇 제어 로직
    # ---------------------------------------------------

    # GR00T 구조에 맞는 dictionary일 경우
    if isinstance(action, dict):

        # -------------------------
        # LEFT ARM
        # -------------------------
        if "action.left_arm" in action:
            left_arm = action["action.left_arm"]
            if isinstance(left_arm, np.ndarray):
                # GR00T는 (T, 8)의 trajectory를 반환 → 첫번째 스텝 사용
                target = left_arm[0]
                print(f"[APPLY] LEFT ARM target[0]: {target}")
                rds.send_joint_trajectory(list(target))  # 필요 시 로봇 포맷에 맞게 변환

        # -------------------------
        # RIGHT ARM
        # -------------------------
        if "action.right_arm" in action:
            right_arm = action["action.right_arm"]
            if isinstance(right_arm, np.ndarray):
                target = right_arm[0]
                print(f"[APPLY] RIGHT ARM target[0]: {target}")
                # 예: 양팔을 하나로 합쳐 publish할 수도 있음
                # rds.send_joint_trajectory(list(target))

        # -------------------------
        # optional: cmd_vel
        # -------------------------
        if "cmd_vel" in action:
            vx, wz = action["cmd_vel"]
            print(f"[APPLY] cmd_vel: vx={vx}, wz={wz}")
            rds.send_cmd_vel(vx, wz)

        return

    # ---------------------------------------------------
    # numpy array type action
    # ---------------------------------------------------
    elif isinstance(action, np.ndarray):
        # 아주 단순히 vx/wz로 매핑하는 예시
        vx = float(action[0])
        wz = float(action[1])
        print(f"[APPLY] ndarray action mapped to cmd_vel: vx={vx}, wz={wz}")
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

            # 🔥 여기서 찍어야 함!!!
            for k, v in data.items():
                print(f"[DEBUG] {k} shape:", np.array(v).shape)


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
