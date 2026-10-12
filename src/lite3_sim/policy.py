"""官方 PPO 行走策略封装。

策略文件来自 DeepRoboticsLab/Lite3_rl_deploy（policy/ppo/policy.onnx，
BSD-3-Clause），观测/动作接口与官方部署代码
run_policy/lite3_test_policy_runner_onnx.hpp 保持一致。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from lite3_sim.robot import Lite3Sim, PDGains

DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[2] / "models" / "lite3" / "policy.onnx"

# 官方部署接口常量（lite3_test_policy_runner_onnx.hpp）
POLICY_DEFAULT_POSE: NDArray[np.float64] = np.array([0.0, -0.65, 1.3] * 4)
ACTION_SCALE: NDArray[np.float64] = np.array([0.125, 0.25, 0.25] * 4)
POLICY_GAINS = PDGains(kp=30.0, kd=1.0)
OMEGA_SCALE = 0.25
DOF_VEL_SCALE = 0.05
OBS_DIM = 45
DECIMATION = 12
MAX_CMD_VEL = 0.8

_FLOAT32 = np.float32


@dataclass(frozen=True)
class VelocityCommand:
    """速度指令：前进 vx、侧移 vy（m/s）、转向 wz（rad/s）。

    官方策略训练范围 ±0.8，超出会被裁剪。
    """

    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0

    def as_obs(self) -> NDArray[_FLOAT32]:
        cmd = np.array([self.vx, self.vy, self.wz], dtype=_FLOAT32)
        return np.clip(cmd, -MAX_CMD_VEL, MAX_CMD_VEL)


class OnnxPolicyController:
    """官方 PPO 策略控制器：观测 → ONNX 推理 → 目标关节角 → PD 力矩。"""

    def __init__(
        self,
        policy_path: Path = DEFAULT_POLICY_PATH,
        decimation: int = DECIMATION,
    ) -> None:
        if not policy_path.is_file():
            raise FileNotFoundError(f"策略文件不存在: {policy_path}")
        self._session = ort.InferenceSession(
            str(policy_path), providers=["CPUExecutionProvider"]
        )
        self._decimation = decimation
        self._count = 0
        self._last_action = np.zeros(12, dtype=_FLOAT32)
        self._target = POLICY_DEFAULT_POSE.copy()

    def reset(self) -> None:
        """清空策略内部状态（上一动作与计数器）。"""
        self._count = 0
        self._last_action[:] = 0.0
        self._target = POLICY_DEFAULT_POSE.copy()

    def _observation(self, sim: Lite3Sim, cmd: VelocityCommand) -> NDArray[_FLOAT32]:
        obs = np.concatenate(
            [
                sim.gyro() * OMEGA_SCALE,
                sim.projected_gravity(),
                cmd.as_obs(),
                sim.joint_positions() - POLICY_DEFAULT_POSE,
                sim.joint_velocities() * DOF_VEL_SCALE,
                self._last_action,
            ]
        )
        if obs.shape != (OBS_DIM,):
            raise RuntimeError(f"观测维度应为 {OBS_DIM}，实际 {obs.shape}")
        return obs.astype(_FLOAT32)[None, :]

    def _infer(self, sim: Lite3Sim, cmd: VelocityCommand) -> None:
        (action,) = self._session.run(
            ["actions"], {"obs": self._observation(sim, cmd)}
        )
        self._last_action = action[0]
        self._target = self._last_action * ACTION_SCALE + POLICY_DEFAULT_POSE

    def step(self, sim: Lite3Sim, cmd: VelocityCommand, n_steps: int = 1) -> None:
        """按 decimation 频率推理策略，其余步 PD 跟踪上一目标。"""
        for _ in range(n_steps):
            if self._count % self._decimation == 0:
                self._infer(sim, cmd)
            self._count += 1
            sim.step_pd(self._target, gains=POLICY_GAINS)
