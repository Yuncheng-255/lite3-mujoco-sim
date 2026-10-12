"""官方 PPO 行走策略测试：速度指令下的位移与稳定性。"""

import numpy as np

from lite3_sim import Lite3Sim, OnnxPolicyController, VelocityCommand
from lite3_sim.policy import POLICY_DEFAULT_POSE, POLICY_GAINS


def _settle_to_default_pose(sim: Lite3Sim, seconds: float = 1.0) -> None:
    """站姿初始化后用 PD 稳定到策略默认站姿。"""
    sim.reset_standing(POLICY_DEFAULT_POSE)
    sim.step_pd(
        POLICY_DEFAULT_POSE, n_steps=int(seconds / sim.timestep), gains=POLICY_GAINS
    )


def test_policy_walks_forward_without_falling() -> None:
    sim = Lite3Sim()
    policy = OnnxPolicyController()
    _settle_to_default_pose(sim)

    start_x, start_y = sim.data.qpos[0], sim.data.qpos[1]
    cmd = VelocityCommand(vx=0.4)
    policy.step(sim, cmd, n_steps=int(5.0 / sim.timestep))

    # 前进位移显著、横向漂移小、躯干未倾倒
    assert sim.data.qpos[0] - start_x > 1.0
    assert abs(sim.data.qpos[1] - start_y) < 0.5
    assert sim.base_height() > 0.15
    quat = sim.base_orientation()
    tilt = 2.0 * np.arccos(np.clip(abs(quat[0]), 0.0, 1.0))
    assert tilt < np.deg2rad(30.0)


def _yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def test_policy_turns_in_place() -> None:
    sim = Lite3Sim()
    policy = OnnxPolicyController()
    _settle_to_default_pose(sim)

    start_yaw = _yaw(sim.base_orientation())
    policy.step(sim, VelocityCommand(wz=0.5), n_steps=int(4.0 / sim.timestep))

    # 转向指令应产生明显航向变化且不倒
    delta = ( _yaw(sim.base_orientation()) - start_yaw + np.pi) % (2 * np.pi) - np.pi
    assert abs(delta) > np.deg2rad(20.0)
    assert sim.base_height() > 0.15
