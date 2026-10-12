"""Lite3 仿真冒烟测试：加载、站立稳定性、离屏渲染。"""

import numpy as np

from lite3_sim import JOINT_NAMES, STAND_POSE, Lite3Sim


def test_model_loads_with_12_joints_and_actuators() -> None:
    sim = Lite3Sim()
    assert sim.model.njnt == 13  # 1 自由基座 + 12 关节
    assert sim.model.nu == 12
    assert sim.model.nq == 19  # 7 (free joint) + 12
    for name in JOINT_NAMES:
        assert sim.model.joint(name) is not None


def test_pd_stand_stabilizes() -> None:
    sim = Lite3Sim()
    steps = int(3.0 / sim.timestep)
    sim.step_pd(STAND_POSE, n_steps=steps)

    # 站立稳定：躯干高度合理、姿态接近水平、关节基本静止
    assert 0.25 < sim.base_height() < 0.45
    quat = sim.base_orientation()
    tilt = 2.0 * np.arccos(np.clip(abs(quat[0]), 0.0, 1.0))
    assert tilt < np.deg2rad(20.0)
    assert np.max(np.abs(sim.joint_velocities())) < 0.5


def test_offscreen_render_shape_and_content() -> None:
    sim = Lite3Sim()
    sim.step_pd(STAND_POSE, n_steps=int(1.0 / sim.timestep))
    rgb = sim.render(320, 240)
    assert rgb.shape == (240, 320, 3)
    assert rgb.dtype == np.uint8
    # 场景非纯背景：存在多种颜色（地面棋盘格 + 机器人）
    assert len(np.unique(rgb.reshape(-1, 3), axis=0)) > 10
    sim.close()
