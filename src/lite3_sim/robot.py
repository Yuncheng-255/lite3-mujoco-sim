"""Lite3 仿真核心：模型加载、PD 关节控制、状态读取与离屏渲染。

模型为官方力矩电机接口（gear=1，±30 N·m），因此位置控制由外层 PD
控制器计算力矩实现，而非修改底层 MJCF。
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

# 仓库内模型默认路径：src/lite3_sim/ -> 仓库根 -> models/lite3
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "lite3" / "Lite3.xml"

JOINT_NAMES: tuple[str, ...] = tuple(
    f"{leg}_{part}_joint"
    for leg in ("FL", "FR", "HL", "HR")
    for part in ("HipX", "HipY", "Knee")
)

# 官方/社区常用站立关节角（rad）；Knee 正值、HipY 负值
STAND_POSE: NDArray[np.float64] = np.array(
    [0.0, -0.9, 1.8] * 4, dtype=np.float64
)

_FLOAT = np.float64


@dataclass(frozen=True)
class PDGains:
    """关节 PD 增益（力矩单位 N·m）。"""

    kp: float = 60.0
    kd: float = 2.0


class Lite3Sim:
    """Lite3 的 MuJoCo 仿真器封装。

    每个实例持有独立的 MjModel/MjData，可安全地创建多实例并行采样。
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        gains: PDGains = PDGains(),
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.gains = gains
        self._renderer: mujoco.Renderer | None = None
        self._qadr = np.array(
            [self.model.joint(name).qposadr[0] for name in JOINT_NAMES]
        )
        self._vadr = np.array(
            [self.model.joint(name).dofadr[0] for name in JOINT_NAMES]
        )
        self._ctrl_range = self.model.actuator_ctrlrange.copy()
        gyro_idx = next(
            i
            for i, sensor_type in enumerate(self.model.sensor_type)
            if sensor_type == mujoco.mjtSensor.mjSENS_GYRO
        )
        self._gyro_adr = int(self.model.sensor_adr[gyro_idx])

    @property
    def timestep(self) -> float:
        """物理步长（秒）。"""
        return float(self.model.opt.timestep)

    def reset(self) -> None:
        """重置到 XML 初始状态（空中 1 m 处释放）。"""
        mujoco.mj_resetData(self.model, self.data)

    def reset_standing(
        self,
        pose: NDArray[_FLOAT] = STAND_POSE,
        height: float = 0.32,
    ) -> None:
        """以站姿直接初始化：躯干置于给定高度、关节置位、速度清零。

        从空中跌落初始化容易形成躯干着地的卡死平衡，站姿初始化是
        官方部署代码采用的方式。
        """
        self.reset()
        self.data.qpos[2] = height
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[7:] = pose
        mujoco.mj_forward(self.model, self.data)

    def joint_positions(self) -> NDArray[_FLOAT]:
        """12 个关节当前角度（rad）。"""
        return self.data.qpos[self._qadr].copy()

    def joint_velocities(self) -> NDArray[_FLOAT]:
        """12 个关节当前角速度（rad/s）。"""
        return self.data.qvel[self._vadr].copy()

    def base_height(self) -> float:
        """躯干（自由基座）离地高度（m）。"""
        return float(self.data.qpos[2])

    def base_orientation(self) -> NDArray[_FLOAT]:
        """躯干姿态四元数 [w, x, y, z]。"""
        return self.data.qpos[3:7].copy()

    def gyro(self) -> NDArray[_FLOAT]:
        """IMU 陀螺仪读数：体坐标系角速度（rad/s）。"""
        return self.data.sensordata[self._gyro_adr : self._gyro_adr + 3].copy()

    def projected_gravity(self) -> NDArray[_FLOAT]:
        """重力方向在躯干坐标系下的投影（RL 策略标准观测）。"""
        rotation = np.zeros(9, dtype=_FLOAT)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
        return rotation.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])

    def pd_torques(
        self,
        target: NDArray[_FLOAT],
        gains: PDGains | None = None,
    ) -> NDArray[_FLOAT]:
        """计算跟踪目标关节角的 PD 力矩，裁剪到执行器范围。"""
        g = gains or self.gains
        torque = g.kp * (target - self.joint_positions()) - g.kd * self.joint_velocities()
        return np.clip(torque, self._ctrl_range[:, 0], self._ctrl_range[:, 1])

    def step_pd(
        self,
        target: NDArray[_FLOAT],
        n_steps: int = 1,
        gains: PDGains | None = None,
    ) -> None:
        """以 PD 控制步进 n_steps 个物理步。"""
        target = np.asarray(target, dtype=_FLOAT)
        if target.shape != (12,):
            raise ValueError(f"目标关节角形状应为 (12,)，实际 {target.shape}")
        for _ in range(n_steps):
            self.data.ctrl[:] = self.pd_torques(target, gains)
            mujoco.mj_step(self.model, self.data)

    def step_torques(self, torques: NDArray[_FLOAT], n_steps: int = 1) -> None:
        """直接以力矩接口步进（供 RL 策略等底层控制使用）。"""
        torques = np.asarray(torques, dtype=_FLOAT)
        if torques.shape != (12,):
            raise ValueError(f"力矩形状应为 (12,)，实际 {torques.shape}")
        clipped = np.clip(torques, self._ctrl_range[:, 0], self._ctrl_range[:, 1])
        for _ in range(n_steps):
            self.data.ctrl[:] = clipped
            mujoco.mj_step(self.model, self.data)

    def render(self, width: int = 640, height: int = 480) -> NDArray[np.uint8]:
        """离屏渲染当前场景，返回 [height, width, 3] uint8 RGB。"""
        if self._renderer is None or (
            self._renderer.width != width or self._renderer.height != height
        ):
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data)
        return self._renderer.render().copy()

    def close(self) -> None:
        """释放渲染器资源。"""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
