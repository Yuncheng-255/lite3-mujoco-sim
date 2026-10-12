# Lite3 MuJoCo 仿真

基于云深处（DeepRobotics）官方 MJCF 模型的 **Lite3 四足机器狗 MuJoCo 仿真**最小可复用工程。

- 纯 MuJoCo + Python，无 ROS 依赖
- 关节力矩接口 + 内置 PD 位置控制器（站立）
- 支持交互式查看器、无头离屏渲染、程序化步进
- 附带 pytest 冒烟测试（模型加载 / 站立稳定 / 离屏渲染）

## 安装

使用 [uv](https://docs.astral.sh/uv/)：

```bash
git clone https://github.com/Yuncheng-255/lite3-mujoco-sim.git
cd lite3-mujoco-sim
uv sync
```

## 快速开始

### 交互式查看器（PD 站立）

```bash
uv run python scripts/viewer.py
```

### 无头渲染一帧 PNG

```bash
uv run python scripts/render_offscreen.py --output lite3.png
```

### 程序化使用

```python
from lite3_sim import Lite3Sim, STAND_POSE

sim = Lite3Sim()                      # 加载 models/lite3/Lite3.xml
for _ in range(2000):                 # 2 kHz 物理步进
    sim.step_pd(STAND_POSE)           # PD 跟踪站立关节角
print(sim.base_height())              # 站立稳定后的躯干高度
rgb = sim.render(640, 480)            # 离屏渲染 [H, W, 3] uint8
```

### 运行测试

```bash
uv run pytest
```

## 模型与关节约定

- 12 个铰链关节，命名 `{FL,FR,HL,HR}_{HipX,HipY,Knee}_joint`
- 12 个力矩电机 `*_joint_ctrl`，`gear=1`，力矩范围 ±30 N·m
- 基座为 free joint；`imu_site` 提供加速度计/陀螺仪传感器
- 默认站立关节角 `STAND_POSE`：HipX=0、HipY=−0.9 rad、Knee=1.8 rad

## 许可与致谢

- 本仓库代码：MIT（见 [LICENSE](LICENSE)）
- `models/lite3/` 下的模型文件来自 [DeepRoboticsLab/deep_robotics_model](https://github.com/DeepRoboticsLab/deep_robotics_model)，BSD-3-Clause 许可，原始许可证保留在 [models/lite3/LICENSE.txt](models/lite3/LICENSE.txt)
