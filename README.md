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

### 官方 PPO 行走策略

```bash
# 录制行走 GIF（跟踪相机，默认前进 0.4 m/s）
uv run python scripts/walk_demo.py --seconds 6 --vx 0.4 --output walk.gif
# 原地转向
uv run python scripts/walk_demo.py --wz 0.5 --output turn.gif
```

```python
from lite3_sim import Lite3Sim, OnnxPolicyController, VelocityCommand
from lite3_sim.policy import POLICY_DEFAULT_POSE

sim = Lite3Sim()
policy = OnnxPolicyController()
sim.reset_standing(POLICY_DEFAULT_POSE)   # 站姿初始化（避免跌落冲击）
policy.step(sim, VelocityCommand(vx=0.4), n_steps=5000)  # 前进 2.5 s
```

策略文件 `models/lite3/policy.onnx` 来自官方 Lite3_rl_deploy（PPO，45 维观测/12 维动作），速度指令范围 ±0.8 m/s（rad/s），推理频率为物理步的 1/12。

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

### 语音遥控（可选）

```bash
uv sync --extra voice          # 安装 faster-whisper + sounddevice
uv run python scripts/voice_control.py            # 语音 + 键盘双通道
uv run python scripts/voice_control.py --semantic # 语义模式：识别自由表述
uv run python scripts/voice_control.py --no-voice # 仅键盘输入调试
```

支持指令：**前进 / 后退 / 左转 / 右转 / 停下 / 起立 / 趴下**。
本地 faster-whisper 中文识别（默认 base 模型）；指令解析有两种模式：

- **关键词模式**（默认）：亚毫秒匹配固定指令词
- **语义模式**（`--semantic`）：bge-small-zh 句向量（ONNX 量化，~24MB，首次自动下载）与指令例句算余弦相似度，阈值 0.6；"宝贝往前走两步"这类自由表述也能识别，无关文本会被拒绝

识别文本先做繁转简（OpenCC）。起立/趴下通过 `PoseTransition` 姿态插值完成（行走策略的 PD 增益不足以从趴下起立）。

### 运行测试

```bash
uv run pytest                  # 基础测试
uv run pytest --extra voice    # 语音相关测试依赖见 [project.optional-dependencies]
```

## 模型与关节约定

- 12 个铰链关节，命名 `{FL,FR,HL,HR}_{HipX,HipY,Knee}_joint`
- 12 个力矩电机 `*_joint_ctrl`，`gear=1`，力矩范围 ±30 N·m
- 基座为 free joint；`imu_site` 提供加速度计/陀螺仪传感器
- 默认站立关节角 `STAND_POSE`：HipX=0、HipY=−0.9 rad、Knee=1.8 rad

## 许可与致谢

- 本仓库代码：MIT（见 [LICENSE](LICENSE)）
- `models/lite3/` 下的模型文件与 `policy.onnx` 来自 [DeepRoboticsLab/deep_robotics_model](https://github.com/DeepRoboticsLab/deep_robotics_model) 和 [DeepRoboticsLab/Lite3_rl_deploy](https://github.com/DeepRoboticsLab/Lite3_rl_deploy)，BSD-3-Clause 许可，原始许可证保留在 [models/lite3/LICENSE.txt](models/lite3/LICENSE.txt)
