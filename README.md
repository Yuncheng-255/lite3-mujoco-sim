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
uv sync --extra voice          # 安装 sherpa-onnx + sounddevice
uv run python scripts/voice_control.py            # 语音 + 键盘双通道
uv run python scripts/voice_control.py --semantic # 语义模式：识别自由表述
uv run python scripts/voice_control.py --no-voice # 仅键盘输入调试

# 大模型模式（需 API key，默认 zju-gateway DeepSeek V4 Flash）
uv sync --extra voice --extra llm
export LITE3_LLM_API_KEY=<你的网关 key>
uv run python scripts/voice_control.py --llm
```

支持指令：**前进 / 后退 / 左转 / 右转 / 停下 / 起立 / 趴下**。
ASR 用阿里 SenseVoiceSmall（sherpa-onnx 推理，int8 模型约 228MB，首次自动下载到 `~/.cache/lite3-sim/`）：原生简体中文输出，CPU 上转写约 0.1 秒/条，中文指令场景准确率显著优于 Whisper。指令解析有三种互斥模式：

- **关键词模式**（默认）：亚毫秒匹配固定指令词
- **语义模式**（`--semantic`，需 `--extra semantic`）：bge-small-zh 句向量（ONNX 量化，~24MB，首次自动下载）与指令例句算余弦相似度，阈值 0.6；"宝贝往前走两步"这类自由表述也能识别，无关文本会被拒绝
- **大模型模式**（`--llm`）：OpenAI 兼容接口输出 JSON（动作 + 速度倍率），能理解"快一点往后退"这类带参数的表述；用 `LITE3_LLM_BASE_URL` / `LITE3_LLM_MODEL` 可切换其他端点

起立/趴下通过 `PoseTransition` 姿态插值完成（行走策略的 PD 增益不足以从趴下起立）。

### 官方二次开发路径（调研结论）

| 层级 | 官方资产 | 能力 |
|---|---|---|
| 关节级 | [Lite3_MotionSDK](https://github.com/DeepRoboticsLab/Lite3_MotionSDK) | UDP 直连运动主机，12 关节 pos/vel/kp/kd/前馈五参数控制，读 IMU/关节/足底力 |
| 策略部署 | [Lite3_rl_deploy](https://github.com/DeepRoboticsLab/Lite3_rl_deploy) | RL 策略 sim-to-sim / sim-to-real（本项目 policy.onnx 来源） |
| 策略训练 | [rl_training](https://github.com/DeepRoboticsLab/rl_training) | 训练新运动技能（上楼梯、新步态等）的官方流程 |
| 高层指令 | 《绝影Lite3运动主机通讯接口》UDP 协议 | 向机载运动程序下发步态/动作模式（仅真机可用） |

仿真中要扩展技能（跳、爬楼梯等），可行路径是 rl_training 训新策略后部署；高层指令接口只存在于真机的机载运动程序。

## 运行测试

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
