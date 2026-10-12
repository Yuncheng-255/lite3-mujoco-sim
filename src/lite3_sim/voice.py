"""语音指令识别：faster-whisper 本地转写 + 关键词映射为运动指令。

依赖为可选安装：``uv sync --extra voice``。
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from lite3_sim.policy import VelocityCommand
from lite3_sim.robot import PDGains, Lite3Sim

_T2S = None


def _to_simplified(text: str) -> str:
    """繁体转简体：Whisper 中文输出常混入繁体字。"""
    global _T2S
    if _T2S is None:
        from opencc import OpenCC

        _T2S = OpenCC("t2s")
    return _T2S.convert(text)

# 官方部署代码中的趴下关节角（lite3 URDF_INIT）
LIE_POSE: NDArray[np.float64] = np.array([0.0, -1.35453, 2.54948] * 4)

# 起立/趴下过渡使用的较强 PD 增益（行走策略的 kp=30 不足以从趴下起立）
POSTURE_GAINS = PDGains(kp=60.0, kd=2.0)
POSTURE_TRANSITION_SECONDS = 2.5


class Action(Enum):
    """语音指令对应的动作模式。"""

    STAND = "stand"
    LIE = "lie"
    MOVE = "move"


@dataclass(frozen=True)
class MotionCommand:
    """解析后的运动指令：动作模式 + 速度（仅 MOVE 有效）。"""

    action: Action
    velocity: VelocityCommand = VelocityCommand()


# 关键词表：先匹配者生效，顺序即优先级
_KEYWORDS: tuple[tuple[tuple[str, ...], MotionCommand], ...] = (
    (("起立", "站起来", "站立", "起身"), MotionCommand(Action.STAND)),
    (("趴下", "躺下", "卧倒"), MotionCommand(Action.LIE)),
    (("停下", "停止", "别动", "站住"), MotionCommand(Action.STAND)),
    (("左转", "往左", "向左"), MotionCommand(Action.MOVE, VelocityCommand(wz=0.5))),
    (("右转", "往右", "向右"), MotionCommand(Action.MOVE, VelocityCommand(wz=-0.5))),
    (("后退", "退后", "往后"), MotionCommand(Action.MOVE, VelocityCommand(vx=-0.3))),
    (("前进", "直行", "往前", "向前", "走"), MotionCommand(Action.MOVE, VelocityCommand(vx=0.4))),
)


def parse_command(text: str) -> MotionCommand | None:
    """从转写文本提取运动指令；无匹配返回 None。"""
    text = _to_simplified(text)
    for keywords, command in _KEYWORDS:
        if any(word in text for word in keywords):
            return command
    return None


_SAMPLE_RATE = 16000


class WhisperListener:
    """后台线程循环录音转写，识别文本经回调分发。"""

    def __init__(
        self,
        on_text: Callable[[str], None],
        model_size: Literal["tiny", "base", "small"] = "base",
        record_seconds: float = 3.0,
    ) -> None:
        self._on_text = on_text
        self._record_seconds = record_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # 延迟加载，避免未安装 voice 依赖时影响包导入
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def _record_chunk(self) -> NDArray[np.float32]:
        import sounddevice as sd

        audio = sd.rec(
            int(self._record_seconds * _SAMPLE_RATE),
            samplerate=_SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        return audio[:, 0]

    def _loop(self) -> None:
        while not self._stop.is_set():
            audio = self._record_chunk()
            if self._stop.is_set():
                break
            segments, _ = self._model.transcribe(audio, language="zh", beam_size=1)
            text = "".join(segment.text for segment in segments).strip()
            if text:
                self._on_text(text)

    def start(self) -> None:
        """启动后台监听线程。"""
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """请求停止；当前录音块结束后线程退出。"""
        self._stop.set()


class PoseTransition:
    """起立/趴下的姿态插值过渡：从当前关节角渐进到目标姿态。"""

    def __init__(
        self,
        sim: Lite3Sim,
        target: NDArray[np.float64],
        seconds: float = POSTURE_TRANSITION_SECONDS,
        gains: PDGains = POSTURE_GAINS,
    ) -> None:
        self._start = sim.joint_positions()
        self._target = target
        self._total = max(1, int(seconds / sim.timestep))
        self._gains = gains
        self._count = 0

    def step(self, sim: Lite3Sim) -> bool:
        """步进一个物理步，返回过渡是否完成。"""
        self._count += 1
        alpha = min(1.0, self._count / self._total)
        sim.step_pd(
            (1.0 - alpha) * self._start + alpha * self._target, gains=self._gains
        )
        return self._count >= self._total


class TextListener:
    """键盘文字输入回退：无麦克风环境下调试用。"""

    def __init__(self, on_text: Callable[[str], None]) -> None:
        self._on_text = on_text

    def start(self) -> None:
        def loop() -> None:
            while True:
                try:
                    text = input()
                except EOFError:
                    return
                if text.strip():
                    self._on_text(text.strip())

        threading.Thread(target=loop, daemon=True).start()


__all__ = [
    "Action",
    "LIE_POSE",
    "MotionCommand",
    "PoseTransition",
    "TextListener",
    "WhisperListener",
    "parse_command",
]
