"""语音指令识别：SenseVoice 本地转写 + 关键词映射为运动指令。

ASR 采用阿里 SenseVoiceSmall（sherpa-onnx 推理，int8 量化约 228MB），
原生输出简体中文，首次使用自动下载模型到 ~/.cache/lite3-sim/。
依赖为可选安装：``uv sync --extra voice``。
"""

import tarfile
import threading
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lite3_sim.policy import VelocityCommand
from lite3_sim.robot import PDGains, Lite3Sim

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
    for keywords, command in _KEYWORDS:
        if any(word in text for word in keywords):
            return command
    return None


_SAMPLE_RATE = 16000
_MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
_MODEL_DIR_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
_PUNCTUATION = "，。！？、,.!? "


def _ensure_model() -> Path:
    """下载并解压 SenseVoice 模型到缓存目录，返回模型目录。"""
    model_dir = Path.home() / ".cache" / "lite3-sim" / _MODEL_DIR_NAME
    if (model_dir / "model.int8.onnx").exists():
        return model_dir
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    archive = model_dir.parent / "sense-voice.tar.bz2"
    print(f"首次使用，下载 SenseVoice 模型（约 236MB）→ {model_dir.parent}")
    urllib.request.urlretrieve(_MODEL_URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(model_dir.parent, filter="data")
    archive.unlink()
    return model_dir


def transcribe_file(recognizer: object, wav_path: str) -> str:
    """转写本地 wav 文件（供离线测试与脚本使用）。"""
    import wave

    with wave.open(wav_path, "rb") as wav:
        assert wav.getframerate() == _SAMPLE_RATE, f"需要 16kHz 单声道 wav: {wav_path}"
        frames = wav.readframes(wav.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    stream = recognizer.create_stream()
    stream.accept_waveform(_SAMPLE_RATE, samples)
    recognizer.decode_stream(stream)
    return stream.result.text.strip().strip(_PUNCTUATION)


class SenseVoiceListener:
    """后台线程循环录音，SenseVoice 转写后经回调分发。"""

    def __init__(
        self,
        on_text: Callable[[str], None],
        record_seconds: float = 3.0,
    ) -> None:
        self._on_text = on_text
        self._record_seconds = record_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # 延迟加载，避免未安装 voice 依赖时影响包导入
        import sherpa_onnx

        model_dir = _ensure_model()
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model_dir / "model.int8.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            use_itn=True,
        )

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
            stream = self._recognizer.create_stream()
            stream.accept_waveform(_SAMPLE_RATE, audio)
            self._recognizer.decode_stream(stream)
            text = stream.result.text.strip().strip(_PUNCTUATION)
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
    "SenseVoiceListener",
    "TextListener",
    "parse_command",
    "transcribe_file",
]
