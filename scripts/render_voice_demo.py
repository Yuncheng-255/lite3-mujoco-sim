#!/usr/bin/env python3
"""渲染语音遥控演示 GIF：SenseVoice 转写 + 指令解析 + 动作执行，带字幕条。

产出 assets/voice_demo.gif（README 引用）。
"""

import wave
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from lite3_sim import Lite3Sim, OnnxPolicyController
from lite3_sim.policy import POLICY_DEFAULT_POSE
from lite3_sim.voice import (
    LIE_POSE,
    POSTURE_GAINS,
    Action,
    PoseTransition,
    _ensure_model,
    parse_command,
)

_ROOT = Path(__file__).resolve().parent.parent
_OUT = _ROOT / "assets" / "voice_demo.gif"
_SAMPLE_RATE = 16000
_FRAME_STEPS = 50  # 每帧物理步数（0.1s）
_FONT_SIZE = 26

# 演示脚本：(tts 文本, 执行秒数)
_SEQUENCE = [
    ("小狗向前走", 3.0),
    ("向左转", 2.5),
    ("小狗趴下", 4.0),
    ("小狗站起来", 4.0),
]


def _synthesize_speech(text: str) -> NDArray[np.float32]:
    """macOS say 合成语音（离线演示可复现）。"""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    aiff_path = wav_path.replace(".wav", ".aiff")
    subprocess.run(["say", "-v", "Tingting", text, "-o", aiff_path], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", aiff_path, wav_path],
        check=True,
    )
    with wave.open(wav_path, "rb") as wav:
        frames = wav.readframes(wav.getnframes())
    Path(aiff_path).unlink(missing_ok=True)
    Path(wav_path).unlink(missing_ok=True)
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _make_recognizer() -> object:
    import sherpa_onnx

    model_dir = _ensure_model()
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(model_dir / "model.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        use_itn=True,
    )


def _load_font() -> ImageFont.FreeTypeFont:
    for candidate in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, _FONT_SIZE)
    return ImageFont.load_default()


def _caption_bar(frame: Image.Image, font: ImageFont.FreeTypeFont, line: str) -> Image.Image:
    """在画面底部加半透明字幕条。"""
    bar = Image.new("RGBA", (frame.width, _FONT_SIZE + 18), (0, 0, 0, 170))
    draw = ImageDraw.Draw(bar)
    bbox = draw.textbbox((0, 0), line, font=font)
    draw.text(((frame.width - bbox[2]) // 2, 8), line, font=font, fill=(255, 255, 255))
    composed = frame.convert("RGBA")
    composed.alpha_composite(bar, (0, frame.height - bar.height))
    return composed.convert("RGB")


def main() -> None:
    recognizer = _make_recognizer()
    font = _load_font()

    sim = Lite3Sim()
    policy = OnnxPolicyController()
    sim.reset_standing(POLICY_DEFAULT_POSE)
    hold_pose = {Action.STAND: POLICY_DEFAULT_POSE, Action.LIE: LIE_POSE}

    renderer = mujoco.Renderer(sim.model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(
        sim.model, mujoco.mjtObj.mjOBJ_BODY, "TORSO"
    )
    camera.distance = 1.8
    camera.elevation = -18.0
    camera.azimuth = 135.0

    frames: list[Image.Image] = []
    transition: PoseTransition | None = None
    for spoken, seconds in _SEQUENCE:
        stream = recognizer.create_stream()
        stream.accept_waveform(_SAMPLE_RATE, _synthesize_speech(spoken))
        recognizer.decode_stream(stream)
        text = stream.result.text.strip("，。！？,.!? ")
        command = parse_command(text)
        assert command is not None, f"未识别: {text}"
        label = f"语音“{text}” → {command.action.value}"
        if command.velocity.vx or command.velocity.wz:
            label += f" vx={command.velocity.vx:.1f} wz={command.velocity.wz:.1f}"
        print(label)
        if command.action in hold_pose:
            transition = PoseTransition(sim, hold_pose[command.action])
        total = int(seconds / sim.timestep)
        for start in range(0, total, _FRAME_STEPS):
            for _ in range(min(_FRAME_STEPS, total - start)):
                if transition is not None:
                    if transition.step(sim):
                        transition = None
                elif command.action is Action.MOVE:
                    policy.step(sim, command.velocity)
                else:
                    sim.step_pd(hold_pose[command.action], gains=POSTURE_GAINS)
            renderer.update_scene(sim.data, camera=camera)
            frames.append(_caption_bar(Image.fromarray(renderer.render()), font, label))
        print(f"  高度={sim.base_height():.3f}")
    renderer.close()

    _OUT.parent.mkdir(exist_ok=True)
    frames[0].save(
        _OUT, save_all=True, append_images=frames[1:], duration=100, loop=0
    )
    print(f"已生成 {_OUT}（{len(frames)} 帧）")


if __name__ == "__main__":
    main()
