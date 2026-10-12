#!/usr/bin/env python3
"""语音遥控 Lite3：说出"前进 / 后退 / 左转 / 右转 / 停下 / 起立 / 趴下"。

需要可选依赖：uv sync --extra voice
无麦克风时可直接在终端输入文字指令（回车发送）。
"""

import argparse
import threading
import time

import mujoco.viewer

from lite3_sim import Lite3Sim, OnnxPolicyController
from lite3_sim.policy import POLICY_DEFAULT_POSE
from lite3_sim.voice import (
    LIE_POSE,
    POSTURE_GAINS,
    Action,
    MotionCommand,
    PoseTransition,
    TextListener,
    WhisperListener,
    parse_command,
)

_HINT = "支持指令：前进 / 后退 / 左转 / 右转 / 停下 / 起立 / 趴下"
_HOLD_POSE = {Action.STAND: POLICY_DEFAULT_POSE, Action.LIE: LIE_POSE}
_STATE_LABEL = {Action.STAND: "站立中", Action.LIE: "趴下中", Action.MOVE: "运动中"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-voice", action="store_true", help="只用键盘输入，不开麦克风"
    )
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--semantic",
        action="store_true",
        help="用语义嵌入匹配替代关键词匹配（识别自由表述）",
    )
    mode.add_argument(
        "--llm",
        action="store_true",
        help="用大模型解析指令（需 LITE3_LLM_API_KEY）",
    )
    args = parser.parse_args()

    if args.llm:
        from lite3_sim.llm import LLMCommandParser

        parse = LLMCommandParser().parse
    elif args.semantic:
        from lite3_sim.semantic import SemanticCommandParser

        parse = SemanticCommandParser().parse
    else:
        parse = parse_command

    sim = Lite3Sim()
    policy = OnnxPolicyController()
    sim.reset_standing(POLICY_DEFAULT_POSE)

    lock = threading.Lock()
    state = {"command": MotionCommand(Action.STAND), "transition": None}

    def on_text(text: str) -> None:
        if args.llm:
            with lock:
                current = _STATE_LABEL[state["command"].action]
            command = parse(text, state=current)
        else:
            command = parse(text)
        if command is None:
            print(f"[未识别] {text}（{_HINT}）")
            return
        print(f"[指令] {text} → {command.action.value} {command.velocity}")
        with lock:
            state["command"] = command
            if command.action in _HOLD_POSE:
                state["transition"] = PoseTransition(sim, _HOLD_POSE[command.action])

    listeners = [TextListener(on_text)]
    if not args.no_voice:
        listeners.append(WhisperListener(on_text, model_size=args.model))
    for listener in listeners:
        listener.start()
    print(f"语音遥控已启动，{_HINT}；键盘可直接输入指令。")

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            with lock:
                command = state["command"]
                transition = state["transition"]
            if transition is not None:
                if transition.step(sim):
                    with lock:
                        state["transition"] = None
            elif command.action is Action.MOVE:
                policy.step(sim, command.velocity)
            else:
                sim.step_pd(_HOLD_POSE[command.action], gains=POSTURE_GAINS)
            viewer.sync()
            elapsed = time.time() - step_start
            if elapsed < sim.timestep:
                time.sleep(sim.timestep - elapsed)


if __name__ == "__main__":
    main()
