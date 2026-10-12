#!/usr/bin/env python3
"""行走演示：跟踪相机录制 Lite3 行走的 GIF 与关键帧 PNG。"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from lite3_sim import Lite3Sim, OnnxPolicyController, VelocityCommand
from lite3_sim.policy import POLICY_DEFAULT_POSE, POLICY_GAINS

RENDER_INTERVAL_STEPS = 50  # 每 50 个物理步取一帧（0.1 s @ 2 kHz）


def _tracking_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "TORSO"
    )
    camera.distance = 1.6
    camera.elevation = -20.0
    camera.azimuth = 90.0
    return camera


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--vx", type=float, default=0.4, help="前进速度 m/s")
    parser.add_argument("--wz", type=float, default=0.0, help="转向速度 rad/s")
    parser.add_argument("--output", type=Path, default=Path("walk.gif"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    sim = Lite3Sim()
    policy = OnnxPolicyController()
    sim.reset_standing(POLICY_DEFAULT_POSE)
    sim.step_pd(
        POLICY_DEFAULT_POSE, n_steps=int(1.0 / sim.timestep), gains=POLICY_GAINS
    )

    renderer = mujoco.Renderer(sim.model, height=args.height, width=args.width)
    camera = _tracking_camera(sim.model)

    frames: list[Image.Image] = []
    cmd = VelocityCommand(vx=args.vx, wz=args.wz)
    total_steps = int(args.seconds / sim.timestep)
    for start in range(0, total_steps, RENDER_INTERVAL_STEPS):
        policy.step(sim, cmd, n_steps=RENDER_INTERVAL_STEPS)
        renderer.update_scene(sim.data, camera=camera)
        frames.append(Image.fromarray(renderer.render()))
    renderer.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=RENDER_INTERVAL_STEPS * sim.timestep * 1000,
        loop=0,
    )
    frames[len(frames) // 2].save(args.output.with_suffix(".mid.png"))
    frames[-1].save(args.output.with_suffix(".final.png"))

    distance = float(np.hypot(sim.data.qpos[0], sim.data.qpos[1]))
    print(
        f"已保存 {args.output}（{len(frames)} 帧）；"
        f"位移 {distance:.2f} m，躯干高度 {sim.base_height():.3f} m"
    )


if __name__ == "__main__":
    main()
