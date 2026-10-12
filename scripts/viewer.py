#!/usr/bin/env python3
"""交互式查看器：Lite3 在 PD 控制下站立，可用鼠标环视场景。"""

import time

import mujoco.viewer

from lite3_sim import STAND_POSE, Lite3Sim


def main() -> None:
    sim = Lite3Sim()
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            sim.step_pd(STAND_POSE)
            viewer.sync()
            elapsed = time.time() - step_start
            if elapsed < sim.timestep:
                time.sleep(sim.timestep - elapsed)


if __name__ == "__main__":
    main()
