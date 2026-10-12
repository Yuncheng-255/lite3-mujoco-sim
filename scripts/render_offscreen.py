#!/usr/bin/env python3
"""无头离屏渲染：让 Lite3 站定后输出一帧 PNG，用于无显示环境冒烟。"""

import argparse
from pathlib import Path

from PIL import Image

from lite3_sim import STAND_POSE, Lite3Sim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("lite3.png"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=2.0,
        help="渲染前 PD 站立的稳定时长（秒）",
    )
    args = parser.parse_args()

    sim = Lite3Sim()
    steps = int(args.settle_seconds / sim.timestep)
    sim.step_pd(STAND_POSE, n_steps=steps)
    rgb = sim.render(args.width, args.height)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(args.output)
    print(f"已保存 {args.output}（base_height={sim.base_height():.3f} m）")


if __name__ == "__main__":
    main()
