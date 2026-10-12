"""LLM 指令解析：OpenAI 兼容接口 → 结构化 MotionCommand。

默认接入 zju-gateway 的 DeepSeek V4 Flash（与 pi 配置一致）。
配置（环境变量，密钥不进仓库）：

    export LITE3_LLM_API_KEY=...      # 必填
    export LITE3_LLM_BASE_URL=...     # 可选，默认 zju-gateway
    export LITE3_LLM_MODEL=...        # 可选，默认 deepseek-v4-flash-780k-local
"""

import json
import os
from typing import Any

from lite3_sim.policy import VelocityCommand
from lite3_sim.voice import Action, MotionCommand, _to_simplified

DEFAULT_BASE_URL = "https://modelgateway.zju.edu.cn/v1"
DEFAULT_MODEL = "deepseek-v4-flash-780k-local"
_SPEED_RANGE = (0.3, 2.0)

_SYSTEM_PROMPT = """\
你是机器狗指令解析器。把用户的话解析成 JSON 输出，格式：
{"action": "...", "speed": 数字或null}

action 只能是以下之一：
- forward：前进、往前走
- backward：后退、退后
- left：左转、向左转
- right：右转、向右转
- stand：起立、站起来、停下、别动
- lie：趴下、躺下、卧倒
- none：与机器狗控制无关，或无法确定意图

speed 是相对速度倍率："快一点"约1.5，"慢一点"约0.5，未提及为null。
只输出 JSON，不要输出任何解释。"""

_MOVE_BASE = {
    "forward": VelocityCommand(vx=0.4),
    "backward": VelocityCommand(vx=-0.3),
    "left": VelocityCommand(wz=0.5),
    "right": VelocityCommand(wz=-0.5),
}


class LLMCommandParser:
    """大模型解析器：自由表述（含速度修饰）→ MotionCommand。"""

    def __init__(
        self,
        client: Any | None = None,
        model: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        if client is None:
            api_key = os.environ.get("LITE3_LLM_API_KEY")
            if not api_key:
                raise RuntimeError("缺少环境变量 LITE3_LLM_API_KEY")
            from openai import OpenAI

            client = OpenAI(
                base_url=os.environ.get("LITE3_LLM_BASE_URL", DEFAULT_BASE_URL),
                api_key=api_key,
                timeout=timeout,
            )
        self._client = client
        self._model = model or os.environ.get("LITE3_LLM_MODEL", DEFAULT_MODEL)

    def parse(self, text: str) -> MotionCommand | None:
        """调用 LLM 解析；动作无法映射或输出非法 JSON 时返回 None。"""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _to_simplified(text)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return self._decode(response.choices[0].message.content)

    def _decode(self, content: str | None) -> MotionCommand | None:
        if not content:
            return None
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None
        action = data.get("action")
        if action == "stand":
            return MotionCommand(Action.STAND)
        if action == "lie":
            return MotionCommand(Action.LIE)
        base = _MOVE_BASE.get(action)
        if base is None:
            return None
        speed = data.get("speed")
        scale = 1.0
        if isinstance(speed, (int, float)):
            scale = min(_SPEED_RANGE[1], max(_SPEED_RANGE[0], float(speed)))
        return MotionCommand(
            Action.MOVE,
            VelocityCommand(vx=base.vx * scale, vy=base.vy * scale, wz=base.wz * scale),
        )


__all__ = ["LLMCommandParser"]
