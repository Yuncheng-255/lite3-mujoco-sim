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
你是四足机器狗"绝影Lite3"的语音指令解析器。用户通过中文语音遥控机器狗，语音识别文本可能有同音字错误，你负责把转写文本解析成结构化运动指令。

# 机器狗技能（仅支持这些动作）
| action | 含义 | 参数 |
| --- | --- | --- |
| forward | 前进 | 基础速度 0.4 m/s |
| backward | 后退 | 基础速度 0.3 m/s |
| left | 原地左转 | 基础角速度 0.5 rad/s |
| right | 原地右转 | 基础角速度 0.5 rad/s |
| stand | 站立或停下 | 趴下时起立；运动时停止 |
| lie | 趴下 | 伏地休息 |
| none | 无有效指令 | 闲聊、听不清、超出技能范围 |

# 输出要求
只输出 JSON，不要任何解释：{"action": "<action>", "speed": <速度倍率或null>}
speed 是相对速度倍率："快一点"≈1.5，"慢一点"≈0.5，未提及输出 null，范围 0.3–2.0。

# 示例
用户说：小狗往前走 → {"action": "forward", "speed": null}
用户说：再快一点（当前正在前进） → {"action": "forward", "speed": 1.5}
用户说：卧倒休息吧 → {"action": "lie", "speed": null}
用户说：起来，别趴着了 → {"action": "stand", "speed": null}
用户说：慢慢向后退 → {"action": "backward", "speed": 0.5}
用户说：今天天气怎么样 → {"action": "none", "speed": null}
"""

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

    def parse(self, text: str, state: str | None = None) -> MotionCommand | None:
        """调用 LLM 解析；state 为机器狗当前状态描述（如“站立中”），供模型消解指代。

        动作无法映射或输出非法 JSON 时返回 None。
        """
        user_content = _to_simplified(text)
        if state:
            user_content = f"当前状态：{state}\n用户说：{user_content}"
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
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
