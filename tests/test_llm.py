"""LLM 指令解析测试：用假客户端验证 JSON 解码与动作映射，不访问网络。"""

import json

import pytest

from lite3_sim.llm import LLMCommandParser
from lite3_sim.voice import Action


class _FakeClient:
    """模拟 openai 客户端的最小接口：chat.completions.create(...)"""

    def __init__(self, payload: str | None) -> None:
        self._payload = payload

    @property
    def chat(self) -> object:
        payload = self._payload

        class _Completions:
            def create(self, **_kwargs: object) -> object:
                message = type("M", (), {"content": payload})()
                choice = type("C", (), {"message": message})()
                return type("R", (), {"choices": [choice]})()

        return type("Chat", (), {"completions": _Completions()})()


def _parser(payload: str | None) -> LLMCommandParser:
    return LLMCommandParser(client=_FakeClient(payload), model="test-model")


def test_forward_with_speed_scale() -> None:
    command = _parser(json.dumps({"action": "forward", "speed": 1.5})).parse("快一点往前走")
    assert command is not None and command.action is Action.MOVE
    assert command.velocity.vx == pytest.approx(0.6)


def test_lie_and_stand() -> None:
    assert _parser(json.dumps({"action": "lie"})).parse("卧倒").action is Action.LIE
    assert _parser(json.dumps({"action": "stand"})).parse("别动").action is Action.STAND


def test_none_and_garbage() -> None:
    assert _parser(json.dumps({"action": "none"})).parse("今天天气如何") is None
    assert _parser("这不是JSON").parse("前进") is None
    assert _parser(None).parse("前进") is None
    assert _parser(json.dumps({"action": "fly"})).parse("飞起来") is None


def test_speed_clamped() -> None:
    command = _parser(json.dumps({"action": "backward", "speed": 99})).parse("飞速后退")
    assert command is not None
    assert command.velocity.vx == pytest.approx(-0.3 * 2.0)
