"""语义指令解析测试：自由表述应映射到正确动作，无关文本应被拒绝。

需要 voice 可选依赖（首次运行下载约 24MB 量化模型）。
"""

import pytest

from lite3_sim.semantic import SemanticCommandParser
from lite3_sim.voice import Action

pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def parser() -> SemanticCommandParser:
    return SemanticCommandParser()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("宝贝，往前走两步", Action.MOVE),
        ("小狗向前移动", Action.MOVE),
        ("给我趴下休息", Action.LIE),
        ("卧倒别动", Action.LIE),
        ("停下来吧", Action.STAND),
        ("起立站好", Action.STAND),
    ],
)
def test_freeform_commands(
    parser: SemanticCommandParser, text: str, expected: Action
) -> None:
    command = parser.parse(text)
    assert command is not None, f"未识别: {text}（score={parser.score(text):.3f}）"
    assert command.action is expected


@pytest.mark.parametrize(
    ("text", "sign"),
    [
        ("往前走", "+vx"),
        ("往后退一点", "-vx"),
        ("向左转弯", "+wz"),
        ("向右拐", "-wz"),
    ],
)
def test_direction_semantics(parser: SemanticCommandParser, text: str, sign: str) -> None:
    command = parser.parse(text)
    assert command is not None, f"未识别: {text}（score={parser.score(text):.3f}）"
    velocity = command.velocity
    assert command.action is Action.MOVE
    if sign == "+vx":
        assert velocity.vx > 0
    elif sign == "-vx":
        assert velocity.vx < 0
    elif sign == "+wz":
        assert velocity.wz > 0
    else:
        assert velocity.wz < 0


def test_unrelated_text_rejected(parser: SemanticCommandParser) -> None:
    assert parser.parse("今天股票行情怎么样") is None
