"""语音指令关键词解析测试（不依赖音频硬件）。"""

from lite3_sim.voice import Action, parse_command


def test_walk_forward_keywords() -> None:
    for text in ("前进", "往前走", "直行", "小狗向前走"):
        command = parse_command(text)
        assert command is not None and command.action is Action.MOVE
        assert command.velocity.vx > 0


def test_turn_and_backward() -> None:
    left = parse_command("左转")
    right = parse_command("向右转")
    back = parse_command("后退一步")
    assert left is not None and left.velocity.wz > 0
    assert right is not None and right.velocity.wz < 0
    assert back is not None and back.velocity.vx < 0


def test_stand_and_lie() -> None:
    assert parse_command("起立") is not None
    assert parse_command("起立").action is Action.STAND  # type: ignore[union-attr]
    assert parse_command("趴下").action is Action.LIE  # type: ignore[union-attr]
    assert parse_command("快停下").action is Action.STAND  # type: ignore[union-attr]


def test_priority_and_unmatched() -> None:
    # “停下”优先于同时出现的“走”
    assert parse_command("别动，不要走").action is Action.STAND  # type: ignore[union-attr]
    assert parse_command("今天天气不错") is None


def test_pose_transition_lie_and_stand_up() -> None:
    """趴下后能经插值过渡重新起立（策略 PD 增益不足以起立）。"""
    from lite3_sim import Lite3Sim
    from lite3_sim.policy import POLICY_DEFAULT_POSE
    from lite3_sim.voice import LIE_POSE, PoseTransition

    sim = Lite3Sim()
    sim.reset_standing(POLICY_DEFAULT_POSE)
    lie = PoseTransition(sim, LIE_POSE)
    while not lie.step(sim):
        pass
    assert sim.base_height() < 0.2

    stand = PoseTransition(sim, POLICY_DEFAULT_POSE)
    while not stand.step(sim):
        pass
    assert sim.base_height() > 0.25
