from tools.inferno_rl.training.schedules import (
    ConstantSchedule,
    LinearSchedule,
    PiecewiseSchedule,
)


def test_constant_schedule() -> None:
    schedule = ConstantSchedule(0.25)
    assert schedule.value(0) == 0.25
    assert schedule.value(1000) == 0.25


def test_linear_schedule() -> None:
    schedule = LinearSchedule(1.0, 0.0, 100)
    assert schedule.value(0) == 1.0
    assert schedule.value(50) == 0.5
    assert schedule.value(100) == 0.0
    assert schedule.value(200) == 0.0


def test_piecewise_schedule() -> None:
    schedule = PiecewiseSchedule(((0, 1.0), (10, 0.5), (20, 0.25)))
    assert schedule.value(0) == 1.0
    assert schedule.value(9) == 1.0
    assert schedule.value(10) == 0.5
    assert schedule.value(19) == 0.5
    assert schedule.value(20) == 0.25
