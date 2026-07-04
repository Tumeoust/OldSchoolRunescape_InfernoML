from __future__ import annotations

from dataclasses import dataclass


class Schedule:
    def value(self, step: int) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantSchedule(Schedule):
    constant: float

    def value(self, step: int) -> float:
        return self.constant


@dataclass(frozen=True)
class LinearSchedule(Schedule):
    initial: float
    final: float
    duration: int

    def value(self, step: int) -> float:
        if self.duration <= 0:
            return self.final
        t = max(0.0, min(1.0, step / float(self.duration)))
        return self.initial + (self.final - self.initial) * t


@dataclass(frozen=True)
class PiecewiseSchedule(Schedule):
    segments: tuple[tuple[int, float], ...]

    def value(self, step: int) -> float:
        if not self.segments:
            raise ValueError("PiecewiseSchedule requires at least one segment")
        value = self.segments[0][1]
        for start_step, segment_value in self.segments:
            if step < start_step:
                break
            value = segment_value
        return value
