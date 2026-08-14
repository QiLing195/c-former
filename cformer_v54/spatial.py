from __future__ import annotations

from dataclasses import dataclass


def _rotate(point: tuple[float, float], quarter_turns: int) -> tuple[float, float]:
    x, y = point
    turns = quarter_turns % 4
    if turns == 0:
        return x, y
    if turns == 1:
        return -y, x
    if turns == 2:
        return -x, -y
    return y, -x


@dataclass(frozen=True)
class FrameTransform:
    frame_id: int
    valid_from: int
    valid_to: int
    quarter_turns: int = 0
    translate_x: float = 0.0
    translate_y: float = 0.0

    def to_global(self, point: tuple[float, float]) -> tuple[float, float]:
        x, y = _rotate(point, self.quarter_turns)
        return x + self.translate_x, y + self.translate_y

    def from_global(self, point: tuple[float, float]) -> tuple[float, float]:
        shifted = (point[0] - self.translate_x, point[1] - self.translate_y)
        return _rotate(shifted, -self.quarter_turns)


class FrameRegistry:
    def __init__(self) -> None:
        self._frames: dict[int, list[FrameTransform]] = {
            0: [FrameTransform(0, -2**31, 2**31 - 1)]
        }

    def add(self, transform: FrameTransform) -> None:
        self._frames.setdefault(transform.frame_id, []).append(transform)

    def resolve(self, frame_id: int, query_time: int) -> FrameTransform | None:
        active = [
            transform
            for transform in self._frames.get(frame_id, ())
            if transform.valid_from <= query_time < transform.valid_to
        ]
        return max(active, key=lambda transform: transform.valid_from) if active else None

    def to_global(
        self, frame_id: int, point: tuple[float, float], query_time: int
    ) -> tuple[float, float] | None:
        transform = self.resolve(frame_id, query_time)
        return transform.to_global(point) if transform is not None else None

    def from_global(
        self, frame_id: int, point: tuple[float, float], query_time: int
    ) -> tuple[float, float] | None:
        transform = self.resolve(frame_id, query_time)
        return transform.from_global(point) if transform is not None else None
