"""The four cardinal directions an Explorer can move.

Row increases downward, column increases rightward (screen/grid convention).
"""

from __future__ import annotations

from enum import Enum


class Direction(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

    @property
    def d_row(self) -> int:
        return self.value[0]

    @property
    def d_col(self) -> int:
        return self.value[1]
