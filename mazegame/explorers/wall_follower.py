"""Right-hand-rule wall follower. Always solves a perfect maze."""

from __future__ import annotations

from mazegame.base_explorer import BaseExplorer


class WallFollowerExplorer(BaseExplorer):
    # Facing: 0 = up, 1 = right, 2 = down, 3 = left
    def __init__(self) -> None:
        super().__init__()
        self.facing = 1  # start facing right

    def solve(self) -> None:
        while not self.is_at_goal():
            right_of = (self.facing + 1) % 4
            if self._try(right_of):
                self.facing = right_of
                continue
            if self._try(self.facing):
                continue
            left_of = (self.facing + 3) % 4
            if self._try(left_of):
                self.facing = left_of
                continue
            # Dead end — only way is back
            back = (self.facing + 2) % 4
            self._try(back)
            self.facing = back

    def _try(self, direction: int) -> bool:
        if direction == 0:
            return self.move_up()
        if direction == 1:
            return self.move_right()
        if direction == 2:
            return self.move_down()
        if direction == 3:
            return self.move_left()
        raise ValueError(f"bad direction {direction}")
