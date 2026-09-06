"""Right-hand-rule wall follower. Always solves a perfect maze."""

from __future__ import annotations

from mazegame.base_explorer import BaseExplorer
from mazegame.direction import Direction
from itertools import cycle, islice


class WallFollowerExplorer(BaseExplorer):

    def solve(self) -> List[Cell]:
        dirs = cycle((Direction.RIGHT, Direction.UP, Direction.LEFT, Direction.DOWN))
        path : List[Cell] = [self.get_cell()]

        while not self.is_at_goal():
            for d in dirs:
                if self.can_move(d):
                    self.move(d)
                    path.append(d)
                    break
            # A right turn relative to the current heading is 3 left turns away,
            # advance by 2 so that the righthand dir is the next in the cycle
            *islice(dirs, 2),

        path.reverse()
        return path


