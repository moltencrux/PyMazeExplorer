"""Simplest (and usually worst) strategy: random direction every step."""

from __future__ import annotations

import random
from typing import Dict, List

from mazegame.direction import Direction
from mazegame.base_explorer import BaseExplorer


class RandomWalkExplorer(BaseExplorer):
    def __init__(self) -> None:
        super().__init__()
        self._rand = random.Random()

    def solve(self) -> List[Cell]:
        came_from: Dict[Cell, Cell] = {}

        while not self.is_at_goal():
            parent = self.get_cell()
            dir_ = self._rand.choice(list(Direction))
            if self.can_move(dir_):
                self.move(dir_)
                came_from.setdefault(self.get_cell(), parent)

        return self._reconstruct_path(came_from, self.get_cell())

    @staticmethod
    def _reconstruct_path(came_from: Dict[Cell, Cell], goal: Cell) -> List[Cell]:
        path = [goal]
        cur = goal
        while cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return path
