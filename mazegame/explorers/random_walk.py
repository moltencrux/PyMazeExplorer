"""Simplest (and usually worst) strategy: random direction every step."""

from __future__ import annotations

import random

from mazegame.base_explorer import BaseExplorer


class RandomWalkExplorer(BaseExplorer):
    def __init__(self) -> None:
        super().__init__()
        self._rand = random.Random()

    def solve(self) -> None:
        while not self.is_at_goal():
            choice = self._rand.randrange(4)
            if choice == 0:
                self.move_up()
            elif choice == 1:
                self.move_down()
            elif choice == 2:
                self.move_left()
            else:
                self.move_right()
