"""
Iterative depth-first search.

Relative coordinate system: the cell where solve() begins is (0, 0).
Physical movement is performed by walking the difference between the path
we just finished and the path we are about to explore (backtrack to the
common prefix, then walk forward).
"""

from __future__ import annotations

from collections import deque
from typing import List, Set

from mazegame.base_explorer import BaseExplorer
from mazegame.cell import Cell
from mazegame.direction import Direction


DIRS = (Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT)


class DFSExplorer(BaseExplorer):
    def solve(self) -> None:
        start = Cell(0, 0)  # relative origin
        visited: Set[Cell] = {start}

        queue: deque[List[Cell]] = deque()
        initial: List[Cell] = [start]
        queue.append(initial)  # front = DFS (appendleft would also work with pop)

        current_path = initial

        while queue:
            path = queue.pop()  # LIFO → DFS

            self._walk_to(current_path, path)
            current_path = path

            if self.is_at_goal():
                return

            terminal = path[-1]
            for direction in DIRS:
                if not self.can_move(direction):
                    continue
                neighbour = terminal.moved(direction)
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                new_path = path + [neighbour]
                queue.append(new_path)

    def _walk_to(self, from_path: List[Cell], to_path: List[Cell]) -> None:
        # Longest common prefix
        i = 0
        limit = min(len(from_path), len(to_path))
        while i < limit and from_path[i] == to_path[i]:
            i += 1

        # Backtrack
        for j in range(len(from_path) - 1, i - 1, -1):
            here = from_path[j]
            prev = from_path[j - 1]
            self.move(self._direction_between(here, prev))

        # Walk forward
        for j in range(i, len(to_path)):
            prev = to_path[j - 1]
            nxt = to_path[j]
            self.move(self._direction_between(prev, nxt))

    @staticmethod
    def _direction_between(a: Cell, b: Cell) -> Direction:
        dr = b.row - a.row
        dc = b.col - a.col
        if dr == -1 and dc == 0:
            return Direction.UP
        if dr == 1 and dc == 0:
            return Direction.DOWN
        if dr == 0 and dc == -1:
            return Direction.LEFT
        if dr == 0 and dc == 1:
            return Direction.RIGHT
        raise ValueError(f"Cells are not adjacent: {a} → {b}")
