"""
Iterative depth-first search.

Uses visit() to jump onto each cell when it is expanded. neighbors are
enqueued without a pre-visit; visit succeeds once the parent has been
expanded (making the neighbor reachable).

Returns the reconstructed start→goal path from solve() so path length is
correct (visit alone does not maintain an engine trail).
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Set

from mazegame.base_explorer import BaseExplorer
from mazegame.cell import Cell
from mazegame.direction import Direction


DIRS = (Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT)


class DFSExplorer(BaseExplorer):
    def solve(self) -> Optional[Sequence[Cell]]:
        start = self.get_cell()
        visited: Set[Cell] = {start}
        came_from: Dict[Cell, Cell] = {}

        stack: deque[Cell] = deque()
        stack.append(start)

        while stack:
            cell = stack.pop()  # LIFO → DFS
            if not self.visit(cell):
                continue
            visited.add(cell)

            if self.is_at_goal():
                return self._reconstruct_path(came_from, cell)

            for direction in DIRS:
                if not self.can_move(direction):
                    continue
                neighbor = cell.moved(direction)
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                came_from[neighbor] = cell
                stack.append(neighbor)

        return None

    @staticmethod
    def _reconstruct_path(came_from: Dict[Cell, Cell], goal: Cell) -> List[Cell]:
        path = [goal]
        cur = goal
        while cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return path
