"""
Tracks the set of visited cells and the open frontier (visited cells that
still have at least one open, unvisited neighbour).

Two instances are used:
  - logical  (worker / algorithm thread)
  - visual   (main / render thread) — advanced only when an animation starts
"""

from __future__ import annotations

from typing import Set

from .cell import Cell
from .direction import Direction
from .maze import Maze


class ExplorationState:
    def __init__(self, maze: Maze, start: Cell) -> None:
        self.maze = maze
        self.visited: Set[Cell] = {start}
        self.frontier: Set[Cell] = set()
        if self._has_open_unvisited_neighbour(start):
            self.frontier.add(start)

    def is_visited(self, cell: Cell) -> bool:
        return cell in self.visited

    def get_frontier(self) -> Set[Cell]:
        return self.frontier

    def can_visit(self, cell: Cell) -> bool:
        """
        True if visiting a *cell* is allowed: already visited, or open
        and orthogonally adjacent to at least one visited cell.
        """
        if cell in self.visited:
            return True
        if not self.maze.is_open_cell(cell):
            return False
        for d in Direction:
            if cell.moved(d) in self.visited:
                return True
        return False

    def visit(self, cell: Cell) -> None:
        """
        Mark *cell* visited and maintain the frontier incrementally.

        Caller is responsible for ensuring this is a legal discovery
        (e.g. an adjacent open cell reached by a normal move).
        """
        if not self.can_visit(cell):
            return

        if cell in self.visited:
            return

        self.visited.add(cell)

        if self._has_open_unvisited_neighbour(cell):
            self.frontier.add(cell)

        # Neighbours that previously had this cell as their only remaining
        # unvisited open neighbour may now be dead-ends.
        for d in Direction:
            neighbour = cell.moved(d)
            if neighbour in self.visited and not self._has_open_unvisited_neighbour(neighbour):
                self.frontier.discard(neighbour)

    def _has_open_unvisited_neighbour(self, cell: Cell) -> bool:
        for d in Direction:
            n = cell.moved(d)
            if self.maze.is_open_cell(n) and n not in self.visited:
                return True
        return False

    def reset(self, start: Cell) -> None:
        """Reset to a fresh state with only *start* visited."""
        self.visited = {start}
        self.frontier = set()
        if self._has_open_unvisited_neighbour(start):
            self.frontier.add(start)
