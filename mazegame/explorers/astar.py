"""
A* search explorer.

Uses teleport() to jump between already-visited cells when expanding the
frontier, and move() only to step into brand-new neighbours.

Open leaves for the camera are derived automatically by the engine from the
visited set (any visited cell that still has an unvisited open neighbour).
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Set, Tuple

from mazegame.base_explorer import BaseExplorer
from mazegame.cell import Cell
from mazegame.direction import Direction


DIRS = (Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT)


class AStarExplorer(BaseExplorer):
    def solve(self) -> None:
        # Rely on the exploration highlight instead of the hopping red dot.
        self.set_show_sprite(False)

        start = self.get_cell()

        # (f, g, tie, cell) — lower f first; tie-break on g then insertion order
        counter = 0
        open_heap: List[Tuple[float, float, int, Cell]] = []
        heapq.heappush(open_heap, (0.0, 0.0, counter, start))

        came_from: Dict[Cell, Cell] = {}
        g_score: Dict[Cell, float] = {start: 0.0}
        closed: Set[Cell] = set()
        in_open: Set[Cell] = {start}

        # self.mark_explored(start)

        while open_heap:
            _f, g, _tie, cell = heapq.heappop(open_heap)
            if cell not in in_open:
                continue  # stale heap entry
            in_open.discard(cell)

            if cell in closed:
                continue
            closed.add(cell)

            # Jump to this node so can_move / get_hint are valid here.
            if not self.teleport(cell):
                continue

            # self.mark_explored(cell)

            if self.is_at_goal():
                return

            # Exact heuristic from the cell we are now standing on.
            h_here = self.get_hint()
            r, c = self.get_row(), self.get_col()

            for direction in DIRS:
                # Must be standing on `cell` to probe; teleport back if needed.
                if self.get_cell() != cell:
                    if not self.teleport(cell):
                        break

                if not self.can_move(direction):
                    continue

                neighbour = Cell(r + direction.d_row, c + direction.d_col)
                if neighbour in closed:
                    continue

                tentative_g = g_score[cell] + 1.0
                if neighbour in g_score and tentative_g >= g_score[neighbour]:
                    continue

                # First time we reach this neighbour: step into it so it
                # becomes visited (required for future teleports).
                if not self.has_visited(neighbour):
                    if not self.move(direction):
                        continue
                    came_from[neighbour] = cell
                    g_score[neighbour] = tentative_g
                    h_n = max(0.0, h_here - 1.0)
                    f_n = tentative_g + h_n
                    counter += 1
                    heapq.heappush(open_heap, (f_n, tentative_g, counter, neighbour))
                    in_open.add(neighbour)
                    # self.mark_explored(neighbour)
                    self.teleport(cell)
                else:
                    # Already visited — just improve the score / parent.
                    came_from[neighbour] = cell
                    g_score[neighbour] = tentative_g
                    h_n = max(0.0, h_here - 1.0)
                    f_n = tentative_g + h_n
                    counter += 1
                    heapq.heappush(open_heap, (f_n, tentative_g, counter, neighbour))
                    in_open.add(neighbour)
                    # self.mark_explored(neighbour)
