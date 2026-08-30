"""
A* search explorer.

Uses visit() to jump onto open-set cells when they are expanded. Visit
is allowed for any visited cell and for any open cell adjacent to the visited
set, so neighbours can be enqueued without a pre-visit move().

heapq has no decrease-key, so improved paths push a new entry; stale entries
are ignored on pop by comparing against the best-known g_score.

Open leaves for the camera are derived automatically by the engine from the
*visual* visited set (any cell that has been animated and still has an
unvisited open neighbour). The visual frontier lags the algorithm so the
camera stays consistent with what has been shown on screen.
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
        h_start = self.get_hint(start)
        heapq.heappush(open_heap, (h_start, 0.0, counter, start))

        g_score: Dict[Cell, float] = {start: 0.0}
        closed: Set[Cell] = set()

        while open_heap:
            _f, g, _tie, cell = heapq.heappop(open_heap)
            # Stale entry: a better path to this cell was found after enqueue.
            if g > g_score.get(cell, float("inf")):
                continue
            if cell in closed:
                continue
            closed.add(cell)

            # Expand from this node; visit marks it visited if needed.
            if not self.visit(cell):
                continue

            if self.is_at_goal():
                return

            for direction in DIRS:
                # Must be standing on `cell` to probe; visit back if needed.
                if self.get_cell() != cell:
                    if not self.visit(cell):
                        break

                if not self.can_move(direction):
                    continue

                neighbour = cell.moved(direction)
                if neighbour in closed:
                    continue

                tentative_g = g + 1.0
                if tentative_g >= g_score.get(neighbour, float("inf")):
                    continue

                # Enqueue without pre-visiting; visit will succeed once this
                # parent (or another path) has made the neighbour reachable.
                # Old heap entries for neighbour become stale via the g-check.
                g_score[neighbour] = tentative_g
                h_n = self.get_hint(neighbour)
                f_n = tentative_g + h_n
                counter += 1
                heapq.heappush(open_heap, (f_n, tentative_g, counter, neighbour))
