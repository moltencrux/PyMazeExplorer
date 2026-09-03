"""
BaseExplorer is the class students subclass to implement a maze-solving
strategy (depth-first search, wall-following, A*, etc.).

How it works:
  - Override solve(). This method is the algorithm's entry point. It runs on
    its own background thread, so you are free to use a loop, recursion, a
    stack, a queue — whatever your algorithm needs — without freezing the
    display.
  - Call move_up() / move_down() / move_left() / move_right() to attempt to
    move one square. Each call blocks until the move animation finishes and
    returns True if the move succeeded, or False if it was blocked by a wall
    (or the edge of the maze). A False return does NOT move you.
  - Call can_move_up() / can_move_down() / can_move_left() / can_move_right()
    to test whether a move would succeed without actually performing it (no
    animation, no path change, no move count).
  - Call visit(cell) to jump to a reachable cell: any previously visited
    cell, or an open cell orthogonally adjacent to a visited cell. First
    visit onto a new cell marks it visited. Returns True on success.
  - Call can_visit(cell) to test whether visit(cell) would succeed
    (no animation, no state change).
  - Call has_visited(cell) to check whether a cell is in the visited set.
  - Call is_at_goal() to check if you've reached the goal.
  - Call get_hint() / get_hint(cell) for a heuristic value (Manhattan
    distance from your current square, or from an arbitrary cell, to the goal).
  - Call get_row() / get_col() to see where you currently are.
  - Call mark_explored(cell) to highlight a cell in the exploration overlay
    (optional; the engine already marks cells you step onto).
  - Call set_show_sprite(False) to hide the red agent dot (useful for
    frontier-style search where the highlight carries the visual).

The camera automatically frames open leaves (cells that have been *visually*
revealed and still have an unvisited open neighbour). Students do not need
to manage that; the visual frontier lags the algorithm so it stays in sync
with the animation.

You do NOT get direct access to the maze's wall layout. The only way to find
out what's around you is to try moving (or call the can_move* helpers).
visit works for visited cells and for open cells next to the visited set.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

from .cell import Cell
from .direction import Direction

if TYPE_CHECKING:
    from .maze_engine import MazeEngine


class BaseExplorer(ABC):
    def __init__(self) -> None:
        self._engine: Optional[MazeEngine] = None

    def bind(self, engine: MazeEngine) -> None:
        """Called internally by the engine — do not call this yourself."""
        self._engine = engine

    @abstractmethod
    def solve(self) -> None:
        """Your maze-solving algorithm goes here."""
        ...

    def _require_engine(self) -> MazeEngine:
        if self._engine is None:
            raise RuntimeError("Explorer is not bound to an engine")
        return self._engine

    def move(self, direction: Direction) -> bool:
        return self._require_engine().attempt_move(direction)

    def move_up(self) -> bool:
        return self.move(Direction.UP)

    def move_down(self) -> bool:
        return self.move(Direction.DOWN)

    def move_left(self) -> bool:
        return self.move(Direction.LEFT)

    def move_right(self) -> bool:
        return self.move(Direction.RIGHT)

    def can_move(self, direction: Direction) -> bool:
        return self._require_engine().can_move(direction)

    def can_move_up(self) -> bool:
        return self.can_move(Direction.UP)

    def can_move_down(self) -> bool:
        return self.can_move(Direction.DOWN)

    def can_move_left(self) -> bool:
        return self.can_move(Direction.LEFT)

    def can_move_right(self) -> bool:
        return self.can_move(Direction.RIGHT)

    def visit(self, cell: Cell) -> bool:
        """
        Jump to a reachable cell: visited, or open and adjacent to visited.
        First visit onto a new cell marks it visited.
        Returns True on success (including a no-op when already there).
        """
        return self._require_engine().attempt_visit(cell)

    def can_visit(self, cell: Cell) -> bool:
        """True if visit(cell) would succeed (no animation / no state change)."""
        return self._require_engine().can_visit(cell)

    def has_visited(self, cell: Cell) -> bool:
        """True if the explorer has previously stepped onto this cell."""
        return self._require_engine().has_visited(cell)

    def get_hint(self, cell: Optional[Cell] = None) -> float:
        """
        Manhattan distance to the goal. Smaller is closer.
        If *cell* is given, returns the hint for that cell without moving;
        otherwise uses the explorer's current logical position.
        """
        return self._require_engine().get_hint(cell)

    def is_at_goal(self) -> bool:
        return self._require_engine().is_at_goal()

    def get_row(self) -> int:
        return self._require_engine().get_row()

    def get_col(self) -> int:
        return self._require_engine().get_col()

    def get_cell(self) -> Cell:
        return Cell(self.get_row(), self.get_col())

    def get_move_count(self) -> int:
        return self._require_engine().get_move_count()

    def set_show_sprite(self, show: bool) -> None:
        """
        Show or hide the red agent dot.
        Frontier-style algorithms (A*, Dijkstra, …) often hide it and rely
        on the exploration highlight instead.
        """
        self._require_engine().set_show_sprite(show)
