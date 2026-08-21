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
  - Call teleport(cell) to jump instantly to a previously visited cell.
    Returns True if the cell has been visited, False otherwise.
  - Call has_visited(cell) to check whether a cell is in the visited set.
  - Call is_at_goal() to check if you've reached the goal.
  - Call get_hint() for a heuristic value (straight-line distance from your
    current square to the goal).
  - Call get_row() / get_col() to see where you currently are.
  - Call mark_explored(cell) to highlight a cell in the exploration overlay
    (optional; the engine already marks cells you step onto).
  - Call set_show_sprite(False) to hide the red agent dot (useful for
    frontier-style search where the highlight carries the visual).

The camera automatically frames open leaves (visited cells that still have
an unvisited open neighbour). Students do not need to manage that.

You do NOT get direct access to the maze's wall layout. The only way to find
out what's around you is to try moving (or call the can_move* helpers).
Teleport only works for cells you have already stepped onto via move_*.
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

    def teleport(self, cell: Cell) -> bool:
        """
        Jump to a previously visited cell.
        Returns True if the teleport succeeded (cell was visited, or is the
        current cell). Returns False if the cell has never been stepped on.
        """
        return self._require_engine().attempt_teleport(cell)

    def has_visited(self, cell: Cell) -> bool:
        """True if the explorer has previously stepped onto this cell."""
        return self._require_engine().has_visited(cell)

    def get_hint(self) -> float:
        """Straight-line (Euclidean) distance to the goal. Smaller is closer."""
        return self._require_engine().get_hint()

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

    def mark_explored(self, cell: Optional[Cell] = None) -> None:
        """
        Highlight a cell in the exploration overlay.
        If cell is None, marks the explorer's current position.
        """
        eng = self._require_engine()
        if cell is None:
            cell = Cell(eng.get_row(), eng.get_col())
        eng.mark_explored(cell)

    def mark_explored_many(self, cells: List[Cell]) -> None:
        """Mark several cells at once."""
        self._require_engine().mark_explored_many(cells)

    def set_show_sprite(self, show: bool) -> None:
        """
        Show or hide the red agent dot.
        Frontier-style algorithms (A*, Dijkstra, …) often hide it and rely
        on the exploration highlight instead.
        """
        self._require_engine().set_show_sprite(show)
