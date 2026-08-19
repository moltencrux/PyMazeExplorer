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
  - Call is_at_goal() to check if you've reached the goal.
  - Call get_hint() for a heuristic value (straight-line distance from your
    current square to the goal).
  - Call get_row() / get_col() to see where you currently are.

You do NOT get direct access to the maze's wall layout. The only way to find
out what's around you is to try moving (or call the can_move* helpers).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

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
