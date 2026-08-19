"""Immutable (row, col) coordinate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .direction import Direction


@dataclass(frozen=True, slots=True)
class Cell:
    row: int
    col: int

    def moved(self, direction: Direction) -> Cell:
        return Cell(self.row + direction.d_row, self.col + direction.d_col)

    def __str__(self) -> str:
        return f"({self.row}, {self.col})"
