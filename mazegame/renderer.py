"""
Pygame renderer for the maze.

Cells on even indices (the "graph lines") are drawn thin (walls / openings);
cells on odd indices (the logical rooms) are drawn thicker. This matches the
dev/var-width-walls branch of the Java app.
"""

from __future__ import annotations

import math
import threading
from typing import List, Optional, Tuple

import pygame

from .cell import Cell
from .direction import Direction
from .maze import Maze

# Pixel size of logical room cells (odd row/col indices).
ROOM_SIZE = 26
# Pixel thickness of wall / opening cells (even row/col indices).
WALL_THICKNESS = 4

MAX_ANIMATION_STEPS = 20

BG = (0xF5, 0xF5, 0xF5)
WALL_COLOR = (0x00, 0xFF, 0xFF)  # cyan walls (matches Java)
OPEN_COLOR = (0x00, 0x00, 0x00)
TRAIL_COLOR = (0x9F, 0xD8, 0xFF)
START_COLOR = (0x4C, 0xAF, 0x50)
GOAL_COLOR = (0xFF, 0xB3, 0x00)
SPRITE_COLOR = (0xE5, 0x39, 0x35)
WHITE = (255, 255, 255)


def _ease_in_out(t: float) -> float:
    if t < 0.5:
        return 2 * t * t
    return 1 - (-2 * t + 2) ** 2 / 2


class MazeRenderer:
    def __init__(self) -> None:
        self.maze: Optional[Maze] = None
        self.path: List[Cell] = []
        self.col_x: List[int] = []
        self.row_y: List[int] = []
        self.total_width = 0
        self.total_height = 0
        self.background: Optional[pygame.Surface] = None
        self.trail_surface: Optional[pygame.Surface] = None
        self.trail_path_size = -1
        self.sprite_x = 0.0
        self.sprite_y = 0.0
        self._anim_duration_ms = 150
        self._animating = False
        self._anim_done: Optional[threading.Event] = None
        self._anim_kind: Optional[str] = None
        self._anim_from: Tuple[float, float] = (0.0, 0.0)
        self._anim_to: Tuple[float, float] = (0.0, 0.0)
        self._anim_steps = 1
        self._anim_step = 0
        self._anim_per_step_ms = 1
        self._anim_elapsed = 0.0
        self._bump_center: Tuple[float, float] = (0.0, 0.0)
        self._bump_target: Tuple[float, float] = (0.0, 0.0)
        self._bump_phase = 0  # 0 = out, 1 = back

    def set_animation_duration_ms(self, ms: int) -> None:
        self._anim_duration_ms = max(0, ms)

    def is_animating(self) -> bool:
        return self._animating

    def set_engine_state(
        self, maze: Maze, path: List[Cell], start_cell: Cell
    ) -> None:
        self.maze = maze
        self.path = list(path)
        self._rebuild_geometry()
        self._rebuild_background()
        self.invalidate_trail()
        self._reset_sprite(start_cell)
        # Cancel any in-flight animation.
        if self._anim_done is not None:
            self._anim_done.set()
        self._animating = False
        self._anim_done = None

    def invalidate_trail(self) -> None:
        self.trail_surface = None
        self.trail_path_size = -1

    def update_path(self, path: List[Cell]) -> None:
        self.path = list(path)
        self.invalidate_trail()

    # ------------------------------------------------------------------
    # Geometry (variable thickness)
    # ------------------------------------------------------------------

    def _rebuild_geometry(self) -> None:
        assert self.maze is not None
        cols, rows = self.maze.cols, self.maze.rows
        self.col_x = [0] * (cols + 1)
        self.row_y = [0] * (rows + 1)
        for c in range(cols):
            w = WALL_THICKNESS if c % 2 == 0 else ROOM_SIZE
            self.col_x[c + 1] = self.col_x[c] + w
        self.total_width = self.col_x[cols]
        for r in range(rows):
            h = WALL_THICKNESS if r % 2 == 0 else ROOM_SIZE
            self.row_y[r + 1] = self.row_y[r] + h
        self.total_height = self.row_y[rows]

    def _cell_width(self, col: int) -> int:
        return self.col_x[col + 1] - self.col_x[col]

    def _cell_height(self, row: int) -> int:
        return self.row_y[row + 1] - self.row_y[row]

    def _cell_center(self, cell: Cell) -> Tuple[float, float]:
        x = self.col_x[cell.col] + self._cell_width(cell.col) / 2.0
        y = self.row_y[cell.row] + self._cell_height(cell.row) / 2.0
        return x, y

    def _rebuild_background(self) -> None:
        assert self.maze is not None
        surf = pygame.Surface((self.total_width, self.total_height))
        surf.fill(BG)
        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                open_ = self.maze.is_open(r, c)
                color = OPEN_COLOR if open_ else WALL_COLOR
                rect = pygame.Rect(
                    self.col_x[c],
                    self.row_y[r],
                    self._cell_width(c),
                    self._cell_height(r),
                )
                pygame.draw.rect(surf, color, rect)
        self._draw_marker(surf, self.maze.start, START_COLOR, "S")
        self._draw_marker(surf, self.maze.goal, GOAL_COLOR, "G")
        self.background = surf

    def _draw_marker(
        self, surf: pygame.Surface, cell: Cell, color: Tuple[int, int, int], label: str
    ) -> None:
        pad = max(2, ROOM_SIZE // 8)
        x = self.col_x[cell.col] + pad
        y = self.row_y[cell.row] + pad
        w = self._cell_width(cell.col) - 2 * pad
        h = self._cell_height(cell.row) - 2 * pad
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surf, color, rect, border_radius=6)
        font = pygame.font.SysFont("sans", max(12, ROOM_SIZE // 2), bold=True)
        text = font.render(label, True, WHITE)
        tx = self.col_x[cell.col] + (self._cell_width(cell.col) - text.get_width()) // 2
        ty = self.row_y[cell.row] + (self._cell_height(cell.row) - text.get_height()) // 2
        surf.blit(text, (tx, ty))

    def _ensure_trail(self) -> None:
        if self.trail_surface is not None and self.trail_path_size == len(self.path):
            return
        surf = pygame.Surface((self.total_width, self.total_height), pygame.SRCALPHA)
        pad = max(2, ROOM_SIZE // 6)
        for idx, c in enumerate(self.path):
            if idx == 0:
                continue  # skip current (head) cell
            rect = pygame.Rect(
                self.col_x[c.col] + pad,
                self.row_y[c.row] + pad,
                self._cell_width(c.col) - 2 * pad,
                self._cell_height(c.row) - 2 * pad,
            )
            pygame.draw.rect(surf, TRAIL_COLOR, rect, border_radius=6)
        self.trail_surface = surf
        self.trail_path_size = len(self.path)

    def _reset_sprite(self, cell: Cell) -> None:
        self.sprite_x, self.sprite_y = self._cell_center(cell)
        self.invalidate_trail()

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def _steps_for_duration(self) -> int:
        if self._anim_duration_ms <= 4:
            return 1
        steps = self._anim_duration_ms // 4
        return max(1, min(MAX_ANIMATION_STEPS, steps))

    def animate_move(
        self, from_cell: Cell, to_cell: Cell, done: threading.Event
    ) -> None:
        to_px = self._cell_center(to_cell)
        steps = self._steps_for_duration()
        if steps <= 1:
            self.sprite_x, self.sprite_y = to_px
            self.path = self.path  # already updated by engine
            self.invalidate_trail()
            done.set()
            return
        from_px = self._cell_center(from_cell)
        self._anim_kind = "move"
        self._anim_from = from_px
        self._anim_to = to_px
        self._anim_steps = steps
        self._anim_step = 0
        self._anim_elapsed = 0.0
        self._anim_per_step_ms = max(1, self._anim_duration_ms // steps)
        self._anim_done = done
        self._animating = True

    def animate_bump(
        self, at: Cell, direction: Direction, done: threading.Event
    ) -> None:
        steps = self._steps_for_duration()
        if steps <= 1:
            done.set()
            return
        center = self._cell_center(at)
        nudge = ROOM_SIZE * 0.28
        target = (
            center[0] + direction.d_col * nudge,
            center[1] + direction.d_row * nudge,
        )
        total_steps = max(2, steps // 2)
        self._anim_kind = "bump"
        self._bump_center = center
        self._bump_target = target
        self._bump_phase = 0
        self._anim_steps = total_steps
        self._anim_step = 0
        self._anim_elapsed = 0.0
        self._anim_per_step_ms = max(1, self._anim_duration_ms // total_steps // 2)
        self._anim_done = done
        self._animating = True

    def tick(self, dt_ms: float) -> None:
        """Advance animation by dt_ms milliseconds. Call every frame."""
        if not self._animating or self._anim_done is None:
            return
        self._anim_elapsed += dt_ms
        while self._anim_elapsed >= self._anim_per_step_ms:
            self._anim_elapsed -= self._anim_per_step_ms
            self._anim_step += 1
            if self._anim_kind == "move":
                t = min(1.0, self._anim_step / self._anim_steps)
                t = _ease_in_out(t)
                fx, fy = self._anim_from
                tx, ty = self._anim_to
                self.sprite_x = fx + (tx - fx) * t
                self.sprite_y = fy + (ty - fy) * t
                if self._anim_step >= self._anim_steps:
                    self.sprite_x, self.sprite_y = tx, ty
                    self._finish_anim()
            elif self._anim_kind == "bump":
                t = self._anim_step / self._anim_steps
                if self._bump_phase == 0:
                    # out
                    t = min(1.0, t)
                    cx, cy = self._bump_center
                    tx, ty = self._bump_target
                    self.sprite_x = cx + (tx - cx) * t
                    self.sprite_y = cy + (ty - cy) * t
                    if self._anim_step >= self._anim_steps:
                        self._bump_phase = 1
                        self._anim_step = 0
                else:
                    # back
                    t = min(1.0, t)
                    cx, cy = self._bump_center
                    tx, ty = self._bump_target
                    self.sprite_x = tx + (cx - tx) * t
                    self.sprite_y = ty + (cy - ty) * t
                    if self._anim_step >= self._anim_steps:
                        self.sprite_x, self.sprite_y = cx, cy
                        self._finish_anim()

    def _finish_anim(self) -> None:
        self._animating = False
        if self._anim_done is not None:
            self._anim_done.set()
            self._anim_done = None
        self._anim_kind = None

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface, offset: Tuple[int, int] = (0, 0)) -> None:
        if self.maze is None or self.background is None:
            return
        ox, oy = offset
        surface.blit(self.background, (ox, oy))
        self._ensure_trail()
        if self.trail_surface is not None:
            surface.blit(self.trail_surface, (ox, oy))
        # Sprite
        radius = ROOM_SIZE * 0.32
        cx = int(self.sprite_x + ox)
        cy = int(self.sprite_y + oy)
        pygame.draw.circle(surface, SPRITE_COLOR, (cx, cy), int(radius))
        pygame.draw.circle(surface, WHITE, (cx, cy), int(radius), 2)
