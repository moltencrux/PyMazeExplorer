"""
Pygame renderer for the maze.

Cells on even indices (the "graph lines") are drawn thin (walls / openings);
cells on odd indices (the logical rooms) are drawn thicker. This matches the
dev/var-width-walls branch of the Java app.

Supports:
  - Camera (pan + soft zoom driven by recent exploration)
  - Fading exploration highlight (fresh → dark grey, never pure unexplored)
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Set, Tuple

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

# Exploration overlay
HIGHLIGHT_COLOR = (0xFF, 0xC1, 0x07)  # warm amber for freshly explored
EXPLORED_GREY = (0x2A, 0x2C, 0x32)  # dark grey – still distinct from pure black
FADE_DURATION_S = 4.0  # seconds from full highlight → grey
RECENT_N = 48  # how many most-recent cells drive the camera
# Critically-damped spring: higher omega = snappier, zeta=1 is critical damping
CAMERA_OMEGA = 3.5          # natural frequency (rad/s) for pan
CAMERA_ZOOM_OMEGA = 2.5     # slightly softer zoom response
MIN_VIEW_CELLS = 18  # minimum focus span in logical rooms (limits how tight we zoom)
PADDING_CELLS = 4  # extra cells of padding around frontier leaves
MAX_ZOOM = 1.25  # never zoom in tighter than this (1.0 = 1 world px → 1 screen px)


def _ease_in_out(t: float) -> float:
    if t < 0.5:
        return 2 * t * t
    return 1 - (-2 * t + 2) ** 2 / 2


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_color(
    c0: Tuple[int, int, int], c1: Tuple[int, int, int], t: float
) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        int(_lerp(c0[0], c1[0], t)),
        int(_lerp(c0[1], c1[1], t)),
        int(_lerp(c0[2], c1[2], t)),
    )


class MazeRenderer:
    def __init__(self) -> None:
        self.maze: Optional[Maze] = None
        self.path: List[Cell] = []
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

        # --- Exploration overlay ---
        # cell -> monotonic time when last marked (for fade timing)
        self._explored: Dict[Cell, float] = {}
        # Cells still transitioning highlight → grey (per-frame updates only)
        self._fading: Set[Cell] = set()
        # Cached world-space overlay; fully-faded cells painted once and left
        self.explore_surface: Optional[pygame.Surface] = None
        # Last scaled visible slices (reuse when camera/zoom unchanged)
        self._scale_cache_key: Optional[tuple] = None
        self._bg_scaled: Optional[pygame.Surface] = None
        self._exp_scaled: Optional[pygame.Surface] = None
        self._trail_scaled: Optional[pygame.Surface] = None
        # ordered recent history (most recent last) — fallback for camera
        self._recent: Deque[Cell] = deque(maxlen=RECENT_N * 2)
        # Open leaves / frontier: cells that still have expansion work left.
        # Camera prefers these over raw recency when the set is non-empty.
        self._frontier: Set[Cell] = set()
        # Worker thread marks cells; main thread reads them for camera/draw.
        self._explore_lock = threading.Lock()

        # --- Camera: view rectangle edges (world pixels), each with its own velocity.
        # cam_x/cam_y/zoom are derived from the view rect for drawing.
        self.view_left = 0.0
        self.view_right = 1.0
        self.view_top = 0.0
        self.view_bottom = 1.0
        self._vel_left = 0.0
        self._vel_right = 0.0
        self._vel_top = 0.0
        self._vel_bottom = 0.0
        self.cam_x = 0.0
        self.cam_y = 0.0
        self.zoom = 1.0  # derived: 1.0 = 1 world-pixel → 1 screen-pixel
        # Manual pan offset (mouse drag) – added on top of the auto camera
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._dragging = False
        self._drag_last: Optional[Tuple[int, int]] = None
        # Red agent dot; frontier-style explorers can hide it
        self.show_sprite = True
        # When True, next update_camera jumps to the target (no spring) so a
        # new maze / reset is immediately framed instead of easing from the
        # previous view.
        self._snap_camera = True
        self._last_viewport_w = 800
        self._last_viewport_h = 600
        # Debug: last focus bounding box in world pixels (min_x, min_y, max_x, max_y)
        self._focus_box: Optional[Tuple[float, float, float, float]] = None
        self.debug_camera = False
        # When True, frame the entire maze instead of the frontier box
        self.overview_mode = False

    # ------------------------------------------------------------------
    # Public API used by engine / main
    # ------------------------------------------------------------------

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
        self._explored.clear()
        self._fading.clear()
        self._recent.clear()
        self._frontier.clear()
        self.explore_surface = None
        self._scale_cache_key = None
        self._bg_scaled = self._exp_scaled = self._trail_scaled = None
        self.mark_explored(start_cell)
        # Cancel any in-flight animation.
        if self._anim_done is not None:
            self._anim_done.set()
        self._animating = False
        self._anim_done = None
        # Reset camera to frame the start cell immediately.
        cx, cy = self._cell_center(start_cell)
        # Reasonable default zoom: a handful of rooms around the start.
        init_box = ROOM_SIZE * max(MIN_VIEW_CELLS, 12)
        vw = max(1, self._last_viewport_w)
        vh = max(1, self._last_viewport_h)
        half_w = init_box * 0.5
        half_h = init_box * 0.5 * (vh / max(1, vw))
        self.view_left = cx - half_w
        self.view_right = cx + half_w
        self.view_top = cy - half_h
        self.view_bottom = cy + half_h
        self._vel_left = self._vel_right = self._vel_top = self._vel_bottom = 0.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.show_sprite = True
        self._snap_camera = True
        self._sync_cam_from_view(vw, vh)
        self._clamp_view_to_maze(vw, vh)
        self._sync_cam_from_view(vw, vh)

    def invalidate_trail(self) -> None:
        self.trail_surface = None
        self.trail_path_size = -1
        self._trail_scaled = None  # force rescale next draw

    def update_path(self, path: List[Cell]) -> None:
        self.path = list(path)
        self.invalidate_trail()

    def mark_explored(self, cell: Cell) -> None:
        """Record that *cell* has been visited / expanded. Safe to call often."""
        now = time.monotonic()
        with self._explore_lock:
            self._explored[cell] = now
            self._fading.add(cell)
            self._recent.append(cell)
        self._paint_explored_cell(cell, HIGHLIGHT_COLOR)

    def mark_explored_many(self, cells: List[Cell]) -> None:
        now = time.monotonic()
        with self._explore_lock:
            for c in cells:
                self._explored[c] = now
                self._fading.add(c)
                self._recent.append(c)
        for c in cells:
            self._paint_explored_cell(c, HIGHLIGHT_COLOR)

    def set_show_sprite(self, show: bool) -> None:
        """Show or hide the red agent dot."""
        self.show_sprite = show

    def set_frontier(self, cells: List[Cell]) -> None:
        """Replace the open-leaf / frontier set used to drive the camera."""
        with self._explore_lock:
            self._frontier = set(cells)

    def add_to_frontier(self, cell: Cell) -> None:
        with self._explore_lock:
            self._frontier.add(cell)

    def remove_from_frontier(self, cell: Cell) -> None:
        with self._explore_lock:
            self._frontier.discard(cell)

    def clear_frontier(self) -> None:
        with self._explore_lock:
            self._frontier.clear()

    # ------------------------------------------------------------------
    # Geometry (variable thickness)
    # ------------------------------------------------------------------

    @staticmethod
    def col_x(col: int) -> int:
        """World-pixel left edge of column *col* (variable-thickness grid)."""
        return (col // 2) * (WALL_THICKNESS + ROOM_SIZE) + WALL_THICKNESS * (col % 2)

    @staticmethod
    def row_y(row: int) -> int:
        """World-pixel top edge of row *row* (variable-thickness grid)."""
        return (row // 2) * (WALL_THICKNESS + ROOM_SIZE) + WALL_THICKNESS * (row % 2)

    def _rebuild_geometry(self) -> None:
        assert self.maze is not None
        self.total_width = self.col_x(self.maze.cols)
        self.total_height = self.row_y(self.maze.rows)

    @staticmethod
    def _cell_width(col: int) -> int:
        return WALL_THICKNESS if col % 2 == 0 else ROOM_SIZE

    @staticmethod
    def _cell_height(row: int) -> int:
        return WALL_THICKNESS if row % 2 == 0 else ROOM_SIZE

    def _cell_center(self, cell: Cell) -> Tuple[float, float]:
        x = self.col_x(cell.col) + self._cell_width(cell.col) / 2.0
        y = self.row_y(cell.row) + self._cell_height(cell.row) / 2.0
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
                    self.col_x(c),
                    self.row_y(r),
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
        x = self.col_x(cell.col) + pad
        y = self.row_y(cell.row) + pad
        w = self._cell_width(cell.col) - 2 * pad
        h = self._cell_height(cell.row) - 2 * pad
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surf, color, rect, border_radius=6)
        font = pygame.font.SysFont("sans", max(12, ROOM_SIZE // 2), bold=True)
        text = font.render(label, True, WHITE)
        tx = self.col_x(cell.col) + (self._cell_width(cell.col) - text.get_width()) // 2
        ty = self.row_y(cell.row) + (self._cell_height(cell.row) - text.get_height()) // 2
        surf.blit(text, (tx, ty))

    def _ensure_trail(self) -> bool:
        if self.trail_surface is not None and self.trail_path_size == len(self.path):
            return False
        surf = pygame.Surface((self.total_width, self.total_height), pygame.SRCALPHA)
        pad = max(2, ROOM_SIZE // 6)
        for idx, c in enumerate(self.path):
            if idx == 0:
                continue  # skip current (head) cell
            rect = pygame.Rect(
                self.col_x(c.col) + pad,
                self.row_y(c.row) + pad,
                self._cell_width(c.col) - 2 * pad,
                self._cell_height(c.row) - 2 * pad,
            )
            pygame.draw.rect(surf, TRAIL_COLOR, rect, border_radius=6)
        self.trail_surface = surf
        return True
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

    def animate_teleport(
        self, from_cell: Cell, to_cell: Cell, done: threading.Event
    ) -> None:
        """Snap the sprite to to_cell. Uses a short fade-style hop when slow."""
        to_px = self._cell_center(to_cell)
        # Teleports are intentionally snappy: at most a few frames.
        steps = min(4, self._steps_for_duration())
        if steps <= 1:
            self.sprite_x, self.sprite_y = to_px
            self.invalidate_trail()
            done.set()
            return
        from_px = self._cell_center(from_cell)
        self._anim_kind = "teleport"
        self._anim_from = from_px
        self._anim_to = to_px
        self._anim_steps = steps
        self._anim_step = 0
        self._anim_elapsed = 0.0
        self._anim_per_step_ms = max(1, max(8, self._anim_duration_ms // 4) // steps)
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
            if self._anim_kind in ("move", "teleport"):
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
    # Camera
    # ------------------------------------------------------------------

    def update_camera(self, viewport_w: int, viewport_h: int, dt: float) -> None:
        """Spring each view-edge independently toward the focus box, then derive cam/zoom."""
        if self.maze is None or viewport_w <= 0 or viewport_h <= 0:
            return

        dt = max(0.0, min(dt, 0.05))
        self._last_viewport_w = viewport_w
        self._last_viewport_h = viewport_h

        # Snapshot under lock so the worker can keep mutating safely
        with self._explore_lock:
            frontier_snapshot = list(self._frontier)
            recent_snapshot = list(self._recent)

        # Prefer open leaves (frontier). Fall back to recent explored cells,
        # then to the sprite, so sequential agents still frame sensibly.
        focus_cells: List[Cell] = frontier_snapshot
        if not focus_cells:
            seen = set()
            for c in reversed(recent_snapshot):
                if c not in seen:
                    seen.add(c)
                    focus_cells.append(c)
                    if len(focus_cells) >= RECENT_N:
                        break

        # --- Target rectangle in world space ---
        if self.overview_mode:
            tl, tr, tt, tb = 0.0, float(self.total_width), 0.0, float(self.total_height)
        elif not focus_cells:
            pad = ROOM_SIZE * PADDING_CELLS
            tl = self.sprite_x - pad
            tr = self.sprite_x + pad
            tt = self.sprite_y - pad
            tb = self.sprite_y + pad
        else:
            tl = tt = float("inf")
            tr = tb = float("-inf")
            for cell in focus_cells:
                px, py = self._cell_center(cell)
                half_w = self._cell_width(cell.col) * 0.5 + ROOM_SIZE * PADDING_CELLS
                half_h = self._cell_height(cell.row) * 0.5 + ROOM_SIZE * PADDING_CELLS
                tl = min(tl, px - half_w)
                tr = max(tr, px + half_w)
                tt = min(tt, py - half_h)
                tb = max(tb, py + half_h)
            pad = ROOM_SIZE * PADDING_CELLS
            tl = min(tl, self.sprite_x - pad)
            tr = max(tr, self.sprite_x + pad)
            tt = min(tt, self.sprite_y - pad)
            tb = max(tb, self.sprite_y + pad)

        # Clip target to maze
        tl = max(0.0, tl)
        tt = max(0.0, tt)
        tr = min(float(self.total_width), tr)
        tb = min(float(self.total_height), tb)
        if tr < tl:
            tl, tr = tr, tl
        if tb < tt:
            tt, tb = tb, tt

        # Enforce a minimum target size (in world px)
        min_span = float(ROOM_SIZE * MIN_VIEW_CELLS)
        if tr - tl < min_span:
            mid = 0.5 * (tl + tr)
            tl, tr = mid - min_span * 0.5, mid + min_span * 0.5
        if tb - tt < min_span:
            mid = 0.5 * (tt + tb)
            tt, tb = mid - min_span * 0.5, mid + min_span * 0.5
        # Re-clip after min-size expand
        tl = max(0.0, tl)
        tt = max(0.0, tt)
        tr = min(float(self.total_width), tr)
        tb = min(float(self.total_height), tb)

        self._focus_box = (tl, tt, tr, tb)

        if self._snap_camera:
            self.view_left, self.view_right = tl, tr
            self.view_top, self.view_bottom = tt, tb
            self._vel_left = self._vel_right = self._vel_top = self._vel_bottom = 0.0
            self.pan_x = self.pan_y = 0.0
            self._snap_camera = False
        else:
            # Independent critically-damped springs on each edge.
            # An edge already on target stays put; only edges that need to
            # move (e.g. frontier grew on the right) accelerate.
            self.view_left, self._vel_left = self._spring_step(
                self.view_left, self._vel_left, tl, CAMERA_OMEGA, dt
            )
            self.view_right, self._vel_right = self._spring_step(
                self.view_right, self._vel_right, tr, CAMERA_OMEGA, dt
            )
            self.view_top, self._vel_top = self._spring_step(
                self.view_top, self._vel_top, tt, CAMERA_OMEGA, dt
            )
            self.view_bottom, self._vel_bottom = self._spring_step(
                self.view_bottom, self._vel_bottom, tb, CAMERA_OMEGA, dt
            )

        # Keep edges ordered and enforce viewport aspect by expanding only.
        self._normalize_view_rect(viewport_w, viewport_h)
        self._clamp_view_to_maze(viewport_w, viewport_h)
        self._sync_cam_from_view(viewport_w, viewport_h)

    def _normalize_view_rect(self, viewport_w: int, viewport_h: int) -> None:
        """Ensure L<R, T<B, and the rect matches the viewport aspect (expand only)."""
        if self.view_right < self.view_left:
            self.view_left, self.view_right = self.view_right, self.view_left
            self._vel_left, self._vel_right = self._vel_right, self._vel_left
        if self.view_bottom < self.view_top:
            self.view_top, self.view_bottom = self.view_bottom, self.view_top
            self._vel_top, self._vel_bottom = self._vel_bottom, self._vel_top

        min_span = float(ROOM_SIZE * MIN_VIEW_CELLS)
        if self.view_right - self.view_left < min_span:
            mid = 0.5 * (self.view_left + self.view_right)
            self.view_left, self.view_right = mid - min_span * 0.5, mid + min_span * 0.5
        if self.view_bottom - self.view_top < min_span:
            mid = 0.5 * (self.view_top + self.view_bottom)
            self.view_top, self.view_bottom = mid - min_span * 0.5, mid + min_span * 0.5

        # Aspect: world_w/world_h should equal viewport_w/viewport_h.
        # Expand the too-narrow axis so we never crop the focus content.
        vw = max(1, viewport_w)
        vh = max(1, viewport_h)
        aspect = vw / vh
        cur_w = self.view_right - self.view_left
        cur_h = self.view_bottom - self.view_top
        if cur_w <= 0 or cur_h <= 0:
            return
        if cur_w / cur_h > aspect:
            # Too wide → expand height
            need_h = cur_w / aspect
            mid = 0.5 * (self.view_top + self.view_bottom)
            self.view_top = mid - need_h * 0.5
            self.view_bottom = mid + need_h * 0.5
        elif cur_w / cur_h < aspect:
            # Too tall → expand width
            need_w = cur_h * aspect
            mid = 0.5 * (self.view_left + self.view_right)
            self.view_left = mid - need_w * 0.5
            self.view_right = mid + need_w * 0.5

    def _clamp_view_to_maze(self, viewport_w: int, viewport_h: int) -> None:
        """Shift the view rect so it stays inside the maze when possible.

        Uses a tiny inward bias so float/raster rounding does not reveal a
        one-pixel gutter of empty background past the maze edge.
        """
        tw = float(self.total_width)
        th = float(self.total_height)
        if tw <= 0 or th <= 0:
            return

        # ~half a world-pixel — enough to hide sub-pixel overscan at the edge.
        eps = 0.5

        w = self.view_right - self.view_left
        h = self.view_bottom - self.view_top
        if w >= tw - 1e-6:
            mid = tw * 0.5
            self.view_left, self.view_right = mid - w * 0.5, mid + w * 0.5
            self._vel_left = self._vel_right = 0.0
        else:
            if self.view_left < eps:
                self.view_right += eps - self.view_left
                self.view_left = eps
                self._vel_left = 0.0
            if self.view_right > tw - eps:
                self.view_left -= self.view_right - (tw - eps)
                self.view_right = tw - eps
                self._vel_right = 0.0
            # Keep width stable if both edges fought each other
            if self.view_right - self.view_left < w - 1e-6:
                mid = 0.5 * (self.view_left + self.view_right)
                self.view_left, self.view_right = mid - w * 0.5, mid + w * 0.5

        if h >= th - 1e-6:
            mid = th * 0.5
            self.view_top, self.view_bottom = mid - h * 0.5, mid + h * 0.5
            self._vel_top = self._vel_bottom = 0.0
        else:
            if self.view_top < eps:
                self.view_bottom += eps - self.view_top
                self.view_top = eps
                self._vel_top = 0.0
            if self.view_bottom > th - eps:
                self.view_top -= self.view_bottom - (th - eps)
                self.view_bottom = th - eps
                self._vel_bottom = 0.0
            if self.view_bottom - self.view_top < h - 1e-6:
                mid = 0.5 * (self.view_top + self.view_bottom)
                self.view_top, self.view_bottom = mid - h * 0.5, mid + h * 0.5

    def _sync_cam_from_view(self, viewport_w: int, viewport_h: int) -> None:
        """Derive cam_x/cam_y/zoom from the view rectangle for the draw path."""
        w = max(1e-6, self.view_right - self.view_left)
        h = max(1e-6, self.view_bottom - self.view_top)
        self.cam_x = 0.5 * (self.view_left + self.view_right)
        self.cam_y = 0.5 * (self.view_top + self.view_bottom)
        # Zoom so the view rect maps onto the viewport (aspect already matched).
        self.zoom = min(viewport_w / w, viewport_h / h)
        self.zoom = max(0.05, self.zoom)

        # Cap zoom-in by expanding the view rect (keeps edge springs consistent).
        if self.zoom > MAX_ZOOM:
            scale = self.zoom / MAX_ZOOM  # > 1 → grow the world rect
            cx, cy = self.cam_x, self.cam_y
            self.view_left = cx - (cx - self.view_left) * scale
            self.view_right = cx + (self.view_right - cx) * scale
            self.view_top = cy - (cy - self.view_top) * scale
            self.view_bottom = cy + (self.view_bottom - cy) * scale
            self.zoom = MAX_ZOOM
            # Expansion can push past the maze — pull back in.
            self._clamp_view_to_maze(viewport_w, viewport_h)
            w = max(1e-6, self.view_right - self.view_left)
            h = max(1e-6, self.view_bottom - self.view_top)
            self.cam_x = 0.5 * (self.view_left + self.view_right)
            self.cam_y = 0.5 * (self.view_top + self.view_bottom)
            self.zoom = min(viewport_w / w, viewport_h / h)
            self.zoom = max(0.05, min(self.zoom, MAX_ZOOM))

    def _min_zoom_to_fit_maze(self, viewport_w: int, viewport_h: int) -> float:
        if self.total_width <= 0 or self.total_height <= 0:
            return 0.05
        margin = 1.01
        zx = viewport_w / (self.total_width * margin)
        zy = viewport_h / (self.total_height * margin)
        return max(0.05, min(zx, zy))

    @staticmethod
    def _spring_step(
        pos: float, vel: float, target: float, omega: float, dt: float
    ) -> Tuple[float, float]:
        """One critically-damped spring integration step (zeta = 1)."""
        # x'' + 2*omega*x' + omega^2*(x - target) = 0
        accel = -2.0 * omega * vel - (omega * omega) * (pos - target)
        vel = vel + accel * dt
        pos = pos + vel * dt
        return pos, vel

    def handle_pan_event(self, event: pygame.event.Event) -> bool:
        """Mouse-drag free look. Returns True if the event was consumed."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._dragging = True
            self._drag_last = event.pos
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._dragging = False
            self._drag_last = None
            return False
        if event.type == pygame.MOUSEMOTION and self._dragging and self._drag_last:
            dx = event.pos[0] - self._drag_last[0]
            dy = event.pos[1] - self._drag_last[1]
            # Dragging the map: opposite direction, scaled by zoom
            self.pan_x -= dx / max(0.01, self.zoom)
            self.pan_y -= dy / max(0.01, self.zoom)
            self._drag_last = event.pos
            return True
        if event.type == pygame.MOUSEWHEEL:
            # Zoom by scaling the view rectangle about its centre.
            # factor > 1 → zoom in (smaller world rect).
            factor = 1.12 if event.y > 0 else 1 / 1.12
            cx = 0.5 * (self.view_left + self.view_right)
            cy = 0.5 * (self.view_top + self.view_bottom)
            w = (self.view_right - self.view_left) / factor
            h = (self.view_bottom - self.view_top) / factor
            # Respect MAX_ZOOM / fit-maze limits via resulting size.
            min_w = self._last_viewport_w / max(MAX_ZOOM, 0.01)
            max_w = float(self.total_width) * 1.01 if self.total_width > 0 else w
            w = max(min_w, min(max_w, w))
            aspect = self._last_viewport_w / max(1, self._last_viewport_h)
            h = w / aspect
            self.view_left = cx - w * 0.5
            self.view_right = cx + w * 0.5
            self.view_top = cy - h * 0.5
            self.view_bottom = cy + h * 0.5
            # Manual zoom temporarily overrides auto-follow; clear edge velocity.
            self._vel_left = self._vel_right = self._vel_top = self._vel_bottom = 0.0
            self._clamp_view_to_maze(self._last_viewport_w, self._last_viewport_h)
            self._sync_cam_from_view(self._last_viewport_w, self._last_viewport_h)
            self._scale_cache_key = None  # force rescale
            return True
        return False

    def reset_pan(self) -> None:
        self.pan_x = 0.0
        self.pan_y = 0.0

    def toggle_debug_camera(self) -> bool:
        self.debug_camera = not self.debug_camera
        return self.debug_camera

    def toggle_overview(self) -> bool:
        """Zoom out to the full maze (or resume frontier framing)."""
        self.overview_mode = not self.overview_mode
        if self.overview_mode:
            self.pan_x = 0.0
            self.pan_y = 0.0
            self._snap_camera = True  # jump so the whole maze is visible immediately
        return self.overview_mode

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(
        self,
        surface: pygame.Surface,
        viewport: pygame.Rect,
    ) -> None:
        """Draw the maze clipped to *viewport* using the current camera."""
        if self.maze is None or self.background is None:
            return

        # World-space centre of the camera
        world_cx = self.cam_x + self.pan_x
        world_cy = self.cam_y + self.pan_y
        z = max(1e-6, self.zoom)

        # World-space rectangle visible in the viewport (plus 1px slack).
        half_w = viewport.w / (2.0 * z)
        half_h = viewport.h / (2.0 * z)
        world_l = world_cx - half_w
        world_t = world_cy - half_h
        world_r = world_cx + half_w
        world_b = world_cy + half_h

        # Source rect on the full-maze surfaces (integer pixel bounds).
        src_x = max(0, int(math.floor(world_l)) - 1)
        src_y = max(0, int(math.floor(world_t)) - 1)
        src_r = min(self.total_width, int(math.ceil(world_r)) + 1)
        src_b = min(self.total_height, int(math.ceil(world_b)) + 1)
        src_w = max(1, src_r - src_x)
        src_h = max(1, src_b - src_y)

        scaled_w = max(1, int(math.ceil(src_w * z)))
        scaled_h = max(1, int(math.ceil(src_h * z)))

        # Screen position of world (0,0); then of the source origin.
        ox = viewport.centerx - world_cx * z
        oy = viewport.centery - world_cy * z
        dest_x = int(ox + src_x * z) - viewport.x
        dest_y = int(oy + src_y * z) - viewport.y

        view_surf = surface.subsurface(viewport) if viewport.x == 0 and viewport.y == 0 and viewport.size == surface.get_size() else None
        if view_surf is None:
            # Draw into a clip region on the main surface when possible.
            view_surf = pygame.Surface((viewport.w, viewport.h))
            owns_view = True
        else:
            owns_view = False
        view_surf.fill(BG)

        # Scale only the visible slice; reuse when the view rect + zoom are unchanged.
        cache_key = (src_x, src_y, src_w, src_h, scaled_w, scaled_h)
        need_rescale = cache_key != self._scale_cache_key

        if need_rescale or self._bg_scaled is None:
            bg_src = self.background.subsurface((src_x, src_y, src_w, src_h))
            self._bg_scaled = pygame.transform.scale(bg_src, (scaled_w, scaled_h))
        view_surf.blit(self._bg_scaled, (dest_x, dest_y))

        # Exploration: update fade paints, then scale only if view changed or fade painted.
        fading_active = self._update_exploration_fade()
        if self.explore_surface is not None:
            if need_rescale or fading_active or self._exp_scaled is None:
                exp_src = self.explore_surface.subsurface((src_x, src_y, src_w, src_h))
                self._exp_scaled = pygame.transform.scale(exp_src, (scaled_w, scaled_h))
            view_surf.blit(self._exp_scaled, (dest_x, dest_y))

        trail_dirty = self._ensure_trail()
        if self.trail_surface is not None:
            if need_rescale or trail_dirty or self._trail_scaled is None:
                trail_src = self.trail_surface.subsurface((src_x, src_y, src_w, src_h))
                self._trail_scaled = pygame.transform.scale(trail_src, (scaled_w, scaled_h))
            view_surf.blit(self._trail_scaled, (dest_x, dest_y))

        self._scale_cache_key = cache_key

        if self.debug_camera and self._focus_box is not None:
            min_x, min_y, max_x, max_y = self._focus_box
            # Same transform as the scaled maze blit
            rx = int(ox + min_x * z) - viewport.x
            ry = int(oy + min_y * z) - viewport.y
            rw = max(1, int((max_x - min_x) * z))
            rh = max(1, int((max_y - min_y) * z))
            pygame.draw.rect(view_surf, (255, 64, 255), pygame.Rect(rx, ry, rw, rh), width=2)

        # Sprite (optional — frontier explorers often hide it)
        if self.show_sprite:
            radius = max(3, int(ROOM_SIZE * 0.32 * z))
            sx = int(ox + self.sprite_x * z) - viewport.x
            sy = int(oy + self.sprite_y * z) - viewport.y
            if -radius < sx < viewport.w + radius and -radius < sy < viewport.h + radius:
                pygame.draw.circle(view_surf, SPRITE_COLOR, (sx, sy), radius)
                pygame.draw.circle(view_surf, WHITE, (sx, sy), radius, max(1, radius // 6))

        if owns_view:
            surface.blit(view_surf, (viewport.x, viewport.y))

    def _ensure_explore_surface(self) -> None:
        if self.explore_surface is not None:
            return
        if self.total_width <= 0 or self.total_height <= 0:
            return
        self.explore_surface = pygame.Surface(
            (self.total_width, self.total_height), pygame.SRCALPHA
        )

    def _paint_explored_cell(self, cell: Cell, color: Tuple[int, int, int]) -> None:
        self._ensure_explore_surface()
        if self.explore_surface is None:
            return
        pad = max(1, ROOM_SIZE // 6)
        x = self.col_x(cell.col) + pad
        y = self.row_y(cell.row) + pad
        w = self._cell_width(cell.col) - 2 * pad
        h = self._cell_height(cell.row) - 2 * pad
        if w <= 0 or h <= 0:
            return
        pygame.draw.rect(self.explore_surface, (*color, 255), (x, y, w, h))

    def _update_exploration_fade(self) -> bool:
        """Advance highlight→grey only for cells still fading (not all visited)."""
        with self._explore_lock:
            if not self._fading:
                return False
            fading = list(self._fading)
            explored = self._explored

        now = time.monotonic()
        done: List[Cell] = []
        for cell in fading:
            t0 = explored.get(cell)
            if t0 is None:
                done.append(cell)
                continue
            age = now - t0
            if age >= FADE_DURATION_S:
                self._paint_explored_cell(cell, EXPLORED_GREY)
                done.append(cell)
            else:
                raw = 1.0 - age / FADE_DURATION_S
                intensity = raw * raw
                color = _lerp_color(EXPLORED_GREY, HIGHLIGHT_COLOR, intensity)
                self._paint_explored_cell(cell, color)

        if done:
            with self._explore_lock:
                for c in done:
                    self._fading.discard(c)
        return True  # at least one cell was still fading / painted

