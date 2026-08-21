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
MIN_VIEW_CELLS = 14  # never zoom tighter than this many logical rooms
PADDING_CELLS = 3  # extra cells of padding around the recent set


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

        # --- Exploration overlay ---
        # cell -> monotonic time when first/last marked
        self._explored: Dict[Cell, float] = {}
        # ordered recent history (most recent last) — fallback for camera
        self._recent: Deque[Cell] = deque(maxlen=RECENT_N * 2)
        # Open leaves / frontier: cells that still have expansion work left.
        # Camera prefers these over raw recency when the set is non-empty.
        self._frontier: Set[Cell] = set()
        # Worker thread marks cells; main thread reads them for camera/draw.
        self._explore_lock = threading.Lock()

        # --- Camera (world-space centre + zoom), with velocity for smooth accel ---
        self.cam_x = 0.0
        self.cam_y = 0.0
        self.zoom = 1.0  # 1.0 = 1 world-pixel → 1 screen-pixel
        self._target_cam_x = 0.0
        self._target_cam_y = 0.0
        self._target_zoom = 1.0
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._vel_zoom = 0.0
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
        self._recent.clear()
        self._frontier.clear()
        self.mark_explored(start_cell)
        # Cancel any in-flight animation.
        if self._anim_done is not None:
            self._anim_done.set()
        self._animating = False
        self._anim_done = None
        # Reset camera to frame the start cell immediately (no leftover pan/zoom).
        cx, cy = self._cell_center(start_cell)
        # Reasonable default zoom: a handful of rooms around the start.
        init_box = ROOM_SIZE * max(MIN_VIEW_CELLS, 12)
        vw = max(1, self._last_viewport_w)
        vh = max(1, self._last_viewport_h)
        init_zoom = min(vw / init_box, vh / init_box, 1.35)
        init_zoom = max(0.22, init_zoom)
        cx, cy = self._clamp_center(cx, cy, vw, vh, init_zoom)
        self.cam_x = cx
        self.cam_y = cy
        self._target_cam_x = cx
        self._target_cam_y = cy
        self.zoom = init_zoom
        self._target_zoom = init_zoom
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._vel_zoom = 0.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.show_sprite = True
        self._snap_camera = True

    def invalidate_trail(self) -> None:
        self.trail_surface = None
        self.trail_path_size = -1

    def update_path(self, path: List[Cell]) -> None:
        self.path = list(path)
        self.invalidate_trail()

    def mark_explored(self, cell: Cell) -> None:
        """Record that *cell* has been visited / expanded. Safe to call often."""
        now = time.monotonic()
        with self._explore_lock:
            self._explored[cell] = now
            self._recent.append(cell)

    def mark_explored_many(self, cells: List[Cell]) -> None:
        now = time.monotonic()
        with self._explore_lock:
            for c in cells:
                self._explored[c] = now
                self._recent.append(c)

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

    # not used yet, but could do something like this for dynamic geometry
    def _cum_col_x(col):
        return (col // 2) * (WALL_THICKNESS + ROOM_SIZE) + WALL_THICKNESS * (col % 2)

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
        """Recompute target from recent exploration, then integrate a damped spring."""
        if self.maze is None or viewport_w <= 0 or viewport_h <= 0:
            return

        dt = max(0.0, min(dt, 0.05))  # clamp for stability if a frame hitches
        now = time.monotonic()

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

        self._last_viewport_w = viewport_w
        self._last_viewport_h = viewport_h

        if self.overview_mode:
            # Frame the entire maze (for debugging / orientation).
            self._target_cam_x = self.total_width / 2.0
            self._target_cam_y = self.total_height / 2.0
            margin = 1.02
            zx = viewport_w / max(1.0, self.total_width * margin)
            zy = viewport_h / max(1.0, self.total_height * margin)
            self._target_zoom = max(0.05, min(zx, zy))
            self._focus_box = (0.0, 0.0, float(self.total_width), float(self.total_height))
        elif not focus_cells:
            self._target_cam_x = self.sprite_x
            self._target_cam_y = self.sprite_y
            init_box = ROOM_SIZE * max(MIN_VIEW_CELLS, 12)
            target_zoom = min(viewport_w / init_box, viewport_h / init_box, 1.35)
            self._target_zoom = max(0.08, target_zoom)
            pad = ROOM_SIZE * PADDING_CELLS
            self._focus_box = (
                self.sprite_x - pad,
                self.sprite_y - pad,
                self.sprite_x + pad,
                self.sprite_y + pad,
            )
        else:
            min_x = min_y = float("inf")
            max_x = max_y = float("-inf")
            for cell in focus_cells:
                px, py = self._cell_center(cell)
                half_w = self._cell_width(cell.col) * 0.5 + ROOM_SIZE * PADDING_CELLS
                half_h = self._cell_height(cell.row) * 0.5 + ROOM_SIZE * PADDING_CELLS
                min_x = min(min_x, px - half_w)
                max_x = max(max_x, px + half_w)
                min_y = min(min_y, py - half_h)
                max_y = max(max_y, py + half_h)

            # Always include the sprite so the active agent/start cannot fall
            # outside the framed rectangle (important at the beginning and for
            # sequential explorers).
            pad = ROOM_SIZE * PADDING_CELLS
            min_x = min(min_x, self.sprite_x - pad)
            max_x = max(max_x, self.sprite_x + pad)
            min_y = min(min_y, self.sprite_y - pad)
            max_y = max(max_y, self.sprite_y + pad)

            self._focus_box = (min_x, min_y, max_x, max_y)

            # Centre of the frontier bounding box (not the centroid of points),
            # so a lopsided cluster cannot push the sparse side off-screen.
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0

            box_w = max(max_x - min_x, ROOM_SIZE * MIN_VIEW_CELLS)
            box_h = max(max_y - min_y, ROOM_SIZE * MIN_VIEW_CELLS)

            # Zoom so the box roughly fits the viewport (with a little margin)
            zoom_x = viewport_w / box_w
            zoom_y = viewport_h / box_h
            target_zoom = min(zoom_x, zoom_y, 1.35)
            target_zoom = max(0.05, target_zoom)

            self._target_cam_x = cx
            self._target_cam_y = cy
            self._target_zoom = target_zoom

        # Keep the *target* inside the maze so the spring never aims off-map.
        self._target_cam_x, self._target_cam_y = self._clamp_center(
            self._target_cam_x, self._target_cam_y, viewport_w, viewport_h, self._target_zoom
        )

        if self._snap_camera:
            # Hard cut used after new maze / reset so the start is on-screen
            # immediately instead of easing in from the previous view.
            self.cam_x = self._target_cam_x
            self.cam_y = self._target_cam_y
            self.zoom = self._target_zoom
            self._vel_x = 0.0
            self._vel_y = 0.0
            self._vel_zoom = 0.0
            self.pan_x = 0.0
            self.pan_y = 0.0
            self._snap_camera = False
        else:
            # Critically-damped spring: accel changes velocity smoothly toward target
            self.cam_x, self._vel_x = self._spring_step(
                self.cam_x, self._vel_x, self._target_cam_x, CAMERA_OMEGA, dt
            )
            self.cam_y, self._vel_y = self._spring_step(
                self.cam_y, self._vel_y, self._target_cam_y, CAMERA_OMEGA, dt
            )
            self.zoom, self._vel_zoom = self._spring_step(
                self.zoom, self._vel_zoom, self._target_zoom, CAMERA_ZOOM_OMEGA, dt
            )
            self.zoom = max(0.12, self.zoom)

        # Clamp the effective view centre (auto cam + manual pan) to the maze.
        self._clamp_camera_to_maze(viewport_w, viewport_h)

    def _view_half_extents(
        self, viewport_w: int, viewport_h: int, zoom: float
    ) -> Tuple[float, float]:
        z = max(0.01, zoom)
        return viewport_w / (2.0 * z), viewport_h / (2.0 * z)

    def _clamp_center(
        self,
        cx: float,
        cy: float,
        viewport_w: int,
        viewport_h: int,
        zoom: float,
    ) -> Tuple[float, float]:
        """Clamp a world-space view centre so the viewport stays inside the maze."""
        if self.total_width <= 0 or self.total_height <= 0:
            return cx, cy
        half_w, half_h = self._view_half_extents(viewport_w, viewport_h, zoom)

        if self.total_width <= 2 * half_w:
            cx = self.total_width / 2.0
        else:
            cx = max(half_w, min(self.total_width - half_w, cx))

        if self.total_height <= 2 * half_h:
            cy = self.total_height / 2.0
        else:
            cy = max(half_h, min(self.total_height - half_h, cy))

        return cx, cy

    def _clamp_camera_to_maze(self, viewport_w: int, viewport_h: int) -> None:
        """
        Ensure cam + pan never reveal empty space past the maze edges.
        Adjusts pan first (manual offset), then cam if still out of range.
        Zeroes velocity on axes that hit a stop so the spring does not fight the clamp.
        """
        eff_x = self.cam_x + self.pan_x
        eff_y = self.cam_y + self.pan_y
        clamped_x, clamped_y = self._clamp_center(
            eff_x, eff_y, viewport_w, viewport_h, self.zoom
        )

        if clamped_x != eff_x:
            # Prefer absorbing the excess into manual pan, then cam.
            self.pan_x += clamped_x - eff_x
            # If pan alone cannot fix (e.g. auto cam far outside), fold into cam.
            eff_x = self.cam_x + self.pan_x
            if abs(eff_x - clamped_x) > 1e-6:
                self.cam_x = clamped_x - self.pan_x
            self._vel_x = 0.0

        if clamped_y != eff_y:
            self.pan_y += clamped_y - eff_y
            eff_y = self.cam_y + self.pan_y
            if abs(eff_y - clamped_y) > 1e-6:
                self.cam_y = clamped_y - self.pan_y
            self._vel_y = 0.0

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
            # Zoom toward cursor would be nicer; for now just scale
            factor = 1.12 if event.y > 0 else 1 / 1.12
            self._target_zoom = max(0.15, min(2.0, self._target_zoom * factor))
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
        z = self.zoom

        # Where the world origin lands on the screen
        # screen_x = viewport.centerx + (world_x - world_cx) * z
        ox = viewport.centerx - world_cx * z
        oy = viewport.centery - world_cy * z

        # Build a temporary surface the size of the viewport and blit scaled pieces
        # For simplicity and correctness we scale the whole maze surface.
        # (Fine for the maze sizes we use; can be optimised later with dirty rects.)
        scaled_w = max(1, int(self.total_width * z))
        scaled_h = max(1, int(self.total_height * z))

        # Clip: only the portion that intersects the viewport
        # Destination top-left of the full scaled maze
        dest_x = int(ox)
        dest_y = int(oy)

        # Create a sub-surface sized to the viewport
        view_surf = pygame.Surface((viewport.w, viewport.h))
        view_surf.fill(BG)

        # Scale background + overlays once
        bg_scaled = pygame.transform.scale(self.background, (scaled_w, scaled_h))
        view_surf.blit(bg_scaled, (dest_x - viewport.x, dest_y - viewport.y))

        # Exploration overlay (drawn under trail)
        self._draw_exploration(view_surf, dest_x - viewport.x, dest_y - viewport.y, z)

        self._ensure_trail()
        if self.trail_surface is not None:
            trail_scaled = pygame.transform.scale(self.trail_surface, (scaled_w, scaled_h))
            view_surf.blit(trail_scaled, (dest_x - viewport.x, dest_y - viewport.y))

        # Debug: focus / frontier bounding box in world space
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

        surface.blit(view_surf, (viewport.x, viewport.y))

    def _draw_exploration(
        self, view_surf: pygame.Surface, origin_x: int, origin_y: int, z: float
    ) -> None:
        """Paint fading highlight for explored cells."""
        with self._explore_lock:
            if not self._explored:
                return
            items = list(self._explored.items())
        now = time.monotonic()
        pad = max(1, int((ROOM_SIZE // 6) * z))

        for cell, t0 in items:
            age = now - t0
            if age < 0:
                continue
            # intensity 1 → 0 over FADE_DURATION_S, then stays at grey
            raw = 1.0 - min(1.0, age / FADE_DURATION_S)
            # ease the fade a bit
            intensity = raw * raw  # quadratic fall-off looks nicer
            color = _lerp_color(EXPLORED_GREY, HIGHLIGHT_COLOR, intensity)

            x = int(origin_x + self.col_x[cell.col] * z) + pad
            y = int(origin_y + self.row_y[cell.row] * z) + pad
            w = max(1, int(self._cell_width(cell.col) * z) - 2 * pad)
            h = max(1, int(self._cell_height(cell.row) * z) - 2 * pad)
            if w <= 0 or h <= 0:
                continue
            # Skip if completely outside the view surface
            if x + w < 0 or y + h < 0 or x > view_surf.get_width() or y > view_surf.get_height():
                continue
            rect = pygame.Rect(x, y, w, h)
            pygame.draw.rect(view_surf, color, rect, border_radius=max(2, int(4 * z)))
