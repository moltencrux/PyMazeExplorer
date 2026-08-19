#!/usr/bin/env python3
"""
Maze Explorer – Python / Pygame port.

A random block-based maze is generated and shown on screen. Students implement
a maze-solving algorithm by subclassing BaseExplorer; the app animates their
agent moving through the maze, leaves a trail behind it (which un-marks itself
when the agent backtracks), and congratulates the player when the goal is
reached.

Run:
    pip install pygame
    python main.py
"""

from __future__ import annotations

import sys
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

import pygame

from mazegame.base_explorer import BaseExplorer
from mazegame.explorers import RandomWalkExplorer, WallFollowerExplorer
from mazegame.maze import Maze
from mazegame.maze_engine import MazeEngine
from mazegame.renderer import MazeRenderer, ROOM_SIZE, WALL_THICKNESS

# ---------------------------------------------------------------------------
# Explorer registry (students add their class here)
# ---------------------------------------------------------------------------

EXPLORERS: Dict[str, Callable[[], BaseExplorer]] = {
    "Random Walk (example)": RandomWalkExplorer,
    "Wall Follower (example)": WallFollowerExplorer,
    # EXPLORERS["My Algorithm"] = MyExplorer
}

DEFAULT_CELLS_WIDE = 36
DEFAULT_CELLS_HIGH = 28

# UI colours
UI_BG = (40, 44, 52)
UI_FG = (220, 220, 220)
UI_ACCENT = (70, 140, 230)
UI_GREEN = (0, 160, 0)
UI_BTN = (60, 66, 78)
UI_BTN_HOVER = (80, 90, 110)
UI_STATUS = (180, 190, 200)


class SolveState(Enum):
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()


class Button:
    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        on_click: Callable[[], None],
        *,
        toggle_labels: Optional[Tuple[str, str]] = None,
    ) -> None:
        self.rect = rect
        self.label = label
        self.on_click = on_click
        self.toggle_labels = toggle_labels  # (play, pause) or None
        self.enabled = True
        self.hovered = False
        self.mode = 0  # for toggle: 0 = first label, 1 = second

    def set_mode(self, mode: int) -> None:
        self.mode = mode

    def current_label(self) -> str:
        if self.toggle_labels is not None:
            return self.toggle_labels[self.mode]
        return self.label

    def handle_event(self, event: pygame.event.Event) -> None:
        if not self.enabled:
            return
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.on_click()

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        color = UI_BTN_HOVER if self.hovered and self.enabled else UI_BTN
        if not self.enabled:
            color = (45, 48, 55)
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        text = font.render(self.current_label(), True, UI_FG if self.enabled else (100, 100, 100))
        tx = self.rect.x + (self.rect.w - text.get_width()) // 2
        ty = self.rect.y + (self.rect.h - text.get_height()) // 2
        surface.blit(text, (tx, ty))


class Slider:
    def __init__(self, rect: pygame.Rect, min_v: int, max_v: int, value: int) -> None:
        self.rect = rect
        self.min_v = min_v
        self.max_v = max_v
        self.value = value
        self.dragging = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self._set_from_x(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(event.pos[0])
            return True
        return False

    def _set_from_x(self, x: int) -> None:
        t = (x - self.rect.x) / max(1, self.rect.w)
        t = max(0.0, min(1.0, t))
        self.value = int(self.min_v + t * (self.max_v - self.min_v))

    def draw(self, surface: pygame.Surface) -> None:
        # track
        mid_y = self.rect.centery
        pygame.draw.line(
            surface, (100, 110, 130),
            (self.rect.x, mid_y), (self.rect.right, mid_y), 4,
        )
        t = (self.value - self.min_v) / max(1, self.max_v - self.min_v)
        kx = int(self.rect.x + t * self.rect.w)
        pygame.draw.circle(surface, UI_ACCENT, (kx, mid_y), 8)


class MazeApp:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Maze Explorer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("sans", 16)
        self.font_bold = pygame.font.SysFont("sans", 16, bold=True)
        self.font_status = pygame.font.SysFont("sans", 14)

        self.renderer = MazeRenderer()
        self.maze: Optional[Maze] = None
        self.engine: Optional[MazeEngine] = None

        self.explorer_names = list(EXPLORERS.keys())
        self.selected_explorer = 0
        self.state = SolveState.IDLE
        self.status = " "
        self._show_goal_dialog = False
        self._goal_moves = 0
        self._goal_path = 0

        self.controls_height = 56
        # Initial size; will resize after first maze
        self.screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
        self._build_controls()
        self.generate_new_maze()

    def _build_controls(self) -> None:
        y = 0  # relative to controls strip; absolute set in draw
        self.btn_new = Button(pygame.Rect(10, y, 100, 36), "New Maze", self.generate_new_maze)
        self.btn_start = Button(
            pygame.Rect(120, y, 90, 36),
            "Start",
            self.on_play_pause,
            toggle_labels=("▶ Start", "⏸ Pause"),
        )
        self.btn_reset = Button(pygame.Rect(220, y, 80, 36), "Reset", self.reset_solving)
        self.btn_reset.enabled = False
        self.slider = Slider(pygame.Rect(420, y + 8, 140, 20), 1, 100, 60)
        self.buttons: List[Button] = [self.btn_new, self.btn_start, self.btn_reset]

    def generate_new_maze(self) -> None:
        if self.engine is not None:
            self.engine.stop_current()
        self.maze = Maze(DEFAULT_CELLS_WIDE, DEFAULT_CELLS_HIGH)
        self.engine = MazeEngine(self.maze, self.renderer)
        self.engine.set_goal_listener(self._on_goal_reached)
        self.apply_speed()
        self.state = SolveState.IDLE
        self.status = "New maze ready. Pick an explorer and press Start."
        self.btn_start.set_mode(0)
        self.btn_start.enabled = True
        self.btn_reset.enabled = False
        self._resize_window()

    def _resize_window(self) -> None:
        if self.renderer.total_width == 0:
            return
        w = max(700, self.renderer.total_width + 40)
        h = self.renderer.total_height + self.controls_height + 40
        self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)

    def apply_speed(self) -> None:
        # Slider 1..100 → animation duration ~ 300 ms (slow) down to ~ 0 ms (fast)
        # Same mapping idea as the Java slider.
        v = self.slider.value
        # high slider = faster → shorter duration
        duration = int(300 * (100 - v) / 99)
        if self.engine is not None:
            self.engine.set_animation_duration_ms(duration)

    def on_play_pause(self) -> None:
        if self.engine is None:
            return
        if self.state == SolveState.IDLE:
            self.start_solving()
        elif self.state == SolveState.RUNNING:
            self.engine.pause()
            self.state = SolveState.PAUSED
            self.btn_start.set_mode(0)
            self.status = "Paused."
        elif self.state == SolveState.PAUSED:
            self.engine.resume()
            self.state = SolveState.RUNNING
            self.btn_start.set_mode(1)
            if self.status.startswith("Paused"):
                self.status = self.status.replace("Paused.", "Solving…")

    def start_solving(self) -> None:
        if self.engine is None:
            return
        name = self.explorer_names[self.selected_explorer]
        factory = EXPLORERS[name]
        explorer = factory()
        self.status = f"Solving with: {name}"
        self.state = SolveState.RUNNING
        self.btn_start.set_mode(1)
        self.btn_reset.enabled = True
        self.engine.start(explorer)

    def reset_solving(self) -> None:
        if self.engine is None:
            return
        was_running = self.engine.is_running() and not self.engine.is_paused()
        if was_running:
            name = self.explorer_names[self.selected_explorer]
            factory = EXPLORERS[name]
            explorer = factory()
            self.status = f"Solving with: {name}"
            self.state = SolveState.RUNNING
            self.btn_start.set_mode(1)
            self.engine.start(explorer)
        else:
            self.engine.reset_to_start()
            self.status = "Reset to start."
            self.state = SolveState.IDLE
            self.btn_start.set_mode(0)
            self.btn_reset.enabled = False

    def _on_goal_reached(self, move_count: int, path_length: int) -> None:
        # Called from the solver thread via the pending-goal mechanism;
        # we just store and show on the main thread.
        self._goal_moves = move_count
        self._goal_path = path_length
        self._show_goal_dialog = True

    def _finish_goal(self) -> None:
        self.state = SolveState.IDLE
        self.btn_start.set_mode(0)
        self.btn_reset.enabled = False
        self.status = (
            f"Solved! {self._goal_moves} moves attempted, "
            f"final path length {self._goal_path}."
        )
        self._show_goal_dialog = False

    def _cycle_explorer(self, delta: int) -> None:
        if self.state != SolveState.IDLE:
            return
        n = len(self.explorer_names)
        self.selected_explorer = (self.selected_explorer + delta) % n

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_LEFT:
                        self._cycle_explorer(-1)
                    elif event.key == pygame.K_RIGHT:
                        self._cycle_explorer(1)
                    elif event.key == pygame.K_SPACE:
                        self.on_play_pause()
                    elif event.key == pygame.K_n:
                        self.generate_new_maze()
                    elif event.key == pygame.K_r:
                        self.reset_solving()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Click on explorer label area to cycle
                    if self._explorer_rect().collidepoint(event.pos):
                        self._cycle_explorer(1)
                for btn in self.buttons:
                    btn.handle_event(event)
                if self.slider.handle_event(event):
                    self.apply_speed()

            # Engine / animation
            if self.engine is not None:
                goal = self.engine.poll()
                if goal is not None:
                    self._goal_moves, self._goal_path = goal
                    self._show_goal_dialog = True
                self.renderer.tick(dt)
                # Keep path in sync for trail (skip while animating so we
                # don't fight the trail-cache rebuild timing).
                if not self.renderer.is_animating():
                    self.renderer.update_path(list(self.engine.path_stack))

            if self._show_goal_dialog:
                self._finish_goal()
                # Simple status update; a full modal is overkill for pygame.
                # User can press New Maze if they want another.

            self._draw()
            pygame.display.flip()

        if self.engine is not None:
            self.engine.stop_current()
        pygame.quit()

    def _explorer_rect(self) -> pygame.Rect:
        return pygame.Rect(310, self.screen.get_height() - self.controls_height + 10, 100, 36)

    def _draw(self) -> None:
        self.screen.fill(UI_BG)
        # Maze
        maze_x = max(0, (self.screen.get_width() - self.renderer.total_width) // 2)
        maze_y = 10
        self.renderer.draw(self.screen, (maze_x, maze_y))

        # Controls strip
        strip_y = self.screen.get_height() - self.controls_height
        pygame.draw.rect(
            self.screen, (30, 34, 40),
            pygame.Rect(0, strip_y, self.screen.get_width(), self.controls_height),
        )

        # Position buttons
        for btn in self.buttons:
            btn.rect.y = strip_y + 10
            btn.draw(self.screen, self.font)

        # Explorer selector (clickable)
        name = self.explorer_names[self.selected_explorer]
        exp_rect = pygame.Rect(310, strip_y + 10, 100, 36)
        pygame.draw.rect(self.screen, UI_BTN, exp_rect, border_radius=6)
        # Truncate long names
        label = name if len(name) < 14 else name[:12] + "…"
        text = self.font.render(label, True, UI_FG)
        self.screen.blit(
            text,
            (exp_rect.x + (exp_rect.w - text.get_width()) // 2,
             exp_rect.y + (exp_rect.h - text.get_height()) // 2),
        )
        # Arrow hints
        hint = self.font_status.render("◀ ▶", True, UI_STATUS)
        self.screen.blit(hint, (exp_rect.right + 4, exp_rect.y + 10))

        # Speed label + slider
        spd = self.font_status.render("Speed", True, UI_STATUS)
        self.screen.blit(spd, (380, strip_y + 18))
        self.slider.rect.y = strip_y + 18
        self.slider.draw(self.screen)

        # Status
        status_surf = self.font_status.render(self.status, True, UI_STATUS)
        self.screen.blit(status_surf, (580, strip_y + 18))


def main() -> None:
    app = MazeApp()
    app.run()


if __name__ == "__main__":
    main()
