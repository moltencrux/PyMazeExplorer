"""
Wires together the Maze (data), the BaseExplorer (student algorithm running
on a background thread) and the MazeRenderer (Pygame drawing on the main
thread).

Threading model:
  - The explorer runs on a worker thread and posts AnimRequests to a bounded
    queue, then continues (async). It only blocks when the queue is full
    (backpressure) or when paused / stopped.
  - Path / sprite animation is applied on the main thread in poll() + tick().
  - frames_per_move and display FPS are owned by main; the engine does not
    throttle the algorithm except via queue depth.

Exploration state:
  - Two ExplorationState instances are kept: *logical* (worker) and *visual*
    (main thread). The logical one drives algorithm queries (has_visited,
    teleport eligibility, etc.). The visual one is advanced only when an
    animation starts, so the camera and debug focus box stay consistent with
    what the player has actually seen.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Deque, List, Optional

from .base_explorer import BaseExplorer
from .cell import Cell
from .direction import Direction
from .exploration_state import ExplorationState
from .maze import Maze
from .renderer import MazeRenderer

# Max outstanding anims the worker may queue before it must wait for main.
ANIM_QUEUE_MAX = 64


class MazeStoppedException(RuntimeError):
    """Raised to unwind an explorer when the maze is reset / stopped."""

    def __init__(self) -> None:
        super().__init__("The maze was reset while the explorer was still running.")


class AnimKind(Enum):
    GOTO = auto()   # move or teleport — anim chosen by adjacency at playback
    BUMP = auto()


@dataclass
class AnimRequest:
    kind: AnimKind
    from_cell: Cell
    to_cell: Optional[Cell]  # None for bump
    direction: Optional[Direction]  # set for bump
    # Path mutation is applied by the main thread when the anim *starts*.
    new_path: Optional[List[Cell]] = None


class MazeEngine:
    def __init__(self, maze: Maze, renderer: MazeRenderer) -> None:
        self.maze = maze
        self.renderer = renderer
        self.path_stack: Deque[Cell] = deque([maze.start])

        # Logical state (worker thread) vs visual state (main / render thread).
        # Camera and debug focus use the visual frontier so they stay in
        # lock-step with the animation instead of running ahead of it.
        self.logical = ExplorationState(maze, maze.start)
        self.visual = ExplorationState(maze, maze.start)

        self.move_count = 0
        self.game_over = False
        self.paused = False
        self._pause_cond = threading.Condition()
        self._solver_thread: Optional[threading.Thread] = None
        self._goal_listener: Optional[Callable[[int, int], None]] = None
        self._anim_queue: Deque[AnimRequest] = deque()
        self._queue_cond = threading.Condition()
        self._lock = threading.Lock()
        self._pending_goal: Optional[tuple[int, int]] = None
        self._play_budget: float = 0.0  # residual edge-units across frames

        self._logic_cell = maze.start
        self._logic_path: List[Cell] = [maze.start]
        renderer.set_engine_state(maze, list(self.path_stack), maze.start)
        self.renderer.set_frontier(list(self.visual.get_frontier()))

    def set_goal_listener(self, listener: Callable[[int, int], None]) -> None:
        self._goal_listener = listener

    def set_frames_per_move(self, frames: float) -> None:
        self.renderer.set_frames_per_move(frames)

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        with self._pause_cond:
            self.paused = False
            self._pause_cond.notify_all()

    def is_paused(self) -> bool:
        return self.paused

    def is_running(self) -> bool:
        t = self._solver_thread
        return t is not None and t.is_alive() and not self.game_over

    def stop_current(self) -> None:
        t = self._solver_thread
        if t is not None and t.is_alive():
            t.interrupt = True  # type: ignore[attr-defined]
            # Also clear any pending pause so the thread can exit.
            with self._pause_cond:
                self.paused = False
                self._pause_cond.notify_all()
            with self._queue_cond:
                self._anim_queue.clear()
                self._queue_cond.notify_all()
        self._solver_thread = None

    def reset_to_start(self) -> None:
        self.stop_current()
        self.path_stack.clear()
        self.path_stack.append(self.maze.start)
        self.logical.reset(self.maze.start)
        self.visual.reset(self.maze.start)
        self.move_count = 0
        self.game_over = False
        self.paused = False
        self._pending_goal = None
        self._play_budget = 0.0
        self._logic_cell = self.maze.start
        self._logic_path = [self.maze.start]
        self.renderer.set_engine_state(self.maze, list(self.path_stack), self.maze.start)
        self.renderer.set_frontier(list(self.visual.get_frontier()))

    def start(self, explorer: BaseExplorer) -> None:
        self.reset_to_start()
        explorer.bind(self)

        def runner() -> None:
            try:
                explorer.solve()
            except MazeStoppedException:
                pass
            except Exception:  # noqa: BLE001
                print("Explorer threw an exception:")
                import traceback
                traceback.print_exc()

        t = threading.Thread(target=runner, name="MazeSolverThread", daemon=True)
        t.interrupt = False  # type: ignore[attr-defined]
        self._solver_thread = t
        t.start()

    def _wait_if_paused(self) -> None:
        with self._pause_cond:
            while self.paused and not self.game_over and not self._is_interrupted():
                self._pause_cond.wait(timeout=0.05)
        if self.game_over or self._is_interrupted():
            raise MazeStoppedException()

    def _is_interrupted(self) -> bool:
        t = threading.current_thread()
        return getattr(t, "interrupt", False)

    def can_move(self, direction: Direction) -> bool:
        current = self._logical_cell()
        target = current.moved(direction)
        return self.logical.can_visit(target)

    def has_visited(self, cell: Cell) -> bool:
        return self.logical.is_visited(cell)

    def _logical_cell(self) -> Cell:
        """Cell the algorithm is at (may be ahead of the sprite)."""
        with self._lock:
            return self._logic_cell

    def _enqueue(self, req: AnimRequest) -> None:
        """Post an anim; block only while the queue is at capacity."""
        with self._queue_cond:
            while len(self._anim_queue) >= ANIM_QUEUE_MAX:
                if self._is_interrupted() or self.game_over:
                    raise MazeStoppedException()
                self._queue_cond.wait(timeout=0.05)
            if self._is_interrupted():
                raise MazeStoppedException()
            self._anim_queue.append(req)

    def attempt_move(self, direction: Direction) -> bool:
        if self.game_over or self._is_interrupted():
            raise MazeStoppedException()
        self._wait_if_paused()

        self.move_count += 1
        current = self._logical_cell()
        target = current.moved(direction)

        if not self.maze.is_open_cell(target):
            self._enqueue(
                AnimRequest(
                    kind=AnimKind.BUMP,
                    from_cell=current,
                    to_cell=None,
                    direction=direction,
                )
            )
            return False

        previous = None
        with self._lock:
            # Maintain a worker-side path for logical backtracking.
            if len(self._logic_path) >= 2 and self._logic_path[-2] == target:
                self._logic_path.pop()
            else:
                self._logic_path.append(target)
            self._logic_cell = target
            new_path = list(self._logic_path)

        # Only the first time we step onto a cell do we expand the logical
        # visited set / frontier. Re-visiting (backtracking) is a no-op.
        self.logical.visit(target)

        self._enqueue(
            AnimRequest(
                kind=AnimKind.GOTO,
                from_cell=current,
                to_cell=target,
                direction=None,
                new_path=new_path,
            )
        )

        if target == self.maze.goal:
            self.game_over = True
            self._pending_goal = (self.move_count, len(new_path))
        return True

    def attempt_teleport(self, cell: Cell) -> bool:
        """
        Instantly move the sprite to a previously visited cell.
        Returns True on success, False if the cell has never been visited
        (or is the current cell — treated as a no-op success).
        """
        if self.game_over or self._is_interrupted():
            raise MazeStoppedException()
        self._wait_if_paused()

        current = self._logical_cell()
        if cell == current:
            return True
        if not self.logical.is_visited(cell):
            return False

        # Teleport does not count as a "move attempt" for the counter, but
        # we still animate so the UI stays in sync.
        new_path = [cell]
        with self._lock:
            self._logic_path = [cell]
            self._logic_cell = cell

        self._enqueue(
            AnimRequest(
                kind=AnimKind.GOTO,
                from_cell=current,
                to_cell=cell,
                direction=None,
                new_path=new_path,
            )
        )

        if cell == self.maze.goal:
            self.game_over = True
            self._pending_goal = (self.move_count, len(new_path))
        return True

    def mark_explored(self, cell: Cell) -> None:
        """Explicitly mark a cell as explored (for frontier-style algorithms)."""
        self.renderer.mark_explored(cell)

    def mark_explored_many(self, cells: List[Cell]) -> None:
        self.renderer.mark_explored_many(cells)

    def set_show_sprite(self, show: bool) -> None:
        """Show or hide the red agent sprite."""
        self.renderer.set_show_sprite(show)

    # ------------------------------------------------------------------
    # Main / render thread
    # ------------------------------------------------------------------

    def has_pending_anims(self) -> bool:
        with self._queue_cond:
            return bool(self._anim_queue)

    def _start_next_anim(self) -> bool:
        """Pop one queue item and start its animation. Returns False if empty."""
        with self._queue_cond:
            if not self._anim_queue:
                return False
            req = self._anim_queue.popleft()
            self._queue_cond.notify_all()

        if req.kind == AnimKind.GOTO and req.to_cell is not None:
            if req.new_path is not None:
                self.path_stack = deque(req.new_path)

            # Advance the *visual* exploration state in lock-step with the
            # animation. This keeps the camera / debug focus box consistent
            # with what the player has actually seen.
            self.visual.visit(req.to_cell)
            self.renderer.set_frontier(list(self.visual.get_frontier()))
            self.renderer.mark_explored(req.to_cell)

            # Slide if graph-adjacent; hop otherwise — independent of move vs teleport API.
            if self._cells_adjacent(req.from_cell, req.to_cell):
                self.renderer.animate_slide(req.from_cell, req.to_cell)
            else:
                self.renderer.animate_hop(req.from_cell, req.to_cell)
        elif req.kind == AnimKind.BUMP and req.direction is not None:
            self.renderer.animate_bump(req.from_cell, req.direction)
        else:
            return False
        return True

    @staticmethod
    def _cells_adjacent(a: Cell, b: Cell) -> bool:
        for d in Direction:
            if a.moved(d) == b:
                return True
        return False

    def advance_playback(self) -> Optional[tuple[int, int]]:
        """Drive anim queue for one display frame using a unified edge budget.

        Each frame adds 1/fpm edge-units. fpm>1 → slow multi-frame moves;
        fpm<1 → multiple edges per frame. Residual budget carries across frames.
        """
        goal_event = self._pending_goal
        if goal_event is not None:
            self._pending_goal = None

        # Hold the queue immediately on pause (solver already blocks on _wait_if_paused).
        if self.paused:
            return goal_event

        fpm = self.renderer.frames_per_move()
        self._play_budget += 1.0 / fpm

        # Safety: never process more than this many edges in one frame
        max_edges = 32
        edges = 0

        while self._play_budget > 1e-9 and edges < max_edges:
            if not self.renderer.is_animating():
                if not self._start_next_anim():
                    break  # queue empty
            # Advance current anim; unused delta stays in budget for next edge
            before = self._play_budget
            self._play_budget = self.renderer.advance_anim(self._play_budget)
            consumed = before - self._play_budget
            if consumed <= 1e-12 and self.renderer.is_animating():
                # No progress (shouldn't happen) — avoid spin
                break
            if not self.renderer.is_animating():
                edges += 1  # completed one edge this frame
            else:
                # Mid-lerp; wait for next display frame
                break

        # If idle with empty queue, don't pile up unbounded budget
        if not self.renderer.is_animating() and not self.has_pending_anims():
            self._play_budget = 0.0

        return goal_event

    def poll(self) -> Optional[tuple[int, int]]:
        """Backward-compatible alias for advance_playback()."""
        return self.advance_playback()

    def get_hint(
self) -> float:
        cur = self._logical_cell()
        goal = self.maze.goal
        dr = cur.row - goal.row
        dc = cur.col - goal.col
        # return math.sqrt(dr * dr + dc * dc)
        return dr + dc

    def is_at_goal(self) -> bool:
        return self._logical_cell() == self.maze.goal

    def get_row(self) -> int:
        return self._logical_cell().row

    def get_col(self) -> int:
        return self._logical_cell().col

    def get_move_count(self) -> int:
        return self.move_count

