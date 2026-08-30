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
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Deque, List, Optional, Set

from .base_explorer import BaseExplorer
from .cell import Cell
from .direction import Direction
from .maze import Maze
from .renderer import MazeRenderer

# Max outstanding anims the worker may queue before it must wait for main.
ANIM_QUEUE_MAX = 64


class MazeStoppedException(RuntimeError):
    """Raised to unwind an explorer when the maze is reset / stopped."""

    def __init__(self) -> None:
        super().__init__("The maze was reset while the explorer was still running.")


class AnimKind(Enum):
    MOVE = auto()
    BUMP = auto()
    TELEPORT = auto()


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
        # Every cell the explorer has successfully stepped onto (or started on).
        # teleport() is only allowed to these cells.
        self.visited: Set[Cell] = {maze.start}
        self.visited_open: Set[Cell] = {maze.start}
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

        self._logic_cell = maze.start
        self._logic_path: List[Cell] = [maze.start]
        renderer.set_engine_state(maze, list(self.path_stack), maze.start)
        self._recompute_frontier()

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
        self.visited = {self.maze.start}
        self.visited_open = {self.maze.start}
        self.move_count = 0
        self.game_over = False
        self.paused = False
        self._pending_goal = None
        self._logic_cell = self.maze.start
        self._logic_path = [self.maze.start]
        self.renderer.set_engine_state(self.maze, list(self.path_stack), self.maze.start)
        self._recompute_frontier()

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
        return self.maze.is_open_cell(target)

    def has_visited(self, cell: Cell) -> bool:
        return cell in self.visited

    def _logical_cell(self) -> Cell:
        """Cell the algorithm is at (may be ahead of the sprite)."""
        with self._lock:
            return self._logic_cell

    def _recompute_frontier(self) -> None:
        """
        Open leaves = visited cells that still have an unvisited open neighbour.
        Derived from the maze + visited set so explorers never manage a frontier.
        """
        frontier: List[Cell] = []
        dead_ends: Set[Cell] = set()
        for cell in self.visited_open:
            for direction in Direction:
                neighbour = cell.moved(direction)
                if self.maze.is_open_cell(neighbour) and neighbour not in self.visited:
                    frontier.append(cell)
                    break
            else:
                dead_ends.add(cell)
        self.visited_open -= dead_ends
        self.renderer.set_frontier(frontier)

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

        self.visited.add(target)
        self.visited_open.add(target)
        self._recompute_frontier()

        self._enqueue(
            AnimRequest(
                kind=AnimKind.MOVE,
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
        if cell not in self.visited:
            return False

        # Teleport does not count as a "move attempt" for the counter, but
        # we still animate so the UI stays in sync.
        new_path = [cell]
        with self._lock:
            self._logic_path = [cell]
            self._logic_cell = cell

        self._enqueue(
            AnimRequest(
                kind=AnimKind.TELEPORT,
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

    def poll(self) -> Optional[tuple[int, int]]:
        """Start the next queued anim if the renderer is idle. Return goal event if any."""
        goal_event = self._pending_goal
        if goal_event is not None:
            self._pending_goal = None

        if self.renderer.is_animating():
            return goal_event

        with self._queue_cond:
            if not self._anim_queue:
                return goal_event
            req = self._anim_queue.popleft()
            self._queue_cond.notify_all()  # worker may be waiting for space

        if req.kind == AnimKind.MOVE and req.new_path is not None:
            self.path_stack = deque(req.new_path)
            self.renderer.update_path(req.new_path)
            # Mark every cell we step onto (including backtracks) so the
            # exploration overlay stays alive for sequential agents.
            if req.to_cell is not None:
                self.renderer.mark_explored(req.to_cell)
                self.renderer.animate_move(req.from_cell, req.to_cell)
        elif req.kind == AnimKind.TELEPORT and req.new_path is not None:
            self.path_stack = deque(req.new_path)
            self.renderer.update_path(req.new_path)
            if req.to_cell is not None:
                self.renderer.mark_explored(req.to_cell)
                self.renderer.animate_teleport(req.from_cell, req.to_cell)
        elif req.kind == AnimKind.BUMP and req.direction is not None:
            self.renderer.animate_bump(req.from_cell, req.direction)

        return goal_event

    def get_hint(self) -> float:
        cur = self._logical_cell()
        goal = self.maze.goal
        dr = cur.row - goal.row
        dc = cur.col - goal.col
        return math.sqrt(dr * dr + dc * dc)

    def is_at_goal(self) -> bool:
        return self._logical_cell() == self.maze.goal

    def get_row(self) -> int:
        return self._logical_cell().row

    def get_col(self) -> int:
        return self._logical_cell().col

    def get_move_count(self) -> int:
        return self.move_count

