"""
Wires together the Maze (data), the BaseExplorer (student algorithm running
on a background thread) and the MazeRenderer (Pygame drawing on the main
thread).

Threading model mirrors the Java version:
  - All mutable path / game-over state is mutated only on the main (render)
    thread via a request queue.
  - The explorer's solve() runs on a dedicated worker thread and calls
    attempt_move(), which posts an animation request and blocks on an Event
    until the animation finishes.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Deque, List, Optional

from .base_explorer import BaseExplorer
from .cell import Cell
from .direction import Direction
from .maze import Maze
from .renderer import MazeRenderer


class MazeStoppedException(RuntimeError):
    """Raised to unwind an explorer when the maze is reset / stopped."""

    def __init__(self) -> None:
        super().__init__("The maze was reset while the explorer was still running.")


class AnimKind(Enum):
    MOVE = auto()
    BUMP = auto()


@dataclass
class AnimRequest:
    kind: AnimKind
    from_cell: Cell
    to_cell: Optional[Cell]  # None for bump
    direction: Optional[Direction]  # set for bump
    done: threading.Event
    # Path mutation is applied by the main thread before starting the anim.
    new_path: Optional[List[Cell]] = None


class MazeEngine:
    def __init__(self, maze: Maze, renderer: MazeRenderer) -> None:
        self.maze = maze
        self.renderer = renderer
        self.path_stack: Deque[Cell] = deque([maze.start])
        self.move_count = 0
        self.game_over = False
        self.paused = False
        self._pause_cond = threading.Condition()
        self._solver_thread: Optional[threading.Thread] = None
        self._goal_listener: Optional[Callable[[int, int], None]] = None
        self._anim_queue: Deque[AnimRequest] = deque()
        self._lock = threading.Lock()

        renderer.set_engine_state(maze, list(self.path_stack), maze.start)

    def set_goal_listener(self, listener: Callable[[int, int], None]) -> None:
        self._goal_listener = listener

    def set_animation_duration_ms(self, ms: int) -> None:
        self.renderer.set_animation_duration_ms(ms)

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
            # Drain pending animations so we don't block.
            with self._lock:
                while self._anim_queue:
                    req = self._anim_queue.popleft()
                    req.done.set()
        self._solver_thread = None

    def reset_to_start(self) -> None:
        self.stop_current()
        self.path_stack.clear()
        self.path_stack.append(self.maze.start)
        self.move_count = 0
        self.game_over = False
        self.paused = False
        self.renderer.set_engine_state(self.maze, list(self.path_stack), self.maze.start)

    def start(self, explorer: BaseExplorer) -> None:
        self.reset_to_start()
        explorer.bind(self)

        def runner() -> None:
            try:
                explorer.solve()
            except MazeStoppedException:
                pass
            except Exception as ex:  # noqa: BLE001
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
        current = self.path_stack[-1]
        target = current.moved(direction)
        return self.maze.is_open_cell(target)

    def attempt_move(self, direction: Direction) -> bool:
        if self.game_over or self._is_interrupted():
            raise MazeStoppedException()
        self._wait_if_paused()

        self.move_count += 1
        current = self.path_stack[-1]
        target = current.moved(direction)

        if not self.maze.is_open_cell(target):
            done = threading.Event()
            req = AnimRequest(
                kind=AnimKind.BUMP,
                from_cell=current,
                to_cell=None,
                direction=direction,
                done=done,
            )
            with self._lock:
                self._anim_queue.append(req)
            self._await(done)
            return False

        # Backing up = moving onto the cell immediately behind us.
        previous = self.path_stack[-2] if len(self.path_stack) >= 2 else None
        backing_up = previous is not None and previous == target

        if backing_up:
            new_path = list(self.path_stack)[:-1]
        else:
            new_path = list(self.path_stack) + [target]

        done = threading.Event()
        req = AnimRequest(
            kind=AnimKind.MOVE,
            from_cell=current,
            to_cell=target,
            direction=None,
            done=done,
            new_path=new_path,
        )
        with self._lock:
            self._anim_queue.append(req)
        self._await(done)

        if target == self.maze.goal:
            self.game_over = True
            # new_path was computed above; path_stack is only mutated on the
            # main thread, so use new_path for the length.
            final_moves = self.move_count
            path_len = len(new_path)
            self._pending_goal = (final_moves, path_len)
        return True

    def _await(self, event: threading.Event) -> None:
        while not event.wait(timeout=0.05):
            if self._is_interrupted() or (
                self.game_over and self.path_stack[-1] != self.maze.goal
            ):
                raise MazeStoppedException()
        if self.game_over and self.path_stack[-1] != self.maze.goal:
            raise MazeStoppedException()

    # ------------------------------------------------------------------
    # Called every frame from the main / render thread
    # ------------------------------------------------------------------

    def poll(self) -> Optional[tuple[int, int]]:
        """Process one pending animation request (if any) and return a
        pending (moves, path_len) goal event if the goal was just reached.
        """
        goal_event: Optional[tuple[int, int]] = getattr(self, "_pending_goal", None)
        if goal_event is not None:
            del self._pending_goal

        if self.renderer.is_animating():
            return goal_event

        with self._lock:
            if not self._anim_queue:
                return goal_event
            req = self._anim_queue.popleft()

        if req.kind == AnimKind.MOVE and req.new_path is not None:
            self.path_stack = deque(req.new_path)
            self.renderer.update_path(req.new_path)
            assert req.to_cell is not None
            self.renderer.animate_move(req.from_cell, req.to_cell, req.done)
        elif req.kind == AnimKind.BUMP and req.direction is not None:
            self.renderer.animate_bump(req.from_cell, req.direction, req.done)
        else:
            req.done.set()

        return goal_event

    def get_hint(self) -> float:
        cur = self.path_stack[-1]
        goal = self.maze.goal
        dr = cur.row - goal.row
        dc = cur.col - goal.col
        return math.sqrt(dr * dr + dc * dc)

    def is_at_goal(self) -> bool:
        return self.path_stack[-1] == self.maze.goal

    def get_row(self) -> int:
        return self.path_stack[-1].row

    def get_col(self) -> int:
        return self.path_stack[-1].col

    def get_move_count(self) -> int:
        return self.move_count
