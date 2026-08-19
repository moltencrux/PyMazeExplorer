# Maze Explorer (Python / Pygame)

A Python + Pygame port of the [Java MazeExplorer](https://github.com/moltencrux/MazeExplorer) teaching tool, including the **variable-thickness walls** layout from the `dev/var-width-walls` branch.

A random block-based maze is generated and shown on screen. Students implement a maze-solving algorithm by subclassing `BaseExplorer`; the app animates their agent moving through the maze, leaves a trail behind it (which un-marks itself when the agent backtracks), and reports success when the goal is reached.

## Requirements

* Python 3.10+
* pygame 2.5+

```bash
pip install -r requirements.txt
```

## Run

```bash
cd python_maze_explorer
python main.py
```

## Controls

| Input | Action |
| --- | --- |
| **Start / Pause** button (or Space) | Start solving / pause / resume |
| **Reset** | Rewind explorer to the start (or restart if currently running) |
| **New Maze** (or N) | Generate a fresh maze |
| **◀ ▶** / click explorer name / Left·Right arrows | Cycle explorer algorithm |
| Speed slider | Animation speed (left = slow, right = fast) |
| Esc | Quit |

## How it works

| Module | Purpose |
| --- | --- |
| `mazegame/maze.py` | Generates a random perfect maze (recursive backtracker) and answers `is_open` queries. |
| `mazegame/direction.py`, `cell.py` | Small value types. |
| `mazegame/base_explorer.py` | **The class students subclass.** Exposes `move_up/down/left/right()`, `can_move_*()`, `get_hint()`, `is_at_goal()`, `get_row()/get_col()`. No direct access to the maze layout. |
| `mazegame/maze_engine.py` | Glue between maze, explorer, and renderer. Runs `solve()` on a background thread; each move blocks until its animation finishes. |
| `mazegame/renderer.py` | Pygame rendering with variable-thickness cells (even indices = thin walls 4 px, odd indices = rooms 26 px). Trail + smooth sprite animation. |
| `main.py` | Window, buttons, explorer picker, speed slider. |
| `mazegame/explorers/` | Example strategies: Random Walk, Wall Follower (right-hand rule), iterative DFS. |

## The assignment (for students)

1. Create a new class that extends `BaseExplorer` (see `mazegame/explorers/random_walk.py` for the simplest example).
2. Implement `solve()`. Inside it you can use:
   - `move_up()`, `move_down()`, `move_left()`, `move_right()` — each returns `True` if the move succeeded and `False` if you hit a wall (you did **not** move in that case).
   - `can_move_up()`, `can_move_down()`, `can_move_left()`, `can_move_right()` — probe without moving or animating.
   - `is_at_goal()`, `get_row()`, `get_col()`, `get_hint()` (Euclidean distance to goal).
3. Register your class in `main.py`:

   ```python
   EXPLORERS["My Algorithm"] = MyExplorer
   ```

4. Run the app, pick your algorithm, and press **Start**.

### Notes on the API

* `solve()` runs on its own thread, so a plain `while` loop with blocking move calls is fine — it will not freeze the UI.
* Moves are one square at a time in one of the four cardinal directions only.
* You never get the maze’s wall grid directly. This is intentional: the exercise is about exploring and remembering, not reading a solved map.
* The trail shown on screen simply mirrors your current path: stepping forward marks a new square; stepping back onto the square you just came from unmarks it.

## Variable-thickness walls

Matches the Java `dev/var-width-walls` branch:

* Even row/col indices → thin wall/opening strips (`WALL_THICKNESS = 4` px)
* Odd indices → room cells (`ROOM_SIZE = 26` px)

Layout uses cumulative offsets so animation, trail, and start/goal markers stay correctly centred.
