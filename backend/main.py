"""Mars Mission Planner - backend.

A* route finding over a terrain grid.
Grid values: 0 = safe terrain, 1 = wall (crater / mountain).
Coordinates are (row, col) indexes into the grid.
"""
import heapq
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

Coord = tuple[int, int]

SAFE = 0
WALL = 1

# Up, down, left, right. Diagonal moves are not allowed.
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# data/grid.json sits next to the backend/ folder in the repo.
GRID_PATH = Path(__file__).resolve().parent.parent / "data" / "grid.json"
DEFAULT_GRID = json.loads(GRID_PATH.read_text())


def manhattan(a: Coord, b: Coord) -> int:
    """Heuristic: straight-line grid distance, never overestimates for 4-way moves."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(start: Coord, end: Coord, grid: list[list[int]]) -> Optional[list[Coord]]:
    """Return the shortest path from start to end as a list of (row, col),
    stepping only on 0 cells. Return None when no path exists.

    Raises ValueError if start or end is outside the grid or on a wall.
    """

    def in_bounds(p: Coord) -> bool:
        r, c = p
        return 0 <= r < len(grid) and 0 <= c < len(grid[r])

    if not in_bounds(start):
        raise ValueError(f"start {start} is outside the grid")
    if not in_bounds(end):
        raise ValueError(f"end {end} is outside the grid")
    if grid[start[0]][start[1]] != SAFE:
        raise ValueError(f"start {start} is on a wall")
    if grid[end[0]][end[1]] != SAFE:
        raise ValueError(f"end {end} is on a wall")

    came_from: dict[Coord, Coord] = {}
    g_score: dict[Coord, int] = {start: 0}
    # Heap items: (f_score, g_score, node)
    open_heap: list[tuple[int, int, Coord]] = [(manhattan(start, end), 0, start)]

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        if current == end:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        # Skip stale heap entries for a node we already reached more cheaply.
        if g > g_score.get(current, g):
            continue

        for dr, dc in MOVES:
            neighbor = (current[0] + dr, current[1] + dc)
            if not in_bounds(neighbor) or grid[neighbor[0]][neighbor[1]] != SAFE:
                continue

            tentative_g = g + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + manhattan(neighbor, end)
                heapq.heappush(open_heap, (f, tentative_g, neighbor))

    return None


class RouteRequest(BaseModel):
    start: tuple[int, int]
    end: tuple[int, int]
    grid: Optional[list[list[int]]] = None  # falls back to data/grid.json


app = FastAPI(title="Mars Mission Planner API")


@app.post("/calculate-route")
def calculate_route(req: RouteRequest):
    grid = req.grid if req.grid is not None else DEFAULT_GRID

    try:
        path = a_star(req.start, req.end, grid)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))

    if path is None:
        return {"found": False, "path": [], "steps": 0}

    return {
        "found": True,
        "path": [list(p) for p in path],
        "steps": len(path) - 1,
    }
