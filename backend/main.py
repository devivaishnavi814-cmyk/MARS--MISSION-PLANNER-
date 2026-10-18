from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import heapq
from typing import Optional

SAFE = 0
WALL = 1

# up, down, left, right
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(start, end, grid):
    def in_bounds(p):
        r, c = p
        return 0 <= r < len(grid) and 0 <= c < len(grid[r])

    if not in_bounds(start) or not in_bounds(end):
        raise ValueError("start or end is outside the grid")
    if grid[start[0]][start[1]] != SAFE or grid[end[0]][end[1]] != SAFE:
        raise ValueError("start or end is on a wall")

    came_from = {}
    g_score = {start: 0}
    open_heap = [(manhattan(start, end), 0, start)]

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        if current == end:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        if g > g_score.get(current, g):
            continue

        for dr, dc in MOVES:
            neighbor = (current[0] + dr, current[1] + dc)
            if not in_bounds(neighbor) or grid[neighbor[0]][neighbor[1]] != SAFE:
                continue
            new_g = g + 1
            if new_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = new_g
                heapq.heappush(
                    open_heap, (new_g + manhattan(neighbor, end), new_g, neighbor)
                )

    return None

DEFAULT_GRID = [
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
]


class RouteRequest(BaseModel):
    start: tuple[int, int]
    end: tuple[int, int]
    grid: Optional[list[list[int]]] = None


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