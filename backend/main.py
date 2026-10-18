"""Mars Mission Planner - backend (whole-Mars version).

The map is a global grid built from USGS MOLA elevation data.
Rows run north to south (latitude), columns run west to east (longitude -180 to 180).
Grid values: 0 = safe terrain, 1 = wall (too steep: crater rims, scarps, volcano flanks).
The map wraps around east-west and every step is measured in real kilometres.
"""
import heapq
import json
import math
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

Cell = tuple[int, int]

SAFE = 0
WALL = 1
MARS_RADIUS_KM = 3396.19

# data/grid.json sits next to the backend/ folder. Create it with backend/build_grid.py.
GRID_PATH = Path(__file__).resolve().parent.parent / "data" / "grid.json"
GRID = json.loads(GRID_PATH.read_text())
ROWS = len(GRID)
COLS = len(GRID[0])
CELL_DEG = 360.0 / COLS
if any(len(row) != COLS for row in GRID) or ROWS != round(180.0 / CELL_DEG):
    raise RuntimeError(
        "data/grid.json must cover the whole planet: twice as many columns as rows "
        "(for example 180 rows x 360 columns). Run backend/build_grid.py to create it."
    )
ARC_KM = math.radians(CELL_DEG) * MARS_RADIUS_KM  # one cell edge along a meridian


def lat_of_row(row: int) -> float:
    return 90.0 - (row + 0.5) * CELL_DEG


def lon_of_col(col: int) -> float:
    return -180.0 + (col + 0.5) * CELL_DEG


def step_km(a: Cell, b: Cell) -> float:
    """Real length of one step. East-west steps get shorter toward the poles."""
    if a[0] != b[0]:
        return ARC_KM
    return ARC_KM * math.cos(math.radians(lat_of_row(a[0])))


def great_circle_km(a: Cell, b: Cell) -> float:
    """Shortest distance over the surface; used as the A* heuristic (never too high)."""
    lat1, lat2 = math.radians(lat_of_row(a[0])), math.radians(lat_of_row(b[0]))
    dlon = math.radians(lon_of_col(b[1]) - lon_of_col(a[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * MARS_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def neighbors(cell: Cell):
    row, col = cell
    if row > 0:
        yield (row - 1, col)
    if row < ROWS - 1:
        yield (row + 1, col)
    yield (row, (col - 1) % COLS)  # the map wraps around east-west
    yield (row, (col + 1) % COLS)


def a_star(start: Cell, end: Cell):
    """Shortest safe route from start to end, stepping only on 0 cells.

    Returns (cells, distance_km), or None when no route exists.
    Raises ValueError if start or end is outside the grid or on a wall.
    """
    for name, (row, col) in (("start", start), ("end", end)):
        if not (0 <= row < ROWS and 0 <= col < COLS):
            raise ValueError(f"{name} {(row, col)} is outside the grid")
        if GRID[row][col] != SAFE:
            raise ValueError(f"{name} {(row, col)} is on a wall (too steep)")

    came_from: dict[Cell, Cell] = {}
    best: dict[Cell, float] = {start: 0.0}
    heap = [(great_circle_km(start, end), 0.0, start)]  # (f, g, cell)

    while heap:
        _, g, current = heapq.heappop(heap)

        if current == end:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, g

        if g > best.get(current, g):  # stale heap entry
            continue

        for nxt in neighbors(current):
            if GRID[nxt[0]][nxt[1]] != SAFE:
                continue
            new_g = g + step_km(current, nxt)
            if new_g < best.get(nxt, math.inf):
                came_from[nxt] = current
                best[nxt] = new_g
                heapq.heappush(heap, (new_g + great_circle_km(nxt, end), new_g, nxt))

    return None


class Point(BaseModel):
    lat: float  # -90 (south) to 90 (north)
    lon: float  # -180 (west) to 180 (east)


class RouteRequest(BaseModel):
    start: Point
    end: Point


def to_cell(point: Point, name: str) -> Cell:
    if not (-90 <= point.lat <= 90 and -180 <= point.lon <= 180):
        raise HTTPException(
            status_code=400, detail=f"{name}: lat must be -90 to 90 and lon -180 to 180"
        )
    row = min(ROWS - 1, int((90.0 - point.lat) / CELL_DEG))
    col = min(COLS - 1, int((point.lon + 180.0) / CELL_DEG))
    return row, col


app = FastAPI(title="Mars Mission Planner API")


@app.get("/grid")
def get_grid():
    """The terrain map, so the frontend can draw it and show the difficult terrain."""
    return {
        "rows": ROWS,
        "cols": COLS,
        "cell_deg": CELL_DEG,
        "lat_top": 90,
        "lon_left": -180,
        "grid": GRID,
    }


@app.post("/calculate-route")
def calculate_route(req: RouteRequest):
    start = to_cell(req.start, "start")
    end = to_cell(req.end, "end")

    try:
        result = a_star(start, end)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))

    if result is None:
        return {"found": False, "steps": 0, "distance_km": 0, "cells": [], "path": []}

    cells, distance_km = result
    return {
        "found": True,
        "steps": len(cells) - 1,
        "distance_km": round(distance_km, 1),
        "cells": [list(c) for c in cells],  # [row, col]
        "path": [[round(lat_of_row(r), 3), round(lon_of_col(c), 3)] for r, c in cells],  # [lat, lon]
    }