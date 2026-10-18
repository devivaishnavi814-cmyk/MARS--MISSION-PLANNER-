"""Build data/grid.json (the whole of Mars) from the USGS MOLA elevation map.

Run from the repo root:
    pip install numpy rasterio
    python backend/build_grid.py

Optional settings:  --cell-deg 1  --slope 15  --share 0.05

Each grid cell covers cell-deg x cell-deg degrees. A cell becomes a wall (1) when at least
`share` of its elevation pixels are steeper than `slope` degrees, otherwise it is safe (0).
Rows run north to south, columns west to east (-180 to 180).
"""
import argparse
import json
import math
import urllib.request
from pathlib import Path

import numpy as np

DEM_URL = "https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif"
DEM_FILE = "/tmp/mars_mola_dem_463m.tif"  # outside the repo: the file is about 2 GB
MARS_RADIUS_M = 3396190.0
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "grid.json"


def download(url, dest):
    dest = Path(dest)
    if dest.exists():
        print("using the file already downloaded:", dest)
        return
    part = dest.with_suffix(".part")
    print("downloading about 2 GB, this takes a few minutes...")
    done, total, next_report = 0, 0, 0
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (mars-mission-planner)"})
    with urllib.request.urlopen(req) as resp, open(part, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if done >= next_report:
                print("  %d MB of %d MB" % (done // 2**20, total // 2**20), flush=True)
                next_report += 200 * 2**20
    if total and done != total:
        raise SystemExit("download was cut off, please run the command again")
    part.rename(dest)


def build_grid(read_rows, width, height, cell_deg, slope_deg, share):
    """read_rows(a, b) must return elevation rows a..b (metres, NaN where missing)."""
    if height * 2 != width:
        raise ValueError("expected a global map that is twice as wide as it is tall")
    px_per_deg = width / 360.0
    block = px_per_deg * cell_deg
    if abs(block - round(block)) > 1e-9 or abs(180 / cell_deg - round(180 / cell_deg)) > 1e-9:
        raise ValueError("cell-deg must divide 180 and give a whole number of pixels")
    block = int(round(block))
    n_rows, n_cols = height // block, width // block

    dy = MARS_RADIUS_M * math.radians(1.0 / px_per_deg)  # metres per pixel, north-south
    steep_share = np.zeros((n_rows, n_cols), dtype=np.float32)

    for i in range(n_rows):
        r0, r1 = i * block, (i + 1) * block
        a, b = max(r0 - 1, 0), min(r1 + 1, height)  # one extra row each side for the slope
        z = read_rows(a, b)
        lat = 90.0 - (np.arange(a, b) + 0.5) / px_per_deg
        # metres per pixel east-west; clamped so the pole rows do not blow up
        dx = dy * np.maximum(np.cos(np.radians(lat)), 0.2)
        gy = np.gradient(z, dy, axis=0)
        gx = (np.roll(z, -1, axis=1) - np.roll(z, 1, axis=1)) / (2.0 * dx[:, None])
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))[r0 - a : r0 - a + block]
        steep = (slope >= slope_deg).reshape(block, n_cols, block)
        steep_share[i] = steep.mean(axis=(0, 2))
        if i % 20 == 0:
            print("  row %d of %d" % (i + 1, n_rows), flush=True)

    return (steep_share >= share).astype(int), steep_share


def main():
    ap = argparse.ArgumentParser(description="Build data/grid.json for the whole of Mars")
    ap.add_argument("--cell-deg", type=float, default=1.0, help="cell size in degrees (1 = about 59 km)")
    ap.add_argument("--slope", type=float, default=15.0, help="steepness in degrees that counts as steep")
    ap.add_argument("--share", type=float, default=0.05, help="share of steep pixels that makes a cell a wall")
    ap.add_argument("--url", default=DEM_URL)
    ap.add_argument("--dem", default=DEM_FILE)
    args = ap.parse_args()

    try:
        import rasterio
        from rasterio.windows import Window
    except ModuleNotFoundError:
        raise SystemExit("first run:  pip install numpy rasterio")

    download(args.url, args.dem)

    with rasterio.open(args.dem) as src:
        width, height = src.width, src.height
        print("elevation map: %d x %d pixels" % (width, height))

        def read_rows(a, b):
            arr = src.read(1, window=Window(0, a, width, b - a), masked=True)
            return np.ma.filled(arr.astype("float32"), np.nan)

        grid, share = build_grid(read_rows, width, height, args.cell_deg, args.slope, args.share)

    walls = 100.0 * grid.mean()
    print("grid: %d rows x %d columns, cell about %.0f km" % (
        grid.shape[0], grid.shape[1], math.radians(args.cell_deg) * MARS_RADIUS_M / 1000))
    print("walls: %.1f%% of cells" % walls)
    print("steep share per cell, 50/90/99th percentile: %.1f%% / %.1f%% / %.1f%%" % tuple(
        100 * np.percentile(share, [50, 90, 99])))
    if walls < 3:
        print("hint: very few walls. Try --slope 10 or --share 0.02")
    elif walls > 40:
        print("hint: too many walls. Try --slope 20 or --share 0.10")

    OUT_PATH.parent.mkdir(exist_ok=True)
    rows = ",\n".join(json.dumps(r.tolist(), separators=(",", ":")) for r in grid)
    OUT_PATH.write_text("[\n" + rows + "\n]\n")
    print("saved", OUT_PATH)


if __name__ == "__main__":
    main()