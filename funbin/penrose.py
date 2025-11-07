import random

from pynrose import Grid, Tiling, Vector

from funbin.geometry import Box, Polygon


def penrose_P3(n_horiz: int, n_vert: int, random_seed: int | None = None) -> list[Polygon]:
    tiling = Tiling(rnd=random.Random(random_seed))
    safety_margin_factor = 1.3
    grid = Grid(Vector(0, 0), Vector(int(n_horiz * safety_margin_factor), int(n_vert * safety_margin_factor)))
    polys = [Polygon.from_rhombus(r) for r in tiling.rhombii(grid.cell(0, 0)) if r]
    box = Box.bounding_all(polys).resized(1 / safety_margin_factor)
    polys = [p.clipped(to=box) for p in polys]
    polys = [p for p in polys if p.area > 1e-9]
    return polys
