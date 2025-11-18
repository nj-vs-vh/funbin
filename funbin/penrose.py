import random

from pynrose import Grid, Tiling, Vector

from funbin.geometry import Box, Polygon, clipped_to_box


def penrose_P3(n_horiz: int, n_vert: int, random_seed: int | None = None) -> list[Polygon]:
    tiling = Tiling(rnd=random.Random(random_seed))
    safety_margin_factor = 1.3
    grid = Grid(Vector(0, 0), Vector(int(n_horiz * safety_margin_factor), int(n_vert * safety_margin_factor)))
    polys = [Polygon.from_rhombus(r) for r in tiling.rhombii(grid.cell(0, 0)) if r]
    return clipped_to_box(
        polys,
        box=Box.bounding_all(polys).resized(1 / safety_margin_factor),
    )
