import random

from pynrose import Grid, Tiling, Vector

from funbin.models import Polygon


def penrose_P3(n_horiz: int, n_vert: int, random_seed: int | None = None) -> list[Polygon]:
    tiling = Tiling(rnd=random.Random(random_seed))
    grid = Grid(Vector(0, 0), Vector(n_horiz, n_vert))
    # FIXME: find fitting box and crop side polygons?
    return [Polygon.from_rhombus(r) for r in tiling.rhombii(grid.cell(0, 0)) if r]
