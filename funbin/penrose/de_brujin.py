import random

import numpy as np
from pynrose import Grid, Tiling, Vector

from funbin.geometry import (
    Box,
    Polygon,
    clipped_to_box,
)


def penrose_P3_de_Brujin(bins: tuple[int, int]) -> list[Polygon]:
    """Generation with de Brujin method facilitated by pynrose library"""
    n_horiz, n_vert = bins
    tiling = Tiling(rnd=random.Random(np.random.random()))
    safety_margin_factor = 1.3
    grid = Grid(Vector(0, 0), Vector(int(n_horiz * safety_margin_factor), int(n_vert * safety_margin_factor)))
    polys = [Polygon.from_rhombus(r) for r in tiling.rhombii(grid.cell(0, 0)) if r]
    return clipped_to_box(
        polys,
        box=Box.bounding_all(polys).resized(1 / safety_margin_factor),
    )
