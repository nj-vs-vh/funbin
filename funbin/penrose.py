import cmath
import math
import random
from typing import Literal

import numpy as np
from pynrose import Grid, Tiling, Vector

from funbin.geometry import Box, Point, Polygon, SpatialIndex, are_segments_close, clipped_to_box

PHI = (5**0.5 + 1) / 2  # Golden ratio


Penrose2TilingKind = Literal["P2", "P3"]
TriangleKind = Literal[0, 1]  # thin, thick
Triangle = tuple[TriangleKind, complex, complex, complex]


def penrose_tiling(
    kind: Penrose2TilingKind,
    divisions: int,
    include_incomplete_tiles: bool = False,
    base_triangles: int = 5,
) -> list[Polygon]:
    tiling = penrose_tiling_triangles(kind=kind, divisions=divisions, base_triangles=base_triangles)
    index = SpatialIndex.from_polygons(tiling, bbox_resize_factor=1.05)

    merged = set[int]()
    to_merge = list[tuple[int, int]]()
    for id, triangle in enumerate(tiling):
        if id in merged:
            continue
        *_, edge_1 = triangle.edges  # last edge is to be merged
        for other_id, other in index.candidate_tiles(edge_1[0]):
            if other_id == id or other_id in merged:
                continue
            *_, edge_2 = other.edges  # last edge is to be merged
            if are_segments_close(edge_1, edge_2):
                merged.add(id)
                merged.add(other_id)
                to_merge.append((id, other_id))
                break

    res: list[Polygon] = []
    for id1, id2 in to_merge:
        res.append(
            Polygon(
                verts=np.concatenate(
                    (
                        tiling[id1].verts,
                        [tiling[id2].verts[1, :]],
                    )
                )
            )
        )

    if include_incomplete_tiles:
        for id, tile in enumerate(tiling):
            if id not in merged:
                res.append(tile)

    return res


def penrose_tiling_triangles(kind: Penrose2TilingKind, divisions: int, base_triangles: int = 5) -> list[Polygon]:
    # Create first layer of triangles
    triangles: list[Triangle] = []
    for i in range(base_triangles * 2):
        v2 = cmath.rect(1, (2 * i - 1) * math.pi / (base_triangles * 2))
        v3 = cmath.rect(1, (2 * i + 1) * math.pi / (base_triangles * 2))
        if i % 2 == 0:
            v2, v3 = v3, v2  # Mirror every other triangle

        if kind == "P2":
            triangles.append((0, v2, 0j, v3))
        else:
            triangles.append((0, 0, v2, v3))

    for i in range(divisions):
        new_triangles: list[Triangle] = []
        for shape, v1, v2, v3 in triangles:
            if kind == "P2":
                if shape == 0:
                    # Subdivide red (sharp isosceles) (half kite) triangle
                    p1 = v1 + (v2 - v1) / PHI
                    p2 = v2 + (v3 - v2) / PHI
                    new_triangles.extend(((1, p2, p1, v2), (0, p1, v1, p2), (0, v3, v1, p2)))
                else:
                    # Subdivide blue (fat isosceles) (half dart) triangle
                    p3 = v3 + (v1 - v3) / PHI
                    new_triangles.extend(((1, v2, p3, v1), (0, p3, v3, v2)))
            else:
                if shape == 0:
                    # Divide thin rhombus
                    p1 = v1 + (v2 - v1) / PHI
                    new_triangles.extend(((1, p1, v3, v1), (0, v3, p1, v2)))
                else:
                    # Divide thicc rhombus
                    p2 = v2 + (v1 - v2) / PHI
                    p3 = v2 + (v3 - v2) / PHI
                    new_triangles.extend(((1, p3, v3, v1), (1, p2, p3, v2), (0, p3, p2, v1)))

        triangles = new_triangles

    return [
        Polygon.from_points(
            (
                Point(v2.real, v2.imag),
                Point(v1.real, v1.imag),
                Point(v3.real, v3.imag),
                # NOTE: in the original script the closing edge of the triangle (v3 -> v2)
                # is never drawn. here we save it so that the "missing" closing edge is last
                # we use it later when merging triangles into tiles
            )
        )
        for _, v1, v2, v3 in triangles
    ]


def penrose_P3_de_Brujin(n_horiz: int, n_vert: int, random_seed: int | None = None) -> list[Polygon]:
    """Generation with de Brujin method facilitated by pynrose library"""
    tiling = Tiling(rnd=random.Random(random_seed))
    safety_margin_factor = 1.3
    grid = Grid(Vector(0, 0), Vector(int(n_horiz * safety_margin_factor), int(n_vert * safety_margin_factor)))
    polys = [Polygon.from_rhombus(r) for r in tiling.rhombii(grid.cell(0, 0)) if r]
    return clipped_to_box(
        polys,
        box=Box.bounding_all(polys).resized(1 / safety_margin_factor),
    )
