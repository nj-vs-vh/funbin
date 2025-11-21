"""
A fast numpy-vectorized rewrite of the original JS code. See direct port in ported.py.
"""

import itertools
from dataclasses import dataclass
from typing import Literal

import numpy as np

from funbin.geometry import Point, Polygon, rotation_matrix


@dataclass
class MetaTile:
    verts: np.ndarray
    poly_start_indices: list[int]

    quad: np.ndarray

    # TODO: track border

    def translate_in_place(self, dp: np.ndarray) -> None:
        self.verts += dp
        self.quad += dp

    def rotate_and_match(self, rotmat: np.ndarray, qidx: int, q_dp: np.ndarray | None) -> "MetaTile":
        res = MetaTile(
            verts=np.matvec(rotmat, self.verts),
            quad=np.matvec(rotmat, self.quad),
            poly_start_indices=self.poly_start_indices,
        )
        if qidx >= 0:
            assert q_dp is not None
            res.translate_in_place(q_dp - res.quad[qidx, :])
        return res

    def as_polygons(self) -> list[Polygon]:
        res: list[Polygon] = []
        bounds = self.poly_start_indices + [self.verts.shape[0]]
        for start, end in itertools.pairwise(bounds):
            res.append(Polygon(verts=self.verts[start:end, :]))
        return res


def pack_points(points: list[Point]) -> np.ndarray:
    return np.array([(v.x, v.y) for v in points])


@dataclass
class State:
    H8: MetaTile
    H7: MetaTile


def base_tile_state(a: float, b: float) -> State:
    # Schematic description of the edges of a shape in the hat
    # continuum.  Each edge's length is one of 'a' or 'b', and the
    # direction d gives the orientation of d*30 degrees relative to
    # the positive X axis.
    edges = [
        ("a", 0),
        ("a", 2),
        ("b", 11),
        ("b", 1),
        ("a", 4),
        ("a", 2),
        ("b", 5),
        ("b", 3),
        ("a", 6),
        ("a", 8),
        ("a", 8),
        ("a", 10),
        ("b", 7),
    ]
    half_sqrt3 = 0.5 * np.sqrt(3)
    angle_dirs = [
        Point(1, 0),
        Point(half_sqrt3, 0.5),
        Point(0.5, half_sqrt3),
        Point(0, 1),
        Point(-0.5, half_sqrt3),
        Point(-half_sqrt3, 0.5),
        Point(-1, 0),
        Point(-half_sqrt3, -0.5),
        Point(-0.5, -half_sqrt3),
        Point(0, -1),
        Point(0.5, -half_sqrt3),
        Point(half_sqrt3, -0.5),
    ]

    prev = Point(0.0, 0.0)
    points = [prev]

    for kind, angle in edges:
        if kind == "a":
            prev += a * angle_dirs[angle]
        else:
            prev += b * angle_dirs[angle]
        points.append(prev)

    quad = pack_points([points[1], points[3], points[9], points[13]])

    flipped_points = [Point(p.x, p.y * -1) for p in reversed(points)]
    dp = points[0] - flipped_points[5]
    flipped_points = [fp + dp for fp in flipped_points]

    return State(
        H8=MetaTile(
            verts=pack_points(points),
            poly_start_indices=[0],
            quad=quad,
        ),
        H7=MetaTile(
            verts=np.vstack(
                (
                    pack_points(points),
                    pack_points(flipped_points),
                )
            ),
            poly_start_indices=[0, len(points)],
            quad=np.vstack((quad, quad)),
        ),
    )


def _merge_metatiles(mts: list[MetaTile], quad: np.ndarray | None = None) -> MetaTile:
    poly_start_indices_merged = []
    offset = 0
    for smt in mts:
        poly_start_indices_merged.extend([idx + offset for idx in smt.poly_start_indices])
        offset += smt.verts.shape[0]

    return MetaTile(
        verts=np.vstack([smt.verts for smt in mts]),
        poly_start_indices=poly_start_indices_merged,
        quad=(
            quad
            if quad is not None
            else np.vstack(
                (
                    mts[1].quad[3, :],
                    mts[2].quad[0, :],
                    mts[4].quad[3, :],
                    mts[6].quad[0, :],
                )
            )
        ),
    )


def extended_state(sys: State) -> State:
    sing = sys.H8
    comp = sys.H7

    sub_metatiles: list[MetaTile] = [sing]
    for rotmat, qidx, axis_quad_idx, flag in [
        (rotation_matrix(np.pi / 3), 2, 0, False),
        (rotation_matrix((2 * np.pi) / 3), 2, 0, False),
        (rotation_matrix(0), 1, 1, True),
        (rotation_matrix((-2 * np.pi) / 3), 2, 2, False),
        (rotation_matrix(-np.pi / 3), 2, 0, False),
        (rotation_matrix(0), 2, 0, False),
    ]:
        sub_metatiles.append(
            (comp if flag else sing).rotate_and_match(
                rotmat,
                qidx,
                sub_metatiles[-1].quad[axis_quad_idx, :],
            ),
        )

    smeta = _merge_metatiles(sub_metatiles)
    cmeta = _merge_metatiles(sub_metatiles[:-1], quad=smeta.quad)
    return State(H8=smeta, H7=cmeta)


AperiodicMonotileConstruction = Literal["H8", "H7"]
AperiodicMonotileKind = Literal["chevron", "hat", "tile(1,1)", "turtle", "comet"] | float


def aperiodic_monotile_raw(
    niter: int,
    construction: AperiodicMonotileConstruction = "H8",
    kind: AperiodicMonotileKind = "hat",
) -> list[Polygon]:
    match kind:
        case "chevron":
            a = 0.0
            b = 1 + np.sqrt(3)
        case "hat":
            a = 1.0
            b = np.sqrt(3)
        case "tile(1,1)":
            a = b = (1 + np.sqrt(3)) / 2
        case "turtle":
            b = 1.0
            a = np.sqrt(3)
        case "comet":
            a = 1 + np.sqrt(3)
            b = 0.0
        case shape_param:
            if not 0.0 <= shape_param <= 1.0:
                raise ValueError("Tile shape param must be between 0 and 1")
            alpha = 1 + np.sqrt(3)
            a = alpha * shape_param
            b = alpha * (1 - shape_param)
    s = base_tile_state(a, b)
    for _ in range(niter):
        s = extended_state(s)

    return (s.H8 if construction == "H8" else s.H7).as_polygons()
