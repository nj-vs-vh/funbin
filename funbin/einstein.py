"""
Aperiodic monotile generation.

A loose Python port of a demo webpage (https://cs.uwaterloo.ca/~csk/hat/h7h8.html)
accompanying the original paper "An aperiodic monotile" by David Smith, Joseph Samuel Myers,
Craig S. Kaplan, Chaim Goodman-Strauss (arXiv:2303.10798v3, 10.5070/C64163843)
"""

import itertools
from dataclasses import dataclass
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection

from funbin.geometry import Box, Point, Polygon


@dataclass(frozen=True)
class ABLinearComb:
    """
    A representation for numbers of the form xa+yb, where x and y
    are numeric coefficients and a and b are symbolic constants.
    Particular values for a and b define specific tiles.
    """

    acoeff: float
    bcoeff: float

    def eval(self, a: float, b: float) -> float:
        return a * self.acoeff + b * self.bcoeff

    def __add__(self, other):
        if isinstance(other, ABLinearComb):
            return ABLinearComb(self.acoeff + other.acoeff, self.bcoeff + other.bcoeff)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, ABLinearComb):
            return ABLinearComb(self.acoeff - other.acoeff, self.bcoeff - other.bcoeff)
        return NotImplemented

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return ABLinearComb(self.acoeff * scalar, self.bcoeff * scalar)
        return NotImplemented

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            return ABLinearComb(self.acoeff / scalar, self.bcoeff / scalar)
        return NotImplemented

    def __repr__(self):
        return f"ABPair({self.acoeff}, {self.bcoeff})"


@dataclass(frozen=True)
class PointInTile:
    x: ABLinearComb
    y: ABLinearComb

    def __add__(self, other):
        if isinstance(other, PointInTile):
            return PointInTile(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, PointInTile):
            return PointInTile(self.x - other.x, self.y - other.y)
        return NotImplemented


LinearMat = np.ndarray  # size: (2, 2)
AffineMat = np.ndarray  # size: (3, 3)


def trot(angle: float) -> LinearMat:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s], [s, c]])


# matrix multiplication on Points in tile
def transAB(M: LinearMat, p: PointInTile) -> PointInTile:
    return PointInTile(
        x=M[0, 0] * p.x + M[0, 1] * p.y,
        y=M[1, 0] * p.x + M[1, 1] * p.y,
    )


def invAffine(am: AffineMat) -> AffineMat:
    A = am[:, :2]
    b = am[:, 2]
    Ainv = np.linalg.inv(A)
    return np.hstack((Ainv, -Ainv @ b))


def mulAffine(A: AffineMat, B: AffineMat) -> AffineMat:
    A = A.flatten()
    B = B.flatten()
    return np.array(
        [
            [
                A[0] * B[0] + A[1] * B[3],
                A[0] * B[1] + A[1] * B[4],
                A[0] * B[2] + A[1] * B[5] + A[2],
            ],
            [
                A[3] * B[0] + A[4] * B[3],
                A[3] * B[1] + A[4] * B[4],
                A[3] * B[2] + A[4] * B[5] + A[5],
            ],
        ]
    )


def ttransAffine(tx: float, ty: float) -> AffineMat:
    return np.array([[1, 0, tx], [0, 1, ty]])


def transAffine(M: AffineMat, p: PointInTile) -> PointInTile:
    return PointInTile(
        x=M[0, 0] * p.x + M[0, 1] * p.y + M[0, 2],
        y=M[1, 0] * p.x + M[1, 1] * p.y + M[1, 2],
    )


@dataclass
class Shape:
    pts: list[PointInTile]
    quad: list[PointInTile]
    label: str

    def translateInPlace(self, dp: PointInTile) -> None:
        self.pts = [p + dp for p in self.pts]
        self.quad = [p + dp for p in self.quad]

    def rotateAndMatch(self, T: LinearMat, qidx: int, P: PointInTile | None) -> "Shape":
        new = Shape(
            pts=[transAB(T, p) for p in self.pts],
            quad=[transAB(T, p) for p in self.quad],
            label=self.label,
        )
        if qidx >= 0:
            if P is None:
                raise ValueError("P is reqiured if qidx is specified")
            new.translateInPlace(P - new.quad[qidx])

        return new

    def as_polygons(self, a: float, b: float) -> list[tuple[Polygon, str]]:
        return [
            (
                Polygon(verts=np.array([(p.x.eval(a, b), p.y.eval(a, b)) for p in self.pts])),
                self.label,
            )
        ]


@dataclass
class Meta:
    geoms: "list[Shape | Meta]"
    quad: list[PointInTile]

    def translateInPlace(self, dp: PointInTile) -> None:
        for g in self.geoms:
            g.translateInPlace(dp)
        self.quad = [q + dp for q in self.quad]

    def rotateAndMatch(self, T: LinearMat, qidx: int, P: PointInTile | None) -> "Meta":
        new = Meta(
            geoms=[g.rotateAndMatch(T, -1, None) for g in self.geoms],
            quad=[transAB(T, q) for q in self.quad],
        )
        if qidx >= 0:
            if P is None:
                raise ValueError("P is reqiured if qidx is specified")
            new.translateInPlace(P - new.quad[qidx])
        return new

    def as_polygons(self, a: float, b: float) -> list[tuple[Polygon, str]]:
        return list(itertools.chain.from_iterable(g.as_polygons(a, b) for g in self.geoms))


@dataclass
class State:
    H8: Shape | Meta
    H7: Meta


def base_tile_state() -> State:
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

    prev = PointInTile(ABLinearComb(0, 0), ABLinearComb(0, 0))
    points = [prev]

    for kind, angle in edges:
        if kind == "a":
            prev = PointInTile(
                x=prev.x + ABLinearComb(angle_dirs[angle].x, 0),
                y=prev.y + ABLinearComb(angle_dirs[angle].y, 0),
            )
        else:
            prev = PointInTile(
                x=prev.x + ABLinearComb(0, angle_dirs[angle].x),
                y=prev.y + ABLinearComb(0, angle_dirs[angle].y),
            )
        points.append(prev)

    quad = [points[1], points[3], points[9], points[13]]

    flipped_points = [PointInTile(p.x, p.y * -1) for p in reversed(points)]
    dp = points[0] - flipped_points[5]
    flipped_points = [fp + dp for fp in flipped_points]

    return State(
        H8=Shape(points, quad, "single"),
        H7=Meta(
            geoms=[
                Shape(points, quad, "unflipped"),
                Shape(flipped_points, quad, "flipped"),
            ],
            quad=quad,
        ),
    )


def extended_state(sys: State) -> State:
    sing = sys.H8
    comp = sys.H7

    smeta = Meta(
        geoms=[],
        quad=[],
    )
    rules = [
        (np.pi / 3, 2, 0, False),
        ((2 * np.pi) / 3, 2, 0, False),
        (0, 1, 1, True),
        ((-2 * np.pi) / 3, 2, 2, False),
        (-np.pi / 3, 2, 0, False),
        (0, 2, 0, False),
    ]

    smeta.geoms.append(sing)  # type: ignore
    for angle, qidx, axis_quad_idx, flag in rules:
        smeta.geoms.append(
            (comp if flag else sing).rotateAndMatch(
                trot(angle),
                qidx,
                smeta.geoms[-1].quad[axis_quad_idx],
            ),
        )

    smeta.quad = [
        smeta.geoms[1].quad[3],
        smeta.geoms[2].quad[0],
        smeta.geoms[4].quad[3],
        smeta.geoms[6].quad[0],
    ]

    cmeta = Meta(geoms=smeta.geoms[:-1].copy(), quad=smeta.quad)
    return State(
        H8=smeta,
        H7=cmeta,
    )


def aperiodic_monotile(
    niter: int,
    construction: Literal["H8", "H7"] = "H8",
    kind: Literal["chevron", "hat", "tile(1,1)", "turtle", "comet"] | float = "hat",
) -> list[Polygon]:
    s = base_tile_state()
    for _ in range(niter):
        s = extended_state(s)

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

    # TODO:
    # - find square box fully covered by the tiling and return it
    # - check max size of box vs. niter and select niter internally

    return [p for p, _ in (s.H8 if construction == "H8" else s.H7).as_polygons(a, b)]


if __name__ == "__main__":
    s = base_tile_state()
    s = extended_state(s)
    s = extended_state(s)
    s = extended_state(s)
    s = extended_state(s)

    labeled_polygons = s.H8.as_polygons(0, 1)
    polygons = [p for p, _ in labeled_polygons]
    fig, ax = plt.subplots()

    colormap = {
        "single": [0.0, 1.0, 0.0],
        "unflipped": [1.0, 0.0, 0.0],
        "flipped": [0.0, 0.0, 1.0],
    }
    pc = PolyCollection(
        [p.verts for p, _ in labeled_polygons],
        facecolors=[colormap[label] for _, label in labeled_polygons],
        edgecolors="black",
        linewidth=0.3,
        # facecolors="none",
    )
    ax.add_collection(pc)
    ax.set_aspect("equal")
    Box.bounding_all(polygons).resized(1.1).fit_axes(ax)

    fig.savefig("aperiodic.png", dpi=300)
