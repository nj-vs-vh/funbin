import functools
import itertools
import math
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from pynrose import Rhombus, RhombusVertex


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        return NotImplemented

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Point(self.x * scalar, self.y * scalar)
        return NotImplemented

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Point(self.x / scalar, self.y / scalar)
        return NotImplemented

    def __repr__(self):
        return f"Point({self.x}, {self.y})"


def is_ccw_order(A: Point, B: Point, C: Point) -> bool:
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)


def do_intersect(AB: tuple[Point, Point], CD: tuple[Point, Point]) -> bool:
    """https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm"""
    A, B = AB
    C, D = CD
    return is_ccw_order(A, C, D) != is_ccw_order(B, C, D) and is_ccw_order(A, B, C) != is_ccw_order(A, B, D)


@dataclass(frozen=True)
class Box:
    anchor: Point  # left bottom
    width: float
    height: float

    @staticmethod
    def bounding(points: np.ndarray) -> "Box":
        xmin = np.min(points[:, 0])
        xmax = np.max(points[:, 0])
        ymin = np.min(points[:, 1])
        ymax = np.max(points[:, 1])
        return Box(
            anchor=Point(xmin, ymin),
            width=xmax - xmin,
            height=ymax - ymin,
        )

    @staticmethod
    def bounding_poly(p: "Polygon") -> "Box":
        return Box.bounding(p.verts)

    @staticmethod
    def bounding_all(polys: "list[Polygon]") -> "Box":
        tiling_verts = np.concatenate([p.verts for p in polys], axis=0)
        return Box.bounding(tiling_verts)

    @functools.cached_property
    def center(self) -> Point:
        return Point(self.anchor.x + self.width / 2, self.anchor.y + self.height / 2)

    @functools.cached_property
    def upper_right(self) -> Point:
        return Point(self.anchor.x + self.width, self.anchor.y + self.height)

    def resized(self, factor: float) -> "Box":
        return Box(
            anchor=Point(
                self.center.x - factor * self.width / 2,
                self.center.y - factor * self.height / 2,
            ),
            width=factor * self.width,
            height=factor * self.height,
        )

    @functools.cached_property
    def as_polygon(self) -> "Polygon":
        return Polygon(
            np.array(
                [
                    (self.anchor.x, self.anchor.y),
                    (self.anchor.x, self.anchor.y + self.height),
                    (self.anchor.x + self.width, self.anchor.y + self.height),
                    (self.anchor.x + self.width, self.anchor.y),
                ]
            )
        )

    def includes(self, p: Point) -> bool:
        return self.anchor.x <= p.x <= self.upper_right.x and self.anchor.y <= p.y <= self.upper_right.y

    def overlaps(self, other: "Box") -> bool:
        return (
            self.upper_right.x > other.anchor.x
            and self.anchor.x < other.upper_right.x
            and self.upper_right.y > other.anchor.y
            and self.anchor.y < other.upper_right.y
        )

    def fit_axes(self, ax: Axes) -> None:
        ax.set_xlim(self.anchor.x, self.anchor.x + self.width)
        ax.set_ylim(self.anchor.y, self.anchor.y + self.height)

    def as_patch(self, **patch_kw) -> Rectangle:
        return Rectangle(
            xy=(self.anchor.x, self.anchor.y),
            width=self.width,
            height=self.height,
            **patch_kw,
        )

    # def almost_equal(self) ->


@dataclass(frozen=True)
class Polygon:
    verts: np.ndarray  # (nvert, 2)

    @functools.cached_property
    def edge_endpoints(self) -> list[Point]:
        vert_points = [Point(*coords) for coords in self.verts]
        return vert_points + [vert_points[0]]

    @functools.cached_property
    def centroid(self) -> Point:
        return Point(*np.mean(self.verts, axis=0))

    @functools.cached_property
    def point_outside(self) -> Point:
        return self.bbox.resized(1.3).anchor

    @functools.cached_property
    def bbox(self) -> Box:
        return Box.bounding(self.verts)

    def includes(self, p: Point) -> bool:
        bbox = self.bbox
        if not (0 <= p.x - bbox.anchor.x <= bbox.width and 0 <= p.y - bbox.anchor.y <= bbox.height):
            return False

        ray = (self.point_outside, p)
        # print(p, [int(do_intersect(ray, edge)) for edge in itertools.pairwise(self.edge_endpoints)])
        intersections = sum(int(do_intersect(ray, edge)) for edge in itertools.pairwise(self.edge_endpoints))
        return intersections % 2 == 1

    @functools.cached_property
    def area(self) -> float:
        return 0.5 * abs(sum((p1.x - p2.x) * (p1.y + p2.y) for (p1, p2) in itertools.pairwise(self.edge_endpoints)))

    @staticmethod
    def from_rhombus(r: Rhombus) -> "Polygon":
        vertices: list[RhombusVertex] = r.vertices()
        return Polygon(np.array([(v.coordinate.x, v.coordinate.y) for v in vertices]))

    def moved(self, from_: Box, to: Box) -> "Polygon":
        x = to.anchor.x + (self.verts[:, 0] - from_.anchor.x) * to.width / from_.width
        y = to.anchor.y + (self.verts[:, 1] - from_.anchor.y) * to.height / from_.height
        return Polygon(verts=np.vstack((x, y)).T)

    def clipped(self, to: Box) -> "Polygon":
        anchor = np.array([to.anchor.x, to.anchor.y])
        upper_right = np.array([to.upper_right.x, to.upper_right.y])
        return Polygon(verts=np.minimum(np.maximum(self.verts, anchor), upper_right))


def fitted_to_box(tiling: list[Polygon], box: Box) -> list[Polygon]:
    orig = Box.bounding_all(tiling)
    return [p.moved(from_=orig, to=box) for p in tiling]


@dataclass(frozen=True)
class IndexedTiling:
    tiles: list[Polygon]

    box: Box
    index_bins: tuple[int, int]
    indexed_tiles: list[list[list[int]]]

    @staticmethod
    def from_polygons(polygons: list[Polygon], bins: tuple[int, int]):
        box = Box.bounding_all(polygons).resized(1.01)
        x_bins, y_bins = bins
        cell_w = box.width / x_bins
        cell_h = box.height / y_bins
        indexed_tiles: list[list[list[int]]] = [[[] for _ in range(y_bins)] for _ in range(x_bins)]
        for id, poly in enumerate(polygons):
            touched_is = []
            touched_js = []
            for vert in poly.bbox.as_polygon.verts:
                touched_is.append(math.floor((vert[0] - box.anchor.x) / cell_w))
                touched_js.append(math.floor((vert[1] - box.anchor.y) / cell_h))
            for i in range(min(touched_is), max(touched_is) + 1):
                for j in range(min(touched_js), max(touched_js) + 1):
                    indexed_tiles[i][j].append(id)
        return IndexedTiling(
            tiles=polygons,
            box=box,
            index_bins=bins,
            indexed_tiles=indexed_tiles,
        )

    def lookup_tile_id(self, p: Point) -> int | None:
        if not self.box.includes(p):
            return None
        cell_w, cell_h = self.index_cell_size
        icell = math.floor((p.x - self.box.anchor.x) / cell_w)
        jcell = math.floor((p.y - self.box.anchor.y) / cell_h)
        for candidate_id in self.indexed_tiles[icell][jcell]:
            if self.tiles[candidate_id].includes(p):
                return candidate_id
        else:
            return None

    @functools.cached_property
    def index_cell_size(self) -> tuple[float, float]:
        return (self.box.width / self.index_bins[0], self.box.height / self.index_bins[1])
