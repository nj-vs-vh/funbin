import functools
import itertools
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from pynrose import Rhombus, RhombusVertex


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __add__(self, other: "Point"):
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __sub__(self, other: "Point"):
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        return NotImplemented

    def __mul__(self, scalar: int | float):
        if isinstance(scalar, (int, float)):
            return Point(self.x * scalar, self.y * scalar)
        return NotImplemented

    def __rmul__(self, scalar: int | float):
        return self.__mul__(scalar)

    def __truediv__(self, scalar: int | float):
        if isinstance(scalar, (int, float)):
            return Point(self.x / scalar, self.y / scalar)
        return NotImplemented

    def __repr__(self):
        return f"({self.x:.2f}, {self.y:.2f})"

    def dot(self, other: "Point") -> float:
        if isinstance(other, Point):
            return self.x * other.x + self.y * other.y
        return NotImplemented

    @functools.cached_property
    def sqabs(self) -> float:
        return self.dot(self)

    @functools.cached_property
    def abs(self) -> float:
        return math.sqrt(self.sqabs)


def is_ccw_order(A: Point, B: Point, C: Point) -> bool:
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)


LineSegment = tuple[Point, Point]


def do_intersect(AB: LineSegment, CD: LineSegment) -> bool:
    """https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm"""
    A, B = AB
    C, D = CD
    return is_ccw_order(A, C, D) != is_ccw_order(B, C, D) and is_ccw_order(A, B, C) != is_ccw_order(A, B, D)


@dataclass(frozen=True)
class Box:
    anchor: Point  # left bottom
    width: float
    height: float

    def __repr__(self) -> str:
        return f"Box({self.anchor}, w={self.width:.2f}, h={self.height:.2f})"

    @staticmethod
    def bounding(points: np.ndarray | Iterable[Point]) -> "Box":
        points_arr = points if isinstance(points, np.ndarray) else np.array([(p.x, p.y) for p in points])

        xmin = np.min(points_arr[:, 0])
        xmax = np.max(points_arr[:, 0])
        ymin = np.min(points_arr[:, 1])
        ymax = np.max(points_arr[:, 1])
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
    def vertices(self) -> list[Point]:
        return [
            Point(self.anchor.x, self.anchor.y),
            Point(self.anchor.x, self.anchor.y + self.height),
            Point(self.anchor.x + self.width, self.anchor.y + self.height),
            Point(self.anchor.x + self.width, self.anchor.y),
        ]

    @functools.cached_property
    def as_polygon(self) -> "Polygon":
        return Polygon(np.array([(v.x, v.y) for v in self.vertices]))

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


@dataclass(frozen=True)
class Polygon:
    verts: np.ndarray  # (nvert, 2)

    @functools.cached_property
    def vertices(self) -> list[Point]:
        return [Point(*coords) for coords in self.verts]

    @functools.cached_property
    def edge_endpoints(self) -> list[Point]:
        return self.vertices + [self.vertices[0]]

    @property
    def edges(self) -> Iterable[LineSegment]:
        return itertools.pairwise(self.edge_endpoints)

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
        intersections = sum(int(do_intersect(ray, edge)) for edge in self.edges)
        res = intersections % 2 == 1
        return res

    @functools.cached_property
    def area(self) -> float:
        return 0.5 * abs(sum((p1.x - p2.x) * (p1.y + p2.y) for (p1, p2) in self.edges))

    @staticmethod
    def from_rhombus(r: Rhombus) -> "Polygon":
        vertices: list[RhombusVertex] = r.vertices()
        return Polygon(np.array([(v.coordinate.x, v.coordinate.y) for v in vertices]))

    @staticmethod
    def from_points(points: list[Point]) -> "Polygon":
        return Polygon(np.array([(v.x, v.y) for v in points]))

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
class SpatialIndex:
    items: list[Polygon | LineSegment]

    box: Box
    bins: tuple[int, int]
    items_in_bin: list[list[list[int]]]
    item_bins: list[list[tuple[int, int]]]

    def append(self, item: Polygon | LineSegment) -> None:
        self.items.append(item)
        self.item_bins.append([])
        id = len(self.items) - 1
        touched_i: list[int] = []
        touched_j: list[int] = []
        item_bbox = Box.bounding_poly(item) if isinstance(item, Polygon) else Box.bounding(item)
        for vert in item_bbox.vertices:
            touched_i.append(math.floor((vert.x - self.box.anchor.x) / self.cell_w))
            touched_j.append(math.floor((vert.y - self.box.anchor.y) / self.cell_h))
        for i in range(min(touched_i), max(touched_i) + 1):
            for j in range(min(touched_j), max(touched_j) + 1):
                self.items_in_bin[i][j].append(id)
                self.item_bins[id].append((i, j))

    # def pop(self, idx: int) -> Polygon | LineSegment:
    #     item = self.items.pop(idx)
    #     item_bins = self.item_bins.pop(idx)
    #     for i, j in item_bins:
    #         self.items_in_bin[i][j].remove(idx)
    #     for row in self.items_in_bin:
    #         for j in range(len(row)):
    #             row[j] = [id - (0 if id <= idx else 1) for id in row[j]]
    #     return item

    @staticmethod
    def _build(box: Box, items: list[Polygon | LineSegment], bins: tuple[int, int] | int):
        if isinstance(bins, tuple):
            x_bins, y_bins = bins
        else:
            x_bins = y_bins = int(round(math.sqrt(bins)))
        res = SpatialIndex(
            items=[],
            box=box.resized(1.01),  # safety margin for outermost points
            bins=(x_bins, y_bins),
            items_in_bin=[[[] for _ in range(y_bins)] for _ in range(x_bins)],
            item_bins=[],
        )
        for item in items:
            res.append(item)
        return res

    @staticmethod
    def from_polygons(polygons: list[Polygon], bins: tuple[int, int] | int):
        return SpatialIndex._build(
            box=Box.bounding_all(polygons),
            items=polygons.copy(),  # type: ignore
            bins=bins,
        )

    @staticmethod
    def from_line_segments(line_segments: list[LineSegment], bins: tuple[int, int] | int):
        return SpatialIndex._build(
            box=Box.bounding(itertools.chain.from_iterable(line_segments)),
            items=line_segments.copy(),  # type: ignore
            bins=bins,
        )

    def lookup_tile_id(self, p: Point) -> int | None:
        if not self.box.includes(p):
            return None
        cell_w, cell_h = self.cell_size
        icell = math.floor((p.x - self.box.anchor.x) / cell_w)
        jcell = math.floor((p.y - self.box.anchor.y) / cell_h)
        for candidate_id in self.items_in_bin[icell][jcell]:
            candidate = self.items[candidate_id]
            if isinstance(candidate, Polygon) and candidate.includes(p):
                return candidate_id
        else:
            return None

    def is_inside_tiles(self, p: Point) -> bool:
        return self.lookup_tile_id(p) is not None

    @functools.cached_property
    def cell_w(self) -> float:
        return self.box.width / self.bins[0]

    @functools.cached_property
    def cell_h(self) -> float:
        return self.box.height / self.bins[1]

    @functools.cached_property
    def cell_size(self) -> tuple[float, float]:
        return self.cell_w, self.cell_h
