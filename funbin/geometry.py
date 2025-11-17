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

    def normalized(self) -> "Point":
        return self / self.abs


def is_ccw_order(A: Point, B: Point, C: Point) -> bool:
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)


LineSegment = tuple[Point, Point]


def do_intersect(AB: LineSegment, CD: LineSegment) -> bool:
    """https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm"""
    A, B = AB
    C, D = CD
    return is_ccw_order(A, C, D) != is_ccw_order(B, C, D) and is_ccw_order(A, B, C) != is_ccw_order(A, B, D)


def segment_intersection(AB: LineSegment, CD: LineSegment) -> Point | None:
    if not do_intersect(AB, CD):
        return None
    A, B = AB
    C, D = CD
    denom = (A.x - B.x) * (C.y - D.y) - (A.y - B.y) * (C.x - D.x)
    if abs(denom) < 1e-10:
        return None
    return Point(
        x=((A.x * B.y - B.x * A.y) * (C.x - D.x) - (C.x * D.y - D.x * C.y) * (A.x - B.x)) / denom,
        y=((A.x * B.y - B.x * A.y) * (C.y - D.y) - (C.x * D.y - D.x * C.y) * (A.y - B.y)) / denom,
    )


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
    def left(self) -> float:
        return self.anchor.x

    @functools.cached_property
    def right(self) -> float:
        return self.anchor.x + self.width

    @functools.cached_property
    def bottom(self) -> float:
        return self.anchor.y

    @functools.cached_property
    def top(self) -> float:
        return self.anchor.y + self.height

    @functools.cached_property
    def lower_left(self) -> Point:
        return self.anchor

    @functools.cached_property
    def lower_right(self) -> Point:
        return Point(self.right, self.bottom)

    @functools.cached_property
    def upper_left(self) -> Point:
        return Point(self.left, self.top)

    @functools.cached_property
    def upper_right(self) -> Point:
        return Point(self.right, self.top)

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

    def scaled(self, a: float) -> "Polygon":
        return Polygon(verts=a * self.verts)

    def translated(self, vec: Point) -> "Polygon":
        return Polygon(self.verts + np.array([[vec.x, vec.y]]))

    def rotated(self, rot: float | np.ndarray) -> "Polygon":
        if isinstance(rot, float):
            rm = rotation_matrix(rot)
        else:
            rm = rot
        return Polygon(verts=np.matvec(rm, self.verts))


def rotation_matrix(angle: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )


def fitted_to_box(tiling: list[Polygon], box: Box) -> list[Polygon]:
    orig = Box.bounding_all(tiling)
    return [p.moved(from_=orig, to=box) for p in tiling]


def rotated(tiling: list[Polygon], angle: float) -> list[Polygon]:
    rm = rotation_matrix(angle)
    return [p.rotated(rm) for p in tiling]


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
    def from_polygons(polygons: list[Polygon], bins: tuple[int, int] | int | None = None):
        return SpatialIndex._build(
            box=Box.bounding_all(polygons),
            items=polygons.copy(),  # type: ignore
            bins=bins or len(polygons),
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

    @functools.cached_property
    def border_edges(self) -> list[LineSegment]:
        eps = 1e-6
        res: list[LineSegment] = []
        for t in self.items:
            if not isinstance(t, Polygon):
                continue
            for start, end in t.edges:
                vec = end - start
                small_normal = eps * Point(vec.y, -vec.x).normalized()
                middle = start + vec / 2
                if self.is_inside_tiles(middle + small_normal) != self.is_inside_tiles(middle - small_normal):
                    res.append((start, end))
        return res

    def is_inscribed(self, poly: Polygon) -> bool:
        return all(self.is_inside_tiles(p) for p in poly.vertices) and all(
            not any(do_intersect(edge, border_edge) for border_edge in self.border_edges) for edge in poly.edges
        )


def rectanglize(tiles: list[Polygon], rotate: bool = True) -> list[Polygon]:
    if rotate:
        angle = np.random.random() * 2 * math.pi
        tiles = rotated(tiles, angle=angle)

    tiles_index = SpatialIndex.from_polygons(tiles, bins=len(tiles) * 5)

    bbox = Box.bounding_all(tiles)

    while True:
        center = Point(
            x=bbox.anchor.x + np.random.random() * bbox.width,
            y=bbox.anchor.y + np.random.random() * bbox.height,
        )
        if tiles_index.is_inside_tiles(center):
            break

    # first, we find a reasonably inscribed square
    side_init = bbox.width / 2
    side = side_init
    side_step = side_init
    unit_square = Polygon.from_points(
        [
            Point(-1.0, -1.0),
            Point(-1.0, 1.0),
            Point(1.0, 1.0),
            Point(1.0, -1.0),
        ]
    ).scaled(0.5)
    box_rect: Polygon | None = None
    for iter in itertools.count():
        box_rect = unit_square.scaled(side).translated(center)
        if tiles_index.is_inscribed(box_rect):
            if iter > 10:
                break
            else:
                side += side_step
        else:
            side -= side_step
        side_step /= 2

    assert box_rect is not None
    box = Box(anchor=box_rect.vertices[0], width=side, height=side)

    # then, we stretch the square in all 4 direction to maximum inscribed size
    top = Point(bbox.left, box.top), Point(bbox.right, box.top)
    bot = Point(bbox.left, box.bottom), Point(bbox.right, box.bottom)
    min_x_candidates = [
        p.x
        for p in itertools.chain.from_iterable(
            [segment_intersection(box_side, edge) for edge in tiles_index.border_edges] for box_side in (top, bot)
        )
        if p is not None and p.x <= box.lower_left.x
    ]
    if min_x_candidates:
        new_min_x = max(min_x_candidates)
        box = Box(
            anchor=Point(
                x=new_min_x,
                y=box.anchor.y,
            ),
            width=box.lower_right.x - new_min_x,
            height=box.height,
        )

    res = [poly.clipped(box) for poly in tiles]
    res = [p for p in res if p.area > 1e-8]

    # TODO: grow square into rectangle
    return res
