import numpy as np
import shapely

from funbin.geometry import Box, Polygon


def voronoi(points: int | np.ndarray) -> list[Polygon]:
    if isinstance(points, int):
        points = np.random.random(size=(points, 2))

    polycoll = shapely.voronoi_polygons(geometry=shapely.MultiPoint(points))
    res = [Polygon.from_shapely(p) for p in polycoll.geoms]
    bbox = Box.bounding(points).resized(1.1)
    return [p.clipped(to=bbox) for p in res]


if __name__ == "__main__":
    voronoi(10)
