import numpy as np
import shapely

from funbin.models import Box, Polygon


def voronoi(points: int | np.ndarray) -> list[Polygon]:
    if isinstance(points, int):
        points = np.random.random(size=(points, 2))

    polycoll = shapely.voronoi_polygons(geometry=shapely.MultiPoint(points))
    res: list[Polygon] = []
    for p in polycoll.geoms:
        x, y = p.boundary.xy
        res.append(Polygon(verts=np.vstack((np.array(x)[:-1], np.array(y)[:-1])).T))

    bbox = Box.bounding(points).resized(1.1)
    return [p.clipped(to=bbox) for p in res]


if __name__ == "__main__":
    voronoi(10)
