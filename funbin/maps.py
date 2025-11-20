import json
import logging
from pathlib import Path
from typing import Any

import shapefile

from funbin.geometry import Point, Polygon, clipped_to_box

logger = logging.getLogger(__name__)


def parse_geo_geometry(geom: dict[str, Any]) -> list[Polygon]:
    try:
        match geom["type"].lower():
            case "polygon":
                return [Polygon.from_points([Point(x, y) for x, y in geom["coordinates"][0]])]
            case "multipolygon":
                return [
                    Polygon.from_points([Point(x, y) for x, y in poly_coords[0]]) for poly_coords in geom["coordinates"]
                ]
    except Exception:
        pass
    return []


def parse_geojson(data) -> list[Polygon]:
    res = []
    for feature in data["features"]:
        polys = parse_geo_geometry(feature["geometry"])
        if not polys:
            logger.warning(f"Ignoring unexpected geometry: {feature['geometry']}")
        res.extend(polys)
    return res


def read_geojson(geojson: Path | str) -> list[Polygon]:
    with open(geojson) as f:
        return parse_geojson(json.load(f))


def read_shapefile(path: Path | str) -> list[Polygon]:
    with shapefile.Reader(path) as shp:
        return parse_geojson(shp.__geo_interface__)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from funbin.geometry import Box, as_poly_collection

    logging.basicConfig(level=logging.INFO)

    fig, ax = plt.subplots()
    polys = read_shapefile("misc/ne_10m_admin_2_counties.zip")
    polys = clipped_to_box(polys, Box(Point(-130, 25), 80, 25))
    ax.add_collection(as_poly_collection(polys, randomize_color=True, edgecolors="none"))
    Box.bounding_all(polys).resized(1.1).fit_axes(ax)
    ax.set_aspect("equal")
    fig.savefig("map.png", dpi=500)
