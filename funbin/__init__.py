import itertools
from typing import Callable, Literal

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import Colormap, Normalize

from funbin.geometry import Box, Point, Polygon, SpatialIndex, fitted_to_box


def funbin(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    tiling: list[Polygon],
    *,
    weights: np.ndarray | None = None,
    cmap: str | Colormap | None = None,
    norm: str | Normalize | Callable[[list[float]], Normalize] = "linear",
    density: bool = True,
    spatial_indexing: bool = True,
    edge_shading: Literal["none", "inverted"] = "none",
    is_spherical: bool = False,
    **poly_coll_kw,
) -> PolyCollection:
    assert x.ndim == 1
    assert y.ndim == 1
    samples = np.vstack((x, y)).T

    samples_bbox = Box.bounding(samples)
    tiling = fitted_to_box(tiling, samples_bbox)

    weight_per_tile = [0.0 for _ in tiling]
    sample_weights = weights if weights is not None else itertools.repeat(1.0)

    if spatial_indexing:
        indexed_tiling = SpatialIndex.from_polygons(tiling, bins=len(tiling))
        for sample, weight in zip(samples, sample_weights):
            # for tile_id in indexed_tiling.lookup_all_tile_ids(Point(*sample)):
            #     weight_per_tile[tile_id] += weight

            tile_id = indexed_tiling.lookup_tile_id(Point(*sample))
            if tile_id is not None:
                weight_per_tile[tile_id] += weight
    else:
        for sample, weight in zip(samples, sample_weights):
            p = Point(*sample)
            for tile_id, poly in enumerate(tiling):
                if poly.includes(p):
                    weight_per_tile[tile_id] += weight
                    break

    tile_values = [
        tile_weight / (1.0 if not density else (poly.area if not is_spherical else poly.solid_angle))
        for tile_weight, poly in zip(weight_per_tile, tiling)
    ]
    match edge_shading:
        case "none":
            poly_coll_kw.setdefault("edgecolors", "face")
        case "inverted":
            empty_tile_border = "gray"
            max_tile_value = max(tile_values)
            tile_values_scaled = [float(v / max_tile_value) for v in tile_values]
            edgecolors = [(empty_tile_border, float(1 - norm_tile_value)) for norm_tile_value in tile_values_scaled]
            poly_coll_kw.setdefault("edgecolors", edgecolors)
    poly_coll_kw.setdefault("linewidth", 0.05)
    pc = PolyCollection([p.verts for p in tiling], **poly_coll_kw)
    pc.set_array(tile_values)
    pc.set_cmap(cmap or matplotlib.rcParams.get("image.cmap", "viridis"))
    if not isinstance(norm, (Normalize, str)):
        pc.set_norm(norm(tile_values))
    else:
        pc.set_norm(norm)
    ax.add_collection(pc)
    if not is_spherical:
        samples_bbox.fit_axes(ax)
    return pc
