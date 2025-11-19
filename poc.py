import itertools
import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import Colormap, Normalize

from funbin.einstein import aperiodic_monotile
from funbin.geometry import Box, Point, Polygon, SpatialIndex, fitted_to_box, rectanglize_tiling
from funbin.penrose import penrose_tiling
from funbin.voronoi import voronoi


def funbin(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    tiling: list[Polygon],
    *,
    weights: np.ndarray | None = None,
    cmap: str | Colormap = "viridis",
    norm: str | Normalize = "linear",
    density: bool = True,
    spatial_indexing: bool = True,
    **poly_coll_kw,
) -> PolyCollection:
    assert x.ndim == 1
    assert y.ndim == 1
    samples = np.vstack((x, y)).T

    samples_bbox = Box.bounding(samples)
    tiling = fitted_to_box(tiling, samples_bbox)

    weight_per_tile = [0 for _ in tiling]

    if spatial_indexing:
        indexed_tiling = SpatialIndex.from_polygons(tiling, bins=len(tiling))
        for sample, weight in zip(samples, weights or itertools.repeat(1.0 / samples.shape[0])):
            tile_id = indexed_tiling.lookup_tile_id(Point(*sample))
            if tile_id is not None:
                weight_per_tile[tile_id] += weight
    else:
        for sample, weight in zip(samples, weights or itertools.repeat(1.0 / samples.shape[0])):
            p = Point(*sample)
            for tile_id, poly in enumerate(tiling):
                if poly.includes(p):
                    weight_per_tile[tile_id] += weight
                    break

    poly_coll_kw.setdefault("edgecolors", "face")
    pc = PolyCollection([p.verts for p in tiling], **poly_coll_kw)
    pc.set_array([tile_weight / (poly.area if density else 1.0) for tile_weight, poly in zip(weight_per_tile, tiling)])
    pc.set_cmap(cmap)
    pc.set_norm(norm)
    ax.add_collection(pc)
    samples_bbox.fit_axes(ax)
    return pc


if __name__ == "__main__":
    from typing import Sequence, cast

    fig, axes = plt.subplots(figsize=(15, 10), ncols=3, nrows=2)
    axes = cast(Sequence[Axes], axes.flatten())
    np.random.seed(1312)
    sample_size = 10000

    gauss_1 = np.random.normal(loc=0, scale=1.0, size=(2, sample_size))
    gauss_2 = np.random.normal(loc=np.expand_dims((2.0, 2.0), axis=1), scale=0.5, size=(2, sample_size))
    samples = np.where(np.random.random(sample_size) > 0.1, gauss_1, gauss_2)
    x, y = samples

    bins = 30
    cmap = "inferno"

    t1 = time.time()
    axes[0].hist2d(x, y, bins=bins, cmap=cmap)
    axes[0].set_title("Regular hist2d")
    t2 = time.time()
    print(f"Regular hist: {t2 - t1:.3f} sec")

    funbin(axes[1], x, y, tiling=penrose_tiling("P3", (bins, bins)), cmap=cmap)
    axes[1].set_title("Penrose P3 (rhombic) tiling")
    t3 = time.time()
    print(f"P3 hist: {t3 - t2:.3f} sec")

    funbin(axes[2], x, y, tiling=penrose_tiling("P2", (bins, bins)), cmap=cmap)
    axes[2].set_title("Penrose P2 (darts and kites) tiling")
    t4 = time.time()
    print(f"P2 hist: {t4 - t3:.3f} sec")

    voronoi_points = bins**2
    funbin(axes[3], x, y, tiling=voronoi(points=voronoi_points), cmap=cmap)
    axes[3].set_title(f"Voronoi diagram of {voronoi_points} random points")
    t5 = time.time()
    print(f"Voronoi: {t5 - t4:.3f} sec")

    raw = aperiodic_monotile(niter=5)
    tiling = rectanglize_tiling(raw, target_bins=(bins, bins), max_tries=30, debug=True)
    pc = funbin(axes[4], x, y, tiling=tiling, cmap=cmap)
    axes[4].set_title("Aperioric monotile")
    t6 = time.time()
    print(f"Aperiodic monotile: {t6 - t5:.3f} sec")

    for ax in axes:
        ax.set_aspect("equal")
        # ax.scatter(x, y, edgecolor="none", color="gray", alpha=0.3, marker=".")
    fig.tight_layout()
    fig.savefig("poc.png")
