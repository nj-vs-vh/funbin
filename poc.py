import itertools
import math
import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import Colormap, Normalize

from funbin.einstein import aperiodic_monotile
from funbin.models import Box, IndexedTiling, Point, Polygon, fitted_to_box
from funbin.penrose import penrose_P3
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
        index_bins = int(round(math.sqrt(len(tiling))))
        indexed_tiling = IndexedTiling.from_polygons(tiling, bins=(index_bins, index_bins))
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

    fig, axes = plt.subplots(figsize=(10, 10), ncols=2, nrows=2)
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
    funbin(axes[1], x, y, tiling=penrose_P3(bins, bins, random_seed=1312), cmap=cmap)
    axes[1].set_title("Penrose P3 tiling")
    t3 = time.time()
    voronoi_points = bins**2
    funbin(axes[2], x, y, tiling=voronoi(points=voronoi_points), cmap=cmap)
    axes[2].set_title(f"Voronoi diagram of {voronoi_points} random points")
    t4 = time.time()
    pc = funbin(axes[3], x, y, tiling=aperiodic_monotile(niter=3), cmap=cmap, spatial_indexing=True)
    axes[3].set_title("Aperioric monotile (WIP)")
    t5 = time.time()

    print(f"Regular hist: {t2 - t1:.3f} sec")
    print(f"P3 hist: {t3 - t2:.3f} sec")
    print(f"Voronoi: {t4 - t3:.3f} sec")
    print(f"Aperiodic monotile: {t5 - t4:.3f} sec")
    # for ax in axes:
    #     ax.scatter(x, y, edgecolor="none", color="gray", alpha=0.3, marker=".")
    fig.tight_layout()
    fig.savefig("poc.png")
