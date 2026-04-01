import logging
import time
from typing import Sequence, cast

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from funbin import funbin
from funbin.einstein import aperiodic_monotile
from funbin.geometry import Box, Point, clipped_to_box
from funbin.maps import read_shapefile
from funbin.penrose import penrose_tiling
from funbin.voronoi import voronoi

if __name__ == "__main__":
    np.random.seed(161)
    logging.basicConfig(level=logging.INFO)

    fig, axes = plt.subplots(figsize=(15, 15), ncols=3, nrows=3)
    axes = cast(Sequence[Axes], axes.flatten())

    sample_size = 50000

    gauss_1 = np.random.normal(loc=0, scale=1.0, size=(2, sample_size))
    gauss_2 = np.random.normal(loc=np.expand_dims((2.0, 2.0), axis=1), scale=0.5, size=(2, sample_size))
    samples = np.where(np.random.random(sample_size) > 0.1, gauss_1, gauss_2)
    x, y = samples

    bins = 40
    cmap = "inferno"

    start = time.time()
    axes[0].hist2d(x, y, bins=bins, cmap=cmap)
    axes[0].set_title("Regular hist2d")
    print(f"Regular hist: {time.time() - start:.3f} sec")

    start = time.time()
    funbin(axes[1], x, y, tiling=penrose_tiling("P3", (bins, bins)), cmap=cmap)
    axes[1].set_title("Penrose P3 (rhombic) tiling")
    print(f"P3 hist: {time.time() - start:.3f} sec")

    start = time.time()
    funbin(axes[2], x, y, tiling=penrose_tiling("P2", (bins, bins)), cmap=cmap)
    axes[2].set_title("Penrose P2 (darts and kites) tiling")
    print(f"P2 hist: {time.time() - start:.3f} sec")

    start = time.time()
    funbin(axes[3], x, y, tiling=penrose_tiling("P1", (bins, bins)), cmap=cmap)
    axes[3].set_title("Penrose P1 tiling")
    print(f"P1 hist: {time.time() - start:.3f} sec")

    start = time.time()
    voronoi_points = bins**2
    funbin(axes[4], x, y, tiling=voronoi(points=voronoi_points), cmap=cmap)
    axes[4].set_title(f"Voronoi diagram of {voronoi_points} random points")
    print(f"Voronoi: {time.time() - start:.3f} sec")

    start = time.time()
    tiling = aperiodic_monotile(bins=(bins, bins))
    pc = funbin(axes[5], x, y, tiling=tiling, cmap=cmap)
    axes[5].set_title("Aperioric monotile")
    print(f"Aperiodic monotile: {time.time() - start:.3} sec")

    start = time.time()
    tiling = read_shapefile("misc/ne_10m_admin_2_counties.zip")
    tiling = clipped_to_box(tiling, Box(Point(-130, 25), 80, 25))
    pc = funbin(axes[6], x, y, tiling=tiling, cmap=cmap)
    axes[6].set_title("Contiguous US counties")
    print(f"US counties: {time.time() - start:.3} sec")

    for ax in axes:
        ax.set_aspect("equal")
        # ax.scatter(x, y, edgecolor="none", color="gray", alpha=0.3, marker=".")
    fig.tight_layout()
    fig.savefig("poc.png")
