import logging

import numpy as np
from matplotlib import pyplot as plt

from funbin import funbin
from funbin.penrose import penrose_tiling

if __name__ == "__main__":
    np.random.seed(161)
    logging.basicConfig(level=logging.INFO)

    fig, ax = plt.subplots(figsize=(8, 8))

    sample_size = 50000

    gauss_1 = np.random.normal(loc=0, scale=1.0, size=(2, sample_size))
    gauss_2 = np.random.normal(loc=np.expand_dims((2.0, 2.0), axis=1), scale=0.5, size=(2, sample_size))
    samples = np.where(np.random.random(sample_size) > 0.1, gauss_1, gauss_2)
    x, y = samples

    bins = 40
    cmap = "inferno"

    funbin(ax, x, y, tiling=penrose_tiling("P3", (bins, bins)), cmap=cmap, edge_shading="inverted")

    ax.set_aspect("equal")
    fig.savefig("edge_shading_inv.png")
