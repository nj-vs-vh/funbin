from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

from funbin.models import Box, IndexedTiling, Point
from funbin.penrose import penrose_P3

it = IndexedTiling.from_polygons(
    polygons=penrose_P3(25, 25, random_seed=161),
    bins=(15, 15),
)

fig, ax = plt.subplots()

icell, jcell = 5, 7

touching_tiles = it.indexed_tiles[icell][jcell]
ax.add_collection(
    PolyCollection(
        [p.verts for p in it.tiles],
        facecolors=["blue" if i in touching_tiles else "none" for i in range(len(it.tiles))],
        edgecolors="darkgray",
    )
)

cell_w, cell_h = it.index_cell_size
for i in range(1, it.index_bins[0]):
    ax.add_line(
        Line2D(
            xdata=[it.box.anchor.x + i * cell_w] * 2,
            ydata=[it.box.anchor.y, it.box.upper_right.y],
            color="gray",
            linewidth=0.3,
        )
    )
for j in range(1, it.index_bins[1]):
    ax.add_line(
        Line2D(
            xdata=[it.box.anchor.x, it.box.upper_right.x],
            ydata=[it.box.anchor.y + j * cell_h] * 2,
            color="gray",
            linewidth=0.3,
        )
    )
ax.add_patch(it.box.as_patch(facecolor="none", edgecolor="black"))
ax.add_patch(
    Box(
        anchor=Point(
            x=it.box.anchor.x + cell_w * icell,
            y=it.box.anchor.y + cell_h * jcell,
        ),
        width=cell_w,
        height=cell_h,
    ).as_patch(facecolor="red", edgecolor="none", alpha=0.3)
)
it.box.resized(1.05).fit_axes(ax)
fig.savefig("indexed_tiling.png")
