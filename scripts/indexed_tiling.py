import argparse
import time

from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

from funbin.einstein import aperiodic_monotile
from funbin.geometry import Box, Point, SpatialIndex

it = SpatialIndex.from_polygons(
    polygons=aperiodic_monotile(niter=2),
    bins=(15, 15),
)


def plot_indexing(cell: tuple[int, int] | None = None, poly_id: int | None = None) -> bool:
    fig, ax = plt.subplots()

    if cell is not None:
        touching_tiles = it.items_in_bin[cell[0]][cell[1]]
        cells = [cell]
    elif poly_id is not None:
        cells = it.item_bins[poly_id]
        touching_tiles = [poly_id]
    else:
        raise RuntimeError("cell or poly_id must be specified!")

    if not touching_tiles:
        return False

    ax.add_collection(
        PolyCollection(
            [p.verts for p in it.items],
            facecolors=["blue" if i in touching_tiles else "none" for i in range(len(it.items))],
            edgecolors="darkgray",
        )
    )

    cell_w, cell_h = it.cell_size
    for i in range(1, it.bins[0]):
        ax.add_line(
            Line2D(
                xdata=[it.box.anchor.x + i * cell_w] * 2,
                ydata=[it.box.anchor.y, it.box.upper_right.y],
                color="gray",
                linewidth=0.3,
            )
        )
    for j in range(1, it.bins[1]):
        ax.add_line(
            Line2D(
                xdata=[it.box.anchor.x, it.box.upper_right.x],
                ydata=[it.box.anchor.y + j * cell_h] * 2,
                color="gray",
                linewidth=0.3,
            )
        )
    ax.add_patch(it.box.as_patch(facecolor="none", edgecolor="black"))
    for icell, jcell in cells:
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
    return True


parser = argparse.ArgumentParser()
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()
if args.animate:
    # for icell in range(it.index_bins[0]):
    #     for jcell in range(it.index_bins[0]):
    #         if plot_indexing(cell=(icell, jcell)):
    #             time.sleep(1.0)
    for poly_id in range(len(it.items)):
        plot_indexing(poly_id=poly_id)
        time.sleep(1.0)
else:
    # plot_indexing(cell=(6, 7))
    plot_indexing(poly_id=0)
