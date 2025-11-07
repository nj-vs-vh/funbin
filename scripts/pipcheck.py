import matplotlib
import matplotlib.collections
import matplotlib.patches
import numpy as np
from matplotlib import pyplot as plt

from funbin.einstein import aperiodic_monotile
from funbin.models import Box, Point, fitted_to_box

points = np.random.random(size=(300, 2))
box = Box.bounding(points).resized(1.1)

polys = aperiodic_monotile(niter=2)
polys = fitted_to_box(polys, box)
cmap = matplotlib.colormaps["jet"]
n = len(polys)
colors = [cmap(np.random.random()) for i in range(len(polys))]

fig, ax = plt.subplots()
# ax.add_patch(matplotlib.patches.Polygon(poly.verts, facecolor="none", edgecolor="black"))
ax.add_collection(
    matplotlib.collections.PolyCollection([p.verts for p in polys], edgecolors=colors, facecolors=colors, alpha=0.3)
)


def point_poly_idx(p: Point) -> int | None:
    indices = np.nonzero(np.array([poly.includes(p) for poly in polys], dtype=int))[0]
    if indices.size == 0:
        return None
    elif indices.size == 1:
        return indices[0]
    else:
        raise ValueError(f"Ambiguous poly attribution: {indices}")


point_poly_indices = [point_poly_idx(Point(*p)) for p in points]
ax.scatter(
    points[:, 0],
    points[:, 1],
    c=np.array([np.array([0.0, 0.0, 0.0, 1.0]) if pid is None else colors[pid] for pid in point_poly_indices]),
    marker=".",
)
box.fit_axes(ax)
fig.savefig("pipcheck.png")
