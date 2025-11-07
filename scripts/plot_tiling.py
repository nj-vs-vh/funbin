import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection

from funbin.einstein import aperiodic_monotile
from funbin.geometry import Box

# points = np.random.random(size=(30, 2)) * 2
# polys = voronoi(points)
# polys = penrose_P3(30, 30)
polys = aperiodic_monotile(niter=3)
pc = PolyCollection([p.verts for p in polys])
# pc.set_array(np.random.random(size=len(polys)))
pc.set_array(np.arange(len(polys)))
pc.set_cmap("viridis")

fig, ax = plt.subplots()
ax.add_collection(pc)
# ax.scatter(points[:, 0], points[:, 1], color="gray", edgecolor="none", marker=".", alpha=0.6)
Box.bounding_all(polys).resized(1.1).fit_axes(ax)
ax.set_aspect("equal")
fig.savefig("tiling.png")
