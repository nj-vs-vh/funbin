import matplotlib
import matplotlib.patches
import numpy as np
from matplotlib import pyplot as plt

from funbin.models import Point, Polygon

points = np.random.random(size=(100, 2))
poly = Polygon(
    np.array(
        [
            [0.3, 0.3],
            [0.5, 0.8],
            [0.6, 0.1],
        ]
    )
)

fig, ax = plt.subplots()
ax.add_patch(matplotlib.patches.Polygon(poly.verts, facecolor="none", edgecolor="black"))
ax.scatter(points[:, 0], points[:, 1], color=["red" if poly.includes(Point(*p)) else "k" for p in points], marker=".")
fig.savefig("pipcheck.png")
