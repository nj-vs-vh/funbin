import cProfile
import logging
import math
import pstats

import numpy as np

from funbin.einstein.vectorized import aperiodic_monotile_raw
from funbin.geometry import rectanglize_tiling

np.random.seed(162)

bins = (30, 30)
target_bins = bins[0] * bins[1]
conserv_bins = 5 * target_bins
niter = int(math.ceil(math.log(conserv_bins / 0.33) / math.log(6)))
print(f"niter = {niter}")

state = aperiodic_monotile_raw(niter=niter, kind="hat")
meta_tile = state.H8
raw = meta_tile.as_polygons()
border_edges = meta_tile.border_edges()

pr = cProfile.Profile()
pr.enable()
logging.basicConfig(level=logging.DEBUG, format="%(relativeCreated)s %(levelname)s:%(module)s: %(message)s")
rectanglize_tiling(
    tiles=raw,
    target_bins=bins,
    border_edges_precomputed=border_edges,
    max_tries=30,
    rotate_index=True,
)
pr.disable()

sortby = pstats.SortKey.CUMULATIVE
with open("profile.log", "w") as f:
    ps = pstats.Stats(pr, stream=f).sort_stats(sortby)
    ps.print_stats()
