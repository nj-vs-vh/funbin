import cProfile
import math
import pstats
from pstats import SortKey

import numpy as np

from funbin.einstein.vectorized import aperiodic_monotile_raw
from funbin.geometry import rectanglize_tiling

pr = cProfile.Profile()

np.random.seed(161)

bins = (20, 20)
target_bins = bins[0] * bins[1]
conserv_bins = 5 * target_bins
niter = int(math.ceil(math.log(conserv_bins / 0.33) / math.log(6)))

state = aperiodic_monotile_raw(niter=niter, kind="hat")
meta_tile = state.H8
raw = meta_tile.as_polygons()
border_edges = meta_tile.border_edges()

pr.enable()
rectanglize_tiling(
    tiles=raw,
    target_bins=bins,
    border_edges_precomputed=border_edges,
)
pr.disable()

sortby = SortKey.CUMULATIVE
with open("profile.log", "w") as f:
    ps = pstats.Stats(pr, stream=f).sort_stats(sortby)
    ps.print_stats()
