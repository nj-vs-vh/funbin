import logging
import math

from funbin.einstein.ported import aperiodic_monotile_raw as aperiodic_monotile_raw_ported
from funbin.einstein.vectorized import AperiodicMonotileConstruction, AperiodicMonotileKind, aperiodic_monotile_raw
from funbin.geometry import LineSegment, Polygon, rectanglize_tiling

logger = logging.getLogger(__name__)


def aperiodic_monotile(
    bins: tuple[int, int],
    kind: AperiodicMonotileKind = "hat",
    construction: AperiodicMonotileConstruction = "H8",
    direct_port_impl: bool = False,
) -> list[Polygon]:
    target_bins = bins[0] * bins[1]
    conserv_bins = 3 * target_bins
    # number of tiles conservatively scales with iterations as 0.33 * 6 ** iter
    niter = int(math.ceil(math.log(conserv_bins / 0.33) / math.log(6)))
    logger.info(f"Running aperiodic monotile algorithm for {niter} iterations")

    if direct_port_impl:
        logger.info("Using direct port implementation, might be slow!")
        raw = aperiodic_monotile_raw_ported(niter=niter, construction=construction, kind=kind)
        border_edges: list[LineSegment] | None = None
    else:
        state = aperiodic_monotile_raw(niter=niter, kind=kind)
        meta_tile = state.H8 if construction == "H8" else state.H7
        raw = meta_tile.as_polygons()
        border_edges = meta_tile.border_edges()

    return rectanglize_tiling(
        tiles=raw,
        target_bins=bins,
        border_edges_precomputed=border_edges,
    )
