import logging
import math

from funbin.einstein.vectorized import AperiodicMonotileConstruction, AperiodicMonotileKind, aperiodic_monotile_raw
from funbin.geometry import Polygon, rectanglize_tiling

logger = logging.getLogger(__name__)


def aperiodic_monotile(
    bins: tuple[int, int],
    kind: AperiodicMonotileKind = "hat",
    construction: AperiodicMonotileConstruction = "H8",
) -> list[Polygon]:
    target_bins = bins[0] * bins[1]
    conserv_bins = 3 * target_bins
    # number of tiles conservatively scales with iterations as 0.33 * 6 ** iter
    niter = int(math.ceil(math.log(conserv_bins / 0.33) / math.log(6)))
    logger.info(f"Running aperiodic monotile algorithm for {niter} iterations")
    raw = aperiodic_monotile_raw(niter=niter, construction=construction, kind=kind)
    logger.info(f"Got {len(raw)} tiles, rectanglizing")
    # TODO: track border edges as we're building the tiling
    return rectanglize_tiling(tiles=raw, target_bins=bins)
