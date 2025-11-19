import math
from typing import Literal

from funbin.geometry import Polygon, rectanglize_tiling
from funbin.penrose.de_brujin import penrose_P3_de_Brujin
from funbin.penrose.deflate import pent1, pent1_deflation_border_conservative, run_defaltion
from funbin.penrose.robinson_triangles import golden_ratio, penrose_tiling_robinson, robinson_triangles_border

__all__ = ["penrose_P3_de_Brujin", "penrose_tiling"]

PenroseTilingKind = Literal["P1", "P2", "P3"]


def penrose_tiling(kind: PenroseTilingKind, bins: tuple[int, int]) -> list[Polygon]:
    target_bins = bins[0] * bins[1]
    if kind in ("P2", "P3"):
        # safety margin + add a bit of random variance to the final patch
        conserv_bins = 2 * target_bins
        # number of tiles (conservatively) scales with divisions as 3 * (1 + tau) ** division
        divisions = int(math.ceil(math.log(conserv_bins / 3) / math.log(1 + golden_ratio)))
        return rectanglize_tiling(
            tiles=penrose_tiling_robinson(
                kind=kind,
                divisions=divisions,
                include_incomplete_tiles=True,
            ),
            target_bins=bins,
            border_edges_precomputed=robinson_triangles_border(),
        )
    elif kind == "P1":
        conserv_bins = 5 * target_bins
        # number of tiles conservatively scales with iterations as 0.5 * 6 ** iter
        iterations = int(math.ceil(math.log(conserv_bins / 0.5) / math.log(6)))
        return rectanglize_tiling(
            tiles=run_defaltion(tiles=pent1, iterations=iterations),
            target_bins=bins,
            border_edges_precomputed=pent1_deflation_border_conservative(),
        )
    else:
        raise ValueError(f"Unexpected kind: {kind}")
