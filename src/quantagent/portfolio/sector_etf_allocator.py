from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SectorHedgeCandidate:
    symbol: str
    sector: str
    hedge_weight: float
    rationale: str


DEFAULT_SECTOR_HEDGES: dict[str, str] = {
    "broad_market": "510300.SH",
    "semiconductor": "512480.SH",
    "technology": "515000.SH",
    "defensive": "510050.SH",
}


def allocate_sector_hedges(
    sector_exposure: dict[str, float],
    sector_risk: dict[str, float],
    total_hedge_weight: float,
    hedge_map: dict[str, str] | None = None,
) -> list[SectorHedgeCandidate]:
    hedge_map = hedge_map or DEFAULT_SECTOR_HEDGES
    pressure = {
        sector: max(0.0, exposure) * max(0.0, sector_risk.get(sector, 0.0))
        for sector, exposure in sector_exposure.items()
    }
    total = sum(pressure.values())
    if total <= 0 or total_hedge_weight <= 0:
        return []
    candidates = []
    for sector, value in sorted(pressure.items(), key=lambda item: item[1], reverse=True):
        if value <= 0:
            continue
        symbol = hedge_map.get(sector, hedge_map.get("broad_market", "510300.SH"))
        candidates.append(
            SectorHedgeCandidate(
                symbol=symbol,
                sector=sector,
                hedge_weight=float(total_hedge_weight * value / total),
                rationale=f"sector_exposure={sector_exposure.get(sector, 0.0):.3f}, sector_risk={sector_risk.get(sector, 0.0):.3f}",
            )
        )
    return candidates
