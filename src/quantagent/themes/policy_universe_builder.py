"""Build a per-date policy-driven stock universe parquet.

This is a pragmatic orchestrator that stitches together:

* **Policy evidence** (from the evidence_store CSV cache produced by
  :mod:`quantagent.cli.v7_evidence`),
* **Theme keyword pack** (:data:`THEMES_15TH_FIVE_YEAR_PLAN`),
* **Theme → industry map** (a static editable mapping, see
  :data:`THEME_TO_INDUSTRY`),
* **Symbol universe** from a local Qlib provider directory.

It outputs a single parquet at ``data/v7/silver/policy_universe.parquet``
with the columns ``[date, symbol, theme, role, score, evidence_count]``.

The output is intentionally **conservative**: when there is no rich
industry mapping for a symbol we still emit it as ``role='core'`` against
its theme, with a score derived from the count of supporting evidence
documents. Downstream code (the v7 training dataset builder, the paper
loop) can choose to filter on ``role`` / ``score`` thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quantagent.config.paths import quant_paths
from quantagent.themes.keyword_packs import THEMES_15TH_FIVE_YEAR_PLAN

# A best-effort theme → industry-token map. The values are substring tokens
# matched against the local instrument list. When no token matches, the
# theme is broadcast to the full universe (with a low score) so downstream
# code still produces output.
THEME_TO_INDUSTRY: dict[str, tuple[str, ...]] = {
    "ai_compute": ("SH688", "SZ300"),  # STAR + ChiNext bias
    "semiconductor_domestic_substitution": ("SH688", "SZ300"),
    "energy_storage": ("SH600", "SZ300"),
    "smart_grid": ("SH600", "SH601"),
    "photovoltaic": ("SH601", "SH600", "SZ300"),
    "wind_power": ("SH600", "SH601"),
    "new_energy_vehicle": ("SH600", "SZ002"),
    "humanoid_robotics": ("SH600", "SZ300"),
    "high_end_manufacturing": ("SH600", "SH601"),
    "commercial_space": ("SH688", "SH600"),
    "low_altitude_economy": ("SH600", "SZ002"),
    "innovative_drug": ("SH688", "SH600", "SZ300"),
    "defense_modernisation": ("SH600", "SZ002"),
    "cyber_security": ("SZ300", "SH688"),
    "seed_industry": ("SH600", "SZ000"),
    "advanced_materials": ("SH600", "SZ002"),
    "rare_earth_strategic": ("SH600",),
    "controlled_fusion": ("SH688",),
    "data_factor": ("SH600", "SZ300"),
}


@dataclass(frozen=True)
class PolicyUniverseConfig:
    as_of_date: str
    qlib_provider_uri: str
    evidence_store_root: Path
    output_path: Path
    role_top_quantile_leader: float = 0.10
    role_top_quantile_core: float = 0.40
    min_evidence_count: int = 1
    fallback_symbols: int = 200


def _read_evidence_cache(root: Path, as_of_date: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not root.exists():
        return pd.DataFrame()
    for path in sorted(root.glob("*.csv")):
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        df = df[df["published_at"].notna() & (df["published_at"] <= pd.Timestamp(as_of_date))]
    return df


def _theme_counts_from_evidence(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if df.empty:
        return counts
    text_col = df.get("theme_candidates")
    if text_col is None:
        # fall back to body+title keyword count using THEMES_15TH_FIVE_YEAR_PLAN
        text = ((df.get("title", "").fillna("") + " " + df.get("body", "").fillna(""))).str.lower()
        for kw, themes in THEMES_15TH_FIVE_YEAR_PLAN.items():
            matches = int(text.str.contains(kw.lower(), regex=False).sum())
            for theme in themes:
                counts[theme] = counts.get(theme, 0) + matches
        return counts
    for cell in text_col.fillna(""):
        if not cell:
            continue
        for theme in str(cell).split(","):
            theme = theme.strip()
            if theme:
                counts[theme] = counts.get(theme, 0) + 1
    return counts


def _qlib_symbol_list(provider_uri: str) -> list[str]:
    instruments = Path(provider_uri) / "instruments" / "all.txt"
    if not instruments.exists():
        return []
    syms: list[str] = []
    with instruments.open() as fh:
        for line in fh:
            parts = line.split()
            if parts:
                syms.append(parts[0])
    return syms


def _assign_role(score: float, leader_cut: float, core_cut: float) -> str:
    if score >= leader_cut:
        return "leader"
    if score >= core_cut:
        return "core"
    return "support"


def build_policy_universe(cfg: PolicyUniverseConfig) -> pd.DataFrame:
    """Materialise a thematic stock universe and persist it to parquet."""
    evidence = _read_evidence_cache(cfg.evidence_store_root, cfg.as_of_date)
    theme_counts = _theme_counts_from_evidence(evidence)
    symbols = _qlib_symbol_list(cfg.qlib_provider_uri)
    if not symbols:
        raise FileNotFoundError(
            f"No instruments found under {cfg.qlib_provider_uri}; ensure qlib data exists."
        )

    # When evidence yields no themes (fully offline first run) we still
    # emit a default "national_strategic_plan" universe so the rest of the
    # pipeline can compose. This makes the first paper-loop iteration
    # debuggable without waiting for live network ingest.
    if not theme_counts:
        theme_counts = {"national_strategic_plan": 1}

    rows: list[dict[str, object]] = []
    total_evidence = float(sum(theme_counts.values()) or 1.0)
    for theme, count in theme_counts.items():
        if count < cfg.min_evidence_count:
            continue
        tokens = THEME_TO_INDUSTRY.get(theme, ())
        matched = [s for s in symbols if any(s.startswith(t) for t in tokens)] if tokens else []
        if not matched:
            # broadcast to a deterministic fallback slice
            matched = symbols[: cfg.fallback_symbols]
        weight = count / total_evidence
        # rank symbols by their numeric tail (stable, fast); top-quantile → leader
        ranked = sorted(matched)
        n = len(ranked)
        leader_idx = max(1, int(n * cfg.role_top_quantile_leader))
        core_idx = max(leader_idx, int(n * cfg.role_top_quantile_core))
        for i, sym in enumerate(ranked):
            if i < leader_idx:
                role = "leader"
            elif i < core_idx:
                role = "core"
            else:
                role = "support"
            score = float(weight) * (1.0 - i / max(1, n)) + 1e-6 * (n - i)
            rows.append(
                {
                    "date": cfg.as_of_date,
                    "symbol": sym,
                    "theme": theme,
                    "role": role,
                    "score": score,
                    "evidence_count": int(count),
                }
            )

    frame = pd.DataFrame(rows, columns=["date", "symbol", "theme", "role", "score", "evidence_count"])
    if frame.empty:
        raise RuntimeError("policy universe is empty — check evidence_store and theme mapping")

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cfg.output_path, index=False)
    return frame


def build_for_today(as_of_date: str) -> pd.DataFrame:
    paths = quant_paths()
    cfg = PolicyUniverseConfig(
        as_of_date=as_of_date,
        qlib_provider_uri=str(paths.raw / "qlib" / "cn_data" / "1d"),
        evidence_store_root=paths.data_root / "v7" / "evidence",
        output_path=paths.data_root / "v7" / "silver" / "policy_universe.parquet",
    )
    return build_policy_universe(cfg)


__all__ = [
    "PolicyUniverseConfig",
    "THEME_TO_INDUSTRY",
    "build_policy_universe",
    "build_for_today",
]
