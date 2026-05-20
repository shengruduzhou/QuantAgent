"""Alpha181 factor library for V7 A-share research.

Alpha181 is the fixed daily feature set used by the V7 training pipeline:

* ``alpha001`` .. ``alpha101``: WorldQuant Alpha101 daily OHLCV
  approximations from :mod:`quantagent.factors.alpha101`.
* ``alpha102`` .. ``alpha181``: the full 80-factor CICC-inspired A-share
  price/volume templates from :mod:`quantagent.factors.cicc_ashare80`.

GA-synthesised factors are loaded as an optional extension and intentionally
keep their ``synth_*`` names so they do not masquerade as reviewed
fixed-library factors. The base ``alpha181`` set is 101+80 = 181 factors;
synth factors add to the runtime total.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantagent.factors.alpha101 import compute_alpha101
from quantagent.factors.cicc_ashare80 import cicc_ashare80_names, compute_cicc_ashare80_factors
from quantagent.factors.factor_synthesis import compute_synthesized_factors


ALPHA181_CICC_COUNT = 80
ALPHA181_NAMES: tuple[str, ...] = tuple(f"alpha{i:03d}" for i in range(1, 182))
ALPHA181_CICC_NAME_MAP: dict[str, str] = {
    source: f"alpha{102 + idx:03d}"
    for idx, source in enumerate(cicc_ashare80_names())
}


def compute_alpha181(
    frame: pd.DataFrame,
    names: list[str] | None = None,
    synthesized_definitions_path: str | Path | None = None,
    *,
    wide: bool = False,
) -> pd.DataFrame:
    """Compute fixed Alpha181 plus optional GA-synthesised factor extensions.

    Output formats
    --------------
    * ``wide=False`` (default): long-form rows
      ``trade_date, symbol, factor_name, factor_value`` (backwards-compatible).
    * ``wide=True``: wide-form columns ``trade_date, symbol, alpha001 ...
      alpha181, synth_*``. This path is the recommended one for streaming
      builds on large universes — it skips the 9.8 GB long-form
      intermediate that the downstream pivot needs to allocate.
    """
    if len(ALPHA181_CICC_NAME_MAP) != ALPHA181_CICC_COUNT:
        raise RuntimeError(
            f"alpha181 expects {ALPHA181_CICC_COUNT} CICC factors, "
            f"got {len(ALPHA181_CICC_NAME_MAP)}"
        )

    requested = set(names) if names else set(ALPHA181_NAMES)
    alpha101_names = [
        name for name in requested
        if name.startswith("alpha") and int(name.removeprefix("alpha")) <= 101
    ]
    cicc_source_names = [
        source
        for source, alpha_name in ALPHA181_CICC_NAME_MAP.items()
        if alpha_name in requested
    ]

    if wide:
        wide_frames: list[pd.DataFrame] = []
        if alpha101_names:
            wide_frames.append(compute_alpha101(frame, names=sorted(alpha101_names), wide=True))
        if cicc_source_names:
            cicc_wide = compute_cicc_ashare80_factors(frame, names=cicc_source_names, wide=True)
            rename_map = {src: ALPHA181_CICC_NAME_MAP[src] for src in cicc_source_names
                          if src in cicc_wide.columns}
            cicc_wide = cicc_wide.rename(columns=rename_map)
            wide_frames.append(cicc_wide)
        if synthesized_definitions_path is not None:
            synthesized = compute_synthesized_factors(frame, synthesized_definitions_path)
            if names:
                synthesized = synthesized[synthesized["factor_name"].isin(requested)]
            if not synthesized.empty:
                synth_wide = synthesized.pivot_table(
                    index=["trade_date", "symbol"],
                    columns="factor_name",
                    values="factor_value",
                    aggfunc="last",
                ).reset_index()
                synth_wide.columns = [str(c) for c in synth_wide.columns]
                wide_frames.append(synth_wide)
        if not wide_frames:
            return pd.DataFrame(columns=["trade_date", "symbol"])
        out = wide_frames[0]
        for piece in wide_frames[1:]:
            out = out.merge(piece, on=["trade_date", "symbol"], how="outer")
        return out

    frames: list[pd.DataFrame] = []
    if alpha101_names:
        frames.append(compute_alpha101(frame, names=sorted(alpha101_names)))
    if cicc_source_names:
        cicc = compute_cicc_ashare80_factors(frame, names=cicc_source_names)
        cicc = cicc.copy()
        cicc["factor_name"] = cicc["factor_name"].map(ALPHA181_CICC_NAME_MAP)
        frames.append(cicc)

    if synthesized_definitions_path is not None:
        synthesized = compute_synthesized_factors(frame, synthesized_definitions_path)
        if names:
            synthesized = synthesized[synthesized["factor_name"].isin(requested)]
        if not synthesized.empty:
            frames.append(synthesized)

    if not frames:
        return pd.DataFrame(columns=["trade_date", "symbol", "factor_name", "factor_value"])
    return pd.concat(frames, ignore_index=True, sort=False)


def alpha181_source_map() -> dict[str, str]:
    """Map Alpha181 output names to their implementation source names."""
    mapping = {f"alpha{i:03d}": f"alpha101.alpha{i:03d}" for i in range(1, 102)}
    mapping.update(
        {target: f"cicc_ashare80.{source}" for source, target in ALPHA181_CICC_NAME_MAP.items()}
    )
    return mapping


__all__ = [
    "ALPHA181_NAMES",
    "ALPHA181_CICC_NAME_MAP",
    "ALPHA181_CICC_COUNT",
    "compute_alpha181",
    "alpha181_source_map",
]
