from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SectorGraphFeatures:
    nodes: pd.Index
    adjacency: pd.DataFrame
    embeddings: pd.DataFrame
    rotation_scores: pd.Series


def build_sector_graph_features(
    sector_returns: pd.DataFrame,
    sector_flows: pd.DataFrame | None = None,
    manual_edges: pd.DataFrame | None = None,
    window: int = 60,
) -> SectorGraphFeatures:
    returns = sector_returns.tail(window).astype(float)
    nodes = returns.columns
    return_corr = returns.corr().fillna(0.0)
    adjacency = return_corr.abs()
    if sector_flows is not None:
        flow_corr = sector_flows.reindex(columns=nodes).tail(window).astype(float).corr().fillna(0.0).abs()
        adjacency = 0.7 * adjacency + 0.3 * flow_corr
    if manual_edges is not None and {"source", "target", "weight"}.issubset(manual_edges.columns):
        for _, row in manual_edges.iterrows():
            source = row["source"]
            target = row["target"]
            if source in adjacency.index and target in adjacency.columns:
                weight = float(row["weight"])
                adjacency.loc[source, target] = max(adjacency.loc[source, target], weight)
                adjacency.loc[target, source] = max(adjacency.loc[target, source], weight)
    np.fill_diagonal(adjacency.values, 0.0)
    row_sum = adjacency.sum(axis=1).replace(0.0, np.nan)
    normalized = adjacency.div(row_sum, axis=0).fillna(0.0)
    latest_return = returns.tail(5).mean()
    neighbor_momentum = normalized @ latest_return.reindex(nodes).fillna(0.0)
    volatility = returns.tail(window).std(ddof=0).replace(0.0, np.nan)
    rotation_scores = (0.6 * latest_return + 0.4 * neighbor_momentum) / volatility
    embeddings = pd.DataFrame(
        {
            "own_momentum": latest_return.reindex(nodes),
            "neighbor_momentum": neighbor_momentum.reindex(nodes),
            "degree": adjacency.sum(axis=1),
            "rotation_score": rotation_scores.reindex(nodes),
        },
        index=nodes,
    ).replace([np.inf, -np.inf], np.nan)
    return SectorGraphFeatures(nodes=nodes, adjacency=adjacency, embeddings=embeddings, rotation_scores=rotation_scores)

