from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd


FactorNodeCompute = Callable[[pd.DataFrame], pd.Series | pd.DataFrame]


@dataclass(frozen=True)
class FactorNode:
    name: str
    compute: FactorNodeCompute
    dependencies: tuple[str, ...] = ()
    required_columns: tuple[str, ...] = ()
    output_column: str | None = None


@dataclass(frozen=True)
class FactorDAGResult:
    frame: pd.DataFrame
    execution_order: tuple[str, ...]


class FactorDAG:
    """Small deterministic factor dependency graph."""

    def __init__(self) -> None:
        self.nodes: dict[str, FactorNode] = {}

    def add(self, node: FactorNode) -> None:
        if node.name in self.nodes:
            raise ValueError(f"Duplicate factor node: {node.name}")
        duplicates = [dep for dep in node.dependencies if node.dependencies.count(dep) > 1]
        if duplicates:
            raise ValueError(f"Duplicate dependencies for {node.name}: {sorted(set(duplicates))}")
        self.nodes[node.name] = node

    def topological_order(self, selected: list[str] | None = None) -> list[str]:
        selected_set = set(selected or self.nodes)
        needed = self._closure(selected_set)
        indegree: dict[str, int] = {name: 0 for name in needed}
        children: dict[str, list[str]] = defaultdict(list)
        for name in needed:
            for dep in self.nodes[name].dependencies:
                if dep not in self.nodes:
                    raise KeyError(f"Missing dependency {dep} for factor {name}")
                if dep in needed:
                    indegree[name] += 1
                    children[dep].append(name)
        queue = deque(sorted(name for name, degree in indegree.items() if degree == 0))
        order: list[str] = []
        while queue:
            name = queue.popleft()
            order.append(name)
            for child in sorted(children[name]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(order) != len(needed):
            raise ValueError("Factor DAG contains a cycle")
        return [name for name in order if name in selected_set or name in needed]

    def execute(self, frame: pd.DataFrame, selected: list[str] | None = None) -> FactorDAGResult:
        result = frame.copy()
        order = self.topological_order(selected)
        for name in order:
            node = self.nodes[name]
            missing = set(node.required_columns).difference(result.columns)
            if missing:
                raise ValueError(f"Missing required columns for {name}: {sorted(missing)}")
            output = node.compute(result)
            column = node.output_column or name
            if isinstance(output, pd.Series):
                result[column] = output.to_numpy()
            else:
                for out_col in output.columns:
                    result[out_col] = output[out_col].to_numpy()
        return FactorDAGResult(frame=result.sort_values(["trade_date", "symbol"]).reset_index(drop=True), execution_order=tuple(order))

    def _closure(self, selected: set[str]) -> set[str]:
        needed: set[str] = set()

        def visit(name: str) -> None:
            if name in needed:
                return
            if name not in self.nodes:
                raise KeyError(f"Unknown factor node: {name}")
            needed.add(name)
            for dep in self.nodes[name].dependencies:
                visit(dep)

        for item in selected:
            visit(item)
        return needed
