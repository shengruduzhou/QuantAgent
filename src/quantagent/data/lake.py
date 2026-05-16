"""Canonical V7 data lake layout.

The V7 lake is split into three medallion tiers under ``data/v7``:

* ``raw/<vendor>/`` — vendor-native dumps (qlib, akshare, tushare, disclosures).
* ``silver/<dataset>/`` — normalised, schema-validated, PIT-tagged tables.
* ``gold/training_dataset/`` — model-ready as-of joined frames with labels.

Every silver/gold writer must also emit a sibling ``manifests/<dataset>.json``
recording provenance and data-quality status. Helpers in this module keep
all those paths in one place so the CLI, bootstrap and dataset builders
agree on where to read and write.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_V7_ROOT = "data/v7"


@dataclass(frozen=True)
class V7LakePaths:
    root: Path
    raw_qlib: Path
    raw_akshare: Path
    raw_tushare: Path
    raw_disclosures: Path
    silver_market_panel: Path
    silver_fundamentals: Path
    silver_valuation: Path
    silver_disclosures: Path
    gold_training_dataset: Path
    manifests: Path

    def ensure(self) -> "V7LakePaths":
        for path in (
            self.root,
            self.raw_qlib,
            self.raw_akshare,
            self.raw_tushare,
            self.raw_disclosures,
            self.silver_market_panel,
            self.silver_fundamentals,
            self.silver_valuation,
            self.silver_disclosures,
            self.gold_training_dataset,
            self.manifests,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


def v7_lake_paths(root: str | Path = DEFAULT_V7_ROOT) -> V7LakePaths:
    base = Path(root)
    return V7LakePaths(
        root=base,
        raw_qlib=base / "raw" / "qlib",
        raw_akshare=base / "raw" / "akshare",
        raw_tushare=base / "raw" / "tushare",
        raw_disclosures=base / "raw" / "disclosures",
        silver_market_panel=base / "silver" / "market_panel",
        silver_fundamentals=base / "silver" / "fundamentals",
        silver_valuation=base / "silver" / "valuation",
        silver_disclosures=base / "silver" / "disclosures",
        gold_training_dataset=base / "gold" / "training_dataset",
        manifests=base / "manifests",
    )


def manifest_path(dataset_name: str, root: str | Path = DEFAULT_V7_ROOT) -> Path:
    return v7_lake_paths(root).manifests / f"{dataset_name}.json"
