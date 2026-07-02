"""Stage 12 Task 1 + 6 — Historical PIT feature registry + factor lifecycle.

A single governance catalog for every feature that may enter a model, recording
its point-in-time contract and its promotion stage so nothing goes to production
without passing the pipeline and no factor is silently re-optimised on data that
post-dates its inception.

Registry fields (Task 1):
  feature_name, source, available_at_rule, lag_rule, feature_version,
  schema_hash, allowed_train_start, leakage_risk, first_live_date

Lifecycle stages (Task 6): candidate -> historical_walkforward -> untouched_oos
-> forward_paper -> capital_shadow -> production. factor_inception_date pins when
the factor was first researched, so later evaluation windows can be checked for
research-overfit leakage.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

REGISTRY_PATH = Path("configs/feature_registry.json")
LIFECYCLE = ("candidate", "historical_walkforward", "untouched_oos",
             "forward_paper", "capital_shadow", "production")
LEAKAGE = ("none", "low", "medium", "high", "forward_only")


@dataclass
class FeatureSpec:
    feature_name: str
    source: str
    available_at_rule: str          # how available_at is derived (PIT contract)
    lag_rule: str                   # applied lag before use
    feature_version: str
    allowed_train_start: str        # earliest date usable for training
    leakage_risk: str               # one of LEAKAGE
    first_live_date: str            # first date the feature exists live
    factor_inception_date: str      # when first researched (overfit guard)
    lifecycle_stage: str = "candidate"
    schema_hash: str = ""
    notes: str = ""

    def compute_hash(self) -> str:
        payload = f"{self.feature_name}|{self.source}|{self.available_at_rule}|{self.lag_rule}|{self.feature_version}"
        self.schema_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return self.schema_hash

    def validate(self) -> list[str]:
        errs = []
        if self.leakage_risk not in LEAKAGE:
            errs.append(f"{self.feature_name}: bad leakage_risk {self.leakage_risk!r}")
        if self.lifecycle_stage not in LIFECYCLE:
            errs.append(f"{self.feature_name}: bad lifecycle_stage {self.lifecycle_stage!r}")
        return errs


@dataclass
class FeatureRegistry:
    features: list[FeatureSpec] = field(default_factory=list)

    def add(self, spec: FeatureSpec) -> None:
        spec.compute_hash()
        self.features = [f for f in self.features if f.feature_name != spec.feature_name] + [spec]

    def get(self, name: str) -> FeatureSpec | None:
        return next((f for f in self.features if f.feature_name == name), None)

    def advance(self, name: str, to_stage: str) -> None:
        f = self.get(name)
        if f is None:
            raise KeyError(name)
        if to_stage not in LIFECYCLE:
            raise ValueError(to_stage)
        if LIFECYCLE.index(to_stage) > LIFECYCLE.index(f.lifecycle_stage) + 1:
            raise ValueError(f"cannot skip stages {f.lifecycle_stage} -> {to_stage}")
        f.lifecycle_stage = to_stage

    def trainable(self, as_of_train_start: str) -> list[FeatureSpec]:
        """Features PIT-safe to train from as_of_train_start (excludes forward-only)."""
        return [f for f in self.features
                if f.leakage_risk != "forward_only" and f.allowed_train_start <= as_of_train_start]

    def validate_all(self) -> list[str]:
        errs = []
        for f in self.features:
            errs += f.validate()
        return errs

    def save(self, path: Path = REGISTRY_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(f) for f in self.features], ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: Path = REGISTRY_PATH) -> "FeatureRegistry":
        if not path.exists():
            return cls()
        return cls([FeatureSpec(**d) for d in json.loads(path.read_text())])
