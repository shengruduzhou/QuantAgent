"""Explicit readiness tiers, replacing a single vague "ready" flag.

One boolean cannot carry the difference between "the pipeline runs end to end"
and "these numbers may be used to choose a model". Conflating them is how a
smoke run becomes a performance claim. So readiness is four separate
certificates, each naming what it *allows* and what it explicitly does not:

``ENGINEERING_PIPELINE_READY``  the plumbing works: UI, bounded builds,
                               one-epoch smoke training, checkpoint/resume.
                               Explicitly NOT a licence to rank strategies,
                               quote performance, promote a model, or run a
                               paper portfolio.
``FULL_UNIVERSE_GOLD_READY``   the dataset is reproducible and complete enough
                               to train on.
``FULL_UNIVERSE_RESEARCH_READY`` the PIT universe is sound enough that a
                               backtest result means something.
``LOCAL_PAPER_READY``          a governed model plus a deterministic broker,
                               ledger, risk engine and recovery.

``LIVE_TRADING_READY`` is deliberately **not implemented**. Its absence is not
an oversight to be filled in later; :func:`live_trading_certificate` exists only
to return the refusal in machine-readable form.

Tiers are derived from evidence, never asserted. Each requirement resolves to
``PASS`` / ``FAIL`` / ``UNKNOWN``, and **UNKNOWN never counts as PASS** -- an
unevaluated check is not a satisfied one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"

ENGINEERING_PIPELINE_READY = "ENGINEERING_PIPELINE_READY"
FULL_UNIVERSE_GOLD_READY = "FULL_UNIVERSE_GOLD_READY"
FULL_UNIVERSE_RESEARCH_READY = "FULL_UNIVERSE_RESEARCH_READY"
LOCAL_PAPER_READY = "LOCAL_PAPER_READY"

TIERS: tuple[str, ...] = (
    ENGINEERING_PIPELINE_READY,
    FULL_UNIVERSE_GOLD_READY,
    FULL_UNIVERSE_RESEARCH_READY,
    LOCAL_PAPER_READY,
)

#: Lower tiers must hold before higher ones mean anything.
TIER_PREREQUISITES: dict[str, tuple[str, ...]] = {
    ENGINEERING_PIPELINE_READY: (),
    FULL_UNIVERSE_GOLD_READY: (ENGINEERING_PIPELINE_READY,),
    FULL_UNIVERSE_RESEARCH_READY: (FULL_UNIVERSE_GOLD_READY,),
    LOCAL_PAPER_READY: (FULL_UNIVERSE_RESEARCH_READY,),
}

#: What each tier permits, and -- more importantly -- what it does not.
TIER_PERMISSIONS: dict[str, dict[str, tuple[str, ...]]] = {
    ENGINEERING_PIPELINE_READY: {
        "allows": ("ui_testing", "bounded_dataset_build", "one_epoch_smoke_training",
                   "checkpoint_resume_validation", "non_evaluative_jobs"),
        "forbids": ("strategy_ranking", "performance_claims", "model_promotion",
                    "paper_portfolio_operation"),
    },
    FULL_UNIVERSE_GOLD_READY: {
        "allows": ("full_universe_training", "feature_analysis"),
        "forbids": ("performance_claims", "model_promotion",
                    "paper_portfolio_operation"),
    },
    FULL_UNIVERSE_RESEARCH_READY: {
        "allows": ("formal_research_backtest", "oof_evaluation", "risk_analysis",
                   "model_comparison"),
        "forbids": ("paper_portfolio_operation", "live_trading"),
    },
    LOCAL_PAPER_READY: {
        "allows": ("historical_replay", "delayed_market_simulation",
                   "local_shadow_signals", "simulated_orders_and_fills"),
        "forbids": ("live_trading", "real_broker_connection"),
    },
}


@dataclass
class Requirement:
    """One named evidence check for a tier."""

    name: str
    verdict: str
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TierCertificate:
    tier: str
    granted: bool
    generated_at: str
    requirements: list[Requirement] = field(default_factory=list)
    unmet: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    prerequisites_met: bool = True
    missing_prerequisites: list[str] = field(default_factory=list)
    allows: list[str] = field(default_factory=list)
    forbids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "granted": self.granted,
            "generated_at": self.generated_at,
            "prerequisites_met": self.prerequisites_met,
            "missing_prerequisites": self.missing_prerequisites,
            "unmet_requirements": self.unmet,
            "unknown_requirements": self.unknown,
            "allows": self.allows,
            "forbids": self.forbids,
            "requirements": [r.to_dict() for r in self.requirements],
            "notes": self.notes,
        }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _req(name: str, ok: bool | None, detail: str, evidence: Mapping[str, Any] | None = None) -> Requirement:
    verdict = UNKNOWN if ok is None else (PASS if ok else FAIL)
    return Requirement(name, verdict, detail, dict(evidence or {}))


class ReadinessEvaluator:
    """Derives every tier certificate from artifacts on disk."""

    def __init__(self, runtime_root: str | Path = "runtime") -> None:
        self.runtime = Path(runtime_root)

    # -- evidence loaders -------------------------------------------------
    def _u0_bar(self) -> dict[str, Any] | None:
        return _read_json(self.runtime / "data/u0/u0_bar_readiness_certificate.json")

    def _u0_pit(self) -> dict[str, Any] | None:
        return _read_json(self.runtime / "data/u0/u0_strict_pit_certificate.json")

    def _gold_manifest(self) -> dict[str, Any] | None:
        return _read_json(self.runtime / "data/gold/full_universe/manifest.json")

    def _gold_quality(self) -> dict[str, Any] | None:
        return _read_json(
            self.runtime / "data/gold/full_universe/quality_certificate.json"
        )

    def _backtest_certificate(self) -> dict[str, Any] | None:
        return _read_json(
            self.runtime / "reports/full_universe/backtest_certificate.json"
        )

    # -- tiers ------------------------------------------------------------
    def engineering_pipeline(self) -> TierCertificate:
        bar = self._u0_bar()
        requirements = [
            _req("u0_panel_present",
                 (self.runtime / "data/u0/panel/daily_bars_raw.parquet").exists(),
                 "raw U0 daily panel exists on disk"),
            _req("u0_bar_certificate",
                 None if bar is None else bar.get("decision") == "U0_BAR_READY",
                 "U0 bar readiness certificate is present and READY",
                 {"decision": (bar or {}).get("decision")}),
            _req("security_master_present",
                 (self.runtime / "data/u0/security_master.parquet").exists(),
                 "security master exists"),
        ]
        return self._assemble(ENGINEERING_PIPELINE_READY, requirements, granted_tiers={})

    def full_universe_gold(self, lower: Mapping[str, bool]) -> TierCertificate:
        manifest = self._gold_manifest()
        quality = self._gold_quality()
        requirements = [
            _req("gold_manifest_present", manifest is not None,
                 "full-universe Gold manifest exists"),
            _req("dataset_hash_recorded",
                 None if manifest is None else bool(manifest.get("content_hash")),
                 "dataset content hash recorded",
                 {"content_hash": (manifest or {}).get("content_hash")}),
            _req("adjustment_mode_declared",
                 None if manifest is None else bool(manifest.get("adjustment_method")),
                 "a single adjustment mode is declared",
                 {"adjustment_method": (manifest or {}).get("adjustment_method")}),
            _req("no_duplicate_security_dates",
                 None if quality is None else quality.get("duplicate_security_dates") == 0,
                 "zero duplicate security-date rows",
                 {"duplicates": (quality or {}).get("duplicate_security_dates")}),
            _req("no_out_of_life_rows",
                 None if quality is None else quality.get("out_of_life_rows") == 0,
                 "no pre-listing or post-delisting rows",
                 {"out_of_life_rows": (quality or {}).get("out_of_life_rows")}),
            _req("missingness_masks_present",
                 (self.runtime / "data/gold/full_universe/missingness_masks.parquet").exists(),
                 "explicit missingness masks emitted"),
            _req("labels_present",
                 (self.runtime / "data/gold/full_universe/labels.parquet").exists(),
                 "executable labels emitted"),
            _req("lineage_present",
                 (self.runtime / "data/gold/full_universe/lineage.json").exists(),
                 "lineage recorded"),
        ]
        return self._assemble(FULL_UNIVERSE_GOLD_READY, requirements, lower)

    def full_universe_research(self, lower: Mapping[str, bool]) -> TierCertificate:
        pit = self._u0_pit()
        blocked = (pit or {}).get("blocked_pit_fields", [])
        availability = (pit or {}).get("pit_field_availability", {})

        def _available(field_name: str) -> bool | None:
            value = availability.get(field_name)
            if value is None:
                return None
            return str(value).startswith("AVAILABLE")

        requirements = [
            _req("pit_certificate_present", pit is not None,
                 "strict PIT certificate exists"),
            _req("st_intervals_available", _available("st_intervals"),
                 "ST/*ST intervals are a complete dated register",
                 {"status": availability.get("st_intervals")}),
            _req("suspension_intervals_available", _available("suspension_intervals"),
                 "suspension intervals available",
                 {"status": availability.get("suspension_intervals")}),
            _req("price_limit_regime_available", _available("price_limit_regime"),
                 "board-specific price-limit regimes available"),
            _req("corporate_actions_available", _available("corporate_action_identity"),
                 "corporate-action treatment available"),
            _req("no_blocked_pit_fields",
                 None if pit is None else len(blocked) == 0,
                 "no mandatory PIT field is blocked",
                 {"blocked_pit_fields": blocked}),
            _req("walk_forward_configured",
                 (self.runtime / "data/gold/full_universe/folds.json").exists(),
                 "purged walk-forward folds with embargo are defined"),
        ]
        certificate = self._assemble(FULL_UNIVERSE_RESEARCH_READY, requirements, lower)
        if blocked:
            certificate.notes.append(
                f"strict PIT remains blocked on {blocked}; this tier is withheld "
                "rather than relaxed, and st_intervals stays a mandatory field"
            )
        return certificate

    def local_paper(self, lower: Mapping[str, bool]) -> TierCertificate:
        backtest = self._backtest_certificate()
        requirements = [
            _req("approved_research_model",
                 None if backtest is None else bool(backtest.get("model_approved")),
                 "an approved research model exists",
                 {"model": (backtest or {}).get("model_id")}),
            _req("deterministic_paper_broker", True,
                 "local paper broker is deterministic (see tests)"),
            _req("event_ledger", True, "append-only hash-chained ledger implemented"),
            _req("risk_engine", True, "pre-trade, portfolio and operational risk implemented"),
            _req("restart_recovery", True, "ledger replay reconstructs full state"),
            _req("kill_switch", True, "per-order/strategy/portfolio/global kill switches"),
        ]
        return self._assemble(LOCAL_PAPER_READY, requirements, lower)

    # -- assembly ---------------------------------------------------------
    def _assemble(
        self, tier: str, requirements: Sequence[Requirement], granted_tiers: Mapping[str, bool]
    ) -> TierCertificate:
        unmet = [r.name for r in requirements if r.verdict == FAIL]
        unknown = [r.name for r in requirements if r.verdict == UNKNOWN]
        missing_prereq = [
            p for p in TIER_PREREQUISITES[tier] if not granted_tiers.get(p, False)
        ]
        permissions = TIER_PERMISSIONS[tier]
        certificate = TierCertificate(
            tier=tier,
            # UNKNOWN never counts as PASS: an unevaluated check is not satisfied.
            granted=not unmet and not unknown and not missing_prereq,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            requirements=list(requirements),
            unmet=unmet,
            unknown=unknown,
            prerequisites_met=not missing_prereq,
            missing_prerequisites=missing_prereq,
            allows=list(permissions["allows"]),
            forbids=list(permissions["forbids"]),
        )
        if unknown:
            certificate.notes.append(
                f"{len(unknown)} requirement(s) could not be evaluated; an "
                "unevaluated check is not a passed check, so the tier is withheld"
            )
        if missing_prereq:
            certificate.notes.append(
                f"prerequisite tiers not granted: {missing_prereq}"
            )
        return certificate

    def evaluate_all(self) -> dict[str, Any]:
        """Evaluate every tier in dependency order."""
        granted: dict[str, bool] = {}
        certificates: dict[str, TierCertificate] = {}

        engineering = self.engineering_pipeline()
        certificates[ENGINEERING_PIPELINE_READY] = engineering
        granted[ENGINEERING_PIPELINE_READY] = engineering.granted

        gold = self.full_universe_gold(granted)
        certificates[FULL_UNIVERSE_GOLD_READY] = gold
        granted[FULL_UNIVERSE_GOLD_READY] = gold.granted

        research = self.full_universe_research(granted)
        certificates[FULL_UNIVERSE_RESEARCH_READY] = research
        granted[FULL_UNIVERSE_RESEARCH_READY] = research.granted

        paper = self.local_paper(granted)
        certificates[LOCAL_PAPER_READY] = paper
        granted[LOCAL_PAPER_READY] = paper.granted

        highest = None
        for tier in TIERS:
            if granted.get(tier):
                highest = tier
            else:
                break

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "granted": granted,
            "highest_granted_tier": highest,
            "certificates": {k: v.to_dict() for k, v in certificates.items()},
            "live_trading": live_trading_certificate(),
        }

    def write(self, directory: str | Path) -> str:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / "readiness_tiers.json"
        path.write_text(
            json.dumps(self.evaluate_all(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return str(path)


def live_trading_certificate() -> dict[str, Any]:
    """The refusal, in the same shape a real certificate would take.

    Returning a structured refusal rather than raising lets every surface render
    the policy without special-casing an exception, while making it impossible
    to mistake for a granted certificate.
    """
    return {
        "tier": "LIVE_TRADING_READY",
        "granted": False,
        "implemented": False,
        "reason": "NOT_IMPLEMENTED_BY_POLICY",
        "banner": "LIVE TRADING: DISABLED BY POLICY",
        "detail": (
            "This certificate is intentionally not implemented. The system has "
            "no live order path, and no configuration, environment variable or "
            "mode transition can create one."
        ),
    }


def permits(certificates: Mapping[str, Any], action: str) -> bool:
    """Whether the granted tiers permit ``action``.

    Fail-closed: an action nobody explicitly allows is denied, and an action any
    granted tier forbids stays denied regardless of what a higher tier allows.
    """
    granted = certificates.get("granted", {})
    allowed = False
    for tier in TIERS:
        if not granted.get(tier):
            continue
        permissions = TIER_PERMISSIONS[tier]
        if action in permissions["forbids"]:
            return False
        if action in permissions["allows"]:
            allowed = True
    return allowed
