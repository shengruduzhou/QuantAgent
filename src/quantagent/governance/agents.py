"""Agent roles: scopes, tools, veto authority and failure behaviour.

An agent here is a *capability boundary*, not a prompt. Each role declares what
it may read, what it may write, whether it can stop a decision outright, and
what happens when it fails. Those declarations are enforced by the protocol,
so "the Risk agent can block" is a property of the system rather than an
instruction a model may or may not follow.

Two authority levels:

``ADVISORY``  the agent's REJECT stops the sequence at its own step but a later
              re-run with better evidence can proceed.
``VETO``      the agent's BLOCK is final for that decision. Data Quality, Risk,
              Compliance and Governance hold it, because each guards a class of
              harm that cannot be traded off against expected return: unknown
              data semantics, uncontrolled loss, unlawful data use, and
              protected-window leakage.

Failure behaviour is declared too. An agent that errors is not "neutral": for a
veto-holding agent, silence must be treated as a block, because the alternative
is that crashing the data-quality check becomes a way to pass it.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Mapping, Sequence

ADVISORY = "ADVISORY"
VETO = "VETO"

#: What an agent's failure means. ``FAIL_CLOSED`` converts an error into a
#: BLOCK; ``FAIL_OPEN`` records the error and lets the sequence continue.
FAIL_CLOSED = "FAIL_CLOSED"
FAIL_OPEN = "FAIL_OPEN"


@dataclass(frozen=True)
class AgentRole:
    name: str
    responsibilities: tuple[str, ...]
    inputs: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    read_scope: tuple[str, ...]
    write_scope: tuple[str, ...]
    output_schema: str
    evidence_requirements: tuple[str, ...]
    authority: str = ADVISORY
    failure_behaviour: str = FAIL_CLOSED
    veto_domains: tuple[str, ...] = ()
    description: str = ""

    @property
    def can_veto(self) -> bool:
        return self.authority == VETO

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"can_veto": self.can_veto}


ORCHESTRATOR = AgentRole(
    name="orchestrator_governance",
    responsibilities=(
        "decompose missions into stage-gated decisions",
        "select the agent sequence for each decision",
        "verify artifact hashes before advancing a stage",
        "prevent protected-window leakage",
        "prevent any live-trading path from activating",
        "collect disagreements and produce the final decision record",
    ),
    inputs=("mission", "stage gates", "agent envelopes"),
    allowed_tools=("read_artifacts", "hash_artifacts", "write_audit_log"),
    read_scope=("runtime/**", "docs/**", "configs/**"),
    write_scope=("runtime/governance/**",),
    output_schema="DecisionRecord",
    evidence_requirements=(
        "every advancing stage cites the artifact hashes it verified",
    ),
    authority=VETO,
    veto_domains=("protected_window_leakage", "live_trading", "missing_evidence"),
    description=(
        "Decides nothing on its own evidence. It sequences agents, enforces "
        "gates and records outcomes; it may not invent evidence and may not "
        "overturn another agent's hard veto."
    ),
)

DATA_ACQUISITION = AgentRole(
    name="data_acquisition",
    responsibilities=(
        "run capability probes", "acquire from entitled providers",
        "resume interrupted acquisition", "respect rate limits",
        "partition raw events immutably", "record provenance and licensing",
    ),
    inputs=("provider capability matrix", "acquisition plan"),
    allowed_tools=("http_client", "vendor_sdk", "raw_event_store"),
    read_scope=("runtime/data/capabilities/**",),
    write_scope=("runtime/data/market_events/**", "runtime/data/u0/**"),
    output_schema="AcquisitionReceipt",
    evidence_requirements=("per-partition write receipts with content hashes",),
    authority=ADVISORY,
    description=(
        "Acquires data. Explicitly may NOT alter validation thresholds -- the "
        "agent that fetches data does not get to decide whether it is good."
    ),
)

DATA_QUALITY = AgentRole(
    name="data_quality_forensics",
    responsibilities=(
        "schema validation", "sequence-gap analysis",
        "tick-to-daily reconciliation", "adjustment forensics",
        "point-in-time checks", "corruption detection",
        "issue or withhold readiness certificates",
    ),
    inputs=("canonical event frames", "U0 daily panel", "integrity reports"),
    allowed_tools=("integrity_checks", "reconciliation", "readiness_certificates"),
    read_scope=("runtime/data/**",),
    write_scope=("runtime/data/**/validation/**", "runtime/governance/**"),
    output_schema="IntegrityReport",
    evidence_requirements=(
        "PASS/WARN/FAIL/NOT_RUN verdict per check",
        "NOT_RUN is never reported as a pass",
    ),
    authority=VETO,
    failure_behaviour=FAIL_CLOSED,
    veto_domains=("unknown_semantics", "failed_integrity", "unverified_reconciliation"),
    description=(
        "Holds a hard veto on datasets whose semantics are unknown. This is the "
        "agent whose silence must never be read as approval."
    ),
)

MICROSTRUCTURE = AgentRole(
    name="market_microstructure",
    responsibilities=(
        "interpret exchange events", "classify auction vs continuous phases",
        "build order-flow features", "judge book-reconstruction validity",
        "set liquidity, impact and latency assumptions",
        "downgrade conclusions when only snapshots or Level-1 exist",
    ),
    inputs=("canonical events", "fidelity decision"),
    allowed_tools=("fidelity_decider", "integrity_checks", "feature_builders"),
    read_scope=("runtime/data/market_events/**",),
    write_scope=("runtime/data/features/microstructure/**",),
    output_schema="FidelityDecision",
    evidence_requirements=(
        "declared data class per input dataset",
        "explicit downgrade list when depth is aggregated",
    ),
    authority=VETO,
    veto_domains=("overstated_fidelity",),
    description=(
        "Must downgrade rather than extrapolate. A queue-position claim on "
        "snapshot data is a veto, not a caveat."
    ),
)

STOCK_SELECTION = AgentRole(
    name="stock_selection",
    responsibilities=(
        "full-universe eligibility", "cross-sectional factors", "neutralisation",
        "liquidity and capacity screens", "candidate ranking",
        "sector and style exposure",
    ),
    inputs=("gold dataset", "eligibility masks"),
    allowed_tools=("factor_library", "neutralisation", "screens"),
    read_scope=("runtime/data/gold/**",),
    write_scope=("runtime/predictions/**", "runtime/proposals/**"),
    output_schema="CandidateBook",
    evidence_requirements=("universe definition", "screen survival counts"),
    authority=ADVISORY,
    failure_behaviour=FAIL_OPEN,
    description=(
        "May not inspect protected blind-window results while selecting "
        "features; doing so is a Governance veto, not a self-policed rule."
    ),
)

STRATEGY_RESEARCH = AgentRole(
    name="strategy_research",
    responsibilities=(
        "register hypotheses before testing", "specify the strategy",
        "state the expected economic mechanism", "declare data requirements",
        "declare invalidation criteria", "declare the parameter budget",
    ),
    inputs=("hypothesis registry", "candidate evidence"),
    allowed_tools=("hypothesis_registry",),
    read_scope=("HYPOTHESIS_REGISTRY.md", "EXPERIMENT_LEDGER.md"),
    write_scope=("HYPOTHESIS_REGISTRY.md",),
    output_schema="HypothesisRegistration",
    evidence_requirements=(
        "pre-registered invalidation criteria",
        "declared parameter budget",
    ),
    authority=ADVISORY,
    failure_behaviour=FAIL_OPEN,
    description="Cannot declare success from in-sample performance.",
)

BACKTEST = AgentRole(
    name="backtest",
    responsibilities=(
        "select the correct simulator fidelity level",
        "construct purged walk-forward splits with embargo",
        "build strict out-of-fold predictions",
        "model costs and execution", "compute robustness metrics",
        "detect leakage", "emit reproducible reports",
    ),
    inputs=("candidate book", "fidelity decision", "cost model"),
    allowed_tools=("simulator", "walk_forward", "cost_model"),
    read_scope=("runtime/data/**",),
    write_scope=("runtime/reports/**",),
    output_schema="BacktestReport",
    evidence_requirements=(
        "disclosed simulator fidelity level",
        "disclosed cost and latency assumptions",
    ),
    authority=VETO,
    veto_domains=("invalid_execution_assumptions", "leakage"),
    description="Holds a hard veto on invalid execution assumptions.",
)

RISK = AgentRole(
    name="risk",
    responsibilities=(
        "pre-trade checks", "exposure and concentration", "liquidity and capacity",
        "drawdown and turnover", "factor crowding", "scenario and stress tests",
        "data, model and operational risk", "kill switches",
    ),
    inputs=("backtest report", "portfolio state", "capacity analysis"),
    allowed_tools=("risk_gate", "kill_switch", "stress_tests"),
    read_scope=("runtime/**",),
    write_scope=("runtime/reports/risk/**",),
    output_schema="RiskAssessment",
    evidence_requirements=("stated limits and the measured value against each",),
    authority=VETO,
    failure_behaviour=FAIL_CLOSED,
    veto_domains=("limit_breach", "capacity", "drawdown", "operational_risk"),
    description="A risk rejection cannot be overridden by majority vote.",
)

EXECUTION = AgentRole(
    name="trading_execution",
    responsibilities=(
        "convert approved targets into simulated orders",
        "select TWAP/VWAP/POV/passive policies",
        "monitor acknowledgements", "reconcile simulated fills",
        "report latency and implementation shortfall",
    ),
    inputs=("approved target weights", "execution policy"),
    allowed_tools=("paper_broker", "execution_simulator"),
    read_scope=("runtime/target_weights/**",),
    write_scope=("runtime/paper/**",),
    output_schema="ExecutionReport",
    evidence_requirements=("simulated fills reconciled against intents",),
    authority=ADVISORY,
    failure_behaviour=FAIL_CLOSED,
    description=(
        "Paper only for this mission: dry-run orders, no real-account "
        "transmission, no credential handling, no broker activation."
    ),
)

CHALLENGER = AgentRole(
    name="independent_challenger",
    responsibilities=(
        "reproduce results independently", "search for leakage",
        "challenge provenance and cost assumptions",
        "test benign alternative explanations", "detect survivorship bias",
        "test adverse periods", "verify code and report agree",
    ),
    inputs=("all artifacts of the decision under review",),
    allowed_tools=("read_artifacts", "rerun_backtest"),
    read_scope=("runtime/**", "src/**"),
    write_scope=("runtime/reports/challenger/**",),
    output_schema="ChallengeReport",
    evidence_requirements=("an independently recomputed number, not a review note",),
    authority=ADVISORY,
    failure_behaviour=FAIL_CLOSED,
    description="Exists to find the benign explanation before the market does.",
)

COMPLIANCE = AgentRole(
    name="compliance_data_rights",
    responsibilities=(
        "verify data-use rights", "prevent unauthorised redistribution",
        "verify provider terms", "classify licensed artifacts",
        "keep licensed raw data out of version control",
        "record retention and deletion rules",
    ),
    inputs=("provider licence records", "artifact inventory"),
    allowed_tools=("licence_registry", "repo_scan"),
    read_scope=("runtime/**", ".gitignore"),
    write_scope=("runtime/governance/compliance/**",),
    output_schema="ComplianceAssessment",
    evidence_requirements=("a licence record per provider actually used",),
    authority=VETO,
    failure_behaviour=FAIL_CLOSED,
    veto_domains=("licensing", "redistribution", "raw_data_in_git"),
    description="Holds a hard veto on data-rights violations.",
)

ALL_ROLES: tuple[AgentRole, ...] = (
    ORCHESTRATOR, DATA_ACQUISITION, DATA_QUALITY, MICROSTRUCTURE,
    STOCK_SELECTION, STRATEGY_RESEARCH, BACKTEST, RISK, EXECUTION,
    CHALLENGER, COMPLIANCE,
)

ROLES_BY_NAME: dict[str, AgentRole] = {role.name: role for role in ALL_ROLES}

#: The approval sequence. Microstructure is conditional: it is required only
#: when the decision touches intraday, tick or Level-2 data, because forcing it
#: onto a daily-bar decision would make it a rubber stamp.
APPROVAL_SEQUENCE: tuple[str, ...] = (
    DATA_ACQUISITION.name,
    DATA_QUALITY.name,
    MICROSTRUCTURE.name,
    STOCK_SELECTION.name,
    STRATEGY_RESEARCH.name,
    BACKTEST.name,
    RISK.name,
    CHALLENGER.name,
    COMPLIANCE.name,
    ORCHESTRATOR.name,
)

#: Agents that only participate when the decision involves sub-daily data.
INTRADAY_ONLY_AGENTS: frozenset[str] = frozenset({MICROSTRUCTURE.name})

#: Agents whose approval is required on every decision regardless of scope.
MANDATORY_AGENTS: frozenset[str] = frozenset({
    DATA_QUALITY.name, RISK.name, COMPLIANCE.name, ORCHESTRATOR.name,
})


def veto_holders() -> tuple[str, ...]:
    return tuple(role.name for role in ALL_ROLES if role.can_veto)


def role_for(name: str) -> AgentRole:
    try:
        return ROLES_BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"unknown agent {name!r}; known agents: {sorted(ROLES_BY_NAME)}"
        ) from None
