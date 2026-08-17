from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from threading import RLock, Thread
import time
from typing import Any, Callable, Iterator
from uuid import uuid4

from quantagent.research.verdict import (
    CONFIGURATION_BLOCKED_EXIT_CODE,
    RESEARCH_REJECTED_EXIT_CODE,
)
from quantagent.safety.operating_mode import reject_live_intent
from services.quant_api.config import (
    PROJECT_ROOT,
    ApiSettings,
    project_relative,
    safe_project_path,
)
from services.quant_api.events import EventBroker
from services.quant_api.services.connections import CREDENTIAL_VARIABLES
from services.quant_api.services.job_diagnostics import JobFailure, diagnose
from services.quant_api.services.process_supervisor import (
    process_is_alive,
    process_start_ticks,
    sample_process,
    signal_tree,
    terminate_tree,
)


# `rejected` is terminal but is not a failure: the run completed and a
# pre-registered gate refused the candidate.
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled", "rejected", "blocked"})
#: Verdicts a run can print on stdout to declare its own terminal conclusion.
TERMINAL_VERDICTS = frozenset({"rejected", "blocked"})
# Retry is for recovering from an interruption, not for repeating a finished
# experiment: a completed run's evidence must not be silently overwritten.
RETRYABLE_STATUSES = frozenset({"failed", "cancelled"})
ACTIVE_STATUSES = frozenset({"queued", "starting", "running", "paused", "cancelling"})
POLL_SECONDS = 1.0
RESOURCE_SAMPLE_SECONDS = 5.0

# The supervisor ships with this code, so its location follows the package
# rather than whatever project root the settings point at.
JOB_SUPERVISOR = PROJECT_ROOT / "scripts" / "job_supervisor.py"

STAGE_LABELS: dict[str, str] = {
    "contract": "研究契约校验",
    "dataset": "PIT 数据集构建",
    "factor_screening": "因子筛选",
    "training": "滚动样本外训练",
    "prediction": "样本外预测",
    "portfolio": "组合与目标权重",
    "portfolio_selection": "组合候选选择",
    "backtest": "A 股回测",
    "risk": "风控与验收闸门",
    "evidence": "证据归档",
    "verdict": "研究结论",
}


@dataclass
class JobRecord:
    id: str
    type: str
    status: str
    commandId: str
    createdAt: str
    startedAt: str | None = None
    finishedAt: str | None = None
    progress: float | None = None
    message: str | None = None
    outputPaths: list[str] = field(default_factory=list)
    error: str | None = None
    logPath: str | None = None
    ownedOutputPaths: list[str] = field(default_factory=list)
    # --- reproducibility -----------------------------------------------------
    # Without the parameters a finished job cannot be re-run, compared, or even
    # explained; "what did this run actually do" required reading the log's
    # first line.
    parameters: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    retryOf: str | None = None
    attempt: int = 1
    # --- observability -------------------------------------------------------
    stage: str | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)
    lastOutputAt: str | None = None
    resources: dict[str, Any] | None = None
    exitCode: int | None = None
    exitStatusObserved: bool = False
    failure: dict[str, Any] | None = None
    verdict: dict[str, Any] | None = None
    # --- process identity (survives an API restart) --------------------------
    pid: int | None = None
    workerPid: int | None = None
    processStartTicks: int | None = None
    workerStartTicks: int | None = None
    statusPath: str | None = None
    adopted: bool = False


COMMANDS: dict[str, dict[str, Any]] = {
    "fetch-tickflow-daily": {
        "type": "data",
        "entrypoint": "scripts/fetch_tickflow_daily_klines.py",
        "required": {"start_date", "end_date", "output", "allow_network"},
        "required_any": (("symbols", "symbols_file"),),
        "allowed": {"symbols", "symbols_file", "start_date", "end_date", "batch_size", "output", "allow_network"},
        "path_inputs": {"symbols_file"},
        "path_outputs": {"output"},
        "control": {"allow_network"},
        "credential_providers": {"tickflow"},
    },
    "fetch-tickflow-minute": {
        "type": "data",
        "entrypoint": "scripts/fetch_tickflow_minute_history.py",
        "required": {"start", "end", "allow_network"},
        "required_any": (("symbols", "symbols_file", "holdings_csv"),),
        "allowed": {"symbols", "symbols_file", "holdings_csv", "start", "end", "sleep", "limit", "allow_network"},
        "path_inputs": {"symbols_file", "holdings_csv"},
        "path_outputs": set(),
        "fixed_outputs": ("runtime/data/v7/silver/minute_bars",),
        "control": {"allow_network"},
        "credential_providers": {"tickflow"},
    },
    "record-tickflow-depth": {
        "type": "data",
        "entrypoint": "scripts/collect_tickflow_depth.py",
        "required": {"allow_network"},
        "required_any": (("symbols", "symbols_file", "book_csv"),),
        "allowed": {"symbols", "symbols_file", "book_csv", "loop_seconds", "max_iterations", "sleep", "allow_network"},
        "path_inputs": {"symbols_file", "book_csv"},
        "path_outputs": set(),
        "fixed_outputs": ("runtime/data/v7/silver/depth_snapshots",),
        "control": {"allow_network"},
        "credential_providers": {"tickflow"},
    },
    "record-tickflow-quotes": {
        "type": "data",
        "entrypoint": "scripts/record_tickflow_quotes.py",
        "required": {"allow_network"},
        "required_any": (("symbols", "symbols_file"),),
        "allowed": {"symbols", "symbols_file", "loop_seconds", "max_iterations", "allow_network"},
        "path_inputs": {"symbols_file"},
        "path_outputs": set(),
        "fixed_outputs": ("runtime/data/v7/silver/tick_snapshots",),
        "control": {"allow_network"},
        "credential_providers": {"tickflow"},
    },
    "data-manager-transfer": {
        "type": "data",
        "entrypoint": "scripts/data_manager_transfer.py",
        "required": {"operation", "source", "output"},
        "allowed": {"operation", "source", "output", "date_column", "symbol_column", "start_date", "end_date", "symbols"},
        "path_inputs": {"source"},
        "path_outputs": {"output"},
        "control": set(),
        "choices": {"operation": {"import", "export"}},
    },
    "run-strict-a-share-backtest-v8": {
        "type": "backtest",
        "required": {"target_weights_path", "market_panel_path", "output_dir"},
        "allowed": {
            "target_weights_path", "market_panel_path", "sector_map_path",
            "factor_weights_path", "output_dir", "slippage_bps", "initial_cash",
        },
        "path_inputs": {"target_weights_path", "market_panel_path", "sector_map_path", "factor_weights_path"},
        "path_outputs": {"output_dir"},
    },
    "train-v8-deep": {
        "type": "train",
        "required": {"dataset_path", "silver_panel_path", "output_dir"},
        "allowed": {
            "horizon_class", "dataset_path", "silver_panel_path", "symbols", "symbols_file",
            "train_start", "train_end", "test_end", "embargo_days", "top_k", "max_epochs",
            "batch_size", "d_token", "n_blocks", "n_heads", "dates_per_step",
            "train_micro_batch", "cross_sectional_norm", "label_norm", "feature_policy",
            "attention_dropout", "ffn_dropout", "weight_decay", "early_stopping_patience",
            "learning_rate", "regime_filter", "regime_min_rows", "require_gpu", "output_dir",
        },
        "path_inputs": {"dataset_path", "silver_panel_path", "symbols_file"},
        "path_outputs": {"output_dir"},
        "dual_boolean_options": {"require_gpu", "label_norm"},
    },
    "synthesize-factors-v7": {
        "type": "factor-discovery",
        "required": {"market_panel_path", "output_dir"},
        "allowed": {
            "market_panel_path", "labels_path", "output_dir", "rd_agent",
            "label_column", "rounds", "factors_per_round", "population",
            "generations", "top_k", "max_depth", "validation_fraction",
            "min_validation_rank_ic", "fitness_sample_dates",
            "fitness_sample_symbols", "seed", "warm_start_fraction",
            "icir_weight", "reference_columns", "max_reference_correlation",
            "max_sota_correlation", "use_llm", "allow_network", "llm_model",
            "llm_start_round", "llm_candidates_per_round",
            "rag_escalation_round", "llm_timeout_seconds", "memory_path",
            "train_end", "exclude_st", "min_validation_icir",
        },
        "path_inputs": {"market_panel_path", "labels_path", "memory_path"},
        "path_outputs": {"output_dir"},
        "control": {"allow_network"},
        "conditional_controls": {"allow_network": "use_llm"},
        "credential_providers": {"openai"},
    },
    "evaluate-factor-library-v7": {
        "type": "factor-evaluation",
        "entrypoint": "scripts/evaluate_factor_library_v7.py",
        "required": {"market_panel_path", "labels_path", "output_dir"},
        "allowed": {
            "market_panel_path", "labels_path", "output_dir", "factor_library",
            "horizon_days", "calibration_days", "holdout_days",
            "min_finite_ratio", "min_abs_rank_ic", "min_abs_rank_icir",
            "min_abs_monotonicity", "max_pairwise_correlation",
        },
        "path_inputs": {"market_panel_path", "labels_path"},
        "path_outputs": {"output_dir"},
        "choices": {
            "factor_library": {
                "all_reviewed", "basic", "alpha101", "alpha181", "cicc_ashare80",
            },
        },
    },
    "research-intraday-t-trading": {
        # A-share has no true T+0. This command's only honest framing is the
        # sellable-inventory one: intraday round trips are bounded by yesterday's
        # closing position, which is why `holdings_csv` is required rather than
        # optional. Removing it would silently manufacture T+0 capability.
        "type": "t-plus-one-research",
        "entrypoint": "scripts/intraday_dot_ev_backtest.py",
        "required": {"minute_dir", "holdings_csv", "market_panel", "output_dir"},
        "allowed": {
            "minute_dir", "holdings_csv", "market_panel", "output_dir",
            "start", "end", "train_end", "validation_end", "order_notional_yuan",
            "horizon_minutes", "backend", "edge_cost_multiple",
            "min_round_trips_enable", "maker_only", "slippage_bps", "spread_bps",
            "commission_rate", "max_symbols", "cache_table",
        },
        "path_inputs": {"minute_dir", "holdings_csv", "market_panel"},
        "path_outputs": {"output_dir", "cache_table"},
        "control": set(),
        "choices": {"backend": {"lightgbm", "xgboost", "catboost", "sklearn"}},
    },
    "search-factor-fusion": {
        "type": "fusion-search",
        "required": {"factor_panel_path", "forward_returns_path", "factor_names", "output_dir"},
        "allowed": {
            "factor_panel_path", "forward_returns_path", "factor_names", "output_dir",
            "forward_column", "horizon_days", "top_k", "n_folds", "embargo_days",
            "min_train_days", "min_test_days", "transaction_cost_bps",
            "include_genetic", "random_controls", "single_factor_baselines", "seed",
            "benchmark_path", "benchmark_symbol", "preference_excess_return",
            "preference_annual_return", "preference_drawdown_control",
            "preference_robustness",
        },
        "path_inputs": {"factor_panel_path", "forward_returns_path", "benchmark_path"},
        "path_outputs": {"output_dir"},
        # `n_trials` is intentionally absent: it is derived from the enumerated
        # search space so an operator cannot deflate their own Sharpe ratio.
        "dual_boolean_options": {"include_genetic"},
    },
    # Every scheme `search-factor-fusion` enumerates is a weight vector over
    # rank-centred factors, i.e. `score = Σ wᵢ · xᵢ`. It cannot express an
    # interaction, so "best fusion" from that command is the best *additive*
    # fusion. This command is the governed way to ask whether letting factors
    # interact adds anything, and without it on the allowlist the workstation
    # had no path to the question at all.
    "audit-nonlinear-factors": {
        "type": "research",
        "required": {"panel_path", "factor_names", "output_dir"},
        "allowed": {
            "panel_path", "factor_names", "label_column", "horizon_days", "top_k",
            "cost_bps", "n_folds", "holdout_folds", "valid_size_days",
            "min_train_days", "embargo_days", "output_dir", "enforce_promotion_gates",
        },
        "path_inputs": {"panel_path"},
        "path_outputs": {"output_dir"},
        "dual_boolean_options": {"enforce_promotion_gates"},
    },
    "predict-alpha-v7": {
        "type": "infer",
        "required": {"model_dir", "feature_dataset", "output"},
        "allowed": {"model_dir", "feature_dataset", "output", "primary_horizon"},
        "path_inputs": {"model_dir", "feature_dataset"},
        "path_outputs": {"output"},
    },
    "build-akshare-market-panel-v7": {
        "type": "data",
        "required": {"start_date", "end_date", "output", "allow_network"},
        "required_any": (("symbols", "symbols_file"),),
        "allowed": {
            "symbols", "symbols_file", "start_date", "end_date", "output_root", "output",
            "allow_network", "adjust", "provider_uri_for_range", "as_of_date",
        },
        "path_inputs": {"symbols_file", "provider_uri_for_range"},
        "path_outputs": {"output_root", "output"},
    },
    "build-market-panel-v7": {
        "type": "data",
        "required": {"provider_uri", "start_date", "end_date", "output_root"},
        "required_any": (("symbols", "symbols_file", "universe"),),
        "allowed": {
            "provider_uri", "start_date", "end_date", "output_root", "symbols",
            "symbols_file", "universe", "region", "require_optional_flags",
        },
        "path_inputs": {"provider_uri", "symbols_file"},
        "path_outputs": {"output_root"},
    },
    "build-labels-v7": {
        "type": "data",
        "required": {"market_panel_path", "output_path", "horizons"},
        "allowed": {"market_panel_path", "output_path", "horizons"},
        "path_inputs": {"market_panel_path"},
        "path_outputs": {"output_path"},
        "option_aliases": {
            "market_panel_path": "market-panel",
            "output_path": "output",
        },
    },
    "build-fundamentals-v7": {
        "type": "data",
        "required": {"start_date", "end_date", "provider", "fundamentals_root", "allow_network"},
        "required_any": (("symbols", "symbols_file"),),
        "allowed": {
            "symbols", "symbols_file", "start_date", "end_date", "provider",
            "fundamentals_root", "allow_network", "token_env",
        },
        "path_inputs": {"symbols_file"},
        "path_outputs": {"fundamentals_root"},
        "credential_providers": {"tushare"},
    },
    "run-full-real-training-v7": {
        "type": "strategy-pipeline",
        "required": {"market_panel_path", "labels_path", "output_dir"},
        "allowed": {
            # A pilot scope makes the difference between an hours-long
            # all-or-nothing run and a run an operator can validate first.
            "symbols", "symbols_file",
            "market_panel_path", "labels_path", "output_dir", "sector_map_path",
            "fundamentals_root", "valuation_path", "disclosures_path",
            "training_dataset_path", "synthesized_factors_path", "factor_library",
            "model", "horizons", "primary_horizon", "split_mode", "n_splits",
            "horizon_blend_method",
            "require_gpu", "top_k", "max_weight_per_name", "max_sector_weight",
            "max_turnover", "objective", "weighting", "initial_cash",
            "benchmark_symbol", "acceptance_max_drawdown",
            "acceptance_min_sharpe", "top_k_candidates",
            "stock_selection_modes", "fundamental_selection_mode",
            "fundamental_selection_threshold", "fundamental_blend_weight",
            "fundamental_threshold_candidates", "fundamental_blend_candidates",
            "selection_max_candidates", "selection_min_oos_days",
            "selection_min_holdout_days", "max_pbo",
            "min_dsr_probability", "max_spa_pvalue",
            "factor_screening_mode", "objective_excess_weight",
            "objective_annual_weight", "objective_drawdown_weight",
            "do_t_mode", "minute_panel_path",
        },
        "path_inputs": {
            "market_panel_path", "labels_path", "sector_map_path",
            "training_dataset_path", "synthesized_factors_path",
            "fundamentals_root", "valuation_path", "disclosures_path",
            "minute_panel_path", "symbols_file",
        },
        "path_outputs": {"output_dir"},
        "option_aliases": {
            "market_panel_path": "market-panel",
            "labels_path": "labels",
            "sector_map_path": "sector-map",
            "training_dataset_path": "training-dataset",
            "synthesized_factors_path": "synthesized-factors",
            "valuation_path": "valuation",
            "disclosures_path": "disclosures",
            "max_weight_per_name": "max-weight",
            "max_sector_weight": "max-sector",
        },
        "choices": {
            "factor_library": {"all_reviewed", "basic", "alpha101", "alpha181", "cicc_ashare80"},
            "model": {"ridge", "ft_transformer"},
            "split_mode": {"rolling", "expanding"},
            "objective": {"max_expected_alpha", "mean_variance", "min_variance"},
            "weighting": {"equal", "rank", "softmax"},
            "factor_screening_mode": {"off", "evaluate_only", "pretrain"},
            "fundamental_selection_mode": {"auto", "fixed", "off"},
            "do_t_mode": {"off", "intraday", "daily_swing", "both"},
            "horizon_blend_method": {
                "adaptive_oos",
                "balanced",
                "short_tactical",
                "long_fundamental",
                "primary_only",
            },
        },
    },
    # ---- H-031 governed operational / full-universe data commands -----------
    # All are parameterless or take a single bounded control; none exposes a
    # free-form shell field. Outputs are fixed Runtime paths; none reads or
    # reports candidate performance.
    "validate-shadow-days": {
        "type": "governance",
        "entrypoint": "scripts/shadow_day_registry.py",
        "required": set(),
        "allowed": {"quiet"},
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": (
            "runtime/paper/fresh_blind/shadow_day_registry.json",
            "runtime/paper/fresh_blind/shadow_accumulating_status.json",
        ),
        "control": set(),
    },
    "certify-s4-batch-replay": {
        "type": "governance",
        "entrypoint": "scripts/s4_batch_replay.py",
        "required": set(),
        "allowed": {"cutoff"},
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": ("runtime/reports/h030/s4_readiness_certificate.json",),
        "control": set(),
    },
    "build-u0-security-master": {
        "type": "data",
        "entrypoint": "scripts/u0_build_security_master.py",
        "required": set(),
        "allowed": set(),
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": (
            "runtime/data/u0/historical_security_master.parquet",
            "runtime/data/u0/pit_field_availability.json",
        ),
        "control": set(),
    },
    "report-u0-provider-coverage": {
        "type": "data",
        "entrypoint": "scripts/u0_provider_coverage.py",
        "required": set(),
        "allowed": set(),
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": (
            "runtime/data/u0/provider_coverage_matrix.parquet",
            "runtime/data/u0/provider_coverage_matrix.csv",
        ),
        "control": set(),
    },
    "assemble-u0-full-universe": {
        "type": "data",
        "entrypoint": "scripts/u0_full_universe_backfill.py",
        "fixed_args": ("assemble",),
        "required": set(),
        "allowed": set(),
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": ("runtime/data/v7/full_universe/full_universe_market_panel.parquet",),
        "control": set(),
    },
    "audit-u0-full-universe": {
        "type": "data",
        "entrypoint": "scripts/u0_audit.py",
        "required": set(),
        "allowed": set(),
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": ("runtime/data/u0/full_universe_readiness_certificate.json",),
        "control": set(),
    },
    "backfill-u0-market-panel": {
        "type": "data",
        "entrypoint": "scripts/u0_full_universe_backfill.py",
        "fixed_args": ("fetch",),
        "required": {"allow_network"},
        "allowed": {"max_minutes", "allow_network", "priority_boards"},
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": ("runtime/data/v7/full_universe/_staging",),
        "control": {"allow_network"},
    },
    "probe-u0-star-bse": {
        "type": "data",
        "entrypoint": "scripts/u0_star_bse_probe.py",
        "required": {"allow_network"},
        "allowed": {"allow_network", "max_symbols"},
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": ("runtime/data/u0/star_bse_probe_report.json",),
        "control": {"allow_network"},
    },
    "benchmark-tickflow-capability": {
        "type": "data",
        "entrypoint": "scripts/tickflow_capability_benchmark.py",
        "required": {"allow_network"},
        "allowed": {"allow_network"},
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": ("runtime/reports/h032b/tickflow_capability_benchmark.json",),
        "control": {"allow_network"},
    },
    "audit-bse-identity": {
        "type": "data",
        "entrypoint": "scripts/u0_bse_identity.py",
        "required": {"allow_network"},
        "allowed": {"allow_network"},
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": ("runtime/data/u0/bse_code_mapping.parquet",
                          "runtime/data/u0/bse_identity_audit.json"),
        "control": {"allow_network"},
    },
    "audit-u0-pit-readiness": {
        "type": "data",
        "entrypoint": "scripts/u0_pit_readiness.py",
        "required": set(),
        "allowed": set(),
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": ("runtime/data/u0/u0_strict_pit_certificate.json",
                          "runtime/data/u0/pit_source_audit.json"),
        "control": set(),
    },
    # --- tick / Level-2 / MT5 / QMT surfaces -------------------------------
    # All read-only or acquisition-only. None of these can reach an order path:
    # the market-data provider and the QMT execution gateway are separate
    # modules with no shared state, enforced by an import-graph test.
    "probe-mt5-capability": {
        "type": "data",
        "entrypoint": "scripts/probe_mt5_capability.py",
        "required": set(),
        "allowed": {"output", "dom_reads"},
        "path_inputs": set(),
        "path_outputs": {"output"},
        "fixed_outputs": ("runtime/data/capabilities/mt5/capability_matrix.json",
                          "runtime/data/capabilities/mt5/terminal.json"),
        "control": set(),
    },
    # Read-only. On a non-Windows host this writes the full capability
    # catalogue marked PLATFORM_UNAVAILABLE rather than an empty file, so the
    # scope of what is unmeasured stays visible.
    "probe-qmt-entitlements": {
        "type": "data",
        "entrypoint": "scripts/probe_qmt_entitlements.py",
        "required": set(),
        "allowed": {"output", "earliest", "latest"},
        "path_inputs": set(),
        "path_outputs": {"output"},
        "fixed_outputs": ("runtime/data/capabilities/qmt/entitlement_matrix.json",
                          "runtime/data/capabilities/qmt/environment.json",
                          "runtime/data/capabilities/qmt/st_probe.json",
                          "runtime/data/capabilities/qmt/level2_probe.json"),
        "control": set(),
    },
    "probe-xtdata-capability": {
        "type": "data",
        "entrypoint": "scripts/probe_xtdata_capability.py",
        "required": set(),
        "allowed": {"symbols", "output"},
        "path_inputs": set(),
        "path_outputs": {"output"},
        "fixed_outputs": ("runtime/data/capabilities/qmt/capability_matrix.json",
                          "runtime/data/capabilities/qmt/runtime.json"),
        "control": set(),
    },
    "probe-tick-l2-source-matrix": {
        "type": "data",
        "entrypoint": "scripts/probe_tick_l2_source_matrix.py",
        "required": set(),
        "allowed": {"symbol", "trade_date", "output", "save_events"},
        "path_inputs": set(),
        "path_outputs": {"output"},
        "fixed_outputs": (
            "runtime/data/capabilities/tick_l2/tick_l2_capability_matrix.json",
        ),
        "control": set(),
    },
    "acquire-ashare-ticks": {
        "type": "data",
        "entrypoint": "scripts/acquire_ashare_ticks.py",
        "required": {"trade_date"},
        "allowed": {"trade_date", "cohort", "symbols", "output", "report", "max_pages"},
        "path_inputs": set(),
        "path_outputs": {"output", "report"},
        "fixed_outputs": ("runtime/data/market_events",),
        "control": set(),
        "choices": {"cohort": {"board-spread"}},
    },
    "export-mt5-custom-symbols": {
        "type": "data",
        "entrypoint": "scripts/export_mt5_custom_symbols.py",
        "required": {"trade_date"},
        "allowed": {"trade_date", "journal", "symbols", "output", "bar_history_days"},
        "path_inputs": {"journal"},
        "path_outputs": {"output"},
        "fixed_outputs": ("runtime/data/mt5_custom_symbols/export_index.json",),
        "control": set(),
    },
    "build-full-universe-gold": {
        "type": "data",
        "entrypoint": "scripts/build_full_universe_gold.py",
        "required": set(),
        "allowed": {"output", "max_symbols", "start_date", "adjustment",
                    "horizons", "seasoning_days", "journal"},
        "path_inputs": {"journal"},
        "path_outputs": {"output"},
        "fixed_outputs": (
            "runtime/data/gold/full_universe/training_slice_certificate.json",
        ),
        "control": set(),
        "choices": {"adjustment": {"none", "qfq", "hfq"}},
    },
    "report-u0-bar-readiness": {
        "type": "data",
        "entrypoint": "scripts/u0_bar_readiness.py",
        "required": set(),
        "allowed": set(),
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": ("runtime/data/u0/u0_bar_readiness_certificate.json",),
        "control": set(),
    },
    "source-u0-pit-metadata": {
        "type": "data",
        "entrypoint": "scripts/u0_pit_metadata.py",
        "required": {"allow_network"},
        "allowed": {"allow_network"},
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": ("runtime/data/u0/pit/pit_metadata_manifest.json",),
        "control": {"allow_network"},
    },
    "audit-tickflow-entitlement": {
        "type": "data",
        "entrypoint": "scripts/u0_tickflow_entitlement.py",
        "required": {"allow_network"},
        "allowed": {"allow_network"},
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": ("runtime/reports/h032c/tickflow_entitlement_audit.json",),
        "control": {"allow_network"},
    },
    "report-u0-reconciliation": {
        "type": "data",
        "entrypoint": "scripts/u0_reconcile_universe.py",
        "required": set(),
        "allowed": set(),
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": ("runtime/data/u0/universe_reconciliation.json",),
        "control": set(),
    },
    # --- A-share data foundation (canonical U0 acquisition pipeline) ---------
    "probe-ashare-capabilities": {
        "type": "data",
        "entrypoint": "scripts/ashare_capability_probe.py",
        "required": {"allow_network"},
        "allowed": {"allow_network", "skip_tickflow"},
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": (
            "runtime/data/u0/capability/provider_capability_matrix.json",
            "runtime/data/u0/capability/provider_capability_matrix.csv",
            "runtime/data/u0/capability/provider_capability_report.md",
        ),
        "control": {"allow_network"},
    },
    "build-u0-live-security-master": {
        "type": "data",
        "entrypoint": "scripts/u0_security_master.py",
        "required": {"allow_network"},
        "allowed": {"allow_network"},
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": (
            "runtime/data/u0/security_master.parquet",
            "runtime/data/u0/security_master_manifest.json",
        ),
        "control": {"allow_network"},
    },
    "acquire-u0-daily-bars": {
        "type": "data",
        "entrypoint": "scripts/u0_acquire_bars.py",
        "required": {"allow_network"},
        "allowed": {"allow_network", "providers", "staging_name", "max_minutes", "limit",
                    "boards", "symbols", "status", "shard", "order", "skip_if_in", "refetch"},
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": ("runtime/data/u0/bars",),
        "control": {"allow_network"},
    },
    "acquire-u0-intraday-bars": {
        "type": "data",
        "entrypoint": "scripts/u0_acquire_intraday.py",
        "required": {"allow_network"},
        "allowed": {"allow_network", "per_board", "frequency", "count", "providers",
                    "max_minutes"},
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": (
            "runtime/data/u0/intraday/minute_bars.parquet",
            "runtime/data/u0/intraday/intraday_manifest.json",
        ),
        "control": {"allow_network"},
    },
    "build-u0-pit-intervals": {
        "type": "data",
        "entrypoint": "scripts/u0_pit_intervals.py",
        "required": {"cmd"},
        "allowed": {"cmd", "allow_network", "max_minutes", "limit", "start"},
        "positional": ("cmd",),
        "choices": {"cmd": ("calendar", "factors", "dividends", "suspension", "st")},
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": ("runtime/data/u0/pit",),
        "control": {"allow_network"},
    },
    "assemble-u0-raw-panel": {
        "type": "data",
        "entrypoint": "scripts/u0_assemble_panel.py",
        "required": set(),
        "allowed": {"flush_every"},
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": (
            "runtime/data/u0/panel/daily_bars_raw.parquet",
            "runtime/data/u0/panel/panel_manifest.json",
            "runtime/data/u0/panel/coverage_matrix.parquet",
            "runtime/data/u0/panel/session_gaps.parquet",
        ),
        "control": set(),
    },
    "validate-u0-data": {
        "type": "data",
        "entrypoint": "scripts/u0_validate.py",
        "required": set(),
        "allowed": {"allow_network"},
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": (
            "runtime/data/u0/validation/validation_report.json",
            "runtime/data/u0/validation/validation_report.md",
        ),
        "control": {"allow_network"},
    },
    "audit-u0-adjustment-forensics": {
        "type": "data",
        "entrypoint": "scripts/u0_adjustment_forensics.py",
        "required": set(),
        "allowed": set(),
        "path_inputs": set(),
        "path_outputs": set(),
        "fixed_outputs": ("runtime/data/u0/validation/adjustment_forensics.json",),
        "control": set(),
    },
}


class JobManager:
    def __init__(
        self,
        settings: ApiSettings,
        events: EventBroker | None = None,
        on_success: Callable[[], None] | None = None,
        connection_environment: Callable[[set[str] | tuple[str, ...]], dict[str, str]] | None = None,
    ) -> None:
        self.settings = settings
        self.events = events
        self.on_success = on_success
        self.connection_environment = connection_environment
        self.state_path = settings.jobs_root / "jobs.json"
        self._lock = RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._cancel_requested: set[str] = set()
        self._purged: set[str] = set()
        self._load()

    def submit(
        self,
        job_type: str,
        command_id: str,
        parameters: dict[str, Any],
        *,
        labels: dict[str, str] | None = None,
        retry_of: str | None = None,
        attempt: int = 1,
    ) -> dict[str, Any]:
        spec, normalized = self._validate(job_type, command_id, parameters)
        job_id = f"job_{uuid4().hex[:16]}"
        log_path = self.settings.jobs_root / f"{job_id}.log"
        declared_outputs = [
            Path(str(normalized[key]))
            for key in spec["path_outputs"]
            if normalized.get(key) not in (None, "")
        ]
        record = JobRecord(
            id=job_id,
            type=job_type,
            status="queued",
            commandId=command_id,
            createdAt=_now(),
            message="queued",
            logPath=project_relative(self.settings, log_path),
            outputPaths=[
                project_relative(self.settings, path) for path in declared_outputs
            ],
            ownedOutputPaths=[
                project_relative(self.settings, path)
                for path in declared_outputs
                if not path.exists()
            ],
            parameters=dict(normalized),
            labels=dict(labels or {}),
            retryOf=retry_of,
            attempt=attempt,
            statusPath=project_relative(self.settings, self.settings.jobs_root / f"{job_id}.status.json"),
        )
        with self._lock:
            self._jobs[job_id] = record
            self._persist()
        self._emit(record)
        Thread(target=self._run, args=(job_id, command_id, normalized, spec, log_path), daemon=True).start()
        return self._public(record)

    def retry(self, job_id: str) -> dict[str, Any]:
        """Re-submit a finished job with exactly the parameters it ran with.

        Re-running is the normal response to a transient failure, and doing it
        by hand means re-typing a parameter set nobody kept. The new job records
        which attempt it is and which job it descends from, so a chain of
        retries stays legible instead of looking like unrelated runs.
        """
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            if record.status not in TERMINAL_STATUSES:
                raise ValueError(
                    f"job {job_id} is {record.status}; only a finished job can be retried"
                )
            if record.status not in RETRYABLE_STATUSES:
                # A retry replays the same parameters into the same output
                # directory. For a run that already produced a conclusion that
                # would overwrite its evidence and reproduce the same answer;
                # a new experiment belongs in a new run directory.
                raise ValueError(
                    f"job {job_id} finished as {record.status}; retrying would overwrite its "
                    "evidence with an identical run. Launch a new run from the strategy instead."
                )
            if not record.parameters:
                raise ValueError(
                    f"job {job_id} predates parameter capture and cannot be replayed; "
                    "resubmit it from the page that created it"
                )
            source = record
            root = source.retryOf or source.id
            attempt = source.attempt + 1
            parameters = dict(source.parameters)
            labels = dict(source.labels)
            job_type, command_id = source.type, source.commandId
        return self.submit(
            job_type,
            command_id,
            parameters,
            labels=labels,
            retry_of=root,
            attempt=attempt,
        )

    def validate(self, job_type: str, command_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
        spec, _ = self._validate(job_type, command_id, parameters)
        outputs = [
            str(parameters[key])
            for key in spec["path_outputs"]
            if parameters.get(key) not in (None, "")
        ]
        outputs.extend(spec.get("fixed_outputs", ()))
        warnings = ["GPU availability is checked by the training process"] if parameters.get("require_gpu") else []
        if command_id == "synthesize-factors-v7":
            warnings.append("Factor discovery writes research candidates only; registration and training remain separate human-gated steps")
            if parameters.get("use_llm"):
                warnings.append("LLM network execution is armed for this research job")
        if command_id == "evaluate-factor-library-v7":
            warnings.extend([
                "Every materialised factor is evaluated from PIT-labelled data; the registry is not mutated",
                "Selection is frozen on an early calibration window; later dates remain an untouched holdout",
            ])
        if command_id == "run-full-real-training-v7":
            warnings.extend([
                "End-to-end strategy pipeline remains research/paper only; no live order route is enabled",
                "Acceptance objectives are evaluated from persisted OOS and backtest artifacts, never promised in advance",
            ])
        return {
            "valid": True,
            "type": job_type,
            "commandId": command_id,
            "entrypoint": spec.get("entrypoint") or "quantagent.cli",
            "outputPaths": outputs,
            "warnings": warnings,
        }

    def _validate(
        self,
        job_type: str,
        command_id: str,
        parameters: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Live-order intent is refused before anything else happens: before the
        # command is resolved, before parameters are normalised, and before a
        # JobRecord exists. Checking later would mean an intent to trade real
        # money had already been routed or queued by the time it was refused.
        reject_live_intent(
            {"job_type": job_type, "command_id": command_id, "parameters": parameters},
            where="job submission",
        )

        spec = COMMANDS.get(command_id)
        if spec is None or spec["type"] != job_type:
            raise ValueError(f"command {command_id!r} is not allowed for {job_type}")
        if command_id == "train-v8-deep" and parameters.get("require_gpu") is not True:
            raise ValueError(
                "train-v8-deep requires require_gpu=true; CPU fallback is disabled "
                "for governed workstation training"
            )
        unknown = set(parameters) - set(spec["allowed"])
        if unknown:
            raise ValueError(f"unsupported parameters: {sorted(unknown)}")
        missing = {
            key for key in spec.get("required", set())
            if parameters.get(key) in (None, "", [])
        }
        if missing:
            raise ValueError(f"missing required parameters: {sorted(missing)}")
        for key, choices in spec.get("choices", {}).items():
            value = parameters.get(key)
            if value is not None and value not in choices:
                raise ValueError(f"{key} must be one of {sorted(choices)}")
        conditional_controls = spec.get("conditional_controls", {})
        for control_key in spec.get("control", set()):
            trigger_key = conditional_controls.get(control_key)
            required = trigger_key is None or parameters.get(trigger_key) is True
            if required and parameters.get(control_key) is not True:
                raise ValueError(f"{control_key} must be explicitly confirmed")
        for group in spec.get("required_any", ()):
            if not any(parameters.get(key) not in (None, "", []) for key in group):
                raise ValueError(f"one of {sorted(group)} is required")
        normalized = self._normalize_parameters(spec, parameters)
        return spec, normalized

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                self._public(record)
                for record in sorted(self._jobs.values(), key=lambda item: item.createdAt, reverse=True)
            ]

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return self._public(record) if record else None


    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            if record.status in TERMINAL_STATUSES:
                raise ValueError(f"job {job_id} already finished with status {record.status}")
            self._cancel_requested.add(job_id)
            was_paused = record.status == "paused"
            target = record.workerPid or record.pid
            if target is None:
                record.status = "cancelled"
                record.message = "cancelled before process start"
                record.finishedAt = _now()
            else:
                record.status = "cancelling"
                record.message = "termination requested"
            self._persist()
            public = self._public(record)
        self._emit(record)
        if target is not None:
            terminate_tree(target, was_paused=was_paused)
        return public

    def pause(self, job_id: str) -> dict[str, Any]:
        """Suspend a running job's process without losing its progress.

        Uses SIGSTOP rather than termination so a long search can yield the
        machine to a higher-priority run and then continue from where it was.
        The process keeps its memory, so a paused job still holds RAM and GPU
        allocations — this is a scheduling control, not a resource release.
        """
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            if record.status != "running":
                raise ValueError(f"job {job_id} is {record.status}, only a running job can be paused")
            target = record.workerPid or record.pid
            if target is None:
                raise ValueError(f"job {job_id} has no live process to pause")
            try:
                signal_tree(target, signal.SIGSTOP)
            except OSError as exc:
                raise ValueError(f"could not pause job {job_id}: {exc}") from exc
            record.status = "paused"
            record.message = "paused by operator; process suspended, memory retained"
            self._persist()
            public = self._public(record)
        self._emit(record)
        return public

    def resume(self, job_id: str) -> dict[str, Any]:
        """Continue a paused job from exactly where it was suspended."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            if record.status != "paused":
                raise ValueError(f"job {job_id} is {record.status}, only a paused job can be resumed")
            target = record.workerPid or record.pid
            if target is None:
                raise ValueError(f"job {job_id} has no live process to resume")
            try:
                signal_tree(target, signal.SIGCONT)
            except OSError as exc:
                raise ValueError(f"could not resume job {job_id}: {exc}") from exc
            record.status = "running"
            record.message = "resumed by operator"
            self._persist()
            public = self._public(record)
        self._emit(record)
        return public

    def purge(self, job_id: str, *, delete_outputs: bool = False) -> dict[str, Any]:
        """Stop a job and remove its operational trace and owned outputs."""

        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            process = self._processes.pop(job_id, None)
            was_paused = record.status == "paused"
            target = record.workerPid or record.pid
            owned_outputs = list(record.ownedOutputPaths)
            self._cancel_requested.add(job_id)
            self._purged.add(job_id)
            self._jobs.pop(job_id, None)
            self._persist()
        if target is not None:
            # Same trap as cancel: a suspended process ignores SIGTERM until it
            # is continued, and purge would then block waiting for it.
            terminate_tree(target, was_paused=was_paused)
        if process is not None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        log_path = self.settings.jobs_root / f"{job_id}.log"
        log_path.unlink(missing_ok=True)
        (self.settings.jobs_root / f"{job_id}.status.json").unlink(missing_ok=True)
        removed_outputs: list[str] = []
        output_errors: list[str] = []
        if delete_outputs:
            for value in owned_outputs:
                try:
                    path = safe_project_path(self.settings, value)
                    runtime = self.settings.runtime_root.resolve()
                    resolved = path.resolve()
                    if resolved == runtime or runtime not in resolved.parents:
                        raise ValueError("owned output is outside the bounded Runtime subtree")
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink(missing_ok=True)
                    removed_outputs.append(value)
                except (OSError, ValueError) as exc:
                    output_errors.append(f"{value}: {exc}")
        return {
            "id": job_id,
            "status": "purged",
            "traceRemoved": True,
            "outputsRemoved": bool(delete_outputs and not output_errors),
            "removedOutputs": removed_outputs,
            "outputErrors": output_errors,
            "sharedOutputsPreserved": True,
        }

    def logs(self, job_id: str, limit: int = 500) -> list[str]:
        with self._lock:
            record = self._jobs.get(job_id)
        if record is None or not record.logPath:
            return []
        path = safe_project_path(self.settings, record.logPath)
        if not path.exists():
            return []
        from services.quant_api.runtime_indexer.parsers import parse_log

        return list(parse_log(path, limit).get("data") or [])

    def stream(self, job_id: str) -> Iterator[str]:
        position = 0
        pending = ""
        path = self.settings.jobs_root / f"{job_id}.log"
        while True:
            record = self.get(job_id)
            if record is None:
                yield f"event: error\ndata: {json.dumps({'message': 'job not found'})}\n\n"
                return
            if path.exists():
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(position)
                    chunk = handle.read()
                    position = handle.tell()
                text = pending + chunk
                lines = text.splitlines(keepends=True)
                pending = ""
                if lines and not lines[-1].endswith(("\n", "\r")):
                    pending = lines.pop()
                for line in lines:
                    yield f"event: log\ndata: {json.dumps({'line': line.rstrip()}, ensure_ascii=False)}\n\n"
            terminal = record["status"] in TERMINAL_STATUSES
            if terminal and pending:
                yield f"event: log\ndata: {json.dumps({'line': pending}, ensure_ascii=False)}\n\n"
                pending = ""
            yield f"event: status\ndata: {json.dumps(record, ensure_ascii=False)}\n\n"
            if terminal:
                return
            import time

            time.sleep(1.0)

    def _run(
        self,
        job_id: str,
        command_id: str,
        parameters: dict[str, Any],
        spec: dict[str, Any],
        log_path: Path,
    ) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or job_id in self._cancel_requested or record.status == "cancelled":
                return
        # `startedAt` is stamped here, but the job is not reported as `running`
        # until its process is registered below. Flipping the status first left a
        # window where an operator could see "running" and have pause rejected
        # for having no process to signal.
        self._update(job_id, status="starting", startedAt=_now(), progress=0.0, message="starting")
        entrypoint = spec.get("entrypoint")
        command = (
            [sys.executable, str(self.settings.project_root / entrypoint)]
            if entrypoint
            else [sys.executable, "-m", "quantagent.cli", command_id]
        )
        # positional subcommands (e.g. `u0_full_universe_backfill.py fetch`) are
        # fixed by the allowlist, never taken from user input.
        command.extend(spec.get("fixed_args", ()))
        # A parameter named in `positional` is passed as a bare argument, but only
        # after being matched against the command's declared choices — a caller
        # can never smuggle an arbitrary token into the argv this way.
        positional = spec.get("positional", ())
        for key in positional:
            value = parameters.get(key)
            choices = spec.get("choices", {}).get(key)
            if value is None or (choices and str(value) not in choices):
                raise ValueError(f"parameter {key} must be one of {sorted(choices or [])}")
            command.append(str(value))
        for key, value in parameters.items():
            if value is None or key in positional:
                continue
            option_name = spec.get("option_aliases", {}).get(key, key.replace("_", "-"))
            option = f"--{option_name}"
            if isinstance(value, bool):
                if value:
                    command.append(option)
                elif key in spec.get("dual_boolean_options", set()):
                    command.append(f"--no-{option_name}")
                continue
            if isinstance(value, list):
                value = ",".join(str(item) for item in value)
            command.extend([option, str(value)])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        status_path = self.settings.jobs_root / f"{job_id}.status.json"
        status_path.unlink(missing_ok=True)
        # The command is wrapped by a supervisor that records the real exit code
        # to disk. Its argv is fixed here; the job's own argv is passed through
        # unchanged after `--`.
        # Invoked by path, deliberately outside the `services.quant_api`
        # namespace: an operator restarting the API with
        # `pkill -f services.quant_api` would otherwise kill the supervisors of
        # every running job along with it.
        supervised = [
            sys.executable,
            str(JOB_SUPERVISOR),
            "--status-file",
            str(status_path),
            "--job-id",
            job_id,
            "--",
            *command,
        ]
        try:
            # The child writes straight to the log file rather than through a
            # pipe this process must keep draining. With a pipe, an API restart
            # left the training process blocked on a full 64KB buffer forever —
            # a job that looked "running" but could never progress again.
            handle = log_path.open("w", encoding="utf-8")
            with handle:
                handle.write(f"$ {' '.join(command)}\n")
                handle.flush()
                with self._lock:
                    if job_id in self._purged or job_id not in self._jobs:
                        return
                    process = subprocess.Popen(
                        supervised,
                        cwd=self.settings.project_root,
                        env=self._process_environment(spec),
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        # Its own session, so restarting or interrupting the API
                        # does not take a multi-hour training down with it.
                        start_new_session=True,
                    )
                    self._processes[job_id] = process
                    cancel_requested = job_id in self._cancel_requested
        except OSError as exc:
            with self._lock:
                purged = job_id in self._purged
            if purged:
                return
            self._update(
                job_id, status="failed", finishedAt=_now(), message="failed to start",
                error=str(exc),
                failure={
                    "code": "spawn_failed",
                    "title": "无法启动任务进程",
                    "detail": str(exc),
                    "remediation": "确认 Python 环境与入口脚本存在，然后重试。",
                    "retryable": True,
                    "logTail": [],
                    "exitCode": None,
                    "signal": None,
                },
            )
            return

        self._update(
            job_id,
            status="running",
            message="running",
            pid=process.pid,
            processStartTicks=process_start_ticks(process.pid),
            statusPath=project_relative(self.settings, status_path),
            lastOutputAt=_now(),
        )
        if cancel_requested:
            terminate_tree(process.pid, was_paused=False)
        self._supervise(
            job_id,
            process=process,
            log_path=log_path,
            status_path=status_path,
            spec=spec,
            parameters=parameters,
        )

    # ------------------------------------------------------------------
    # supervision
    # ------------------------------------------------------------------

    def _supervise(
        self,
        job_id: str,
        *,
        process: subprocess.Popen | None,
        log_path: Path,
        status_path: Path,
        spec: dict[str, Any],
        parameters: dict[str, Any],
        pid: int | None = None,
        start_ticks: int | None = None,
    ) -> None:
        """Follow a job to completion, whether we spawned it or adopted it.

        An adopted job passes ``process=None`` and is followed by PID instead;
        the record's own ``adopted`` flag is what the UI reads.
        """
        position = 0
        next_sample = 0.0
        worker_pid: int | None = None
        while True:
            with self._lock:
                if job_id in self._purged or job_id not in self._jobs:
                    return
            if worker_pid is None:
                worker_pid = self._read_worker_pid(status_path)
                if worker_pid is not None:
                    self._update(
                        job_id,
                        workerPid=worker_pid,
                        workerStartTicks=process_start_ticks(worker_pid),
                    )
            position = self._drain_log(job_id, log_path, position)
            monotonic = time.monotonic()
            if monotonic >= next_sample:
                next_sample = monotonic + RESOURCE_SAMPLE_SECONDS
                self._sample_resources(job_id, worker_pid or pid or (process.pid if process else None))
            if process is not None:
                if process.poll() is not None:
                    break
            elif not process_is_alive(pid, start_ticks):
                break
            time.sleep(POLL_SECONDS)

        # A final drain: the last lines are usually the ones that explain the end.
        position = self._drain_log(job_id, log_path, position)
        with self._lock:
            self._processes.pop(job_id, None)
            cancelled = job_id in self._cancel_requested
            purged = job_id in self._purged
        if purged:
            return
        exit_code, observed = self._resolve_exit_code(process, status_path)
        self._finalize(
            job_id,
            exit_code=exit_code,
            exit_observed=observed,
            cancelled=cancelled,
            log_path=log_path,
            spec=spec,
            parameters=parameters,
        )

    def _resolve_exit_code(
        self,
        process: subprocess.Popen | None,
        status_path: Path,
    ) -> tuple[int | None, bool]:
        """Prefer the supervisor's on-disk record; fall back to our own wait()."""
        status = _read_json(status_path)
        if isinstance(status, dict) and status.get("state") in {"exited", "start_failed"}:
            code = status.get("exitCode")
            if isinstance(code, int):
                return code, True
        if process is not None:
            return process.returncode, process.returncode is not None
        return None, False

    def _read_worker_pid(self, status_path: Path) -> int | None:
        status = _read_json(status_path)
        if isinstance(status, dict):
            worker = status.get("workerPid")
            if isinstance(worker, int):
                return worker
        return None

    def _drain_log(self, job_id: str, log_path: Path, position: int) -> int:
        """Read new log bytes and turn structured lines into job state."""
        if not log_path.exists():
            return position
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(position)
                chunk = handle.read()
                position = handle.tell()
        except OSError:
            return position
        if not chunk:
            return position
        changes: dict[str, Any] = {"lastOutputAt": _now()}
        last_text = ""
        for line in chunk.splitlines():
            text = line.strip()
            if not text:
                continue
            last_text = text
            event = _parse_event_line(text)
            if event is None:
                continue
            if event.get("verdict") in TERMINAL_VERDICTS:
                changes["verdict"] = event["payload"]
            if event.get("progress") is not None:
                changes["progress"] = event["progress"]
            if event.get("stage"):
                changes["stage"] = event["stage"]
                changes["stages"] = self._advance_stage(
                    job_id,
                    stage=str(event["stage"]),
                    message=str(event.get("message") or ""),
                    progress=event.get("progress"),
                )
            if event.get("message"):
                changes["message"] = str(event["message"])[:240]
        # Even unstructured output is evidence the job is alive; showing the last
        # line beats showing "running" for an hour with no other signal.
        changes.setdefault("message", last_text[:240] or "running")
        self._update(job_id, **changes)
        return position

    def _advance_stage(
        self,
        job_id: str,
        *,
        stage: str,
        message: str,
        progress: float | None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            record = self._jobs.get(job_id)
            stages = list(record.stages) if record else []
        now = _now()
        if stages and stages[-1]["id"] == stage:
            stages[-1].update({"message": message or stages[-1].get("message"), "progress": progress})
            return stages
        if stages:
            stages[-1]["completedAt"] = now
            stages[-1]["status"] = "completed"
        stages.append({
            "id": stage,
            "label": STAGE_LABELS.get(stage, stage),
            "startedAt": now,
            "completedAt": None,
            "status": "running",
            "message": message,
            "progress": progress,
        })
        return stages

    def _sample_resources(self, job_id: str, pid: int | None) -> None:
        if not pid:
            return
        sample = sample_process(pid, now=_now())
        if sample is not None:
            self._update(job_id, resources=sample.to_dict())

    def _finalize(
        self,
        job_id: str,
        *,
        exit_code: int | None,
        exit_observed: bool,
        cancelled: bool,
        log_path: Path,
        spec: dict[str, Any],
        parameters: dict[str, Any],
    ) -> None:
        outputs = [
            project_relative(self.settings, value)
            for key, value in parameters.items()
            if key in spec["path_outputs"] and value is not None
        ]
        outputs.extend(spec.get("fixed_outputs", ()))
        with self._lock:
            record = self._jobs.get(job_id)
            stages = list(record.stages) if record else []
            verdict = dict(record.verdict) if record and record.verdict else None
        for stage in stages:
            if stage.get("completedAt") is None:
                stage["completedAt"] = _now()
                stage["status"] = "completed" if exit_code == 0 else "stopped"

        if cancelled:
            self._update(
                job_id, status="cancelled", finishedAt=_now(), stages=stages,
                message="cancelled by operator", outputPaths=outputs,
                exitCode=exit_code, exitStatusObserved=exit_observed,
            )
            return
        if exit_code == 0:
            if self.on_success is not None:
                try:
                    self.on_success()
                except Exception:
                    pass
            self._update(
                job_id, status="succeeded", finishedAt=_now(), progress=1.0, stages=stages,
                message="completed; runtime catalog invalidated", outputPaths=outputs,
                exitCode=0, exitStatusObserved=exit_observed, failure=None,
            )
            return
        if exit_code == CONFIGURATION_BLOCKED_EXIT_CODE:
            # Nothing was measured: the configuration could not answer its own
            # question. Filing this as "rejected" would imply a hypothesis was
            # tested and refused, and filing it as "failed" would imply a bug.
            self._update(
                job_id, status="blocked", finishedAt=_now(), progress=1.0, stages=stages,
                message=(verdict or {}).get("title") or "配置无法执行该研究协议",
                outputPaths=outputs, exitCode=exit_code, exitStatusObserved=exit_observed,
                verdict=verdict or {
                    "verdict": "blocked",
                    "title": "配置无法执行该研究协议",
                    "reasons": (verdict or {}).get("reasons") or [],
                },
                failure=None,
            )
            return
        if exit_code == RESEARCH_REJECTED_EXIT_CODE:
            # A pre-registered gate refused the candidate. The run itself
            # completed: its evidence is intact and worth reading.
            if self.on_success is not None:
                try:
                    self.on_success()
                except Exception:
                    pass
            reasons = (verdict or {}).get("reasons") or []
            self._update(
                job_id, status="rejected", finishedAt=_now(), progress=1.0, stages=stages,
                message=(verdict or {}).get("title") or "研究闸门否决了该候选",
                outputPaths=outputs, exitCode=exit_code, exitStatusObserved=exit_observed,
                verdict=verdict or {
                    "verdict": "rejected",
                    "title": "研究闸门否决了该候选",
                    "reasons": reasons,
                },
                failure=None,
            )
            return

        lines = _read_tail(log_path, 400)
        if not exit_observed:
            failure = JobFailure(
                code="exit_status_unknown",
                title="任务在服务重启期间结束，退出状态未知",
                detail=(
                    "进程在本 API 进程之外结束，没有留下退出码。"
                    "下方日志尾部是它最后写出的内容。"
                ),
                remediation="根据日志判断是否已完成；需要时用相同参数重试。",
                retryable=True,
                log_tail=lines[-40:],
                exit_code=None,
            )
        else:
            failure = diagnose(lines, exit_code=exit_code, cancelled=False)
        self._update(
            job_id,
            status="failed",
            finishedAt=_now(),
            stages=stages,
            message=failure.title,
            error=failure.detail,
            failure=failure.to_dict(),
            exitCode=exit_code,
            exitStatusObserved=exit_observed,
            outputPaths=outputs,
        )

    def _process_environment(self, spec: dict[str, Any]) -> dict[str, str]:
        environment = {
            name: value
            for name, value in os.environ.items()
            if name not in CREDENTIAL_VARIABLES
        }
        providers = set(spec.get("credential_providers", set()))
        if providers and self.connection_environment is not None:
            environment.update(self.connection_environment(providers))
        return environment

    def _normalize_parameters(self, spec: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in parameters.items():
            if value is None:
                normalized[key] = None
                continue
            if isinstance(value, str) and ("\x00" in value or "\n" in value or "\r" in value):
                raise ValueError(f"invalid control character in {key}")
            if key in spec["path_inputs"] | spec["path_outputs"]:
                path = safe_project_path(self.settings, str(value))
                if key in spec["path_inputs"] and not path.exists():
                    raise ValueError(f"input path does not exist: {key}")
                if key in spec["path_outputs"]:
                    runtime = self.settings.runtime_root.resolve()
                    if path != runtime and runtime not in path.parents:
                        raise ValueError(f"output path must be inside runtime: {key}")
                normalized[key] = str(path)
            else:
                if isinstance(value, str) and not re.fullmatch(r"[\w.,:+/ -]*", value):
                    raise ValueError(f"unsupported characters in {key}")
                normalized[key] = value
        return normalized

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or job_id in self._purged:
                return
            for key, value in changes.items():
                setattr(record, key, value)
            self._persist()
        self._emit(record)

    def _emit(self, record: JobRecord) -> None:
        if self.events is None:
            return
        self.events.publish(
            topic=f"jobs:{record.id}",
            event_type="job.status",
            payload={"job": self._public(record)},
            source="quant_api.jobs",
            correlation_id=record.id,
        )

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            self._jobs = {}
            return
        known = {item.name for item in fields(JobRecord)}
        self._jobs = {}
        for item in payload:
            if not isinstance(item, dict) or "id" not in item:
                continue
            # Records written by older builds lack the newer fields; drop unknown
            # keys rather than refusing to load the entire history.
            self._jobs[item["id"]] = JobRecord(**{k: v for k, v in item.items() if k in known})
        for record in list(self._jobs.values()):
            if record.status in ACTIVE_STATUSES:
                self._recover(record)
        self._persist()

    def _recover(self, record: JobRecord) -> None:
        """Reconcile one in-flight job against reality after an API restart.

        Three outcomes, in order of how much we actually know:
        the supervisor recorded an exit code (finalize from it); the process is
        demonstrably still running (re-adopt and keep following it); or neither
        (say the exit status is unknown instead of inventing "failed").
        """
        job_id = record.id
        status_path = (
            safe_project_path(self.settings, record.statusPath)
            if record.statusPath
            else self.settings.jobs_root / f"{job_id}.status.json"
        )
        log_path = (
            safe_project_path(self.settings, record.logPath)
            if record.logPath
            else self.settings.jobs_root / f"{job_id}.log"
        )
        spec = COMMANDS.get(record.commandId) or {"path_outputs": set()}
        status = _read_json(status_path)
        exit_code = status.get("exitCode") if isinstance(status, dict) else None
        state = status.get("state") if isinstance(status, dict) else None

        if state in {"exited", "start_failed"} and isinstance(exit_code, int):
            record.status = "running"  # so _finalize's transition is meaningful
            Thread(
                target=self._finalize,
                args=(job_id,),
                kwargs={
                    "exit_code": exit_code,
                    "exit_observed": True,
                    "cancelled": False,
                    "log_path": log_path,
                    "spec": spec,
                    "parameters": record.parameters,
                },
                daemon=True,
            ).start()
            return

        worker_pid = status.get("workerPid") if isinstance(status, dict) else None
        if isinstance(worker_pid, int):
            record.workerPid = worker_pid
        # Follow whichever half of the pair is still alive. The supervisor can
        # die while the training it started keeps running for hours, and a
        # workstation that declared that training dead would be lying about the
        # most expensive thing on the machine.
        follow_pid, follow_ticks = None, None
        if process_is_alive(record.pid, record.processStartTicks):
            follow_pid, follow_ticks = record.pid, record.processStartTicks
        elif process_is_alive(record.workerPid, record.workerStartTicks):
            follow_pid, follow_ticks = record.workerPid, record.workerStartTicks
        if follow_pid is not None:
            record.adopted = True
            record.message = (
                "已在服务重启后重新接管该进程"
                if follow_pid == record.pid
                else "监督进程已消失，但训练进程仍在运行，已直接接管"
            )
            Thread(
                target=self._supervise,
                args=(job_id,),
                kwargs={
                    "process": None,
                    "log_path": log_path,
                    "status_path": status_path,
                    "spec": spec,
                    "parameters": record.parameters,
                    "pid": follow_pid,
                    "start_ticks": follow_ticks,
                },
                daemon=True,
            ).start()
            return

        lines = _read_tail(log_path, 400)
        verdict_line = next(
            (
                line
                for line in reversed(lines)
                if '"verdict"' in line
                and any(f'"{name}"' in line for name in TERMINAL_VERDICTS)
            ),
            None,
        )
        if verdict_line is not None:
            event = _parse_event_line(verdict_line)
            verdict = (event or {}).get("payload") or {}
            # A run that recorded its own conclusion keeps it across an API
            # restart; recovering it as a generic failure would discard both the
            # verdict and the evidence behind it.
            record.status = "blocked" if verdict.get("verdict") == "blocked" else "rejected"
            record.verdict = verdict or None
            record.progress = 1.0
            record.message = verdict.get("title") or (
                "配置无法执行该研究协议" if record.status == "blocked" else "研究闸门否决了该候选"
            )
            record.finishedAt = record.finishedAt or _now()
            return

        record.status = "failed"
        record.finishedAt = _now()
        failure = JobFailure(
            code="lost_after_restart",
            title="服务重启后该任务的进程已不存在",
            detail=(
                "API 重启时该任务仍在运行，重启后进程已消失且没有留下退出码，"
                "因此无法断定它是完成还是中断。"
            ),
            remediation="查看日志尾部与输出目录判断进度；需要时用相同参数重试。",
            retryable=True,
            log_tail=lines[-40:],
            exit_code=None,
        )
        record.failure = failure.to_dict()
        record.error = failure.detail
        record.message = failure.title
        record.exitStatusObserved = False

    def _persist(self) -> None:
        # A per-write temp name: supervision threads and recovery threads persist
        # concurrently, and a shared "jobs.tmp" meant one thread's rename could
        # delete the file another was about to rename.
        with self._lock:
            snapshot = [asdict(record) for record in self._jobs.values()]
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_name(f"jobs.{os.getpid()}.{uuid4().hex[:8]}.tmp")
        try:
            temp.write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp.replace(self.state_path)
        except OSError:
            temp.unlink(missing_ok=True)
            raise

    @staticmethod
    def _public(record: JobRecord) -> dict[str, Any]:
        data = asdict(record)
        data.pop("logPath", None)
        data.pop("ownedOutputPaths", None)
        data.pop("processStartTicks", None)
        data.pop("statusPath", None)
        data["terminal"] = record.status in TERMINAL_STATUSES
        data["elapsedSeconds"] = _elapsed_seconds(record)
        data["canRetry"] = bool(record.status in RETRYABLE_STATUSES and record.parameters)
        return data


def _elapsed_seconds(record: JobRecord) -> float | None:
    if not record.startedAt:
        return None
    started = _parse_time(record.startedAt)
    if started is None:
        return None
    end = _parse_time(record.finishedAt) if record.finishedAt else datetime.now(timezone.utc)
    if end is None:
        return None
    return round((end - started).total_seconds(), 1)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_tail(path: Path, limit: int) -> list[str]:
    """Last `limit` lines without loading a multi-gigabyte log into memory."""
    if not path.exists():
        return []
    try:
        size = path.stat().st_size
        window = min(size, 512 * 1024)
        with path.open("rb") as handle:
            handle.seek(size - window)
            data = handle.read(window)
    except OSError:
        return []
    text = data.decode("utf-8", errors="replace")
    if window < size:
        text = text.split("\n", 1)[-1]
    return text.splitlines()[-limit:]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_event_line(line: str) -> dict[str, Any] | None:
    """Extract progress, stage, message and verdict from one output line.

    Commands publish structured JSON lines; anything else is treated as plain
    output. The previous implementation read only `progress` and then used the
    entire raw JSON blob as the human-facing message, which is why the UI could
    never say which stage a run was in.
    """
    text = line.strip()
    bracket = re.search(r"\[(\d+)\s*/\s*(\d+)\]", text)
    if bracket and int(bracket.group(2)) > 0:
        return {
            "progress": min(0.99, int(bracket.group(1)) / int(bracket.group(2))),
            "stage": None,
            "message": text[:240],
        }
    if not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None

    event: dict[str, Any] = {"progress": None, "stage": None, "message": None}
    if payload.get("verdict") in TERMINAL_VERDICTS:
        event["verdict"] = payload["verdict"]
        event["payload"] = payload
    value = payload.get("progress")
    if isinstance(value, (int, float)):
        event["progress"] = min(1.0, max(0.0, float(value)))
    else:
        for current_key, total_key in (
            ("batch", "total_batches"),
            ("iteration", "total_iterations"),
            ("rows_written", "total_rows"),
            ("fold", "total_folds"),
        ):
            current, total = payload.get(current_key), payload.get(total_key)
            if isinstance(current, (int, float)) and isinstance(total, (int, float)) and total > 0:
                event["progress"] = min(0.99, max(0.0, float(current) / float(total)))
                break
    if isinstance(payload.get("stage"), str):
        event["stage"] = payload["stage"]
    if isinstance(payload.get("message"), str):
        event["message"] = payload["message"]
    if event["progress"] is None and event["stage"] is None and event["message"] is None:
        return None
    return event


    value = payload.get("progress")
    if isinstance(value, (int, float)):
        return min(0.99, max(0.0, float(value)))
    for current_key, total_key in (
        ("batch", "total_batches"),
        ("iteration", "total_iterations"),
        ("rows_written", "total_rows"),
    ):
        current, total = payload.get(current_key), payload.get(total_key)
        if isinstance(current, (int, float)) and isinstance(total, (int, float)) and total > 0:
            return min(0.99, max(0.0, float(current) / float(total)))
    return None
