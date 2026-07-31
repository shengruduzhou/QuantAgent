from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from threading import RLock, Thread
from typing import Any, Callable, Iterator
from uuid import uuid4

from quantagent.safety.operating_mode import reject_live_intent
from services.quant_api.config import ApiSettings, project_relative, safe_project_path
from services.quant_api.events import EventBroker
from services.quant_api.services.connections import CREDENTIAL_VARIABLES


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
            "minute_panel_path",
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

    def submit(self, job_type: str, command_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
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
        )
        with self._lock:
            self._jobs[job_id] = record
            self._persist()
        self._emit(record)
        Thread(target=self._run, args=(job_id, command_id, normalized, spec, log_path), daemon=True).start()
        return self._public(record)

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
        terminal = {"succeeded", "failed", "cancelled"}
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            if record.status in terminal:
                raise ValueError(f"job {job_id} already finished with status {record.status}")
            self._cancel_requested.add(job_id)
            process = self._processes.get(job_id)
            if process is None:
                record.status = "cancelled"
                record.message = "cancelled before process start"
                record.finishedAt = _now()
            else:
                record.status = "cancelling"
                record.message = "termination requested"
            self._persist()
            public = self._public(record)
        self._emit(record)
        if process is not None:
            try:
                process.terminate()
            except OSError:
                pass
        return public

    def purge(self, job_id: str, *, delete_outputs: bool = False) -> dict[str, Any]:
        """Stop a job and remove its operational trace and owned outputs."""

        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            process = self._processes.pop(job_id, None)
            owned_outputs = list(record.ownedOutputPaths)
            self._cancel_requested.add(job_id)
            self._purged.add(job_id)
            self._jobs.pop(job_id, None)
            self._persist()
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except OSError:
                pass
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        log_path = self.settings.jobs_root / f"{job_id}.log"
        log_path.unlink(missing_ok=True)
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
            terminal = record["status"] in {"succeeded", "failed", "cancelled"}
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
        self._update(job_id, status="running", startedAt=_now(), progress=0.0, message="running")
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
        try:
            with log_path.open("w", encoding="utf-8") as handle:
                handle.write(f"$ {' '.join(command)}\n")
                handle.flush()
                with self._lock:
                    if job_id in self._purged or job_id not in self._jobs:
                        return
                    process = subprocess.Popen(
                        command,
                        cwd=self.settings.project_root,
                        env=self._process_environment(spec),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    self._processes[job_id] = process
                    cancel_requested = job_id in self._cancel_requested
                if cancel_requested:
                    process.terminate()
                assert process.stdout is not None
                for line in process.stdout:
                    with self._lock:
                        if job_id in self._purged:
                            process.terminate()
                            break
                    handle.write(line)
                    handle.flush()
                    progress = _progress_from_line(line)
                    if progress is not None:
                        self._update(
                            job_id,
                            progress=progress,
                            message=line.strip()[:240] or "running",
                        )
                code = process.wait()
            with self._lock:
                self._processes.pop(job_id, None)
                cancelled = job_id in self._cancel_requested
                purged = job_id in self._purged
            if purged:
                return
            outputs = [
                project_relative(self.settings, value)
                for key, value in parameters.items()
                if key in spec["path_outputs"] and value is not None
            ]
            outputs.extend(spec.get("fixed_outputs", ()))
            if cancelled:
                self._update(
                    job_id, status="cancelled", finishedAt=_now(),
                    message="cancelled", outputPaths=outputs,
                )
            elif code == 0:
                if self.on_success is not None:
                    try:
                        self.on_success()
                    except Exception:
                        pass
                self._update(
                    job_id, status="succeeded", finishedAt=_now(), progress=1.0,
                    message="completed; runtime catalog invalidated", outputPaths=outputs,
                )
            else:
                self._update(
                    job_id, status="failed", finishedAt=_now(), message=f"exit code {code}",
                    error=f"command exited with code {code}",
                )
        except OSError as exc:
            with self._lock:
                self._processes.pop(job_id, None)
                cancelled = job_id in self._cancel_requested
                purged = job_id in self._purged
            if purged:
                return
            if cancelled:
                self._update(job_id, status="cancelled", finishedAt=_now(), message="cancelled")
            else:
                self._update(
                    job_id, status="failed", finishedAt=_now(), message="failed to start",
                    error=str(exc),
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
            self._jobs = {item["id"]: JobRecord(**item) for item in payload}
            for record in self._jobs.values():
                if record.status in {"queued", "running", "cancelling"}:
                    record.status = "failed"
                    record.finishedAt = _now()
                    record.error = "API process restarted before job completed"
        except (OSError, ValueError, TypeError):
            self._jobs = {}

    def _persist(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(
            json.dumps([asdict(record) for record in self._jobs.values()], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp.replace(self.state_path)

    @staticmethod
    def _public(record: JobRecord) -> dict[str, Any]:
        data = asdict(record)
        data.pop("logPath", None)
        data.pop("ownedOutputPaths", None)
        return data


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _progress_from_line(line: str) -> float | None:
    """Parse provider progress without requiring a provider-specific protocol."""
    match = re.search(r"\[(\d+)\s*/\s*(\d+)\]", line)
    if match and int(match.group(2)) > 0:
        return min(0.99, int(match.group(1)) / int(match.group(2)))
    try:
        payload = json.loads(line)
    except (TypeError, ValueError):
        return None
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
