# R9_debug findings (Round 29) — incremental

## A. Branch adjudication (read-only; main=22b3f6c)

| Branch | ahead/behind main | Verdict | Evidence |
|---|---|---|---|
| origin/agent/authoritative-acceptance-calendar | 9/189 | **VALUABLE-UNMERGED** | `execution/acceptance_calendar.py` (306 l, digest-bound exchange session set) absent in main; main `live_model_trust_v2.py:401-408` still counts FRESH `trading_days` from the evidence file's own unique dates, no authoritative calendar. Branch integration was never finished: 4/9 commits are CI chores that apply `scripts/issue71_calendar_binding_patch.py` (string-replace patcher) via a workflow. Port = module + tests + hand wiring into live_model_trust_v2; drop the CI patch scaffolding. Low urgency (live disabled; FRESH first read ~2026-11). |
| origin/agent/canonical-execution-prefix-binding | 1/199 | **SUPERSEDED** | Same concept merged as #131 `e4b6c77` "bind terminal execution evidence to canonical ledger prefixes" → `paper/canonical_receipt.py` (build/verify_canonical_prefix_receipt) used by `paper/continuous_execution.py:585`. |
| origin/agent/continuous-paper-account | 13/232 | **SUPERSEDED** | All 5 added files exist in main via #80 `f3c4f88` / #83 `7643d9b`, and main's versions are larger (execution_journal +365/-52, pending_signal +202/-35 vs branch tip). |
| origin/agent/fuyao-hithink-integration-20260807 | 5/337 | **SUPERSEDED** | `hithink_finance_provider.py` replaced by `data/providers/fuyao_provider.py` from `9fd56d6` (same base URL, `DEFAULT_TOKEN_ENV="HITHINK_FINANCE_API_KEY"`), `cli/fuyao.py`, `data/fuyao_dump.py`; `.env.example:29` already has the var. |
| origin/agent/parent-child-execution-tca | 4/197 | **SUPERSEDED** | `execution/parent_child.py`, `tca.py` + tests merged as #133 `27991fb` (session-bound planning, fill-id TCA); main versions are newer (parent_child +291/-99). |
| origin/agent/rl-executable-clock-governance | 13/204 | **SUPERSEDED** (clock) | Clock fix merged as #130 `0473178` (T→T+1 exec→T+1/T+2 reward, quarantine). Branch's extra deletions not in main: `scripts/rl_strict_eval.py`, `scripts/rl_train_eval_2026.py` (ported by this round's deletion batch), `rl/portfolio_env.py` (kept on purpose: Round 21 ruled it a governed deprecation, ctor raises without ack flag), `scripts/forward_rl_book.py` (#130 turned it into a tombstone, tested by `tests/rl/test_forward_rl_tombstone.py`). |
| origin/agent/paper-shadow-next-session-state-v2 (#90) | 19/231 | **SUPERSEDED** | `paper/pending_execution.py` replaced by `paper/continuous_execution.py` (#83 `7643d9b`, then #126/#128/#129/#131): next-proven-session, OrderManager→PaperBrokerAdapter→PaperBroker, prefix receipts. Its dated 2026 ST-limit/market_rules commits are already in main byte-identical (`market_rules/__init__.py`, `ashare_rules.py`, ST tests: zero diff). |
| origin/agent/paper-shadow-next-session-state-v3 (#94) | 2/230 | **SUPERSEDED** | Same `pending_execution.py` design as v2 (docstring "pending signal → idempotency → OMS → paper adapter → venue"); main's `continuous_execution.py` docstring covers every bullet plus account-genesis binding and indeterminate-freeze. |
| origin/agent/quant-foundations-fuyao-parity (#47) | 3/331 | **SUPERSEDED** | `cli/research_foundations.py audit-nonlinear-promotion` ≈ main `cli/nonlinear.py audit-nonlinear-factors` + `research/nonlinear_promotion.py`; main's `black_scholes.py`/`performance.py` are supersets (main has `_arbitrage_bounds`, `_zero_vol_result`, `newey_west_auto_lag`, `_spa_recenter_means` the branch lacks); correctness tests live in `tests/research/test_quant_correctness_regressions.py`. |
| origin/feat/fuyao-u0-integration-20260807 (#41) | 11/337 | **SUPERSEDED** (minor residue) | Replaced by `9fd56d6` (FuyaoProvider/dump/CLI) and deterministic `max_candidates` truncation in `models/interactions.py:264-308`. Only residue: 27-line Fuyao registration in `scripts/u0_acquire_bars.py` and `docs/quant_model_foundations.md` — a different design from what main chose; do not port as-is. |

### Local cleanup candidates (0 commits ahead of main; chair decides)
- Gone-upstream & merged: `1`, `s`, `robustness-mission`, `agent/ashare-mt5-tick-multiagent`, `agent/full-universe-training-paper-workstation`, `agent/h032a-u0-readiness`, `agent/h032b-tickflow-u0-closure`, `agent/h032c-u0-finalize`, `agent/local-paper-full-universe-workstation`, `agent/qmt-data-closure-ci-recovery`.
- Merged, no upstream: `agent/pr-a-recovery-ci`, `agent/round24-rl-verdict`, `agent/u0-ashare-data-foundation`, `agent/u0-assembly-scale`, `agent/u0-data-closure`, 9× `worktree-agent-*` (b56ae57/b313153/22b3f6c).
- Stale worktree: `.claude/worktrees/agent-a8615b40ef65e5897` on merged `agent/round24-cleanup` (9e094e7).
- Detached scratch worktree `round29/R11_rl/pr156` (f273546 = PR #156, merged as f1b122e) — R11's; remove after R11 finishes.

## B. Bugs

### R9_debug-F01 [P1] Full-pipeline acceptance producer still forges max_drawdown/sharpe/excess = 0.0 (DEF-025 shape survives DEF-023)
- Location: src/quantagent/cli/v7_train.py:2256-2262 (`_build_full_pipeline_acceptance_metrics`)
- Claim: `paper_summary.get("max_drawdown", 0.0)`, `.get("sharpe", training.get("sharpe", 0.0))`, `.get("turnover_adjusted_net_return", ..., 0.0)`, `.get("excess_return_after_costs", ..., 0.0)`. The consumer (`v7_quality_gates.add_measured_gate`) was hardened to report `unknown` on a missing value, but the producer supplies 0.0, so a missing drawdown reaches the gate as a *measured* pass.
- Reproduction: `repro/repro_acceptance_forged_defaults.py` → `{'max_drawdown': 0.0, 'sharpe': 0.0, 'turnover_adjusted_net_return': 0.0, 'excess_return_after_costs': 0.0}`; gates: `max_drawdown pass 0.0`, `turnover_adjusted_net_return fail 0.0`, `excess_return_after_costs fail 0.0` (gate comment says excess must be "not measured", not 0).
- Reachability: latent in the CLI path today — `backtest/paper_report.py:343-370` always emits these keys (value may be None, which `.get` passes through). Any other caller (tests already call it with `paper_summary={}`) gets forged values.
- Second defect, same function: `benchmark_status`, `benchmark_sessions_covered`, `benchmark_sessions_missing` are not forwarded, so a run whose benchmark has holes is told "没有基准，指定 benchmarkSymbol" (status `absent`) instead of the DEF-022 "基准数据不完整 N/M" remediation. Repro output: `benchmark_status forwarded: None`, `excess gate: unknown absent`.
- Fix: drop the 0.0 defaults; forward the three coverage keys. Test asserts gate status `unknown` for empty summary and `incomplete` remediation.
- Confidence: CONFIRMED

### R9_debug-F02 [P1] `train-rl-agent --env-config` silently drops valid PITPortfolioEnvConfig keys (incl. risk terms, max_book, reward_end_date_limit) and admits stale ones
- Location: src/quantagent/cli/v7_train.py:3053-3077 (`_load_env_config`)
- Claim: `allowed = set(PITPortfolioEnvConfig.__dataclass_fields__) if "PITPortfolioEnvConfig" in globals() else {...}`. The class is only imported *locally* inside `train_rl_agent`, so the `globals()` test is always False and the hard-coded fallback set (legacy PortfolioEnv keys: top_n, max_delta, max_weight_per_name, max_turnover, drawdown_limit, kill_switch_drawdown, initial_nav, ...) is always used. Only `max_gross`, `cost_bps`, `drawdown_lambda` overlap with the real config.
- Reproduction (verbatim): YAML `{max_book: 40, volatility_lambda: 0.5, reward_end_date_limit: "2025-08-29", drawdown_lambda: 0.2}` → `kept: {'drawdown_lambda': 0.2}` → `max_book 60 volatility_lambda 0.0 reward_end_date_limit None`. YAML `{max_turnover: 0.3}` → `TypeError: PITPortfolioEnvConfig.__init__() got an unexpected keyword argument 'max_turnover'`.
- Impact: an operator who turns on the volatility risk term or shrinks the book via YAML trains a different policy than configured, with no error. `reward_end_date_limit` loss is mitigated: `train_ppo._quarantine_clamped_env_config` clamps to the quarantine boundary regardless (train_ppo.py:96-117), so this is not a holdout leak.
- Fix: derive `allowed` from `dataclasses.fields(PITPortfolioEnvConfig)` (import at function scope) and reject unknown keys loudly. Test: YAML round-trip keeps every real field and raises on a legacy key.
- Confidence: CONFIRMED

### R9_debug-F03 [P2] `v7/agent_contracts.py` declares 6 "existing_extension_points" that do not exist; nothing checks them
- Location: src/quantagent/v7/agent_contracts.py:71,137,145,174,189 (`existing_extension_points`)
- Claim: `agents/policy_agent.py`, `agents/news_agent.py`, `agents/sentiment_agent.py`, `models/v6_model_system.py`, `models/v6_outputs.py`, `portfolio/v6_portfolio_service.py` are listed as existing extension points; none exists. The only reader of `V7_AGENT_SPECS` outside the module is `tests/test_v7_architecture_contracts.py`, which never checks the field.
- Reproduction: `grep -rhoE "src/quantagent/[A-Za-z0-9_/.-]+\.py" src | sort -u | while read p; do [ -e "$p" ] || echo MISSING $p; done` → the 6 paths above.
- Impact: this file is the reason Rounds 23/24 kept `quant_math/factor_attribution.py`, `portfolio/sector_etf_allocator.py`, `factors/governance.py`, `execution/audit_replay.py`, `fundamental/{quality,scores,valuation_agent}.py` (path-string references). A contract that already points at 6 non-existent files is not evidence that those 7 zero-importer modules are live; it is the same "documented capability that does not exist" defect.
- Fix (proposed, not applied — needs an owner decision): add a test that every `existing_extension_points` path exists, then either delete the 6 stale entries or the whole field. Once the contract is honest, the 7 kept modules (~530 lines) become ordinary zero-importer candidates.
- Confidence: CONFIRMED

## C. Dead code (proof method)
- Import graph: `import_graph.py` (AST: absolute + relative imports, `importlib.import_module("quantagent...")` literals, package `__init__` re-exports) + `reach.py` (transitive reachability from roots = `src/quantagent/cli/**`, `services/**`, `scripts/**`). 532 modules, 455 reachable, 72 unreachable (34 with no test importer either).
- Text proof per module: `verify2.sh <pkg/mod> <every public symbol>` greps dotted name, `pkg/mod.py`, `pkg.mod`, and each symbol (word-bounded) over the whole repo excluding .git/node_modules/runtime/rd-agent/.claude/dist/venv and `egg-info/SOURCES.txt`. Output saved in `verify_batch_a.txt`.
- Static kwarg check `kwcheck.py`: every dataclass constructor call with keywords the dataclass lacks → only `scripts/rl_train_eval_2026.py:48` and `scripts/rl_strict_eval.py:44` (`PITPortfolioEnvConfig(max_turnover=...)`).

### Batch D1 — zero-reference modules (no importer, no test, no path string, no doc)
| file | lines | verify2 result |
|---|---|---|
| src/quantagent/factors/dag.py | 98 | empty |
| src/quantagent/quant_math/conformal.py | 84 | empty (calibration lives in `ensemble/calibration.py`) |
| src/quantagent/quant_math/labels.py | 91 | empty |
| src/quantagent/quant_math/neutralization.py | 71 | only `factors/preprocessing.py:11` defines its *own* `winsorize_by_date` (duplicate helper, not an import) |
| src/quantagent/quant_math/position_sizing.py | 23 | empty (`fractional_kelly`, `volatility_target_weight`, `confidence_adjusted_alpha`) |
| src/quantagent/fundamental/statements.py | 94 | empty |
| src/quantagent/training/losses.py | 71 | only the Round 24 proposal doc (`docs/audits/round24/09_cleanup.md:117-130`, which recommended this deletion after `composite_loss.py` went) |

### Kept after review (with reason)
- `risk/risk_gate.py` RiskGate + `risk/risk_limits.py` V6RiskLimits: KEEP. RiskGate is the required `risk_gate` argument of `execution/live_session.LiveTradingSession` (the dormant, fail-closed live arming boundary; `services/quant_api/services/production_readiness.py` imports that module), exercised by 5 test files, and AGENTS.md requires "QMT submit 前必须通过 risk gate". Dormant ≠ dead. Only change: drop the unused `V6RiskLimits` import from `risk/decision_chain.py` (pyflakes), which implied the 15-gate chain reads those limits.
- `rl/reward_ablation_experiment.py`: KEEP — documented `python -m` reproduction entry in `docs/audits/round23/11_rl_ablation.md:279`, `round24/11_rl_verdict.md:60`.
- `cli/__main__.py`: KEEP — `python -m quantagent.cli` entry point.
- AGENTS-path-string modules (`factor_attribution`, `sector_etf_allocator`, `factors/governance`, `execution/audit_replay`, `fundamental/{quality,scores,valuation_agent}`): kept this round; see F03 for why the reason is weak.

### R9_debug-F04 [P2] Model observability API surfaces burned-holdout PPO numbers as model metrics with no holdout/trust stamp
- Location: services/quant_api/adapters/models.py:15-26 (`MODEL_METADATA_NAMES` includes `eval_2026.json`, `strict_eval_2026.json`), :66-75 (`evaluations`), :741-755 (`_normalized_metrics` flattens every numeric leaf).
- Claim: `runtime/models/v88_rl_overlay/{eval_2026.json,strict_eval_2026.json}` exist (keys: `annualized, maxDD, sharpe, window_days` / `strict_annualized, strict_sharpe, strict_maxDD, excess_vs_paper_bench_ann, ...`). They were produced by the two scripts deleted in this round on a 2026-YTD window inside the burned holdout; `eval_2026.json` is the frictionless env-analytic number its own producer labelled "REJECTED as evaluation basis". The adapter lists both as evaluations and flattens all numeric fields into the model's `metrics` with no quarantine / trust-class marker (backtests got such a stamp in 91a9a56; models did not).
- Reproduction: `python3 -c "import json;print(sorted(json.load(open('runtime/models/v88_rl_overlay/eval_2026.json'))))"` → `['annualized', 'maxDD', 'note', 'sharpe', 'window_days']`; adapter code above. (Did not read the values.)
- Fix: drop the two names from `MODEL_METADATA_NAMES`/`evaluations`, or stamp any evaluation whose window overlaps `configs/quarantined_windows.json` like backtests are stamped. UI-role file; not changed here.
- Confidence: CONFIRMED (code path + artifact presence); rendering not checked in a browser.

### Batch D2 — shims with only doc/path-string references (docs fixed in the same commit)
| commit | file(s) | lines | references and how resolved |
|---|---|---|---|
| ac58ed6 | execution/risk_kill_switch.py | -95 | 0 importers; path strings in scripts/phase2_ledger.py:894 (M5-07) + docs/architecture/phase2_ledger.json:752 → now cite paper/risk.py + paper/broker.py (regenerated JSON identical for M5-07, stage unchanged `backtest_only`); docs/V7_live_readiness_gates.md:58, docs/quant_ui_code_map.md:164 rewritten to the real kill switch. Two "used to come from KillSwitchLimits" prose comments (adapters/risk.py:80, tests/quant_ui/test_adapters.py:127) left as history. Layer check: risk.kill_switch.KillSwitch still used by risk_gate, decision_chain, 3 tests; portfolio.position_state.StopDecision still used by strategy/risk_gate + portfolio/__init__ ⇒ no new orphan. |
| 77e762d | clean_room/dataset.py, clean_room/risk.py | -303 (+docstring fix) | only importer = clean_room/__init__ re-export; `__init__` now exports engine only; stale "STILL OPEN" interior-bar claim corrected (DEF-038 closed). |
| 3e275e1 | scripts/rl_train_eval_2026.py, scripts/rl_strict_eval.py | -199 | TypeError on `max_turnover` (reproduced), 2026-YTD burned-holdout windows; only reference a filename excerpt in docs/research/代码清理与删除证据.md, now annotated. |
| 00423bf | cli/v8.py (2 unreachable lines), risk/decision_chain.py (unused import), risk/risk_gate.py (+keep docstring) | -3/+9 | pyflakes undefined-name / unused-import. |

### Batch D3 — AFML lookalikes (chair rule: delete unless a test documents an intended consumer)
| commit | change | lines |
|---|---|---|
| 7b602b9 | delete quant_math/triple_barrier.py (triple_barrier_labels, daily_volatility, sample_weights_by_uniqueness, meta_label) and purged_cv.combinatorial_purged_split; inventory test keeps WIRED checks + guard that the 5 names are not re-added (guard: 5 defs on parent tree, 0 after) | -131 src, test 219→~70 |

### R9_debug-F05 [P1, low reachability] Target-weights builder selects a name that has no market bar that day; every tradability block reads the gap as "tradable"
- Location: src/quantagent/portfolio/v7_target_weights.py:336 (`day.merge(day_market, how="left")`) + :356 (`blocked = merged[column].fillna(False)`)
- Claim: DEF-040 fixed the *column*-absent case, but a *row*-absent case remains: a predicted symbol without a market row on the date gets NaN in is_suspended/is_st/is_limit_up/is_limit_down → `fillna(False)` → not blocked, and no rejection is recorded. With the default `min_amount_yuan=0.0` the illiquidity rule (which would fail closed via `amount.fillna(0.0)`) is off.
- Reproduction: `repro/repro_target_weights_missing_bar.py` (run twice, identical md5 `bcf19ecd...`): 10 names, best-scored `000001.SZ` has no bar → output weights `000001.SZ 0.1, 000002.SZ 0.1, 000003.SZ 0.1`, `rejected: []`.
- Impact: in the standard pipeline predictions are built from the same panel, so a same-day missing row is rare; it bites when `--market-panel` differs from the dataset panel or after `_restrict_market_for_paper`. The downstream simulator then cannot fill it (no price) → silent cash drag, and the selection report claims the name passed ST/suspension checks it never had data for.
- Fix (proposed, not applied — production selection path, wants an owner): merge with `indicator=True`; when any tradability block is requested, reject `_merge == "left_only"` rows with reason `no_market_row`. Test = this repro asserting 000001.SZ absent and a `no_market_row` rejection.
- Confidence: CONFIRMED

### R9_debug-F06 [P3, REFUTED as behaviour risk] pandas "Downcasting object dtype arrays on .fillna" FutureWarnings
- Locations flagged: universe/filters.py:279,432; data/v7_label_builder.py:141; quant_math/ashare.py:530-531; data/v7_dataset_builder.py:170; portfolio/v7_target_weights.py:356; backtest/full_pipeline_backtester.py:178.
- Check: ran the 44 test files that import these modules under `pd.set_option("future.no_silent_downcasting", True)` (plugin `r9_nodowncast.py`): **372 passed, 14 skipped, 0 failed**. Direct probes: `enforce_tradability` with missing can_buy/can_sell rows returns the same capped weights; `_untradable_at_exit` returns dtype bool and `~mask` inverts correctly. Every flagged site either ends in `.astype(bool)` or feeds a bool op that pandas re-infers.
- Verdict: noise today, not a silent behaviour change under the opt-in future mode. Cleanup = `.astype("boolean").fillna(False).astype(bool)` or `infer_objects`; not done (no behaviour gain).
- Other pandas-3 warnings still present (not downcasting): `groupby.apply` on grouping columns (training/v7_experiment.py:1229-1230, data/v7_quality_gates.py:1129, diagnostics/stratified_ic.py), setitem incompatible dtype (data/providers/st_pit.py:225 — off-limits area), concat with all-NA (data/sector/sector_mapping.py:454). Not measured.

### Also checked, no defect
- pyflakes over src/services/scripts: 19 "undefined name" hits; all are string annotations / TYPE_CHECKING-style forward refs except cli/v8.py:1454-1455 (unreachable, removed) and cli/v7_train.py:3066 (the F02 `globals()` test).
- Static dataclass-kwarg scan: only the two deleted RL scripts constructed a dataclass with a non-existent field.
- `except Exception: return <default>` scan: model_comparison._lightgbm_available (fine), intraday_dot_factor_combo._time_to_minute→0.0 on unparsable time (P3, research-only, not changed).
