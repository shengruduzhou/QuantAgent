# R9_debug summary (Round 29)

Branch `agent/round29-deadcode` = `80ad515` (chair tip, fast-forwarded) + 8 commits; net 27 files, +233 / −1471.

## Commits
| sha | subject | +/− |
|---|---|---|
| e8fdcfd | delete 7 zero-reference helpers (factors/dag, quant_math/{conformal,labels,neutralization,position_sizing}, fundamental/statements, training/losses) | −532 |
| ac58ed6 | delete execution/risk_kill_switch shim; phase2 ledger M5-07 + 2 docs re-pointed to paper/risk.py, paper/broker.py | +6/−101 |
| 77e762d | delete clean_room/{dataset,risk}.py (only re-exported); fix stale DEF-038 "STILL OPEN" docstring | +7/−313 |
| 3e275e1 | delete scripts/rl_train_eval_2026.py, rl_strict_eval.py (TypeError max_turnover; burned-holdout window) | +2/−199 |
| 00423bf | unreachable v8 tail, unused V6RiskLimits import, RiskGate keep-reason docstring | +11/−3 |
| 7b602b9 | delete unwired AFML lookalikes (triple_barrier.py, combinatorial_purged_split) | +44/−306 |
| 65bf186 | F01 fix + 4 tests (3 fail pre-fix) | +97/−4 |
| d4ce8e8 | F02 fix + 3 tests (2 fail pre-fix) | +66/−13 |

## Full-suite counts
- baseline d4e3d3c (worktree): 1 failed, 3745 passed, 47 skipped (failure = model_comparison label-availability test, fixed by chair 80ad515)
- baseline 80ad515 (worktree): 3800 passed, 47 skipped, 0 failed
- d4ce8e8, archive snapshot (all 8 commits): 3794 passed, 47 skipped, 0 failed (= 3800 − 13 removed AFML tests + 7 new)
- d4ce8e8, worktree: see pytest_final.log
- D1/D2 intermediate snapshot runs were killed by the session interruption; the cumulative run covers them.

## Verified correct / refuted (do not re-audit)
- pandas downcasting FutureWarnings: 44 affected test files pass under `future.no_silent_downcasting=True` (372 passed) — not a behaviour change.
- RiskGate/V6RiskLimits: dormant (LiveTradingSession only), not dead — kept with docstring.
- Only the two deleted RL scripts call a dataclass with a non-existent kwarg (static scan).

## Not done / open
- F03 agent_contracts stale extension points (6 missing files) — owner decision.
- F04 model API shows burned-holdout PPO numbers unstamped — UI role.
- F05 target-weights missing-bar fail-open — production selection path, owner decision.
- Frontend: `pages/WalkForwardRiskPage.tsx` has zero importers/no route (backend /api/walkforward-risk/* mounted, no consumer); `components/CandlestickChart.tsx` imported only by its test. Not deleted: eng-ui branch is editing apps/quant-ui; adding/removing a route is an IA decision.
- Test-only clusters (unreachable from cli/services/scripts, kept alive only by tests): strategy/{decision_engine,position_sizing,risk_gate,score_fusion} (US-ticker toy), agents/{ashare_specialists,flow_agent,financial_statement_agent,quality_gate,sector_rotation_agent}, backtest/{tplus1_engine,trace_proven_strict_v8}, portfolio/{market_regime_detector,position_policy,robust_pareto,sector_rotation,decision_chain,state_machine}, quant_math/{black_scholes,constraints,ic_analysis,signal_fusion}, training/{metrics,regime_sub_models}, data/{v7_sources,fundamental/extended_ranker,sector/decision_pool,credibility}, diagnostics/post_mortem, execution/{qmt_gateway,parent_child,tca} — full list in reach.py output. AGENTS.md:38 counts tests as references, so these need an explicit chair ruling.
