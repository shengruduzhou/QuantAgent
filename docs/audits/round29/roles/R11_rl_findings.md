# R11_rl — Round 29 independent audit (recorded by chair from the agent's hand-back text; the harness refused the agent's own report-file writes)

Binding pre-registration: repro/prereg.json, hashed 2026-09-30T15:54:34+09:00 before any run (data_evidence/prereg.sha256 fb299dee…).
Counts: P0 0 · P1 1 · P2 5 · P3 3.

## Usefulness test — DO NOT ENABLE (pre-registered rule applied verbatim)
Setup: certified gold close + v7 limit flags, walk-forward OOS alpha_5d, hold-band book (30 names, exit rank 90, min hold 10, rebalance 5); 4 expanding folds V1–V4 all ≤2025-08-29; PPO 150k steps × 5 seeds + passive + untrained arms; 44/44 runs complete.
PPO excess IQM negative on every fold; every 95% stratified-bootstrap CI excludes 0 from below:
V1 −1.97% [−3.72, −0.36]; V2 −2.49% [−3.02, −1.71]; V3 −3.04% [−4.57, −2.76]; V4 −1.51% [−2.34, −1.15]; aggregate −2.35% [−2.78, −1.94].
0/20 PPO runs beat the book; Sharpe-change IQM −0.125; untrained ≈ 0. MaxDD worse in V1 (+0.32pp); better in V2–V4 only with ~5pp less gross.
Decomposition: extra turnover cost −1.76% (turnover 0.34 vs 0.21) + selection/de-grossing residual −0.78%.
Sanity: zero action ⇒ value-add exactly 0.0 on real data; V4 seed 42 fresh-process rerun bit-identical.
Evidence: data_evidence/verdict_prereg.json, runs_prereg.csv, decomposition_prereg.csv, prep_report.json.
Exploratory (post-hoc, non-binding, interrupted): 400k steps (6/20 runs) V1 IQM ≈ −2.4%, turnover 0.63; random 60-step windows w/o time feature (8/20) V1 IQM ≈ −3.1%. Neither rescues RL.

## PR #156 — MERGE-WITH-CHANGES (minor)
127 passed (tests/rl + evidence tests), 22 passed affected non-RL suites; the PR's 7 new tests all fail on main (real tests).
build_pit_evidence_slice raise: 0 production callers; stock-pool-gate branch not reachable from pipeline. Env arrays identical main vs PR on real data (48/48). NaN prices never reach reward.
Requested change: bound the gap-session union to the panel date range (u0 gaps cover 1990–2026; 2,337 gap dates not in the v7 panel ⇒ a quarantine-truncated panel + unfiltered gaps raises at construction instead of trimming the last signals). P3: regime feature fillna(0) drops the market move across a gap day.

## Findings
- R11-F01 [P1, CONFIRMED] `train_ppo_policy` (train_ppo.py; reached from `train-rl-agent` and `autopilot`, cli/v7_train.py:2418, :2783) has no quarantine `check_window` and no reward-end censoring; callers pass full predictions + full v7 panel (to 2026-08-05). Test `test_train_ppo_policy_refuses_quarantined_rewards` → DID NOT RAISE; trained on frozen-holdout returns and wrote status: passed.
- R11-F02 [P2, CONFIRMED, latent] reward_ablation_experiment.py:76 coerces unknown/non-numeric execution flags to False ("tradable"), defeating the env's strict flag check (DID NOT RAISE). Real 2022–25 flags have 0 NaN ⇒ no reported number affected.
- R11-F03 [P2, code reading] rl_pit_train_eval.py:601-602 null-baseline check defaults True when --random-baselines 0; research gate allows MaxDD 5pp worse (:606).
- R11-F04 [P2, code reading] services/quant_api/adapters/models.py:390-392 lists an unevaluated RL policy as status=ready, verdict=passed (UI rendering not verified).
- R11-F05 [P2, code reading] reward risk terms depend on NAV/peak/maxDD which are absent from the observation (pit_portfolio_env.py:662-690) ⇒ non-Markov; agent can only learn unconditional de-grossing (explains round-24 "89% de-grossing").
- R11-F06 [P2, PLAUSIBLE] training replays one deterministic episode ~250–630× with a t/T time feature ⇒ path memorisation; exploratory fix did not change verdict.
- R11-F07 [P3] runtime/paper/forward/C_rl/targets_latest.csv (2026-05-07) still on disk; tombstone never run; only consumer is manual `intraday_dot_decisions --book C_rl`.
- R11-F08 [P3] scripts/rl_train_eval_2026.py and rl_strict_eval.py are dead (pass max_turnover ⇒ TypeError; target burned holdout).
- R11-F09 [P3] PR #156 nits above.

## RL cannot reach the order path
No PPO.load/policy.zip consumer in src/ or services/ (only read-only discovery in models adapter); no RL command in job allowlists or cron; forward exporter tombstoned; no enable flag exists at all (future flag must require a verdict artifact).

## Verified correct
repro/test_r11_adversarial.py 22/22 pass on main: dollar-ledger oracle on 12 random markets (NAV, cash, frozen/limit constraints, fees from cash, drift), observation invariance to future-price shocks (6 markets), reward-end censoring (4 markets). Clock T / T+1 / T+1→T+2 confirmed. DEF-036, holding period, round-27 fixes, PR #130 not regressed.

## Literature
Potential-based shaping (Ng 1999): round-22 rejection holds. Differential Sharpe (Moody & Saffell 2001): rejection incomplete — differential IR on excess keeps zero-action at 0. EIIE (Jiang 2017): costs inside log-return reward + previous weights fed back (repo does both). FinRL executes at the observed close and its portfolio env reward lacks costs (repo stricter). Henderson 2018 / Agarwal 2021 applied. Unfetched: FinRL reward in README, arXiv 2412.18563 abstract only, EIIE PDF unparsable.

## Not checked
Full suite on PR branch; strict-simulator replay (moot); exploratory grid 14/60.
