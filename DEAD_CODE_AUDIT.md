# DEAD_CODE_AUDIT — 死代码与误导路径审计（Stage C / Phase 3）

> 2026-07-03，分支 `robustness-mission`。依赖图方法：对 `scripts/` 全部 155 个脚本做全仓交叉引用扫描（scripts/src/tests/docs/services/README/AGENTS/systemd，排除 runtime/venv/apps）；对模型类做 import 追踪。原始数据在本文档表格内。
> 原则（AGENTS.md）：删除前必须证明未被 imports/CLI/tests/README/AGENTS/docs 引用；不确定 → 先 deprecate。

## 1. 脚本层：155 个中 **95 个零引用（orphan）**，60 个被引用

### 1a. 被引用的核心链（保留，勿动）

- 生产/评测：`baseline_protocol.py`（3 引用）、`materialize_production_composite.py`（closed loop）、`run_v89_closed_loop.sh`、`evaluate_discovered_factors.py`、`rl_pit_train_eval.py`。
- 定时管线（systemd 单元引用）：`run_daily_pipeline.sh` / `run_weekly_pipeline.sh` / `run_monthly_pipeline.sh` / `run_forward_daily.sh` / `daily_health_check.sh` + 它们调用的 fetch_* / forward_* / update_market_panel_daily / enrich_market_panel 等 ~20 个。
- stage10 日频（`stage10_daily.sh` 引用 4 个）、UI（run_quant_ui*、quant_ui_api_smoke）。
- sweep 入口（docs 引用）：`run_v8_deep_sweep.py`、`run_v8_sweep.py`、两个 tmux launcher。

### 1b. orphan 95 个的分类（详表可由扫描脚本再生）

| 类别 | 数量级 | 代表 | 处置建议 |
|---|---|---|---|
| 做T/intraday 一次性研究（结论=无 edge，已 REJECT） | ~15 | `intraday_dot_ev_*`, `dot_overlay_backtest.py`, `tickflow_intraday_factor_combo_train.py`, `selective_dot_walkforward.py` | **deprecate 标注**；六个月后无人调用再删 |
| stage1–13 一次性研究脚本（各自写简化回测） | ~25 | `stage1_base_book_audit.py`, `stage3a/3b/4/5/6/7/8/9_*.py`, `stage11–13_*.py` | deprecate 标注 + 引用 census；其中 stage6_classical_walkforward 链（引用 stage6_full_walkforward / stage6_policy_search）保留待 Phase 6 复用评估 |
| rankfix 取证（使命完成） | 4 | `rankfix_*.py` | deprecate；保留（是 v8.8 事故的证据链） |
| 已被 P4/新流程接管的搜索 | 3 | `ensemble_weight_search.py`, `factor_combo_search.py`, `regime_strategy_search.py` | **已加固**（必填窗口参数 + guard），保留为受控工具 |
| 旧 build/enrich 变体 | ~8 | `build_intraday_panel_2026.py` vs `_full.py` vs `build_intraday_minute_panel.py`, `enrich_market_panel_boardfix.py` | 合并/删除候选（Phase 7），先 deprecate |
| 杂项一次性实验 | ~40 | `mix_weight_experiment.py`(实被 cli/v8.py 引用→非 orphan 复查), `board_chase_eval.py`, `paper_replay_2026.py`, `overlay_regime_split.py` 等 | deprecate 标注批处理 |

**注意**：orphan ≠ 可安全删除 —— 部分靠人工在 tmux 手跑（无引用痕迹）。因此本阶段**零删除**，统一走"deprecate 标注 → 观察期 → PRUNE_PLAN 执行"。

## 2. 模型类层（import 追踪结果）

| 对象 | 引用者 | 判定 | 已做 |
|---|---|---|---|
| `models/v7_deep_alpha.py`（启发式 towers） | `v7_classical_alpha.py`、`v7_deep_trainer.py`、`services/v7_pipeline_service.py` | **活代码但严重误导命名**（agentic V7 fallback） | ✅ docstring 顶部加 STATUS WARNING（非生产、未训练） |
| `models/v7_multi_horizon.py` | `v7/agent_contracts.py`、`v7_pipeline_service.py` | 同上 | ✅ 同上 |
| `training/v8_pipeline.py`（GA 管线） | 仅 `cli/v8.py` | legacy，非生产 | ✅ STATUS note；CLI 命令保留（Phase 7 再议摘除） |
| `training/v7_deep_trainer.py` MLP | trainer 被 CLI v7 族用；`run_walk_forward_deep_training` 是**有用基建** | 保留；模型头已知无独立 edge | 文档已载（MODEL_FLOW_MAP） |
| `run_v8_deep_sweep.blend()` | 唯一调用方 closed loop **已切走** | sweep 场景仍用 | 保留，closed loop 不再调用 |
| `scripts/forward_daily_inference.py` | systemd forward 管线 **仍在用** | **高危**：钉死 v8.8 corrupted 代际 | ✅ 启动时 stderr 大字警告（P6 修复前禁作证据） |

## 3. CLI 层（22 模块）

- 生产用：`v8_deep.py`、`v7_data.py`、部分 `v7_backtest.py` / `v7_train.py`。
- `v8.py` 引用 legacy v8_pipeline + 多个研究脚本 → 待 Phase 7 分流。
- 其余（v7_bond/v7_policy/v7_evidence/v8_gated/v8_intraday/…）被 `cli/__init__.py` 注册、tests 部分覆盖 —— 未见死命令的直接证据，**未列删除**；逐命令 `--help`+调用扫描排 Phase 7。

## 4. 本轮已实施的最小加固（本 commit）

1. 三个污染源搜索脚本：OOS 窗口参数**必填化**（去掉 2025-09-01 危险默认），`regime_strategy_search.py` 额外加显式 quarantine guard（它绕过 bp.evaluate 直调 strict 引擎）。
2. 两个启发式模型 + GA 管线：STATUS/deprecation 头。
3. `forward_daily_inference.py`：运行时警告（v8.8 钉死 + 特征不可复现）。

## 5. 残余风险

- **27 个脚本直调 `run_strict_backtest_v8`**（绕过 guard）。提议：在 strict_v8 加"软警告"（stderr 一行，不改行为）——因触及可信评测器文件，**待用户批准**（PRUNE_PLAN §P-G）。
- orphan 判定基于静态引用，人工手跑无法检测 —— 故全部走 deprecate-first。
