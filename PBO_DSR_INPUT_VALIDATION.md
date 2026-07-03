# PBO_DSR_INPUT_VALIDATION — 输入校验（Phase 2.5）

> 全部检查以命令实测为准（pyarrow footer 元数据 + 单列读取，无全表载入）。

## 1. 校验结果

| # | 检查项 | 结果 | 证据 |
|---|---|---|---|
| 1 | 27 候选共享同一 validation 窗 | ✅ | 全部由同一次 `ensemble_weight_search.py` 运行生成（同 argv：val 2024-08-28→2025-08-31）；打分帧全部派生自同一 `ranked` 母帧 |
| 2 | 日索引对齐 | ✅ | 27 文件 row_count 全等（1,410,354）；首/末文件 `trade_date` 列逐行 `equals()` = True；val 窗内 257 个唯一交易日 |
| 3 | 候选文件不含 2025-09-01+？ | ⚠ **文件层面含**（打分行至 2026-05-07——原搜索复用同一帧做赢家 heldout 复跑所致）。**处理：分析读取时硬截断 `trade_date ≤ 2025-08-31`，且重建回测的 panel 同样截断 ≤2025-08-31，NAV 不产生任何隔离窗日期。** 这是显式截断，非静默近似 |
| 4 | 输入不含 final-holdout 派生的选择元数据 | ✅（带注） | 打分帧只有 (date,symbol,score)；`summary.json` 内含赢家 heldout 块 —— 该块**不进入**分析输入（只取 27 个 val 行与 weights/k） |
| 5 | 日收益有限性 | ✅ | 打分列 null_count=0（27 文件全查）；重建日收益在脚本内断言 `np.isfinite` 全真 |
| 6 | NAV 基点可比 | ✅ | 重建统一 `initial_cash=1,000,000`、slippage 8bps、variant C 配置——与原搜索 `bp.evaluate` 参数一致 |
| 7 | 无重复候选 | ✅ | 27 个 (weights, top_k) 组合唯一（9 权重 × 3 k） |

## 2. 关键事实：候选日收益不在盘上 → 确定性重建

原搜索只落盘汇总指标（`summary.json`），NAV 序列丢弃。PBO/CSCV 需要 T×27 日收益矩阵，因此：

- **重建 = 原评测的一比一重放**：同 27 个打分帧、同 top_k、同 eligible+delay1 规则（`bp._target_weights`）、同 strict 引擎（`run_strict_backtest_v8`，无随机源，确定性）、同成本参数。
- **不是新搜索**：不新增候选、不改窗口、不做任何选择；产物只用于统计诊断。
- **与原运行的两处受控偏差**（quarantine 纯净化，显式声明）：
  1. 原运行 panel 缓冲到 `end+10d`（→2025-09-10），末尾 delay-1 信号可在 2025-09-01 成交；重建把 panel 与信号都截断在 2025-08-31，隔离窗零接触。
  2. 因此重建窗比原运行短 ~1 个执行日。
- **保真校验（fidelity gate）**：重建的 27 个 val CAGR 与 `summary.json` 逐一对照，容忍度 |Δ| ≤ 2.0pp（预期主要来自上述末日截断）；超限则停止并调查，不得带病出数。

## 3. 资源核算（Phase 2.5 约束 ≤16 GiB）

- panel 读取：pyarrow `filters=[trade_date ∈ [2024-08-14, 2025-08-31]]` + 13 列裁剪 → ~1.4M 行，~0.2 GiB。
- 打分帧读取：逐候选、列裁剪+日期过滤（~0.87M 行/候选），用后即弃。
- 27 次 strict 回测串行；日收益矩阵 245×27 ≈ 50 KB。
- 脚本记录 peak RSS 并写入结果；预算 < 4 GiB。

## 4. 判定

**输入 VALID（附第 3 项显式截断声明）。** 可以进入 PBO/DSR 计算，方法与局限见 `PBO_DSR_ANALYSIS.md`。
