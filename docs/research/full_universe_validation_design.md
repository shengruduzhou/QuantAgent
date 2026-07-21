# full_universe_validation_design — 下一代全宇宙验证协议（H-030 §9，**设计完成·尚未执行**）

> 本文只**准备**协议，不执行任何训练/评测。执行的前提 = Track U0 数据门全过（`FULL_UNIVERSE_DATA_READY`）。
> **绝对边界**：本协议与冻结的 S1–S4 / 2026-11-11 首读**完全隔离**；Track-F 的 FRESH 窗**不得**被当作新全宇宙模型的干净盲窗复用（该窗将被 S1–S4 首读消费，对新模型不再干净）。

## 1. 模型面（冻结，不扩张）

- **基线** = 现行 LightGBM 截面 ranker（EXP-022 参数逐项沿用）。
- **候选** = 现行 FT-Transformer 三 sleeve 架构（short_5d / mid_5d_30d / long_30d_120d，d_token 256 / n_blocks 6 / n_heads 8 / dropout 0.25 等生产参数原样）。
- **禁止**：新模型族（H-027 已判 REJECT_ALL_ALTERNATIVE_MODELS）、新因子挖掘（H-025/026 已判特征线收敛）、架构搜索、损失函数搜索。本协议只回答一个问题：**在真正的全宇宙 PIT 数据上，现有两个模型面的表现如何**——不是"哪个新模型更好"。

## 2. 数据与标签

- 宇宙 = `daily_full_universe_eligible.parquet`（逐日 PIT eligible；含 STAR/北交/2020 后新股/退市名）。
- **新股 60 交易日内默认 ineligible**（H-028 预注册规则，防新股首日效应制造 phantom breadth）。
- 标签 = delay-1 executable（与冻结链一致：`close(t+1+h)/close(t+1)−1`，t+1 涨停/停牌入场不可行剔除）。
- 执行规则 = variant-C 全同（T+1、涨跌停、停牌、ST、lot、参与≤10%、成本+slippage、√冲击诊断叠加）。

## 3. 折叠与泄漏控制

- **purged walk-forward**，expanding；**embargo ≥ 最长标签视界**（120/126d 标签 ⇒ embargo ≥126 交易日，修正历史 30d 的次级泄漏风险 —— EVALUATION_PROTOCOL_V2 §4a 已列此项）。
- 严格 OOF：任何折的预测只来自该折之前的数据训练的模型。
- 全部预处理（winsor/scale/impute/中性化/选择）**折内拟合**。
- 宇宙变更本身是 PIT 的：每日 eligible 由当日可得信息决定，不得用未来成分。

## 4. 全新盲钟（与 Track-F 完全分离）

配置冻结后，另起独立盲钟：

| 相位 | 长度 | 用途 |
|---|---|---|
| **Phase A** | 前 126 个交易日 | 第一次一次性读数 |
| **Phase B** | 其后 126 个交易日 | 确认读数（Phase A 结论的独立复核） |

- 起点 = 全宇宙配置冻结日之后的第一个交易日（届时按真实交易日历程序化计算，禁日历日近似）。
- 每相位每配置**只读一次**，预注册于独立配置文件（不复用 `configs/preregistered_evals.json`，避免与 S1–S4 首读混淆）。
- Track-F FRESH 窗（2026-05-19→2026-11-11）**明确不可复用**。

## 5. 验收门（沿用 ACCEPTANCE_RULES，全 AND）

零 PIT 违规 · PBO ≤0.25 · DSR 按累计 N 校正显著 · 亏损折比 ≤40% · 15bps 正超额 · 25bps 衰减 ≤40% · 折中位 maxDD ≤20% / 最差 ≤30% · 单行业 ≤30% · 单票 ≤10% · 容量 1M/10M 通过（√冲击模型已就位，30M 需重新论证）· beta/Jensen alpha 并报 · 执行完整性无失败。

## 6. 累计试验数

新宇宙 = 新族，但 DSR 校正**不清零**：沿用全库累计 N（H-030 时点 115）作为下界，新族试验数在其上累加并单独记账。理由：同一研究者、同一数据生态下的搜索历史仍构成多重检验负担。

## 7. 先决条件清单（执行前必须全绿）

1. `FULL_UNIVERSE_DATA_READY`（缺失 symbol=0、退市后行=0、重复=0、未发布收盘=0、ST/停牌/涨跌停旗标按板别重建）；
2. 全宇宙 benchmark 重算（等权 eligible 全A —— 与固定 cohort 的旧 benchmark **不可比**，旧数字不迁移）；
3. 冲击/容量模型在新宇宙上重新标定；
4. 独立盲钟预注册文件落盘并提交；
5. Track-F S1–S4 首读已完成（避免两条评测线在同一时期争夺注意力与算力）。
