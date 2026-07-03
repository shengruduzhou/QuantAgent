# WALK_FORWARD_PROTOCOL_H008 — C3+EMA challenger 的走式验证协议（规格，未开跑）

> 状态：**协议定稿，等待 GPU 授权**（预估超出"多小时训练"红线，按规矩先批后跑）。
> 目的：在**非选择窗**上比较 blend 家族（C1/C2/C3/C3+EMA×3），消除"在自己选择窗上评自己"的结构偏向（EXP-001 的根本局限）。

## 1. 折叠定义（expanding，每折重训 sleeve）

| 折 | train 窗 | embargo 间隔 | OOS 预测窗 | 重训需求 |
|---|---|---|---|---|
| F1 | 2018-01-02 → 2022-12-30 | ≈126 交易日 | 2023-07-03 → 2023-12-29 | short+mid+long |
| F2 | 2018-01-02 → 2023-06-30 | 同 | 2024-01-02 → 2024-06-28 | short+mid+long |
| F3 | 2018-01-02 → 2023-12-29 | 同 | 2024-07-01 → 2024-12-31 | short+mid+long |
| F4 | 2018-01-02 → 2024-06-30 | 同 | 2025-01-02 → 2025-08-29 | **复用 retrain_plus7_20260620_0300（零 GPU）** |

- 拼接 OOS 总跨度 2023-07→2025-08（~26 个月，含 2024 年初微盘崩塌与 2024Q3 拉升两种压力态），全程在隔离窗之前。
- **Embargo = 126 交易日**（最长 label 126d），修复现行 30d embargo 相对 120/126d label 不足的缺陷（LEAKAGE 跟进项）。train_end 与 OOS start 之间的间隔按交易日历落实（上表日期已按 ≥126 交易日排布，执行时以日历精确校验并写入 fold manifest）。

## 2. 数据与 schema 纪律

- 数据集：`training_dataset_alpha181_exec_v89_plus7clean.parquet`（sha256 已锁 `configs/production_blend.json`）。
- **已知披露（必须随结果引用）**：+7 synth 因子的验收用了 →2025-08 的信息；对 F1–F3 的训练存在因子选择 vantage 偏置。该偏置**对 C1/C2/C3/C3+EMA 全体候选同等作用**，因此 blend 间的**相对比较内部有效**；绝对量级不可引用（这也与隔离制一致）。可选加严版：改用 `v88_rankfix`（无 synth）重跑一遍作敏感性——列为 H-008b，不阻塞主问题。
- 每折 trainer 钉同一 feature schema（`--expected-feature-schema` 语义；折间 schema_hash 全等断言，run manifest 记录）。
- 超参 = 生产参数**冻结不动**（d_token 256 / n_blocks 6 / n_heads 8 / dropout 0.25 / lr 5e-4 / wd 1e-3 / batch 8192 / dates_per_step 1 / rank norm + label norm / judgment policy / seed 1729）⇒ 架构层零新试验。

## 3. 候选与选择规则（预注册，N=6；blend 族累计 N=50）

C1（0.30/0.45/0.25 平均）、C2（rank(1,1,0)）、C3（rank 中位）、C3+EMA α∈{0.3,0.5,0.7}。
**选择指标（先验声明）**：全部 gate 通过者中**折中位 CAGR 最高**；并列取最差折更优者。任何"看完结果再改指标"=作废。

## 4. 每折评测（variant-C 强制）

每折对 6 候选跑 `baseline_protocol` variant C（k=10，slippage 8bps；quarantine guard 在位）：CAGR / maxDD / Sharpe / turnover / vs 等权 bench 超额。拼接层：
- 6 候选 × 4 折矩阵 → 折块 CSCV-PBO（S=4 折块，C(4,2)=6 划分——粒度粗，主要证据为逐折分布本身）；
- 赢家 DSR：日收益拼接序列，N=50（累计台账）；
- 报告口径 = **逐折分布**（min/median/max/亏损折数），禁止只报拼接均值。

## 5. 验收门（ACCEPTANCE_RULES 全套适用）

challenger 采纳（更新 `configs/production_blend.json` + trust 升级为 `walk_forward_oos`）当且仅当：
1. 中位折 CAGR ≥ C2 的中位折，且最差折 ≥ C2 最差折；2. 亏损折 ≤ C2；3. 换手 ≤0.10/日（EMA 变体）；
4. 折内 maxDD 中位 ≤20%、最差 ≤30%；5. PBO（粗粒度）不劣于 0.5 且 DSR 不反对；6. 8→15bps 成本敏感衰减 ≤40%；
7. 行业集中 ≤30%（decision-chain 门不放松）；8. FRESH 窗保持零接触（最终裁决仍等 2026-11 首读）。

## 6. 资源预估与守卫

| 项 | 预估 | 上限/守卫 |
|---|---|---|
| GPU 重训 | 3 折 × 3 sleeve = 9 次 × 40–90 min ≈ **7–10 GPU·h**（夜间串行） | 单折 >3h 中止；`--require-gpu`；每折记录 `torch.cuda.max_memory_allocated()` |
| VRAM | 生产同参数（历史峰值 <24G；long sleeve `--train-micro-batch 1024`） | >22G 即降 micro-batch 重试一次，再超即中止 |
| RAM | 数据载入+训练 ~12–20G/折 | >48G 中止（监控脚本） |
| 磁盘 | 9×(checkpoint+preds) ≈ 3–5G | 单实验 ≤5G 预算内；中间物结束即清 |
| 评测 | 24 次 variant-C 回测 ≈ 24×15s ≈ 6 min CPU | — |
| 时钟 | 2 个夜间批次 | — |

**中止条件**：val loss NaN / schema hash 不等 / OOM 二次触发 / RAM 超限 / 折超时。中止即存诊断退出，不焖烧机器。

## 7. 精确命令（执行时逐折替换日期；全部入 EXPERIMENT_LEDGER）

```bash
# 每折每 sleeve（示例 F1 short）：
AI_quant_venv/bin/python3 -m quantagent.cli train-v8-deep --horizon-class short_5d \
  --dataset-path runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus7clean.parquet \
  --silver-panel-path runtime/data/v7/silver/market_panel/market_panel.parquet \
  --symbols-file runtime/data/v7/universe_v88_comma.txt \
  --train-start 2018-01-02 --train-end 2022-12-30 --test-end 2023-12-29 \
  --embargo-days 126 --top-k 30 --max-epochs 80 --batch-size 8192 \
  --d-token 256 --n-blocks 6 --n-heads 8 --dates-per-step 1 \
  --cross-sectional-norm rank --label-norm --attention-dropout 0.25 --ffn-dropout 0.25 \
  --weight-decay 0.001 --early-stopping-patience 8 --learning-rate 0.0005 \
  --feature-policy judgment --require-gpu \
  --output-dir runtime/reports/v89_closed_loop/wf_h008/F1/short_5d
# 折内评测驱动（待写）：scripts/analysis/exp008_walkforward_eval.py
#   —— 对每折 OOS 窗物化 6 候选（复用 materializer blend 函数 + EMA）→ variant-C → 汇总
```

## 8. 交付物

`runtime/reports/v89_closed_loop/wf_h008/{F1..F4}/…` + `wf_summary.json` + `EXPERIMENT_LEDGER` EXP-008 条目 + 若采纳则 `configs/production_blend.json` 变更（附完整 selection 档案）。

---
**待批事项：按上表启动 9 次 GPU 重训（约 7–10 GPU·h，夜间串行，守卫全开）。批准前本协议不产生任何 GPU 负载。**
