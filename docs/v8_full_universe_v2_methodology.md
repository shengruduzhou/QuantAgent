# 全宇宙 v2 — 抗过拟合方法论

## 背景

v1 全宇宙训练 (3658 sym, 80ep/256/6) 失败：mid/long 模型在 epoch-0 即达最佳
val_loss（随机权重最优 = 完全没学到 OOS 信号），门控后年化仅 8-12%，远不如
top-500 的 51%。但 top-500 的 51% 是**前视偏差**产物——universe 用 2024 年成交额
选股，而 OOS 窗口正是 2024 年。

用户洞察（正确）：只在 top-k 上训练，真实交易时模型没见过垃圾票分布，会买到
没学过的烂票，没有部署价值。**正确做法是全宇宙训练得到通用权重 + PIT 选股。**

## 四项改造

### A. 按日横截面标准化特征 (`_cross_sectional_normalize`)
每个 trade_date 内把每个特征转成当日横截面 rank∈[-0.5,0.5]（或 z-score）。
- **为什么**：原 trainer 用单一全局均值/方差，茅台(1300)和微盘(3)同参数标准化，
  模型只能学绝对水平。全宇宙跨度太大 → mid/long epoch-0 过拟合。
- **leak-free**：date t 的变换只用 date-t 的横截面，t 时点完全已知。
- rank 比 z-score 更稳健（不受微盘极端特征值影响）。

### B. 按日标签 winsorize + z-score (`_normalize_label_per_date`)
训练标签每日 winsorize(1%/99%) + z-score（仅训练集，OOS 标签保持原始供回测）。
- **为什么**：微盘连板 +50% 的极端 forward return 主导 Huber loss → 模型追垃圾票。
- 配合已有的 per-date listwise rank loss，组合目标更干净。

### C. PIT 滚动可交易 universe (`decision_chain.liquidity_window`)
训练用全宇宙（模型学完整分布、学会避开烂票），**选股/回测只在 PIT 可交易集**内
选 top-K：trailing-60d 成交额 + 非 ST + 非停牌（决策链 15-gate 已实现，窗口现可配）。
- 消除前视：date t 的可交易判定只用 t 之前的滚动数据。
- 训练全宇宙 → 打分全宇宙 → PIT 过滤 → top-K → 15-gate → 下单。这就是用户要的
  "通用权重 + 选股策略" 架构。

### D. 正则化 widened 模型（保持 256/6）
attention_dropout/ffn_dropout 0.10 → 0.25，weight_decay 1e-4 → 1e-3，
early-stop patience 10 → 8。用更强正则而非缩小模型来防过拟合。

## 用满数据

v1 只用到 2024-12（浪费 2025-2026）。v2:
- 训练: 2018-01-02 → 2024-06-30 (6.5 年)
- embargo: 30 bday
- OOS: 2024-08 → 2026-05 (~1.75 年, 含最新 regime)

## 工程修复

- **OOM (SIGKILL, 62GB RSS)**：原 `pd.read_parquet` 读全部 246 列×7.3M 行
  (float64 ≈14GB) + 归一化 `df.copy()` 翻倍 → 爆 62GB 系统内存。
  修复：(1) 只读该 horizon 需要的列 (~64) (2) 特征降 float32
  (3) 归一化原地改写不整帧复制。RSS 62GB → ~30GB。
- **GPU OOM**：`dates_per_step=1`（每步 ~3658 行而非 8 天×3658）。

## CLI

```bash
MAX_EPOCHS=80 D_TOKEN=256 N_BLOCKS=6 BATCH_SIZE=8192 DATES_PER_STEP=1 \
CROSS_SECTIONAL_NORM=rank LABEL_NORM=1 \
ATTENTION_DROPOUT=0.25 FFN_DROPOUT=0.25 WEIGHT_DECAY=0.001 EARLY_STOP_PATIENCE=8 \
TRAIN_START=2018-01-02 TRAIN_END=2024-06-30 TEST_END=2026-05-15 EMBARGO=30 \
UNIVERSE_FILE=runtime/reports/v8/pipeline/universe_full.txt \
bash scripts/launch_v8_deep_sweep_tmux.sh
```

## 验收标准

1. mid/long 不再 epoch-0 过拟合（val_loss 应持续下降数个 epoch）
2. 全宇宙门控年化跑赢等权基准 (6%) 且回撤 < v1 的 23-34%
3. 该权重是**无前视、可部署**的真实数字（不像 top-500 的 51% 虚高）
