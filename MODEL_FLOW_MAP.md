# MODEL_FLOW_MAP — 模型数据流图（Phase 1）

> 只画**生产路径**与其 contract；旁路/废弃路径见 `ARCHITECTURE_AUDIT.md` §2。
> 每个节点给出：代码位置 → 输入 → 输出 artifact → 契约（contract）。

```
[1] 数据采集 providers (src/quantagent/data/providers/*, ~25 个)
      TickFlow(仅日线) / QLib(分钟) / akshare(财务·资金·宏观) / tushare / baostock
      └─ MultiSourceDataRouter 容错路由
          ↓  contract: available_at 必须存在; DataManifest 伴随每个 silver/gold 表
[2] silver 层  runtime/data/v7/silver/
      market_panel/market_panel.parquet   (symbol, trade_date, OHLCV, is_suspended/is_st/is_limit_*)
      sector_map/sector_map.parquet       (⚠ current-snapshot, survivorship 已知)
      fundamentals / valuation / disclosures (PIT 四键)
          ↓
[3] 因子层
      factors/alpha101.py  → alpha001..101 (向量化, workers= 并行)
      factors/cicc_ashare80.py → alpha102..181
      factors/factor_synthesis.py + llm_factor_proposer.py
        → synth_* (LLM/GP 产出, 限 factors/expr.py DSL, 永不写自由 Python)
        → 验收: tradability-aware IC, ICIR≥0.2, 去相关, --train-end 洁净截断
          ↓
[4] gold 训练集  data/dataset_builder/v7_training_dataset.py (`build-training-dataset-v7`)
      输出: runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_*.parquet
      labels: forward_return_{1,5,20,60,120,126}d + forward_tradable_return_*d
      contract: feature_schema.json (feature_version + schema_hash);
                --expected-feature-schema 钉死列集; 禁 synthetic fallback;
                board-aware 涨跌停旗标强制重推 (main10/创业科创20/北交30/ST5)
          ↓
[5] 模型层 — 生产 = FT-Transformer 三 sleeve
      架构: models/ft_transformer.py (per-feature tokenizer + missing-mask embed
             + [CLS] + pre-norm Transformer + per-horizon linear heads)
      训练: training/ft_transformer_trainer.py 经 cli/v8_deep.py `train-v8-deep`
      生产参数 (run_v89_plus7_retrain / closed_loop step4):
             d_token 256, n_blocks 6, n_heads 8, attn/ffn dropout 0.25,
             batch 8192, dates_per_step 1, lr 5e-4, wd 1e-3, epochs≤80,
             early-stop 8, embargo 30d, train 2018-01-02→2024-06-30,
             cross-sectional rank norm + label norm, AMP, --require-gpu
      loss: Huber + listwise top-K softmax rank loss (per trade_date 截面)
      sleeves: short_5d / mid_5d_30d / long_30d_120d (training/horizon_models.py,
               per-sleeve feature whitelist)
      输出: <RETRAIN>/{sleeve}/predictions.parquet + checkpoint + schema + metrics
          ↓
[6] Blend → composite_score   ⚠ 此处存在双轨 (见 AUDIT §5.3):
      a) 脚本默认: run_v8_deep_sweep.blend() = 0.30/0.45/0.25 三 sleeve 加权平均
      b) 生产实际 (PRODUCTION_CONFIG.json): short+mid 两 sleeve per-date
         cross-sectional rank 相加, 丢 long sleeve —— 来自
         scripts/ensemble_weight_search.py 在 validation(2024-08..2025-08) 上的胜者
      输出: ensemble_composite.parquet (trade_date, symbol, composite_score)
          ↓
[7] 组合/风控层
      risk/decision_chain.py: 15-gate (ST/停牌/涨跌停排除, sector pool, 流动性,
        趋势质量) → top-K 等权 target_weights; 上限: 单票 5%, 单行业 30%
      ensemble/strict_policy_search.py: regime meta-policy
        (bull/neutral_up/neutral_down/bear/crisis × horizon 权重 × gross scale),
        直接以严格回测为目标函数
          ↓
[8] 唯一可信评测  scripts/baseline_protocol.py  ★ variant C ★
      C = flags ON + eligible ranking(排除停牌/ST/涨停封板) + t+1 fill
      引擎: backtest/strict_v8.py → backtest/ashare_execution_simulator.py
        (T+1, lot 100, 涨跌停不成交, ST 禁买, 成交量参与≤10%, slippage 8bps,
         佣金/印花/过户, FIFO round-trip PnL, 全审计)
      benchmark: 无摩擦等权全A
      beta 分解: backtest/beta_decomposition.py (Jensen alpha vs 全A/沪深300)
      输出: <dir>/backtest/{metrics.json, nav.csv} (UI 可发现)
          ↓
[9] UI  services/quant_api (indexer/adapters) + apps/quant-ui
```

## 模型三层级速查（防混淆）

| 层级 | 文件 | 是否训练 | 地位 |
|---|---|---|---|
| **FT-Transformer** | `models/ft_transformer.py` + `training/ft_transformer_trainer.py` | ✅ GPU | **生产模型** |
| MLP `V7DeepAlphaTrainer` | `training/v7_deep_trainer.py` | ✅（小） | 基建脚手架；模型本身已证无独立 edge；其 `run_walk_forward_deep_training`（schema-locked 折叠）仍是有用组件 |
| 启发式 towers | `models/v7_deep_alpha.py`, `models/v7_multi_horizon.py` | ❌ 未训练 | agentic V7 主题管线的降级 baseline，**不是模型** |

## 当前生产数字（provenance 必须一起引用）

| 读数 | artifact | 口径 |
|---|---|---|
| CAGR **+38.6%** / Calmar 3.26 / maxDD 11.85% | `v89_closed_loop/ensemble_search_plus7/winner_heldout/backtest/metrics.json` | variant C, top10, 2-sleeve rank blend, holdout 2025-09-01..2026-05-15 |
| 同窗 3-sleeve 平均 blend | `realtest_plus7_holdout_top10/backtest/metrics.json` | **仅 +8.3%** |
| 同窗 2-sleeve 平均 blend | `retrain_plus7_.../realtest_2sleeve_holdout_top10/...` | **+14.5%** |
| 同窗 benchmark 等权全A | 同上 metrics 内 | **+19.9%** |

⚠ 解读：holdout 已被多方案反复评测（见 AUDIT §5.1），38.6% 的稳健性未经 multiple-testing 校正，是 Phase 2 的第一优先审计对象。在此之前**不得**以 38.6% 作为改进 baseline 的对照真值。
