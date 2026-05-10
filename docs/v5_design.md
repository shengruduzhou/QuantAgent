# QuantAgent V5 设计 / V5 Design

> 面向大 A 市场的下一代 AI Quant OS。在 V4「三塔预测 + 因子组门控 + 结构化 agent + 约束优化 + 双模式组合」骨架之上，V5 不做加法堆叠，而是把每一层做厚、做活、做可治理。

---

## 0. V5 设计原则 / Design Principles

V4 已经把"研究—回测—执行准备"的骨架搭起来，但有四个结构性短板：

1. **模型不闭环**：`itransformer.py` / `alpha_transformer.py` / `PatchTST` 这些 V4 PDF 推荐的骨干虽然已经写好，但 `v4_multitower.py` 实际只用 `SimpleSequenceBackbone`；`factor_gate` 头部输出了却没有任何下游消费；`quantile_head` 输出了却没接 conformal。
2. **因子治理离线化**：`factors/lifecycle.py`、`evaluation.py`、`composite.py` 三套都是研究工具，没有进入"训练 → 推理 → 组合"的主链。
3. **agent 智能度参差**：`commodity_agent.py` 用硬编码 `COMMODITY_SECTOR_BETA` dict；`debate.py` 是简单 confidence 求和；`AgentRouter` 没有在线可信度反馈；`sleeve_allocator_agent.py` 是无逻辑包装。
4. **冗余与遗留**：`ashare_rules.py` 是 re-export 壳；`data/labels.py` 与 `quant_math/labels.py` 平行存在；`ic_analysis.py` 与 `factors/evaluation.py` 重复；`build_daily_features.py`、`hedge_leg.py`、`reject_reason.py`、`reconciliation.py`、`sleeve_allocator_agent.py` 处于"半 stub"状态；测试与 config 仍保留 V2/V3 残留。

V5 的核心信条是：

- **智能 ≠ 复杂**：单一深度堆叠不如多个被治理好的弱模型 + 在线反馈聪明。
- **减法优于加法**：先把 V4 重复模块合并、stub 删除、命名收敛，再谈新增。
- **闭环高于精度**：所有"研究级工具"必须接进训练 / 推理 / 组合 / 风控主链，否则下架。
- **可解释高于黑箱**：每个 alpha → view → weight 的链路都必须可追溯到 evidence、factor、model_version。
- **制度敏感高于学术指标**：A 股的板块差异化涨跌幅、T+1、最小申报、印花税、停复牌、ST 切换，是第一性约束，不能让模型层覆盖。

---

## 1. V5 总体架构 / Overall Architecture

V5 用四个认知层（Perception / Reasoning / Cognition / Action）封装 V4 现有模块，并显式把"治理 / 反馈 / 解释"作为横切关注点：

```text
┌──────────── 横切层 Governance & Feedback ─────────────┐
│  factor_lifecycle  ·  agent_reliability  ·  audit_chain │
│       conformal_calibrator  ·  ablation_runner          │
└─────────────────────────────────────────────────────────┘
        │                  │                  │
┌─ Perception 感知 ┐ ┌─ Reasoning 推理 ┐ ┌─ Cognition 认知 ┐ ┌─ Action 行动 ┐
│ FeatureStore    │ │ V5 MultiTower   │ │ Agent Committee │ │ Optimizer    │
│ Factor DAG      │→│ MoE Fusion      │→│ EvidenceRecord  │→│ Regime-aware │
│ EventStore      │ │ Conformal Head  │ │ AgentRouter+IR  │ │ HRP Fallback │
│ PIT Joiner      │ │ SSL Pretrain    │ │ BL Posterior    │ │ A-share Rules│
└─────────────────┘ └─────────────────┘ └─────────────────┘ └──────────────┘
                                                                   │
                                                          ┌────────▼────────┐
                                                          │  Execution      │
                                                          │  OrderManager   │
                                                          │  QMT Gateway    │
                                                          │  (dry-run gate) │
                                                          └─────────────────┘
```

数据流（trade date `t` 当日）：

```text
1. FeatureStore.snapshot(t)
     ↳ 价量 + 财务 PIT + 资金流 + 事件 token
2. FactorDAG.compute(t)
     ↳ alpha101 + cicc_hf + sector_rotation + fundamental + composite
3. V5MultiTower.forward(seq_x, snap_x, event_x)
     ↳ {alpha, direction, q_low, q_high, factor_gate, confidence, risk_score}
4. ConformalCalibrator.adjust(alpha, q_low, q_high)
     ↳ 校准后的预测区间
5. AgentRouter.route(evidence_records, universe)
     ↳ AgentView 列表（带 online IR 反馈调整后的 q / omega）
6. blend_alpha_and_views(alpha_post_conformal, views)
     ↳ BL 后验 alpha
7. solve_v5_portfolio(...) [regime-aware]
     ↳ target_weights / rejected / diagnostics
8. EventDrivenBacktester.run() 或 OrderManager.intend()
     ↳ paper trading / shadow portfolio / live (默认 dry-run)
9. PostTrade.attribute(weights, returns)
     ↳ 把 IC / IR 反馈给 factor_lifecycle 和 agent_reliability
```

---

## 2. 感知层 / Perception Layer

### 2.1 统一 FeatureStore（保留 V4，强化两点）

V4 的 `data/feature_store.py` + `point_in_time.py` + `event_store.py` + `universe.py` 已经形成可用的 PIT 流水线，**V5 不重构其骨架**，只做两件事：

1. **PIT 强一致性 audit**：每次 `FeatureStore.snapshot(t)` 写一条 JSONL，记录每个数据源的 `as_of_time / cutoff / row_count / null_ratio`。这条 audit 是"未来不会作弊"的物证，离线回测和实盘共用同一段代码路径。
2. **特征版本指纹**：每张 panel 输出附带 `feature_version = sha1(factor_set + preprocessing + universe + label_horizons)`。版本指纹随订单一并写入 `audit.py`，使任何一笔交易都能反向定位到当时的特征集合。

合并清单（V5 删除 / 内联）：

| 文件 | 处置 | 理由 |
|---|---|---|
| `data/build_daily_features.py` | 删除 | argparse 入口仅用 V3 路径；已被 `services/build_features_service.py` 覆盖 |
| `data/labels.py` | 删除函数，改为 import | 与 `quant_math/labels.py` 重复，统一由后者实现，data 层只做调用 |

### 2.2 Factor DAG 2.0

V4 已经有 `factors/registry.py` + `dag.py` + `lifecycle.py` + `governance.py` + `composite.py` + `evaluation.py` + `operators.py` + `preprocessing.py`。问题不在缺失，而在**没串成闭环**：

- `composite.py` 输出的加权复合因子没有进入 `services/build_features_service.py`；
- `lifecycle.py` 的 `FactorLifecycleReport` 没有被任何线上服务读取，因此降权 / 退役不会真正生效；
- `governance.py` 是静态白名单，不响应 lifecycle 状态。

V5 改造：

```text
FactorRegistry
   │  register(meta, compute_fn, group, dependencies)
   ▼
FactorDAG.topological(group)
   │  返回 (factor_name, fn) 序列
   ▼
FactorPipeline.run(panel, t)
   │  → 写 cache; 写 IC; 写 turnover; 写 capacity
   ▼
FactorLifecycleEngine.tick(t)   ◀── 每日一次
   │  滚动 RankIC / IC_decay / 相关性 / 拥挤度
   │  状态机：active → watch → degraded → retired
   ▼
FactorComposite.assemble(active_factors, learned_gate)
   │  权重 = softmax(learned_gate × ICIR × (1 - crowding))
   ▼
SnapshotTensor                     给 V5MultiTower snapshot 塔
```

**关键创新**：`learned_gate` 直接来自 `V5MultiTower.factor_gate` 头部（V4 已经输出，V5 第一次真正消费）。这让"模型告诉因子层用谁"成为闭环。

因子分组（沿用 V4，明确口径）：

- **price_volume**：alpha101 / 技术指标 / 实现波动率
- **micro_structure**：cicc_high_freq（分钟聚合）
- **breadth_rotation**：sector_rotation
- **fundamental**：valuation / quality / dupont / forensic_accounting / target_price
- **flow**：北向 / 融资融券 / ETF / 龙虎榜 / 主力流（来自 `ashare/fund_flow.py`）
- **event_policy**：来自 EventStore 的结构化事件 token

---

## 3. 推理层 / Reasoning Layer：V5MultiTower

### 3.1 V4 现状回顾

`v4_multitower.py` 的 `V4MultiTowerModel` 是一个 64-dim 三塔 + concat + 线性融合 + 7 头模型，结构合理但有四个缺陷：

1. 序列塔只能选 `SimpleSequenceBackbone`，`alpha_transformer.py` / `itransformer.py` 没有接进来。
2. 三塔融合是简单 `torch.cat`，缺少"按制度 / 风格学习权重"的能力。
3. `factor_gate` 输出，下游不消费。
4. `q_low / q_high` 输出，没有 conformal 校准。

### 3.2 V5 多塔模型：四点升级

```text
                        ┌────────────────────┐
                        │  Regime Embedding  │
                        │  (HMM state, vix,  │
                        │   breadth, turnover)│
                        └────────┬───────────┘
                                 │
        ┌─Sequence Backbone (configurable)─┐
seq_x ─▶│  SimpleSeq | iTransformer | PatchTST | TFT  │─▶ z_seq
        └────────────────────────────────────────────┘
        ┌─Snapshot Tower (TabularResNet)──┐
snap_x ▶│  factor groups + learned gate    │─▶ z_snap
        └──────────────────────────────────┘
        ┌─Event Tower (StructuredEvent)───┐
ev_x  ─▶│  policy / news / flow events     │─▶ z_event
        └──────────────────────────────────┘

                z_seq, z_snap, z_event, z_regime
                          │
                    MoE Fusion Gate
                          │
                  shared hidden (h)
                          │
       ┌──────────┬──────────┬──────────┬──────────┐
       ▼          ▼          ▼          ▼          ▼
   alpha_head  q_low/high  cls_head  gate_head  conf_head
                          │
                Conformal Calibrator (val)
                          │
                  calibrated alpha + PI
```

#### 升级 1：BackboneBase 接入多骨干

`models/backbone_base.py` 已经有 `BackboneSpec`，V5 把它扩成接口：

```python
class SequenceBackbone(Protocol):
    def forward(x: Tensor, mask: Tensor | None = None) -> Tensor: ...

# 注册表：
{
  "simple_seq": SimpleSequenceBackbone,
  "alpha_transformer": AlphaTransformerBackbone,
  "itransformer": iTransformerBackbone,
  "patchtst": PatchTSTBackbone,
}
```

V5 的 `V5MultiTowerConfig.sequence_backbone` 是字符串，按配置实例化。这把 V4 PDF 推荐却没有接通的三个骨干一次性激活，且不需要重写 `v4_multitower.py` 主体逻辑——只是把 `self.sequence_tower = SimpleSequenceBackbone(...)` 改为工厂调用。

#### 升级 2：MoE Fusion（Mixture-of-Experts 融合）

简单 cat 不能反映"震荡市更看资金流、趋势市更看价量、政策市更看事件"的事实。V5 把融合层改成轻量 MoE：

```python
# regime 信号 = [hmm_state, market_breadth, realized_vol, turnover]
gate_logits = MoEGate(z_regime)          # shape (B, num_experts)
gate_weights = softmax(gate_logits, dim=-1)
# 每个 expert 是一个独立的轻量融合头（hidden_dim -> hidden_dim）
experts = [Expert_i(cat(z_seq, z_snap, z_event)) for i in range(num_experts)]
h_fused = sum(gate_weights[:, i] * experts[i] for i in range(num_experts))
```

`num_experts` 默认 3，分别对应"价量主导 / 资金事件主导 / 基本面主导"。MoE Gate 是 2 层 MLP，参数量 < 1 万，不会显著增加训练开销。这比"一个超大单塔"更聪明、更易解释，因为可以直接看 `gate_weights` 在不同日期上的分布。

#### 升级 3：factor_gate 真正消费

`V5MultiTower.factor_gate_head` 输出向量维度 = 因子组数（price_volume / micro / breadth / fundamental / flow / event）。在推理时：

1. 模型层输出 `factor_gate ∈ (0, 1)^G`；
2. 该向量作为 `FactorComposite.assemble()` 的可学习先验权重；
3. 与 `lifecycle.py` 的滚动 ICIR / 拥挤度做 element-wise 调制；
4. 形成的复合因子重新进入下一日 snapshot 塔输入。

这是一个**自指闭环**：模型的 gate 决定下一日因子如何组合 → 新因子又作为下一日模型的输入。这正是 V4 PDF 中提到但未实现的 "differentiable gate"。

#### 升级 4：Conformal 校准头部

V4 的 `quant_math/conformal.py` 已经实现 split conformal 与 CQR，但没有任何调用方。V5 把它接到训练循环：

```text
train phase   : 训练 V5MultiTower 输出 alpha / q_low / q_high
val   phase   : 在 holdout 上计算 conformal residuals
                → ConformalCalibrator.fit(residuals)
infer phase   : alpha_calibrated, lower, upper = calibrator.attach_interval(...)
```

校准后的 `lower / upper` 提供两个下游能力：

1. **风险闸**：`(upper - lower) > threshold` 视作高不确定性，optimizer 降低其上限权重。
2. **位置规模**：`position_sizing` 用 1 / interval_width 而非 1 / variance 作为信号强度，对偏态友好。

#### 升级 5（可选）：自监督预训练（SSL）

A 股标注数据稀疏，但 OHLCV 与因子时间序列充裕。V5 加一段 SSL pretraining：

- **Masked Factor Reconstruction**（类 MAE）：随机 mask 15% 截面因子值，让 snapshot tower 重建。
- **Next-bar prediction**（类 BERT MLM）：mask 序列中一段，让 sequence tower 预测。
- Loss 加权进入主目标，权重 < 0.1。

SSL 本身不预测 alpha，但能让 backbone 学到稳健的截面表征，提升小样本场景（如行业切换初期）的鲁棒性。这是 Qlib / Microsoft 在 2023 之后的主线之一。

### 3.3 复合损失 v2

V4 的 `training/composite_loss.py` 已包含 rank + huber + cls + quantile + factor_gate + turnover + risk 七项。V5 加两项：

```text
L_total = λ1·rank_loss          # pairwise / listwise 排序
        + λ2·huber_loss         # 数值拟合
        + λ3·cls_loss           # 涨跌方向 / 三障碍标签
        + λ4·quantile_loss      # pinball loss for q_low / q_high
        + λ5·gate_loss          # 因子 gate 监督（用 ICIR 当软标签）
        + λ6·turnover_loss      # 换手惩罚
        + λ7·risk_loss          # 风格 / 行业 / beta 暴露惩罚
        + λ8·consistency_loss   # 多 horizon (1d/5d/20d) 排序一致性
        + λ9·ssl_loss           # masked factor reconstruction
```

`consistency_loss` 是 V5 新增的关键正则：要求同一只股票在 1d / 5d / 20d 上的 alpha 排序方向一致（用 Kendall tau 的可微近似）。这能压低"短期信号噪声、长期方向反向"的伪 alpha，对 A 股这种短周期噪声很大的市场尤其重要。

### 3.4 训练协议（强制 Purged + CPCV）

V4 已有 `quant_math/purged_cv.py`、`training/walk_forward.py`、`training/ablation_runner.py`，但 `train_v4_service.py` 只是 synthetic 演示。V5 把训练协议规范化：

| 阶段 | 协议 | 目的 |
|---|---|---|
| 模型选择 | Purged K-Fold (k=5, embargo=10d) | 防 label / feature 泄漏 |
| 超参 search | CPCV (combinatorial purged CV) | 多组合下计算 PBO |
| 实战测试 | Walk-Forward (rolling train/val/test) | 模拟真实推进 |
| 上线前 | Ablation Runner | 删掉每一层后看剩余 alpha |
| 上线后 | Live Conformal Recalibration | 覆盖率漂移触发重训 |

每次训练自动写一份报告：
- ICIR / Rank-ICIR / Newey-West t-stat
- PSR / DSR / PBO
- Ablation 矩阵：去掉 event / agent / gate / conformal / cost penalty 后的 OOS Sharpe

---

## 4. 认知层 / Cognition Layer：智能化的 Agent Committee

### 4.1 V4 现状回顾

V4 已有 9 个 agent（policy / flow / commodity / sector_rotation / financial_statement / sleeve_allocator / debate / arbitration / bl_views），统一通过 `EvidenceRecord → AgentRouter → AgentView → blend_alpha_and_views → BL posterior` 链路接入组合。

问题：

1. **commodity_agent** 用硬编码 `{"oil": {"oil_gas": 0.8}, ...}` 映射，不学习、不更新。
2. **AgentRouter** 用静态 `base_view_scale=0.03` 把所有 agent 一视同仁，没有"哪个 agent 历史上准"的反馈。
3. **debate** 简单加和 bull / bear confidence，不是后验。
4. **sleeve_allocator_agent** 是无逻辑包装。
5. **financial_statement_agent** 输出 signal，但 main pipeline 没消费。

### 4.2 V5 Agent Committee 设计

V5 的 agent 哲学（与 V4 PDF 一致）："agent 不输出订单，只输出结构化证据"。V5 在这条线上做四点增强：

#### 增强 1：可学习的 sector-beta 映射

`commodity_agent.py` 的 `COMMODITY_SECTOR_BETA` 改造为：

```text
config/commodity_beta.yaml:
  oil:
    oil_gas: 0.80          # 先验
    transport: -0.30
    chemical: 0.50
  copper:
    nonferrous: 0.85
    new_energy: 0.30
  ...
```

并新增 `AgentRouter.update_beta(commodity, sector, realized_correlation, eta=0.05)`：每月把当月商品收益与板块收益的相关系数对先验做指数加权更新。这把"硬编码常识"变成"先验 + 在线学习"。同样的做法应用到 `policy_agent.py` 的政策受益板块映射。

#### 增强 2：Agent Reliability 在线追踪

V5 在 `agents/agent_router.py` 增加：

```text
AgentReliability
├── per-agent rolling IC（agent 给出 view 后该 view 在 5d / 20d 的命中率）
├── per-agent decay（最近 60 个交易日加权）
└── reliability score → 调制 AgentRouter.base_view_scale
```

`base_view_scale` 不再是 0.03 常数，而是 `0.03 × reliability[agent_name]`。这意味着：

- policy agent 在政策密集期表现好 → `reliability ↑` → 政策视图 q / 1-omega ↑
- commodity agent 在商品平静期表现差 → `reliability ↓` → 商品视图被自动压低

实现上只需要 `bl_views.py` 在生成订单后保留 `view_id → expected_return`，事后 T+5 / T+20 写实际收益 → 调用 `update_reliability`。

#### 增强 3：Debate 升级为 Bayesian Arbitration

`agents/debate.py` 当前实现是 `score = sum(bull_confidence) - sum(bear_confidence)`。V5 升级：

```text
posterior_P(up | evidence) = sigmoid(
    log_prior(symbol)
  + Σ_i log[ L(evidence_i | up) / L(evidence_i | down) ]
)

L(evidence | up)   = reliability[agent_i] × evidence_quality_i × confidence_i
L(evidence | down) = (1 - reliability[agent_i]) × ...
```

输出是后验概率 ∈ (0, 1)，而非"投票差"。这让多 agent 冲突时的输出更稳定，可直接当作 cls 头的软标签或 BL view 的 q。

#### 增强 4：合并冗余与新增 SentimentAgent

- **删除** `agents/sleeve_allocator_agent.py`：它只是 `SleeveAllocator.allocate` 的包装；`portfolio_build_service.py` 不会调用它。直接用 `portfolio/allocator.py`。
- **新增** `agents/sentiment_agent.py`（可降级版本）：
  - 输入：新闻 / 研报标题 + 摘要（从 `data/event_store.py` 取）。
  - 默认实现：金融领域词表 + 否定 / 加强词 + 截面 z-score。
  - 升级路径：`transformers` 可选依赖，加载中文金融 BERT。
  - 输出：每只股票一个 sentiment score → EvidenceRecord。
- **financial_statement_agent 接入主链**：`AgentRouter.route` 把它的输出转成 BL view，与价量 alpha 融合。

### 4.3 Evidence → View → Posterior 闭环

V5 agent committee 的运行步骤（每个交易日）：

```text
1. policy_agent.run(events_t)
2. flow_agent.run(fund_flow_t)
3. commodity_agent.run(commodity_returns_t)
4. sector_rotation_agent.run(panel_t)
5. financial_statement_agent.run(fundamentals_t)
6. sentiment_agent.run(news_t)
   → 全部输出 EvidenceRecord

7. AgentRouter.route(evidence_records, universe)
   → AgentView 列表（已用 reliability 调制 q / omega）

8. (optional) debate.bayesian_arbitrate(views)
   → 冲突视图融合

9. agent_views_to_bl_views → BL posterior alpha
```

`AgentView.symbols` / `q` / `omega` / `evidence` 的契约（沿用 V4 的 `views_schema.py`）不变，下游 BL 不需要任何改动。

---

## 5. 行动层 / Action Layer

### 5.1 Regime-aware Optimizer

V4 的 `quant_math/hmm_regime.py` 是 Gaussian HMM，但没有任何调用方。V5 在 `solve_v5_portfolio()` 入口处：

```text
regime = HMMRegimeDetector.predict(recent_returns)
risk_aversion = base_risk_aversion × {
    "trending_bull":   0.8,
    "trending_bear":   1.5,
    "high_vol":        1.6,
    "crash":           2.5,
    "recovery":        1.0,
}[regime]
```

`OptimizerConfig.risk_aversion` 不再是常量；同样地，`turnover_penalty` 在 crash 状态加倍。这把"市场制度"作为优化器的一类约束，**不需要重训模型**就能在 crash 中保守、trending 中放手。

### 5.2 HRP 作为二级 fallback

当前 `optimizer.py._solve_fallback` 是 "score / risk" 简单加权，缺乏聚类视角。V5 的 fallback 链：

```text
solve_v5_portfolio
├── try CVXPY 求解
├── except: try HRP (quant_math/hrp.py)
└── except: 简单 score-based fallback
```

HRP 在 covariance 病态、cvxpy 缺失、universe 过大时表现稳定，且不需要矩阵求逆，适合极端市场。

### 5.3 模式自动切换

V4 已有 `long_only_enhancement / hedged_alpha / market_neutral_placeholder` 三种模式。V5 加 `auto` 模式：

```text
breadth = (上涨股数 - 下跌股数) / 总股数
volatility_regime = HMM.predict()

if regime in {"crash", "high_vol"} and breadth < -0.3:
    mode = "hedged_alpha"
elif regime in {"trending_bull"} and breadth > 0.3:
    mode = "long_only_enhancement"
else:
    mode = "market_neutral_placeholder"  # 默认偏中性
```

无需对冲腿（A 股个人很多账户没有股指期货 / 融券权限）的用户可以禁用 hedged_alpha，回退到纯 long-only。

### 5.4 Tradability 与 A 股规则

`quant_math/ashare.py` 已有 board-aware 涨跌幅、suspension、T+1。V5 不重写，只做两件事：

1. **删除 `quant_math/ashare_rules.py`**：纯 re-export 壳，无逻辑。下游统一 `from quantagent.quant_math.ashare import ...`。
2. **把 2025-06-27 上交所 / 深交所主板 ST 涨跌幅由 5% 调整至 10% 的征求意见做成配置项**（V4 PDF 已经强调过）：`configs/ashare_rules.yaml` 里加一个 `st_main_board_limit: 0.05`，runtime 可改。

---

## 6. 横切层 / Cross-cutting

### 6.1 Audit Chain

每一笔订单 / target weight 都写一条 audit 记录：

```json
{
  "ts": "...",
  "symbol": "600519.SH",
  "target_weight": 0.034,
  "feature_version": "f5a2c1b",
  "model_version": "v5.2024-05-11",
  "view_ids": ["a1b2", "c3d4"],
  "regime": "trending_bull",
  "calibration_coverage": 0.91,
  "rejected_reason": null
}
```

实现：在 `execution/audit.py` 已有 `AuditLogger`，V5 把 `feature_version` / `model_version` / `view_ids` / `regime` 显式加进事件 schema。

### 6.2 Conformal 持续校准

`ConformalCalibrator` 在线追踪 90% 覆盖率：

- 实际覆盖 < 0.85：触发"模型可能漂移"告警，optimizer 自动降低当日仓位 30%。
- 实际覆盖 < 0.75：触发"重训"任务，写入 ops queue。

这一机制把不确定性量化变成"自我熔断"信号，不需要复杂监督学习。

### 6.3 Ablation 强制化

`training/ablation_runner.py` 在 V4 已有；V5 把它放进 `train-v5` CLI 的强制步骤。每次发布模型必须输出：

| 删除模块 | OOS Sharpe | 留存比例 |
|---|---|---|
| baseline (full) | 1.85 | 100% |
| - event tower | 1.62 | 87% |
| - factor gate | 1.71 | 92% |
| - conformal | 1.78 | 96% |
| - agent views | 1.50 | 81% |
| - cost penalty | 1.30 | 70% |
| - optimizer constraint | 0.95 | 51% |

只有 baseline 显著强于所有 ablation 时才允许上线。

---

## 7. 代码结构改造 / Refactoring Plan

### 7.1 删除（清理 stub 与重复）

| 文件 | 处置 | 理由 |
|---|---|---|
| `src/quantagent/quant_math/ashare_rules.py` | 删除 | 纯 re-export，无逻辑 |
| `src/quantagent/backtest/hedge_leg.py` | 删除 | 15 行 placeholder dataclass，无消费方 |
| `src/quantagent/backtest/reject_reason.py` | 删除 | 未使用 Enum；engine.py 用字符串 |
| `src/quantagent/execution/reconciliation.py` | 保留但内联 | 28 行；功能合并入 `qmt_gateway.py` 私有方法 |
| `src/quantagent/agents/sleeve_allocator_agent.py` | 删除 | 无逻辑包装；直接用 `portfolio/allocator.py` |
| `src/quantagent/data/build_daily_features.py` | 删除 | V3 argparse 入口，已被 services 覆盖 |
| `src/quantagent/data/labels.py` | 改为薄 import | 实际函数移入 `quant_math/labels.py` |
| `src/quantagent/quant_math/ic_analysis.py` | 删除 | 功能与 `factors/evaluation.py` 重复 |
| `tests/test_agents_sota.py` | 删除 | V2 / SOTA 旧基线 |
| `tests/test_ai_quant_os_v2.py` | 删除 | V2 编排测试 |
| `tests/test_quant_math_sota.py` | 删除 | V2 / V3 SOTA 旧基线 |
| `tests/test_v3_backtest_integration.py` | 删除 | V3 集成测试 |
| `configs/ai_quant_os_v2.yaml` | 删除 | V2 配置 |
| `cli.py: run_v3_backtest` | 删除命令 | V3 命令，主链已是 V4 / V5 |

### 7.2 合并（统一接口）

| 来源 | 目标 | 说明 |
|---|---|---|
| `data/labels.py` 的 `add_forward_return_labels` | `quant_math/labels.py` | 合入并保留兼容 alias |
| `strategy/position_sizing.py` | `quant_math/position_sizing.py` | 保留一份，strategy 调用 |
| `ic_analysis.dynamic_model_weights` | `factors/evaluation.py` | 因子评估统一入口 |

### 7.3 新增 / 重命名

| 文件 | 类型 | 内容 |
|---|---|---|
| `src/quantagent/models/v5_multitower.py` | 新增 | V5 多塔，BackboneBase 工厂 + MoE Fusion |
| `src/quantagent/models/backbone_registry.py` | 新增 | 序列骨干注册表（simple / alpha_transformer / itransformer / patchtst） |
| `src/quantagent/agents/sentiment_agent.py` | 新增 | 词表 + 截面 z-score 版本，预留 BERT 升级 |
| `src/quantagent/agents/agent_reliability.py` | 新增 | per-agent 滚动 IC 与 reliability score |
| `src/quantagent/quant_math/regime_aware_optimizer.py` | 新增 | 包装现有 optimizer，按 regime 调整 risk_aversion |
| `src/quantagent/training/conformal_calibrator.py` | 新增 | 把 `quant_math/conformal.py` 真正接到训练 / 推理 |
| `src/quantagent/training/factor_gate_loss.py` | 新增 | gate 头部监督（ICIR 软标签） |
| `configs/v5.default.yaml` | 新增 | V5 默认配置 |

### 7.4 V4 → V5 兼容策略

- **保留** 所有 V4 服务入口（`services/build_features_service.py` 等）。
- **V5MultiTower 与 V4MultiTower 并存**，旧路径不影响。
- 配置 `model_version: v4 | v5` 在 `services/train_v4_service.py` 切换。
- 默认 CLI 仍然指向 V4，加 `quantagent train-v5 / infer-v5 / backtest-v5` 命令。
- 等 V5 离线 + paper 双跑 4 周稳定后，再把 default 切到 V5。

---

## 8. 实施路线图 / Roadmap

| 里程碑 | 关键交付 | 估计人周 |
|---|---|---|
| M1 结构性减法 | 删除 / 合并清单完成；测试基线绿；CI 流转 | 1–2 |
| M2 因子闭环 | factor_gate → composite → snapshot 自指闭环；lifecycle 在线化 | 2–3 |
| M3 多骨干 + MoE | BackboneRegistry；V5MultiTower 跑通三种骨干 | 3–4 |
| M4 Conformal + SSL | 校准接进训练；SSL 预训练 pipeline；ablation runner 跑通 | 3–4 |
| M5 Agent 智能化 | reliability 在线更新；bayesian arbitration；sentiment_agent | 2–3 |
| M6 Regime-aware optimizer | hmm 接 risk_aversion；HRP fallback；mode auto switch | 2–3 |
| M7 Paper trading 完整闭环 | qmt dry-run loop；audit chain；daily reconciliation | 3–5 |
| M8 灰度 | shadow portfolio；小资金 canary；监控看板 | 4–6 |

V5 最小可上线版本（M1–M6）约 13–19 人周；完整版（含 M7–M8）约 20–30 人周。

---

## 9. 关键差异化（V4 vs V5 一表）

| 维度 | V4 | V5 |
|---|---|---|
| 序列骨干 | SimpleSequenceBackbone（硬编码） | 4 种可配置（simple / alpha_transformer / itransformer / patchtst） |
| 塔融合 | torch.cat + 线性 | MoE Gate（regime-conditional） |
| factor_gate | 输出但不消费 | 直接驱动 composite 因子权重 |
| 不确定性 | quantile 输出但不校准 | Conformal split / CQR，自我熔断 |
| 标签 | future returns（数值） | + triple barrier meta-label + cross-horizon consistency |
| SSL 预训练 | 无 | masked factor reconstruction（可选） |
| Agent commodity_beta | 硬编码 dict | YAML 先验 + 在线 EWMA 更新 |
| Agent 可信度 | 静态 base_view_scale | per-agent rolling IC × decay |
| Debate | bull/bear 加和 | Bayesian posterior |
| 优化器 risk_aversion | 常量 | regime-conditional |
| 优化器 fallback | score-based | HRP → score |
| 模式 | 手动配置 | auto（breadth + regime） |
| Backtest | walk-forward | + purged CV + CPCV + PBO |
| 上线门 | 测试通过 | + ablation 矩阵 + conformal coverage |
| 审计 | 订单事件 | + feature_version + model_version + view_ids + regime |
| 冗余 | 重复 / stub 多 | 经 M1 减法清理 |

---

## 10. 风险与边界 / Risks & Boundaries

1. **V5 不是 LLM agent autonomy 系统**。LLM 只用于结构化事件抽取（policy / news），不直接产生订单。
2. **deep model 永远不能覆盖 A 股制度约束**：限价、T+1、最小申报、停牌、ST 升降均由 `quant_math/ashare.py` 在 optimizer 与 backtest 双重把关。
3. **conformal 不是免疫卡**：覆盖率漂移只能告警和降仓，不能保证 OOS 收益。
4. **MoE 不是越多越好**：默认 3 个 expert；超过 5 个会引入训练不稳定。
5. **SSL 预训练数据 ≠ 标签**：仅作为正则，不能替代有标签训练。
6. **agent reliability 反馈延迟**：5d / 20d 才能闭环；冷启动期需要全局 prior。
7. **shadow portfolio ≠ live**：滑点 / 排队 / 拒单仍可能与 live 有显著差异。任何 live 上线必须经过小资金 canary。
8. **2025-06-27 上交所 / 深交所 ST 涨跌幅 5% → 10% 征求意见**仍未见正式实施文件；V5 把 ST 涨跌幅做成配置项，**默认 5%**，正式文件出台后由 ops 修改 yaml。

---

## 11. 写在最后 / Closing

V5 不是把 V4 推倒重来。V4 已经把"量化骨架 + agent 证据 + 组合优化 + A 股制度"四件事的接口拉通；V5 要做的是：

- **把每个接口背后的逻辑从 stub 升级为可治理的智能模块**；
- **把分散在多处的实现（labels / ic_analysis / position_sizing）收敛到单一来源**；
- **把"研究级工具"（conformal / hmm / hrp / purged_cv / triple_barrier / composite）真正接进训练 → 推理 → 组合主链**；
- **把 agent 从硬编码常识升级为可学习先验 + 在线反馈**；
- **把模型层的 factor_gate / quantile / confidence 输出从"展示性"变成"驱动下游决策"**。

按本文 M1–M6 推进 13–19 人周后，QuantAgent 才真正具备"AI Quant OS"的形态：一个在大 A 市场上有结构、有反馈、有治理、有审计的研究 + 生产闭环系统。
