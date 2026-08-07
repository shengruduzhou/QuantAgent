# QuantAgent 研究基础与 Fuyao 产品契约审计

> 审计日期：2026-08-08。该文档记录“论文原则 → QuantAgent 实现 → 本次发现/修复”，避免未来把引用论文当装饰性 bibliography。

## 1. 基础研究映射

| 参考 | QuantAgent 应承担的约束 | 当前落点 | 本次判断 |
|---|---|---|---|
| Sharpe (1964) CAPM | 市场/系统性风险与基准风险暴露必须可区分；不能只看绝对收益 | benchmark/excess-return、风险归因、组合风险 | 保留；不把 CAPM 当单因子选股器 |
| Black & Scholes (1973) | 期权/衍生品定价、无套利和波动率敏感度的理论边界 | 当前 A 股现货主链没有期权定价模块 | 不强行混入现货多因子分数；未来有期权链时单独进入 derivatives/risk layer |
| Fama & French (1992) | 横截面规模/价值等系统特征需要基线和增量检验 | 因子实验、横截面 RankIC、线性基线 | 保留；任何 ML/interaction 必须对简单基线报告增量 |
| Carhart (1997) | 动量与交易成本不可忽略；绩效持续性不能直接解释为 alpha | momentum 因子、成本后回测 | 保留；所有经济指标使用成本后结果 |
| Kakushadze (2016) 101 Formulaic Alphas | 大量公式 alpha 应作为候选库，不应按单一 in-sample IC 直接晋级 | `src/quantagent/factors/alpha101.py` | 已实现；继续受同一 OOS/多重试验门控 |
| Gu, Kelly & Xiu (2020) | 非线性收益主要来自 predictor interactions；必须比较线性/非线性且严格 OOS | `models/interactions.py`, `research/model_comparison.py` | 模型分类与 interaction 构造正确；生产统计门控需加严 |
| Qlib / arXiv:2009.11189 | 数据/handler/model/workflow/recorder/backtest/online 的基础设施；processor 只能 train-fit | `src/quantagent/qlib/*` | PR #46 已完整桥接；QuantAgent PIT/holdout 继续为上层治理 |

## 2. 非线性多因子审计

当前 `ModelClass` 明确区分：

1. linear additive；
2. rank-weighted additive；
3. per-factor nonlinear transform；
4. explicit factor interaction；
5. regime interaction；
6. nonlinear learner；
7. nonlinear objective；
8. ensemble。

这个区分保持不变。`x_i*x_j` interaction 在横截面 rank-normalise 后构造，interaction 选择使用剔除两个 parent main effects 后的 residual IC，而不是 raw product IC。该设计符合“复杂度必须证明增量信息”的要求，因此本次不改动。

需要修复的是**晋级层**而非 interaction 数学本身：

- 研究模块曾存在 `max_pbo=0.50` 的宽松默认；
- fusion Pareto preference 可以给出“preferred”，但它本质是研究排序，不等于生产批准；
- DSR/SPA 数学工具已经存在，但没有统一接到所有候选晋级路径；
- benchmark / PIT / final holdout 缺失时，需要统一 fail closed。

本次新增 `quantagent.research.foundation_gates`，把生产政策与 optimiser/search 解耦：

```text
PBO <= 0.25
DSR probability >= 0.95
SPA p-value <= 0.05
explicit benchmark required
PIT validation required
final holdout untouched required
close signal => execution lag >= 1 trading day
```

`search-factor-fusion` 每次运行额外生成 `promotion_gate.json`。Pareto preferred 仍可用于研究，但只有该文件显示 `promotionEligible=true` 才能进入后续生产讨论。

## 3. Fuyao / Financial-API 16 场景审计

此前 QuantAgent 已有 Market Workbench、Market Intelligence、财务视图和 16 场景映射，但“映射存在”不等于“产品契约完整”。本次新增后端机器可读 registry：

`services/quant_api/services/fuyao_best_practices.py`

并暴露：

`GET /api/market/best-practices`

每个场景记录：

- 上游真实 endpoint；
- QuantAgent 实际工作站；
- 必须输出；
- 计算/执行契约；
- 禁止越界的语义。

特别锁定：

- 13 价格成交量突破：前 55 日高点排除当天、20 日退出、量比/MA60、T close -> T+1 open、复权事件、成本/滑点、假突破和敏感性；
- 14 时间序列动量：120d momentum、MA120、60d vol、周/月重采样、等权/逆波动率、T+1、无 active = 100% cash、状态泳道/风险贡献；
- 15 短期反转：5d relative return、流动性/MA120/异常跌幅、bottom decile、T+1、持有/冷却、decile/RankIC/regime/sensitivity；
- 16 龙虎榜资金拓扑：交易日级聚合、默认 range_days=1、多概念净额等分保证守恒，禁止伪装成 intraday。

## 4. 六类 Fuyao 数据产品

统一登记：

1. 行情与历史数据；
2. 财务与公司数据；
3. 基金数据；
4. 盘面特色数据；
5. 交易日历与市场基础；
6. 任务型结果输出。

原始全量取数仍由 PR #45 的 Fuyao exhaustive sync / capability registry 负责。本次不复制一套数据下载器，而是补齐**产品/UI/研究使用契约**。

## 5. 报告系统修复

旧 Evidence Center 只能下载 JSON。现在增加单文件离线 HTML 导出，要求：

- CSS 与报告数据内联；
- 不包含 API Key；
- 写明生成时间、来源/endpoint、计算口径与非投资建议；
- unavailable 保持 unavailable；
- 显示统一晋级门槛；
- 只有持久化 artifact 才能进入报告，不根据结果反推 signal reason/factor contribution/fill。

## 6. 保持不改的正确部分

本次没有为了“融合更多论文”而重写以下已经正确的机制：

- `available_at` PIT 数据契约；
- purged walk-forward / embargo；
- Gu-Kelly-Xiu 风格横截面 rank normalisation；
- interaction residual incremental selection；
- Qlib processor train-only fit；
- 101 Formulaic Alphas 候选库；
- Fuyao 完整 REST/MCP/dump 获取与文档 drift audit；
- 当前指数成分不得回填历史日期；
- 财务 `report_date_ms` 优先于 `period_end_ms`。

这些部分继续作为基线，而不是为了代码 churn 重写。
