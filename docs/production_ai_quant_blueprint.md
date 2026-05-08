# QuantAgent 生产级 AI 量化技术方案

> 目标：构建一个可训练、可回测、可解释、可风控、可逐步接入实盘的 AI 量化系统。本文是技术方案，不构成投资建议，也不承诺收益。

## 1. 对参考项目的判断

你给的两个项目很有价值，但定位要分清：

- [TradingAgents](https://github.com/TauricResearch/TradingAgents) 是多智能体 LLM 金融交易研究框架，README 显示它包含 fundamental、sentiment、news、technical、trader、risk management、portfolio manager 等角色，并在 2026-04 的 v0.2.4 中加入了结构化输出、LangGraph checkpoint、持久化 decision log、多模型提供商和 Docker 支持。
- [ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) 是 AI hedge fund 概念验证，包含价值、成长、情绪、基本面、技术、风险经理、组合经理等 agent，但项目 README 明确说明它是 educational / proof of concept，且不实际交易。

结论：这两个项目适合借鉴“多角色研究流程”和“决策解释”，但不能直接作为实盘交易核心。生产级系统应该采用更硬的量化架构：

```text
数据层 -> 特征层 -> Alpha / Risk / Text 模型 -> 组合优化 -> 风控门 -> 执行 -> 监控
              LLM Agent 只做结构化抽取、校验、解释、研究辅助
```

## 2. 总体架构

```text
Market Data
  OHLCV, corporate actions, fundamentals, estimates, ownership, macro
        |
Feature Store
  point-in-time features, labels, embeddings, event scores
        |
Model Layer
  Short Alpha Model       -> 1d / 3d / 5d / 20d excess return rank
  Long Holding Model      -> 3m / 6m / 12m risk-adjusted return rank
  Risk Model              -> volatility, drawdown, crash probability
  Text/Event Model        -> news, earnings, policy, filings impact
        |
Signal Fusion
  normalized ShortScore, LongScore, NewsScore, LLMScore, RiskScore
        |
Portfolio & Risk
  exposure limits, sector limits, turnover, liquidity, stop rules, kill switch
        |
Execution
  paper trading -> broker adapter -> order audit -> monitoring
```

## 3. 最推荐的第一阶段

第一阶段只做“短线截面排序模型”，这是最容易形成闭环、最能验证 AI 量化有效性的部分。

```text
任务：每天预测股票池中未来 5 日最可能跑赢基准的股票
标签：future_5d_return - future_5d_benchmark_return
模型：TCN 或 Transformer Encoder
损失：SmoothL1 + 每日 Rank IC 评估
交易：只买 Top 5%-10% 且通过风控门的股票
验证：walk-forward，不允许随机切分
```

输入样本：

```text
X = [ticker, date, lookback_days=60, features=50~150]
y = [ret_1d, ret_5d, ret_20d, vol_5d, drawdown_risk]
```

当前仓库已经提供 `src/quantagent/models/alpha_transformer.py` 和 `src/quantagent/training/losses.py`，后续接入真实数据即可训练。

## 4. 长短线结合方式

生产级上不要让“短线”和“长线”互相打架。建议角色划分：

```text
LongScore  决定一只股票是否值得进入长期候选池
ShortScore 决定何时买入、加仓、暂停
RiskScore  决定买多少、是否降仓、是否禁止交易
NewsScore  捕捉事件冲击
LLMScore   用于解释、交叉验证、发现遗漏，不单独触发订单
```

当前实现的默认融合公式：

```text
FinalScore =
  0.35 * ShortScore
+ 0.30 * LongScore
+ 0.15 * NewsScore
+ 0.10 * LLMScore
- 0.20 * RiskScore
```

风控门：

```text
if LongScore < 50:
    exit_or_block
elif RiskScore >= 70:
    reduce_position
elif LongScore >= 75 and ShortScore >= 65 and RiskScore <= 40:
    allow_buy
else:
    hold
```

代码位置：

- `src/quantagent/strategy/signal_fusion.py`
- `src/quantagent/strategy/risk_gate.py`
- `src/quantagent/strategy/position_sizing.py`
- `src/quantagent/strategy/decision_engine.py`

## 5. 模型路线

### 短线 Alpha 模型

优先级最高。预测未来 1d / 5d / 20d 超额收益和短期风险。

推荐：

- 第一版：MLP / TCN / Transformer Encoder
- 第二版：PatchTST / TFT 风格模型
- 不建议第一版：强化学习直接选股

### 长线持有模型

长线不要主要看 K 线，要看季度财报、估值、预期、机构持仓、行业周期、政策暴露。

推荐输入：

```text
过去 8-16 个季度财报
估值历史分位
收入/利润/现金流质量
ROE / ROIC / 毛利率 / 负债
分析师预期修正
机构和基金持仓变化
政策扶持暴露分
行业相对强弱
```

推荐模型：Fundamental MLP / Tabular Transformer / FT-Transformer 类模型。

### 风险模型

风险模型要独立存在，不能只是 Alpha 模型的副产品。

输出：

```text
volatility_20d_forecast
drawdown_risk
crash_probability_5d
liquidity_risk
event_risk
```

### 文本模型

本地 3090 适合微调 FinBERT / DeBERTa-small / MiniLM，不适合从零训练大语言模型。

任务：

```text
新闻情绪
财报语气
政策利好程度
公告事件分类
风险事件识别
```

LLM API 做高层抽取和总结，小模型做稳定、便宜、可批处理的打分。

## 6. 推荐技术栈

研究和训练：

- Qlib：AI-oriented quant research、数据管理、因子、模型训练和回测。Microsoft Qlib README 描述其支持 supervised learning、market dynamics modeling、RL。
- PyTorch：训练 TCN / Transformer / 文本模型。
- Parquet / DuckDB / Polars：特征存储和批处理。

回测和实盘：

- LEAN：生产级事件驱动回测和实盘引擎，支持自定义数据、费用、滑点、经纪商模型。
- vectorbt：快速研究和参数扫描。
- 自研 `strategy` 层：保持模型输出、风控、仓位逻辑在本仓库内可审计。

强化学习：

- FinRL / FinRL-X：适合作为后期仓位调整、执行优化、RL 研究环境。不要第一阶段让 RL 直接控制实盘选股。

## 7. 实盘上线门槛

实盘前必须通过：

```text
1. 数据 point-in-time 校验，无未来函数
2. walk-forward 样本外回测
3. 手续费、滑点、冲击成本后仍有效
4. 分年度、分行业、分市场状态表现稳定
5. turnover 和容量可接受
6. paper trading 至少 1-3 个月
7. 订单审计、异常熔断、最大亏损限制
8. 小资金灰度上线
```

关键指标：

```text
Rank IC
ICIR
Top quantile excess return
long-short return
Sharpe / Sortino
max drawdown
turnover
hit rate
profit factor
capacity
```

## 8. 当前仓库下一步实现顺序

```text
Phase 1:
  接入 OHLCV 数据
  构建技术因子和标签
  训练短线 Alpha Transformer
  walk-forward 回测

Phase 2:
  接入新闻/财报/公告数据
  LLM 抽取事件，FinBERT 或小模型打分
  验证 NewsScore 的边际贡献

Phase 3:
  接入基本面、分析师预期、机构持仓
  训练 LongScore
  用 LongScore 过滤长期股票池

Phase 4:
  接入组合优化、风险预算、paper trading
  生成每日信号、仓位、风险报告

Phase 5:
  小资金实盘
  增加 broker adapter、订单审计、kill switch、监控告警
```

## 9. 一个重要判断

这套系统最核心的竞争力不是“LLM 会不会说买入”，而是：

```text
可回测的数据闭环
可训练的 Alpha / Risk / Text 模型
可审计的风控和仓位系统
可复现的样本外验证
可逐步灰度的实盘工程
```

所以本仓库第一版先把确定性交易决策层、模型接口、训练骨架搭起来，这是后面所有数据和模型迭代的底座。

## 10. 参考来源

- TradingAgents README: https://github.com/TauricResearch/TradingAgents
- AI Hedge Fund README: https://github.com/virattt/ai-hedge-fund
- Microsoft Qlib: https://github.com/microsoft/qlib
- AI4Finance FinRL: https://github.com/AI4Finance-Foundation/FinRL
- QuantConnect LEAN: https://www.lean.io/

## 11. QMT 的位置：后置执行网关

QMT / MiniQMT 后续应该作为 execution gateway，而不是训练系统的一部分。

成熟结构是：

```text
AI 训练系统
  生成 ShortScore / LongScore / RiskScore / target_weight
        |
策略层
  风控门、仓位 sizing、组合约束
        |
OrderManager
  订单状态、去重、撤单、回报同步
        |
QMTGateway
  xtdata / xttrader 查询、下单、撤单、成交回报
```

当前阶段先不接 QMT，优先完成：

```text
1. 日频 OHLCV 数据标准化
2. 技术因子和未来收益标签
3. 短线 Alpha Transformer 训练
4. walk-forward 样本外验证
5. Rank IC、ICIR、Top quantile return 评估
```

等训练和回测稳定后，再新增：

```text
src/quantagent/execution/broker_base.py
src/quantagent/execution/qmt_gateway.py
src/quantagent/execution/order_manager.py
```

这样 QMT 可以被替换，模型训练也不会被本地客户端、券商登录状态、交易时间、网络断开等实盘因素干扰。

## 12. 已落地的数学层模块

当前仓库已经新增：

```text
src/quantagent/quant_math/labels.py
src/quantagent/quant_math/neutralization.py
src/quantagent/quant_math/ic_analysis.py
src/quantagent/quant_math/signal_fusion.py
src/quantagent/quant_math/regime.py
src/quantagent/quant_math/covariance.py
src/quantagent/quant_math/risk_metrics.py
src/quantagent/quant_math/transaction_cost.py
src/quantagent/quant_math/optimizer.py
src/quantagent/quant_math/constraints.py
```

这些模块对应：

```text
AI 预测 + 统计检验 + 概率置信度 + Regime 调整
+ 协方差估计 + VaR/CVaR + 成本模型 + 组合优化 + A 股整数手约束
```

详细说明见 `docs/math_optimization_layer.md`。

## 13. AI Quant OS v2 架构修正

最新架构把系统定义为：

```text
数学信号和组合优化为执行核心
LLM 多 Agent 为信息抽取与解释核心
基本面估值和宏观政策为长期约束
风控为最高优先级
```

硬约束：

```text
1. LLM / Multi-Agent 不直接决定买卖。
2. 所有 Agent 输出结构化 AgentSignal。
3. 所有研究层输出最终统一为 TargetWeight。
4. 没有样本外验证的策略不能上线。
5. QMT 只作为后续 execution gateway。
```

当前已新增 Phase 1 不训练基线：

```text
technical_indicators.py  RSI / MACD / Bollinger / ATR / ADX / Donchian / VWAP
rule_signals.py          均值回归和动量突破规则信号
valuation.py             DCF / Reverse DCF / relative valuation
quality.py               Quality score / Fraud risk / LongScore
arbitration.py           Agent 置信度、证据质量、历史误差加权仲裁
weight_adapter.py        short weight + long weight -> TargetWeight
performance.py           Sharpe / Sortino / MaxDD / Calmar / hit ratio / profit factor
```

详细边界见 `docs/ai_quant_os_v2.md`。
