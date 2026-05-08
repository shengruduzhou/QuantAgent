# QuantAgent 数学化决策层

当前系统正在从“AI 模型给分”升级为：

```text
AI 预测
  -> 统计检验
  -> 概率置信度
  -> Regime 调整
  -> 成本和风险估计
  -> 组合优化
  -> 风控约束
```

QMT 仍然后置为交易网关。这里的数学层只负责研究、训练后验证、目标权重生成，不连接账户。

## 模块

```text
src/quantagent/quant_math/
  labels.py            多周期 log return、风险、行业中性标签
  neutralization.py    winsorize、robust z-score、行业/市值中性化
  ic_analysis.py       IC、Rank IC、IR、正 IC 占比、decay curve
  signal_fusion.py     精度加权 Alpha 融合、ensemble confidence、Black-Litterman
  regime.py            市场状态识别和 RegimeMultiplier
  covariance.py        sample / shrinkage / EWMA covariance
  risk_metrics.py      portfolio vol、VaR、CVaR、drawdown multiplier
  transaction_cost.py  commission、tax、slippage、market impact
  optimizer.py         连续均值-方差优化器，支持 fallback
  constraints.py       A 股 100 股整数手、流动性权重限制
```

## 核心优化问题

第一版连续优化器近似求解：

```text
max_w alpha^T w
      - lambda w^T Sigma w
      - gamma ||w - w_prev||_1
      - eta cost^T |w - w_prev|
```

约束：

```text
sum(w) <= max_total_weight
0 <= w_i <= max_position_weight
||w - w_prev||_1 <= max_turnover
sum(w_sector) <= max_sector_weight
|beta_p - beta_target| <= beta_limit
```

如果安装了 `cvxpy`，使用凸优化求解；如果没有安装，使用可运行的启发式 fallback，方便先跑通研究闭环。

## 推荐使用顺序

```text
1. 模型输出 alpha、risk、confidence
2. ic_analysis 计算近 60 日 Rank IC 和 error variance
3. signal_fusion 做 precision-weighted alpha
4. regime 检测市场状态并调整 alpha
5. covariance 估计 Sigma
6. transaction_cost 估计交易成本
7. optimizer 求 target_weights
8. constraints 转为 A 股整数手目标
9. strategy/risk_gate 做硬风控
```

## 安装优化依赖

```powershell
pip install -e ".[optimization]"
```

训练依赖仍然是：

```powershell
pip install -e ".[training]"
```

## 设计边界

这里生成的是 `target_weights`，不是实盘订单。

后续接 QMT 时，路径应是：

```text
target_weights
  -> OrderManager
  -> A-share constraints
  -> RiskGate
  -> QMTGateway
```

模型、统计检验、组合优化都留在 QMT 外部，这样更可测试、可回测、可替换。
