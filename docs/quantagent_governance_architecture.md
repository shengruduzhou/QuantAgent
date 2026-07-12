# QuantAgent Governance Architecture

> 状态：PR-3 至 PR-12 累计重构后的目标结构。本文描述工程边界，不构成收益承诺或实盘批准。

## 1. 单一可信链路

```text
raw public evidence / market data
  -> PIT normalization and manifests
  -> governed feature products
  -> factor admission gates
  -> short / mid / long sleeve models
  -> calibration and uncertainty
  -> regime-aware sleeve blend
  -> hard-constrained Pareto portfolio
  -> A-share execution simulation
  -> reconciliation and paper report
  -> trusted evaluator / live-readiness gates
```

生产配置必须提供完整 lineage：数据 hash、feature schema、模型版本、训练窗口、校准窗口、选择协议、组合约束和 evaluator trust class。

## 2. 数据与证据

### Policy

`quantagent.data.policy.builder` 分离：

- `published_at`
- `public_available_at`
- `ingested_at`
- `available_at`
- `effective_at`
- `revised_at`
- `superseded_at`

历史研究可使用公开可得时间；严格在线回放可选择本地摄取时间。政策记录包含原文 hash、发文机构权威度、文件类型、地域与显式资金规模。

### Bond and fiscal liquidity

`quantagent.data.bond.builder` 将利率统一为 percent、资金流统一为 CNY billion，并构造：

- OMO / MLF net injection
- central and local government net financing
- local special-bond net financing
- policy-bank bond net financing
- government-deposit drain
- monetary and fiscal liquidity impulse

单位不可信或信用利差方向异常时，manifest gate 关闭。

### State-team inference

`quantagent.data.state_team.posterior` 只接受公开证据，并始终标记 `inferred`。普通前十大股东不会进入国家队事件；持仓使用真实公告时间，固定 45 个工作日估算默认关闭。多条高度相关的 ETF 证据按独立证据族去重。

## 3. Feature contracts

`quantagent.training.feature_contract` 明确区分：

- `required`
- `optional`
- `forbidden`

研究任务可以记录缺失后继续；production 合同缺少必须产品时 fail-closed。训练和推理必须引用同一个 contract 名称及 schema hash。

## 4. 因子治理

`quantagent.factors.governance_metrics` 在因子进入模型搜索前评估：

- PIT coverage
- cross-sectional RankIC
- IC information ratio
- losing-period rate
- decay curve
- active-library correlation
- ADV capacity

单一全样本 IC 不构成 admission。高度相关因子按 correlation cluster 管理，避免对同一风险轴重复计权。

## 5. 模型选择治理

`quantagent.research.selection_governance` 提供：

- append-only cumulative trial registry
- nested purged selection
- outer-fold-only evaluation
- PBO
- Deflated Sharpe Ratio
- SPA
- losing-fold gate

最终候选由 inner selection 决定；outer performance 不参与二次选优。所有失败和重复搜索都计入 trial family。

## 6. Calibration and sleeve blend

`quantagent.ensemble.calibration` 使用无强制 sklearn 依赖的 isotonic PAV 与 split-conformal uncertainty。

`quantagent.ensemble.regime_sleeve_blend`：

- 将 sleeve raw score 转为逐日横截面 rank；
- 依据市场 regime 分配 short / mid / long 基础权重；
- 按 uncertainty 收缩有效权重；
- 低 blend confidence 时输出 `cash_preferred`，而不是强制满仓。

## 7. Portfolio construction

`quantagent.portfolio.pareto_allocator` 先执行硬约束：

- single-name weight
- sector weight
- style exposure
- turnover
- ADV participation
- minimum cash
- gross exposure
- minimum number of names

仅对可行解计算 Pareto frontier。收益、风险、成本、换手和集中度不再由一个可被搜索器利用的无限制 scalar loss 决定。

## 8. Execution and reconciliation

`quantagent.execution.auction_impact` 用于 paper/research：

- opening-auction indicative price
- matched and unmatched queue
- cancellation risk
- price-limit blocking
- fill probability
- temporary/permanent impact

`quantagent.execution.reconciliation` 使用证券代码归一化、A 股 lot tolerance、现金误差、fill rate、slippage 和 unresolved-order 门。任何 report 通过都不等于授权实盘。

## 9. CLI boundary

默认 CLI 仅加载受治理的数据、训练、可信评测、paper、readiness 和存储入口。以下 legacy 模块只有设置：

```text
QUANTAGENT_ENABLE_LEGACY_CLI=1
```

才会注册；完整列表在 `configs/legacy_cli_manifest.json`。Legacy 入口只能用于历史复现，不能生成 production-approved 声明。

## 10. Validation status

本轮累计分支按用户要求未执行测试。合并前至少需要：

```bash
python -m compileall -q src services scripts
python -m pytest tests/ -q
git diff --check
```

CI 安装入口为：

```bash
python -m pip install -e ".[test]"
```

任何测试失败、schema 不一致或 trusted evaluator gate 失败，都应阻止合并或 production promotion。
