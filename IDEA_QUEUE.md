# IDEA_QUEUE — 研究想法队列（Stage D，按优先级排序）

> 出队条件：转成 HYPOTHESIS_REGISTRY 预注册条目。每条含：来源 / 机制 / 泄漏风险 / 预估资源。

| # | 想法 | 层 | 来源 | 机制一句话 | 泄漏风险 | 资源估计 | 状态 |
|---|---|---|---|---|---|---|---|
| 1 | 族稳健聚合（median/等权/先验平均） | ensemble | PBO 0.886 + 组合文献 | 消除选择方差 | 低 | CPU 5min | → **H-001 已注册** |
| 2 | score EMA / no-trade band 换手平滑 | book | Gârleanu–Pedersen | 部分调仓保留信号、砍成本 | 低 | CPU 10min | → **H-002 已注册** |
| 3 | 新鲜数据入库+冻结 | data | EVALUATION_PROTOCOL_V2 §2 | FRESH 窗成熟的前提 | 无（不评测） | CPU ~30min + 网络 | → **H-003 待批** |
| 4 | sector 集中度/单票上限压力测试 | book | decision_chain 已有 30%/5% 门 | 收紧至 20% 看稳健性代价 | 低 | CPU 10min | 排队 |
| 5 | 长 sleeve 诊断：何窗有增量信息（IC 分解 by regime） | ensemble | 长 sleeve weight=0 从未独立验证 | 判断"丢长"是真信息还是 val 噪声 | 低 | CPU 15min | 排队 |
| 6 | 稳定性加权 blend（子窗 rank 稳定度作权重） | ensemble | ensemble pruning 文献 | 奖励稳定而非均值 | 中（权重本身是拟合物，需预注册函数形式） | CPU 15min | 排队 |
| 7 | drawdown/regime 感知 gross 缩放（先验规则，非搜索） | book | strict_policy_search 基建已有 | 熊市降 gross 保 Calmar | 中（规则须先验声明） | CPU 20min | 排队 |
| 8 | DSL 因子新批次（cap=20 候选/批，tradability-aware 验收，oos-end 强制记录） | factor | RD-Agent 闭环（已可信化） | 补充截面信息源 | 低（DSL+PIT 契约） | LLM 调用 + CPU 1–2h | 排队（依赖 H-001 定参考配置） |
| 9 | offline RL turnover-controller（CQL 类，保守；reward=净收益−成本−换手−DD 惩罚；flat-book/no-trade 双基线；PBO 门槛适用） | RL | Stage D 授权范围 | RL 只管"何时不动"，不产 alpha | 中（env 构造须 PIT；train 窗 ≤2025-08） | GPU 数小时（需授权） | 排队（远期） |
| 10 | walk-forward 重训协议（模型层变更的唯一评测路径；embargo 修至 ≥126d） | training | EVALUATION_PROTOCOL_V2 §4a | 模型改动的选择基础 | 低（协议本身） | GPU 多小时/折（需授权） | 排队（Phase 6 后段） |
| 11 | 市场冲击 √participation 容量模型精化 | eval | 市场冲击文献 | 容量估算从 ADV 比例升级 | 低 | CPU 小 | 排队 |
| 12 | UI trust_class 徽标透传 | infra | OUTPUT_ARTIFACT_AUDIT §4 | 防"漂亮数字"复流 | 无 | 20 行 | 排队（P-H） |
| 13 | 池因子线性层集成（carrier rank-blend 层加 RC7/M12，非 GBM 特征空间） | ensemble | EXP-026 ridge 诊断（线性 ΔIC +0.0058，3/3 折正；GBM 冗余） | survivor 信息线性模型不可从基座重构，而书级 blend 恰是线性 | 中（书级集成 fold-informed 风险 ⇒ 必须 post-FRESH 首读后） | CPU ~20min | 排队（post-FRESH 硬前置） |

## 已否决（勿重提，证据在案）

1 分钟 OHLCV 做T（净负）；执行 overlay open/VWAP/TWAP（符号翻转）；naive sector momentum（whipsaw）；regime 因子子集搜索（死路）；打板（逆选择 −2%/板）；全宇宙原始 MLP（无 edge）；在 SEARCH 窗上继续加密 blend/k 搜索（PBO 0.886 判死）。
