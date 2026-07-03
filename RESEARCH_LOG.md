# RESEARCH_LOG — 外部研究笔记（Stage D/F，持续维护）

> 规则：每条含来源；外部想法必须转成 PIT-safe、可检验假设（进 `HYPOTHESIS_REGISTRY.md`）后才可实现。

## 2026-07-03 #R1 — CPCV / PBO 的适用性与局限

- 来源：Bailey, Borwein, López de Prado, Zhu, *The Probability of Backtest Overfitting*, Journal of Computational Finance (2017)（[ResearchGate](https://www.researchgate.net/publication/318600389_The_probability_of_backtest_overfitting)）；López de Prado, *Advances in Financial Machine Learning* (2018) ch.7/11–12；[Arian et al. 2024, "Backtest overfitting in the machine learning era", ScienceDirect/KBS](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110)；[Purged CV 概览, Wikipedia](https://en.wikipedia.org/wiki/Purged_cross-validation)；[QuantBeckman CPCV 实现笔记](https://www.quantbeckman.com/p/with-code-combinatorial-purged-cross)。
- 要点：Arian et al. (2024) 的受控合成环境比较发现 **CPCV 在抗过拟合上优于 walk-forward/k-fold**，但 walk-forward 仍是贴近实盘的行业标准 ⇒ 我们的双轨（WF 主力 + CSCV/PBO 校正）与文献一致。CPCV 的 2025 批评：**假设未来是历史的重组合**，无法预见结构断裂；短历史下窗口不足。
- 应用：Phase 2.5 已按 CSCV 实现 PBO（S=8/16 一致，0.886）。局限对我们的映射：候选相关 0.805 → PBO 是"族内选择是否噪声"的检验，不回答"族整体是否有 alpha"（那要靠 WF 逐折分布 + 新鲜 forward 窗）。
- 衍生假设：[[H-001]]（族稳健聚合优于单点赢家）直接由 PBO 结论驱动。

## 2026-07-03 #R2 — DSR 与试验数核算

- 来源：Bailey & López de Prado, *The Deflated Sharpe Ratio* (Journal of Portfolio Management, 2014)。
- 要点：E[max SR] 随 N 与截面方差增长；偏度/峰度修正必不可少（我们的赢家 skew −0.65 / kurt 7.35，修正非平凡）。相关候选使独立-N 公式偏保守（对接受方不利）——可接受的从严方向。
- 应用：`ACCEPTANCE_RULES.md` R2 已内置；`HYPOTHESIS_REGISTRY.md` 增设**累计试验台账**（同族 N 累加）。

## 2026-07-03 #R3 — 稳健聚合 vs 选择（shrinkage/model averaging）

- 来源：经典 forecast combination 文献（Bates & Granger 1969 组合优于单一预测者；Timmermann 2006 handbook：等权组合在参数不确定下顽强）+ 本仓 PBO 直接证据。
- 机制：当候选间相关高、排名不稳时，选择噪声主导 ⇒ 等权/中位数聚合把"挑错赢家"的方差整段移除，代价是放弃（大概率虚假的）选择溢价 18.9pp。
- 应用：[[H-001]] 的四个先验配置（无搜索）。

## 2026-07-03 #R4 — 换手感知构建（turnover-aware）

- 来源：Gârleanu & Pedersen, *Dynamic Trading with Predictable Returns and Transaction Costs* (JF 2013)（aim-portfolio/部分调仓）；实践常用 score-EMA / no-trade band。
- 机制：日频 rank 信号噪声大，部分向目标移动可大幅降换手而保留大部分截面信息。本仓证据：27 候选换手 0.15–0.21/日（≈8bps 单边成本下年化拖累 ~6–10%），regime 赢家 0.81–0.86/日（不可接受）。
- 应用：[[H-002]] score EMA / band 平滑，先验 3 档参数，不做网格膨胀。

## 待研主题（进 IDEA_QUEUE，未立项）

A 股微观结构成本（涨跌停排队不可得性、打板 adverse selection 已有内证）；offline RL 作 turnover controller（CQL/IQL 类保守策略 + do-no-harm 基线）；ensemble pruning（按稳定性而非均值裁剪）；市场冲击模型（√participation 定律）用于容量估算精化。
